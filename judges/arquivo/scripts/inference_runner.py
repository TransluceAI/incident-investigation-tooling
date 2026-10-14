"""Offline, resumable inference on prepared UTF-8 JSONL units. No corpus parsing."""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as out:
        out.write(text)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def units(path, limit=None):
    seen = set()
    with open(path, encoding='utf-8') as stream:
        for idx, line in enumerate(stream):
            if limit is not None and idx >= limit:
                break
            row = json.loads(line)
            if not isinstance(row.get('unit_id'), str) or not row['unit_id']:
                raise ValueError(f'Invalid unit ID at row {idx}')
            if row['unit_id'] in seen:
                raise ValueError(f'Duplicate unit ID: {row["unit_id"]}')
            seen.add(row['unit_id'])
            if type(row.get('chunk_idx')) is not int or row['chunk_idx'] < 0:
                raise ValueError('Invalid chunk index')
            if not isinstance(row.get('text'), str):
                raise ValueError('Missing text')
            if digest(row['text'].encode('utf-8')) != row.get('text_sha256'):
                raise ValueError(f'Text hash mismatch: {row["unit_id"]}')
            yield row


def committed(directory, identity):
    rows = {}
    for part in sorted(directory.glob('part-*.jsonl')):
        with part.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['run_identity'] != identity:
                    raise ValueError(f'Run mismatch: {part}')
                if row['unit_id'] in rows:
                    raise ValueError(f'Duplicate committed ID: {row["unit_id"]}')
                rows[row['unit_id']] = row
    return rows


def messages(row, config):
    return [{'role': 'system', 'content': config['system_prompt']},
            {'role': 'user', 'content': '<document>\n' + row['text'] + '\n</document>'}]


def batches(rows, tokenizer, config, skip):
    batch, count = [], 0
    for row in rows:
        if row['unit_id'] in skip:
            if row['text_sha256'] != skip[row['unit_id']]['text_sha256']:
                raise ValueError('Committed text hash does not match input')
            continue
        chat = messages(row, config)
        ids = tokenizer.apply_chat_template(chat, tokenize=True, return_dict=False, add_generation_prompt=True,
                                            **config['chat_template_kwargs'])
        if not isinstance(ids, list) or any(not isinstance(token, int) for token in ids):
            raise TypeError('Expected a flat list of token IDs from the chat template')
        tokens = len(ids)
        if tokens + config['sampling']['max_tokens'] > config['engine']['max_model_len']:
            raise ValueError(f'Oversize unit {row["unit_id"]}: {tokens} prompt tokens')
        if tokens > config['host_batch_tokens']:
            raise ValueError('Single unit exceeds configured host batch token cap')
        if batch and (len(batch) >= config['host_batch_records'] or
                      count + tokens > config['host_batch_tokens']):
            yield batch
            batch, count = [], 0
        batch.append((row, chat, tokens))
        count += tokens
    if batch:
        yield batch


def metrics_snapshot(llm):
    if not hasattr(llm, 'get_metrics'):
        return {'available': False, 'preemptions': None, 'metrics': []}
    metrics = []
    for metric in llm.get_metrics():
        name = getattr(metric, 'name', type(metric).__name__)
        value = getattr(metric, 'value', None)
        if isinstance(value, (int, float)):
            metrics.append({'name': name, 'value': value, 'labels': getattr(metric, 'labels', {})})
    preempt = [m['value'] for m in metrics if 'preempt' in m['name']]
    return {'available': True, 'preemptions': sum(preempt) if preempt else None, 'metrics': metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fixtures', '--input-manifest', dest='fixtures', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--worker-id', type=int, required=True)
    parser.add_argument('--stop-at', help='Optional ISO 8601 deadline with timezone')
    parser.add_argument('--max-records', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.worker_id < 0 or (args.max_records is not None and args.max_records < 1):
        parser.error('worker-id must be nonnegative and max-records positive')
    deadline = float('inf')
    if args.stop_at:
        try:
            stop = datetime.fromisoformat(args.stop_at.replace('Z', '+00:00'))
        except ValueError:
            parser.error('--stop-at must be an ISO 8601 timestamp')
        if stop.tzinfo is None:
            parser.error('--stop-at must include its timezone')
        deadline = stop.timestamp()
    if time.time() >= deadline:
        raise RuntimeError('Deadline has passed')
    config = json.loads(Path(args.config).read_text())
    identity_data = {'config': config, 'manifest_hash': file_hash(args.fixtures),
                     'runner_hash': file_hash(__file__), 'max_records': args.max_records}
    identity = digest(json.dumps(identity_data, sort_keys=True).encode())
    directory = Path(args.output_dir) / f'worker-{args.worker_id}'
    directory.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(config['schema'])
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True,
                                              trust_remote_code=False)
    # Validate every selected unit before instantiating a GPU model.
    input_info = {r['unit_id']: r['text_sha256'] for r in units(args.fixtures, args.max_records)}
    for batch in batches(units(args.fixtures, args.max_records), tokenizer, config, {}):
        pass
    if args.validate_only:
        print(json.dumps({'event': 'validated', 'units': len(input_info), 'identity': identity}), flush=True)
        return
    lock = open(directory / '.writer.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity_path = directory / 'identity.json'
    if identity_path.exists():
        if not args.resume or json.loads(identity_path.read_text())['identity'] != identity:
            raise ValueError('Existing output requires --resume with matching identity')
    else:
        if list(directory.glob('part-*.jsonl')):
            raise ValueError('Parts exist without identity')
        write_json(identity_path, {'identity': identity, **identity_data})
    done = committed(directory, identity)
    if not set(done).issubset(input_info):
        raise ValueError('Committed IDs outside selected manifest')
    for key, row in done.items():
        if row['text_sha256'] != input_info[key]:
            raise ValueError('Committed/input hash mismatch')
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        print(json.dumps({'event': 'stop_requested', 'signal': signum}), flush=True)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    llm = None
    if len(done) < len(input_info):
        if time.time() >= deadline:
            raise RuntimeError('Deadline reached before model load')
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        import torch
        print(json.dumps({'event': 'loading', 'utc': now(), 'worker': args.worker_id,
                          'job_id': os.environ.get('SLURM_JOB_ID'),
                          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                          'cuda_count': torch.cuda.device_count(),
                          'gpu': torch.cuda.get_device_name(0),
                          'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                          'identity': identity}), flush=True)
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expected exactly one Slurm-allocated CUDA device')
        llm = LLM(model=config['model'], tokenizer=config['model'], **config['engine'])
        sampling = SamplingParams(**config['sampling'],
                                  structured_outputs=StructuredOutputsParams(json=config['schema']))
        write_json(directory / f'startup-{os.environ.get("SLURM_JOB_ID", os.getpid())}.json',
                   {'utc': now(), 'load_seconds': time.monotonic()-started,
                    'metrics': metrics_snapshot(llm)})
        for batch in batches(units(args.fixtures, args.max_records), tokenizer, config, done):
            if stopping or time.time() >= deadline:
                break
            begin = time.monotonic()
            print(json.dumps({'event': 'batch_start', 'utc': now(), 'committed': len(done),
                              'units': len(batch), 'prompt_tokens': sum(b[2] for b in batch)}), flush=True)
            outputs = llm.chat([b[1] for b in batch], sampling_params=sampling,
                               chat_template_kwargs=config['chat_template_kwargs'], use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError('Missing model outputs; batch not committed')
            results = []
            for (row, chat, count), output in zip(batch, outputs):
                answer = output.outputs[0]
                error = None
                parsed = None
                try:
                    if answer.finish_reason != 'stop':
                        raise ValueError(f'Nonterminal/length finish: {answer.finish_reason}')
                    if len(output.prompt_token_ids) != count:
                        raise ValueError('Rendered token count differs from engine input')
                    if '<think>' in answer.text:
                        raise ValueError('Unexpected reasoning preamble')
                    parsed = json.loads(answer.text)
                    validator.validate(parsed)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                results.append({'run_identity': identity, 'unit_id': row['unit_id'],
                                'text_sha256': row['text_sha256'], 'chunk_idx': row['chunk_idx'],
                                'status': 'failed' if error else 'success', 'result': parsed,
                                'error': error, 'raw_output': answer.text,
                                'input_tokens': count, 'output_tokens': len(answer.token_ids),
                                'finish_reason': answer.finish_reason, 'utc': now()})
            part = directory / f'part-{len(list(directory.glob("part-*.jsonl"))):06d}.jsonl'
            if part.exists():
                raise RuntimeError('Refusing to overwrite committed part')
            atomic_write(part, ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in results))
            done.update({r['unit_id']: r for r in results})
            metrics = metrics_snapshot(llm)
            progress = {'utc': now(), 'committed': len(done), 'expected': len(input_info),
                        'failed': sum(r['status'] != 'success' for r in done.values()),
                        'batch_units': len(batch), 'batch_seconds': time.monotonic()-begin,
                        'batch_input_tokens': sum(r['input_tokens'] for r in results),
                        'batch_output_tokens': sum(r['output_tokens'] for r in results),
                        'metrics': metrics}
            write_json(directory / 'progress.json', progress)
            with open(directory / 'timings.jsonl', 'a') as stream:
                stream.write(json.dumps(progress)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps({'event': 'batch_committed', **progress}), flush=True)
    final = {'utc': now(), 'identity': identity, 'expected': len(input_info),
             'committed': len(done), 'unfinished': len(input_info)-len(done),
             'failed': sum(r['status'] != 'success' for r in done.values()),
             'session_seconds': time.monotonic()-started,
             'metrics': metrics_snapshot(llm) if llm else None}
    write_json(directory / 'summary.json', final)
    print(json.dumps({'event': 'finished', **final}), flush=True)
    if final['unfinished'] or final['failed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
