"""Offline, resumable inference on prepared (window, axis) units. No corpus parsing.

Adapted from the shared judges/arquivo runner. Changes: each unit names its
axis and evidence shape; prompts come from evidence_judge.prompts; each axis has
its own output schema; and the probability of the label is read from the
label token's logprobs when ``label_logprobs`` is configured.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evidence_judge import prompts  # noqa: E402

LABELS = ('yes', 'no', 'insufficient_evidence')
LABEL_KEY = re.compile(r'"label"\s*:\s*"$')


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
    """Yield one request per (window, axis). limit counts manifest windows."""
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
            axes = row.get('axes')
            if (not isinstance(axes, list) or not axes or len(set(axes)) != len(axes)
                    or not set(axes) <= set(prompts.AXES) or not isinstance(row.get('shape'), str)):
                raise ValueError(f'Invalid axes/shape: {row["unit_id"]}')
            if digest(row['text'].encode('utf-8')) != row.get('text_sha256'):
                raise ValueError(f'Text hash mismatch: {row["unit_id"]}')
            # One manifest row per window; its axis requests stay adjacent so they
            # share a batch and the cached document prefix.
            for axis in axes:
                yield {**row, 'unit_id': f'{row["unit_id"]}:{axis}', 'window_id': row['unit_id'],
                       'axis': axis}


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
    return prompts.messages(config, row['shape'], row['axis'], row['text'])


def label_probabilities(answer, final_start=0):
    """Diagnostic first-token branch scores, NOT calibrated label probabilities.

    Require an unambiguous label boundary and candidates for every label. A
    missing top-k candidate is unknown, never zero. Multi-token labels are
    represented by their unique first prefix only; no sequence probability is
    claimed. Combined boundary/value tokens conservatively produce no score.
    """
    logprobs = getattr(answer, 'logprobs', None)
    ids = list(answer.token_ids)
    if not logprobs or len(logprobs) != len(ids):
        return None
    text = ''
    for i in range(final_start, len(ids)):
        table = logprobs[i]
        if not table:
            return None
        if LABEL_KEY.search(text):
            # Parse the prefix to ensure this is the actual top-level field,
            # rather than a quoted label-looking string inside the rationale.
            try:
                prefix = json.loads(text + '__score_boundary__"}')
            except (ValueError, TypeError):
                return None
            if prefix.get('label') != '__score_boundary__':
                return None
            mass = dict.fromkeys(LABELS, 0.0)
            found = set()
            for candidate in table.values():
                piece = (candidate.decoded_token or '').rstrip('" ,}')
                matches = [l for l in LABELS if piece and l.startswith(piece)]
                if len(matches) == 1:
                    lp = candidate.logprob
                    if math.isnan(lp) or lp > 0:
                        return None
                    mass[matches[0]] += math.exp(lp)
                    found.add(matches[0])
            if found != set(LABELS):
                return None
            total = sum(mass.values())
            return {k: v / total for k, v in mass.items()} if total > 0 else None
        chosen = table.get(ids[i])
        if chosen is None or chosen.decoded_token is None:
            return None
        text += chosen.decoded_token
    return None


def thinking_enabled(config):
    # Only an explicit opt-in activates the new handling. Keep legacy template
    # kwargs untouched when the flag is absent (Qwen itself defaults to thinking).
    enabled = config['chat_template_kwargs'].get('enable_thinking', False)
    if type(enabled) is not bool:
        raise ValueError('chat_template_kwargs.enable_thinking must be true or false')
    return enabled


def engine_options(config):
    """Resolve the JSON grammar gate without changing saved application settings."""
    options = dict(config['engine'])
    if thinking_enabled(config):
        structured = dict(options.get('structured_outputs_config') or {})
        for value in (options.get('reasoning_parser'), structured.get('reasoning_parser')):
            if value not in (None, '', 'qwen3'):
                raise ValueError('Thinking currently supports the Qwen3 parser only')
        if structured.get('enable_in_reasoning', False) is not False:
            raise ValueError('JSON constraints must not apply inside reasoning')
        structured.update(reasoning_parser='qwen3', enable_in_reasoning=False)
        options['structured_outputs_config'] = structured
    return options


def thinking_tokens(tokenizer):
    vocab = tokenizer.get_vocab()
    try:
        start, end = vocab['<think>'], vocab['</think>']
    except KeyError as exc:
        raise ValueError('Thinking requires Qwen <think>/</think> tokens') from exc
    if start == end:
        raise ValueError('Reasoning boundary tokens must be distinct')
    return start, end


def split_answer(answer, tokenizer, enabled):
    """Split Qwen offline output by token boundary, not JSON-looking reasoning.

    The prompt already opens <think>. Offline output may hide special tokens in
    answer.text, so use generated IDs. Counts include generated delimiters/EOS
    and add up exactly to output_tokens. Never recover JSON from an unfinished
    reasoning block.
    """
    ids = list(answer.token_ids)
    if not enabled:
        return {'reasoning': None, 'final_output': answer.text,
                'reasoning_tokens': 0, 'final_tokens': len(ids),
                'reasoning_complete': None}
    start, end = thinking_tokens(tokenizer)
    if end not in ids:
        return {'reasoning': tokenizer.decode(ids, skip_special_tokens=True),
                'final_output': '', 'reasoning_tokens': len(ids),
                'final_tokens': 0, 'reasoning_complete': False}
    boundary = ids.index(end)
    # Permit a generated opener only at the start; never accept multiple blocks.
    if any(token == start and i != 0 for i, token in enumerate(ids[:boundary])):
        raise ValueError('Unexpected nested reasoning block')
    if any(token in (start, end) for token in ids[boundary + 1:]):
        raise ValueError('Unexpected reasoning boundary in final answer')
    return {'reasoning': tokenizer.decode(ids[:boundary], skip_special_tokens=True),
            'final_output': tokenizer.decode(ids[boundary + 1:], skip_special_tokens=True),
            'reasoning_tokens': boundary + 1, 'final_tokens': len(ids) - boundary - 1,
            'reasoning_complete': True}


def validate_answer(answer, prompt_ids, expected_tokens, tokenizer, validator, enabled):
    parsed = None
    fields = {'reasoning': None, 'final_output': None, 'reasoning_tokens': None,
              'final_tokens': None, 'reasoning_complete': None}
    try:
        fields = split_answer(answer, tokenizer, enabled)
        if answer.finish_reason != 'stop':
            raise ValueError(f'Nonterminal/length finish: {answer.finish_reason}')
        if len(prompt_ids) != expected_tokens:
            raise ValueError('Rendered token count differs from engine input')
        # Preserve legacy validation order and allow quoted closing tags in JSON.
        if not enabled and '<think>' in answer.text:
            raise ValueError('Unexpected reasoning preamble')
        if enabled and not fields['reasoning_complete']:
            raise ValueError('Reasoning ended without </think>; final JSON unavailable')
        parsed = json.loads(fields['final_output'])
        validator.validate(parsed)
    except Exception as exc:
        # Legacy failures retain parsed JSON for inspection; status stays failed.
        return parsed if not enabled else None, f'{type(exc).__name__}: {exc}', fields
    return parsed, None, fields


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
        if thinking_enabled(config):
            start, end = thinking_tokens(tokenizer)
            last_start = max((i for i, token in enumerate(ids) if token == start), default=-1)
            last_end = max((i for i, token in enumerate(ids) if token == end), default=-1)
            if last_start <= last_end:
                raise ValueError('Thinking template must leave an open <think> block')
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


def run(resources):
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
    enabled = thinking_enabled(config)
    resolved_engine = engine_options(config)
    identity_data = {'config': config, 'manifest_hash': file_hash(args.fixtures),
                     'runner_hash': file_hash(__file__), 'prompts_hash': file_hash(prompts.__file__),
                     'max_records': args.max_records}
    identity = digest(json.dumps(identity_data, sort_keys=True).encode())
    directory = Path(args.output_dir) / f'worker-{args.worker_id}'
    directory.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    from jsonschema import Draft202012Validator
    validators = {axis: Draft202012Validator(s) for axis, s in config['schemas'].items()}
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True,
                                              trust_remote_code=False)
    # Validate every selected unit before instantiating a GPU model.
    input_info = {r['unit_id']: r['text_sha256'] for r in units(args.fixtures, args.max_records)}
    for batch in batches(units(args.fixtures, args.max_records), tokenizer, config, {}):
        pass
    if args.validate_only:
        print(json.dumps({'event': 'validated', 'units': len(input_info), 'identity': identity}), flush=True)
        return
    lock = resources.enter_context(open(directory / '.writer.lock', 'a'))
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
        llm = LLM(model=config['model'], tokenizer=config['model'], **resolved_engine)
        extra = {'logprobs': config['label_logprobs']} if config.get('label_logprobs') else {}
        sampling = {axis: SamplingParams(**config['sampling'], **extra,
                                         structured_outputs=StructuredOutputsParams(json=s))
                    for axis, s in config['schemas'].items()}
        write_json(directory / f'startup-{os.environ.get("SLURM_JOB_ID", os.getpid())}.json',
                   {'utc': now(), 'load_seconds': time.monotonic()-started,
                    'enable_thinking': enabled, 'engine': resolved_engine,
                    'sampling': config['sampling'],
                    'metrics': metrics_snapshot(llm)})
        for batch in batches(units(args.fixtures, args.max_records), tokenizer, config, done):
            if stopping or time.time() >= deadline:
                break
            begin = time.monotonic()
            print(json.dumps({'event': 'batch_start', 'utc': now(), 'committed': len(done),
                              'units': len(batch), 'prompt_tokens': sum(b[2] for b in batch)}), flush=True)
            outputs = llm.chat([b[1] for b in batch], sampling_params=[sampling[b[0]['axis']] for b in batch],
                               chat_template_kwargs=config['chat_template_kwargs'], use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError('Missing model outputs; batch not committed')
            results = []
            for (row, chat, count), output in zip(batch, outputs):
                answer = output.outputs[0]
                parsed, error, answer_fields = validate_answer(
                    answer, output.prompt_token_ids, count, tokenizer, validators[row['axis']], enabled)
                final_start = answer_fields['reasoning_tokens'] or 0
                probs = label_probabilities(answer, final_start) if not error else None
                results.append({'run_identity': identity, 'unit_id': row['unit_id'],
                                'window_id': row['window_id'], 'axis': row['axis'],
                                'text_sha256': row['text_sha256'], 'chunk_idx': row['chunk_idx'],
                                'label_probs': probs,
                                'label_score_kind': 'first_token_branch_score_not_calibrated',
                                'status': 'failed' if error else 'success', 'result': parsed,
                                'error': error, 'raw_output': answer.text,
                                **answer_fields,
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
                        'batch_reasoning_tokens': sum(r['reasoning_tokens'] or 0 for r in results),
                        'batch_final_tokens': sum(r['final_tokens'] or 0 for r in results),
                        'batch_unclassified_tokens': sum(r['output_tokens'] for r in results
                                                         if r['reasoning_tokens'] is None),
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
             'enable_thinking': enabled,
             'output_tokens': sum(r.get('output_tokens', 0) for r in done.values()),
             'reasoning_tokens': sum(r.get('reasoning_tokens') or 0 for r in done.values()),
             'final_tokens': sum(r.get('final_tokens') or 0 for r in done.values()),
             'unclassified_tokens': sum(r.get('output_tokens', 0) for r in done.values()
                                        if r.get('reasoning_tokens') is None),
             'session_seconds': time.monotonic()-started,
             'metrics': metrics_snapshot(llm) if llm else None}
    write_json(directory / 'summary.json', final)
    print(json.dumps({'event': 'finished', **final}), flush=True)
    if final['unfinished'] or final['failed']:
        raise SystemExit(2)


def main():
    with contextlib.ExitStack() as resources:
        run(resources)


if __name__ == '__main__':
    main()
