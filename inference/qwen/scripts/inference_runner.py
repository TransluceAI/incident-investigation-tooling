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


def thinking_enabled(config):
    # Qwen templates can default to thinking; parsing and generation must agree.
    enabled = config.get('chat_template_kwargs', {}).get('enable_thinking')
    if type(enabled) is not bool:
        raise ValueError('chat_template_kwargs.enable_thinking must be true or false')
    return enabled


def engine_options(config):
    """Resolve the JSON grammar gate without changing saved application settings."""
    options = dict(config['engine'])
    for key in ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size'):
        if key in options and (type(options[key]) is not int or options[key] != 1):
            raise ValueError(f'{key} must be 1 for this one-GPU-per-worker launcher')
    if thinking_enabled(config):
        structured = dict(options.get('structured_outputs_config') or {})
        for value in (options.get('reasoning_parser'), structured.get('reasoning_parser')):
            if value not in (None, '', 'qwen3'):
                raise ValueError('Thinking currently supports the Qwen3 parser only')
        if structured.get('enable_in_reasoning', False) is not False:
            raise ValueError('JSON constraints must not apply inside reasoning')
        structured.update(reasoning_parser='qwen3', enable_in_reasoning=False)
        options['structured_outputs_config'] = structured
    reasoning = options.get('reasoning_config')
    if reasoning is not None:
        # Needed only for sampling.thinking_token_budget; kept as JSON here and
        # converted to vLLM's ReasoningConfig when the engine is built.
        if not thinking_enabled(config):
            raise ValueError('engine.reasoning_config requires enable_thinking')
        if not isinstance(reasoning, dict) or set(reasoning) - {'reasoning_parser', 'reasoning_end_str'}:
            raise ValueError('engine.reasoning_config accepts reasoning_parser and reasoning_end_str only')
        if reasoning.get('reasoning_parser') not in (None, '', 'qwen3'):
            raise ValueError('Thinking currently supports the Qwen3 parser only')
        options['reasoning_config'] = {**reasoning, 'reasoning_parser': 'qwen3'}
    return options


def decoding_controls(config):
    """Validate the opt-in decoding controls; absent keys keep legacy behaviour."""
    sampling = config['sampling']
    if type(sampling.get('n', 1)) is not int or sampling.get('n', 1) != 1:
        raise ValueError('The runner requires sampling.n=1; multiple outputs cannot be committed per unit')
    repetition = sampling.get('repetition_detection')
    if repetition is not None and (
            not isinstance(repetition, dict) or not repetition
            or set(repetition) - {'max_pattern_size', 'min_pattern_size', 'min_count'}
            or any(type(v) is not int for v in repetition.values())):
        raise ValueError('repetition_detection takes integer max_pattern_size, min_pattern_size and min_count')
    if repetition is not None:
        low, high, count = (repetition.get(k, 0) for k in ('min_pattern_size', 'max_pattern_size', 'min_count'))
        if low < 0 or high < 0 or low > high or (high > 0 and count < 2):
            raise ValueError('Invalid repetition_detection bounds: require 0 <= min_pattern_size <= max_pattern_size'
                             ' and min_count >= 2 when enabled')
    budget = sampling.get('thinking_token_budget')
    if budget is not None:
        if not thinking_enabled(config):
            raise ValueError('thinking_token_budget requires enable_thinking')
        if type(budget) is not int or not 0 < budget < sampling['max_tokens']:
            raise ValueError('thinking_token_budget must be a positive integer below max_tokens')
        if 'reasoning_config' not in config['engine']:
            raise ValueError('thinking_token_budget requires engine.reasoning_config')
    citations = config.get('citation_constraints', False)
    if type(citations) is not bool:
        raise ValueError('citation_constraints must be true or false')
    retry = config.get('retry')
    if retry is not None and (not isinstance(retry, dict) or set(retry) != {'max_attempts'}
                              or type(retry['max_attempts']) is not int or retry['max_attempts'] < 1):
        raise ValueError('retry must be {"max_attempts": <positive integer>}')
    return {'citations': citations, 'attempts': retry['max_attempts'] if retry else 1,
            'budget': budget}


def check_reasoning_end(config, tokenizer):
    """The forced end string must close reasoning with exactly one </think>.

    The JSON grammar starts after </think>, so nothing may follow it.
    """
    budget = config['sampling'].get('thinking_token_budget')
    end_str = (config['engine'].get('reasoning_config') or {}).get('reasoning_end_str')
    if budget is not None and not end_str:
        end_str = '</think>'  # qwen3 parser default in the supported vLLM version
    if end_str:
        ids = tokenizer.encode(end_str, add_special_tokens=False)
        end = thinking_tokens(tokenizer)[1]
        if ids.count(end) != 1 or ids[-1] != end:
            raise ValueError('reasoning_end_str must end with a single </think> token')
        if budget is not None and budget + len(ids) >= config['sampling']['max_tokens']:
            raise ValueError('thinking_token_budget plus reasoning_end_str must leave room for final JSON')


def citation_schema(schema, row):
    """Per-unit copy of the schema whose citations can name only this unit's chunks.

    Under grammar-constrained decoding the model then cannot emit a miscopied
    segment ID or an out-of-range chunk number, and reference arrays name at least
    one known segment. Original field constraints remain in the narrowed schema;
    application-level evidence validation is still the caller's responsibility.
    """
    chunks = row.get('evidence_chunks')
    if not isinstance(chunks, dict) or not chunks:
        raise ValueError(f'citation_constraints need evidence_chunks: {row["unit_id"]}')
    if any(not isinstance(sid, str) or not sid or not isinstance(entries, list)
           for sid, entries in chunks.items()):
        raise ValueError('evidence_chunks must map nonempty segment IDs to chunk lists')
    counts = {sid: len(entries) for sid, entries in sorted(chunks.items())}
    citable = [(sid, n) for sid, n in counts.items() if n]
    if not citable:
        raise ValueError(f'No citable chunks: {row["unit_id"]}')
    found = set()

    def restrict_enum(original, allowed):
        if not isinstance(original, dict):
            raise ValueError('Citation fields require object-form schemas')
        values = [value for value in allowed
                  if 'enum' not in original or value in original['enum']]
        return {**original, 'enum': values}

    def narrow(node):
        if isinstance(node, list):
            return [narrow(x) for x in node]
        if not isinstance(node, dict):
            return node
        node = {k: narrow(v) for k, v in node.items()}
        props = node.get('properties')
        if not isinstance(props, dict):
            return node
        if 'segment_id' in props and 'chunk_id' in props:
            found.add('citation')
            chunk = props['chunk_id']
            if not isinstance(chunk, dict) or chunk.get('type') != 'integer':
                raise ValueError('Citation chunk_id must have an integer schema')
            return {'anyOf': [{**node, 'properties': {
                **props, 'segment_id': restrict_enum(props['segment_id'], [sid]),
                'chunk_id': {**chunk, 'minimum': max(1, chunk.get('minimum', 1)),
                            'maximum': min(n, chunk.get('maximum', n))}}}
                for sid, n in citable]}
        if 'evidence_ids' in props:
            ids = props['evidence_ids']
            if not isinstance(ids, dict) or ids.get('type') != 'array':
                raise ValueError('Citation evidence_ids must have an array schema')
            node['properties'] = {**props, 'evidence_ids': {
                **ids, 'minItems': max(1, ids.get('minItems', 1)),
                'items': restrict_enum(ids.get('items', {}), list(counts))}}
        return node

    narrowed = narrow(schema)
    if not found:
        raise ValueError('citation_constraints need a schema citation with segment_id and chunk_id')
    return narrowed


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
        # Boundary-looking strings inside valid JSON are ordinary application data.
        if not enabled and answer.text.lstrip().startswith('<think>'):
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
    controls = decoding_controls(config)
    legacy = not controls['citations'] and controls['attempts'] == 1
    identity_data = {'config': config, 'manifest_hash': file_hash(args.fixtures),
                     'runner_hash': file_hash(__file__), 'max_records': args.max_records}
    identity = digest(json.dumps(identity_data, sort_keys=True).encode())
    directory = Path(args.output_dir) / f'worker-{args.worker_id}'
    directory.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(config['schema'])
    Draft202012Validator.check_schema(config['schema'])
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True,
                                              trust_remote_code=False)
    # Validate every selected unit before instantiating a GPU model.
    check_reasoning_end(config, tokenizer)
    input_info = {r['unit_id']: r['text_sha256'] for r in units(args.fixtures, args.max_records)}
    for batch in batches(units(args.fixtures, args.max_records), tokenizer, config, {}):
        if controls['citations']:
            for row, _, _ in batch:
                Draft202012Validator.check_schema(citation_schema(config['schema'], row))
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
        engine_kwargs = dict(resolved_engine)
        if 'reasoning_config' in engine_kwargs:
            from vllm.config.reasoning import ReasoningConfig
            engine_kwargs['reasoning_config'] = ReasoningConfig(**engine_kwargs['reasoning_config'])
        llm = LLM(model=config['model'], tokenizer=config['model'], **engine_kwargs)
        base_sampling = dict(config['sampling'])
        if 'repetition_detection' in base_sampling:
            from vllm.sampling_params import RepetitionDetectionParams
            base_sampling['repetition_detection'] = RepetitionDetectionParams(**base_sampling['repetition_detection'])

        def sampling_for(row, attempt):
            kwargs = dict(base_sampling)
            # A retry with an explicit sampling seed draws a new sample; greedy
            # retries may produce the same answer; recovery is not guaranteed.
            if attempt > 1 and type(kwargs.get('seed')) is int:
                kwargs['seed'] += attempt - 1
            schema = citation_schema(config['schema'], row) if controls['citations'] else config['schema']
            return SamplingParams(**kwargs, structured_outputs=StructuredOutputsParams(json=schema))

        sampling = sampling_for(None, 1) if not controls['citations'] else None
        startup = {'utc': now(), 'load_seconds': time.monotonic()-started,
                   'enable_thinking': enabled, 'engine': resolved_engine,
                   'sampling': config['sampling'],
                   'metrics': metrics_snapshot(llm)}
        if not legacy:
            startup['decoding_controls'] = controls
        write_json(directory / f'startup-{os.environ.get("SLURM_JOB_ID", os.getpid())}.json', startup)
        for batch in batches(units(args.fixtures, args.max_records), tokenizer, config, done):
            if stopping or time.time() >= deadline:
                break
            begin = time.monotonic()
            print(json.dumps({'event': 'batch_start', 'utc': now(), 'committed': len(done),
                              'units': len(batch), 'prompt_tokens': sum(b[2] for b in batch)}), flush=True)
            # Model-output failures are re-run within the batch, before commit, so
            # resume semantics are unchanged: committed rows are never retried.
            results, pending, history = [None] * len(batch), list(range(len(batch))), {}
            for attempt in range(1, controls['attempts'] + 1):
                # Finish committing the completed attempt rather than start more
                # GPU work after a stop request or deadline.
                if attempt > 1 and (stopping or time.time() >= deadline):
                    break
                params = sampling if legacy else [
                    sampling if sampling is not None and attempt == 1 else sampling_for(batch[i][0], attempt)
                    for i in pending]
                outputs = llm.chat([batch[i][1] for i in pending], sampling_params=params,
                                   chat_template_kwargs=config['chat_template_kwargs'], use_tqdm=False)
                if len(outputs) != len(pending):
                    raise RuntimeError('Missing model outputs; batch not committed')
                retry = []
                for i, output in zip(pending, outputs):
                    row, chat, count = batch[i]
                    if len(output.outputs) != 1:
                        raise RuntimeError('Expected exactly one model output per unit; batch not committed')
                    answer = output.outputs[0]
                    parsed, error, answer_fields = validate_answer(
                        answer, output.prompt_token_ids, count, tokenizer, validator, enabled)
                    result = {'run_identity': identity, 'unit_id': row['unit_id'],
                              'text_sha256': row['text_sha256'], 'chunk_idx': row['chunk_idx'],
                              'status': 'failed' if error else 'success', 'result': parsed,
                              'error': error, 'raw_output': answer.text,
                              **answer_fields,
                              'input_tokens': count, 'output_tokens': len(answer.token_ids),
                              'finish_reason': answer.finish_reason, 'utc': now()}
                    if controls['budget'] is not None:
                        # Natural reasoning cannot exceed the budget, so more tokens
                        # mean vLLM forced the end string.
                        result['thinking_budget_exhausted'] = (answer_fields['reasoning_tokens'] or 0) > controls['budget']
                    if controls['attempts'] > 1:
                        result['attempt'] = attempt
                        result['previous_attempts'] = list(history.get(i, ()))
                    results[i] = result
                    if error and attempt < controls['attempts']:
                        history.setdefault(i, []).append({'attempt': attempt, 'error': error,
                                                          'finish_reason': answer.finish_reason,
                                                          'output_tokens': len(answer.token_ids)})
                        retry.append(i)
                pending = retry
                if not pending:
                    break
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
            if controls['attempts'] > 1:
                progress['batch_retried_units'] = sum(bool(r['previous_attempts']) for r in results)
                progress['batch_retry_output_tokens'] = sum(a['output_tokens'] for r in results
                                                            for a in r['previous_attempts'])
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
    if controls['attempts'] > 1:
        # Committed output_tokens count final attempts only; earlier attempts are listed here.
        final['retried_units'] = sum(bool(r.get('previous_attempts')) for r in done.values())
        final['retry_output_tokens'] = sum(a['output_tokens'] for r in done.values()
                                           for a in r.get('previous_attempts', ()))
    write_json(directory / 'summary.json', final)
    print(json.dumps({'event': 'finished', **final}), flush=True)
    if final['unfinished'] or final['failed']:
        raise SystemExit(2)


def main():
    with contextlib.ExitStack() as resources:
        run(resources)


if __name__ == '__main__':
    main()
