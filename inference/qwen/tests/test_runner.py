import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import contextlib
import io
import sys
import importlib.util
runner_path = Path(__file__).with_name('inference_runner.py')
if not runner_path.exists():
    runner_path = Path(__file__).resolve().parents[1] / 'scripts/inference_runner.py'
spec = importlib.util.spec_from_file_location('inference_runner', runner_path)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class FakeTokenizer:
    all_special_ids = [1, 2, 3]

    def apply_chat_template(self, chat, **kwargs):
        ids = self.encode(chat[-1]['content'])
        return ids + ([1, 10] if kwargs.get('enable_thinking') else [])

    def get_vocab(self):
        return {'<think>': 1, '</think>': 2}

    def encode(self, text):
        return [ord(c) + 100 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(i - 100) for i in ids if i >= 100)


class DeadlineTests(unittest.TestCase):
    def test_effective_deadline_uses_earlier_cutoff(self):
        with patch.dict(os.environ, {'SLURM_JOB_END_TIME': '200'}):
            self.assertEqual(r.job_deadline(None), 200)
            self.assertEqual(r.job_deadline('1970-01-01T00:01:40+00:00'), 100)
            self.assertEqual(r.job_deadline('1970-01-01T00:05:00+00:00'), 200)
        with patch.dict(os.environ, {'SLURM_JOB_END_TIME': ''}):
            self.assertEqual(r.job_deadline(None), float('inf'))
            with self.assertRaisesRegex(ValueError, 'timezone'):
                r.job_deadline('2026-10-16T12:00:00')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.row = {'unit_id': 'unit-1', 'chunk_idx': 0, 'text': 'café 日本語'}
        self.row['text_sha256'] = r.digest(self.row['text'].encode())

    def test_input_integrity_and_selection(self):
        p = self.root/'input.jsonl'
        p.write_text((json.dumps(self.row)+'\n')*2)
        self.assertEqual(list(r.units(p,1)),[self.row])
        with self.assertRaises(ValueError):
            list(r.units(p))
        p.write_text(json.dumps({**self.row,'text':'changed'})+'\n')
        with self.assertRaises(ValueError):
            list(r.units(p))

    def test_malformed_manifest_rows_report_path_and_line(self):
        path = self.root / 'input.jsonl'
        for invalid in ('[]', 'null', 'true', '42', '"text"', '', '{'):
            with self.subTest(invalid=invalid):
                path.write_text(json.dumps(self.row) + '\n' + invalid + '\n')
                with self.assertRaises(ValueError) as error:
                    list(r.units(path))
                self.assertIn(f'{path}:2:', str(error.exception))
                self.assertIn('Invalid JSON' if invalid in ('', '{') else 'Expected a JSON object',
                              str(error.exception))

    def test_interrupted_part_and_duplicate_detection(self):
        row = {**self.row, 'run_identity': 'run-a', 'status': 'success', 'input_tokens': 3,
               'output_tokens': 2, 'reasoning_tokens': 0, 'final_tokens': 2}
        r.atomic_write(self.root/'part-000000.jsonl', json.dumps(row)+'\n')
        (self.root/'part-000001.jsonl.tmp').write_text('{incomplete')
        self.assertEqual(list(r.committed(self.root,'run-a')),['unit-1'])
        # Resume state is a slim record; the full row (reasoning, logprobs) stays on disk.
        self.assertEqual(set(r.committed(self.root,'run-a')['unit-1']), {'fingerprint','status','tokens','retry_tokens'})
        with self.assertRaises(ValueError):
            r.committed(self.root,'run-b')
        r.atomic_write(self.root/'part-000001.jsonl', json.dumps(row)+'\n')
        with self.assertRaises(ValueError):
            r.committed(self.root,'run-a')

    def test_batch_limits_resume_and_context(self):
        cfg={'system_prompt':'x','chat_template_kwargs':{'enable_thinking':False},
             'sampling':{'max_tokens':10},'engine':{'max_model_len':1000},
             'host_batch_tokens':1000,'host_batch_records':2}
        rows=[{**self.row,'unit_id':str(i)} for i in range(5)]
        batches=list(r.batches(iter(rows),FakeTokenizer(),cfg,{'0':{'fingerprint':r.fingerprint(rows[0])}}))
        self.assertEqual([len(b) for b in batches],[2,2])
        cfg['engine']['max_model_len']=10
        with self.assertRaises(ValueError):
            list(r.batches(iter(rows),FakeTokenizer(),cfg,{}))


class ReasoningTests(unittest.TestCase):
    def setUp(self):
        from jsonschema import Draft202012Validator
        self.tokenizer = FakeTokenizer()
        self.schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
                       'required': ['ok'], 'additionalProperties': False}
        self.validator = Draft202012Validator(self.schema)
        self.cfg = {'model': '/fake/qwen', 'system_prompt': 'Return JSON.',
                    'schema': self.schema, 'chat_template_kwargs': {'enable_thinking': True},
                    'engine': {'max_model_len': 20000}, 'sampling': {'max_tokens': 16384},
                    'host_batch_tokens': 20000, 'host_batch_records': 2}

    def answer(self, reasoning='Check evidence.', final='{"ok":true}', closed=True,
               finish='stop', opener=False):
        ids = ([1] if opener else []) + self.tokenizer.encode(reasoning)
        if closed:
            ids += [2] + self.tokenizer.encode(final) + [3]  # EOS
        return SimpleNamespace(token_ids=ids, text=self.tokenizer.decode(ids), finish_reason=finish)

    def validate(self, answer, enabled=True):
        return r.validate_answer(answer, [20], 1, self.tokenizer, self.validator, enabled)

    def test_thinking_uses_final_json_not_json_inside_reasoning(self):
        for opener in (False, True):
            answer = self.answer(reasoning='Candidate {"ok":false}. Recheck.', opener=opener)
            result, error, fields = self.validate(answer)
            self.assertIsNone(error)
            self.assertEqual(result, {'ok': True})
            self.assertIn('Recheck', fields['reasoning'])
            self.assertEqual(fields['reasoning_tokens'] + fields['final_tokens'], len(answer.token_ids))

    def test_unclosed_reasoning_and_length_exhaustion_never_become_results(self):
        for answer in (self.answer(reasoning='{"ok":true}', closed=False),
                       self.answer(closed=False, finish='length'),
                       self.answer(finish='length')):
            result, error, fields = self.validate(answer)
            self.assertIsNone(result)
            self.assertIsNotNone(error)
            self.assertEqual(fields['reasoning_tokens'] + fields['final_tokens'], len(answer.token_ids))

    def test_missing_malformed_and_schema_invalid_final_answers(self):
        for final in ('', '{', '{"ok":"true"}', '```json\n{"ok":true}\n```'):
            result, error, _ = self.validate(self.answer(final=final))
            self.assertIsNone(result)
            self.assertIsNotNone(error)

    def test_repeated_reasoning_boundaries_are_rejected(self):
        for ids in ([1, 1, 2], [2, 1], [2, 2]):
            answer = SimpleNamespace(token_ids=ids, text='', finish_reason='stop')
            self.assertIsNotNone(self.validate(answer)[1])

    def test_off_mode_still_validates_json_directly(self):
        text = '{"ok":true}'
        answer = SimpleNamespace(token_ids=self.tokenizer.encode(text), text=text, finish_reason='stop')
        result, error, fields = self.validate(answer, enabled=False)
        self.assertEqual(result, {'ok': True})
        self.assertIsNone(error)
        self.assertEqual(fields['reasoning_tokens'], 0)
        answer.text = '<think>reason</think>' + text
        self.assertIsNotNone(self.validate(answer, enabled=False)[1])

    def test_rendered_prompt_mismatch_still_fails(self):
        result, error, _ = r.validate_answer(self.answer(), [], 1, self.tokenizer, self.validator, True)
        self.assertIsNone(result)
        self.assertIn('Rendered token count', error)

    def test_off_mode_allows_literal_reasoning_tags_in_json(self):
        from jsonschema import Draft202012Validator
        for value in ({'text': 'Literal <think> and </think> tags.'},
                      ['<think>'], '<think>'):
            text = json.dumps(value)
            answer = SimpleNamespace(token_ids=self.tokenizer.encode(text), text=text, finish_reason='stop')
            result, error, _ = r.validate_answer(answer, [20], 1, self.tokenizer,
                                                Draft202012Validator({}), False)
            self.assertEqual(result, value)
            self.assertIsNone(error)
        answer.text = ' \n<think>unexpected</think>{"ok":true}'
        self.assertIn('Unexpected reasoning preamble', self.validate(answer, enabled=False)[1])

    def test_off_mode_preserves_schema_invalid_parsed_values(self):
        # The old runner retained any parsed value, including arrays and scalars.
        for text in ('{"ok":"true"}', '[]', 'false', '0', 'null'):
            with self.subTest(text=text):
                answer = SimpleNamespace(token_ids=self.tokenizer.encode(text), text=text, finish_reason='stop')
                result, error, _ = self.validate(answer, enabled=False)
                self.assertEqual(result, json.loads(text))
                self.assertTrue(error.startswith('ValidationError:'))

    def test_off_mode_preserves_error_precedence(self):
        for finish, ids, expected in (
                ('length', [], 'Nonterminal/length finish: length'),
                ('length', [20], 'Nonterminal/length finish: length'),
                ('stop', [], 'Rendered token count differs from engine input'),
                ('stop', [20], 'Unexpected reasoning preamble')):
            with self.subTest(finish=finish, ids=ids):
                answer = SimpleNamespace(token_ids=[1], text='<think>{', finish_reason=finish)
                result, error, _ = r.validate_answer(answer, ids, 1, self.tokenizer, self.validator, False)
                self.assertIsNone(result)
                self.assertEqual(error, 'ValueError: ' + expected)

    def test_omitted_toggle_is_rejected_before_model_load(self):
        self.cfg['chat_template_kwargs'] = {}
        with self.assertRaisesRegex(ValueError, 'enable_thinking'):
            r.engine_options(self.cfg)

    def test_multi_gpu_settings_are_rejected(self):
        for key in ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size'):
            self.cfg['engine'][key] = 2
            with self.assertRaisesRegex(ValueError, 'one-GPU'):
                r.engine_options(self.cfg)
            del self.cfg['engine'][key]

    def test_engine_gate_and_strict_toggle(self):
        original = json.dumps(self.cfg, sort_keys=True)
        options = r.engine_options(self.cfg)
        self.assertEqual(options['structured_outputs_config'],
                         {'reasoning_parser': 'qwen3', 'enable_in_reasoning': False})
        self.assertEqual(json.dumps(self.cfg, sort_keys=True), original)
        self.cfg['chat_template_kwargs']['enable_thinking'] = False
        self.assertEqual(r.engine_options(self.cfg), self.cfg['engine'])
        for value in ('false', 1, None):
            self.cfg['chat_template_kwargs']['enable_thinking'] = value
            with self.assertRaises(ValueError):
                r.engine_options(self.cfg)

    def test_conflicting_reasoning_configuration_rejected(self):
        for structured in ({'reasoning_parser': 'glm47'}, {'enable_in_reasoning': True}):
            self.cfg['engine']['structured_outputs_config'] = structured
            with self.assertRaises(ValueError):
                r.engine_options(self.cfg)

    def test_output_allowance_and_thinking_template_count_toward_context(self):
        row = {'unit_id': 'a', 'text': 'evidence', 'text_sha256': r.digest(b'evidence')}
        prompt_len = len(self.tokenizer.apply_chat_template(r.messages(row, self.cfg), enable_thinking=True))
        self.cfg['engine']['max_model_len'] = prompt_len + 16384
        self.assertEqual(len(list(r.batches([row], self.tokenizer, self.cfg, {}))), 1)
        self.cfg['engine']['max_model_len'] -= 1
        with self.assertRaisesRegex(ValueError, 'Oversize'):
            list(r.batches([row], self.tokenizer, self.cfg, {}))
        with patch.object(self.tokenizer, 'apply_chat_template', return_value=[1, 2]):
            with self.assertRaisesRegex(ValueError, 'open <think>'):
                list(r.batches([row], self.tokenizer, self.cfg, {}))
        with patch.object(self.tokenizer, 'get_vocab', return_value={}):
            with self.assertRaisesRegex(ValueError, 'requires Qwen'):
                list(r.batches([row], self.tokenizer, self.cfg, {}))

    def test_full_runner_commits_failure_and_preserves_resume_identity(self):
        # Exercise main/atomic output/resume with real JSON Schema validation;
        # only GPU/model dependencies are simulated. No GPU or network used.
        tokenizer, outer = self.tokenizer, self
        loads = []
        class FakeLLM:
            def __init__(self, **kwargs):
                loads.append(kwargs)
            def chat(self, chats, **kwargs):
                return [SimpleNamespace(
                    prompt_token_ids=tokenizer.apply_chat_template(chat, **kwargs['chat_template_kwargs']),
                    outputs=[outer.answer() if i == 0 else outer.answer(closed=False, finish='length')])
                    for i, chat in enumerate(chats)]
        modules = {
            'transformers': SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)),
            'torch': SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1, get_device_name=lambda i: 'fake'),
                                     __version__='fake', version=SimpleNamespace(cuda='fake')),
            'vllm': SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **k: k),
            'vllm.sampling_params': SimpleNamespace(StructuredOutputsParams=lambda **k: k)}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                patch.object(r.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            cfg, manifest = root/'config.json', root/'input.jsonl'
            cfg.write_text(json.dumps(self.cfg))
            manifest.write_text(''.join(json.dumps({'unit_id': str(i), 'chunk_idx': 0,
                'text': 'evidence', 'text_sha256': r.digest(b'evidence')})+'\n' for i in range(2)))
            argv = ['runner', '--config', str(cfg), '--input-manifest', str(manifest),
                    '--output-dir', str(root/'outputs'), '--worker-id', '0']
            with patch.object(sys, 'argv', argv):
                with self.assertRaises(SystemExit) as stopped:
                    r.main()
                self.assertEqual(stopped.exception.code, 2)
            worker = root/'outputs/worker-0'
            rows = [json.loads(line) for line in (worker/'part-000000.jsonl').read_text().splitlines()]
            self.assertEqual([x['status'] for x in rows], ['success', 'failed'])
            self.assertIsNone(rows[1]['result'])
            self.assertFalse(rows[1]['reasoning_complete'])
            self.assertEqual(loads[0]['structured_outputs_config']['reasoning_parser'], 'qwen3')
            before = (worker/'part-000000.jsonl').read_bytes()
            with patch.object(sys, 'argv', argv+['--resume']):
                with self.assertRaises(SystemExit):
                    r.main()
                self.assertEqual(len(loads), 1)  # committed failures are not retried
                self.assertEqual(before, (worker/'part-000000.jsonl').read_bytes())
                self.cfg['chat_template_kwargs']['enable_thinking'] = False
                cfg.write_text(json.dumps(self.cfg))
                with self.assertRaisesRegex(ValueError, 'matching identity'):
                    r.main()
            summary = json.loads((worker/'summary.json').read_text())
            self.assertEqual(summary['reasoning_tokens'] + summary['final_tokens'], summary['output_tokens'])
            self.assertNotIn('retried_units', summary)
            self.assertFalse({'attempt', 'previous_attempts', 'thinking_budget_exhausted'} & set(rows[0]))


class EndTokenizer(FakeTokenizer):
    def encode(self, text, add_special_tokens=None):
        head, sep, tail = text.partition('</think>')
        return super().encode(head) + ([2] + self.encode(tail) if sep else [])


class DecodingControlTests(unittest.TestCase):
    def setUp(self):
        self.schema = {
            'type': 'object', 'properties': {
                'evidence': {'type': 'array', 'items': {
                    'type': 'object', 'properties': {
                        'segment_id': {'type': 'string'},
                        'chunk_id': {'type': 'integer', 'minimum': 1}},
                    'required': ['segment_id', 'chunk_id'], 'additionalProperties': False}},
                'references': {'type': 'array', 'items': {
                    'type': 'object', 'properties': {
                        'evidence_ids': {'type': 'array', 'items': {'type': 'string'}}},
                    'required': ['evidence_ids'], 'additionalProperties': False}}},
            'required': ['evidence', 'references'], 'additionalProperties': False}
        self.row = {'unit_id': 'e:w0'}
        self.cfg = {'chat_template_kwargs': {'enable_thinking': True},
                    'engine': {'max_model_len': 20000,
                               'reasoning_config': {'reasoning_end_str': 'Answer now.\n</think>'}},
                    'sampling': {'max_tokens': 16384, 'thinking_token_budget': 14336,
                                 'repetition_detection': {'max_pattern_size': 256, 'min_pattern_size': 8,
                                                          'min_count': 8}},
                    'retry': {'max_attempts': 2}}

    def result(self, evidence=(), action_ids=None):
        return {'evidence': [{'segment_id': s, 'chunk_id': c} for s, c in evidence],
                'references': [] if action_ids is None else [{'evidence_ids': action_ids}]}


    def test_controls_are_validated_and_absent_controls_are_legacy(self):
        self.assertEqual(r.decoding_controls(self.cfg), {'attempts': 2, 'budget': 14336, 'temperatures': None})
        self.assertEqual(r.engine_options(self.cfg)['reasoning_config'],
                         {'reasoning_end_str': 'Answer now.\n</think>', 'reasoning_parser': 'qwen3'})
        legacy = {'chat_template_kwargs': {'enable_thinking': False}, 'engine': {}, 'sampling': {'max_tokens': 10}}
        self.assertEqual(r.decoding_controls(legacy), {'attempts': 1, 'budget': None, 'temperatures': None})
        self.assertEqual(r.decoding_controls({**self.cfg, 'retry': {'max_attempts': 3, 'temperatures': [0.2, 0.5]}}),
                         {'attempts': 3, 'budget': 14336, 'temperatures': [0.2, 0.5]})
        self.assertEqual(r.engine_options(legacy), {})
        for path, value in ((('sampling', 'thinking_token_budget'), 16384),
                            (('sampling', 'thinking_token_budget'), 0),
                            (('sampling', 'thinking_token_budget'), 1.5),
                            (('sampling', 'repetition_detection'), {'max_pattern_size': '8'}),
                            (('sampling', 'repetition_detection'), {'window': 8}),
                            (('sampling', 'repetition_detection'), {'max_pattern_size': -1}),
                            (('sampling', 'repetition_detection'), {'max_pattern_size': 4, 'min_pattern_size': 8, 'min_count': 8}),
                            (('sampling', 'repetition_detection'), {'max_pattern_size': 8, 'min_count': 1}),
                            (('sampling', 'n'), 2),
                            (('sampling', 'n'), True),
                            (('citation_constraints',), True),  # applications narrow citations
                            (('citation_constraints',), False),
                            (('retry',), {'max_attempts': 0}),
                            (('retry',), {'temperatures': []}),
                            (('retry',), {'max_attempts': 2, 'temperatures': [0.2, 0.5]}),
                            (('retry',), {'max_attempts': 2, 'temperatures': [3]}),
                            (('retry',), {'max_attempts': 2, 'reask': True})):
            with self.subTest(path=path, value=value):
                cfg = json.loads(json.dumps(self.cfg))
                target = cfg
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    r.decoding_controls(cfg)
        cfg = json.loads(json.dumps(self.cfg))
        del cfg['engine']['reasoning_config']
        with self.assertRaisesRegex(ValueError, 'requires engine.reasoning_config'):
            r.decoding_controls(cfg)
        cfg = json.loads(json.dumps(self.cfg))
        cfg['chat_template_kwargs']['enable_thinking'] = False
        with self.assertRaisesRegex(ValueError, 'requires enable_thinking'):
            r.engine_options(cfg)
        with self.assertRaisesRegex(ValueError, 'requires enable_thinking'):
            r.decoding_controls(cfg)
        cfg = json.loads(json.dumps(self.cfg))
        cfg['engine']['reasoning_config']['reasoning_parser'] = 'deepseek_r1'
        with self.assertRaisesRegex(ValueError, 'Qwen3 parser'):
            r.engine_options(cfg)

    def test_only_format_failures_are_retryable(self):
        from jsonschema import ValidationError
        cases = ((json.JSONDecodeError('Expecting value', '', 0), True),
                 (ValidationError('bad'), True),
                 (ValueError('Nonterminal/length finish: length'), True),
                 (ValueError('Nonterminal/length finish: repetition'), True),
                 (ValueError('Reasoning ended without </think>; final JSON unavailable'), True),
                 (ValueError('Rendered token count differs from engine input'), False),
                 (ValueError('Unexpected nested reasoning block'), False))
        for exc, retryable in cases:
            with self.subTest(exc=exc):
                error = r.failure(exc)
                self.assertEqual(error.retryable, retryable)
                self.assertEqual(error, f'{type(exc).__name__}: {exc}')  # error text is unchanged

    def test_budget_exhaustion_needs_the_forced_closing_string_after_a_full_budget(self):
        tokenizer, budget = EndTokenizer(), 5
        end_ids = tokenizer.encode('Answer now.\n</think>')

        def exhausted(ids):
            answer = SimpleNamespace(token_ids=ids + tokenizer.encode('{}'), text='', finish_reason='stop')
            return r.budget_exhausted(answer, r.split_answer(answer, tokenizer, True), budget, end_ids, 1)

        natural = tokenizer.encode('x' * budget) + [2]  # ends on its last budgeted token
        self.assertFalse(exhausted(natural))
        self.assertFalse(exhausted(tokenizer.encode('Answer now.\n') + [2]))  # phrase, short reasoning
        self.assertTrue(exhausted(tokenizer.encode('x' * budget + 'Answer now.\n') + [2]))
        self.assertTrue(exhausted([1] + tokenizer.encode('x' * budget + 'Answer now.\n') + [2]))
        answer = SimpleNamespace(token_ids=tokenizer.encode('x' * 50), text='', finish_reason='length')
        self.assertFalse(r.budget_exhausted(answer, r.split_answer(answer, tokenizer, True), budget, end_ids, 1))

    def test_controls_example_config_is_valid(self):
        path = Path(__file__).resolve().parents[1] / 'config/qwen3.8-27b-fp8.thinking.controls.json'
        cfg = json.loads(path.read_text())
        self.assertEqual(r.decoding_controls(cfg), {'attempts': 2, 'budget': 14336, 'temperatures': None})
        self.assertEqual(r.engine_options(cfg)['reasoning_config']['reasoning_parser'], 'qwen3')
        r.check_reasoning_end(cfg, EndTokenizer())

    def test_reasoning_end_string_must_close_with_one_end_token(self):
        tokenizer = EndTokenizer()
        r.check_reasoning_end(self.cfg, tokenizer)
        for end in ('Answer now.', '</think> then more', '</think></think>'):
            self.cfg['engine']['reasoning_config']['reasoning_end_str'] = end
            with self.assertRaisesRegex(ValueError, 'single </think>'):
                r.check_reasoning_end(self.cfg, tokenizer)

    def test_reasoning_budget_accounts_for_forced_closing_tokens(self):
        tokenizer = EndTokenizer()
        closing = len(tokenizer.encode(self.cfg['engine']['reasoning_config']['reasoning_end_str']))
        self.cfg['sampling']['max_tokens'] = 14336 + closing
        with self.assertRaisesRegex(ValueError, 'leave room for final JSON'):
            r.check_reasoning_end(self.cfg, tokenizer)
        self.cfg['sampling']['max_tokens'] += 1
        r.check_reasoning_end(self.cfg, tokenizer)
        self.cfg['engine']['reasoning_config'] = {}
        self.cfg['sampling']['max_tokens'] = 14337
        with self.assertRaisesRegex(ValueError, 'leave room for final JSON'):
            r.check_reasoning_end(self.cfg, tokenizer)

    def test_gpu_allocation_is_checked_before_device_lookup(self):
        for count in (0, 2):
            with self.subTest(count=count):
                self._run_retry_case(gpu_count=count)

    def test_full_runner_with_disabled_statistics(self):
        for explicit in (True, False):
            with self.subTest(explicit=explicit):
                self.cfg['engine']['disable_log_stats'] = True
                if not explicit:
                    del self.cfg['engine']['disable_log_stats']
                self._run_retry_case()

    def test_full_runner_retries_before_commit_with_per_unit_controls(self):
        self._run_retry_case()

    def test_retry_stops_after_signal_or_deadline_and_commits_completed_attempt(self):
        for mode in ('signal', 'deadline', 'drain', 'slurm', 'no_load', 'default_drain', 'adaptive_retry'):
            with self.subTest(mode=mode):
                self._run_retry_case(mode)

    def _run_retry_case(self, stop_mode=None, gpu_count=1):
        tokenizer = EndTokenizer()
        calls, loads = [], []
        clock, handlers = [0], {}
        final = json.dumps(self.result([('e:0', 1)]))

        def answer(final_text=final, finish='stop', reasoning='Check.'):
            ids = tokenizer.encode(reasoning) + [2] + tokenizer.encode(final_text) + [3]
            return SimpleNamespace(token_ids=ids, text=tokenizer.decode(ids), finish_reason=finish)

        class FakeLLM:
            def __init__(self, **kwargs):
                loads.append(kwargs)

            def get_metrics(self):
                if loads[-1].get('disable_log_stats', True):
                    raise AssertionError('Stat logging disabled')
                return [SimpleNamespace(name='preemptions', value=0, labels={})]

            def chat(self, chats, **kwargs):
                calls.append(kwargs['sampling_params'])
                first = len(calls) == 1
                if first and stop_mode == 'signal':
                    handlers[r.signal.SIGTERM](r.signal.SIGTERM, None)
                if first and stop_mode in ('deadline', 'drain', 'slurm'):
                    clock[0] = 2
                if first and stop_mode == 'adaptive_retry':
                    clock[0] = 600
                return [SimpleNamespace(
                    prompt_token_ids=tokenizer.apply_chat_template(chat, **kwargs['chat_template_kwargs']),
                    outputs=[answer(finish='repetition') if first and i == 0 else
                             answer(reasoning='x' * 14336 + 'Answer now.\n') if not first else answer()])
                    for i, chat in enumerate(chats)]

        def gpu_name(index):
            self.assertEqual(gpu_count, 1, 'Device lookup ran before allocation validation')
            return 'fake'

        modules = {
            'transformers': SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)),
            'torch': SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: gpu_count, get_device_name=gpu_name),
                                     __version__='fake', version=SimpleNamespace(cuda='fake')),
            'vllm': SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **k: k),
            'vllm.config': SimpleNamespace(),
            'vllm.config.reasoning': SimpleNamespace(ReasoningConfig=lambda **k: ('ReasoningConfig', k)),
            'vllm.sampling_params': SimpleNamespace(StructuredOutputsParams=lambda **k: k,
                                                    RepetitionDetectionParams=lambda **k: ('Repetition', k))}
        self.cfg.update(model='/fake/qwen', system_prompt='Return JSON.', schema=self.schema,
                        host_batch_tokens=40000, host_batch_records=2,
                        retry={'max_attempts': 2, 'temperatures': [0.7]})
        self.cfg['sampling']['seed'] = 5
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                patch.dict(os.environ, {'SLURM_JOB_END_TIME': '3' if stop_mode == 'slurm' else ''}), \
                patch.object(r.signal, 'signal', side_effect=lambda sig, fn: handlers.update({sig: fn})), \
                patch.object(r.time, 'time', side_effect=lambda: clock[0]), \
                patch.object(r.time, 'monotonic', side_effect=lambda: clock[0]), \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            cfg, manifest = root/'config.json', root/'input.jsonl'
            cfg.write_text(json.dumps(self.cfg))
            manifest.write_text(''.join(json.dumps({**self.row, 'unit_id': f'e:w{i}', 'chunk_idx': i,
                                                    'text': 'evidence', 'text_sha256': r.digest(b'evidence')})+'\n'
                                        for i in range(4 if stop_mode in ('drain', 'slurm', 'adaptive_retry') else 2)))
            argv = ['runner', '--config', str(cfg), '--input-manifest', str(manifest),
                    '--output-dir', str(root/'outputs'), '--worker-id', '0']
            if stop_mode == 'deadline':
                argv += ['--stop-at', '1970-01-01T00:00:01+00:00', '--drain-seconds', '0']
            elif stop_mode == 'drain':
                argv += ['--stop-at', '1970-01-01T00:00:03+00:00', '--drain-seconds', '1']
            elif stop_mode == 'slurm':
                argv += ['--drain-seconds', '1']
            elif stop_mode == 'adaptive_retry':
                argv += ['--stop-at', '1970-01-01T00:28:20+00:00']
            elif stop_mode == 'default_drain':
                argv += ['--stop-at', '1970-01-01T00:10:00+00:00']
            elif stop_mode == 'no_load':
                argv += ['--stop-at', '1970-01-01T00:00:03+00:00', '--drain-seconds', '4']
            with patch.object(sys, 'argv', argv):
                if gpu_count != 1:
                    with self.assertRaisesRegex(RuntimeError, 'exactly one Slurm-allocated'):
                        r.main()
                    self.assertEqual(loads, [])
                    return
                if stop_mode:
                    with self.assertRaises(SystemExit) as stopped:
                        r.main()
                    self.assertEqual(stopped.exception.code, 2)
                else:
                    r.main()  # every unit succeeded after retry: no SystemExit(2)
            worker = root/'outputs/worker-0'
            if stop_mode in ('no_load', 'default_drain'):
                self.assertEqual(loads, [])
                self.assertEqual(list(worker.glob('part-*.jsonl')), [])
                self.assertEqual(json.loads((worker/'summary.json').read_text())['unfinished'], 2)
                return
            rows = [json.loads(line) for line in (worker/'part-000000.jsonl').read_text().splitlines()]
            summary = json.loads((worker/'summary.json').read_text())
            self.assertEqual(summary['metrics']['available'], not self.cfg['engine'].get('disable_log_stats', True))
        if stop_mode:
            self.assertEqual(len(calls), 1)
            self.assertEqual([row['status'] for row in rows], ['failed', 'success'])
            self.assertEqual(summary['committed'], 2)
            self.assertEqual(summary['unfinished'], 2 if stop_mode in ('drain', 'slurm', 'adaptive_retry') else 0)
            self.assertEqual(summary['failed'], 1)
            self.assertEqual(summary['retried_units'], 0)
            return
        self.assertEqual(loads[0]['reasoning_config'],
                         ('ReasoningConfig', {'reasoning_end_str': 'Answer now.\n</think>', 'reasoning_parser': 'qwen3'}))
        self.assertEqual([len(c) for c in calls], [2, 1])  # only the failed unit is re-run
        first, retry = calls[0][0], calls[1][0]
        self.assertEqual(first['repetition_detection'], ('Repetition', self.cfg['sampling']['repetition_detection']))
        self.assertEqual((first['seed'], retry['seed']), (5, 6))
        self.assertEqual((first.get('temperature'), retry['temperature']), (None, 0.7))
        self.assertEqual(first['structured_outputs']['json'], self.schema)
        self.assertEqual([x['status'] for x in rows], ['success', 'success'])
        self.assertEqual((rows[0]['attempt'], rows[1]['attempt']), (2, 1))
        self.assertEqual(rows[0]['previous_attempts'][0]['finish_reason'], 'repetition')
        self.assertIn('raw_output', rows[0]['previous_attempts'][0])  # every attempt is recorded
        self.assertEqual(rows[1]['previous_attempts'], [])
        self.assertEqual((rows[0]['thinking_budget_exhausted'], rows[1]['thinking_budget_exhausted']), (True, False))
        self.assertEqual(summary['retried_units'], 1)
        self.assertEqual(summary['retry_output_tokens'], rows[0]['previous_attempts'][0]['output_tokens'])


if __name__ == '__main__':
    unittest.main()
