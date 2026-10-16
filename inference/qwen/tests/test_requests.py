"""Request rows: each line carries its own messages, schema and logprob options."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_runner import FakeTokenizer, r

SCHEMA = {'type': 'object', 'properties': {'label': {'enum': ['yes', 'no']}},
          'required': ['label'], 'additionalProperties': False}
OTHER = {'type': 'object', 'properties': {'score': {'type': 'integer'}},
         'required': ['score'], 'additionalProperties': False}


def request(custom_id, content='evidence', schema=SCHEMA, **body):
    return {'custom_id': custom_id,
            'body': {'messages': [{'role': 'system', 'content': 'Judge.'},
                                  {'role': 'user', 'content': content}],
                     'response_format': {'type': 'json_schema',
                                         'json_schema': {'name': 'judgment', 'schema': schema}},
                     **body}}


class RequestRowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def manifest(self, rows):
        path = self.root / 'requests.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def test_request_rows_normalize_with_a_content_hash(self):
        row = request('a', logprobs=True, top_logprobs=5)
        unit, = r.units(self.manifest([row]))
        self.assertEqual((unit['unit_id'], unit['custom_id']), ('a', 'a'))
        self.assertEqual(unit['request_sha256'], r.request_hash(row['body']))
        self.assertEqual(r.messages(unit, {}), row['body']['messages'])
        self.assertEqual(r.row_schema(unit, {}), SCHEMA)
        self.assertEqual(r.top_logprobs(unit), 5)
        # OpenAI batch lines may also carry method/url.
        openai = {**request('b'), 'method': 'POST', 'url': '/v1/chat/completions'}
        self.assertEqual(next(r.units(self.manifest([openai])))['custom_id'], 'b')

    def test_example_request_is_valid(self):
        example = Path(__file__).resolve().parents[1] / 'examples/requests.jsonl'
        unit, = r.units(example)
        self.assertEqual(r.top_logprobs(unit), 5)

    def test_malformed_requests_are_rejected(self):
        bad = [
            {**request('a'), 'extra': 1},
            {**request('a'), 'url': '/v1/embeddings'},
            {'custom_id': '', 'body': request('a')['body']},
            request('a', temperature=0.7),  # sampling belongs to the config
            {'custom_id': 'a', 'body': {'messages': []}},
            {'custom_id': 'a', 'body': {'messages': [{'role': 'tool', 'content': 'x'}]}},
            {'custom_id': 'a', 'body': {'messages': [{'role': 'user', 'content': 1}]}},
            {'custom_id': 'a', 'body': {**request('a')['body'], 'response_format': {'type': 'json_object'}}},
            request('a', top_logprobs=5),
            request('a', logprobs=True, top_logprobs=21),
            request('a', logprobs='yes'),
            {'custom_id': 'a', 'body': {**request('a')['body'], 'response_format': 5}},
        ]
        for row in bad:
            with self.subTest(row=row), self.assertRaises(ValueError):
                list(r.units(self.manifest([row])))

    def test_duplicate_ids_and_mixed_manifests_are_rejected(self):
        text = {'unit_id': 't', 'chunk_idx': 0, 'text': 'x', 'text_sha256': r.digest(b'x')}
        for rows in ([request('a'), request('a')], [request('a'), text], [text, request('a')]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                list(r.units(self.manifest(rows)))

    def test_schema_falls_back_to_config_and_citations_need_text_rows(self):
        body = {'messages': [{'role': 'user', 'content': 'x'}]}
        unit, = r.units(self.manifest([{'custom_id': 'a', 'body': body}]))
        self.assertEqual(r.row_schema(unit, {'schema': OTHER}), OTHER)
        with self.assertRaisesRegex(ValueError, 'No response_format'):
            r.row_schema(unit, {})
        cfg = {'citation_constraints': True, 'chat_template_kwargs': {'enable_thinking': False},
               'sampling': {'max_tokens': 10}, 'engine': {'max_model_len': 1000},
               'host_batch_tokens': 1000, 'host_batch_records': 2}
        with self.assertRaisesRegex(ValueError, 'citation_constraints'):
            list(r.batches(iter([unit]), FakeTokenizer(), cfg, {}))

    def test_final_logprobs_cover_only_the_answer_and_never_invent_values(self):
        lp = lambda token, value, rank: SimpleNamespace(decoded_token=token, logprob=value, rank=rank)
        tables = [{7: lp('reason', -0.1, 1)}, {2: lp('</think>', 0.0, 1)},
                  {8: lp('{', -0.2, 1), 9: lp('[', -2.0, 2)},
                  {10: lp('}', float('nan'), 1)}]
        answer = SimpleNamespace(token_ids=[7, 2, 8, 10], logprobs=tables)
        content = r.final_logprobs(answer, 2)['content']
        self.assertEqual([c['token'] for c in content], ['{', '}'])
        self.assertEqual(content[0]['top_logprobs'], [{'token': '{', 'logprob': -0.2},
                                                      {'token': '[', 'logprob': -2.0}])
        self.assertIsNone(content[1]['logprob'])  # NaN is not valid JSON or evidence
        tables[3] = {11: lp('x', -1.0, 1)}  # sampled token absent from the table
        self.assertEqual(r.final_logprobs(answer, 2)['content'][1]['token'], None)
        for missing in (None, tables[:2]):
            answer.logprobs = missing
            self.assertIsNone(r.final_logprobs(answer, 2))

    def test_full_runner_uses_each_requests_messages_schema_and_logprobs(self):
        tokenizer = FakeTokenizer()
        calls, loads = [], []

        def candidates(text):
            return [{100 + ord(c): SimpleNamespace(decoded_token=c, logprob=math.log(.5), rank=1),
                     1000: SimpleNamespace(decoded_token='~', logprob=math.log(.25), rank=2)} for c in text]

        def answer(text, logprobs=None):
            ids = tokenizer.encode(text)
            return SimpleNamespace(token_ids=ids, text=text, finish_reason='stop', logprobs=logprobs)

        class FakeLLM:
            def __init__(self, **kwargs):
                loads.append(kwargs)

            def chat(self, chats, **kwargs):
                calls.append((chats, kwargs['sampling_params']))
                outputs = []
                for chat, params in zip(chats, kwargs['sampling_params']):
                    # The second request's own schema needs a score, so a label fails it.
                    text = '{"label":"yes"}'
                    logprobs = candidates(text) if 'logprobs' in params else None
                    outputs.append(SimpleNamespace(
                        prompt_token_ids=tokenizer.apply_chat_template(chat, **kwargs['chat_template_kwargs']),
                        outputs=[answer(text, logprobs)]))
                return outputs

        modules = {
            'transformers': SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)),
            'torch': SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1, get_device_name=lambda i: 'fake'),
                                     __version__='fake', version=SimpleNamespace(cuda='fake')),
            'vllm': SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **k: k),
            'vllm.sampling_params': SimpleNamespace(StructuredOutputsParams=lambda **k: k)}
        cfg = {'model': '/fake/qwen', 'chat_template_kwargs': {'enable_thinking': False},
               'engine': {'max_model_len': 20000}, 'sampling': {'max_tokens': 100, 'temperature': 0},
               'host_batch_tokens': 20000, 'host_batch_records': 8}
        rows = [request('w0:hacking', 'first', logprobs=True, top_logprobs=3),
                request('w0:coordination', 'second', schema=OTHER)]
        with patch.dict(sys.modules, modules), patch.object(r.signal, 'signal'), \
                contextlib.redirect_stdout(io.StringIO()):
            config, manifest = self.root / 'config.json', self.manifest(rows)
            config.write_text(json.dumps(cfg))
            argv = ['runner', '--config', str(config), '--input-manifest', str(manifest),
                    '--output-dir', str(self.root / 'outputs'), '--worker-id', '0']
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit) as stopped:
                r.main()
            self.assertEqual(stopped.exception.code, 2)  # the second request failed its schema
            worker = self.root / 'outputs/worker-0'
            out = [json.loads(line) for line in (worker / 'part-000000.jsonl').read_text().splitlines()]
            with patch.object(sys, 'argv', argv + ['--resume']), self.assertRaises(SystemExit):
                r.main()
            self.assertEqual(len(loads), 1)  # committed requests are not rerun
        (chats, params), = calls
        self.assertEqual(chats, [rows[0]['body']['messages'], rows[1]['body']['messages']])
        self.assertEqual([p['structured_outputs']['json'] for p in params], [SCHEMA, OTHER])
        self.assertEqual(params[0]['logprobs'], 3)
        self.assertNotIn('logprobs', params[1])
        self.assertEqual([o['custom_id'] for o in out], ['w0:hacking', 'w0:coordination'])
        self.assertEqual([o['status'] for o in out], ['success', 'failed'])
        self.assertEqual(out[0]['request_sha256'], r.request_hash(rows[0]['body']))
        self.assertFalse({'text_sha256', 'chunk_idx'} & set(out[0]))
        content = out[0]['logprobs']['content']
        self.assertEqual(''.join(c['token'] for c in content), '{"label":"yes"}')
        self.assertEqual(len(content[0]['top_logprobs']), 2)
        self.assertNotIn('logprobs', out[1])


if __name__ == '__main__':
    unittest.main()
