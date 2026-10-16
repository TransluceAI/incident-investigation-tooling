"""Contract test: prepared requests -> shared inference/qwen runner -> report.

Runs the repository's shared runner with simulated vLLM/torch/tokenizer modules.
Skips when the runner is absent; set QWEN_RUNNER to its inference_runner.py to
test against another checkout.
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from screening_judge.common import iter_jsonl, write_jsonl
from screening_judge.prepare import prepare
from screening_judge.report import reconcile, write_report
from test_pipeline import config, result

RUNNER = Path(os.environ.get('QWEN_RUNNER') or
              Path(__file__).resolve().parents[3] / 'inference/qwen/scripts/inference_runner.py')


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 100 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(i - 100) for i in ids if i >= 100)

    def get_vocab(self):
        return {'<think>': 1, '</think>': 2}

    def apply_chat_template(self, chat, **kwargs):
        return self.encode(''.join(m['content'] for m in chat))


def candidates(text, at, branch):
    """Per-character logprob tables; the label's first character gets a branch."""
    tables = []
    for i, c in enumerate(text):
        options = branch if i == at else {c: 1.0}
        tables.append({ord(t) + 100: SimpleNamespace(decoded_token=t, logprob=math.log(p), rank=rank)
                       for rank, (t, p) in enumerate(options.items(), 1)})
    return tables


@unittest.skipUnless(RUNNER.is_file(), f'shared runner not found at {RUNNER}')
class SharedRunnerContractTests(unittest.TestCase):
    def test_requests_run_and_reconcile(self):
        spec = importlib.util.spec_from_file_location('qwen_runner', RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        tokenizer, seen = CharTokenizer(), []

        class FakeLLM:
            def __init__(self, **kwargs):
                pass

            def chat(self, chats, **kwargs):
                outputs = []
                for chat, params in zip(chats, kwargs['sampling_params']):
                    schema = params['structured_outputs']['json']
                    axis = 'hacking' if 'categories' in schema['properties'] else (
                        'agentic_activity' if 'claimed_actor' in schema['properties'] else 'coordination')
                    seen.append((axis, params.get('logprobs')))
                    text = json.dumps(result(axis), separators=(',', ':'))
                    at = text.index('"label":"') + len('"label":"')
                    ids = tokenizer.encode(text)
                    answer = SimpleNamespace(token_ids=ids, text=text, finish_reason='stop',
                                             logprobs=candidates(text, at, {'n': .7, 'y': .2, 'i': .1}))
                    outputs.append(SimpleNamespace(prompt_token_ids=tokenizer.apply_chat_template(chat),
                                                   outputs=[answer]))
                return outputs

        modules = {
            'transformers': SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)),
            'torch': SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1, get_device_name=lambda i: 'fake'),
                                     __version__='fake', version=SimpleNamespace(cuda='fake')),
            'vllm': SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **k: k),
            'vllm.sampling_params': SimpleNamespace(StructuredOutputsParams=lambda **k: k)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data/resources/sample').mkdir(parents=True)
            (root / 'data/one.txt').write_text('ordinary public text')
            from screening_judge.common import sha256_text
            write_jsonl(root / 'data/resources/sample/records.jsonl',
                        [{'observation_id': 'one', 'kind': 'raw_paste', 'document_path': 'one.txt',
                          'text_sha256': sha256_text('ordinary public text')}])
            run = root / 'run'
            prepare(root / 'data', run, lambda chat: sum(len(m['content']) for m in chat), 15000,
                    {**config(), 'citation_constraints': True}, with_controls=False)
            argv = ['runner', '--config', str(run / 'engine.json'), '--input-manifest', str(run / 'requests.jsonl'),
                    '--output-dir', str(run / 'outputs'), '--worker-id', '0']
            with patch.dict(sys.modules, modules), patch.object(runner.signal, 'signal'), \
                    patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                runner.main()  # every request validated against its own schema
            parts = reconcile(run, [run / 'outputs'])
            summary = write_report(run / 'report', *parts, None)
        self.assertEqual(sorted(seen), [('agentic_activity', 10), ('coordination', 10), ('hacking', 10)])
        self.assertEqual(summary['failed'], 0)
        self.assertEqual(summary['pending'], 0)
        for judgment in parts[2]:
            self.assertEqual(judgment['validation_status'], 'validated')
            self.assertAlmostEqual(judgment['p_yes'], .2)
        self.assertEqual({a['assessment'] for item in parts[1] for a in item['axes'].values()}, {'no_evidence'})


if __name__ == '__main__':
    unittest.main()
