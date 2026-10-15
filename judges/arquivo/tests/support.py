"""Shared fixtures: runner-style output directories and a stand-in tokenizer."""
import sys
import types

from arquivo_judge.common import write_jsonl
from arquivo_judge.pipeline import runner_identity, save_json


def write_worker(outputs, cfg, rows, worker_id=0, manifest_hash='0' * 64):
    """Commit rows the way scripts/inference_runner.py does, identity file included."""
    data = {'config': cfg, 'manifest_hash': manifest_hash, 'runner_hash': '1' * 64, 'max_records': None}
    identity = runner_identity(data)
    worker = outputs / f'worker-{worker_id}'
    worker.mkdir(parents=True)
    save_json(worker / 'identity.json', {'identity': identity, **data})
    write_jsonl(worker / 'part-000000.jsonl', [{'run_identity': identity, **r} for r in rows])
    return worker


class CharTokenizer:
    """One token per character of the rendered chat; no model files needed."""

    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(m['role']) + len(m['content']) for m in messages)))


def install_fake_transformers():
    module = types.ModuleType('transformers')
    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: CharTokenizer())
    sys.modules['transformers'] = module
