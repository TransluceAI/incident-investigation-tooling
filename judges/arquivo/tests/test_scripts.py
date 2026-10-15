"""shard_all.py and audit.py on a prepared synthetic run."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from arquivo_judge.common import read_jsonl, write_jsonl
from arquivo_judge.pipeline import prepare
from support import install_fake_transformers

TOOL = Path(__file__).resolve().parents[1]


def script(name, *args):
    return subprocess.run([sys.executable, '-O', str(TOOL / 'scripts' / name), *map(str, args)],
                          capture_output=True, text=True)


class ScriptTests(unittest.TestCase):
    def setUp(self):
        install_fake_transformers()
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / 'engine.json').write_text((TOOL / 'config/engine.example.json').read_text())
        self.run_dir = root / 'prepared'
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(SimpleNamespace(data_root=TOOL / 'examples/corpus', engine=root / 'engine.json',
                                    rubric=TOOL / 'config/rubric.txt', schema=TOOL / 'config/schema.json',
                                    out=self.run_dir))

    def tearDown(self):
        self.temp.cleanup()

    def test_audit_checks_the_shards_the_launcher_runs(self):
        result = script('audit.py', self.run_dir)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('run scripts/shard_all.py first', result.stderr)
        result = script('shard_all.py', '--run', self.run_dir, '--out', self.run_dir / 'full-shards', '--shards', 3)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list((self.run_dir / 'full-shards').glob('part-*.jsonl'))), 3)
        result = script('audit.py', self.run_dir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['shards'], 3)

    def test_audit_fails_under_optimization_when_a_unit_is_unassigned(self):
        self.assertEqual(script('shard_all.py', '--run', self.run_dir, '--out', self.run_dir / 'full-shards').returncode, 0)
        parts = sorted((self.run_dir / 'full-shards').glob('part-*.jsonl'))
        rows = [(p, read_jsonl(p)) for p in parts]
        part, units = next((p, r) for p, r in rows if r)
        write_jsonl(part, units[1:])
        result = script('audit.py', self.run_dir)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('do not cover every unit', result.stderr)


if __name__ == '__main__':
    unittest.main()
