import json
import os
import signal
import tempfile
import time
import unittest
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest import mock

from screening_judge.common import iter_jsonl, sha256_text, write_jsonl
from screening_judge.prepare import prepare, usable_cpus
from test_pipeline import config


def count(chat):
    return sum(len(m['content']) for m in chat) // 3


class ParallelPrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        rows = {'alpha': [], 'beta': []}
        for n in range(14):
            body = f'ordinary text {n} ' * (2000 if n % 5 == 0 else 3)
            if n == 3:  # an unknown kind whose body makes it a scanner item during preparation
                body = json.dumps({'task': {'url': 'https://example.test/'}, 'page': {'n': n}})
            path = self.data / f'doc{n}.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
            rec = {'observation_id': f'obs{n}', 'kind': 'unknown_kind' if n == 3 else 'raw_paste',
                   'document_path': path.name, 'text_sha256': sha256_text(body)}
            if n == 7:
                rec['text_sha256'] = 'bad'
            if n == 9:
                rec['document_path'] = 'missing.txt'
            rows['alpha' if n % 2 else 'beta'].append(rec)
        for source, recs in rows.items():
            write_jsonl(self.data / f'resources/{source}/records.jsonl', recs)

    def run_prepare(self, name, **kw):
        with mock.patch('screening_judge.prepare.secrets.token_hex', return_value='0' * 32):
            return prepare(self.data, self.root / name, count, 6000, config(), shards=2, **kw)

    def test_workers_write_the_same_bytes_as_one_process(self):
        serial = self.run_prepare('serial')
        parallel = self.run_prepare('parallel', workers=3)
        self.assertEqual(parallel, serial)
        self.assertGreater(serial['windowed_items'], 0)
        self.assertEqual(serial['excluded_integrity'], 2)
        self.assertIn('scanner', {i['shape'] for i in iter_jsonl(self.root / 'serial/items.jsonl')
                                  if i['source'] == 'alpha'})
        files = sorted(p.relative_to(self.root / 'serial') for p in (self.root / 'serial').rglob('*') if p.is_file())
        self.assertEqual(files, sorted(p.relative_to(self.root / 'parallel')
                                       for p in (self.root / 'parallel').rglob('*') if p.is_file()))
        for rel in files:
            self.assertEqual((self.root / 'parallel' / rel).read_bytes(),
                             (self.root / 'serial' / rel).read_bytes(), str(rel))

    def test_more_workers_than_cpus_warns_and_still_prepares(self):
        serial = self.run_prepare('serial')
        with mock.patch('screening_judge.prepare.usable_cpus', return_value=2):
            with self.assertWarnsRegex(RuntimeWarning, 'workers=3 exceeds the 2 CPUs'):
                parallel = self.run_prepare('parallel', workers=3)
        self.assertEqual(parallel, serial)

    def test_slurm_allocation_caps_usable_cpus(self):
        def cpus(**env):
            with mock.patch.dict(os.environ):
                for name in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
                    os.environ.pop(name, None)
                os.environ.update(env)
                return usable_cpus()
        mask = cpus()
        if mask is None:
            self.skipTest('CPU count unknown on this platform')
        self.assertEqual(cpus(SLURM_CPUS_PER_TASK='1'), 1)
        self.assertEqual(cpus(SLURM_CPUS_ON_NODE='1'), 1)
        self.assertEqual(cpus(SLURM_CPUS_PER_TASK='3', SLURM_CPUS_ON_NODE='2'), min(2, mask))
        for value in (str(mask + 50), '', 'x', '0'):
            self.assertEqual(cpus(SLURM_CPUS_PER_TASK=value), mask, value)

    def test_workers_need_python_312(self):
        with mock.patch('screening_judge.prepare.sys.version_info', (3, 11, 9)):
            with self.assertRaisesRegex(ValueError, 'Python 3.12'):
                self.run_prepare('old', workers=2)
            self.run_prepare('serial')  # one process is unaffected

    def test_a_failing_worker_publishes_no_requests(self):
        def failing(chat):
            if 'ordinary text 11' in chat[-1]['content']:
                raise RuntimeError('tokenizer failed')
            return count(chat)
        with self.assertRaisesRegex(RuntimeError, 'tokenizer failed'):
            prepare(self.data, self.root / 'failed', failing, 6000, config(), workers=3)
        self.assertFalse((self.root / 'failed/requests.jsonl').exists())
        self.assertFalse((self.root / 'failed/windows.jsonl').exists())

    def test_a_failure_stops_a_blocked_sibling(self):
        def blocked(chat):
            if 'ordinary text 1 ' in chat[-1]['content']:
                time.sleep(3600)
            if 'ordinary text 11 ' in chat[-1]['content']:
                raise RuntimeError('tokenizer failed')
            return count(chat)
        def hung(signum, frame):
            raise AssertionError('preparation waited for a blocked worker')
        previous = signal.signal(signal.SIGALRM, hung)
        self.addCleanup(signal.signal, signal.SIGALRM, previous)
        self.addCleanup(signal.alarm, 0)
        signal.alarm(20)
        with self.assertRaisesRegex(RuntimeError, 'tokenizer failed'):
            prepare(self.data, self.root / 'blocked', blocked, 6000, config(), workers=3)
        self.assertFalse((self.root / 'blocked/requests.jsonl').exists())
        self.assertFalse((self.root / 'blocked/.parts').exists())

    def test_a_killed_worker_fails_instead_of_hanging(self):
        def killed(chat):
            if 'ordinary text 11' in chat[-1]['content']:
                os._exit(1)  # as the kernel OOM killer would
            return count(chat)
        with self.assertRaises(BrokenProcessPool):
            prepare(self.data, self.root / 'killed', killed, 6000, config(), workers=3)
        self.assertFalse((self.root / 'killed/requests.jsonl').exists())

    def test_workers_must_be_a_positive_integer(self):
        for workers in (0, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'workers'):
                prepare(self.data, self.root / 'bad', count, 6000, config(), workers=workers)


if __name__ == '__main__':
    unittest.main()
