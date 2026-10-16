"""Exercise draining and resume with a real runner process and engine child.

The child substitutes for GPU generation. These tests do not simulate Slurm's
signal forwarding or prove GPU/Slurm shutdown behavior.
"""
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

from test_runner import FakeTokenizer, r


def engine_child(root):
    for call, line in enumerate(sys.stdin, 1):
        (root / 'engine-ready').touch()
        while not (root / 'release-engine').exists():
            time.sleep(0.01)
        with (root / 'generated.jsonl').open('a') as out:
            out.write(line)
        finishes = ['stop'] * len(json.loads(line))
        fault_path = root / 'fault.json'
        if fault_path.exists():
            fault = json.loads(fault_path.read_text())
            if call == fault['call']:
                finishes[-1] = fault['reason']
            elif fault.get('retry') and call == fault['call'] - 1:
                finishes[-1] = 'length'
        durations = root / 'durations.json'
        if durations.exists():
            seconds = json.loads(durations.read_text())[call - 1]
            (root / 'clock').write_text(str(float((root / 'clock').read_text()) + seconds))
        print(json.dumps(finishes), flush=True)


def run_with_engine(root, argv):
    tokenizer = FakeTokenizer()
    children = []

    class Engine:
        def __init__(self, **kwargs):
            self.child = subprocess.Popen(
                [sys.executable, '-B', __file__, 'engine', str(root)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            children.append(self.child)

        def chat(self, chats, **kwargs):
            self.child.stdin.write(json.dumps(chats) + '\n')
            self.child.stdin.flush()
            response = self.child.stdout.readline().strip()
            if not response:
                raise RuntimeError('Engine child died during generation')
            finishes = json.loads(response)
            text = '{}'
            return [SimpleNamespace(
                prompt_token_ids=tokenizer.apply_chat_template(chat),
                outputs=[SimpleNamespace(token_ids=tokenizer.encode(text), text=text, finish_reason=finish)])
                for chat, finish in zip(chats, finishes)]

    sys.modules.update({
        'transformers': SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)),
        'torch': SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1, get_device_name=lambda i: 'fake'),
                                 __version__='fake', version=SimpleNamespace(cuda='fake')),
        'vllm': SimpleNamespace(LLM=Engine, SamplingParams=lambda **k: k),
        'vllm.sampling_params': SimpleNamespace(StructuredOutputsParams=lambda **k: k)})
    # Advance the runner's wall clock deterministically while its child is busy.
    r.time.time = lambda: float((root / 'clock').read_text())
    r.time.monotonic = r.time.time
    sys.argv = ['runner', *argv]
    try:
        r.main()
    finally:
        for child in children:
            child.stdin.close()
            child.wait(timeout=5)
            child.stdout.close()


class ProcessTests(unittest.TestCase):
    def test_deadline_drains_current_batch_and_resume_skips_it(self):
        self.exercise_stop('deadline')

    def test_runner_sigterm_leaves_engine_alive_and_commits(self):
        self.exercise_stop('signal')

    def test_engine_abort_and_error_leave_batch_resumable(self):
        for reason in ('abort', 'error'):
            for retry in (False, True):
                with self.subTest(reason=reason, retry=retry), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    (root / 'clock').write_text('0')
                    (root / 'release-engine').touch()
                    (root / 'fault.json').write_text(json.dumps({
                        'call': 3 if retry else 2, 'reason': reason, 'retry': retry}))
                    config = {'model': '/fake/model', 'schema': {'type': 'object'},
                              'chat_template_kwargs': {'enable_thinking': False},
                              'engine': {'max_model_len': 100}, 'sampling': {'max_tokens': 10},
                              'host_batch_records': 2, 'host_batch_tokens': 100,
                              'retry': {'max_attempts': 2}}
                    (root / 'config.json').write_text(json.dumps(config))
                    (root / 'input.jsonl').write_text(''.join(json.dumps({
                        'custom_id': str(i), 'body': {'messages': [{'role': 'user', 'content': str(i)}]}
                    }) + '\n' for i in range(5)))
                    argv = [sys.executable, '-B', __file__, 'runner', str(root),
                            '--config', str(root / 'config.json'), '--input-manifest', str(root / 'input.jsonl'),
                            '--output-dir', str(root / 'outputs'), '--worker-id', '0']
                    env = {**os.environ, 'SLURM_JOB_END_TIME': ''}
                    interrupted = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(interrupted.returncode, 0)
                    self.assertIn(f'Engine terminated generation ({reason})', interrupted.stderr)
                    worker = root / 'outputs/worker-0'
                    first = worker / 'part-000000.jsonl'
                    original = first.read_bytes()
                    self.assertEqual(list(worker.glob('part-*.jsonl')), [first])
                    generated_before = (root / 'generated.jsonl').read_text().splitlines()
                    self.assertEqual(len(generated_before), 3 if retry else 2)
                    (root / 'fault.json').unlink()
                    resumed = subprocess.run(argv + ['--resume'], env=env, capture_output=True,
                                             text=True, timeout=10)
                    self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                    self.assertEqual(first.read_bytes(), original)
                    rows = [json.loads(line) for part in sorted(worker.glob('part-*.jsonl'))
                            for line in part.read_text().splitlines()]
                    self.assertEqual([row['custom_id'] for row in rows], list(map(str, range(5))))
                    self.assertTrue(all(row['status'] == 'success' for row in rows))
                    generated_after = (root / 'generated.jsonl').read_text().splitlines()[len(generated_before):]
                    self.assertEqual([[chat[-1]['content'] for chat in json.loads(line)]
                                      for line in generated_after], [['2', '3'], ['4']])

    def test_adaptive_margin_retains_slowest_batch_across_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'clock').write_text('0')
            (root / 'release-engine').touch()
            (root / 'durations.json').write_text('[1100, 800, 1]')
            config = {'model': '/fake/model', 'system_prompt': 'Return JSON', 'schema': {'type': 'object'},
                      'chat_template_kwargs': {'enable_thinking': False}, 'engine': {'max_model_len': 100},
                      'sampling': {'max_tokens': 10}, 'host_batch_records': 2, 'host_batch_tokens': 100}
            (root / 'config.json').write_text(json.dumps(config))
            (root / 'input.jsonl').write_text(''.join(json.dumps({
                'unit_id': str(i), 'chunk_idx': 0, 'text': str(i),
                'text_sha256': r.digest(str(i).encode())}) + '\n' for i in range(6)))
            argv = [sys.executable, '-B', __file__, 'runner', str(root),
                    '--config', str(root / 'config.json'), '--input-manifest', str(root / 'input.jsonl'),
                    '--output-dir', str(root / 'outputs'), '--worker-id', '0']
            env = {**os.environ, 'SLURM_JOB_END_TIME': '4000'}
            worker = root / 'outputs/worker-0'
            stopped = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(stopped.returncode, 2, stopped.stdout + stopped.stderr)
            progress = json.loads((worker / 'progress.json').read_text())
            self.assertEqual(progress['batch_seconds'], 800)
            self.assertEqual(progress['longest_batch_seconds'], 1100)
            summary = json.loads((worker / 'summary.json').read_text())
            self.assertEqual((summary['committed'], summary['unfinished']), (4, 2))
            generated = (root / 'generated.jsonl').read_bytes()
            # 2100 seconds remain, exceeding the default minimum but not twice
            # the observed 1100-second batch. Resume must not load another engine.
            stopped = subprocess.run(argv + ['--resume'], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(stopped.returncode, 2, stopped.stdout + stopped.stderr)
            self.assertNotIn('"event": "loading"', stopped.stdout)
            self.assertEqual((root / 'generated.jsonl').read_bytes(), generated)
            # A fresh allocation has enough time to finish the remaining batch.
            (root / 'clock').write_text('0')
            finished = subprocess.run(argv + ['--resume'], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(finished.returncode, 0, finished.stdout + finished.stderr)
            summary = json.loads((worker / 'summary.json').read_text())
            self.assertEqual((summary['committed'], summary['unfinished'], summary['failed']), (6, 0, 0))

    def exercise_stop(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'clock').write_text('0')
            config = {'model': '/fake/model', 'system_prompt': 'Return JSON', 'schema': {'type': 'object'},
                      'chat_template_kwargs': {'enable_thinking': False}, 'engine': {'max_model_len': 100},
                      'sampling': {'max_tokens': 10}, 'host_batch_records': 2, 'host_batch_tokens': 100}
            (root / 'config.json').write_text(json.dumps(config))
            (root / 'input.jsonl').write_text(''.join(json.dumps({
                'unit_id': str(i), 'chunk_idx': 0, 'text': str(i),
                'text_sha256': r.digest(str(i).encode())}) + '\n' for i in range(3)))
            argv = [sys.executable, '-B', __file__, 'runner', str(root),
                    '--config', str(root / 'config.json'), '--input-manifest', str(root / 'input.jsonl'),
                    '--output-dir', str(root / 'outputs'), '--worker-id', '0', '--drain-seconds', '100']
            env = {**os.environ, 'SLURM_JOB_END_TIME': '1000'}
            worker = root / 'outputs/worker-0'
            with subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, start_new_session=True) as process:
                try:
                    end = time.monotonic() + 10
                    while not (root / 'engine-ready').exists() and process.poll() is None:
                        if time.monotonic() >= end:
                            self.fail('Engine did not start')
                        time.sleep(0.01)
                    self.assertIsNone(process.poll(), 'Runner exited before generation')
                    if mode == 'deadline':
                        (root / 'clock').write_text('900')
                    else:
                        process.send_signal(signal.SIGTERM)
                        # Reading the event confirms the real handler ran before releasing generation.
                        while True:
                            line = process.stdout.readline()
                            self.assertTrue(line, 'Runner exited before handling SIGTERM')
                            if json.loads(line)['event'] == 'stop_requested':
                                break
                    (root / 'release-engine').touch()
                    stdout, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 2, stdout + stderr)
                finally:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
            summary = json.loads((worker / 'summary.json').read_text())
            self.assertEqual((summary['committed'], summary['unfinished'], summary['failed']), (2, 1, 0))
            first = worker / 'part-000000.jsonl'
            original = first.read_bytes()
            (root / 'clock').write_text('0')
            env['SLURM_JOB_END_TIME'] = '3000'
            resumed = subprocess.run(argv + ['--resume'], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertEqual(first.read_bytes(), original)
            generated = [json.loads(line) for line in (root / 'generated.jsonl').read_text().splitlines()]
            self.assertEqual([len(batch) for batch in generated], [2, 1])
            summary = json.loads((worker / 'summary.json').read_text())
            self.assertEqual((summary['committed'], summary['unfinished']), (3, 0))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'engine':
        engine_child(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == 'runner':
        run_with_engine(Path(sys.argv[2]), sys.argv[3:])
    else:
        unittest.main()
