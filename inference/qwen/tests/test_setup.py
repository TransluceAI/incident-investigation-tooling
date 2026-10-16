import contextlib
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = load('submit')
download = load('download_model')


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.config = self.root / 'config.json'
        self.config.write_text('{}\n')
        self.input = self.root / 'input.jsonl'
        self.input.write_text((ROOT / 'examples/input.jsonl').read_text())

    def invoke(self, *args, active='', reject=False, run_dir=None):
        def scheduler(command, **kwargs):
            if command[0] == 'sbatch' and reject:
                raise subprocess.CalledProcessError(1, command)
            return active if command[0] == 'squeue' else '123;cluster\n'

        self.errors = io.StringIO()
        old_umask = os.umask(0o022)
        try:
            with patch.object(sys, 'argv', ['submit', '--run-dir', str(run_dir or self.run), *args]), \
                    patch.object(submit.subprocess, 'check_output', side_effect=scheduler) as call, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(self.errors):
                submit.main()
            return call
        finally:
            os.umask(old_umask)

    def fresh(self, *args, **kwargs):
        return self.invoke('--manifest', str(self.input), '--config', str(self.config),
                           '--python', sys.executable, *args, **kwargs)

    def job_start_check(self, run_dir):
        # The hash block that run.sbatch executes before launching the runner.
        manifest = (self.run / 'inputs.txt').read_text().split()[0]
        check = (self.run / 'run.sbatch').read_text().split("<<'PYHASH'\n")[1].split('\nPYHASH\n')[0]
        return subprocess.run([sys.executable, '-I', '-', str(run_dir), manifest],
                              input=check, capture_output=True, text=True)

    def test_snapshot_and_resume_preserve_runtime_and_code(self):
        call = self.fresh('--gpus', '4', '--partition', 'compute')
        command = call.call_args.args[0]
        self.assertIn('--array=0-0%1', command)
        self.assertIn('--partition=compute', command)
        self.assertEqual(command[-3], sys.executable)
        self.assertEqual(command[-1], '900')
        self.assertEqual((self.run / 'inference_runner.py').read_bytes(),
                         (ROOT / 'scripts/inference_runner.py').read_bytes())
        self.assertEqual((self.run / 'run.sbatch').read_bytes(),
                         (ROOT / 'scripts/run.sbatch').read_bytes())
        self.assertFalse((self.run / '.submission-pending').exists())
        hashes = json.loads((self.run / 'hashes.json').read_text())
        self.assertIn('python.txt', hashes)  # run files by name, inputs by absolute path
        self.assertIn(str(self.input.resolve()), hashes)
        self.config.write_text('{"changed": true}')  # original config is no longer used
        call = self.invoke('--resume')
        self.assertEqual(call.call_args.args[0][-3], sys.executable)
        self.assertEqual((self.run / 'config.json').read_text(), '{}\n')

    def test_resume_rejects_changed_files_active_jobs_and_unknown_submission(self):
        self.fresh()
        with self.assertRaises(SystemExit):
            self.invoke('--resume', active='123_0\n')
        (self.run / '.submission-pending').touch()
        with self.assertRaises(SystemExit):
            self.invoke('--resume')
        (self.run / '.submission-pending').unlink()
        (self.run / 'config.json').write_text('{"changed": true}\n')
        with self.assertRaises(SystemExit):
            self.invoke('--resume')
        self.assertIn('Saved config or code changed', self.errors.getvalue())

    def test_changed_inputs_are_rejected_at_job_start(self):
        self.fresh()
        self.input.write_text('changed\n')
        self.invoke('--resume')  # submission checks run files only; each task hashes its shard
        result = self.job_start_check(self.run)
        self.assertEqual(result.returncode, 1)
        self.assertIn('changed after submission', result.stderr)
        self.input.write_text((ROOT / 'examples/input.jsonl').read_text())
        result = self.job_start_check(self.run)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_resume_through_another_path_to_the_run_checks_the_same_files(self):
        self.fresh()
        alias = self.root / 'alias'
        alias.symlink_to(self.root)
        self.invoke('--resume', run_dir=alias / 'run')
        result = self.job_start_check(alias / 'run')
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.run / 'config.json').write_text('{"changed": true}\n')
        with self.assertRaises(SystemExit):
            self.invoke('--resume', run_dir=alias / 'run')
        self.assertIn('Saved config or code changed', self.errors.getvalue())

    def test_wall_time_must_exceed_the_drain_margin(self):
        soon = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        for args in (('--time', '00:15:00'), ('--time', '00:10:00', '--drain-seconds', '600'), ('--stop-at', soon)):
            with self.subTest(args=args):
                with self.assertRaises(SystemExit):
                    self.fresh(*args)
                self.assertIn('drain margin', self.errors.getvalue())
        self.fresh('--time', '00:15:01')

    def test_rejected_submission_leaves_no_pending_marker(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.fresh(reject=True)
        self.assertFalse((self.run / '.submission-pending').exists())
        self.invoke('--resume')  # resubmission needs no manual cleanup

    def test_resume_reports_missing_saved_files(self):
        self.fresh()
        for name in ('inputs.txt', 'hashes.json', 'config.json', 'inference_runner.py', 'run.sbatch', 'python.txt'):
            with self.subTest(name=name):
                saved = self.run / name
                original = saved.read_bytes()
                saved.unlink()
                with self.assertRaises(SystemExit) as error:
                    self.invoke('--resume')
                self.assertEqual(error.exception.code, 2)
                self.assertIn(f'Incomplete run directory: missing {name}', self.errors.getvalue())
                saved.write_bytes(original)
        self.input.unlink()
        with self.assertRaises(SystemExit):
            self.invoke('--resume')
        self.assertIn('Each input must be a nonempty file', self.errors.getvalue())

    def test_shards_use_capped_array_concurrency(self):
        shards = self.root / 'shards'
        shards.mkdir()
        for index in range(3):
            (shards / f'part-{index}.jsonl').write_text(self.input.read_text())
        call = self.invoke('--shard-dir', str(shards), '--config', str(self.config),
                           '--python', sys.executable, '--gpus', '2')
        self.assertIn('--array=0-2%2', call.call_args.args[0])

    def test_drain_margin_is_forwarded_and_validated(self):
        call = self.fresh('--drain-seconds', '90')
        self.assertEqual(call.call_args.args[0][-1], '90')
        with self.assertRaises(SystemExit):
            self.invoke('--resume', '--drain-seconds', '-1')

    def test_resume_rejects_replacement_runtime(self):
        self.fresh()
        with self.assertRaises(SystemExit):
            self.invoke('--resume', '--python', sys.executable)

    def test_python_pin_resolves_release_alias_without_bypassing_venv(self):
        release = self.root / 'release/venv/bin'
        release.mkdir(parents=True)
        (release / 'python').symlink_to(sys.executable)
        current = self.root / 'current'
        current.symlink_to(release.parents[1], target_is_directory=True)
        pinned = submit.pinned_python(current / 'venv/bin/python')
        self.assertEqual(pinned, release.resolve() / 'python')
        self.assertNotEqual(pinned, Path(sys.executable).resolve())

    def test_install_refuses_to_replace_existing_environment(self):
        runtime = self.root / 'runtime'
        (runtime / 'venv').mkdir(parents=True)
        marker = runtime / 'venv/keep'
        marker.write_text('existing runtime')
        result = subprocess.run(['bash', str(ROOT / 'scripts/install_runtime.sh'), str(runtime)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Runtime already exists', result.stderr)
        self.assertEqual(marker.read_text(), 'existing runtime')


class DownloadTests(unittest.TestCase):
    def test_profiles_resolve_to_local_configs_without_mutating_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            (snapshot / 'model.safetensors').write_bytes(b'fake test weights')
            (snapshot / 'config.json').write_text('{}')
            for profile in sorted((ROOT / 'config').glob('*.json')):
                original = profile.read_bytes()
                config = json.loads(original)
                with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(
                        snapshot_download=lambda model, **kwargs: str(snapshot))}):
                    output = download.prepare(profile, root / 'runtime')
                resolved = json.loads(output.read_text())
                self.assertEqual(resolved, {**config, 'model': str(snapshot)})
                self.assertEqual(profile.read_bytes(), original)

    def test_another_model_profile_is_selected_by_config_and_requires_a_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            (snapshot / 'model.safetensors').write_bytes(b'fake')
            profile = root / 'other-qwen.json'
            profile.write_text(json.dumps({'model': 'test/other-qwen', 'model_revision': 'a' * 40}))
            from unittest.mock import Mock
            fetch = Mock(return_value=str(snapshot))
            with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(snapshot_download=fetch)}):
                download.prepare(profile, root / 'runtime')
                self.assertEqual(fetch.call_args.args, ('test/other-qwen',))
                self.assertEqual(fetch.call_args.kwargs['revision'], 'a' * 40)
                profile.write_text(json.dumps({'model': 'test/other-qwen', 'model_revision': 'main'}))
                with self.assertRaisesRegex(ValueError, 'pinned'):
                    download.prepare(profile, root / 'runtime')
                self.assertEqual(fetch.call_count, 1)


if __name__ == '__main__':
    unittest.main()
