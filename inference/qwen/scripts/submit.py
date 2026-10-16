"""Submit offline Qwen inference as one-GPU Slurm array tasks."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess

RELEASE = Path(__file__).resolve().parent.parent


def pinned_python(path):
    # Resolve release-directory aliases, but retain the venv's bin/python entry:
    # resolving that final symlink would bypass the virtual environment.
    path = path.expanduser().absolute()
    return path.parent.resolve(strict=True) / path.name


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--manifest', type=Path)
    source.add_argument('--shard-dir', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--python', type=Path, help='Cluster Python with vLLM installed; saved for resume')
    p.add_argument('--partition', default='all')
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--gpus', type=int, choices=range(1, 9), default=1)
    p.add_argument('--time', default='01:00:00', help='Per-task HH:MM:SS')
    p.add_argument('--stop-at', help='Optional absolute deadline in ISO 8601 with timezone; no default')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    user = pwd.getpwuid(os.getuid()).pw_name
    run = args.run_dir.expanduser().absolute()
    if run.is_symlink() or (run.exists() and run.stat().st_uid != os.getuid()):
        p.error('Run directory must belong to you and cannot be a symlink')
    match = re.fullmatch(r'(\d+):([0-5]\d):([0-5]\d)', args.time)
    if not match:
        p.error('Use --time HH:MM:SS')
    h, m, s = map(int, match.groups())
    duration = h * 3600 + m * 60 + s
    if duration < 1:
        p.error('--time must be positive')
    if args.stop_at:
        try:
            deadline = datetime.fromisoformat(args.stop_at.replace('Z', '+00:00'))
        except ValueError:
            p.error('--stop-at must be an ISO 8601 timestamp')
        if deadline.tzinfo is None:
            p.error('--stop-at must include its timezone')
        remaining = int(deadline.timestamp() - datetime.now(timezone.utc).timestamp())
        if remaining < 60:
            p.error('--stop-at must be at least one minute in the future')
        duration = min(duration, remaining)
    if args.resume:
        if args.config or args.manifest or args.shard_dir or args.python:
            p.error('Resume uses saved inputs/config; omit new input flags')
        paths = [Path(line) for line in (run/'inputs.txt').read_text().splitlines()]
        for name in ['config.json', 'inference_runner.py', 'run.sbatch', 'python.txt']:
            if not (run/name).is_file():
                p.error(f'Incomplete run directory: missing {name}')
    else:
        if run.exists() and any(run.iterdir()):
            p.error('Choose a fresh run directory or use --resume')
        if args.manifest:
            paths = [args.manifest.resolve(strict=True)]
        elif args.shard_dir:
            files = list(args.shard_dir.glob('part-*.jsonl'))
            paths = [args.shard_dir/f'part-{i}.jsonl' for i in range(len(files))]
            if not paths or set(paths) != set(files):
                p.error('Use consecutive shard names: part-0.jsonl, part-1.jsonl, ...')
            paths = [path.resolve(strict=True) for path in paths]
        else:
            p.error('Provide --manifest or --shard-dir')
    if not paths or len(set(paths)) != len(paths):
        p.error('Input paths must be nonempty and distinct')
    if any('\n' in str(path) or not path.is_file() or path.stat().st_size == 0 for path in paths):
        p.error('Each input must be a nonempty file with no newline in its path')
    config = args.config or RELEASE/'.runtime/config/qwen3.8-27b-fp8.json'
    try:
        python = (Path((run/'python.txt').read_text().strip()) if args.resume else
                  pinned_python(args.python or RELEASE/'.runtime/venv/bin/python'))
    except OSError as exc:
        p.error(f'Python runtime is unavailable; install it or pass --python: {exc}')
    if not python.is_file() or not os.access(python, os.X_OK):
        p.error('Python runtime is missing or not executable; install it or pass --python')
    if not args.resume and not config.is_file():
        p.error('Config file is missing')
    os.umask(0o022)
    shared = Path('/data/qwen-inference/runs')
    if run.is_relative_to(shared):
        parts = run.relative_to(shared).parts
        if len(parts) < 2 or parts[0] != user:
            p.error('Use /data/qwen-inference/runs/<your username>/<run name>')
        parent = shared/user
        parent.mkdir(exist_ok=True, mode=0o755)
        if parent.is_symlink() or parent.stat().st_uid != os.getuid():
            p.error('Shared user directory has an unexpected owner')
        parent.chmod(0o755)
    run.mkdir(parents=True, exist_ok=True, mode=0o755)
    run.chmod(0o755)
    with (run/'.submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipts = run/'job-ids.txt'
        if receipts.exists():
            active = subprocess.check_output(['squeue','-h','-u',user,'-o','%i'], text=True)
            active_ids = {line.strip().split('_')[0] for line in active.splitlines()}
            if active_ids.intersection(receipts.read_text().splitlines()):
                p.error('A previous job for this run is still active')
        if (run/'.submission-pending').exists():
            p.error('Previous submission outcome is unknown: inspect squeue, then remove .submission-pending before retrying')
        (run/'logs').mkdir(exist_ok=True, mode=0o755)
        if not args.resume:
            if (run/'inputs.txt').exists():
                p.error('Run was already prepared; use --resume')
            shutil.copyfile(config, run/'config.json')
            shutil.copyfile(RELEASE/'scripts/inference_runner.py', run/'inference_runner.py')
            shutil.copyfile(RELEASE/'scripts/run.sbatch', run/'run.sbatch')
            (run/'python.txt').write_text(str(python)+'\n')
            (run/'inputs.txt').write_text(''.join(str(path)+'\n' for path in paths))
            # A release built by build_release.sh records its commit; keep it with the run.
            saved = ['config.json','inference_runner.py','run.sbatch','inputs.txt','python.txt']
            if (RELEASE/'RELEASE').is_file():
                shutil.copyfile(RELEASE/'RELEASE', run/'release.txt')
                saved.append('release.txt')
            hashes = {str(path): sha(path) for path in paths}
            hashes.update({str(run/name):sha(run/name) for name in saved})
            (run/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
        hashes = json.loads((run/'hashes.json').read_text())
        if any(sha(path) != expected for path,expected in hashes.items()):
            p.error('Saved config, code or input changed; use a new run directory')
        command = ['sbatch','--parsable',f'--partition={args.partition}','--nodes=1','--ntasks=1',
                   '--gres=gpu:1','--cpus-per-task=8','--mem=64G',
                   f'--time={duration//3600:02}:{duration//60%60:02}:{duration%60:02}',
                   f'--array=0-{len(paths)-1}%{min(args.gpus,len(paths))}',
                   '--job-name=qwen-batch',f'--chdir={run}',f'--output={run}/logs/%A_%a.log',
                   '--export=NIL',str(run/'run.sbatch'),str(run),
                   str(python),args.stop_at or '']
        (run/'.submission-pending').touch()
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unknown submission response: inspect squeue before retrying')
        with receipts.open('a') as f:
            f.write(job+'\n')
            f.flush()
            os.fsync(f.fileno())
        (run/'.submission-pending').unlink()
        print(f'Job {job}: {len(paths)} shard(s), at most {min(args.gpus,len(paths))} GPU(s)\nRun: {run}')


if __name__ == '__main__':
    main()
