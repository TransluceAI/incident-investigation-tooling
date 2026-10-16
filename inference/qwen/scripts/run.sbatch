#!/bin/bash
set -euo pipefail
umask 022
run=$1
python=$2
stop_at=${3:-}
drain_seconds=${4:-900}
worker=${SLURM_ARRAY_TASK_ID:-0}
mapfile -t inputs < "$run/inputs.txt"
manifest=${inputs[$worker]}
export PATH="${python%/*}:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
worker_home=$("$python" -I -c 'import os,pwd; print(pwd.getpwuid(os.getuid()).pw_dir)')
export USER=$(id -un) LOGNAME=$(id -un)
export LANG=C.UTF-8 LC_ALL=C.UTF-8 SLURM_EXPORT_ENV=ALL
unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$worker_home/.cache/qwen-inference"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$XDG_CACHE_HOME"
command -v ninja >/dev/null
"$python" -I - "$run" "$manifest" <<'PYHASH'
import hashlib,json,sys
from pathlib import Path
run=Path(sys.argv[1])
expected=json.loads((run/'hashes.json').read_text())
for path in [Path(sys.argv[2])]+[run/name for name in ['config.json','inference_runner.py','run.sbatch','inputs.txt','python.txt']]:
    with path.open('rb') as source:
        actual=hashlib.file_digest(source,'sha256').hexdigest()
    if actual != expected[str(path)]:
        raise RuntimeError(f'Input/config/code changed after submission: {path}')
PYHASH
ulimit -n 65536
command=(srun --ntasks=1 "$python" -I "$run/inference_runner.py" \
  --config "$run/config.json" --input-manifest "$manifest" \
  --output-dir "$run/outputs" --worker-id "$worker" --resume --drain-seconds "$drain_seconds")
if [[ -n "$stop_at" ]]; then
  command+=(--stop-at "$stop_at")
  remaining=$("$python" -I -c 'import sys,time; from datetime import datetime; print(int(datetime.fromisoformat(sys.argv[1].replace("Z","+00:00")).timestamp()-time.time()))' "$stop_at")
  test "$remaining" -gt 0
  # The runner drains before this hard cutoff. TERM asks srun to terminate
  # the step, including engine children; kill-after only bounds srun cleanup.
  exec timeout --signal=TERM --kill-after=60s "${remaining}s" "${command[@]}"
fi
exec "${command[@]}"
