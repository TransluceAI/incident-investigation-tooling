#!/usr/bin/env bash
# Run the full corpus: one token-balanced shard per GPU worker.
# The array matches shard_all.py's default eight shards; pass a matching --array to sbatch otherwise.
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00

set -euo pipefail
umask 022

: "${TOOL_DIR:?Set TOOL_DIR to the absolute judges/arquivo directory}"
: "${PREPARED_DIR:?Set PREPARED_DIR to the absolute prepared-run directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh absolute inference-output directory}"
: "${JUDGE_PYTHON:?Set JUDGE_PYTHON to the absolute GPU environment Python executable}"

worker=${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}
manifest="$PREPARED_DIR/full-shards/part-$worker.jsonl"
test -f "$manifest"

export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec srun --ntasks=1 "$JUDGE_PYTHON" "$TOOL_DIR/scripts/inference_runner.py" \
  --config "$PREPARED_DIR/config.json" \
  --input-manifest "$manifest" \
  --output-dir "$OUTPUT_DIR" \
  --worker-id "$worker" --resume
