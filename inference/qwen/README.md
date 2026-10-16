# Qwen cluster inference

Standalone offline inference for compatible Qwen models using vLLM and Slurm.
Prepared JSONL shards become Slurm array tasks, with one model replica on one GPU
per task. Each worker validates structured JSON answers and commits resumable
output parts. Application prompts and JSON schemas are supplied in configuration
or in each request row; this tool has no dependency on a judge package or dataset.
Applications write requests and read results ([format](REQUESTS.md)); they do not
carry their own copy of the runner.

## Layout

- `scripts/`: release build, runtime installation, pinned model download, submission,
  Slurm worker, and inference runner.
- `config/`: named model profiles with synthetic text-transport prompts/schemas.
- `examples/`: synthetic text and request inputs for smoke checks.
- `REQUESTS.md`: the request/result row format shared with applications.
- `tests/`: CPU tests with simulated model and scheduler dependencies.
- `requirements-gpu.txt`: package-version snapshot from the existing cluster setup.

## Current profile and model selection

The starting profile is `Qwen/Qwen3.8-27B-FP8`, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. It carries forward the existing H100 80GB
setup: tensor parallelism 1, 262,144 context tokens, 16 sequences, 16,384 scheduled
tokens, chunked prefill, prefix caching, 0.90 GPU utilization, and `kv_cache_dtype=auto`.
The runtime snapshot includes vLLM 0.31.0, Torch 2.13.0 and Transformers 5.17.0;
the installation script selects Python 3.12.12. These are the inherited settings,
not a fresh compatibility certification of every package or model.

Profiles select the model, pinned revision, engine limits, sampling, thinking,
prompt and schema. To use another Qwen model, copy a profile under a new name,
change `model` and `model_revision`, and adjust memory/context settings for its
architecture and size. Download it with `--profile`, then submit the resulting
local config with `--config`. The worker loads the configured model at startup;
there is no in-process model swap. This launcher currently requires models that
fit one GPU. Thinking support requires the Qwen3 parser, `<think>`/`</think>` token
boundaries and a compatible chat template. Other Qwen architectures require
their own validation rather than assuming interchangeable behavior.

## Install on the cluster

Build each release from a commit, not from a working tree, into storage visible at
the same absolute path on the Slurm login and worker nodes. `build_release.sh`
exports `inference/qwen` at that commit with `git archive` and writes the commit to
`RELEASE`; submissions copy it into the run directory as `release.txt`. Use a Linux
CUDA environment compatible with the recorded runtime, and an installed `uv`
command. Installation and download happen on the login node; inference jobs use
offline model files.

```bash
inference/qwen/scripts/build_release.sh <commit> /absolute/path/to/releases/<name>
cd /absolute/path/to/releases/<name>
bash scripts/install_runtime.sh
.runtime/venv/bin/python scripts/download_model.py
```

This creates an ignored `.runtime/` directory containing the environment, model
cache, version manifests and resolved configs. The managed Python interpreter is
also installed there, so the environment does not depend on the installer's
private home directory. Installation refuses an existing `venv`; use a new runtime
directory for upgrades. Keep its absolute path stable. The tracked profile's `model` is a
Hugging Face repository ID; the downloader writes an absolute local snapshot path
into `.runtime/config/<profile-name>.json`. Submit the resolved config, not the
tracked template. Cache directories and their symlink targets must remain intact.
The model metadata manifest records metadata hashes and weight sizes, not full
weight checksums.

To use a separate runtime directory, pass it as the installation script's first
argument and as `download_model.py --runtime-dir`. Supply both `--python` and
`--config` to the launcher for that location. An existing cluster runtime and
model snapshot can also be reused: copy a profile, set `model` to the absolute
snapshot path, and pass that config plus the existing runtime's Python. All
paths must be readable by the worker account. Keep the environment and model
immutable for the lifetime of a resumable run.

## Prepare and submit

Each input line must contain a unique string `unit_id`, nonnegative integer
`chunk_idx`, string `text`, and `text_sha256` covering the exact UTF-8 text.
The system prompt comes from config; text is wrapped in `<document>` tags in the
user message. Alternatively, each line can be a self-contained request with
`custom_id` and a `body` holding its messages, optional JSON Schema and optional
logprobs; see [REQUESTS.md](REQUESTS.md). One manifest uses one row format.
Prepare disjoint shards named `part-0.jsonl`, `part-1.jsonl`, etc.
Global uniqueness across shards is the caller's responsibility.

Copy the resolved config and replace `system_prompt` and `schema` for your task.
The bundled prompt only extracts synthetic marker values; it is not a classifier.
Preflight uses the real tokenizer without loading GPU weights:

```bash
.runtime/venv/bin/python scripts/inference_runner.py \
  --config .runtime/config/qwen3.8-27b-fp8.json \
  --input-manifest examples/input.jsonl \
  --output-dir runs/preflight --worker-id 0 --validate-only

.runtime/venv/bin/python scripts/submit.py \
  --manifest examples/input.jsonl --run-dir "$PWD/runs/smoke-v1" \
  --gpus 1 --time 00:15:00

.runtime/venv/bin/python scripts/submit.py \
  --config /absolute/path/to/application.json \
  --shard-dir /absolute/path/to/shards --run-dir "$PWD/runs/workload-v1" \
  --gpus 4 --time 04:00:00
```

`--gpus` limits concurrent one-GPU array tasks (1–8); shard count can be larger.
Each task requests 8 CPUs and 64 GiB RAM. `--partition` defaults to `all`.
`--time` is per-task wall time. Optional `--stop-at` accepts an absolute ISO 8601
deadline with timezone and must be repeated on resume. The worker also enforces
that deadline after queueing. New submissions snapshot the runner, batch script,
configuration and selected Python path; input files are referenced and hashed.
Run files use mode 0755/0644 and are readable by other cluster users with access
to the parent directory. Place sensitive workloads in an appropriately restricted
parent directory.

The launcher resolves directory aliases such as `current/venv/bin/python` to a
specific release before saving the Python path. It retains the venv's `bin/python`
entry rather than following its final symlink to the base interpreter. Old runs
created by the pre-extraction cluster launcher must still use their original
launcher to resume; they do not contain this launcher's `python.txt` metadata.

Inspect `job-ids.txt`, `logs/`, and `outputs/worker-*/summary.json` in the run
directory. Completion requires `unfinished=0` and `failed=0`; the runner exits 2
otherwise. `squeue -u "$USER"` shows queued/running jobs, and
`sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed` shows accounting.

## Reasoning and decoding controls

- `qwen3.8-27b-fp8.json`: thinking explicitly off, greedy (temperature 0), 2,048 output tokens.
- `qwen3.8-27b-fp8.thinking.json`: thinking on with the model card's thinking sampling
  (temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition
  penalty 1, seed 17), 16,384 output tokens.
- `qwen3.8-27b-fp8.thinking.controls.json`: additionally sets a 14,336-token thinking
  budget, repetition detection, and up to two attempts before commit.

Download a named profile to resolve its model path, then submit its generated
config. These profiles do not establish optimal sampling or application accuracy.
Output budgets include reasoning and final JSON. Preflight rejects
prompt-plus-output budgets that exceed the configured context size.

Runs are not bit-reproducible, including at temperature 0: vLLM output depends on
batch composition unless batch invariance is enabled, and that is deferred (beta,
untested on this model), as is an FP8 KV cache. Reproducibility comes from keeping
the outputs: each run stores its config, runner, inputs and their hashes, and each
result row its request hash.

Always set `chat_template_kwargs.enable_thinking` explicitly to `true` or `false`.
An omitted value is rejected: Qwen's template default must not disagree with the
runner's output parser. The thinking profiles follow the
[model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8#best-practices) sampling
rather than greedy decoding. Compare thinking and non-thinking settings on labelled,
representative inputs in separate run directories. Retry count and token budgets
remain opt-in.

Thinking leaves reasoning unconstrained and applies the JSON grammar to the final
answer. The runner splits on token boundaries and records reasoning/final text
and token counts. Missing boundaries, invalid JSON, schema violations and
nonterminal/length finishes become explicit failed rows. Optional
`sampling.thinking_token_budget` requires `engine.reasoning_config` and room for
the closing string plus final JSON. `thinking_budget_exhausted` is true when the
reasoning ends with the forced closing string after the full budget; a natural stop
on the last budgeted token is not a cap (with a bare `</think>` end string the two
cannot be told apart). Only one output per unit (`sampling.n=1`) is supported.

`retry.max_attempts` (off unless set) retries format failures within the current
batch: invalid JSON, schema violations, length or repetition stops, and reasoning
that never closed. Engine, template and reasoning-boundary errors are not retried.
Every attempt is recorded with its error, finish reason, token count and raw output.
An explicit sampling seed increments on retry, and optional `retry.temperatures`
(one value per retry) raises the temperature on each retry, as olmOCR does for
repetition loops; greedy retries may repeat the same answer. Deciding to judge an
input again, such as a second pass with reasoning, belongs to the application.
Resume never retries committed failures. A stop signal or elapsed deadline prevents another
retry and commits the attempt already completed; an in-flight generation can
still be terminated by Slurm before the batch is committed.

The runner has no application logic. An application that wants citations limited
to known sources writes the narrowed JSON Schema into each request; a config with
`citation_constraints` is rejected. Semantic checks belong to the calling application.

## Resume

After the previous job exits:

```bash
.runtime/venv/bin/python scripts/submit.py --resume \
  --run-dir "$PWD/runs/workload-v1" --gpus 4 --time 04:00:00
```

Resume uses the saved code, config and Python path, verifies saved file/input
hashes, and refuses to overlap a recorded active job. Committed parts are skipped;
an interrupted uncommitted batch may be recomputed. Changed model, config, code,
or input requires a new run directory. The launcher records the Python path,
not a hash of the full runtime/model contents; those must not be modified in place.
After a failed run, select failed inputs into a new manifest and start a fresh run
with corrected settings. If `.submission-pending` exists, inspect `squeue`, record
any accepted job in `job-ids.txt`, and only then remove that marker before resuming.

## Local verification and provenance

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -v
bash -n scripts/build_release.sh scripts/install_runtime.sh scripts/run.sbatch
```

This extraction combines the cluster launcher/setup artifacts with the runner
changes from the separate `inference-improvements` worktree: reasoning parsing,
structured output, token budgets, repetition detection, retries, validation and
resume behavior. Tests use synthetic schemas and data; no judge prompts, datasets
or pilot outputs are included. This worktree has not been deployed or GPU-tested
as a new release. Use tokenizer preflight and a one-GPU smoke run before broader
cluster use. The [implementation review](docs/review-2026-10-16.md) records the
live cluster comparison, targeted fixes, and validation limits.

## Scope and tuning

Keep the offline `LLM.chat()` worker and Slurm arrays for independent document
workloads. vLLM owns request scheduling and continuous batching within each
worker; the host batch controls checkpoint size and host memory, not GPU
concurrency. A larger host batch amortizes overhead but increases recomputation
after interruption. Balance shards by expected token work and keep them large
enough to amortize model startup. Resume retains committed row metadata and
outputs in memory, so very large workloads should use bounded shards.

The current model fits one H100, so independent replicas avoid cross-GPU model
communication. Tensor/pipeline/data parallel sizes greater than one are rejected
by this launcher. For models that need multiple GPUs, change allocation and
parallelism together in a separate extension. Add an API server only if interactive
requests or persistent low-latency serving become requirements; this batch tool
does not need Ray, Kubernetes, a database, or a second scheduling service.

The 262K context and batch limits are a measured starting point for this model,
not universal defaults for every Qwen checkpoint. Tune context, `max_num_seqs`
and `max_num_batched_tokens` using representative token lengths, throughput,
preemption metrics and memory headroom. Preserve the compile cache between jobs.
Keep `kv_cache_dtype=auto` unless a separate quantized-cache trial establishes
quality and memory behavior. See vLLM's [offline inference](https://docs.vllm.ai/en/v0.31.0/serving/offline_inference/),
[structured outputs](https://docs.vllm.ai/en/v0.31.0/features/structured_outputs/),
[tuning](https://docs.vllm.ai/en/v0.31.0/configuration/optimization/), and Together's
[Slurm guide](https://docs.together.ai/docs/slurm).
