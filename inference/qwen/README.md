# Qwen cluster inference

Standalone offline inference for compatible Qwen models using vLLM and Slurm.
Prepared JSONL shards become Slurm array tasks, with one model replica on one GPU
per task. Each worker validates structured JSON answers and commits resumable
output parts. Application prompts and JSON schemas are supplied in configuration
or in each request row. See [input and output formats](#input-and-output-formats).

## Layout

- `scripts/`: runtime installation, pinned model download, submission,
  Slurm worker, and inference runner.
- `config/`: named model profiles with synthetic text-transport prompts/schemas.
- `examples/`: synthetic text and request inputs for smoke checks.
- `tests/`: CPU tests with simulated model and scheduler dependencies.
- `requirements-gpu.txt`: pinned GPU runtime dependencies.

## Model profiles

The default profile is `Qwen/Qwen3.8-27B-FP8`, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. It targets an H100 80GB GPU with
tensor parallelism 1, 262,144 context tokens, 16 sequences, 16,384 scheduled tokens,
chunked prefill, prefix caching, 0.90 GPU utilization, and `kv_cache_dtype=auto`.
The runtime snapshot includes vLLM 0.31.0, Torch 2.13.0 and Transformers 5.17.0;
the installation script selects Python 3.12.12.

Profiles select the model, pinned revision, engine limits, sampling, thinking,
prompt and schema. To use another Qwen model, copy a profile under a new name,
change `model` and `model_revision`, and adjust memory/context settings for its
architecture and size. Download it with `--profile`, then submit the resulting
local config with `--config`. The worker loads the configured model at startup;
there is no in-process model swap. This launcher currently requires models that
fit one GPU. Thinking support requires the Qwen3 parser, `<think>`/`</think>` token
boundaries and a compatible chat template.

## Install on the cluster

Clone the repository onto shared storage visible at the same absolute path on
Slurm login and worker nodes, then check out the commit to run. Use a Linux CUDA
environment compatible with the pinned runtime and an installed `uv` command.
Installation and download happen on the login node. Inference jobs use offline
model files.

```bash
git clone <repository-url> /absolute/path/to/incident-investigation-tooling
cd /absolute/path/to/incident-investigation-tooling
git checkout --detach <commit>
cd inference/qwen
bash scripts/install_runtime.sh
.runtime/venv/bin/python scripts/download_model.py
```

This creates an ignored `.runtime/` directory containing the environment, model
cache, version manifests and resolved configs. The managed Python interpreter is
also installed there, so the environment does not depend on the installer's
private home directory. Installation refuses an existing `venv`; use a new runtime
directory for upgrades. Keep its absolute path stable. The tracked profile's
`model` is a Hugging Face repository ID; the downloader writes an absolute local snapshot path
into `.runtime/config/<profile-name>.json`. Submit the resolved config, not the
tracked template. Cache directories and their symlink targets must remain intact.

To use a separate runtime directory, pass it as the installation script's first
argument and as `download_model.py --runtime-dir`. Supply both `--python` and
`--config` to the launcher for that location. An existing cluster runtime and
model snapshot can also be reused: copy a profile, set `model` to the absolute
snapshot path, and pass that config plus the existing runtime's Python. All
paths must be readable by the worker account. Keep the environment and model
immutable for the lifetime of a resumable run.

## Prepare and submit

Prepare a JSONL manifest using [text or request rows](#input-and-output-formats).
For multiple workers, split it into nonempty, disjoint shards named `part-0.jsonl`,
`part-1.jsonl`, etc. IDs must be unique across shards. Text rows use the profile's
`system_prompt` and `schema`; request rows carry their own messages and optional
schema. The bundled examples extract synthetic marker values.

Preflight uses the real tokenizer without loading GPU weights. Pointed at the
run's `outputs` (worker N for shard N), it records its pass so the workers skip
theirs, and `submit.py` still accepts the run as fresh:

```bash
.runtime/venv/bin/python scripts/inference_runner.py \
  --config .runtime/config/qwen3.8-27b-fp8.json \
  --input-manifest examples/input.jsonl \
  --output-dir "$PWD/runs/smoke-v1/outputs" --worker-id 0 --validate-only

.runtime/venv/bin/python scripts/submit.py \
  --manifest examples/input.jsonl --run-dir "$PWD/runs/smoke-v1" \
  --gpus 1 --time 00:30:00

.runtime/venv/bin/python scripts/submit.py \
  --config /absolute/path/to/application.json \
  --shard-dir /absolute/path/to/shards --run-dir "$PWD/runs/workload-v1" \
  --gpus 4 --time 04:00:00
```

`--gpus` limits concurrent one-GPU array tasks (1–8); shard count can be larger.
Each task requests 8 CPUs and 64 GiB RAM. `--partition` defaults to `all`.
`--time` is per-task wall time. Optional `--stop-at` accepts an absolute ISO 8601
deadline with timezone and must be repeated on resume. The worker also enforces
that deadline after queueing. The runner uses the earlier of `--stop-at` and
Slurm's allocated end time. By default, it stops starting batches and retries
with a margin of at least 900 seconds (15 minutes). The margin grows to twice the longest
observed generation or batch duration when that is larger, allowing time for
ongoing generation and writes. Saved batch timings carry this estimate across
resume. `--drain-seconds` sets the minimum margin on submission or resume;
submission rejects a wall time or remaining `--stop-at` window that does not
exceed it. For jobs shorter than 15 minutes, choose a smaller minimum only when
batch timings support it. A batch that outlasts the margin can still be terminated at the hard
cutoff. New submissions snapshot the runner, batch script, configuration and
selected Python path. Input files are referenced and hashed.
Run files use mode 0755/0644 and are readable by other cluster users with access
to the parent directory. Place sensitive workloads in an appropriately restricted
parent directory.

Inspect `job-ids.txt`, `logs/`, and `outputs/worker-*/summary.json` in the run
directory. Completion requires `unfinished=0` and `failed=0`; the runner exits 2
otherwise. `squeue -u "$USER"` shows queued/running jobs, and
`sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed` shows accounting.

## Input and output formats

A manifest contains either text rows or request rows, never both.

### Text rows

- `unit_id`: unique, nonempty string.
- `chunk_idx`: nonnegative integer.
- `text`: input string.
- `text_sha256`: SHA-256 of the exact UTF-8 text.

The config supplies `system_prompt` and `schema`; the user message wraps `text`
in `<document>` tags.

### Request rows

Each line is a self-contained chat request:

```json
{"custom_id": "item7",
 "body": {"messages": [{"role": "system", "content": "..."},
                       {"role": "user", "content": "..."}],
          "response_format": {"type": "json_schema",
                              "json_schema": {"name": "response", "schema": {"type": "object"}}},
          "logprobs": true,
          "top_logprobs": 10}}
```

- `custom_id`: unique, nonempty string. Results are keyed by it; never rely on line order.
- `body.messages`: nonempty list of `{role, content}` with role `system`, `user` or `assistant`.
- `body.response_format`: optional JSON Schema for this request. Without it the
  config's `schema` applies. Validators use a bounded cache of 4,096 schemas.
- `body.logprobs` / `body.top_logprobs`: optional; `top_logprobs` (0–20) requires
  `logprobs: true`.
- Other keys, including sampling settings, are rejected: sampling, thinking and
  decoding controls come from the config profile so that one run uses one setting.

The runner hashes each body: `request_sha256` is the SHA-256 of the UTF-8 JSON
with sorted keys, `(',', ':')` separators and non-ASCII characters unescaped
(Python `json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':'))`).
Resume and application validation use this hash to bind results to requests.
Applications supply citation constraints in each request's schema and perform
semantic validation on the results.

### Result rows

Workers write `outputs/worker-<id>/part-*.jsonl`. Request results carry:

- `custom_id`, `request_sha256`, `status` (`success`/`failed`), `error`;
- `result`: the parsed JSON, or null on failure;
- `content`: the final answer text; `reasoning` and `reasoning_complete` with thinking on;
- `finish_reason` and `usage: {prompt_tokens, completion_tokens, reasoning_tokens, final_tokens}`;
- `attempts`: every attempt in order, the committed one last, each with `attempt`,
  `error`, `finish_reason` and `output_tokens`; earlier attempts also keep `raw_output`;
- `raw_output` (full generated text), `utc`, `run_identity`, and `unit_id` (equals `custom_id`);
- `thinking_budget_exhausted` when a thinking budget is configured.

When a request asked for logprobs, a successful row also has
`logprobs: {"content": [{"token", "logprob", "top_logprobs": [{"token", "logprob"}]}]}`
covering the final answer only (reasoning and trailing special tokens such as
EOS are excluded). Usage counts still include generated special tokens. Candidates
are those vLLM returned: the top `top_logprobs` plus the sampled token. Missing or
non-finite values are `null`. Missing or misaligned logprob tables fail the
request as an engine error, without a format retry. A failed row has
`logprobs: null`. Applications compute label scores from these values.

`examples/requests.jsonl` is a synthetic request for preflight checks.

## Reasoning and decoding controls

- `qwen3.8-27b-fp8.json`: thinking explicitly off, greedy (temperature 0), 2,048 output tokens.
- `qwen3.8-27b-fp8.thinking.json`: thinking on with the model card's thinking sampling
  (temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0, repetition
  penalty 1, seed 17), 16,384 output tokens.
- `qwen3.8-27b-fp8.thinking.controls.json`: additionally sets a 14,336-token thinking
  budget, repetition detection, and up to two attempts before commit.

Output budgets include reasoning and final JSON. Preflight rejects
prompt-plus-output budgets that exceed the configured context size.

The supplied profiles do not enable batch invariance and use `kv_cache_dtype=auto`.
Outputs can vary with batch composition, including at temperature 0.

Always set `chat_template_kwargs.enable_thinking` explicitly to `true` or `false`.
An omitted value is rejected. The thinking profiles use the
[model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8#best-practices) sampling settings.

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
An explicit sampling seed increments on retry, and optional `retry.temperatures`
(one value per retry) sets the temperature for each retry; greedy retries may
repeat the same answer. Resume never retries committed failures. The drain cutoff
or a SIGTERM/SIGINT sent directly to the runner prevents another batch or retry
and lets the current generation finish before committing. Slurm cancellation
and the hard deadline can terminate the engine too. Engine aborts and errors
leave the entire current batch uncommitted, including when they occur during a
retry. Resume regenerates that batch and skips earlier committed parts.

## Resume

After the previous job exits:

```bash
.runtime/venv/bin/python scripts/submit.py --resume \
  --run-dir "$PWD/runs/workload-v1" --gpus 4 --time 04:00:00
```

Resume uses the saved code, config and Python path, verifies the saved run-file
hashes at submission (each task verifies its input shard at job start), and
refuses to overlap a recorded active job. Committed parts are skipped;
an interrupted uncommitted batch may be recomputed. Before loading the model, a
worker validates every selected unit, which tokenizes the whole shard; after it
passes, `outputs/worker-*/validated.json` records the run identity and digests of
the Python version, the installed packages and the tokenizer as loaded (class,
options, vocabulary and chat template), and a resume with the same record skips
that pass. A missing, unreadable or mismatched record means a full pass;
`--validate-only` always validates and records its pass in the worker directory
unless that directory holds another run, so point a preflight's `--output-dir` at
the run's `outputs` directory to spare the workers a second pass. Changed model, config, code,
or input requires a new run directory. The launcher records the Python path,
not a hash of the full runtime/model contents; those must not be modified in place.
Directory aliases are resolved at submission so switching a release symlink does
not change the saved Python path.

After a failed run, select failed inputs into a new manifest and start a fresh run
with corrected settings. If `.submission-pending` exists, inspect `squeue`, record
any accepted job in `job-ids.txt`, and only then remove that marker before resuming.

## Tests

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -v
bash -n scripts/install_runtime.sh scripts/run.sbatch
```

The CPU tests use synthetic schemas and data with simulated model and scheduler
dependencies. They cover request validation, reasoning parsing, decoding controls,
result formatting, submission and resume. Use tokenizer preflight and a one-GPU
smoke run to check a model profile and runtime on the target cluster.

## Tuning

vLLM schedules requests within each worker. `host_batch_records` and
`host_batch_tokens` control checkpoint size and host memory; `max_num_seqs`
controls GPU request concurrency. A larger host batch amortizes overhead but
increases recomputation after interruption. Balance shards by expected token work and keep requests with
shared prompt prefixes adjacent to improve prefix-cache reuse. Resume keeps
compact per-unit hashes, status and token counts in memory; full outputs stay
on disk. Use bounded shards for very large workloads.

Tensor, pipeline and data parallel sizes must each be 1 for this launcher.

The 262K context and batch limits are specific to the supplied model profile.
Tune context, `max_num_seqs` and `max_num_batched_tokens` using representative
token lengths, throughput, preemption metrics and memory headroom. Preserve the compile cache between jobs.
Keep `kv_cache_dtype=auto` unless a separate quantized-cache trial establishes
quality and memory behavior. See vLLM's [offline inference](https://docs.vllm.ai/en/v0.31.0/serving/offline_inference/),
[structured outputs](https://docs.vllm.ai/en/v0.31.0/features/structured_outputs/),
[tuning](https://docs.vllm.ai/en/v0.31.0/configuration/optimization/), and Together's
[Slurm guide](https://docs.together.ai/docs/slurm).
