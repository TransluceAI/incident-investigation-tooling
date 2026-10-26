# Activity Screening Judge

An LLM judge for screening preserved public-web evidence across datasets. It independently assesses hacking attempts, autonomous AI-agent activity, and agent coordination, then produces source-cited judgments and review queues.

The judge supports captures and redirects, scanner reports, wiki content, messages, and general text. It preserves request context, decodes embedded URLs and programs as inert text, and splits long items into overlapping windows that each repeat the item's context and provenance (or, when provenance is too long, its leading lines). Numbered source chunks let the model cite locations without copying escaped text. It never executes evidence or follows evidence URLs.

This is exploratory evidence triage. Labels require human review, and source-location validation does not establish classification accuracy or a successful compromise.

## Code

```text
config/core.txt         Shared evidence and attribution rules
config/axes/            Separate hacking, agent-activity, and coordination rubrics
config/shapes/          Guidance for each evidence format
src/screening_judge/    Adapters, schemas, preparation, requests, label scores, reporting, and evaluation
tests/                  CPU regression tests and synthetic corpus fixtures
```

Inference uses [`inference/qwen`](../../inference/qwen/README.md). `prepare` writes one [request](../../inference/qwen/README.md#request-rows) per (window, axis); `report` validates the results and computes label scores. Model profiles control engine, sampling, thinking, and decoding settings.

## Install

Python 3.10+ is required. Reporting and evaluation run on a CPU with `jsonschema`. Exact preparation additionally needs the model tokenizer. Install and run inference with [`inference/qwen`](../../inference/qwen/README.md) on the cluster.

```sh
cd judges/screening
python -m pip install -e .              # Report and evaluate existing outputs
python -m pip install -e '.[prepare]'   # Also prepare inputs with Transformers
```

Pass a resolved runner profile as `--engine`, for example `inference/qwen/.runtime/config/qwen3.8-27b-fp8.json` after its model download. Its `model` must be the local model path, because preparation counts tokens with that tokenizer. The `model_revision` field records the intended snapshot; it does not verify the weights at that path. The profile's transport `system_prompt` and `schema` are not used: each request carries the judge's own prompts and per-axis schema. Preparation writes the profile's engine and sampling settings to `engine.json` for the runner.

The baseline profile disables thinking; the `thinking` and `thinking.controls` profiles enable it.

Each request's schema is narrowed to the window's citable segments and chunk numbers, so constrained decoding cannot emit an unknown citation; `--no-citation-constraints` sends the plain axis schema instead. Prompts include field descriptions generated from the response schema. `--label-logprobs K` (default 10, 0 disables) sets the logprob candidates requested for label scores.

## Prepare and run

Run the runner commands below from `inference/qwen`. `DATA_ROOT` must contain an export matching [the input contract](#input-format). Keep run artifacts outside the repository.

```sh
export DATA_ROOT=/absolute/path/to/exported-corpus
export ENGINE_CONFIG=/absolute/path/to/inference/qwen/.runtime/config/qwen3.8-27b-fp8.json
export RUN_ROOT=/absolute/path/to/new-screening-run
export PREPARED_DIR="$RUN_ROOT/prepared"

screening-judge prepare \
  --data-root "$DATA_ROOT" --engine "$ENGINE_CONFIG" \
  --out "$PREPARED_DIR" --window-tokens 160000 --shards 8 --workers 8

# From inference/qwen: tokenizer preflight on one shard, then submit every shard.
.runtime/venv/bin/python scripts/inference_runner.py \
  --config "$PREPARED_DIR/engine.json" \
  --input-manifest "$PREPARED_DIR/shards/part-0.jsonl" \
  --output-dir "$RUN_ROOT/preflight" --worker-id 0 --validate-only

.runtime/venv/bin/python scripts/submit.py \
  --config "$PREPARED_DIR/engine.json" --shard-dir "$PREPARED_DIR/shards" \
  --run-dir "$RUN_ROOT/inference" --gpus 8 --time 04:00:00
export OUTPUT_DIR="$RUN_ROOT/inference/outputs"
```

`--shards` controls the shard count. Shards are named `part-N.jsonl` for the launcher's `--shard-dir`; a window's three axis requests stay together and adjacent so they share the prefix cache. For one worker, prepare with `--shards 1` and submit `$PREPARED_DIR/requests.jsonl` with `--manifest`. Synthetic controls are included by default and stay separate from corpus item counts.

`--workers` sets how many processes build windows. Window search re-tokenizes each candidate prompt, so one process is CPU-bound on corpora with long documents. Workers take contiguous ranges of items and their output is joined in item order, so every prepared file and hash is byte-identical to a one-process run with the same boundary salt (each run draws a new one). Give the job at least that many CPUs; preparation warns when `--workers` exceeds the CPUs the process may use (its affinity mask, capped by the job's Slurm CPU allocation). Workers are forked, so use it on Linux with Python 3.12 or later.

Use a fresh preparation directory for each run. Oversized requests are rejected before inference. See the runner README for [resume](../../inference/qwen/README.md#resume) and [decoding controls](../../inference/qwen/README.md#reasoning-and-decoding-controls).

For CPU-only preparation without a tokenizer, add `--estimate-tokens 4` to the preparation command. This is a layout smoke test with estimated token counts, not validation of model compatibility or context limits.

## Report

Reporting can run on a CPU, including before inference to inspect pending work:

```sh
screening-judge report \
  --run "$PREPARED_DIR" --outputs "$OUTPUT_DIR" \
  --out "$RUN_ROOT/report"
```

- `summary.json`: item/request counts, validation status, control results, and assessments by axis and shape. `requests_expected`, `validated` and label-score diagnostics include controls; item counts and reviewer files exclude them.
- `item-results.jsonl`: per-item assessments and completion status for all three axes.
- `judgments.jsonl`: all committed judgments, including synthetic controls, validation failures and resolved evidence. Cascade reconciliation reads this complete file.
- `review-queue.jsonl`: every request needing a human: verified positives on any axis, unverified positives, abstentions, and failures, each with its `queue_reason`, in that order. Within each reason it is sorted by descending `confidence` (missing scores last), then by `p_yes`. An item can contribute several rows.
- `failures.jsonl`, `pending.jsonl`, `unverified-candidates.jsonl`: failures, unfinished requests, and unresolved positive claims kept separate from negatives.
- `near-misses.jsonl`, `injection-review.jsonl`: additional candidates for review.
- `control-results.jsonl`: synthetic controls, excluded from corpus item counts. Request totals include controls. Reviewer files exclude controls; `summary.controls.failed`, `.pending` and `.unverified_candidates` track their execution issues separately. Runs prepared with `--no-controls` have no control checks and zero control counts. In a pass-2 report, control axes that were not selected are untested: they count in `controls.not_requested`, not in `passed` or `total`.

Reports check prepared artifact hashes, worker configuration against `engine.json`, membership in each worker's selected manifest, and that each result's `request_sha256` matches its prepared request. Positive judgments require schema validity and resolved citations; action references must cite resolved evidence. The model's free-text `location` is unverified. Archive replay responses cannot establish attack success.

Every axis answer includes `confidence`, the model's 0–1 estimate that the evidence shows the behaviour that axis asks about. It ranks evidence, not certainty in the chosen label: a confidently negative answer should score near 0. `summary.json` counts validated negatives with confidence above 0.5 under `confidence_on_negatives`. Item results report `confidence_positive_max`, the highest confidence among positive answers, per axis. `label_probs` and `p_yes` are diagnostic first-token branch scores. Neither score is a calibrated probability. The report computes them from the runner's raw final-answer logprobs; incomplete logprobs produce no score.

## Two-pass cascade

The cascade runs all requests with reasoning off and selected requests with reasoning on:

```sh
screening-judge escalate \
  --run "$PREPARED_DIR" --report "$RUN_ROOT/report" \
  --engine /absolute/path/to/inference/qwen/.runtime/config/qwen3.8-27b-fp8.thinking.controls.json \
  --out "$RUN_ROOT/pass2" --audit-rate 0.03 --shards 8
# Run inference on "$RUN_ROOT/pass2" exactly as for pass 1, then:
screening-judge report --run "$RUN_ROOT/pass2" --outputs "$RUN_ROOT/pass2-inference/outputs" --out "$RUN_ROOT/pass2-report"
screening-judge cascade \
  --pass1-report "$RUN_ROOT/report" --pass2-run "$RUN_ROOT/pass2" \
  --pass2-report "$RUN_ROOT/pass2-report" --out "$RUN_ROOT/cascade"
```

`escalate` selects:

- Positives, unverified positives, abstentions, and failures on any axis.
- Negatives with P(yes) in the `--borderline` band (default 0.2–0.8).
- Every axis of windows matching a pass-0 rule.
- A seeded random `--audit-rate` sample of remaining negatives.

Pass-0 rules match non-default backend ports, Host-header or TLS-server-name
overrides, `dav/versions` paths, and CAPTCHA/OCR chains in one URL or program.
They inspect request context, excluding bodies and body-derived URL views;
`dav/versions` is matched everywhere. A rule hit triggers escalation, not a label.

The pass-2 directory uses the standard prepared-run format with the selected
requests and its own engine profile. `selection.jsonl` records selection reasons.
Both commands verify report/run identity.

`cascade` takes validated pass-2 answers and otherwise retains pass 1:

| Output | Contents |
| --- | --- |
| `final-judgments.jsonl` | Merged corpus judgments |
| `review-queue.jsonl` | One row per request needing review, with `queue_reasons` for both passes; the highest-priority answer determines its position |
| `demoted.jsonl` | Pass-1 positives that pass 2 no longer supports |
| `control-judgments.jsonl` | Synthetic controls, separate from corpus judgments and queues |
| `pending.jsonl` | Unfinished corpus requests when `--allow-pending` is used, with `pass` identifying the unfinished pass |
| `summary.json` | Promotions, demotions, failures, and observed positive flips in validated audits |

Both commands reject pending requests in their input pass unless `--allow-pending`
is set, including pending controls. Pending pass-1 requests remain explicit
`validation_status: "pending"` rows with `pass1_pending`; pending pass-2 requests
retain their pass-1 answer and are marked `pass2_pending`. Synthetic controls
stay in `control-judgments.jsonl`, with pending counts under `summary.controls`.
If pass 2 fails, the retained pass-1 judgment includes the failed `pass2` result;
a queue row led by pass 2 includes its `pass1` result. Audit flips are observations, not estimates of total misses or
recall. Shards never outnumber selected windows; an empty selection needs no
inference run and writes no shards.

## Tests

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests use synthetic data and simulated runner outputs; no model, GPU, or network access is required. `test_shared_runner.py` runs prepared requests through the shared runner with a simulated engine; it skips when `inference/qwen` is absent, and `QWEN_RUNNER=/path/to/inference_runner.py` selects an alternative runner path.

## Input format

The export root contains `resources/*/records.jsonl` or recursive `resources/**/DIRECT_RECORDS.jsonl`, one observation per line. Each observation needs a string `observation_id` and `kind`; when `observation_id` is absent, a string `id` is used, as in `DIRECT_RECORDS.jsonl` exports. Multi-record events use source-specific join adapters in `prepare.ADAPTERS`. The `arquivo.pt` adapter joins CDX rows, capture bodies, and replay redirects by original URL and capture time. Other sources use individual records. Stored text is referenced by `document_path` relative to the export root, with its UTF-8 SHA-256 in `text_sha256`. Records may also carry source, timestamp, and other provenance metadata.

Adapters map recognized kinds to capture, scanner, wiki, message, or text guidance; unknown kinds retain their text. Adding a dataset may require adapting its record format or kind mapping in `records.py` and `shapes.py`. Scanner views preserve nonempty scalar fields with JSON pointers. Captures with different request URLs remain separate even when their bodies match; other items merge only when source, shape, body hash, and evidence metadata match.

Empty exports fail visibly. Missing or corrupt referenced documents remain incomplete, and malformed or unlinked records appear in `gaps.jsonl`. Prepared evidence, schemas, prompts, settings, and implementation hashes are recorded with the run. Model weights and runtime dependencies require separate verification.

A neutral benchmark manifest can be used instead of source records:

```sh
screening-judge prepare \
  --data-root /absolute/path/to/benchmark-package \
  --benchmark-manifest benchmark/inputs.jsonl \
  --engine "$ENGINE_CONFIG" --out /absolute/path/to/new-benchmark-run
```

Rows require `id`, `document_path`, and `text_sha256`, with optional `metadata`; document paths resolve against `--data-root`. Only `source_family` and `format` metadata enter prompt provenance. Case IDs, paths, labels, and review notes stay outside prompts, and preparation never opens the answer key. Cases remain separate even when bodies match.

## Evaluation

```sh
screening-judge evaluate \
  --results "$RUN_ROOT/report/item-results.jsonl" \
  --key /absolute/path/to/answer_key.jsonl --out "$RUN_ROOT/evaluation"
```

Each key row requires a unique `id` and a `document_sha256`, with labels in `overall_label` and/or `axis_labels`. Axis label keys are `hacking`, `agentic_activity`, and `collusion` (mapped to coordination). `overall` means any of the three axes. Missing axis labels remain unlabeled. Optional `leakage_group` and `evaluation_partition` fields define related cases and explicit train/dev/test assignments; repeated document hashes are allowed and remain in the same partition. Conflicting assignments fail.

Evaluation reports coverage, missing/ambiguous/undecided cases, conditional TPR/FPR/precision/MCC (decided cases only, so read coverage alongside them), end-to-end positive recall, and a `projection` of precision and review load at assumed prevalences, which holds only if the measured rates transfer. Ambiguous shared-body joins require an explicit `item_id`, or a matching benchmark case ID. Confidence intervals resample connected leakage groups; degenerate intervals do not establish zero error.

Test metrics are excluded unless explicitly requested with `evaluate --include-test` for a frozen candidate. Holdout data must be independent of model and prompt tuning; assigning a partition does not establish that independence.

## Limitations and run artifacts

Windows preserve text but can separate related context; cross-document reasoning is not implemented.

For a model run, record the code revision, model snapshot, runtime versions, dataset identity, configuration, completed/failed/pending counts, control outcomes, and token/runtime totals alongside links to external artifacts. Accuracy claims additionally need the labeling method, split provenance, coverage, and per-axis metrics. Corpus files, weights, raw outputs, and reports belong outside this repository.
