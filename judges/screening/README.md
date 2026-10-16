# Activity Screening Judge

An LLM judge for screening preserved public-web evidence across datasets. It independently assesses hacking attempts, autonomous AI-agent activity, and agent coordination, then produces source-cited judgments and review queues.

The judge supports captures and redirects, scanner reports, wiki content, messages, and general text. It preserves request context, decodes embedded URLs and programs as inert text, and splits long items into overlapping windows that each repeat the item's context and provenance (or, when provenance is too long, its leading lines). Numbered source chunks let the model cite locations without copying escaped text. It never executes evidence or follows evidence URLs.

This is exploratory evidence triage. Labels require human review, and source-location validation does not establish classification accuracy or a successful compromise. The local framework is tested; model accuracy and calibration have not yet been measured.

## Code

```text
config/core.txt         Shared evidence and attribution rules
config/axes/            Separate hacking, agent-activity, and coordination rubrics
config/shapes/          Guidance for each evidence format
src/screening_judge/    Adapters, schemas, preparation, requests, label scores, reporting, and evaluation
tests/                  CPU regression tests and synthetic corpus fixtures
```

The package is `screening-judge`, its Python module is `screening_judge`, and its installed CLI is `screening-judge`. The wheel includes the rubrics.

The judge has no GPU code. Inference uses the repository's shared runner, [`inference/qwen`](../../inference/qwen/): `prepare` writes one self-contained request per (window, axis) in its [request format](../../inference/qwen/REQUESTS.md), the runner returns one result per request, and `report` validates those results and computes label scores. Engine, sampling, thinking, and decoding settings come from the runner's model profiles.

## Install

Python 3.10+ is required. Reporting and evaluation run on a CPU with `jsonschema`. Exact preparation additionally needs the model tokenizer. Install and run inference with [`inference/qwen`](../../inference/qwen/README.md) on the cluster.

```sh
cd judges/screening
python -m pip install -e .              # Report and evaluate existing outputs
python -m pip install -e '.[prepare]'   # Also prepare inputs with Transformers
```

Pass a resolved runner profile as `--engine`, for example `inference/qwen/.runtime/config/qwen3.8-27b-fp8.json` after its model download. Its `model` must be the local model path, because preparation counts tokens with that tokenizer. The `model_revision` field records the intended snapshot; it does not verify the weights at that path. The profile's transport `system_prompt` and `schema` are not used: each request carries the judge's own prompts and per-axis schema. Preparation writes the profile's engine and sampling settings to `engine.json` for the runner.

The baseline profile disables thinking; the `thinking` and `thinking.controls` profiles enable it. Settings need comparative evaluation before choosing a default.

Each request's schema is narrowed to the window's citable segments and chunk numbers, so constrained decoding cannot emit an unknown citation; `--no-citation-constraints` sends the plain axis schema instead. Each prompt also describes the response fields one by one after the axis question, as vLLM's structured-output guide recommends, generated from the same schema. `--label-logprobs K` (default 10, 0 disables) sets the logprob candidates requested for label scores.

## Prepare and run

Run `screening-judge` from this tool's directory and the runner commands from `inference/qwen`. `DATA_ROOT` must contain an export matching [the input contract](#input-format); the tool does not collect or crawl sources. Keep run artifacts outside the repository.

```sh
export DATA_ROOT=/absolute/path/to/exported-corpus
export ENGINE_CONFIG=/absolute/path/to/inference/qwen/.runtime/config/qwen3.8-27b-fp8.json
export RUN_ROOT=/absolute/path/to/new-screening-run
export PREPARED_DIR="$RUN_ROOT/prepared"

screening-judge prepare \
  --data-root "$DATA_ROOT" --engine "$ENGINE_CONFIG" \
  --out "$PREPARED_DIR" --window-tokens 160000 --shards 8

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

Use a fresh preparation directory for each experiment. See the runner README for resume, deadlines, and decoding controls. The runner counts prompt tokens and rejects oversized requests rather than truncating them. Unclosed reasoning blocks, length-limited answers and schema-invalid answers are failures.

For CPU-only preparation without a tokenizer, add `--estimate-tokens 4` to the preparation command. This is a layout smoke test with estimated token counts, not validation of model compatibility or context limits.

## Report

Reporting can run on a CPU, including before inference to inspect pending work:

```sh
screening-judge report \
  --run "$PREPARED_DIR" --outputs "$OUTPUT_DIR" \
  --out "$RUN_ROOT/report"
```

- `summary.json`: item/request counts, validation status, control results, and assessments by axis and shape.
- `item-results.jsonl`: per-item assessments and completion status for all three axes.
- `judgments.jsonl`: committed judgments, including validation failures and resolved evidence for validated results.
- `review-queue.jsonl`: every request needing a human: verified positives on any axis, unverified positives, abstentions, and failures, each with its `queue_reason`. It is sorted by descending `confidence` (missing scores last), then by `p_yes`. An item can contribute several rows.
- `failures.jsonl`, `pending.jsonl`, `unverified-candidates.jsonl`: failures, unfinished requests, and unresolved positive claims kept separate from negatives.
- `near-misses.jsonl`, `injection-review.jsonl`: additional candidates for review.
- `control-results.jsonl`: synthetic controls, excluded from corpus item counts. Request totals include controls.

Reports check prepared artifact hashes, worker configuration against `engine.json`, membership in each worker's selected manifest, and that each result's `request_sha256` matches its prepared request. Positive judgments require schema validity and resolved citations; action references must cite resolved evidence. The model's free-text `location` is unverified. Archive replay responses cannot establish attack success.

Every axis answer includes `confidence`, the model's 0–1 estimate that the evidence shows the behaviour that axis asks about, as in the Arquivo judge's hacking confidence. It ranks evidence, not certainty in the chosen label: a confidently negative answer should score near 0. Item results report `confidence_max` per axis. `label_probs` and `p_yes` are diagnostic first-token branch scores. Neither score is a calibrated probability. The report computes them from the runner's raw final-answer logprobs; incomplete logprobs produce no score. Optional `--thresholds /path/to/thresholds.json` applies exploratory development-set ranking bands to already verified positives; it never promotes negative or abstaining judgments. Bands are bound to the classifier and inference implementation. Without applicable bands, positives remain `positive_uncalibrated`.

## Two-pass cascade

Run every request once with reasoning off, then re-judge only the requests that matter with reasoning on:

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

`escalate` selects pass-1 positives on any axis, unverified positives, abstentions, failures, negatives with P(yes) in the `--borderline` band (default 0.2–0.8), every axis of a window that a pass-0 rule flags, and a seeded random `--audit-rate` sample of the remaining negatives. Pass-0 rules (`rules.py`) are cheap triggers for patterns pass 1 under-flagged: non-default backend ports, request code overriding the Host header or TLS server name, `dav/versions` paths, and a CAPTCHA endpoint with an OCR API in the same URL or program. A full-corpus scan found every true positive in request or CDX URLs and their decoded views, and the noise in bodies, so the rules read only non-body segments (except the precise `dav/versions`). They are not yet measured against labelled data. The pass-2 directory is an ordinary prepared run over that subset of the same requests with the pass-2 runner profile, and `selection.jsonl` records each request's reasons. Escalation refuses a pass 1 with pending requests unless `--allow-pending` is given.

`cascade` uses the pass-2 answer wherever pass 2 validated one and otherwise keeps pass 1. It writes `final-judgments.jsonl`, `review-queue.jsonl`, and `demoted.jsonl`: pass-1 positives that pass 2 no longer supports, kept visible at low priority until calibration shows reasoning does not lose real attacks. Pass-2 answers that fail validation leave the pass-1 classification in place but still enter the review queue, so an uncited pass-2 yes is not lost. Synthetic controls go to `control-judgments.jsonl` and stay out of the queues and counts. `summary.json` counts demotions, promotions and pass-2 failures, and how many audited pass-1 negatives pass 2 validated and answered yes; these are observed flips, not an estimate of total misses. Each report's judgments carry the hash of their prepared run (`preparation_sha256`, which covers the requests and runner profile), and `escalate`/`cascade` refuse reports from any other run. Shards never outnumber windows, and an escalation that selects nothing writes no shards. Cost relative to reasoning on everything, and whether the cascade matches its recall, still need measuring on labelled data.

## Tests

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite uses synthetic data and simulated runner outputs. It covers evidence preservation, provenance in every window, request contents, citation narrowing and resolution, label and confidence scores, the review queue, the two-pass cascade and pass-0 rules, worker assignment, failure handling, benchmark isolation, and evaluation joins. It requires no model, GPU, or network access. `test_shared_runner.py` runs prepared requests through the shared runner with a simulated engine; it skips when `inference/qwen` is absent, and `QWEN_RUNNER=/path/to/inference_runner.py` selects another checkout.

## Input format

The export root contains `resources/*/records.jsonl` or recursive `resources/**/DIRECT_RECORDS.jsonl`, one observation per line. Each observation needs a string `observation_id` and `kind`; when `observation_id` is absent, a string `id` is used, as in `DIRECT_RECORDS.jsonl` exports. Each self-contained record needs no source-specific code. A source whose export splits one event across several records registers a join adapter in `prepare.ADAPTERS`, keyed by source name. The only one is `arquivo.pt`, whose CDX index rows, capture bodies and replay redirects are joined into one event per original URL and capture time; the same kinds from any other source are ordinary items. Stored text is referenced by `document_path` relative to the export root, with its UTF-8 SHA-256 in `text_sha256`. Records may also carry source, timestamp, and other provenance metadata.

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

screening-judge compare \
  --a /absolute/path/to/baseline/item-results.jsonl \
  --b "$RUN_ROOT/report/item-results.jsonl" \
  --key /absolute/path/to/answer_key.jsonl --axis overall --split dev
```

Each key row requires a unique `id` and a `document_sha256`, with labels in `overall_label` and/or `axis_labels`. Axis label keys are `hacking`, `agentic_activity`, and `collusion` (mapped to coordination). `overall` means any of the three axes. Missing axis labels remain unlabeled. Optional `leakage_group` and `evaluation_partition` fields define related cases and explicit train/dev/test assignments; repeated document hashes are allowed and remain in the same partition. Conflicting assignments fail.

Evaluation reports coverage, missing/ambiguous/undecided cases, conditional TPR/FPR/precision/MCC, and end-to-end positive recall. Ambiguous shared-body joins require an explicit `item_id`, or a matching benchmark case ID. Confidence intervals resample connected leakage groups; degenerate intervals do not establish zero error.

Test metrics are excluded unless explicitly requested with `evaluate --include-test` or `compare --split test` for a frozen candidate. The existing 600-case benchmark was described as a development set: reserving a partition now does not establish historical independence. `thresholds.json` contains exploratory development ranking bands, not probability calibration or a release criterion.

## Recorded results

Local verification on 2026-10-16: **66 CPU tests pass**, including the shared-runner contract test against the `inference/qwen` runner from the separate inference branch; without that runner the contract test is skipped. Earlier, the renamed package built successfully as a wheel; its bundled policies and CLI entry point were checked from an extracted installation outside the checkout.

No actual-model accuracy or calibration results are recorded for this judge yet. GPU compatibility, injection resistance, repeatability, and full-corpus memory/throughput remain unmeasured. Windows preserve text but can separate related context; cross-document reasoning is not implemented. Arquivo experiment results do not establish this judge's performance.

For a model run, record the code revision, model snapshot, runtime versions, dataset identity, configuration, completed/failed/pending counts, control outcomes, and token/runtime totals alongside links to external artifacts. Accuracy claims additionally need the labeling method, split provenance, coverage, and per-axis metrics. Corpus files, weights, raw outputs, and reports belong outside this repository.
