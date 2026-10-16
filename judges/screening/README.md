# Activity Screening Judge

An LLM judge for screening preserved public-web evidence across datasets. It independently assesses hacking attempts, autonomous AI-agent activity, and agent coordination, then produces source-cited judgments and review queues.

The judge supports captures and redirects, scanner reports, wiki content, messages, and general text. It preserves request context, decodes embedded URLs and programs as inert text, and splits long items into overlapping windows. Numbered source chunks let the model cite locations without copying escaped text. It never executes evidence or follows evidence URLs.

This is exploratory evidence triage. Labels require human review, and source-location validation does not establish classification accuracy or a successful compromise. The local framework is tested; model accuracy and calibration have not yet been measured.

## Code

```text
config/core.txt         Shared evidence and attribution rules
config/axes/            Separate hacking, agent-activity, and coordination rubrics
config/shapes/          Guidance for each evidence format
config/engine*.json     Engine and sampling settings templates
src/screening_judge/    Adapters, schemas, preparation, reporting, and evaluation
scripts/inference_runner.py
                        Offline, resumable vLLM inference runner
tests/                  CPU regression tests and synthetic corpus fixtures
```

The package is `screening-judge`, its Python module is `screening_judge`, and its installed CLI is `screening-judge`. The wheel includes rubrics and example configs; run the inference script from the source checkout.

## Install

Python 3.10+ is required. Reporting and evaluation run on a CPU with `jsonschema`. Exact preparation additionally needs the model tokenizer; inference needs a compatible Linux CUDA environment with vLLM and enough GPU memory for the configured model and context.

```sh
cd judges/screening
python -m pip install -e .              # Report and evaluate existing outputs
python -m pip install -e '.[prepare]'   # Also prepare inputs with Transformers
# On the GPU host, use a model-compatible vLLM environment:
python -m pip install -e '.[gpu]'
```

Copy `config/engine.example.json` to a local configuration outside the repository and set `model` to the absolute path of your local weights and tokenizer. The examples target Qwen 27B. The `model_revision` field records the intended snapshot; it does not verify the weights at that path.

The baseline disables thinking. `engine.thinking.example.json` enables it; the two `engine.vendor-*.example.json` files provide alternative sampling settings for experiments. These configurations need comparative evaluation before choosing a default. Thinking mode also requires validation of the reasoning parser and structured-output support in the installed vLLM version.

## Prepare and run

Run these commands from this tool's directory. `DATA_ROOT` must contain an export matching [the input contract](#input-format); the tool does not collect or crawl sources. Keep run artifacts outside the repository.

```sh
export DATA_ROOT=/absolute/path/to/exported-corpus
export ENGINE_CONFIG=/absolute/path/to/engine.local.json
export RUN_ROOT=/absolute/path/to/new-screening-run
export PREPARED_DIR="$RUN_ROOT/prepared"
export OUTPUT_DIR="$RUN_ROOT/outputs"

screening-judge prepare \
  --data-root "$DATA_ROOT" --engine "$ENGINE_CONFIG" \
  --out "$PREPARED_DIR" --window-tokens 160000 --shards 8

python scripts/inference_runner.py \
  --config "$PREPARED_DIR/config.json" \
  --input-manifest "$PREPARED_DIR/shards/manifest-0000.jsonl" \
  --output-dir "$OUTPUT_DIR" --worker-id 0 --validate-only

# After validation, run inference on the same shard:
python scripts/inference_runner.py \
  --config "$PREPARED_DIR/config.json" \
  --input-manifest "$PREPARED_DIR/shards/manifest-0000.jsonl" \
  --output-dir "$OUTPUT_DIR" --worker-id 0
```

These runner commands cover shard 0 only. Run each remaining shard with its own worker ID to complete the corpus. `--shards` controls the shard count; each worker currently expects exactly one visible CUDA device. For one worker, prepare with `--shards 1` and use `$PREPARED_DIR/manifest.jsonl` directly. Synthetic controls are included by default and stay separate from corpus item counts.

Use a fresh preparation directory for each experiment. Add `--resume` only after the previous worker exits and with identical configuration, manifest, and runner code. Resume skips committed failures as well as successes; it does not retry them. The runner counts prompt tokens and rejects oversized inputs rather than truncating them. Unclosed reasoning blocks and length-limited answers are failures.

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
- `review-queue.jsonl`: positive windows with resolved citations; an item can contribute multiple windows.
- `failures.jsonl`, `pending.jsonl`, `unverified-candidates.jsonl`: failures, unfinished requests, and unresolved positive claims kept separate from negatives.
- `near-misses.jsonl`, `injection-review.jsonl`: additional candidates for review.
- `control-results.jsonl`: synthetic controls, excluded from corpus item counts. Request totals include controls.

Reports check prepared artifact hashes, worker configuration, and membership in each worker's selected manifest. Positive judgments require schema validity and resolved citations; action references must cite resolved evidence. The model's free-text `location` is unverified. Archive replay responses cannot establish attack success.

`label_probs` and `p_yes` are diagnostic first-token branch scores, not calibrated probabilities. Incomplete logprobs produce no score. Optional `--thresholds /path/to/thresholds.json` applies exploratory development-set ranking bands to already verified positives; it never promotes negative or abstaining judgments. Bands are bound to the classifier and inference implementation. Without applicable bands, positives remain `positive_uncalibrated`.

## Tests

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite uses synthetic data and a simulated inference engine. It covers evidence preservation, citation resolution, worker assignment, failure and resume handling, benchmark isolation, and evaluation joins. It requires no model, GPU, or network access.

## Input format

The export root contains `resources/*/records.jsonl` or recursive `resources/**/DIRECT_RECORDS.jsonl`, one observation per line. Each observation needs a string `observation_id` and `kind`. Stored text is referenced by `document_path` relative to the export root, with its UTF-8 SHA-256 in `text_sha256`. Records may also carry source, timestamp, and other provenance metadata.

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

Local verification on 2026-10-16: **47 CPU tests pass**. The renamed package also built successfully as a wheel; its bundled policies and CLI entry point were checked from an extracted installation outside the checkout.

No actual-model accuracy or calibration results are recorded for this judge yet. GPU compatibility, injection resistance, repeatability, and full-corpus memory/throughput remain unmeasured. Windows preserve text but can separate related context; cross-document reasoning is not implemented. Arquivo experiment results do not establish this judge's performance.

For a model run, record the code revision, model snapshot, runtime versions, dataset identity, configuration, completed/failed/pending counts, control outcomes, and token/runtime totals alongside links to external artifacts. Accuracy claims additionally need the labeling method, split provenance, coverage, and per-axis metrics. Corpus files, weights, raw outputs, and reports belong outside this repository.
