# Reusable evidence judge

Screens preserved evidence independently for hacking attempts, autonomous AI-agent activity, and agent coordination. It prepares capture/redirect, scanner, wiki, message, and general-text evidence for an offline Qwen 27B runner. Unknown kinds retain their text. It never executes evidence or follows evidence URLs.

**Status: local framework tested; model accuracy and calibration not yet measured.** CPU tests exercise the pipeline and simulate the inference engine. They do not establish Qwen accuracy, injection resistance, GPU compatibility, or full-corpus throughput. See [validation notes](docs/local-validation.md) for evidence and outstanding work.

## Local use

From this directory, install the package with `python -m pip install -e .` in your Python 3.10+ environment. The core dependency is `jsonschema`. Preparation with exact counts also needs `transformers`; inference needs a compatible `vllm`, Transformers, and local model snapshot. The wheel bundles prompts and example configs; the inference script is run from the source checkout.

```sh
# Tests: no model, network, or server required.
PYTHONPATH=src python -m unittest discover -s tests -v

# CPU-only preparation smoke test. Counts are explicitly estimates.
PYTHONPATH=src python -m evidence_judge.pipeline prepare \
  --data-root /path/to/export --engine config/engine.example.json \
  --out /path/to/new-run --estimate-tokens 4 --window-tokens 160000 --shards 8

# Exact preparation: set model in a copy of the engine config to a local
# snapshot, and omit --estimate-tokens. Existing output folders are refused.
```

Accepted inputs are `resources/*/records.jsonl` and recursive `resources/**/DIRECT_RECORDS.jsonl`, matching the corpus profiler. Each record needs a string `observation_id` and `kind`; optional `document_path` resolves inside the export root, and `text_sha256` is checked. Empty exports fail visibly. Missing/corrupt documents remain incomplete in reports. Malformed records and unlinked archive records remain in `gaps.jsonl`.

Each capture retains its request/time context; different request URLs do not collapse just because their error bodies match. Other items merge only when source, shape, body hash, and evidence metadata match. This is more conservative than body-hash deduplication and may require more model calls. Scanner views preserve every nonempty scalar field, including hashes: arbitrary request bodies can contain meaningful fields with those names. This also increases token use relative to the unfinished implementation.

Windows have exact character-offset citation maps and overlap. Preparation measures every axis prompt. The runner independently counts tokenizer tokens and rejects oversized input; it never truncates to fit. Shards keep all axis requests for a window together and balance approximately by text length. Preparation streams windows, but item indexing/report reconciliation still use memory proportional to the number of items/results; full 15 GB corpus resource use is not measured.

## Runner and reporting

These commands describe later local GPU use. No GPU run or server upload was performed during the continuation.

```sh
python scripts/inference_runner.py \
  --config /path/to/new-run/config.json \
  --input-manifest /path/to/new-run/shards/manifest-0000.jsonl \
  --output-dir /path/to/new-run/outputs --worker-id 0 --validate-only

# Remove --validate-only to infer. Use a different manifest/worker ID per shard.
# Add --resume only with identical settings and input. Failed committed
# responses remain failed; resume does not silently retry them.

PYTHONPATH=src python -m evidence_judge.pipeline report \
  --run /path/to/new-run --outputs /path/to/new-run/outputs \
  --out /path/to/new-run/report
```

The runner currently requires exactly one visible CUDA device per worker. Thinking is explicit; an unclosed reasoning block or length-limited output is a failure. The reasoning parser and structured-output configuration require validation on the installed vLLM version. The example `model_revision` records the intended snapshot; it does not prove that an arbitrary local weights directory contains that revision. Use a verified immutable snapshot.

Reports verify prepared file hashes and worker identities/configuration/manifest membership. Each committed unit must also belong to that worker's own manifest and its `--max-records` selection, when present. Positive assessments require valid schema and resolvable citations. A citation proves that the cited text exists; it does **not** prove that the text supports the model's interpretation. The model's free-text `location` is explicitly unverified. Action references must have resolved citations; archive replay status cannot establish attack success.

Outputs include item results, validated/failed judgments, pending units, unverified positives, near misses, positive review candidates, and injection review. Synthetic controls cover every shape and are reported separately. Abstention does not pass a negative control. Item counts exclude controls; request-level totals include them. Positive-rate-by-window-count is a descriptive rate, **not** false-positive rate without labels.

## Neutral benchmark input

The recorded 600-case input manifest is also supported directly:

```sh
PYTHONPATH=src python -m evidence_judge.pipeline prepare \
  --data-root /path/to/benchmark-package \
  --benchmark-manifest benchmark/inputs.jsonl \
  --engine config/engine.example.json --out /path/to/new-benchmark-run
```

Paths in `inputs.jsonl` resolve against `--data-root`. Rows require `id`, `document_path`, and `text_sha256`, with optional `metadata`. Only neutral metadata fields `source_family` and `format` enter prompt provenance. IDs, paths, labels and review fields remain outside prompts. The adapter never opens the answer key. Every case stays separate even if bodies match; reporting carries `benchmark_case_id` for evaluator joins. The path convention must be checked against the actual local export when it becomes available.

Preparation code hashes are saved in the configuration. Exploratory score bands bind to that classifier fingerprint and to runner/prompt implementation hashes. Installed model weights and runtime packages still require separate verification.

## Evaluation

```sh
PYTHONPATH=src python -m evidence_judge.pipeline evaluate \
  --results /path/to/report/item-results.jsonl \
  --key /path/to/answer_key.jsonl --out /path/to/evaluation

PYTHONPATH=src python -m evidence_judge.pipeline compare \
  --a /path/to/baseline/item-results.jsonl \
  --b /path/to/candidate/item-results.jsonl \
  --key /path/to/answer_key.jsonl --axis overall --split dev
```

The recovered benchmark documentation calls the existing 600 cases a development set, not a historically sealed holdout. Reserving a test partition now only protects it from subsequent tuning; evaluation reports historical independence as unverified. A defensible final accuracy claim needs reviewed provenance or new untouched examples.

The key requires unique `id`, `document_sha256`, optional `leakage_group`, optional explicit `evaluation_partition`, `overall_label`, and/or `axis_labels` (`hacking`, `agentic_activity`, `collusion`). `overall` means any of the three axes. Missing axis labels stay unlabeled; overall positives are never invented as per-axis positives.

Connected leakage groups and repeated document hashes stay together. Conflicting explicit splits fail. Recognized partitions are train/dev/test plus explicit aliases in `evaluate.py`; unknown names fail rather than silently reassigning cases. Test metrics are excluded unless `evaluate --include-test` is explicitly requested for a frozen candidate. `compare` defaults to dev; test requires `--split test`. These switches prevent accidental metric exposure, not unauthorized filesystem access to the key.

A document hash matching multiple event contexts is ambiguous; supply a matching `item_id` in the key to resolve it. Do not max-pool unrelated captures for evaluation. Reports show missing/ambiguous/undecided counts, coverage, conditional TPR/FPR/MCC, and end-to-end positive recall. Cluster bootstrap resamples connected groups. Degenerate intervals do not prove zero error; one-group intervals are withheld. Paired comparisons provide clustered TPR/FPR delta intervals and omit McNemar p-values when paired cases are dependent.

`label_probs`/`p_yes` are compatibility field names for **diagnostic first-token branch scores**, not calibrated probabilities. Missing top-k candidates, combined boundary/value tokens, or unavailable logprobs produce no score. Multi-token labels make sequence-probability claims inappropriate. No/abstain responses are never promoted based on scores alone.

The optional `thresholds.json` contains exploratory dev ranking bands among already verified positives, with ties handled together. It is bound to classifier configuration (excluding the per-run delimiter nonce), and it requires both labeled classes. It does not tune a detection recall floor, estimate calibrated probability, guarantee production precision, or establish a release gate. Without applicable scores/bands, positives remain `positive_uncalibrated`. Supplying bands to `report --thresholds ...` can label verified positives `likely`/`possible`; these are review bands only.

`config/engine.example.json` and `engine.thinking.example.json` retain the inherited greedy baselines. `engine.vendor-instruct.example.json` and `engine.vendor-thinking.example.json` use the exact model card's recommended sampling parameters as **untested experiment configurations**. Compare them on dev before selecting a candidate. Full inference, held-out quality, actual-model canaries, and repeatability are still required before release.
