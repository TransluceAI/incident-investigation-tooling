# Arquivo judge

An LLM judge for screening preserved Arquivo.pt capture events for hacking attempts. It independently assesses AI-agent activity, coordination, and attribution, then produces source-cited judgments and a review queue.

The judge preserves original URLs and capture provenance, decodes embedded URLs and programs as inert text, and supplies complete evidence windows. Numbered source chunks let the model cite locations without copying escaped text. Validation resolves those pointers to source text and character offsets; unresolved citations and incomplete windows remain explicit.

This is exploratory evidence triage. Labels require human review, and source-location validation does not establish classification accuracy or a successful compromise. Archive responses and intermediary responses are distinguished from ultimate-target responses.

## Code

```text
config/                 Judge rubric, JSON schema, and engine settings template
src/arquivo_judge/      Event ledger, inert decoding, evidence rendering, reporting
scripts/audit.py        Prepared-input coverage and shard-assignment checks
scripts/shard_all.py    All corpus windows assigned once across GPU shards (default eight)
scripts/inference_runner.py
                        Shared vLLM inference runner
scripts/run.sbatch      Portable Slurm launcher for those eight shards
examples/corpus/        Small synthetic export illustrating the input format
docs/run-results.md     Recorded results and external artifact location
```

The judge uses `config/rubric.txt` and `config/schema.json` with the existing shared vLLM inference runner. The Slurm launcher accepts user-supplied paths.

## Install

Python 3.10+ is required for reporting. Preparation additionally needs the model tokenizer; inference needs a Linux CUDA environment with vLLM and enough GPU memory for the configured model and context. The completed run used one model replica on each of eight H100 GPUs. Install into a suitable existing environment:

```sh
cd judges/arquivo
python -m pip install -e .              # Report existing outputs, CPU only
python -m pip install -e '.[prepare]'   # Also prepare inputs with Transformers
# On the GPU host, use a model-compatible vLLM environment:
python -m pip install -e '.[gpu]'
```

Place the model weights and tokenizer in a local directory. Copy `config/engine.example.json` to `config/engine.local.json` and set `model` to that directory's absolute path. Preparation and the runner use local model files only. The example preserves the completed run's engine and sampling settings; other models or hardware may require different settings and validation. Preparation keeps the file's `host_batch_records` and `host_batch_tokens`, and sizes input windows to `min(160000, max_model_len - sampling.max_tokens, host_batch_tokens)` tokens (recorded as `window_budget` in `prepared/config.json`; set `window_budget` in the file to lower it).

## Prepare and run

Run these commands from this tool's directory. `DATA_ROOT` must contain an export matching [the input contract](#input-format); the tool does not collect or crawl Arquivo.pt.

```sh
export TOOL_DIR="$PWD"
export DATA_ROOT=/absolute/path/to/exported-corpus
export PREPARED_DIR="$PWD/runs/full/prepared"
export OUTPUT_DIR="$PWD/runs/full/outputs"
export JUDGE_PYTHON="$(command -v python)"

"$JUDGE_PYTHON" -m arquivo_judge.pipeline prepare \
  --data-root "$DATA_ROOT" --engine config/engine.local.json \
  --rubric config/rubric.txt --schema config/schema.json \
  --out "$PREPARED_DIR"

"$JUDGE_PYTHON" scripts/shard_all.py \
  --run "$PREPARED_DIR" --out "$PREPARED_DIR/full-shards"
"$JUDGE_PYTHON" scripts/audit.py "$PREPARED_DIR"

mkdir -p runs/full/logs
sbatch --export=ALL --output="$PWD/runs/full/logs/%A_%a.log" scripts/run.sbatch
```

Use fresh preparation and output paths for each new experiment. `shard_all.py` includes every canonical corpus window exactly once and excludes synthetic controls; `audit.py` then checks character coverage and that `full-shards/` assigns every window exactly once, and fails with a message rather than an `assert`. The commands here do not evaluate the synthetic controls: `controls: []` and `controls_passed: 0` in the report mean no controls were run, and `ignored_control_outputs` counts control outputs left out of a report. `prepare` also writes `pilot.jsonl` and `shard-N.jsonl` for a pilot workflow; the commands here do not run them. Choose any partition/account options required by your cluster when submitting. `shard_all.py --shards N` (default 8) sets the number of single-GPU workers; submit the launcher with a matching array, e.g. `sbatch --array=0-$((N-1)) ...`. The model and Python environment must be accessible from every worker.

An event too long for one window is split. Every segment other than BODY (metadata, URL, decoded views, redirects) repeats in each window and only the BODY text is sliced, with a `PART i OF n` marker, so request-side evidence and the response are judged together. If those other segments alone would use more than a third of the window, only EVENT and URL repeat and each other segment is sliced on its own.

For a single GPU, run one shard directly instead:

```sh
"$JUDGE_PYTHON" scripts/inference_runner.py \
  --config "$PREPARED_DIR/config.json" \
  --input-manifest "$PREPARED_DIR/full-shards/part-0.jsonl" \
  --output-dir "$OUTPUT_DIR" --worker-id 0 --resume
```

That command covers shard 0 only. Run the other shards to complete the corpus. Each worker expects exactly one visible CUDA device. Resume interrupted work only after the previous worker exits; configuration, manifest, and runner identities must match. Resume skips all committed rows, including failures. Automated retries and result replacement are not supported; reporting rejects duplicate window IDs across output directories, and outputs whose runner identity does not match the prepared run's `config.json`.

## Report

After inference finishes, reporting can run on a CPU:

```sh
python -m arquivo_judge.pipeline report \
  --run "$PREPARED_DIR" --manifest "$PREPARED_DIR/units.jsonl" \
  --outputs "$OUTPUT_DIR" --out runs/full/report
```

- `summary.json`: event/window counts, per-axis event counts, validation status, preparation gap counts, and token totals.
- `judgments.jsonl`: every committed judgment, including validation failures.
- `predictions.jsonl`: validated judgments with extracted source locations. Each action's `action_evidence_sources` entry marks a referenced segment `cited` when the judgment also cites a chunk there, or `segment_reference_only` when the action only names the segment; the span given is the whole segment either way.
- `event-results.jsonl`: event-level assessments and completion status. `assessment` is the hacking label; `assessments` gives every axis (hacking, agent activity, coordination, provider attribution) and `positive_axes` lists the positive ones.
- `review-queue.jsonl`: windows positive on any axis, including explicitly unverified positives, each with `positive_axes`, sorted by descending `hacking_confidence` (0–1). Higher scores mean stronger model-estimated evidence of a hacking attempt; they are not calibrated probabilities or confidence in a negative label, so windows positive only on other axes usually sort low. Validation status remains explicit regardless of score.
- `failures.jsonl`, `pending.jsonl`, `unverified-candidates.jsonl`: exceptions kept visible rather than treated as negatives.

A positive claim on any axis, including `supported_by_supplied_evidence` attribution, needs at least one resolved citation; supported attribution also needs a non-empty `basis`.

New judgments require `hacking_attempt.confidence` in the schema. The queue also exposes it as `hacking_confidence` for sorting. Historical runs are validated against their saved schema and remain reportable; missing scores are `null` and sort last. To obtain scores for old evidence, prepare a fresh run and rerun inference rather than reusing or resuming old outputs.

The review queue contains window records; an event can have multiple windows. Use `event-results.jsonl` for event-level counts. `prepared/config.json` records the exact prompt, schema, model path/revision, and settings used for a particular run.

## Tests

```sh
python -m unittest discover -s tests -v
```

## Input format

The export root contains `resources/arquivo.pt/records.jsonl`, one observation per line. Every observation has a unique `observation_id`, a `kind`, and, when it references stored text, `document_path` relative to the export root and the SHA-256 of that UTF-8 text in `text_sha256`.

Supported kinds:

- `arquivo_index`: its document contains CDX JSONL rows with `url` and `timestamp`, plus optional `mime`, `status`, `digest`, `length`, and `collection`.
- `arquivo_capture`: stored response text with `metadata.original_url` and `metadata.capture_timestamp` (also accepted under `original_record.metadata`).
- `provider_redirect`: `source_url` identifies an archive replay URL; its document stores the redirect destination as text.

Records may also carry `scope`, `time_basis`, and `raw_path` provenance. Malformed or unsupported observations (unparseable lines, missing IDs, fields of the wrong type, captures without metadata, documents that cannot be read, empty redirects) are recorded in `gaps.jsonl` and skipped. A capture whose text is missing, has no hash, or cannot be read keeps its event: the model sees `BODY: unreadable`, and the event is reported incomplete (`unreadable_captures`), never negative. Duplicate observation IDs and documents that do not match their recorded hash stop preparation. See `examples/corpus/` for a synthetic index and response body; it illustrates the format, not judge accuracy.

## Recorded results

The judge processed 7,926 events across 8,198 windows, with 8,175 validated results and 23 validation exceptions. Event assessments included 117 likely and 313 possible candidates, plus one additional unverified positive in the review queue. Accuracy has not been calibrated against an independent human-labeled set.

See [recorded results and artifact access](docs/run-results.md). The corpus, model weights, raw outputs, and reports are hosted outside this repository; cluster access is required for the recorded run artifacts.
