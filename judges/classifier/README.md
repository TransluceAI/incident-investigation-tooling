# Classifier judge

A one-request-per-window yes/no classifier for the same three questions as the [screening judge](../screening/): hacking attempts, autonomous AI-agent activity, and agent coordination. It tests the suggestion from the review of #2 that open models handle a short prompt and a small answer format better than a large structured schema, and that a plain classifier pass may be enough for screening.

It does not build its own evidence. It reads a run prepared by `screening-judge prepare` and writes one request per window instead of one per axis, so both judges see byte-identical evidence, windows and boundary tags, and the same engine profile. The answer is four fields, generated in this order:

```json
{"reason": "...", "agentic_activity": "yes|no", "coordination": "yes|no", "hacking": "yes|no"}
```

There are no citations, actions, abstentions or confidence scores. An axis is positive for an item when any window says yes. An item with a failed or missing window and no yes is `incomplete`, never negative. `item-results.jsonl` has the screening report's shape, so `screening-judge evaluate` scores both judges with the same code.

The prompt is [`config/prompt.txt`](config/prompt.txt). It keeps the screening rubrics' main exclusions in one line each.

## Campaign-aware prompt

[`config/recall_prompt.txt`](config/recall_prompt.txt) lists the agent patterns seen in confirmed incidents (timed benchmark cohorts, signalling channels, scanner-as-browser fetches, archive and carrier relays, probes) and asks for recall. Its agent-activity question says that a scanner's own browser behaviour, API or feed submission, URL shorteners and the operator's collection labels are not evidence, and that a scan of an ordinary site page is a no unless the requested URL itself shows a task. Pass it with `--prompt`. `--hints` adds a `PATTERN HINTS` block after the evidence from the regex matcher in [`hints.py`](src/classifier_judge/hints.py); the prompt tells the judge hints are not evidence.

```sh
classifier-judge prepare --source-run "$RUN_ROOT/screening/prepared" \
  --out "$RUN_ROOT/campaign/prepared" --shards 8 \
  --prompt config/recall_prompt.txt --hints
```

[`EVALUATION.md`](EVALUATION.md) has the pre-registered tests behind this prompt and their limits. On a fresh set of 21 positives and 76 negatives it found 15/21 with 0 false alarms, against 8/21 and 0 for the v5 screening judge. On 89 later negatives that no judge saw before, it raised 0 false alarms, against 6 for the untuned prompt and 0 for v5.

## Install

Python 3.10+. Install the screening judge first; it supplies windows, controls, output checks and evaluation.

```sh
cd judges
python -m pip install -e screening -e classifier
```

## Prepare and run

```sh
screening-judge prepare --data-root "$DATA_ROOT" --engine "$ENGINE_CONFIG" \
  --out "$RUN_ROOT/screening/prepared" --window-tokens 160000 --shards 8

classifier-judge prepare --source-run "$RUN_ROOT/screening/prepared" \
  --out "$RUN_ROOT/classifier/prepared" --shards 8
```

Run inference on `$RUN_ROOT/classifier/prepared` with [`inference/qwen`](../../inference/qwen/README.md) exactly as for the screening judge (`submit.py --shard-dir .../shards`, or `inference_runner.py --input-manifest .../requests.jsonl` for one worker). The source run must not change after classifier preparation; reporting checks its hashes.

## Report

```sh
classifier-judge report --run "$RUN_ROOT/classifier/prepared" \
  --outputs "$RUN_ROOT/classifier/inference/outputs" --out "$RUN_ROOT/classifier/report"
```

- `item-results.jsonl`: per-item assessments (`positive`, `no_evidence`, `incomplete`) in the screening report's shape.
- `judgments.jsonl`: every committed window answer, with its reason, including validation failures.
- `failures.jsonl`, `pending.jsonl`: windows that failed validation or have no output; their items are incomplete.
- `control-results.jsonl` and `summary.json`: the screening judge's synthetic controls scored on the three axes. Its two prompt-injection checks are skipped because this format has no injection field.

## Compare with the screening judge

`scripts/compare.py` scores both judges on a labeled key (the findings benchmark's `answer_key.jsonl` format) and reports end-to-end rates: an axis counts as flagged only when the item result is positive, so failures and abstentions count as not flagged and are listed separately. It also reports per-family and per-length results, every case where the judges disagree, runner token and GPU-time totals, and, with `--axis-key`, per-axis recall on hand-labeled positives.

```sh
python scripts/compare.py --key answer_key.jsonl --axis-key hand_labels.jsonl \
  --screening-results "$RUN_ROOT/screening/report/item-results.jsonl" \
  --screening-outputs "$RUN_ROOT/screening/inference/outputs" \
  --classifier-results "$RUN_ROOT/classifier/report/item-results.jsonl" \
  --classifier-outputs "$RUN_ROOT/classifier/inference/outputs" --out comparison.json
```

The findings benchmark key labels its partition `development`, which `screening-judge evaluate` does not accept; `compare.py` maps it to `dev`.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests use synthetic data and simulated runner outputs; no model or GPU is needed.
