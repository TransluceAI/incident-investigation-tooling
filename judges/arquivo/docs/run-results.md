# Recorded results

Slurm job 66 screened the entire preserved corpus supplied to the judge on eight GPUs in 21m37s. Coverage refers to that export, not all of Arquivo.pt.

| Measure | Result |
| --- | ---: |
| Events | 7,926 |
| Evidence windows | 8,198 |
| Validated windows | 8,175 |
| Validation exceptions | 23 |
| Runner failures / pending windows | 0 / 0 |
| Likely / possible events | 117 / 313 |
| Additional unverified positive | 1 |
| Input / output tokens | 61,534,677 / 3,619,454 |

The 23 exceptions comprise 17 unknown evidence chunks, five unsupported success stage/system combinations, and one unresolved action reference. They remain reviewable and are not accepted as negative findings. Labels do not establish successful attacks or AI involvement, and classification accuracy is not calibrated.

The run used `Qwen/Qwen3.8-27B-FP8`, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, temperature 0, seed 17, and thinking disabled. The recorded environment used vLLM 0.31.0, Transformers 5.17.0, PyTorch 2.13.0, and jsonschema 4.26.0. The package's optional dependencies are not pinned; these are the versions used for the recorded run. The shared inference runner is taken from that run; the rubric and schema now also request a confidence score as described below. The corpus preparation default path and Slurm launcher have been adapted for portability; no new GPU run is claimed for this packaging.

The recorded run also predates later reporting and preparation changes: reports now check runner identities, list every axis per event, queue windows positive on any axis, and require evidence and a basis for supported provider attribution; preparation now slices only BODY text when an event needs several windows. Re-reporting the recorded run applies the new reporting rules, so its counts can differ from the table above; the preparation changes apply only to new runs.

The recorded run predates the confidence-score addition. Its saved rubric/schema and outputs have no confidence scores; the totals above are unchanged. New runs use the updated rubric/schema to produce scores. Re-reporting this historical run leaves scores `null`; it does not backfill model estimates.

## Artifact access

The complete artifacts are readable to users on the investigation cluster at:

```text
/data/qwen-inference/runs/ayushb/arquivo-full-20261014/
```

Within that root:

- `README.md`: overview.
- `prepared/config.json`: exact prompt, schema, model location, and inference settings used.
- `prepared/`: source provenance, model inputs, and coverage/assignment audits.
- `outputs/` and `logs/`: committed raw outputs and execution logs.
- `report/completion-audit.json`: completion and accounting checks.
- `report/summary.json` and `report/event-results.jsonl`: window and event totals.
- `report/review-queue.jsonl` and `report/failures.jsonl`: candidates and validation exceptions.

These are cluster filesystem paths, not public download links. Reproducing the recorded run requires its original exported corpus and pinned model environment. Using another export creates a new screening run.
