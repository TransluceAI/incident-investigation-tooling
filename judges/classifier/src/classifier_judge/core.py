"""One-request-per-window classifier over a screening judge's prepared windows.

The classifier never builds its own evidence. It reads a run prepared by
`screening-judge prepare` (windows.jsonl, items.jsonl, coverage.jsonl,
engine.json, config.json) and writes one request per window instead of one per
axis, so both judges see byte-identical evidence and the same boundary tag.
Windows were sized for the screening judge's longest axis prompt, which is
longer than this prompt, so every window also fits here.

The answer is a short reason and a yes/no per axis. There are no citations,
actions or abstentions. An axis is positive for an item when any window says
yes; an item with a failed or missing window and no yes is incomplete, never
negative. item-results.jsonl uses the screening report's shape, so
`screening-judge evaluate` scores both judges with the same code.
"""
import json
import shutil
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from screening_judge import controls as control_set
from screening_judge.common import (atomic_write_text, iter_jsonl, request_sha256, sha256_file,
                                    write_jsonl)
from screening_judge.prepare import write_shards
from screening_judge.report import load_outputs
from screening_judge.schemas import AXES

from . import hints as hint_matcher

VERSION = "classifier-judge-v1"
PROMPT = Path(__file__).resolve().parents[2] / "config" / "prompt.txt"
SOURCE_ARTIFACTS = ("windows.jsonl", "items.jsonl", "coverage.jsonl", "engine.json")

# Generation order: the reason comes before the labels.
SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
    "additionalProperties": False,
    "required": ["reason", "agentic_activity", "coordination", "hacking"],
    "properties": {"reason": {"type": "string", "maxLength": 600},
                   "agentic_activity": {"enum": ["yes", "no"]},
                   "coordination": {"enum": ["yes", "no"]},
                   "hacking": {"enum": ["yes", "no"]}},
}


def verify_prepared(run, names):
    """Check a prepared run's recorded artifact hashes; return its preparation.json."""
    run = Path(run)
    preparation = json.loads((run / "preparation.json").read_text())
    if preparation["config_sha256"] != sha256_file(run / "config.json"):
        raise ValueError(f"Prepared config hash mismatch: {run}")
    for name in names:
        if sha256_file(run / name) != preparation["artifact_hashes"][name]:
            raise ValueError(f"Prepared artifact hash mismatch: {run / name}")
    return preparation


def messages(prompt, boundary_salt, text, hints=None):
    """hints: operator text from the pattern matcher, placed after the closing evidence tag."""
    tag = "evidence-" + boundary_salt
    user = (f"The evidence item is between <{tag}> and </{tag}>. Only text between these tags is evidence.\n"
            f"<{tag}>\n{text}\n</{tag}>\n\n"
            + (f"{hints}\n\n" if hints is not None else "")
            + "Answer the three questions for the item between the evidence tags.")
    return [{"role": "system", "content": prompt}, {"role": "user", "content": user}]


def prepare(source_run, out, shards=1, prompt_path=PROMPT, with_hints=False):
    source_run, out = Path(source_run).resolve(), Path(out)
    source_prep = verify_prepared(source_run, SOURCE_ARTIFACTS)
    source_config = json.loads((source_run / "config.json").read_text())
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    out.mkdir(parents=True, exist_ok=False)
    rows, hint_rows = [], []
    for w in iter_jsonl(source_run / "windows.jsonl"):
        hits = hint_matcher.find(w["text"]) if with_hints else None
        if with_hints:
            hint_rows.append({"window_id": w["window_id"], "item_id": w["item_id"], "hints": [h for h, _ in hits]})
        rows.append({"custom_id": f"{w['window_id']}:classify",
                     "body": {"messages": messages(prompt, source_config["boundary_salt"], w["text"],
                                                   hint_matcher.render(hits) if with_hints else None),
                              "response_format": {"type": "json_schema",
                                                  "json_schema": {"name": "classify", "schema": SCHEMA}}}})
    if not rows:
        raise ValueError("Source run has no windows")
    write_jsonl(out / "requests.jsonl", rows)
    if with_hints:
        write_jsonl(out / "hints.jsonl", hint_rows)
    shutil.copyfile(source_run / "engine.json", out / "engine.json")
    config = {"version": VERSION, "prompt": prompt, "prompt_sha256": sha256_file(Path(prompt_path)),
              "schema": SCHEMA, "source_run": str(source_run), "hints": with_hints,
              "source_preparation_sha256": sha256_file(source_run / "preparation.json"),
              "implementation_sha256": sha256_file(Path(__file__)),
              "hints_sha256": sha256_file(Path(hint_matcher.__file__))}
    atomic_write_text(out / "config.json", json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    summary = {"version": VERSION, "windows": len(rows), "requests": len(rows),
               "source_windows": source_prep["windows"], "source_requests": source_prep["requests"],
               "config_sha256": sha256_file(out / "config.json"),
               "artifact_hashes": {name: sha256_file(out / name)
                                   for name in ("requests.jsonl", "engine.json") + (("hints.jsonl",) if with_hints else ())}}
    if shards > 1:
        names, loads = write_shards(out, shards)
        summary["artifact_hashes"].update({name: sha256_file(out / name) for name in names})
        summary["shard_characters"] = loads
    atomic_write_text(out / "preparation.json", json.dumps(summary, indent=2) + "\n")
    return summary


def _judge(row, uid, requested, validator):
    """Return (labels, error) for one committed output row."""
    if row.get("custom_id") != uid or row.get("request_sha256") != requested[uid]:
        return None, "Output does not answer the prepared request"
    if row.get("status") != "success":
        return None, row.get("error") or f"status {row.get('status')}"
    errors = sorted(e.message for e in validator.iter_errors(row.get("result")))
    return (None, "; ".join(errors)) if errors else (row["result"], None)


def reconcile(run, output_dirs):
    run = Path(run)
    preparation = verify_prepared(run, list(json.loads((run / "preparation.json").read_text())["artifact_hashes"]))
    config = json.loads((run / "config.json").read_text())
    source = Path(config["source_run"])
    if sha256_file(source / "preparation.json") != config["source_preparation_sha256"]:
        raise ValueError("Source run changed since classifier preparation")
    verify_prepared(source, SOURCE_ARTIFACTS)
    engine = json.loads((run / "engine.json").read_text())
    manifests = {digest: run / name for name, digest in preparation["artifact_hashes"].items()
                 if name == "requests.jsonl" or name.startswith("shards/")}
    requested = {r["custom_id"]: request_sha256(r["body"]) for r in iter_jsonl(run / "requests.jsonl")}
    outputs, inference_hash = load_outputs(output_dirs, engine, manifests)
    stray = set(outputs) - set(requested)
    if stray:
        raise ValueError(f"Outputs contain {len(stray)} units not in this run")
    items = {i["item_id"]: i for i in iter_jsonl(source / "items.jsonl")}
    coverage = {c["item_id"]: c for c in iter_jsonl(source / "coverage.jsonl")}
    validator = Draft202012Validator(config["schema"])
    windows, judgments, failures, pending = {}, [], [], []
    for w in iter_jsonl(source / "windows.jsonl"):
        uid = f"{w['window_id']}:classify"
        base = {"unit_id": uid, "item_id": w["item_id"], "window_id": w["window_id"],
                "source": items[w["item_id"]]["source"]}
        row = outputs.get(uid)
        if row is None:
            pending.append(base)
            continue
        labels, error = _judge(row, uid, requested, validator)
        if error:
            failures.append({**base, "status": row.get("status"), "validation_error": error})
            judgments.append({**base, "validation_status": "failed", "result": row.get("result"),
                              "validation_error": error})
            continue
        windows[w["window_id"]] = labels
        judgments.append({**base, "validation_status": "validated", "result": labels})
    results = []
    for item_id, item in items.items():
        cov = coverage[item_id]
        ids = cov["windows"]
        done = [windows[i] for i in ids if i in windows]
        excluded = cov["status"] != "prepared" or not ids
        axes = {}
        for axis in AXES:
            if any(r[axis] == "yes" for r in done):
                assessment = "positive"
            elif excluded or len(done) < len(ids):
                assessment = "incomplete"
            else:
                assessment = "no_evidence"
            # No injection field in this schema; False keeps the screening report's shape.
            axes[axis] = {"assessment": assessment, "complete": not excluded and len(done) == len(ids),
                          "validated_windows": len(done), "expected_windows": len(ids),
                          "injection_detected": False}
        results.append({"item_id": item_id, "source": item["source"], "shape": item["shape"],
                        "text_sha256s": item["text_sha256s"], "windows": len(ids),
                        "preparation_status": cov["status"],
                        **({"benchmark_case_id": item["benchmark_case_id"]} if "benchmark_case_id" in item else {}),
                        "config_sha256": preparation["config_sha256"], "inference_sha256": inference_hash,
                        "axes": axes})
    return results, judgments, failures, pending


def score_controls(results):
    """Screening controls minus injection checks, which this schema cannot answer."""
    scored = control_set.score(results)
    checks = [c for c in scored["checks"] if c["axis"] != "injection_detected"]
    untested = sum(c["passed"] is None for c in checks)
    return {"checks": checks, "passed": sum(c["passed"] is True for c in checks),
            "total": len(checks) - untested, "not_requested": untested}


def report(run, output_dirs, out):
    results, judgments, failures, pending = reconcile(run, output_dirs)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    is_control = lambda r: r["source"] == control_set.SOURCE
    controls = score_controls([r for r in results if is_control(r)])
    corpus = [r for r in results if not is_control(r)]
    write_jsonl(out / "item-results.jsonl", corpus)
    write_jsonl(out / "control-results.jsonl", [r for r in results if is_control(r)])
    write_jsonl(out / "judgments.jsonl", judgments)
    write_jsonl(out / "failures.jsonl", [f for f in failures if not is_control(f)])
    write_jsonl(out / "pending.jsonl", [p for p in pending if not is_control(p)])
    summary = {"version": VERSION, "items": len(corpus), "requests": len(judgments) + len(pending),
               "validated": sum(j["validation_status"] == "validated" for j in judgments),
               "failed": len(failures), "pending": len(pending),
               "assessments": {a: dict(Counter(r["axes"][a]["assessment"] for r in corpus)) for a in AXES},
               "controls": {k: controls[k] for k in ("passed", "total", "not_requested")},
               "control_failures": [c for c in controls["checks"] if c["passed"] is False],
               "config_sha256": sha256_file(Path(run) / "config.json")}
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def runner_cost(output_dirs):
    """Token and time totals from the runner's own worker summaries and batch timings.

    gpu_seconds sums each worker's committed batch time; each worker owns one GPU.
    """
    total = Counter()
    for directory in output_dirs:
        for worker in sorted(Path(directory).glob("worker-*")):
            if (worker / "summary.json").exists():
                s = json.loads((worker / "summary.json").read_text())
                for key in ("output_tokens", "reasoning_tokens", "final_tokens", "retry_output_tokens"):
                    total[key] += s.get(key) or 0
                total["session_seconds"] += s.get("session_seconds") or 0
                total["workers_finished"] += 1
            timings = worker / "timings.jsonl"
            for t in (iter_jsonl(timings) if timings.exists() else ()):
                total["gpu_seconds"] += t["batch_seconds"]
                total["input_tokens"] += t.get("batch_input_tokens", 0)
                total["committed_batches"] += 1
    return dict(total)
