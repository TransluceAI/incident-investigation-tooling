"""Reconcile committed outputs into per-item, per-axis results and review queues.

Item assessment per axis:
- positive: some window answered yes AND its citations resolved. With calibrated
  thresholds the positive is graded likely (P(yes) >= t_likely) or possible;
  without them it is positive_uncalibrated.
- incomplete: no verified positive and some expected window failed or is pending.
  Failures are never negatives.
- insufficient_evidence / no_evidence: every window validated and none positive.

Schema-valid yes answers whose citations failed go to unverified-candidates.
Windows answered no with P(yes) >= near_miss_threshold go to near-misses for
recall audits; they are never counted as positive.

The review queue holds every request needing a human: verified positives on any
axis, unverified positives, abstentions, and failures, in that order. Within
each reason it is sorted by the model's 0-1 confidence for that axis (missing
scores last), then by P(yes) from the label logprobs. Neither score is
calibrated; both only order the queue. Confidence represents support for the
behaviour. A run expects exactly the requests in its requests.jsonl, so a second-pass
run over a subset of requests reports only those.

Outputs come from the shared runner (inference/qwen). Each committed row must
answer exactly the prepared request: its custom_id and request hash are checked
against requests.jsonl. Label scores are computed here from the raw logprobs.
"""
import json
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from . import controls as control_set
from .common import atomic_write_text, iter_jsonl, write_jsonl, request_sha256, sha256_text, sha256_file
from .schemas import AXES, FINDINGS_CYBERATTACK
from .scores import label_probabilities
from .validate import ValidationError, validate


# Fields reconciliation reads. Reasoning and token logprobs are dropped once
# scored, so memory grows with the number of requests, not generation length.
KEPT = ("unit_id", "custom_id", "request_sha256", "run_identity", "status", "error", "result")


def load_outputs(dirs, engine, manifests):
    outputs = {}
    code_identity = None
    for directory in dirs:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Output directory missing: {directory}")
        for worker in sorted(directory.glob("worker-*")):
            identity_path = worker / "identity.json"
            if not identity_path.exists():
                if list(worker.glob("part-*.jsonl")):
                    raise ValueError("Committed outputs missing identity: " + str(worker))
                continue
            identity = json.loads(identity_path.read_text())
            digest = identity.pop("identity")
            if digest != sha256_text(json.dumps(identity, sort_keys=True)):
                raise ValueError("Worker identity hash mismatch")
            if identity["config"] != engine:
                raise ValueError("Worker config does not match prepared run")
            if identity["manifest_hash"] not in manifests:
                raise ValueError("Worker manifest does not belong to prepared run")
            limit = identity.get("max_records")
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("Invalid worker manifest selection limit")
            selected = islice(iter_jsonl(manifests[identity["manifest_hash"]]), limit)
            allowed = {row["custom_id"] for row in selected}
            code = identity["runner_hash"]
            if code_identity is not None and code != code_identity:
                raise ValueError("Mixed runner implementations")
            code_identity = code
            for part in sorted(worker.glob("part-*.jsonl")):
                for row in iter_jsonl(part):
                    if row.get("run_identity") != digest:
                        raise ValueError("Output run identity mismatch")
                    if row.get("unit_id") not in allowed:
                        raise ValueError("Output unit outside worker selected manifest: " + str(row.get("unit_id")))
                    if row["unit_id"] in outputs:
                        raise ValueError("Duplicate committed unit: " + row["unit_id"])
                    outputs[row["unit_id"]] = {
                        **{k: row[k] for k in KEPT if k in row},
                        "label_probs": label_probabilities(row.get("logprobs")) if row.get("status") == "success" else None}
    inference_hash = sha256_text(json.dumps(code_identity)) if code_identity is not None else None
    return outputs, inference_hash


def grade(p_yes, thresholds, axis):
    t = (thresholds or {}).get("axes", {}).get(axis)
    if not t:
        return "positive_uncalibrated"
    if p_yes is None:
        return "positive_uncalibrated"
    return "likely" if p_yes >= t["t_likely"] else "possible"


def reconcile(run, output_dirs, thresholds=None, near_miss_threshold=0.2):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    engine = json.loads((run / "engine.json").read_text())
    items = {i["item_id"]: i for i in iter_jsonl(run / "items.jsonl")}
    preparation = json.loads((run / "preparation.json").read_text())
    if preparation["config_sha256"] != sha256_file(run / "config.json"):
        raise ValueError("Prepared config hash mismatch")
    for name, digest in preparation["artifact_hashes"].items():
        if sha256_file(run / name) != digest:
            raise ValueError("Prepared artifact hash mismatch: " + name)
    if thresholds is not None and thresholds.get("classifier_sha256") != preparation["classifier_sha256"]:
        raise ValueError("Threshold config does not match prepared run")
    coverage = {c["item_id"]: c for c in iter_jsonl(run / "coverage.jsonl")}
    manifests = {digest: run / name for name, digest in preparation["artifact_hashes"].items()
                 if name == "requests.jsonl" or name.startswith("shards/")}
    requested = {row["custom_id"]: request_sha256(row["body"]) for row in iter_jsonl(run / "requests.jsonl")}
    # preparation.json hashes requests.jsonl and engine.json, so this names the exact execution.
    preparation_sha256 = sha256_file(run / "preparation.json")
    outputs, inference_hash = load_outputs(output_dirs, engine, manifests)
    if thresholds is not None and thresholds.get("inference_sha256") != inference_hash:
        raise ValueError("Threshold inference implementation does not match this run")
    expected = defaultdict(list)
    judgments, failures, pending, unverified, near = [], [], [], [], []
    by_item_axis = defaultdict(list)
    window_counts = Counter()
    seen = set()
    for w in iter_jsonl(run / "windows.jsonl"):
        window_counts[w["item_id"]] += 1
        for axis in AXES:
            uid = f"{w['window_id']}:{axis}"
            if uid not in requested:
                continue
            seen.add(uid)
            expected[(w["item_id"], axis)].append(uid)
            row = outputs.get(uid)
            if row is None:
                pending.append({"unit_id": uid, "item_id": w["item_id"], "axis": axis,
                                "source": items[w["item_id"]]["source"]})
                continue
            probs = row["label_probs"]
            base = {"unit_id": uid, "item_id": w["item_id"], "axis": axis, "window_id": w["window_id"],
                    "result": row.get("result"), "label_probs": probs, "p_yes": (probs or {}).get("yes"),
                    "run_identity": row.get("run_identity"), "config_sha256": preparation["config_sha256"],
                    "preparation_sha256": preparation_sha256,
                    "source": items[w["item_id"]]["source"]}
            try:
                if row.get("custom_id") != uid or row.get("request_sha256") != requested.get(uid):
                    raise ValidationError("Output does not answer the prepared request")
                evidence = validate(row, w, config["schemas"][axis])
            except ValidationError as exc:
                fail = {**base, "status": row.get("status"), "validation_error": str(exc)}
                failures.append(fail)
                judgments.append({**fail, "validation_status": "failed"})
                res = row.get("result")
                if isinstance(res, dict) and res.get("label") == "yes" and row.get("status") == "success":
                    unverified.append({**fail, "validation_status": "unverified_candidate",
                                       "segment_sources": w["segment_sources"]})
                continue
            p_yes = base["p_yes"]
            ok = {**base, "validation_status": "validated", "evidence": evidence,
                  "segment_sources": w["segment_sources"]}
            judgments.append(ok)
            by_item_axis[(w["item_id"], axis)].append(ok)
            if row["result"]["label"] == "no" and p_yes is not None and p_yes >= near_miss_threshold:
                near.append(ok)
    stray = set(outputs) - seen
    if stray:
        raise ValueError(f"Outputs contain {len(stray)} units not in this run")
    results = []
    for item_id, item in items.items():
        axes = {}
        excluded = coverage[item_id]["status"] != "prepared" or not coverage[item_id]["windows"]
        for axis in AXES:
            rows = by_item_axis[(item_id, axis)]
            n = len(expected[(item_id, axis)])
            labels = [r["result"]["label"] for r in rows]
            yes = [r for r in rows if r["result"]["label"] == "yes"]
            probs = [r["p_yes"] for r in rows if r["p_yes"] is not None]
            if yes:
                best = max(yes, key=lambda r: -1 if r["p_yes"] is None else r["p_yes"])
                assessment = grade(best["p_yes"], thresholds, axis)
            elif not n and not excluded:
                assessment = "not_requested"  # e.g. a second pass over selected requests
            elif excluded or len(rows) < n:
                assessment = "incomplete"
            else:
                assessment = "insufficient_evidence" if "insufficient_evidence" in labels else "no_evidence"
            axes[axis] = {"assessment": assessment, "complete": not excluded and len(rows) == n,
                          "validated_windows": len(rows), "expected_windows": n,
                          "p_yes_max": max(probs) if probs else None,
                          "p_yes_positive_max": max((r["p_yes"] for r in yes if r["p_yes"] is not None), default=None),
                          "confidence_positive_max": max((r["result"]["confidence"] for r in yes), default=None),
                          "injection_detected": any(r["result"]["injection_detected"] for r in rows)}
        results.append({"item_id": item_id, "source": item["source"], "shape": item["shape"],
                        "text_sha256s": item["text_sha256s"], "windows": window_counts[item_id],
                        "preparation_status": coverage[item_id]["status"],
                        **({"benchmark_case_id": item["benchmark_case_id"]} if "benchmark_case_id" in item else {}),
                        "config_sha256": preparation["config_sha256"],
                        "classifier_sha256": preparation["classifier_sha256"],
                        "inference_sha256": inference_hash,
                        "axes": axes})
    queue = review_queue(judgments, items)
    return config, results, judgments, failures, pending, unverified, near, queue


QUEUE_ORDER = ("positive", "unverified_positive", "abstention", "failure")


def queue_reason(judgment):
    result = judgment.get("result") if isinstance(judgment.get("result"), dict) else {}
    if judgment["validation_status"] == "validated":
        return {"yes": "positive", "insufficient_evidence": "abstention"}.get(result.get("label"))
    if result.get("label") == "yes" and judgment.get("status") == "success":
        return "unverified_positive"
    return "failure"


def review_queue(judgments, items):
    """Positives on any axis, unverified positives, abstentions and failures; by confidence then P(yes) within each."""
    queue = []
    for j in judgments:
        reason = queue_reason(j)
        if reason is None:
            continue
        item = items[j["item_id"]]
        result = j["result"] if isinstance(j.get("result"), dict) else {}
        confidence = result.get("confidence")
        confidence = confidence if type(confidence) in (int, float) and 0 <= confidence <= 1 else None
        row = {**j, "queue_reason": reason, "confidence": confidence,
               "shape": item["shape"], "request_urls": item["request_urls"]}
        if j["axis"] == "hacking" and reason in ("positive", "unverified_positive"):
            categories = j["result"].get("categories")
            row["findings_cyberattack"] = [FINDINGS_CYBERATTACK[c] for c in dict.fromkeys(categories or ())
                                           if c in FINDINGS_CYBERATTACK]
        queue.append(row)
    return sorted(queue, key=lambda r: (QUEUE_ORDER.index(r["queue_reason"]),
                                        r["confidence"] is None, -(r["confidence"] or 0),
                                        r["p_yes"] is None, -(r["p_yes"] or 0), r["unit_id"]))


def _bucket(n):
    return "1" if n == 1 else "2-4" if n <= 4 else "5-19" if n <= 19 else "20+"


def write_report(dest, config, results, judgments, failures, pending, unverified, near, queue, thresholds):
    """Write report files. Synthetic controls are scored separately and excluded from corpus counts."""
    dest = Path(dest)
    control_rows = [r for r in results if r["source"] == control_set.SOURCE]
    results = [r for r in results if r["source"] != control_set.SOURCE]
    queue = [q for q in queue if q["source"] != control_set.SOURCE]
    control_counts = {}
    for name, rows in (("failed", failures), ("pending", pending), ("unverified_candidates", unverified)):
        control_counts[name] = sum(r["source"] == control_set.SOURCE for r in rows)
    failures = [r for r in failures if r["source"] != control_set.SOURCE]
    all_pending = len(pending)
    pending = [r for r in pending if r["source"] != control_set.SOURCE]
    unverified = [r for r in unverified if r["source"] != control_set.SOURCE]
    near = [r for r in near if r["source"] != control_set.SOURCE]
    dest.mkdir(parents=True, exist_ok=True)
    write_jsonl(dest / "control-results.jsonl", control_rows)
    for name, rows in (("item-results", results), ("judgments", judgments), ("failures", failures),
                       ("pending", pending), ("unverified-candidates", unverified),
                       ("near-misses", near), ("review-queue", queue),
                       ("injection-review", [j for j in judgments if j.get("source") != control_set.SOURCE
                                             and isinstance(j.get("result"), dict)
                                             and j["result"].get("injection_detected") is True])):
        write_jsonl(dest / f"{name}.jsonl", rows)
    by_axis = {a: dict(Counter(r["axes"][a]["assessment"] for r in results)) for a in AXES}
    positives = {"likely", "possible", "positive_uncalibrated"}
    negative = [j for j in judgments if j["validation_status"] == "validated" and j["result"]["label"] == "no"]
    window_rate = {}
    for a in AXES:
        groups = defaultdict(list)
        for r in results:
            groups[_bucket(r["windows"])].append(r["axes"][a]["assessment"] in positives)
        window_rate[a] = {b: {"items": len(v), "positive_rate": sum(v) / len(v)} for b, v in sorted(groups.items())}
    summary = {
        "version": config.get("version"), "thresholds": thresholds or None,
        "controls": {**control_set.score(control_rows), **control_counts},
        "items": len(results), "requests_expected": len(judgments) + all_pending,
        "validated": sum(j["validation_status"] == "validated" for j in judgments),
        "failed": len(failures), "pending": len(pending), "unverified_candidates": len(unverified),
        "near_misses": len(near), "assessments": by_axis,
        "review_queue": dict(Counter(q["queue_reason"] for q in queue)),
        "assessments_by_shape": {a: {s: dict(Counter(r["axes"][a]["assessment"] for r in results if r["shape"] == s))
                                     for s in sorted({r["shape"] for r in results})} for a in AXES},
        "positive_rate_by_window_count": window_rate,
        "injection_detected_items": sum(any(r["axes"][a]["injection_detected"] for a in AXES) for r in results),
        "label_probs_available": sum(j.get("label_probs") is not None for j in judgments),
        # Confidence should be near 0 on a "no"; a high value means the model read it as
        # certainty in its label. Tracked until the field's meaning is shown reliable.
        "confidence_on_negatives": {"validated_no": len(negative), "above_half": sum(
            type(j["result"].get("confidence")) in (int, float) and j["result"]["confidence"] > .5 for j in negative)},
        "failure_reasons": dict(Counter(f["validation_error"].split(":")[0] for f in failures)),
        "scope": "Screening for expert review. requests_expected, validated and label-score diagnostics include controls; item counts and reviewer files exclude them.",
        "score_semantics": "label_probs/p_yes are diagnostic first-token branch scores, not calibrated probabilities.",
        "release_ready": False,
        "release_note": "Requires held-out accuracy, per-axis labels, actual-model canaries and runtime validation.",
    }
    atomic_write_text(dest / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary
