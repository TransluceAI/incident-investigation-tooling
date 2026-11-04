"""Compare the classifier and the screening judge on a labeled benchmark.

End-to-end rates: an axis counts as flagged only when the judge's item result is
positive. Missing, incomplete, abstained and ambiguous cases count as not
flagged and are listed, so a judge cannot gain by failing. Overall means any of
the three axes. Intervals resample leakage groups (screening evaluate.py).

Pre-registered rules (Lab Notebook TR1, 2026-10-20), simple vs screening:
  P1 overall TPR  >= screening - 0.05
  P2 overall FPR  <= screening + 0.03
  P3 output tokens <= 0.5x and GPU-seconds <= 0.6x
  P4 per-axis recall on hand labels >= screening - 0.10, for axes with >= 15 yes labels
TR1 passes if P1, P2 and P3 pass; P4 is recorded on its own.

Usage:
  python scripts/compare.py --key answer_key.jsonl [--axis-key hand_labels.jsonl] \
    --screening-results S/report/item-results.jsonl --screening-outputs S/inference/outputs \
    --classifier-results C/report/item-results.jsonl --classifier-outputs C/inference/outputs \
    --out comparison.json
"""
import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from urllib.parse import urlparse

from classifier_judge.core import runner_cost
from screening_judge.common import iter_jsonl
from screening_judge.evaluate import KEY_AXIS, case_prediction, cluster_bootstrap, predictions, validated_key
from screening_judge.schemas import AXES

JUDGES = ("screening", "classifier")
TARGETS = ("overall",) + AXES


def family(row):
    if row["overall_label"] == "negative":
        return "negative"
    host = urlparse(row.get("source_url") or "").netloc
    return "collusion.wiki" if "collusion" in host else "findings"


def band(row):
    n = row.get("estimated_tokens_o200k") or 0
    return "<=1k" if n <= 1000 else "<=8k" if n <= 8000 else "<=64k" if n <= 64000 else ">64k"


def flagged(by_hash, row, target):
    pred = case_prediction(by_hash, row, target)
    return pred is True, ("decided" if pred in (True, False) else str(pred))


def rate(cases, positive):
    """cases: [(group, flagged)] all with the same truth. Returns rate and group-bootstrap CI."""
    if not cases:
        return {"n": 0, "flagged": 0, "rate": None, "ci95": None}
    k = sum(f for _, f in cases)
    boot = cluster_bootstrap([(g, positive, f) for g, f in cases])
    return {"n": len(cases), "flagged": k, "rate": k / len(cases),
            "ci95": boot.get("tpr" if positive else "fpr")}


def load_key(path):
    """The benchmark key says "development"; screening evaluate.py only accepts "dev" and its aliases."""
    rows = [{**r, "evaluation_partition": "dev"} if r.get("evaluation_partition") == "development" else r
            for r in iter_jsonl(path)]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    try:
        return validated_key(f.name)
    finally:
        os.unlink(f.name)


def compare(key_path, axis_key_path, results, outputs):
    key = load_key(key_path)
    hashes = {j: predictions(list(iter_jsonl(results[j]))) for j in JUDGES}
    report = {"rates": {}, "status_counts": {}, "by_family": {}, "by_length": {}, "disagreements": []}
    for j in JUDGES:
        report["rates"][j], report["status_counts"][j] = {}, {}
        for target in TARGETS:
            pos, neg, status = [], [], Counter()
            for row in key:
                truth = row["overall_label"] == "positive"
                if target != "overall" and truth:
                    continue  # positives carry no per-axis labels in the benchmark key
                hit, how = flagged(hashes[j], row, target)
                status[how] += 1
                (pos if truth else neg).append((row["_group"], hit))
            report["rates"][j][target] = {"tpr": rate(pos, True), "fpr": rate(neg, False)}
            report["status_counts"][j][target] = dict(status)
        fam, length = defaultdict(Counter), defaultdict(Counter)
        for row in key:
            hit, _ = flagged(hashes[j], row, "overall")
            for table, name in ((fam, family(row)), (length, band(row))):
                table[name]["n"] += 1
                table[name]["flagged"] += hit
        report["by_family"][j] = {k: dict(v) for k, v in sorted(fam.items())}
        report["by_length"][j] = {k: dict(v) for k, v in sorted(length.items())}
    for row in key:
        preds = {j: {t: flagged(hashes[j], row, t)[0] for t in TARGETS} for j in JUDGES}
        if preds["screening"] != preds["classifier"]:
            report["disagreements"].append({"id": row["id"], "label": row["overall_label"],
                                            "family": family(row), "source_url": row.get("source_url"),
                                            **{j: [t for t in TARGETS if preds[j][t]] for j in JUDGES}})
    report["cost"] = {j: runner_cost(outputs[j]) for j in JUDGES}
    if axis_key_path:
        report["hand_labeled"] = hand_labeled(axis_key_path, key, hashes)
    report["verdict"] = verdict(report)
    return report


def wilson(k, n, z=1.96):
    """Document-level 95% interval; treats related documents as independent, so it is too narrow."""
    if not n:
        return {"n": 0, "flagged": 0, "rate": None, "wilson95": None}
    p = k / n
    centre, half = (p + z * z / (2 * n)), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return {"n": n, "flagged": k, "rate": p, "wilson95": [(centre - half) / d, (centre + half) / d]}


def hand_labeled(path, key, hashes):
    """Per-axis recall on hand-labeled positives. Labels: yes / no / unclear.

    Nearly all benchmark positives share one leakage group, so intervals here are
    document-level Wilson intervals (Lab Notebook TR1 amendment A1), not the group bootstrap.
    """
    by_id = {r["id"]: r for r in key}
    out = {}
    labels = list(iter_jsonl(path))
    for axis in AXES:
        yes = [by_id[r["id"]] for r in labels if r["axis_labels"][KEY_AXIS[axis]] == "yes"]
        no = [by_id[r["id"]] for r in labels if r["axis_labels"][KEY_AXIS[axis]] == "no"]
        out[axis] = {"yes_labels": len(yes), "no_labels": len(no),
                     "unclear": sum(r["axis_labels"][KEY_AXIS[axis]] == "unclear" for r in labels)}
        for j in JUDGES:
            hit = lambda rows: sum(flagged(hashes[j], r, axis)[0] for r in rows)
            out[axis][j] = {"recall": wilson(hit(yes), len(yes)), "flag_rate_on_no": wilson(hit(no), len(no))}
    return out


def verdict(report):
    r, c = report["rates"], report["cost"]
    s_tpr, c_tpr = r["screening"]["overall"]["tpr"]["rate"], r["classifier"]["overall"]["tpr"]["rate"]
    s_fpr, c_fpr = r["screening"]["overall"]["fpr"]["rate"], r["classifier"]["overall"]["fpr"]["rate"]
    ratio = lambda k: c["classifier"].get(k, 0) / c["screening"][k] if c["screening"].get(k) else None
    tokens, gpu = ratio("output_tokens"), ratio("gpu_seconds")
    out = {"P1": {"screening_tpr": s_tpr, "classifier_tpr": c_tpr, "pass": c_tpr >= s_tpr - 0.05},
           "P2": {"screening_fpr": s_fpr, "classifier_fpr": c_fpr, "pass": c_fpr <= s_fpr + 0.03},
           "P3": {"output_token_ratio": tokens, "gpu_seconds_ratio": gpu,
                  "pass": None if tokens is None or gpu is None else tokens <= 0.5 and gpu <= 0.6}}
    hl = report.get("hand_labeled")
    if hl:
        axes = {a: {"screening": v["screening"]["recall"]["rate"], "classifier": v["classifier"]["recall"]["rate"],
                    "pass": v["classifier"]["recall"]["rate"] >= v["screening"]["recall"]["rate"] - 0.10}
                for a, v in hl.items() if v["yes_labels"] >= 15}
        out["P4"] = {"axes": axes, "pass": all(a["pass"] for a in axes.values()) if axes else None}
    out["TR1_pass"] = None if out["P3"]["pass"] is None else out["P1"]["pass"] and out["P2"]["pass"] and out["P3"]["pass"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True)
    p.add_argument("--axis-key")
    for j in JUDGES:
        p.add_argument(f"--{j}-results", required=True)
        p.add_argument(f"--{j}-outputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    report = compare(a.key, a.axis_key, {j: getattr(a, f"{j}_results") for j in JUDGES},
                     {j: getattr(a, f"{j}_outputs") for j in JUDGES})
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["verdict"], indent=2))


if __name__ == "__main__":
    main()
