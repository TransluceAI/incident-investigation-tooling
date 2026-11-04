"""Score OR-ensembles of judges on the findings benchmark key (Lab Notebook TR2).

An item is flagged by a judge when its item result is positive on any axis; an
ensemble flags it when any member does. Undecided, failed and missing items count
as not flagged. The pattern matcher alone flags an item when any of its windows
has any hint (hints.jsonl from `classifier-judge prepare --hints`).

Pre-registered rules (TR2, 2026-10-22), ensemble = screening OR campaign-aware judge:
  P1 recall >= 190/200;  P2 false positives <= 8/400;  P3 Findings-website recall >= 40/47
TR2 passes if P1 and P2 pass.

Usage:
  python scripts/ensemble.py --key answer_key.jsonl --screening S/item-results.jsonl \
    --recall R/item-results.jsonl --hints R/prepared/hints.jsonl [--classifier C/item-results.jsonl] \
    [--recall-outputs R/inference/outputs] --out tr2.json
"""
import argparse
import json
from collections import defaultdict
from urllib.parse import urlparse

from classifier_judge.core import runner_cost
from screening_judge.common import iter_jsonl
from screening_judge.schemas import AXES


def flagged_cases(path):
    return {r["benchmark_case_id"] for r in iter_jsonl(path)
            if "benchmark_case_id" in r and any(r["axes"][a]["assessment"] == "positive" for a in AXES)}


def hinted_cases(hints_path, results_path):
    case = {r["item_id"]: r.get("benchmark_case_id") for r in iter_jsonl(results_path)}
    hit = defaultdict(bool)
    for row in iter_jsonl(hints_path):
        hit[row["item_id"]] |= bool(row["hints"])
    return {case[i] for i, h in hit.items() if h and case.get(i)}


def family(row):
    if row["overall_label"] == "negative":
        return "negative"
    return "collusion.wiki" if "collusion" in urlparse(row.get("source_url") or "").netloc else "findings"


def score(key, flagged):
    out = {"positives": 0, "tp": 0, "negatives": 0, "fp": 0, "by_family": defaultdict(lambda: [0, 0])}
    for row in key:
        f = row["id"] in flagged
        fam = out["by_family"][family(row)]
        fam[0] += f
        fam[1] += 1
        if row["overall_label"] == "positive":
            out["positives"] += 1
            out["tp"] += f
        else:
            out["negatives"] += 1
            out["fp"] += f
    out["recall"] = out["tp"] / out["positives"]
    out["fpr"] = out["fp"] / out["negatives"]
    out["by_family"] = {k: {"flagged": v[0], "n": v[1]} for k, v in sorted(out["by_family"].items())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True)
    p.add_argument("--screening", required=True)
    p.add_argument("--recall", required=True)
    p.add_argument("--hints", required=True)
    p.add_argument("--classifier")
    p.add_argument("--recall-outputs", nargs="*")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = list(iter_jsonl(a.key))
    S, R = flagged_cases(a.screening), flagged_cases(a.recall)
    arms = {"screening": S, "recall_judge": R, "ensemble": S | R,
            "matcher_alone": hinted_cases(a.hints, a.recall)}
    if a.classifier:
        C = flagged_cases(a.classifier)
        arms["classifier"] = C
        arms["ensemble_plus_classifier"] = S | R | C
    report = {name: score(key, f) for name, f in arms.items()}
    e = report["ensemble"]
    findings = e["by_family"].get("findings", {"flagged": 0})
    report["verdict"] = {"P1": {"tp": e["tp"], "pass": e["tp"] >= 190},
                         "P2": {"fp": e["fp"], "pass": e["fp"] <= 8},
                         "P3": {"findings_flagged": findings["flagged"], "pass": findings["flagged"] >= 40},
                         "TR2_pass": e["tp"] >= 190 and e["fp"] <= 8}
    if a.recall_outputs:
        report["recall_judge_cost"] = runner_cost(a.recall_outputs)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    for name in arms:
        r = report[name]
        print(f"{name:26s} recall {r['tp']}/{r['positives']}  FP {r['fp']}/{r['negatives']}  {r['by_family']}")
    print(json.dumps(report["verdict"]))


if __name__ == "__main__":
    main()
