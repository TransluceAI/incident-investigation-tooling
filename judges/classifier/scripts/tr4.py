"""Score the restricted ensemble against main's v5 screener on the fresh set (Lab Notebook TR4).

Same fresh set and campaign-aware judge outputs as TR3; only the screening arm
changes to main's v5 rubric (a060ef0). Ensemble E = v5 screening positive on any
axis OR campaign positive on coordination or hacking. Undecided, failed and
missing items count as not flagged.

Pre-registered rules (TR4, 2026-11-03):
  P1 E recall >= V recall + 0.05 on the fresh positives
  P2 E false-positive rate <= 0.03 on the fresh negatives
  P3 V recall <= TR3 screening recall (9/21)
TR4 passes if P1 and P2 pass; P3 is recorded on its own.

Usage:
  python scripts/tr4.py --key answer_key.jsonl --v5 V/item-results.jsonl \
    --campaign R/item-results.jsonl --tr3-screening S/item-results.jsonl --out tr4.json
"""
import argparse
import json
from collections import defaultdict

from screening_judge.common import iter_jsonl
from tr3 import flagged, incomplete, score

TR3_GAINED = ("fresh-0d1f5928bae7d881", "fresh-aa76f07be351815c")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True)
    p.add_argument("--v5", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--tr3-screening", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = list(iter_jsonl(a.key))
    V, S = flagged(a.v5), flagged(a.tr3_screening)
    C_restricted = flagged(a.campaign, ("coordination", "hacking"))
    arms = {"v5_screening": V, "tr3_screening": S, "campaign_coord_hack": C_restricted,
            "ensemble_v5": V | C_restricted}
    report = {name: score(key, f) for name, f in arms.items()}
    v, e = report["v5_screening"], report["ensemble_v5"]
    report["verdict"] = {
        "P1": {"ensemble_recall": e["recall"], "v5_recall": v["recall"],
               "pass": e["recall"] >= v["recall"] + 0.05},
        "P2": {"ensemble_fpr": e["fpr"], "pass": e["fpr"] <= 0.03},
        "P3": {"v5_tp": v["tp"], "tr3_screening_tp": 9, "pass": v["tp"] <= 9},
    }
    report["verdict"]["TR4_pass"] = report["verdict"]["P1"]["pass"] and report["verdict"]["P2"]["pass"]
    by_finding = defaultdict(lambda: {"n": 0, "v5": 0, "ensemble": 0, "tr3_screening": 0})
    for k in key:
        if k["overall_label"] == "positive":
            f = by_finding[k.get("finding")]
            f["n"] += 1
            f["v5"] += k["id"] in V
            f["ensemble"] += k["id"] in V | C_restricted
            f["tr3_screening"] += k["id"] in S
    report["positives_by_finding"] = dict(by_finding)
    report["tr3_gained_flagged_by_v5"] = {i: i in V for i in TR3_GAINED}
    report["v5_vs_tr3_screening"] = {"v5_only": sorted(V - S), "tr3_only": sorted(S - V)}
    report["incomplete"] = {"v5": incomplete(a.v5), "campaign": incomplete(a.campaign)}
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    for name in arms:
        r = report[name]
        print(f"{name:20s} recall {r['tp']}/{r['positives']}  FP {r['fp']}/{r['negatives']}  {r['by_family']}")
    print(json.dumps(report["verdict"]))


if __name__ == "__main__":
    main()
