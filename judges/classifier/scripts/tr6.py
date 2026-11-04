"""Score the tuned campaign judge on new, untouched negatives (Lab Notebook TR6).

TR6 has no positives: it checks whether TR5's precision gain was fit to the
TR5 negatives. Arms: F = tuned campaign judge, C = untuned campaign judge,
V = main's v5 screener, each positive on any axis. Undecided, failed and
missing items count as not flagged.

Pre-registered rules (TR6, 2026-11-03):
  P1 F false-positive rate <= 0.03 on the TR6 negatives
  P2 F false-positive rate <= 0.10 on the TR6 scanner negatives
  P3 C false-positive rate >= 0.10 on the TR6 scanner negatives (recorded on its own)
TR6 passes if P1 and P2 pass; with fewer than 10 scanner negatives it is inconclusive.

Usage:
  python scripts/tr6.py --key answer_key.jsonl --tuned F/item-results.jsonl \
    --untuned C/item-results.jsonl --v5 V/item-results.jsonl \
    --tuned-outputs F-inf/outputs --untuned-outputs C-inf/outputs --out tr6.json
"""
import argparse
import json

from screening_judge.common import iter_jsonl
from screening_judge.schemas import AXES
from tr3 import incomplete
from tr5_dev import reasons


def axes_by_item(path):
    return {r["benchmark_case_id"]: [a for a in AXES if r["axes"][a]["assessment"] == "positive"]
            for r in iter_jsonl(path) if "benchmark_case_id" in r}


def item_ids(path):
    return {r["benchmark_case_id"]: r["item_id"] for r in iter_jsonl(path) if "benchmark_case_id" in r}


def score(key, hits):
    out = {}
    for fam in ("all", "scanner", "wiki"):
        rows = [k for k in key if fam == "all" or k["family"] == fam]
        fp = [k["id"] for k in rows if hits.get(k["id"])]
        out[fam] = {"fp": len(fp), "n": len(rows), "fpr": len(fp) / len(rows) if rows else None,
                    "per_axis": {a: sum(a in hits.get(k["id"], []) for k in rows) for a in AXES},
                    "false_alarms": sorted(fp)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True)
    p.add_argument("--tuned", required=True)
    p.add_argument("--untuned", required=True)
    p.add_argument("--v5", required=True)
    p.add_argument("--tuned-outputs")
    p.add_argument("--untuned-outputs")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = [k for k in iter_jsonl(a.key) if k["overall_label"] == "negative"]
    paths = {"tuned": a.tuned, "untuned": a.untuned, "v5": a.v5}
    hits = {name: axes_by_item(path) for name, path in paths.items()}
    report = {name: score(key, h) for name, h in hits.items()}
    f, c = report["tuned"], report["untuned"]
    scanner_n = f["scanner"]["n"]
    report["verdict"] = {
        "P1": {"fp": f["all"]["fp"], "n": f["all"]["n"], "pass": f["all"]["fpr"] <= 0.03},
        "P2": {"fp": f["scanner"]["fp"], "n": scanner_n,
               "pass": None if scanner_n < 10 else f["scanner"]["fpr"] <= 0.10},
        "P3": {"fp": c["scanner"]["fp"], "n": scanner_n,
               "pass": None if scanner_n < 10 else c["scanner"]["fpr"] >= 0.10},
    }
    v = report["verdict"]
    v["TR6"] = "INCONCLUSIVE" if scanner_n < 10 else ("PASS" if v["P1"]["pass"] and v["P2"]["pass"] else "FAIL")
    for name, outputs in (("tuned", a.tuned_outputs), ("untuned", a.untuned_outputs)):
        if not outputs:
            continue
        why, ids = reasons(outputs), item_ids(paths[name])
        report[name]["reasons"] = {cid: [t for c_id, t in why.items() if c_id.startswith(ids[cid] + ":")]
                                   for cid in report[name]["all"]["false_alarms"]}
    report["incomplete"] = {name: incomplete(path) for name, path in paths.items()}
    with open(a.out, "w") as out:
        json.dump(report, out, indent=2)
    for name in paths:
        r = report[name]
        print(f"{name:8s} " + "  ".join(f"{fam} {r[fam]['fp']}/{r[fam]['n']} {r[fam]['per_axis']}" for fam in ("all", "scanner", "wiki")))
    print(json.dumps(v))


if __name__ == "__main__":
    main()
