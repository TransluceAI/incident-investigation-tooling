"""Score one TR5 tuning round on a benchmark split (Lab Notebook TR5).

Prints recall, false positives by family and per axis, and writes the reason
for every false alarm so a round's change can target them. Families: scanner
(urlscan/urlquery reports) and wiki (everything else among negatives).

Usage:
  python scripts/tr5_dev.py --key key_tune.jsonl --results R/item-results.jsonl \
    --outputs R/inference/outputs --out round.json
"""
import argparse
import json
from pathlib import Path

from screening_judge.common import iter_jsonl
from screening_judge.schemas import AXES


def family(row):
    if row["overall_label"] == "positive":
        return "positive"
    url = row.get("source_url") or ""
    return "scanner" if any(s in url for s in ("urlscan", "urlquery", "scan")) else "wiki"


def reasons(outputs):
    out = {}
    for path in Path(outputs).glob("worker-*/part-*.jsonl"):
        for r in iter_jsonl(path):
            try:
                out[r["custom_id"]] = json.loads(r["content"]).get("reason", "")
            except (KeyError, TypeError, ValueError):
                continue
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--outputs")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    key = {k["id"]: k for k in iter_jsonl(a.key)}
    res = {r["benchmark_case_id"]: r for r in iter_jsonl(a.results) if r.get("benchmark_case_id") in key}
    pos = lambda r, ax: r is not None and r["axes"][ax]["assessment"] == "positive"
    table = {fam: {ax: 0 for ax in (*AXES, "any")} | {"n": 0}
             for fam in ("positive", "scanner", "wiki")}
    false_alarms = []
    for cid, k in key.items():
        fam, r = family(k), res.get(cid)
        table[fam]["n"] += 1
        hits = [ax for ax in AXES if pos(r, ax)]
        for ax in hits:
            table[fam][ax] += 1
        table[fam]["any"] += bool(hits)
        if hits and fam != "positive":
            false_alarms.append({"id": cid, "family": fam, "axes": hits, "url": k.get("source_url")})
    why = reasons(a.outputs) if a.outputs else {}
    for fa in false_alarms:
        item = res[fa["id"]]["item_id"] + ":"
        fa["reasons"] = [v for c, v in why.items() if c.startswith(item)] or None
    report = {"table": table, "missing": len(key) - len(res), "false_alarms": false_alarms}
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    for fam, row in table.items():
        print(f"{fam:9s} n={row['n']:3d}  " + "  ".join(f"{ax}={row[ax]}" for ax in (*AXES, "any")))
    print("missing results:", report["missing"])


if __name__ == "__main__":
    main()
