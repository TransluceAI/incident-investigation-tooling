"""Measure the judge against a labeled answer key.

Protocol (practitioner guidance plus the research report in reports/):
- Split by leakage group so related cases never straddle partitions. The key's
  own evaluation_partition is used when present; otherwise groups are hashed
  into train (15%, few-shot only), dev (40%, tuning) and test (45%, run once
  per release candidate).
- Report per-class rates (TPR, FPR, precision, MCC) and confusion counts,
  never raw agreement, with 95% intervals from a bootstrap that resamples
  leakage groups.
- Missing and incomplete predictions are reported as coverage, not negatives.
- Precision on the key reflects the key's prevalence; production precision and
  review load are projected separately for assumed prevalences.
"""
import json
import math
import random
from collections import Counter, defaultdict

from .common import iter_jsonl, sha256_text, sha256_file, atomic_write_text
from .schemas import AXES

KEY_AXIS = {"hacking": "hacking", "agentic_activity": "agentic_activity", "coordination": "collusion"}
POSITIVE_LABELS = {"likely", "possible", "yes", "positive", "flag", "flagged", "true", "1"}
NEGATIVE_LABELS = {"no_evidence", "no", "negative", "none", "false", "0", "not_flagged"}
PREDICTED_POSITIVE = {"positive"}
UNDECIDED = {"incomplete", "insufficient_evidence"}


def truth(value):
    v = str(value).strip().lower() if value is not None else ""
    if v in POSITIVE_LABELS:
        return True
    if v in NEGATIVE_LABELS:
        return False
    return None


PARTITIONS = {"train": "train", "dev": "dev", "test": "test", "validation": "dev",
              "val": "dev", "tune": "dev", "tuning": "dev", "holdout": "test",
              "heldout": "test", "held_out": "test", "sealed": "test"}


def partition(row):
    given = str(row.get("evaluation_partition") or "").lower()
    if given:
        if given not in PARTITIONS:
            raise ValueError(f"Unknown evaluation partition: {given}")
        return PARTITIONS[given]
    group = row.get("leakage_group") or row.get("document_sha256") or row.get("id")
    if not group:
        raise ValueError("Missing stable split/group identity")
    bucket = int(sha256_text("partition|" + str(group))[:8], 16) % 100
    return "train" if bucket < 15 else "dev" if bucket < 55 else "test"


def validated_key(path):
    """Unify connected leakage groups/document hashes before assigning partitions.

    Explicit assignments are preserved, but conflicting assignments fail closed.
    The same connected component is also the bootstrap sampling unit.
    """
    rows = list(iter_jsonl(path))
    parents = {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)
    ids = set()
    for row in rows:
        if not row.get("id") or row["id"] in ids or not row.get("document_sha256"):
            raise ValueError("Key requires unique IDs and document hashes")
        ids.add(row["id"])
        group = "group:" + str(row.get("leakage_group") or row["id"])
        doc = "document:" + row["document_sha256"]
        union(group, doc)
        row["_node"] = group
    components = defaultdict(list)
    for row in rows:
        components[find(row.pop("_node"))].append(row)
    for group, members in components.items():
        explicit = {partition(r) for r in members if r.get("evaluation_partition")}
        if len(explicit) > 1:
            raise ValueError("Cross-partition leakage group/document: " + group)
        part = next(iter(explicit)) if explicit else partition({"leakage_group": group})
        for row in members:
            row["evaluation_partition"] = part
            row["_group"] = group
    return rows


def predictions(item_results):
    """Map document sha256 -> list of item results containing that text."""
    by_hash = defaultdict(list)
    for r in item_results:
        for h in r["text_sha256s"]:
            by_hash[h].append(r)
    return by_hash


def predict(rows, axis):
    """A body hash alone cannot select among different capture/provenance contexts."""
    if not rows:
        return None
    if len(rows) != 1:
        return "ambiguous"
    assessment = rows[0]["axes"][axis]["assessment"]
    if assessment in PREDICTED_POSITIVE:
        return True
    return False if assessment == "no_evidence" else "undecided"


def case_prediction(by_hash, row, axis):
    rows = by_hash.get(row["document_sha256"], [])
    if any("benchmark_case_id" in r for r in rows):
        rows = [r for r in rows if r.get("benchmark_case_id") == row["id"]]
    if row.get("item_id"):
        rows = [r for r in rows if r["item_id"] == row["item_id"]]
    if axis != "overall":
        return predict(rows, axis)
    preds = [predict(rows, a) for a in AXES]
    if any(p is True for p in preds):
        return True
    if not rows:
        return None
    if any(p == "ambiguous" for p in preds):
        return "ambiguous"
    return False if all(p is False for p in preds) else "undecided"


def confusion(pairs):
    c = Counter()
    for y, p in pairs:
        c[("tp" if p else "fn") if y else ("fp" if p else "tn")] += 1
    return c


def rates(c):
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    def ratio(a, b):
        return a / b if b else None
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": ratio(tp, tp + fn), "fpr": ratio(fp, fp + tn), "precision": ratio(tp, tp + fp),
            "mcc": (tp * tn - fp * fn) / denom if denom else None}


def cluster_bootstrap(cases, n=1000, seed=17):
    """cases: [(group, y, pred)]. 95% percentile intervals over resampled groups."""
    groups = defaultdict(list)
    for g, y, p in cases:
        groups[g].append((y, p))
    keys = sorted(groups)
    if not keys:
        return {}
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(n):
        pairs = [pair for g in (rng.choice(keys) for _ in keys) for pair in groups[g]]
        for k, v in rates(confusion(pairs)).items():
            if k in ("tpr", "fpr", "precision", "mcc") and v is not None:
                samples[k].append(v)
    out = {}
    for k, vals in samples.items():
        vals.sort()
        out[k] = [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]
    return out


def project(tpr, fpr, prevalences=(0.001, 0.01, 0.05), items=150_000):
    """Precision and review load at assumed production prevalence (Bayes' rule)."""
    out = []
    for prev in prevalences:
        if tpr is None or fpr is None:
            continue
        flagged = tpr * prev + fpr * (1 - prev)
        out.append({"prevalence": prev, "precision": tpr * prev / flagged if flagged else None,
                    "expected_flags": round(flagged * items), "expected_true": round(tpr * prev * items)})
    return out


def evaluate(item_results_path, key_path, out_path=None, include_test=False):
    results = list(iter_jsonl(item_results_path))
    by_hash = predictions(results)
    key = validated_key(key_path)
    report = {"key_rows": len(key), "partitions": dict(Counter(partition(r) for r in key)), "axes": {},
              "test_evaluated": include_test}
    targets = [("overall", None)] + [(a, KEY_AXIS[a]) for a in AXES]
    for name, key_axis in targets:
        per_part = {}
        for part in (("train", "dev", "test") if include_test else ("train", "dev")):
            cases = []
            missing = undecided = ambiguous = unlabeled = positive_total = negative_total = 0
            for row in key:
                if partition(row) != part:
                    continue
                y = truth(row.get("overall_label")) if key_axis is None else truth((row.get("axis_labels") or {}).get(key_axis))
                if y is None:
                    unlabeled += 1
                    continue
                positive_total += y
                negative_total += not y
                pred = case_prediction(by_hash, row, name)
                if pred is None:
                    missing += 1
                elif pred == "ambiguous":
                    ambiguous += 1
                elif pred == "undecided":
                    undecided += 1
                else:
                    cases.append((row["_group"], y, pred))
            r = rates(confusion([(y, p) for _, y, p in cases]))
            labeled = positive_total + negative_total
            ci = cluster_bootstrap(cases)
            warnings = []
            if len({g for g, _, _ in cases}) < 2:
                warnings.append("Fewer than two decided independent groups; interval unavailable.")
                ci = {}
            elif any(v[0] == v[1] for v in ci.values()):
                warnings.append("Degenerate empirical bootstrap interval does not establish certainty or zero population error.")
            per_part[part] = {**r, "decided": len(cases), "missing": missing, "undecided": undecided,
                              "ambiguous": ambiguous, "unlabeled": unlabeled,
                              "labeled_positives": positive_total, "labeled_negatives": negative_total,
                              "coverage": len(cases) / labeled if labeled else None,
                              "end_to_end_positive_recall": r["tp"] / positive_total if positive_total else None,
                              "ci95": ci, "interval_warnings": warnings,
                              "projection": project(r["tpr"], r["fpr"])}
        report["axes"][name] = per_part
    report["key_sha256"] = sha256_file(key_path)
    report["results_sha256"] = sha256_file(item_results_path)
    if out_path:
        atomic_write_text(out_path, json.dumps(report, indent=2) + "\n")
    return report
