"""Two-pass screening: select pass-1 requests for a reasoning pass, then merge.

Pass 1 runs every request with reasoning off. escalate() selects requests for
pass 2 (reasoning on): positives on any axis, unverified positives, abstentions,
failures, borderline P(yes), pass-0 rule hits, and a seeded random audit of
pass-1 negatives, which is the only way to estimate what pass 1 misses. The
pass-2 directory is an ordinary prepared run over that subset of the same
requests, with its own runner profile, so the runner and report are unchanged.

combine() takes the pass-2 answer where pass 2 validated one and otherwise
keeps pass 1. Pass-2 demotions of pass-1 positives stay visible in their own
low-priority queue until calibration shows reasoning does not lose real attacks.
"""
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from . import rules
from .controls import SOURCE as CONTROL_SOURCE
from .common import atomic_write_text, iter_jsonl, sha256_file, sha256_text, write_jsonl
from .prepare import runner_profile, write_shards
from .report import queue_reason, review_queue

COPIED = ("config.json", "windows.jsonl", "items.jsonl", "coverage.jsonl", "gaps.jsonl")
POSITIVE = ("positive", "unverified_positive")


def _judgments(report, preparation_sha256):
    """A report's judgments, refusing any reconciled against a different prepared run."""
    rows = {j["unit_id"]: j for j in iter_jsonl(Path(report) / "judgments.jsonl")}
    if any(j.get("preparation_sha256") != preparation_sha256 for j in rows.values()):
        raise ValueError("Report does not belong to this prepared run")
    return rows


def _audited(seed, uid, rate):
    # Deterministic per request: the same seed selects the same audit sample.
    return int(sha256_text(f"{seed}:{uid}"), 16) / 16 ** 64 < rate


def _pass1_rank(demoted):
    """Order demotions by the pass-1 scores that queued them: confidence, then P(yes)."""
    first = demoted["pass1"]
    result = first["result"] if isinstance(first.get("result"), dict) else {}
    confidence, p_yes = result.get("confidence"), first.get("p_yes")
    return (confidence is None, -(confidence or 0), p_yes is None, -(p_yes or 0), demoted["unit_id"])


def escalate(run, report, out, engine_template, borderline=(0.2, 0.8), audit_rate=0.03, seed=0,
             shards=1, allow_pending=False):
    run, out = Path(run), Path(out)
    lo, hi = borderline
    if not 0 <= lo <= hi <= 1 or not 0 <= audit_rate <= 1 or type(shards) is not int or shards < 1:
        raise ValueError("Invalid borderline band, audit rate or shard count")
    preparation = json.loads((run / "preparation.json").read_text())
    if preparation.get("pass", 1) != 1:
        raise ValueError("Escalate from a pass-1 run")
    judgments = _judgments(report, sha256_file(run / "preparation.json"))
    runner = runner_profile(engine_template)
    reasons = defaultdict(list)
    axes = list(json.loads((run / "config.json").read_text())["schemas"])
    expected = set()
    for w in iter_jsonl(run / "windows.jsonl"):
        fired = rules.hits(w)
        for axis in axes:
            uid = f"{w['window_id']}:{axis}"
            expected.add(uid)
            j = judgments.get(uid)
            if j is None:
                continue
            reasons[uid] += [f"rule:{name}" for name in fired]
            why = queue_reason(j)
            if why:
                reasons[uid].append(why)
            elif j.get("p_yes") is not None and lo <= j["p_yes"] <= hi:
                reasons[uid].append("borderline")
            elif not reasons[uid] and _audited(seed, uid, audit_rate):
                reasons[uid].append("audit")
    if set(judgments) - expected:
        raise ValueError("Pass-1 report contains units outside the prepared run")
    pending = expected - set(judgments)
    if pending and not allow_pending:
        raise ValueError(f"Pass 1 has {len(pending)} pending requests; finish it or pass allow_pending")
    selected = {uid for uid, why in reasons.items() if why}
    out.mkdir(parents=True, exist_ok=False)
    for name in COPIED:
        shutil.copyfile(run / name, out / name)
    write_jsonl(out / "requests.jsonl", (r for r in iter_jsonl(run / "requests.jsonl") if r["custom_id"] in selected))
    write_jsonl(out / "selection.jsonl", ({"custom_id": uid, "reasons": reasons[uid]} for uid in
                                          (r["custom_id"] for r in iter_jsonl(out / "requests.jsonl"))))
    atomic_write_text(out / "engine.json", json.dumps(runner, ensure_ascii=False, indent=2) + "\n")
    counts = Counter(reason.split(":")[0] if reason.startswith("rule:") else reason
                     for uid in selected for reason in reasons[uid])
    summary = {**{k: preparation[k] for k in ("version", "config_sha256")},
               "pass": 2, "requests": len(selected), "pass1_requests": len(judgments),
               "selection": dict(counts), "borderline": [lo, hi], "audit_rate": audit_rate, "seed": seed,
               "parent": {"run": str(run.resolve()), "preparation_sha256": sha256_file(run / "preparation.json"),
                          "judgments_sha256": sha256_file(Path(report) / "judgments.jsonl")},
               "artifact_hashes": {name: sha256_file(out / name) for name in
                                   ("requests.jsonl", "engine.json", "selection.jsonl", *COPIED[1:])}}
    if not selected:
        summary["note"] = "No requests selected; there is nothing to run for pass 2."
    elif shards > 1:
        names, loads = write_shards(out, shards)
        summary["artifact_hashes"].update({name: sha256_file(out / name) for name in names})
        summary["shard_characters"] = loads
    atomic_write_text(out / "preparation.json", json.dumps(summary, indent=2) + "\n")
    return summary


def combine(pass1_report, pass2_run, pass2_report, out, allow_pending=False):
    pass2_run, out = Path(pass2_run), Path(out)
    preparation = json.loads((pass2_run / "preparation.json").read_text())
    if preparation.get("pass") != 2:
        raise ValueError("Not a pass-2 run")
    if sha256_file(Path(pass1_report) / "judgments.jsonl") != preparation["parent"]["judgments_sha256"]:
        raise ValueError("Pass-1 report differs from the one pass 2 was selected from")
    first = _judgments(pass1_report, preparation["parent"]["preparation_sha256"])
    second = _judgments(pass2_report, sha256_file(pass2_run / "preparation.json"))
    axes = json.loads((pass2_run / "config.json").read_text())["schemas"]
    expected = {f"{w['window_id']}:{a}": {"item_id": w["item_id"], "axis": a, "window_id": w["window_id"]}
                for w in iter_jsonl(pass2_run / "windows.jsonl") for a in axes}
    if set(first) - set(expected):
        raise ValueError("Pass-1 report contains units outside the prepared run")
    pending1 = set(expected) - set(first)
    if pending1 and not allow_pending:
        raise ValueError(f"Pass 1 has {len(pending1)} pending requests; finish it or pass allow_pending")
    selection = {r["custom_id"]: r["reasons"] for r in iter_jsonl(pass2_run / "selection.jsonl")}
    if set(selection) - set(first):
        raise ValueError("Pass-2 selection contains units absent from pass 1")
    if set(second) - set(selection):
        raise ValueError("Pass-2 report contains requests that were not selected")
    unfinished = set(selection) - set(second)
    if unfinished and not allow_pending:
        # Merging now would keep pass-1 negatives that were selected for another look.
        raise ValueError(f"Pass 2 has {len(unfinished)} pending requests; finish it or pass allow_pending")
    items = {i["item_id"]: i for i in iter_jsonl(pass2_run / "items.jsonl")}
    final, demoted, unresolved, controls, pending, changes = [], [], [], [], [], Counter()
    audit = Counter()
    for uid in sorted(pending1):
        source = items[expected[uid]["item_id"]]["source"]
        row = {"unit_id": uid, **expected[uid], "source": source, "pass": None,
               "validation_status": "pending", "pass1_pending": True, "selection_reasons": []}
        (controls if source == CONTROL_SOURCE else final).append(row)
        if source != CONTROL_SOURCE:
            changes["pass1_pending"] += 1
            pending.append({**row, "pass": 1})
    for uid, j1 in first.items():
        j2 = second.get(uid)
        reasons = selection.get(uid, [])
        if j1.get("source") == CONTROL_SOURCE:
            # Synthetic controls are kept apart, as in the ordinary report.
            controls.append({**(j2 if j2 is not None and j2["validation_status"] == "validated" else j1),
                             "pass": 2 if j2 is not None and j2["validation_status"] == "validated" else 1,
                             "selection_reasons": reasons,
                             **({"pass2_pending": True} if uid in unfinished else {})})
            continue
        if uid in unfinished:
            changes["pass2_pending"] += 1
            pending.append({"unit_id": uid, "item_id": j1["item_id"], "axis": j1["axis"],
                            "source": j1["source"], "pass": 2, "selection_reasons": reasons})
        if j2 is not None and j2["validation_status"] == "validated":
            chosen = {**j2, "pass": 2, "pass1_reason": queue_reason(j1)}
            before, after = queue_reason(j1), queue_reason(j2)
            if before in POSITIVE and after not in POSITIVE:
                changes["demoted"] += 1
                demoted.append({**chosen, "pass1": j1, "selection_reasons": reasons})
            elif before not in POSITIVE and after == "positive":
                changes["promoted"] += 1
            if reasons == ["audit"]:
                audit["audited"] += 1
                audit["pass2_positive"] += after == "positive"
        else:
            chosen = {**j1, "pass": 1, **({"pass2_pending": True} if uid in unfinished else {})}
            if j2 is not None:
                # Keep the pass-1 classification, but the failed or unverified pass-2
                # answer (for example an uncited pass-2 yes) still goes to review.
                changes["pass2_failed"] += 1
                chosen["pass2_error"] = j2.get("validation_error")
                chosen["pass2"] = j2
                unresolved.append({**j2, "pass": 2, "pass1": j1, "pass1_reason": queue_reason(j1),
                                   "selection_reasons": reasons})
        final.append({**chosen, "selection_reasons": reasons})
    lowered = {d["unit_id"] for d in demoted}
    ranked = review_queue([j for j in final if j["unit_id"] not in lowered
                           and j["validation_status"] != "pending"] + unresolved, items)
    # review_queue ranks by urgency: retain that answer and combine the triggers.
    queued = {}
    for row in ranked:
        kept = queued.setdefault(row["unit_id"], {**row, "queue_reasons": []})
        if row["queue_reason"] not in kept["queue_reasons"]:
            kept["queue_reasons"].append(row["queue_reason"])
    queue = list(queued.values())
    if len(final) + len(controls) != len(expected):
        raise ValueError("Cascade did not account for every prepared request")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "final-judgments.jsonl", final)
    write_jsonl(out / "control-judgments.jsonl", controls)
    write_jsonl(out / "review-queue.jsonl", queue)
    write_jsonl(out / "demoted.jsonl", sorted(demoted, key=_pass1_rank))
    write_jsonl(out / "pending.jsonl", pending)
    summary = {"requests": len(final), "pass2_requests": len(selection), "pass2_validated": sum(
                   j["pass"] == 2 for j in final), **dict(changes),
               "review_queue": dict(Counter(q["queue_reason"] for q in queue)), "demoted_queue": len(demoted),
               "pass2_unresolved_queued": len(unresolved),
               "controls": {"requests": len(controls),
                            "pass1_pending": sum(c.get("pass1_pending", False) for c in controls),
                            "pass2_pending": sum(c.get("pass2_pending", False) for c in controls)},
               "audit": {**dict(audit), "note": "Counts audited pass-1 negatives that pass 2 validated, and how many "
                                                "of those pass 2 answered yes. Observed flips only: not an estimate "
                                                "of total misses or recall."},
               "selection": preparation["selection"]}
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary
