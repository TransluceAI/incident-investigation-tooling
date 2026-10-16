"""Build Items from an exported data root and write inference requests.

Every record ends in exactly one place: an Item, the Arquivo ledger, or a gap.
coverage.jsonl and gaps.jsonl make that accounting auditable.

requests.jsonl holds one self-contained request per (window, axis) for the
shared inference runner (inference/qwen, REQUESTS.md): finished messages, the
axis schema and logprob options. engine.json is the runner profile for the run.
"""
import contextlib
import json
import os
import secrets
from collections import Counter
from pathlib import Path

from . import arquivo_ledger, benchmark, controls, records, shapes
from .common import atomic_write_text, sha256_file, sha256_text, write_jsonl, classifier_hash
from .prompts import load_prompts, messages, schemas
from .render import render_citable, windows
from .schemas import AXES, citation_schema

VERSION = "screening-judge-v1"
# Profile keys the judge consumes; the runner receives only engine/sampling settings.
NOT_FOR_RUNNER = {"system_prompt", "schema", "citation_constraints", "token_count_mode"}


class Corpus:
    """Reads stored texts relative to the data root, verifying declared hashes."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def read(self, rel):
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Path outside corpus: " + rel)
        text = path.read_bytes().decode("utf-8")
        self.hashes[rel] = sha256_text(text)
        return text


@contextlib.contextmanager
def jsonl_stream(path):
    """Append rows to path.tmp; publish atomically only after a clean finish."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        yield lambda row: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _new_item(item_id, source, shape):
    return {"item_id": item_id, "source": source, "shape": shape, "kinds": set(),
            "observations": [], "time_basis": set(), "request_urls": [], "documents": [],
            "redirects": []}


def _observe(item, rec):
    item["kinds"].add(str(rec.get("kind")))
    if rec.get("time_basis"):
        item["time_basis"].add(str(rec["time_basis"]))
    item["observations"].append({"observation_id": rec["observation_id"], "kind": rec.get("kind"),
                                 "line": records.provenance_line(rec)})
    if item["shape"] == "capture":
        url = records.request_url(rec)
        if url and url not in item["request_urls"]:
            item["request_urls"].append(url)
    if rec.get("document_path"):
        doc = (rec["document_path"], rec.get("text_sha256"))
        if doc not in item["documents"]:
            item["documents"].append(doc)


def _arquivo_items(source, recs, corpus, items, gaps):
    safe = []
    for rec in recs:
        if rec["kind"] in {"arquivo_index", "provider_redirect"}:
            candidate = _new_item("unreadable-" + sha256_text(source + rec["observation_id"])[:24], source, "capture")
            _observe(candidate, rec)
            if item_bodies(candidate, corpus, gaps) is None:
                items[candidate["item_id"]] = candidate
                continue
        meta = records.metadata(rec)
        if rec["kind"] == "arquivo_capture" and any(not isinstance(meta.get(k), str)
                                                        for k in ("original_url", "capture_timestamp")):
            meta = {}
        safe.append({**rec, "metadata": meta, "original_record": {}})
    ledger = arquivo_ledger.build_ledger(safe, corpus.read)
    by_obs = {r["observation_id"]: r for r in recs}
    for ev in ledger.rows():
        item = _new_item(sha256_text(source + "|" + ev["event_id"])[:24], source, "capture")
        for oid in ev["observation_ids"]:
            _observe(item, by_obs[oid])
        item["request_urls"] = [ev["original_url"]]
        item["time_basis"].add(f"capture_timestamp={ev['capture_timestamp']}")
        if ev["cdx"]:
            item["observations"].append({"observation_id": None, "kind": "cdx",
                                         "line": "cdx: " + " ; ".join(json.dumps(c, sort_keys=True) for c in ev["cdx"])})
        item["documents"] = [(b["document_path"], b["text_sha256"])
                             for b in ([ev["body"]] if ev["body"] else []) + ev["extra_bodies"]
                             if b["document_path"]]
        item["redirects"] = ev["redirects"]
        items[item["item_id"]] = item
    for m in ledger.observation_map:
        if m["disposition"] == "redirect_unlinked":
            rec = by_obs[m["observation_id"]]
            item = _new_item("redirect-" + sha256_text(source + "|" + rec["observation_id"])[:24], source, "capture")
            _observe(item, rec)
            item["request_urls"] = [rec.get("source_url")] if rec.get("source_url") else []
            items[item["item_id"]] = item
    represented = {o["observation_id"] for item in items.values() if item["source"] == source
                   for o in item["observations"]}
    for rec in recs:
        if rec["observation_id"] not in represented:
            item = _new_item("unlinked-" + sha256_text(source + "|" + rec["observation_id"])[:24], source, "capture")
            _observe(item, rec)
            items[item["item_id"]] = item
            gaps.append({"source": source, "observation_id": rec["observation_id"], "gap": "archive_record_unlinked"})
    gaps.extend({**g, "source": source} for g in ledger.gaps)


def build_items(root, corpus, gaps):
    items, arquivo, seen = {}, {}, set()
    for source, path in records.sources(Path(root)):
        for line_no, rec, error in records.iter_source_records(path):
            where = {"source": source, "file": str(path.relative_to(root)), "line": line_no}
            if (error or not isinstance(rec, dict) or not isinstance(rec.get("observation_id"), str)
                    or not rec["observation_id"] or not isinstance(rec.get("kind"), str)
                    or (rec.get("document_path") is not None and not isinstance(rec["document_path"], str))):
                gaps.append({**where, "gap": "unreadable_record", "error": error})
                continue
            if (source, rec["observation_id"]) in seen:
                gaps.append({**where, "gap": "duplicate_observation", "observation_id": rec["observation_id"]})
                continue
            seen.add((source, rec["observation_id"]))
            if rec.get("kind") in arquivo_ledger.KINDS:
                arquivo.setdefault(source, []).append(rec)
                continue
            shape = shapes.shape_for(rec.get("kind"), source=source)
            key = records.item_key(source, shape, rec)
            item = items.setdefault(key, _new_item(sha256_text(key)[:24], source, shape))
            _observe(item, rec)
    for source, recs in arquivo.items():
        _arquivo_items(source, recs, corpus, items, gaps)
    return sorted(items.values(), key=lambda i: (i["source"], i["item_id"]))


def item_bodies(item, corpus, gaps):
    """Return verified [(path, text)], or None when integrity fails (never judged as clean)."""
    out = []
    for path, declared in item["documents"]:
        try:
            text = corpus.read(path)
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            gaps.append({"item_id": item["item_id"], "gap": "document_unreadable", "path": path,
                         "error": f"{type(exc).__name__}: {exc}"})
            return None
        if declared and sha256_text(text) != declared:
            gaps.append({"item_id": item["item_id"], "gap": "document_hash_mismatch", "path": path})
            return None
        out.append((path, text))
    return out


def prepare(data_root, out, count, window_tokens, config_template, config_dir=None, with_controls=True, shards=1,
            benchmark_manifest=None, label_logprobs=10, citation_constraints=True):
    if window_tokens <= 0 or type(shards) is not int or shards < 1:
        raise ValueError("window_tokens and shards must be positive")
    if type(label_logprobs) is not int or not 0 <= label_logprobs <= 20:
        raise ValueError("label_logprobs must be an integer from 0 to 20")
    runner = runner_profile(config_template)
    if type(citation_constraints) is not bool:
        raise ValueError("citation_constraints must be true or false")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    corpus, gaps = Corpus(data_root), []
    config = {**config_template, "version": VERSION, "prompts": load_prompts(config_dir) if config_dir else load_prompts(),
              "schemas": schemas(), "boundary_salt": secrets.token_hex(16),
              "window_tokens": window_tokens, "label_logprobs": label_logprobs,
              "citation_constraints": citation_constraints,
              "preparation_implementation": {
                  name: sha256_file(Path(__file__).with_name(name + ".py"))
                  for name in ("arquivo_ledger", "benchmark", "common", "controls", "decode", "prepare",
                               "prompts", "records", "render", "scanner", "schemas", "shapes")}}
    widest = None  # Each axis is measured; character length does not order token counts.
    items = (benchmark.build_items(corpus.root, benchmark_manifest, gaps) if benchmark_manifest
             else build_items(corpus.root, corpus, gaps))
    if not items:
        write_jsonl(out / "gaps.jsonl", gaps)
        raise ValueError("No judgeable records found; refusing an empty corpus run")
    coverage, n_windows = [], 0
    with jsonl_stream(out / "windows.jsonl") as put_window, jsonl_stream(out / "requests.jsonl") as put_request:
        for item in items:
            n_windows += _prepare_item(item, corpus, gaps, config, count, widest, window_tokens,
                                       put_window, put_request, coverage)
        if with_controls:
            for item, bodies, _, _ in controls.control_items():
                items.append(item)
                n_windows += _prepare_item(item, corpus, gaps, config, count, widest, window_tokens,
                                           put_window, put_request, coverage, bodies)
    item_rows = [{**{k: v for k, v in i.items() if k not in ("kinds", "time_basis")},
                  "kinds": sorted(i["kinds"]), "time_basis": sorted(i["time_basis"]),
                  "text_sha256s": sorted({d[1] or corpus.hashes.get(d[0]) for d in i["documents"]
                                          if d[1] or corpus.hashes.get(d[0])})} for i in items]
    write_jsonl(out / "items.jsonl", item_rows)
    write_jsonl(out / "coverage.jsonl", coverage)
    write_jsonl(out / "gaps.jsonl", gaps)
    atomic_write_text(out / "config.json", json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(out / "engine.json", json.dumps(runner, ensure_ascii=False, indent=2) + "\n")
    summary = {"version": VERSION, "items": len(items), "windows": n_windows,
               "requests": n_windows * len(AXES),
               "items_by_shape": dict(Counter(i["shape"] for i in items)),
               "items_by_source": dict(Counter(i["source"] for i in items)),
               "windowed_items": sum(len(c["windows"]) > 1 for c in coverage),
               "excluded_integrity": sum(c["status"] == "excluded_integrity" for c in coverage),
               "gaps": dict(Counter(g["gap"] for g in gaps)),
               "config_sha256": sha256_file(out / "config.json"),
               "classifier_sha256": classifier_hash(config),
               "artifact_hashes": {name: sha256_file(out / name) for name in
                                   ("requests.jsonl", "engine.json", "windows.jsonl", "items.jsonl",
                                    "coverage.jsonl", "gaps.jsonl")}}
    if shards > 1:
        names, loads = write_shards(out, shards)
        summary["artifact_hashes"].update({name: sha256_file(out / name) for name in names})
        summary["shard_characters"] = loads
    atomic_write_text(out / "preparation.json", json.dumps(summary, indent=2) + "\n")
    return summary


def runner_profile(config_template):
    """The runner's config: the profile's engine and sampling settings only."""
    if "label_logprobs" in config_template:
        raise ValueError("Set label scores with --label-logprobs; runner profiles do not carry label_logprobs")
    if "citation_constraints" in config_template:
        raise ValueError("Citation narrowing is a judge option (--no-citation-constraints), not a runner profile key")
    return {k: v for k, v in config_template.items() if k not in NOT_FOR_RUNNER}


def write_shards(out, shards):
    """Split requests.jsonl into shards/part-N.jsonl (the runner launcher's --shard-dir naming).

    A window's axis requests stay together and adjacent so they share the prefix cache.
    """
    from itertools import groupby
    from .common import iter_jsonl
    (out / "shards").mkdir()
    loads = [0] * shards
    names = [f"shards/part-{i}.jsonl" for i in range(shards)]
    with contextlib.ExitStack() as stack:
        writers = [stack.enter_context(jsonl_stream(out / name)) for name in names]
        window = lambda row: row["custom_id"].rsplit(":", 1)[0]
        for _, group in groupby(iter_jsonl(out / "requests.jsonl"), key=window):
            worker = min(range(shards), key=lambda i: (loads[i], i))
            for row in group:
                writers[worker](row)
                loads[worker] += sum(len(m["content"]) for m in row["body"]["messages"])
    return names, loads


def _prepare_item(item, corpus, gaps, config, count, widest, window_tokens, put_window, put_request, coverage,
                  bodies=None):
    """Write one item's windows; return how many were written."""
    if bodies is None:
        bodies = item_bodies(item, corpus, gaps)
    if bodies is None:
        coverage.append({"item_id": item["item_id"], "status": "excluded_integrity", "windows": []})
        return 0
    if item["shape"] == "text" and bodies:
        item["shape"] = shapes.shape_for(sorted(item["kinds"])[0], bodies[0][1], source=item["source"])
    segs = shapes.build_segments(item, bodies)

    def measure(text, shape=item["shape"]):
        return max(count(messages(config, shape, axis, text)) for axis in AXES)

    ids = []
    for idx, (parts, offsets) in enumerate(windows(segs, measure, window_tokens)):
        text, spans, chunks = render_citable(parts, offsets)
        wid = f"{item['item_id']}:w{idx}"
        ids.append(wid)
        digest = sha256_text(text)
        put_window({"window_id": wid, "item_id": item["item_id"], "chunk_idx": idx,
                    "shape": item["shape"], "text": text, "text_sha256": digest,
                    "segment_spans": spans, "source_offsets": offsets, "evidence_chunks": chunks,
                    "segment_sources": {s["segment_id"]: s["source"] for s in parts},
                    "segment_kinds": {s["segment_id"]: s["kind"] for s in parts}})
        for axis in AXES:
            schema = citation_schema(axis, chunks) if config.get("citation_constraints") else config["schemas"][axis]
            body = {"messages": messages(config, item["shape"], axis, text),
                    "response_format": {"type": "json_schema", "json_schema": {"name": axis, "schema": schema}}}
            if config["label_logprobs"]:
                body.update(logprobs=True, top_logprobs=config["label_logprobs"])
            put_request({"custom_id": f"{wid}:{axis}", "body": body})
    coverage.append({"item_id": item["item_id"], "status": "prepared", "windows": ids,
                     "segment_lengths": {s["segment_id"]: len(s["text"]) for s in segs}})
    return len(ids)
