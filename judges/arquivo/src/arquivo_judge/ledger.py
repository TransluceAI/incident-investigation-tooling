"""Event ledger: one row per distinct (original URL, capture timestamp) on arquivo.pt."""
import json
import re
from dataclasses import dataclass, field

from .common import sha256_text

PROVIDER = "arquivo.pt"
REPLAY_RE = re.compile(r"arquivo\.pt/(?:noFrame/replay|wayback)/(\d{14})(?:[a-z]{2}_)?/(.+)$")
KINDS = ("arquivo_index", "arquivo_capture", "provider_redirect")


def parse_replay_url(url):
    m = REPLAY_RE.search(url or "")
    return (m.group(1), m.group(2)) if m else None


def make_event_id(url: str, ts: str) -> str:
    return sha256_text(f"{PROVIDER}|{url}|{ts}")[:24]


def capture_metadata(rec):
    meta = rec.get("metadata") or (rec.get("original_record") or {}).get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


@dataclass
class Ledger:
    events: dict = field(default_factory=dict)
    observation_map: list = field(default_factory=list)
    gaps: list = field(default_factory=list)

    def rows(self):
        return [self.events[k] for k in sorted(self.events)]


def _event(ledger, url, ts):
    eid = make_event_id(url, ts)
    ev = ledger.events.get(eid)
    if ev is None:
        ev = {"event_id": eid, "provider": PROVIDER, "original_url": url, "capture_timestamp": ts,
              "cdx": [], "body": None, "extra_bodies": [], "body_gaps": [], "redirects": [], "observation_ids": [],
              "scope_labels": [], "time_basis": []}
        ledger.events[eid] = ev
    return ev


def _note(ev, rec):
    for key, dest in (("scope", "scope_labels"), ("time_basis", "time_basis")):
        value = rec.get(key)
        if value and value not in ev[dest]:
            ev[dest].append(value)
    if rec["observation_id"] not in ev["observation_ids"]:
        ev["observation_ids"].append(rec["observation_id"])


def _read(read_text, rec):
    path = rec.get("document_path")
    return read_text(path) if path else ""


def build_ledger(records, read_text) -> Ledger:
    ledger = Ledger()
    by_kind = {k: [] for k in KINDS}
    for rec in records:
        if rec.get("kind") in by_kind:
            by_kind[rec["kind"]].append(rec)
        else:
            ledger.gaps.append({"observation_id": rec.get("observation_id"), "gap": "unknown_kind", "kind": rec.get("kind")})
            ledger.observation_map.append({"observation_id": rec.get("observation_id"), "kind": rec.get("kind"),
                                           "disposition": "parse_error", "event_ids": []})
    for rec in by_kind["arquivo_index"]:
        if not rec.get("document_path"):
            ledger.gaps.append({"observation_id": rec["observation_id"], "gap": "index_missing_document"})
        linked = []
        for row_no, line in enumerate(_read(read_text, rec).splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                url, ts = row["url"], row["timestamp"]
                if not isinstance(url, str) or not isinstance(ts, str):
                    raise ValueError("url/timestamp not strings")
            except Exception as exc:  # malformed line: keep going, record the gap
                ledger.gaps.append({"observation_id": rec["observation_id"], "gap": "index_parse_error",
                                    "row_no": row_no, "error": f"{type(exc).__name__}: {exc}"})
                continue
            ev = _event(ledger, url, ts)
            ev["cdx"].append({k: row.get(k) for k in ("mime", "status", "digest", "length", "collection")}
                             | {"index_observation_id": rec["observation_id"], "row_no": row_no})
            _note(ev, rec)
            linked.append(ev["event_id"])
        ledger.observation_map.append({"observation_id": rec["observation_id"], "kind": "arquivo_index",
                                       "disposition": "index_container", "event_ids": sorted(set(linked))})
    for rec in by_kind["arquivo_capture"]:
        meta = capture_metadata(rec)
        url, ts = meta.get("original_url"), meta.get("capture_timestamp")
        if not isinstance(url, str) or not isinstance(ts, str) or not url or not ts:
            ledger.gaps.append({"observation_id": rec["observation_id"], "gap": "capture_missing_metadata"})
            ledger.observation_map.append({"observation_id": rec["observation_id"], "kind": "arquivo_capture",
                                           "disposition": "parse_error", "event_ids": []})
            continue
        ev = _event(ledger, url, ts)
        if not rec.get("document_path"):
            rec = {**rec, "body_gap": "capture_missing_document"}
            ledger.gaps.append({"observation_id": rec["observation_id"], "gap": rec["body_gap"], "event_id": ev["event_id"]})
        if rec.get("body_gap"):
            # The capture exists but its text is missing, unreadable or unverified: keep
            # the event (its URL is still evidence) and record why it has no body.
            ev["body_gaps"].append({"observation_id": rec["observation_id"], "gap": rec["body_gap"]})
            _note(ev, rec)
            ledger.observation_map.append({"observation_id": rec["observation_id"], "kind": "arquivo_capture",
                                           "disposition": "unreadable_capture", "event_ids": [ev["event_id"]]})
            continue
        body = {"observation_id": rec["observation_id"], "document_path": rec.get("document_path"),
                "text_sha256": rec.get("text_sha256"), "raw_path": rec.get("raw_path"), "cdx_digest": meta.get("cdx_digest")}
        if ev["body"] is None:
            ev["body"] = body
        elif ev["body"]["text_sha256"] != body["text_sha256"]:
            ev["extra_bodies"].append(body)
            ledger.gaps.append({"observation_id": rec["observation_id"], "gap": "body_conflict", "event_id": ev["event_id"]})
        _note(ev, rec)
        ledger.observation_map.append({"observation_id": rec["observation_id"], "kind": "arquivo_capture",
                                       "disposition": "event", "event_ids": [ev["event_id"]]})
    for rec in by_kind["provider_redirect"]:
        parsed = parse_replay_url(rec.get("source_url", ""))
        target = _read(read_text, rec).strip()
        eid = make_event_id(parsed[1], parsed[0]) if parsed else None
        if eid in ledger.events:
            ev = ledger.events[eid]
            ev["redirects"].append({"observation_id": rec["observation_id"],
                                    "from_replay_url": rec.get("source_url"), "to_replay_url": target})
            _note(ev, rec)
            ledger.observation_map.append({"observation_id": rec["observation_id"], "kind": "provider_redirect",
                                           "disposition": "redirect", "event_ids": [eid]})
        else:
            ledger.gaps.append({"observation_id": rec["observation_id"], "gap": "redirect_unlinked",
                                "source_url": rec.get("source_url"), "to_replay_url": target})
            ledger.observation_map.append({"observation_id": rec["observation_id"], "kind": "provider_redirect",
                                           "disposition": "redirect_unlinked", "event_ids": []})
    return ledger
