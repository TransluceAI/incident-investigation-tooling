"""Discover source records and group observations into judgeable Items.

An Item is the unit whose evidence is judged once. Grouping follows one rule:
reuse a judgment only when the complete evidence is identical.

- capture shape: keyed by (request URL, capture time). The same error body is
  routinely returned for different probe URLs, so a shared body hash is not
  shared evidence.
- every other shape: keyed by source, shape, stored-text hash and evidence
  metadata. Identical bodies with different provenance remain distinct.

Records whose fields this module does not understand are never dropped: they
become text-shape Items or explicit gaps.
"""
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from .common import sha256_text

URL_KEYS = ("original_url", "url", "request_url", "target_url", "source_url", "destination",
            "destination_url", "long_url", "page_url", "href")
TIME_KEYS = ("capture_timestamp", "timestamp", "capture_time", "published_at", "created_at",
             "revision_time", "source_time", "date", "observed_at", "retrieved_at")
URLISH = re.compile(r"(url|uri|href|destination|target|link)", re.I)
TIMEISH = re.compile(r"(time|date|_at$|timestamp)", re.I)


def sources(root: Path):
    """Yield (source_name, records_file) under resources/. Source = path relative to resources."""
    base = root / "resources"
    if not base.is_dir():
        raise ValueError("No resources directory in corpus")
    files = set(base.glob("*/records.jsonl")) | set(base.glob("**/DIRECT_RECORDS.jsonl"))
    for f in sorted(files):
        yield f.parent.relative_to(base).as_posix(), f


def iter_source_records(path: Path):
    """Yield (line_no, record_or_None, error) for one JSONL file."""
    with open(path, encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield i, json.loads(line), None
            except Exception as exc:
                yield i, None, f"{type(exc).__name__}: {exc}"


def metadata(rec: dict) -> dict:
    meta = rec.get("metadata")
    if not isinstance(meta, dict):
        original = rec.get("original_record")
        meta = original.get("metadata") if isinstance(original, dict) else None
    return meta if isinstance(meta, dict) else {}


def _scalars(obj, prefix="", depth=0):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _scalars(v, f"{prefix}{k}.", depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _scalars(v, f"{prefix}{i}.", depth + 1)
    elif isinstance(obj, (str, int, float, bool)) and obj != "":
        yield prefix[:-1], obj


def _pick(rec, names, pattern, want_url, skip=None):
    meta = metadata(rec)
    for scope in (meta, rec):
        for name in names:
            v = scope.get(name)
            if isinstance(v, str) and v and v != skip and (not want_url or v.startswith(("http://", "https://"))):
                return v
    for key, v in _scalars({"metadata": meta, **{k: x for k, x in rec.items() if k != "original_record"}}):
        if isinstance(v, str) and v != skip and pattern.search(key.rsplit(".", 1)[-1]):
            if not want_url or v.startswith(("http://", "https://")):
                return v
    return None


def request_url(rec, source):
    """The URL the record observed. An import location never qualifies."""
    return _pick(rec, URL_KEYS, URLISH, True, skip=import_location(rec, source))


def import_location(rec, source):
    """Where an imported record was copied from, unless that is the source site itself.

    Such a value names an earlier collection (a repository, export file or local
    path), not the observed activity, so prompts must not read it as evidence.
    """
    url = rec.get("source_url")
    if rec.get("collection_method") != "imported" or not isinstance(url, str) or not url:
        return None
    host = (urlsplit(url).hostname or "").removeprefix("www.")
    site = source.split("/", 1)[0].lower().removeprefix("www.")
    return None if host and (host == site or host.endswith("." + site)) else url


def observed_time(rec):
    return _pick(rec, TIME_KEYS, TIMEISH, False)


def _provenance_fields(rec) -> dict:
    skip = {"document_path", "raw_path", "text_sha256", "original_record"}
    fields = {k: v for k, v in rec.items() if k not in skip}
    meta = rec.get("original_record", {}).get("metadata") if isinstance(rec.get("original_record"), dict) else None
    if isinstance(meta, dict) and "metadata" not in rec:
        fields["metadata"] = meta
    return fields


def _line(fields) -> str:
    return " | ".join(f"{k}={v}" for k, v in _scalars(fields))


def provenance_line(rec) -> str:
    """One compact, complete line of scalar provenance for an observation."""
    return _line(_provenance_fields(rec))


# Fields that describe the observed activity. Everything else a collector wrote
# (selection reasons, run names, window checks, verdict-like notes) can steer the
# judge without being evidence, so prompts omit it and the audit line keeps it.
ACTIVITY_FIELDS = frozenset({
    "kind", "source", "site", "wiki", "collection_method", "time_basis",
    "title", "page", "comment", "summary", "revision", "revision_id", "parent_id",
    "user", "username", "author", "editor", "actor", "ip", "method", "user_agent", "referrer",
    "status", "status_code", "http_status", "original_status", "original_location",
    "mime", "content_type", "representation", "body_withheld", "credentials_redacted",
    "request", "requests", "response", "responses", "headers", "payload", "params", "query", "body", "cookies"})


# For prompts, a key names a URL or time only when its last word does: request_url,
# fetched_at, createdAt. A substring is not enough (candidate_reason, url_selection_reason).
_LAST_WORD = r"(?:^|[_.-]|(?<=[a-z0-9])(?=[A-Z]))"
URL_NAMED = re.compile(_LAST_WORD + r"(?i:url|uri|href|destination|target|link)s?$")
TIME_NAMED = re.compile(_LAST_WORD + r"(?i:time|date|datetime|timestamp|at)s?$")


def _describes_activity(key) -> bool:
    return key in ACTIVITY_FIELDS or bool(URL_NAMED.search(key) or TIME_NAMED.search(key))


def prompt_provenance(rec):
    """The provenance line shown to the model (activity fields only) and the omitted field names."""
    kept, omitted = {}, []
    for k, v in _provenance_fields(rec).items():
        if k == "metadata" and isinstance(v, dict):
            kept[k] = {mk: mv for mk, mv in v.items() if _describes_activity(mk)}
            omitted += [f"metadata.{mk}" for mk in v if mk not in kept[k]]
        elif _describes_activity(k):
            kept[k] = v
        else:
            omitted.append(k)
    line = _line(kept)
    if omitted:
        line += f"{' | ' if line else ''}[collection fields omitted: {len(omitted)}; kept in audit record]"
    return line, omitted


def item_key(source, shape, rec):
    if shape == "capture":
        url, ts = request_url(rec, source), observed_time(rec)
        if url:
            return "capture|" + sha256_text(f"{source}|{url}|{ts or rec['observation_id']}")[:24]
    if rec.get("text_sha256"):
        # Different provenance can change attribution, timing, and shape. Only
        # collapse exact evidence equivalents, excluding storage/observation IDs.
        context = {k: v for k, v in rec.items()
                   if k not in {"observation_id", "document_path", "raw_path", "text_sha256"}}
        signature = json.dumps(context, sort_keys=True, ensure_ascii=False)
        return "text|" + sha256_text(f"{source}|{shape}|{rec['text_sha256']}|{signature}")[:24]
    return "obs|" + sha256_text(f"{source}|{rec.get('observation_id')}")[:24]
