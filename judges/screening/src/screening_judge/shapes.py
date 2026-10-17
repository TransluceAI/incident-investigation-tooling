"""Evidence shapes: which record kinds share a rendering, and how an Item becomes segments.

Five shapes cover every kind in the profiled data. An unknown kind falls back to
the text shape and is counted, never dropped.
"""
import json
import re

from . import scanner
from .decode import URL_RE, decode_url

SHAPES = ("capture", "scanner", "wiki", "message", "text")
KIND_SHAPE = {
    # Archived or redirected requests: the request URL is primary evidence.
    "arquivo_capture": "capture", "arquivo_index": "capture", "provider_redirect": "capture",
    "wayback_capture": "capture", "redirect_observation": "capture", "redirect_mapping": "capture",
    "shortlink": "capture", "live_shortlink_mapping": "capture", "prior_shortlink_mapping": "capture",
    "live_mapping": "capture", "prior_mapping": "capture", "destination_text": "capture",
    "lookup_json": "capture", "lookup_error_response": "capture",
    # Third-party scanner reports of a submitted URL.
    "urlquery_report": "scanner", "urlscan_report": "scanner", "urlscan_search": "scanner",
    # Wiki pages, histories and diffs: authorship and revision time matter.
    "wiki_page": "wiki", "wiki_history": "wiki", "wiki_changes": "wiki", "wiki_index": "wiki",
    "mw_html": "wiki", "live_historical_revision": "wiki", "prior_revision_diff": "wiki",
    "public_wiki_diff": "wiki", "current_wiki_html": "wiki", "change_index": "wiki",
    "change_rss": "wiki",
    # Short published messages.
    "retained_message": "message",
}
MAX_BODY_URL_VIEWS = 50


def shape_for(kind: str, body: str | None = None, source: str | None = None) -> str:
    # The corpus uses this kind for both notification messages and shortlink
    # pages. A kind alone does not justify notification-channel guidance.
    if kind == "live_public_content" and (source or "").split("/", 1)[0] == "ntfy.sh":
        return "message"
    shape = KIND_SHAPE.get(kind)
    if shape is None and body and body.lstrip().startswith("{"):
        try:
            if scanner.detect(json.loads(body)):
                return "scanner"
        except ValueError:
            pass
    return shape or "text"


def _add(segs, item_id, kind, source, text):
    segs.append({"segment_id": f"S{len(segs)}", "kind": kind, "source": source, "text": text})


def _url_views(segs, item_id, url, label):
    views, gaps = decode_url(url)
    for v in views:
        _add(segs, item_id, v["kind"], f"{label}.{v['source']}", v["text"])
    for g in gaps:
        _add(segs, item_id, "DECODE_GAP", f"{label}.{g['source']}", f"not decoded: {g['gap']}")


# This repository collects website revisions; its directory names are not
# independent authorship evidence. Preserve the full observation in items.jsonl.
COLLECTION_URL = re.compile(
    r'https://(?:raw\.githubusercontent\.com|github\.com)/'
    r'JoshuaDavid/WikiAgentSwarmInvestigation/[^\s|]+', re.I)


def neutral_collection_context(segments):
    """Remove collection URLs from metadata, including request_url fallbacks.

    Document text and its decoded views are evidence, so never redact them.
    Full original metadata and request URLs remain in items.jsonl. Any other
    mention of the collection name in metadata stops preparation.
    """
    for segment in segments:
        if segment["kind"] in BULK_KINDS or segment["source"].startswith("document:"):
            continue
        original = segment["text"]
        segment["text"] = COLLECTION_URL.sub("[collection archive URL; retained in audit record]", original)
        if segment["text"] != original:
            segment["source"] += ":neutral_collection_view"
        if "wikiagentswarminvestigation" in segment["text"].casefold():
            raise ValueError("Collection repository name leaked into prompt metadata: " + segment["segment_id"])
    return segments


def build_segments(item: dict, bodies: list) -> list:
    """bodies: [(document_path, text)] in observation order; empty when no text is stored.

    Segment kinds: ITEM, PROVENANCE, URL, URL_DECODED, NESTED_URL, PROGRAM,
    DECODED_B64, DECODE_GAP, REDIRECTS, BODY, BODY_VIEW. BODY_VIEW is a derived,
    pointer-addressed scanner view; every other kind is source text or an inert
    decoding of it.
    """
    iid, shape = item["item_id"], item["shape"]
    segs = []
    _add(segs, iid, "ITEM", "item", "\n".join([
        f"source: {item['source']}",
        f"shape: {shape}",
        f"record_kinds: {', '.join(sorted(item['kinds']))}",
        f"observations: {len(item['observations'])}",
        f"time_basis: {', '.join(sorted(item['time_basis'])) or 'unknown'}",
        f"stored_text: {'available' if bodies else 'missing'}",
    ]))
    _add(segs, iid, "PROVENANCE", "records", "\n".join(o["line"] for o in item["observations"]))
    for url in item["request_urls"]:
        _add(segs, iid, "URL", "request_url", url)
        _url_views(segs, iid, url, "request_url")
    if item.get("redirects"):
        _add(segs, iid, "REDIRECTS", "redirects",
             "\n".join(f"{r['from_replay_url']} -> {r['to_replay_url']}" for r in item["redirects"]))
    if not bodies:
        _add(segs, iid, "BODY", "none", "BODY: missing (no stored text for this item)")
        return neutral_collection_context(segs)
    for path, text in bodies:
        doc = None
        if shape == "scanner":
            try:
                doc = json.loads(text)
            except ValueError:
                doc = None
        if isinstance(doc, dict) and scanner.detect(doc):
            view, _ = scanner.view(doc)
            _add(segs, iid, "BODY_VIEW", f"document:{path}#json-pointer", view)
        else:
            view = text
            _add(segs, iid, "BODY", f"document:{path}", view)
        if shape in ("message", "text", "wiki", "scanner", "capture"):
            urls = list(dict.fromkeys(URL_RE.findall(view)))
            for n, url in enumerate(urls[:MAX_BODY_URL_VIEWS]):
                _url_views(segs, iid, url, f"document:{path}:body_url:{n}")
            if len(urls) > MAX_BODY_URL_VIEWS:
                _add(segs, iid, "DECODE_GAP", "body_url",
                     f"not decoded: {len(urls) - MAX_BODY_URL_VIEWS} further URLs remain readable in BODY")
    return neutral_collection_context(segs)


# Bulk segments are windowed; every other segment, including PROVENANCE, is context
# repeated in each window when it fits (render.windows).
BULK_KINDS = frozenset({"BODY", "BODY_VIEW"})
