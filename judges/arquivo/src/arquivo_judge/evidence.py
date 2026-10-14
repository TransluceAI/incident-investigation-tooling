"""Per-event evidence view: ordered, citable segments and a deterministic rendering."""
from .decode import decode_url

MISSING_BODY = "BODY: missing (no exported capture for this event)"
SEG_OPEN = "[SEGMENT {sid} {kind}]\n"
SEG_CLOSE = "\n[/SEGMENT]\n"


def build_segments(event: dict, body_text) -> list:
    segs = []

    def add(kind, source, text):
        segs.append({"segment_id": f"{event['event_id']}:{len(segs)}", "kind": kind, "source": source, "text": text})

    cdx = event.get("cdx") or []
    statuses = sorted({str(c.get("status")) for c in cdx if c.get("status")})
    mimes = sorted({str(c.get("mime")) for c in cdx if c.get("mime")})
    colls = sorted({str(c.get("collection")) for c in cdx if c.get("collection")})
    add("EVENT", "ledger", "\n".join([
        f"provider: {event.get('provider', 'arquivo.pt')}",
        f"capture_timestamp: {event['capture_timestamp']}",
        f"cdx_status: {', '.join(statuses) or 'unknown'}",
        f"mime: {', '.join(mimes) or 'unknown'}",
        f"collection: {', '.join(colls) or 'unknown'}",
        f"index_rows: {len(cdx)}",
        f"body: {'available' if body_text is not None else 'missing'}",
    ]))
    add("URL", "ledger.original_url", event["original_url"])
    views, gaps = decode_url(event["original_url"])
    for v in views:
        add(v["kind"], v["source"], v["text"])
    for g in gaps:
        add("DECODE_GAP", g["source"], f"not decoded: {g['gap']}")
    if event.get("redirects"):
        add("REDIRECTS", "ledger.redirects",
            "\n".join(f"{r['from_replay_url']} -> {r['to_replay_url']}" for r in event["redirects"]))
    if body_text is None:
        add("BODY", "none", MISSING_BODY)
    else:
        add("BODY", f"document:{(event.get('body') or {}).get('document_path')}", body_text)
    return segs


def render(segments):
    text, spans = "", {}
    for s in segments:
        text += SEG_OPEN.format(sid=s["segment_id"], kind=s["kind"])
        start = len(text)
        text += s["text"]
        spans[s["segment_id"]] = [start, len(text)]
        text += SEG_CLOSE
    return text, spans


def render_citable(segments, source_offsets=None):
    """Number exact, contiguous source slices; model returns a pointer, not a copy.

    Chunk numbers are local to the segment in this window. Offsets are Unicode
    character offsets, as elsewhere in the manifest, not UTF-8 byte offsets.
    """
    text, spans, chunks = '', {}, {}
    for s in segments:
        sid = s['segment_id']
        base = source_offsets[sid][0] if source_offsets is not None else 0
        text += SEG_OPEN.format(sid=sid, kind=s['kind'])
        begin = len(text)
        chunks[sid] = []
        for start in range(0, len(s['text']), 240):
            value = s['text'][start:start + 240]
            text += f'[CHUNK {len(chunks[sid]) + 1}]\n'
            pos = len(text)
            text += value
            chunks[sid].append({'text_start': pos, 'text_end': len(text),
                                'source_start': base + start,
                                'source_end': base + start + len(value)})
            text += '\n[/CHUNK]\n'
        spans[sid] = [begin, len(text)]
        text += SEG_CLOSE
    return text, spans, chunks


def validate_chunk_map(unit):
    """Check complete source-offset coverage and bounds of the extraction map."""
    chunks = unit['evidence_chunks']
    if set(chunks) != set(unit['segment_spans']) or set(chunks) != set(unit['source_offsets']):
        raise ValueError('Chunk map segment mismatch')
    for sid, entries in chunks.items():
        source_start, source_end = unit['source_offsets'][sid]
        lo, hi = unit['segment_spans'][sid]
        if not (0 <= source_start <= source_end and 0 <= lo <= hi <= len(unit['text'])):
            raise ValueError('Invalid segment bounds: ' + sid)
        cursor, rendered_end = source_start, lo
        for c in entries:
            a, b = c['text_start'], c['text_end']
            if not (0 <= lo <= rendered_end <= a < b <= hi <= len(unit['text'])
                    and c['source_start'] == cursor
                    and 0 < b - a == c['source_end'] - cursor <= 240):
                raise ValueError('Invalid chunk bounds: ' + sid)
            cursor, rendered_end = c['source_end'], b
        if cursor != source_end:
            raise ValueError('Incomplete chunk coverage: ' + sid)
