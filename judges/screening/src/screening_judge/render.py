"""Citable rendering and complete windowing (adapted from judges/arquivo).

The model cites [CHUNK N] numbers inside a segment rather than copying text;
validation resolves those pointers to exact source slices. Every character of
every segment is supplied to some window; nothing is truncated.

Change from Arquivo: every small context segment (metadata, URLs and all of
their decoded views) is repeated in each window when it fits, not only the
EVENT and URL segments. In Arquivo a long event split across windows never
showed its decoded program next to its body.
"""
from .shapes import BULK_KINDS

SEG_OPEN = "[SEGMENT {sid} {kind}]\n"
SEG_CLOSE = "\n[/SEGMENT]\n"
CHUNK = 240


def render_citable(segments, source_offsets=None):
    """Number exact, contiguous source slices; offsets are Unicode character offsets."""
    text, spans, chunks = "", {}, {}
    for s in segments:
        sid = s["segment_id"]
        base = source_offsets[sid][0] if source_offsets is not None else 0
        text += SEG_OPEN.format(sid=sid, kind=s["kind"])
        begin = len(text)
        chunks[sid] = []
        for start in range(0, len(s["text"]), CHUNK):
            value = s["text"][start:start + CHUNK]
            text += f"[CHUNK {len(chunks[sid]) + 1}]\n"
            pos = len(text)
            text += value
            chunks[sid].append({"text_start": pos, "text_end": len(text),
                                "source_start": base + start,
                                "source_end": base + start + len(value)})
            text += "\n[/CHUNK]\n"
        spans[sid] = [begin, len(text)]
        text += SEG_CLOSE
    return text, spans, chunks


def validate_chunk_map(unit):
    """Check complete source-offset coverage and bounds of the extraction map."""
    chunks = unit["evidence_chunks"]
    if set(chunks) != set(unit["segment_spans"]) or set(chunks) != set(unit["source_offsets"]):
        raise ValueError("Chunk map segment mismatch")
    for sid, entries in chunks.items():
        source_start, source_end = unit["source_offsets"][sid]
        lo, hi = unit["segment_spans"][sid]
        if not (0 <= source_start <= source_end and 0 <= lo <= hi <= len(unit["text"])):
            raise ValueError("Invalid segment bounds: " + sid)
        cursor, rendered_end = source_start, lo
        for c in entries:
            a, b = c["text_start"], c["text_end"]
            if not (0 <= lo <= rendered_end <= a < b <= hi <= len(unit["text"])
                    and c["source_start"] == cursor
                    and 0 < b - a == c["source_end"] - cursor <= CHUNK):
                raise ValueError("Invalid chunk bounds: " + sid)
            cursor, rendered_end = c["source_end"], b
        if cursor != source_end:
            raise ValueError("Incomplete chunk coverage: " + sid)


def windows(segments, count, budget, overlap=2000):
    """Yield (segments, offsets) covering every character of every segment.

    count(text) must return the full prompt token count for a rendered window.
    """
    full = render_citable(segments)[0]
    if count(full) <= budget:
        yield segments, {s["segment_id"]: [0, len(s["text"])] for s in segments}
        return
    fixed, variable = [], []
    for s in segments:
        if s["kind"] not in BULK_KINDS and count(render_citable(fixed + [s])[0]) < budget // 3:
            fixed.append(s)
        else:
            variable.append(s)
    for s in variable:
        start = 0
        while start < len(s["text"]) or (start == 0 and not s["text"]):
            remaining = len(s["text"]) - start
            lo = min(remaining, budget * 2)
            while True:
                part = {**s, "text": s["text"][start:start + lo]}
                if count(render_citable(fixed + [part])[0]) <= budget:
                    break
                if lo == 0:
                    raise ValueError("Context segments alone exceed window budget")
                lo = int(lo * 0.8)
            if not lo and remaining:
                raise ValueError("No window fits; refusing to omit evidence")
            end = start + lo
            spans = {x["segment_id"]: [0, len(x["text"])] for x in fixed}
            spans[s["segment_id"]] = [start, end]
            yield fixed + [{**s, "text": s["text"][start:end]}], spans
            if end == len(s["text"]):
                break
            start = end - min(overlap, lo // 4)
