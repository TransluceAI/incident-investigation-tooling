"""Citable rendering and complete windowing for screening evidence.

The model cites [CHUNK N] numbers inside a segment rather than copying text;
validation resolves those pointers to exact source slices. Every character of
every segment is supplied to some window; nothing is truncated.

Every small context segment (metadata, URLs and all of their decoded views)
is repeated in each window when it fits, so decoded programs remain visible
alongside the body even when a long item spans several windows.
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


def provenance_header(segment, fixed, count, limit):
    """Leading provenance that fits beside the fixed context.

    Whole lines when they fit, otherwise a character prefix of the first line.
    Raises when not even a short prefix fits, rather than leaving body windows
    without provenance.
    """
    sid, text = segment["segment_id"], segment["text"]
    lines = text.split("\n")

    def header(body):
        return {**segment, "segment_id": sid + "h", "kind": "PROVENANCE_HEADER", "text": body}

    def fits(candidate):
        return count(render_citable(fixed + [candidate])[0]) < limit

    def leading(n):
        return header("\n".join(lines[:n] + [
            f"[{len(lines) - n} further provenance lines are in the windows of segment {sid}]"]))

    lo, hi = 0, len(lines) - 1  # most leading whole lines that fit; lo is always a measured fit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(leading(mid)):
            lo = mid
        else:
            hi = mid - 1
    if lo:
        return leading(lo)
    note = f"\n[provenance continues in the windows of segment {sid}]"
    lo, hi = 0, len(lines[0])  # longest first-line prefix that fits
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(header(lines[0][:mid] + note)):
            lo = mid
        else:
            hi = mid - 1
    if not lo:
        raise ValueError("No provenance prefix fits the window budget; refusing windows without provenance")
    return header(lines[0][:lo] + note)


def windows(segments, count, budget, overlap=2000):
    """Yield (segments, offsets) covering every character of every segment.

    count(text) must return the full prompt token count for a rendered window.
    """
    full = render_citable(segments)[0]
    full_tokens = count(full)
    if full_tokens <= budget:
        yield segments, {s["segment_id"]: [0, len(s["text"])] for s in segments}
        return
    # count() includes the prompt around the evidence, so repeated context gets a
    # third of the room left after that overhead, not a third of the whole budget.
    overhead = count(render_citable([])[0])
    limit = overhead + max(budget - overhead, 0) // 3
    fixed, variable, fixed_tokens = [], [], overhead
    for s in segments:
        # Each context segment is counted once on its own; a join adds at most a token.
        alone = None if s["kind"] in BULK_KINDS else count(render_citable([s])[0]) - overhead
        if alone is not None and fixed_tokens + alone + 1 < limit:
            fixed.append(s)
            fixed_tokens += alone + 1
        else:
            variable.append(s)
    while fixed and count(render_citable(fixed)[0]) >= limit:
        variable.insert(0, fixed.pop())  # one exact check; trims only on tokenizer join effects
    for s in [v for v in variable if v["kind"] == "PROVENANCE"]:
        # Provenance too long to repeat whole is windowed like a body; every window
        # still carries its leading lines so no body window loses its provenance.
        fixed.append(provenance_header(s, fixed, count, limit))
    # Density is only a starting guess; every accepted slice is counted exactly.
    chars_per_token = len(full) / max(full_tokens - overhead, 1)
    fixed_tokens = count(render_citable(fixed)[0])
    guess = max(1, int((budget - fixed_tokens) * chars_per_token * 0.97))
    for s in variable:
        start = 0
        while start < len(s["text"]) or (start == 0 and not s["text"]):
            remaining = len(s["text"]) - start
            def fits(size):
                part = {**s, "text": s["text"][start:start + size]}
                return count(render_citable(fixed + [part])[0]) <= budget

            size = min(remaining, guess)
            lo, hi = 0, size
            if fits(size):
                lo = size
                while lo < remaining:
                    hi = min(remaining, max(lo + 1, lo * 2))
                    if not fits(hi):
                        break
                    lo = hi
            # Stop within 1% of the fitting slice, avoiding token-by-token search.
            # A tokenizer need not be monotone: only measured fits become `lo`.
            while hi - lo > max(1, lo // 100):
                mid = (lo + hi) // 2
                if fits(mid):
                    lo = mid
                else:
                    hi = mid
            if not remaining and not fits(0):
                raise ValueError("Context segments alone exceed window budget")
            if not lo and remaining:
                raise ValueError("No window fits; refusing to omit evidence")
            end = start + lo
            spans = {x["segment_id"]: [0, len(x["text"])] for x in fixed}
            spans[s["segment_id"]] = [start, end]
            yield fixed + [{**s, "text": s["text"][start:end]}], spans
            if end == len(s["text"]):
                break
            start = end - min(overlap, lo // 4)
