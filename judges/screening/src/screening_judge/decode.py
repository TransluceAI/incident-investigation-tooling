"""Inert decoded views of a URL: query parameters, nested URLs, programs, base64, zlib/gzip.

Nothing here executes, fetches or interprets content. Every view is text derived
from the original URL so the model can read encoded material it would otherwise
misjudge. The Arquivo review found 224 events whose base64 path segments or
base64+zlib parameters were never decoded; inspected ones resolved to ordinary
query parameters that the judge had read as possible injection payloads.
"""
import base64
import re
import zlib
from urllib.parse import parse_qsl, unquote, urlsplit

MAX_DEPTH = 3
MAX_BYTES = 10 * 1024 * 1024
MAX_INFLATED = 1024 * 1024
URL_RE = re.compile(r"https?://[^\s'\"<>()\\]+")
B64_RE = re.compile(r"^[A-Za-z0-9+/=_-]{16,}$")
CODE_KEYS = {"function", "code", "script", "js", "eval", "payload", "cmd", "command", "exec"}
CODE_HINTS = ("=>", "function", "eval(", "fetch(", "<script", "select ", "union ", "../", "${", "{{", "<?php")


def looks_like_code(value: str) -> bool:
    low = value.lower()
    return len(value) >= 12 and any(h in low for h in CODE_HINTS)


def _printable(text: str) -> bool:
    return len(text) >= 8 and all(ch.isprintable() or ch in "\n\r\t" for ch in text)


def _inflate(raw: bytes):
    """Return (method, text) for zlib/gzip/raw-deflate bytes, size-capped; else None."""
    for method, wbits in (("zlib", 15), ("gzip", 31), ("deflate", -15)):
        try:
            d = zlib.decompressobj(wbits)
            out = d.decompress(raw, MAX_INFLATED + 1)
            if len(out) > MAX_INFLATED or d.unconsumed_tail:
                return method, None  # over cap: report as a gap, never truncate silently
            if not d.eof:
                return "incomplete_" + method, None
            if d.unused_data:
                return "trailing_data_" + method, None
            text = out.decode("utf-8")
        except Exception:
            continue
        if _printable(text):
            return method, text
    return None


def decode_b64(value: str):
    """Return [(label, text)] views of a base64 value, or [] when it is not decodable text.

    A view may be the decoded text itself, or its zlib/gzip/deflate inflation.
    (None, gap) marks an inflation that exceeded the cap.
    """
    value = unquote(value)
    if not B64_RE.match(value) or len(value) % 4 == 1:
        return []
    alt = b"-_" if ("-" in value or "_" in value) else None
    try:
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=alt)
    except Exception:
        return []
    try:
        text = raw.decode("utf-8")
        if _printable(text):
            return [("base64", text)]
    except UnicodeDecodeError:
        pass
    inflated = _inflate(raw)
    if inflated is None:
        return []
    method, text = inflated
    return [(f"base64+{method}", text)] if text is not None else [(None, method if method.startswith(("incomplete_", "trailing_data_")) else f"{method}_size_exceeded")]


def decode_url(url: str, max_depth: int = MAX_DEPTH, max_bytes: int = MAX_BYTES):
    views, gaps = [], []
    _walk(url, "URL", 0, views, gaps, max_depth, max_bytes, seen=set())
    return views, gaps


def _b64_views(value, source, depth, views, gaps, max_depth, max_bytes, seen):
    for label, text in decode_b64(value):
        if label is None:
            gaps.append({"source": source, "gap": text})
            continue
        views.append({"kind": "DECODED_B64", "source": f"{source}:{label}:{depth + 1}", "text": text})
        for m in URL_RE.findall(text):
            _walk(m, f"{source}.{label}.embedded", depth + 1, views, gaps, max_depth, max_bytes, seen)


def _walk(url, source, depth, views, gaps, max_depth, max_bytes, seen):
    if url in seen:
        return
    seen.add(url)
    if depth > max_depth:
        gaps.append({"source": source, "gap": "depth_exceeded"})
        return
    if len(url.encode("utf-8")) > max_bytes:
        gaps.append({"source": source, "gap": "size_exceeded"})
        return
    try:
        parts = urlsplit(url)
    except ValueError:
        gaps.append({"source": source, "gap": "malformed_url"})
        return
    params = parse_qsl(parts.query, keep_blank_values=True)
    if params:
        lines = "\n".join(f"{k}={v}" for k, v in params)
        views.append({"kind": "URL_DECODED" if depth == 0 else "NESTED_URL",
                      "source": f"{source}.query:percent-decode:{depth + 1}",
                      "text": lines if depth == 0 else f"{url}\n{lines}"})
    elif depth > 0:
        views.append({"kind": "NESTED_URL", "source": f"{source}:percent-decode:{depth}", "text": url})
    for m in URL_RE.findall(unquote(parts.path)):
        _walk(m, f"{source}.path.embedded", depth + 1, views, gaps, max_depth, max_bytes, seen)
    for i, segment in enumerate(s for s in parts.path.split("/") if s):
        _b64_views(segment, f"{source}.path.{i}", depth, views, gaps, max_depth, max_bytes, seen)
    for k, v in params:
        if len(v.encode("utf-8")) > max_bytes:
            gaps.append({"source": f"{source}.query.{k}", "gap": "size_exceeded"})
            continue
        if v.lower().startswith(("http://", "https://")):
            _walk(v, f"{source}.query.{k}", depth + 1, views, gaps, max_depth, max_bytes, seen)
        elif k.lower() in CODE_KEYS or looks_like_code(v):
            views.append({"kind": "PROGRAM", "source": f"{source}.query.{k}:percent-decode:{depth + 1}", "text": v})
            for m in URL_RE.findall(v):
                _walk(m, f"{source}.query.{k}.embedded", depth + 1, views, gaps, max_depth, max_bytes, seen)
        # Code keys may also carry compressed/base64 programs.
        _b64_views(v, f"{source}.query.{k}", depth, views, gaps, max_depth, max_bytes, seen)
