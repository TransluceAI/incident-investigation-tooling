"""Inert decoded views of a URL: query parameters, nested URLs, programs, base64. Never executes."""
import base64
import re
from urllib.parse import parse_qsl, urlsplit

MAX_DEPTH = 3
MAX_BYTES = 10 * 1024 * 1024
URL_RE = re.compile(r"https?://[^\s'\"<>()\\]+")
B64_RE = re.compile(r"^[A-Za-z0-9+/=_-]{16,}$")
CODE_KEYS = {"function", "code", "script", "js", "eval", "payload", "cmd", "command", "exec"}
CODE_HINTS = ("=>", "function", "eval(", "fetch(", "<script", "select ", "union ", "../", "${", "{{", "<?php")


def looks_like_code(value: str) -> bool:
    low = value.lower()
    return len(value) >= 12 and any(h in low for h in CODE_HINTS)


def _b64(value: str):
    if not B64_RE.match(value) or len(value) % 4 == 1:
        return None
    alt = b"-_" if ("-" in value or "_" in value) else None
    try:
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=alt)
        text = raw.decode("utf-8")
    except Exception:
        return None
    if len(text) < 8 or not all(ch.isprintable() or ch in "\n\r\t" for ch in text):
        return None
    return text


def decode_url(url: str, max_depth: int = MAX_DEPTH, max_bytes: int = MAX_BYTES):
    views, gaps = [], []
    _walk(url, "URL", 0, views, gaps, max_depth, max_bytes, seen=set())
    return views, gaps


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
    for m in URL_RE.findall(parts.path):
        _walk(m, f"{source}.path.embedded", depth + 1, views, gaps, max_depth, max_bytes, seen)
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
        else:
            decoded = _b64(v)
            if decoded is not None:
                views.append({"kind": "DECODED_B64", "source": f"{source}.query.{k}:base64:{depth + 1}", "text": decoded})
