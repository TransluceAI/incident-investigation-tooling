"""Pointer-addressed scanner view preserving all nonempty scalar values.

No global key-name deny-list: a request payload can legitimately contain
fields called hash, _id, or sha256. Values retain RFC 6901 JSON pointers.
Request/response sections are ordered first without changing their contents.
"""
import json

PRIORITY = {
    "urlquery": ("url", "final", "submit", "settings", "status", "date", "tags", "detection",
                 "summary", "http", "javascript", "artifacts", "sensors", "files", "ip", "stats"),
    "urlscan": ("task", "page", "submitter", "verdicts", "lists", "data", "meta", "stats"),
}


def detect(doc) -> str | None:
    if not isinstance(doc, dict):
        return None
    if "report_id" in doc and ("http" in doc or "javascript" in doc):
        return "urlquery"
    if "task" in doc and ("page" in doc or "data" in doc):
        return "urlscan"
    return None


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _walk(value, pointer, out):
    if isinstance(value, dict):
        for k, v in value.items():
            _walk(v, f"{pointer}/{_escape(str(k))}", out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _walk(v, f"{pointer}/{i}", out)
    elif value is None or value == "" or value == [] or value == {}:
        return
    else:
        out.append((pointer, value if isinstance(value, str) else json.dumps(value)))


def view(doc) -> tuple[str, list]:
    """Return (text, lines) where lines are (pointer, rendered value) in output order."""
    flavor = detect(doc)
    order = PRIORITY.get(flavor, ())
    keys = [k for k in order if k in doc] + [k for k in doc if k not in order]
    lines = []
    for k in keys:
        _walk(doc[k], "/" + _escape(str(k)), lines)
    return "\n".join(f"{p} = {v}" for p, v in lines), lines

