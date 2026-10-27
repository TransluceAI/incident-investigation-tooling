"""A small synthetic data root covering every evidence shape (no real evidence)."""
import base64
import json
import zlib
from hashlib import sha256
from pathlib import Path

URLQUERY_REPORT = {
    "report_id": "r-1", "version": 1, "status": "done", "tags": [], "date": "2026-06-20T10:00:00Z",
    "url": {"schema": "https", "fqdn": "api.example", "addr": "https://api.example/run?function=x"},
    "submit": {"type": "public", "user_agent": "Mozilla/5.0"},
    "http": [{"request": {"method": "POST", "url": "https://stats.example/q",
                          "post_data": {"data": "{\"year\":2020}"}},
              "response": {"status": 200, "sha256": "f" * 64, "md5": "a" * 32}}],
    "javascript": {"script": [{"url": {"addr": "https://cdn.example/lib.js"}, "sha1": "b" * 40,
                               "tlshash": "T1" + "c" * 68, "data": ""}]},
}


def _put(root, source, rel, text):
    path = root / "resources" / source / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return f"resources/{source}/{rel}", sha256(text.encode()).hexdigest()


def build(root: Path) -> Path:
    root = Path(root)
    recs = {}

    def rec(source, **fields):
        recs.setdefault(source, []).append(fields)

    payload = base64.urlsafe_b64encode(zlib.compress(b'{"year":2020,"country":"PT"}')).decode()
    for source, kind, body in (
        ("urlquery.net", "urlquery_report", json.dumps(URLQUERY_REPORT)),
        ("urlscan.io", "urlscan_report", json.dumps({"task":{"url":"https://stats.example/"},"page":{"status":200}})),
        ("wiki.example", "wiki_page", "Public encyclopedia page. "+"café 日本語 "*3000),
        ("ntfy.sh", "retained_message", "nightly backup complete, zero errors"),
        ("paste.example", "raw_paste", "Public data query https://stats.example/"+payload),
        ("new.example", "new_record_kind", "Preserved unknown-source text."),
        ("wayback.example", "wayback_capture", "404 Not Found"),
    ):
        rel, digest = _put(root, source, "body.txt", body)
        rec(source, observation_id=source+":one", kind=kind, document_path=rel, text_sha256=digest,
            metadata={"original_url":"https://stats.example/public", "capture_timestamp":"20260601000000"})
    rel, digest = _put(root, "bad.example", "body.txt", "corrupted text")
    rec("bad.example", observation_id="bad:one", kind="raw_paste", document_path=rel, text_sha256="0"*64)
    for source, rows in recs.items():
        path = root / "resources" / source / "records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    return root
