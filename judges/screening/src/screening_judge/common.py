"""Hashing, UTC time and atomic file helpers shared by every module."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def request_sha256(body: dict) -> str:
    """Canonical request hash; must match the shared runner's request_sha256."""
    return sha256_text(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path, write) -> None:
    """write(stream) -> flush -> fsync -> rename on the same filesystem, then fsync the directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as out:
        write(out)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path, text: str) -> None:
    _atomic_write(path, lambda out: out.write(text))


def write_jsonl(path, rows) -> None:
    """Stream rows; the file is never held in memory as one string."""
    _atomic_write(path, lambda out: out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def iter_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path) -> list:
    return list(iter_jsonl(path))


def classifier_hash(config):
    """Settings that affect classification, excluding run nonce/counting diagnostics."""
    stable = {k: v for k, v in config.items() if k not in {"boundary_salt", "token_count_mode"}}
    return sha256_text(json.dumps(stable, sort_keys=True, ensure_ascii=False))
