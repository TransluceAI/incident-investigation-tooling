# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Fetch recently created Hugging Face models and datasets.

Reads the keyless Hugging Face Hub list APIs, newest first, and writes
one JSON Lines record per model or dataset whose creation time falls
inside the requested window. An optional ``HF_TOKEN`` environment
variable raises the rate limit; the keyless path works without it and
no token is ever hardcoded. Fetched payloads are parsed as data only
and never executed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

USER_AGENT = (
    "url-scanners/1.0 "
    "(+https://github.com/TransluceAI/incident-investigation-tooling)"
)
SOURCE = "huggingface.co"
ENDPOINTS = {
    "model": "https://huggingface.co/api/models",
    "dataset": "https://huggingface.co/api/datasets",
}


class DegradedError(Exception):
    """Raised when the source is reachable but unusable (rate limit, block)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=["models", "datasets", "both"],
        default="both",
        help="Hub listing to read (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Entries to request per listing (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (else SCANNER_OUT_DIR, else ./out)",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24,
        help="Look-back window in hours (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="HTTP timeout in seconds (default: %(default)s)",
    )
    return parser.parse_args(argv)


def fetch_json(url: str, token: str | None, timeout: float) -> list:
    """GET a JSON document with retries and backoff.

    Args:
        url: URL to fetch.
        token: Optional Hugging Face token, sent as a Bearer header.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JSON list.

    Raises:
        DegradedError: On HTTP 401/403/429 (auth, rate limit, or block).
        urllib.error.URLError: On transport failure after retries.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 429):
                raise DegradedError(
                    f"Hugging Face returned HTTP {exc.code}; degraded, "
                    "not retrying (check HF_TOKEN if 401)"
                ) from exc
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def parse_time(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp to an aware UTC datetime.

    Args:
        value: Timestamp string from the API.

    Returns:
        Aware datetime in UTC, or None if unparsable.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def build_record(
    item: dict, kind: str, fetched_at: str, window: str, provenance: str
) -> dict:
    """Build one output record from a Hub listing entry.

    Args:
        item: One entry from the models or datasets list.
        kind: "model" or "dataset".
        fetched_at: UTC ISO-8601 fetch timestamp.
        window: Actual window label, e.g. "24h".
        provenance: URL that was queried.

    Returns:
        JSON-serialisable output record.
    """
    repo_id = item.get("id")
    path_prefix = "datasets/" if kind == "dataset" else ""
    return {
        "source": SOURCE,
        "fetched_at": fetched_at,
        "window": window,
        "provenance": provenance,
        "kind": kind,
        "repo_id": repo_id,
        "author": item.get("author"),
        "created_at": item.get("createdAt"),
        "last_modified": item.get("lastModified"),
        "likes": item.get("likes"),
        "downloads": item.get("downloads"),
        "private": item.get("private"),
        "gated": item.get("gated"),
        "tags": item.get("tags"),
        "url": f"https://huggingface.co/{path_prefix}{repo_id}",
    }


def main(argv: list[str] | None = None) -> int:
    """Run the Hugging Face scanner.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Exit code: 0 ok, 2 degraded, 1 error.
    """
    args = parse_args(argv)
    now = datetime.now(UTC)
    start = now - timedelta(hours=args.window_hours)
    window = f"{args.window_hours:g}h"
    fetched_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    token = os.environ.get("HF_TOKEN") or None
    kinds = (
        ["model", "dataset"]
        if args.kind == "both"
        else (["model"] if args.kind == "models" else ["dataset"])
    )
    records: list[dict] = []
    degraded = False
    for kind in kinds:
        params = {"sort": "createdAt", "limit": str(args.limit)}
        url = ENDPOINTS[kind] + "?" + urllib.parse.urlencode(params)
        try:
            payload = fetch_json(url, token, args.timeout)
        except DegradedError as exc:
            print(f"degraded: {exc}", file=sys.stderr)
            degraded = True
            continue
        except Exception as exc:  # Network/parse failure after retries.
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for item in payload:
            if not isinstance(item, dict):
                continue
            created = parse_time(item.get("createdAt") or "")
            if created is None or not (start <= created <= now):
                continue
            records.append(build_record(item, kind, fetched_at, window, url))
    out_dir = Path(args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"huggingface-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records to {out_path}")
    return 2 if degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
