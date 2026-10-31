"""Common Crawl index scanner (keyless).

Lists indexes from ``https://index.commoncrawl.org/collinfo.json``,
picks the latest, and queries its CDX API for a URL. Verified live
2026-10-31: collinfo returns ``CC-MAIN-2026-43`` (October 2026 Index)
first, and ``CC-MAIN-2026-43-index?url=example.com&output=json``
returns capture records.

HONEST WINDOW: Common Crawl indexes are periodic (roughly monthly),
not 24-hour feeds. Every record stamps the actual crawl window from
collinfo (``from``/``to``) plus the index name, and never claims 24h.
``--window-hours`` is accepted for CLI uniformity but does not filter
crawl captures, because a crawl index cannot answer a 24h question.

Usage:
    uv run commoncrawl.py --out-dir ./out --url example.com --limit 20
"""

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from typing import Any

USER_AGENT = (
    "url-scanners/1.0 "
    "(+https://github.com/TransluceAI/incident-investigation-tooling)"
)
COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in ``Z``.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_text(url: str, timeout: float) -> tuple[int, str]:
    """GET a URL with retries and return its body text.

    Args:
        url: URL to fetch.
        timeout: Per-attempt timeout in seconds.

    Returns:
        Tuple of (HTTP status code, body text).

    Raises:
        urllib.error.URLError: If all attempts fail at the transport level.
    """
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}, method="GET"
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read().decode(
                    "utf-8", errors="replace"
                )
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise urllib.error.URLError(str(last_error))


def format_capture_time(raw: str) -> str:
    """Format a CDX timestamp (YYYYMMDDHHMMSS) as ISO-8601.

    Args:
        raw: 14-digit CDX timestamp.

    Returns:
        ISO-8601 string, or the raw value if it does not parse.
    """
    try:
        parsed = datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return raw


def write_jsonl(records: list[dict[str, Any]], out_dir: str, stem: str) -> str:
    """Write records as JSON Lines and return the output path.

    Args:
        records: Records to write.
        out_dir: Directory to write into (created if missing).
        stem: Filename stem; a UTC timestamp suffix is appended.

    Returns:
        Path of the file written.
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"{stem}-{stamp}.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Query the latest Common Crawl index."
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (else SCANNER_OUT_DIR, else ./out).",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Accepted for CLI uniformity; does not filter crawl captures.",
    )
    parser.add_argument(
        "--url",
        default="example.com",
        help="URL/domain to look up in the index (documented example).",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Max capture records to emit."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Common Crawl scanner.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        0 on success, 2 degraded (index query unavailable), 1 on error.
    """
    args = parse_args(argv)
    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    fetched_at = utcnow_iso()
    try:
        status, body = fetch_text(COLLINFO_URL, args.timeout)
    except urllib.error.URLError as exc:
        print(
            f"error: Common Crawl collinfo request failed: {exc}",
            file=sys.stderr,
        )
        return 1
    if status != 200:
        print(
            f"degraded: collinfo returned HTTP {status} "
            "(the service is intermittently unavailable; retry later).",
            file=sys.stderr,
        )
        return 2
    try:
        indexes = json.loads(body)
    except ValueError:
        print("error: collinfo did not return JSON.", file=sys.stderr)
        return 1
    if not isinstance(indexes, list) or not indexes:
        print("error: collinfo returned no indexes.", file=sys.stderr)
        return 1
    latest = indexes[0]
    index_id = latest.get("id", "")
    cdx_api = latest.get(
        "cdx-api", f"https://index.commoncrawl.org/{index_id}-index"
    )
    crawl_window = f"{latest.get('from', '?')} to {latest.get('to', '?')}"
    window_label = f"latest crawl index {index_id} ({crawl_window})"
    query = urllib.parse.urlencode({"url": args.url, "output": "json"})
    query_url = f"{cdx_api}?{query}"
    try:
        status, body = fetch_text(query_url, args.timeout)
    except urllib.error.URLError as exc:
        print(f"error: Common Crawl index query failed: {exc}", file=sys.stderr)
        return 1
    if status != 200:
        print(
            f"degraded: Common Crawl index query returned HTTP {status} "
            "(the index API is intermittently slow; retry later).",
            file=sys.stderr,
        )
        return 2
    records: list[dict[str, Any]] = []
    for line in body.splitlines():
        if len(records) >= args.limit:
            break
        line = line.strip()
        if not line or line.startswith("<"):
            continue
        try:
            capture = json.loads(line)
        except ValueError:
            continue
        if not isinstance(capture, dict):
            continue
        records.append(
            {
                "source": "commoncrawl",
                "fetched_at": fetched_at,
                "window": window_label,
                "provenance": query_url,
                "crawl_index": index_id,
                "crawl_window": crawl_window,
                "queried_url": args.url,
                "capture_url": capture.get("url"),
                "capture_time": format_capture_time(
                    str(capture.get("timestamp", ""))
                ),
                "mime": capture.get("mime"),
                "status": capture.get("status"),
                "digest": capture.get("digest"),
                "warc_filename": capture.get("filename"),
            }
        )
    path = write_jsonl(records, out_dir, "commoncrawl")
    print(
        f"commoncrawl: wrote {len(records)} records to {path} ({window_label})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
