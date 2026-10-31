"""URLhaus recent-URLs scanner.

Fetches recently added malicious URLs from
``https://urlhaus-api.abuse.ch/v1/urls/recent/`` (GET; a
``/limit/<n>/`` variant exists). Live probe 2026-10-31: without a key
the endpoint returns HTTP 401 ``{"error": "Unauthorized"}``. This
matches abuse.ch's October 2026 change requiring authentication for
exports. If ``ABUSECH_API_KEY`` is set it is sent as the ``Auth-Key``
header; without it this scanner exits 2 (degraded) and emits no
records. The recent feed covers roughly the last 3 days; records are
further filtered to ``--window-hours`` when ``date_added`` parses.
Entries whose ``date_added`` does not parse are kept but stamped
``window: "undated"`` so they are never misread as in-window.

Usage:
    ABUSECH_API_KEY=... uv run urlhaus.py --out-dir ./out --limit 100
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
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any

USER_AGENT = (
    "url-scanners/1.0 "
    "(+https://github.com/TransluceAI/incident-investigation-tooling)"
)
API_BASE = "https://urlhaus-api.abuse.ch/v1"


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in ``Z``.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_urlhaus_time(value: str) -> datetime | None:
    """Parse a URLhaus ``date_added`` value into an aware datetime.

    Args:
        value: String like ``2026-10-31 12:00:00 UTC``.

    Returns:
        Aware datetime in UTC, or None if unparseable.
    """
    if not value:
        return None
    text = value.strip().replace(" UTC", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def fetch_json(
    url: str, headers: dict[str, str], timeout: float
) -> tuple[int, Any]:
    """GET a URL with retries and return the parsed JSON payload.

    Args:
        url: URL to fetch.
        headers: HTTP headers to send.
        timeout: Per-attempt timeout in seconds.

    Returns:
        Tuple of (HTTP status code, parsed JSON or None).

    Raises:
        urllib.error.URLError: If all attempts fail at the transport level.
    """
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(
                    response.read().decode("utf-8")
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                return exc.code, json.loads(body)
            except ValueError:
                return exc.code, None
        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            OSError,
            http.client.HTTPException,
        ) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise urllib.error.URLError(str(last_error))


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
    parser = argparse.ArgumentParser(description="Fetch URLhaus recent URLs.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (else SCANNER_OUT_DIR, else ./out).",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Only keep URLs added within this many hours.",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Max recent URLs to request."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the URLhaus scanner.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        0 on success, 2 degraded (no key / key rejected), 1 on error.
    """
    args = parse_args(argv)
    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    api_key = os.environ.get("ABUSECH_API_KEY")
    if not api_key:
        print(
            "degraded: ABUSECH_API_KEY is not set; URLhaus recent URLs "
            "require an abuse.ch Auth-Key (keyless probe returned HTTP "
            "401). Get a free key at auth.abuse.ch, set ABUSECH_API_KEY, "
            "and re-run.",
            file=sys.stderr,
        )
        return 2
    fetched_at = utcnow_iso()
    cutoff = datetime.now(UTC) - timedelta(hours=args.window_hours)
    url = f"{API_BASE}/urls/recent/limit/{args.limit}/"
    headers = {"User-Agent": USER_AGENT, "Auth-Key": api_key}
    try:
        status, payload = fetch_json(url, headers, args.timeout)
    except urllib.error.URLError as exc:
        print(f"error: URLhaus request failed: {exc}", file=sys.stderr)
        return 1
    if status in (401, 403, 429):
        print(
            "degraded: URLhaus rejected or rate-limited the request "
            f"(HTTP {status}); check ABUSECH_API_KEY or retry later.",
            file=sys.stderr,
        )
        return 2
    if status != 200 or not isinstance(payload, dict):
        print(
            f"error: URLhaus returned unexpected HTTP {status}.",
            file=sys.stderr,
        )
        return 1
    records: list[dict[str, Any]] = []
    for entry in payload.get("urls", []):
        if not isinstance(entry, dict):
            continue
        added = parse_urlhaus_time(str(entry.get("date_added", "")))
        if added is not None and added < cutoff:
            continue
        # An entry with no parseable date_added was not window-filtered;
        # stamp it "undated" instead of claiming the window.
        entry_window = (
            f"{args.window_hours:g}h" if added is not None else "undated"
        )
        records.append(
            {
                "source": "urlhaus",
                "fetched_at": fetched_at,
                "window": entry_window,
                "provenance": url,
                "urlhaus_id": entry.get("id"),
                "url": entry.get("url"),
                "threat": entry.get("threat"),
                "tags": entry.get("tags", []),
                "date_added": entry.get("date_added"),
                "reporter": entry.get("reporter"),
            }
        )
    path = write_jsonl(records, out_dir, "urlhaus")
    print(f"urlhaus: wrote {len(records)} records to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
