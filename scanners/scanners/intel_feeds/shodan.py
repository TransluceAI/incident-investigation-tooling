"""Shodan host-search scanner (keyed only).

Queries the Shodan REST API at ``https://api.shodan.io`` using the
``/shodan/host/search`` endpoint documented at developer.shodan.io.
There is no keyless path: without ``SHODAN_API_KEY`` this scanner
exits 2 (degraded) with a keyed-upgrade message. The key is sent only
as a query parameter to api.shodan.io and is never written to output
records or logs; the provenance URL recorded omits the key.

Default query (documented example, an agent-harness facet-style
query): ``http.component:mcp OR http.title:"agent board"``. Override
with ``--query``. Search results carry Shodan scan timestamps; records
are kept only when their timestamp falls inside the window. Matches
with no parseable timestamp are kept but stamped ``window:
"undated"`` so they are never misread as in-window findings.

Usage:
    SHODAN_API_KEY=... uv run shodan.py --out-dir ./out --query 'apache'
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
from datetime import UTC, datetime, timedelta
from typing import Any

USER_AGENT = (
    "url-scanners/1.0 "
    "(+https://github.com/TransluceAI/incident-investigation-tooling)"
)
API_BASE = "https://api.shodan.io"
DEFAULT_QUERY = 'http.component:mcp OR http.title:"agent board"'


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in ``Z``.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: Any) -> datetime | None:
    """Parse a Shodan timestamp into an aware UTC datetime.

    Args:
        value: Timestamp string such as ``2026-10-31T12:00:00.000000``.

    Returns:
        Aware datetime in UTC, or None if unparseable.
    """
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def fetch_json(url: str, timeout: float) -> tuple[int, Any]:
    """GET a URL with retries and return the parsed JSON payload.

    Args:
        url: URL to fetch (key included only for the live request).
        timeout: Per-attempt timeout in seconds.

    Returns:
        Tuple of (HTTP status code, parsed JSON or None).

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
    parser = argparse.ArgumentParser(description="Search Shodan hosts (keyed).")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (else SCANNER_OUT_DIR, else ./out).",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Window in hours for scan timestamps, when present.",
    )
    parser.add_argument(
        "--query", default=DEFAULT_QUERY, help="Shodan search query string."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Shodan scanner.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        0 on success, 2 degraded (no key or key rejected), 1 on error.
    """
    args = parse_args(argv)
    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    api_key = os.environ.get("SHODAN_API_KEY")
    if not api_key:
        print(
            "degraded: SHODAN_API_KEY is not set; Shodan is keyed-only "
            "and has no keyless search path. Get a free key at "
            "shodan.io, set SHODAN_API_KEY, and re-run.",
            file=sys.stderr,
        )
        return 2
    fetched_at = utcnow_iso()
    cutoff = datetime.now(UTC) - timedelta(hours=args.window_hours)
    params = urllib.parse.urlencode(
        {"key": api_key, "query": args.query, "minify": "true"}
    )
    url = f"{API_BASE}/shodan/host/search?{params}"
    # Provenance must never carry the key.
    safe_params = urllib.parse.urlencode({"query": args.query})
    provenance = f"{API_BASE}/shodan/host/search?{safe_params}"
    try:
        status, payload = fetch_json(url, args.timeout)
    except urllib.error.URLError as exc:
        print(f"error: Shodan request failed: {exc}", file=sys.stderr)
        return 1
    if status in (401, 403, 429):
        print(
            "degraded: Shodan rejected or rate-limited the request "
            f"(HTTP {status}); check SHODAN_API_KEY or retry later.",
            file=sys.stderr,
        )
        return 2
    if status != 200 or not isinstance(payload, dict):
        print(
            f"error: Shodan returned unexpected HTTP {status}.", file=sys.stderr
        )
        return 1
    records: list[dict[str, Any]] = []
    for match in payload.get("matches", []):
        if not isinstance(match, dict):
            continue
        scanned = parse_time(match.get("timestamp"))
        if scanned is not None and scanned < cutoff:
            continue
        # A match with no parseable timestamp was not window-filtered;
        # stamp it "undated" instead of claiming the window.
        match_window = (
            f"{args.window_hours:g}h" if scanned is not None else "undated"
        )
        records.append(
            {
                "source": "shodan",
                "fetched_at": fetched_at,
                "window": match_window,
                "provenance": provenance,
                "query": args.query,
                "ip": match.get("ip_str"),
                "port": match.get("port"),
                "org": match.get("org"),
                "hostnames": match.get("hostnames", []),
                "product": match.get("product"),
                "scanned_at": match.get("timestamp"),
                "total_results": payload.get("total"),
            }
        )
    path = write_jsonl(records, out_dir, "shodan")
    print(f"shodan: wrote {len(records)} records to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
