# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Fetch recent urlscan.io scan results for a query.

Reads the public urlscan.io search API (keyless) and writes one JSON
Lines record per scan whose task time falls inside the requested window.
An optional ``URLSCAN_API_KEY`` raises the rate limit and unlocks the
detail endpoint, but the keyless search path works without it.
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
BASE_URL = "https://urlscan.io/api/v1/search/"
SOURCE = "urlscan.io"


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
        "--query",
        default="*",
        help="urlscan.io search query (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum results to request (default: %(default)s)",
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


def fetch_json(url: str, api_key: str | None, timeout: float) -> dict:
    """GET a JSON document with retries and backoff.

    Args:
        url: URL to fetch.
        api_key: Optional urlscan.io API key, sent as a header.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JSON object.

    Raises:
        DegradedError: On HTTP 403/429 (rate limit or block).
        urllib.error.URLError: On transport failure after retries.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if api_key:
        headers["API-Key"] = api_key
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                try:
                    return json.loads(response.read().decode("utf-8"))
                except ValueError as exc:
                    raise DegradedError(
                        "urlscan.io returned a non-JSON (block) page"
                    ) from exc
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise DegradedError(
                    f"urlscan.io returned HTTP {exc.code}; "
                    "degraded, not retrying"
                ) from exc
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def parse_task_time(value: str) -> datetime | None:
    """Parse a urlscan task time string to an aware datetime.

    Args:
        value: ISO-8601 timestamp string from the API.

    Returns:
        Aware datetime in UTC, or None if unparsable.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def build_record(
    result: dict, fetched_at: str, window: str, provenance: str
) -> dict:
    """Build one output record from a urlscan search result.

    Args:
        result: One entry from the API ``results`` list.
        fetched_at: UTC ISO-8601 fetch timestamp.
        window: Actual window label, e.g. ``"24h"``.
        provenance: URL that was queried.

    Returns:
        JSON-serialisable output record.
    """
    task = result.get("task", {}) if isinstance(result, dict) else {}
    page = result.get("page", {}) if isinstance(result, dict) else {}
    return {
        "source": SOURCE,
        "fetched_at": fetched_at,
        "window": window,
        "provenance": provenance,
        "scan_id": result.get("_id"),
        "scan_url": result.get("result"),
        "task_time": task.get("time"),
        "task_url": task.get("url"),
        "page_url": page.get("url"),
        "page_domain": page.get("domain"),
        "page_ip": page.get("ip"),
        "page_country": page.get("country"),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the urlscan.io scanner.

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
    # Client-side window filter on task time: the search API accepts a
    # date filter in the query, but its exact qualifier behaviour varies,
    # so the window is enforced here and reported honestly.
    params = {"q": args.query, "size": str(args.limit)}
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    try:
        payload = fetch_json(
            url, os.environ.get("URLSCAN_API_KEY"), args.timeout
        )
    except DegradedError as exc:
        print(f"degraded: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Network/parse failure after retries.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"urlscan-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    kept = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for result in payload.get("results", []):
            task_time = parse_task_time(
                (result.get("task") or {}).get("time", "")
            )
            if task_time is not None and not (start <= task_time <= now):
                continue
            record = build_record(result, fetched_at, window, url)
            handle.write(json.dumps(record) + "\n")
            kept += 1
    print(f"wrote {kept} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
