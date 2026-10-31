# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Fetch recent RubyGems activity from the public activity API.

Reads the keyless RubyGems activity endpoints (latest gems and
just-updated gems) and writes one JSON Lines record per gem version
whose creation time falls inside the requested window. No API key is
needed. Fetched payloads are parsed as data only and never executed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

USER_AGENT = (
    "url-scanners/1.0 "
    "(+https://github.com/TransluceAI/incident-investigation-tooling)"
)
SOURCE = "rubygems.org"
ENDPOINTS = {
    "latest": "https://rubygems.org/api/v1/activity/latest.json",
    "just_updated": "https://rubygems.org/api/v1/activity/just_updated.json",
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
        "--activity",
        choices=["latest", "just_updated", "both"],
        default="both",
        help="Activity endpoint to read (default: %(default)s)",
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


def fetch_json(url: str, timeout: float) -> list:
    """GET a JSON document with retries and backoff.

    Args:
        url: URL to fetch.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JSON list.

    Raises:
        DegradedError: On HTTP 403/429 (rate limit or block).
        urllib.error.URLError: On transport failure after retries.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise DegradedError(
                    f"RubyGems returned HTTP {exc.code}; degraded, not retrying"
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
    gem: dict, activity: str, fetched_at: str, window: str, provenance: str
) -> dict:
    """Build one output record from a RubyGems activity entry.

    Args:
        gem: One entry from the activity list.
        activity: Activity endpoint key ("latest" or "just_updated").
        fetched_at: UTC ISO-8601 fetch timestamp.
        window: Actual window label, e.g. "24h".
        provenance: URL that was queried.

    Returns:
        JSON-serialisable output record.
    """
    metadata = gem.get("metadata") or {}
    return {
        "source": SOURCE,
        "fetched_at": fetched_at,
        "window": window,
        "provenance": provenance,
        "activity": activity,
        "name": gem.get("name"),
        "version": gem.get("version"),
        "version_created_at": gem.get("version_created_at"),
        "authors": gem.get("authors"),
        "info": gem.get("info"),
        "licenses": gem.get("licenses"),
        "yanked": gem.get("yanked"),
        "sha": gem.get("sha"),
        "homepage_uri": metadata.get("homepage_uri"),
        "gem_url": f"https://rubygems.org/gems/{gem.get('name')}",
    }


def main(argv: list[str] | None = None) -> int:
    """Run the RubyGems scanner.

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
    activities = list(ENDPOINTS) if args.activity == "both" else [args.activity]
    records: list[dict] = []
    degraded = False
    for activity in activities:
        url = ENDPOINTS[activity]
        try:
            payload = fetch_json(url, args.timeout)
        except DegradedError as exc:
            print(f"degraded: {exc}", file=sys.stderr)
            degraded = True
            continue
        except Exception as exc:  # Network/parse failure after retries.
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for gem in payload:
            if not isinstance(gem, dict):
                continue
            created = parse_time(gem.get("version_created_at") or "")
            if created is None or not (start <= created <= now):
                continue
            records.append(build_record(gem, activity, fetched_at, window, url))
    out_dir = Path(args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rubygems-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records to {out_path}")
    return 2 if degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
