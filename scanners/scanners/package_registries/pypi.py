# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Fetch recent PyPI activity from the public RSS feeds.

Reads the keyless PyPI RSS feeds (recent updates and newest packages)
and writes one JSON Lines record per item whose publication date falls
inside the requested window. No API key exists or is needed for these
feeds. Fetched payloads are parsed as data only and never executed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

USER_AGENT = (
    "url-scanners/1.0 "
    "(+https://github.com/TransluceAI/incident-investigation-tooling)"
)
SOURCE = "pypi.org"
FEEDS = {
    "updates": "https://pypi.org/rss/updates.xml",
    "packages": "https://pypi.org/rss/packages.xml",
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
        "--feed",
        choices=["updates", "packages", "both"],
        default="both",
        help="RSS feed to read (default: %(default)s)",
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


def fetch_text(url: str, timeout: float) -> str:
    """GET a text document with retries and backoff.

    Args:
        url: URL to fetch.
        timeout: HTTP timeout in seconds.

    Returns:
        Response body decoded as UTF-8.

    Raises:
        DegradedError: On HTTP 403/429 (rate limit or block).
        urllib.error.URLError: On transport failure after retries.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml"}
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise DegradedError(
                    f"PyPI returned HTTP {exc.code}; degraded, not retrying"
                ) from exc
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def parse_pub_date(value: str) -> datetime | None:
    """Parse an RSS pubDate string to an aware UTC datetime.

    Args:
        value: RFC 822 date string from the feed.

    Returns:
        Aware datetime in UTC, or None if unparsable.
    """
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_feed(
    body: str,
    feed_name: str,
    fetched_at: str,
    window: str,
    provenance: str,
    start: datetime,
    now: datetime,
) -> list[dict]:
    """Parse one RSS feed body into in-window records.

    Args:
        body: Raw RSS XML text.
        feed_name: Feed key ("updates" or "packages").
        fetched_at: UTC ISO-8601 fetch timestamp.
        window: Actual window label, e.g. "24h".
        provenance: URL that was queried.
        start: Window start (aware UTC datetime).
        now: Fetch time (aware UTC datetime).

    Returns:
        List of JSON-serialisable records inside the window.
    """
    root = ET.fromstring(body)
    records: list[dict] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        summary = (item.findtext("description") or "").strip()
        published = parse_pub_date(item.findtext("pubDate") or "")
        if published is None or not (start <= published <= now):
            continue
        name, _, version = title.partition(" ")
        if feed_name == "packages":
            # Newest-packages titles read "<name> added to PyPI".
            name = title.split(" added to PyPI")[0].strip()
            version = ""
        records.append(
            {
                "source": SOURCE,
                "fetched_at": fetched_at,
                "window": window,
                "provenance": provenance,
                "feed": feed_name,
                "event_type": "new_package"
                if feed_name == "packages"
                else "update",
                "name": name,
                "version": version,
                "title": title,
                "link": link,
                "summary": summary,
                "published_at": published.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    """Run the PyPI scanner.

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
    feed_names = list(FEEDS) if args.feed == "both" else [args.feed]
    records: list[dict] = []
    degraded = False
    for feed_name in feed_names:
        url = FEEDS[feed_name]
        try:
            body = fetch_text(url, args.timeout)
        except DegradedError as exc:
            print(f"degraded: {exc}", file=sys.stderr)
            degraded = True
            continue
        except Exception as exc:  # Network/parse failure after retries.
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            records.extend(
                parse_feed(body, feed_name, fetched_at, window, url, start, now)
            )
        except ET.ParseError as exc:
            print(f"error: unparsable RSS from {url}: {exc}", file=sys.stderr)
            return 1
    out_dir = Path(args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pypi-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records to {out_path}")
    return 2 if degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
