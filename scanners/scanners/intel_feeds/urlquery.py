# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Fetch the latest public reports from urlquery.net, keyless.

Verified live 2026-10-31: the public home page (``https://urlquery.net/``,
"Recent reports") and the search page
(``https://urlquery.net/search/reports?q=<query>``) are server-rendered
HTML tables with one row per report: a ``<time datetime="...">`` UTC
timestamp, a link to ``/report/<uuid>``, the submitted URL, and the
resolved IP. No API key is needed and no keyed JSON endpoint was
verified, so this scanner uses only the keyless HTML path. Rows are
filtered client-side to the requested window by their timestamp.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import re
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
SOURCE = "urlquery.net"
HOME_URL = "https://urlquery.net/"
SEARCH_URL = "https://urlquery.net/search/reports"

ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
TIME_RE = re.compile(r'<time datetime="([^"]+)"', re.IGNORECASE)
REPORT_RE = re.compile(
    r'<a[^>]*href="/report/([0-9a-f-]{36})"[^>]*>([^<]+)</a>', re.IGNORECASE
)
IP_RE = re.compile(
    r'<span class="mono ip-literal">([^<]+)</span>', re.IGNORECASE
)


class DegradedError(Exception):
    """Raised when urlquery.net rate-limits or blocks the request."""


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
        default=None,
        help="Search query; if omitted, latest reports "
        "from the home page are used",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max reports to emit (default: %(default)s)",
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
    """GET an HTML page with retries and backoff.

    Args:
        url: URL to fetch.
        timeout: HTTP timeout in seconds.

    Returns:
        Response body as text.

    Raises:
        DegradedError: On HTTP 403/429.
        urllib.error.URLError: On transport failure after retries.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT}, method="GET"
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise DegradedError(
                    f"urlquery.net returned HTTP {exc.code}; degraded, "
                    "not retrying"
                ) from exc
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def parse_rows(page_html: str) -> list[dict]:
    """Parse report rows out of a urlquery.net table page.

    Args:
        page_html: HTML of the home or search-reports page.

    Returns:
        List of dicts with report_id, timestamp, submitted_url, ip.
        Rows missing a timestamp or report link are skipped.
    """
    rows: list[dict] = []
    for match in ROW_RE.finditer(page_html):
        block = match.group(1)
        time_match = TIME_RE.search(block)
        report_match = REPORT_RE.search(block)
        if not time_match or not report_match:
            continue
        ip_match = IP_RE.search(block)
        rows.append(
            {
                "report_id": report_match.group(1),
                "timestamp": time_match.group(1),
                "submitted_url": html_lib.unescape(
                    report_match.group(2).strip()
                ),
                "ip": ip_match.group(1).strip() if ip_match else None,
            }
        )
    return rows


def parse_time(value: str) -> datetime | None:
    """Parse a urlquery.net timestamp to an aware datetime.

    Args:
        value: ISO-8601 timestamp string.

    Returns:
        Aware datetime in UTC, or None if unparsable.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    """Run the urlquery.net scanner.

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
    out_dir = Path(args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"urlquery-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    records: list[dict] = []
    try:
        if args.query is None:
            page_html = fetch_text(HOME_URL, args.timeout)
            pages = [(HOME_URL, parse_rows(page_html))]
        else:
            pages = []
            offset = 0
            fetched_rows = 0
            while fetched_rows < args.limit:
                params = (
                    {"q": args.query, "offset": str(offset)}
                    if offset
                    else {"q": args.query}
                )
                url = SEARCH_URL + "?" + urllib.parse.urlencode(params)
                page_html = fetch_text(url, args.timeout)
                rows = parse_rows(page_html)
                if not rows:
                    break
                pages.append((url, rows))
                fetched_rows += len(rows)
                offset += len(rows)
                if len(rows) < 24:  # Last page of search results.
                    break
        for provenance, rows in pages:
            for row in rows:
                seen = parse_time(row["timestamp"])
                if seen is not None and not (start <= seen <= now):
                    continue
                records.append(
                    {
                        "source": SOURCE,
                        "fetched_at": fetched_at,
                        "window": window,
                        "provenance": provenance,
                        "report_id": row["report_id"],
                        "report_url": f"https://urlquery.net/report/{row['report_id']}",
                        "timestamp": row["timestamp"],
                        "submitted_url": row["submitted_url"],
                        "ip": row["ip"],
                    }
                )
                if len(records) >= args.limit:
                    break
            if len(records) >= args.limit:
                break
    except DegradedError as exc:
        print(f"degraded: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Network failure after retries.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
