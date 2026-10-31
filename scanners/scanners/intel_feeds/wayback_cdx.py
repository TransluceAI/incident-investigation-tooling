# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Query the Wayback Machine CDX index for watched domains.

Uses the keyless CDX search API
(``https://web.archive.org/cdx/search/cdx``). Each domain is queried
as ``<domain>/*`` with ``from``/``to`` bounding the requested window,
so the emitted window is the window actually requested from the index.
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
SOURCE = "wayback-cdx"
CDX_URL = "https://web.archive.org/cdx/search/cdx"

DEFAULT_DOMAINS = [
    "swarmsay.com",
    "commons.dooza.ai",
    "robots.wiki",
    "agenthaven.org",
    "agentcolony.one",
    "msgboard.dev",
    "moltbook.com",
    "commonlog.ai",
]


class DegradedError(Exception):
    """Raised when the CDX API rate-limits or blocks the request."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        default=",".join(DEFAULT_DOMAINS),
        help="Comma-separated domains (default: curated default domains)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max captures per domain (default: %(default)s)",
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
    """GET a CDX JSON document with retries and backoff.

    Args:
        url: URL to fetch.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JSON list (first row is the header).

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
                try:
                    return json.loads(response.read().decode("utf-8"))
                except ValueError as exc:
                    raise DegradedError(
                        "CDX returned a non-JSON page "
                        "(e.g. Temporarily Offline)"
                    ) from exc
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise DegradedError(
                    f"CDX returned HTTP {exc.code}; degraded, not retrying"
                ) from exc
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def main(argv: list[str] | None = None) -> int:
    """Run the Wayback CDX scanner.

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
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    out_dir = Path(args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wayback_cdx-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    kept = 0
    degraded = False
    with out_path.open("w", encoding="utf-8") as handle:
        for domain in domains:
            params = {
                "url": f"{domain}/*",
                "from": start.strftime("%Y%m%d%H%M%S"),
                "to": now.strftime("%Y%m%d%H%M%S"),
                "output": "json",
                "limit": str(args.limit),
            }
            url = CDX_URL + "?" + urllib.parse.urlencode(params)
            try:
                rows = fetch_json(url, args.timeout)
            except DegradedError as exc:
                print(f"degraded: {exc} (domain {domain})", file=sys.stderr)
                degraded = True
                continue
            except Exception as exc:  # Network/parse failure after retries.
                print(f"error: {domain}: {exc}", file=sys.stderr)
                degraded = True
                continue
            if not rows:
                continue
            header, data = rows[0], rows[1:]
            for row in data:
                entry = dict(zip(header, row, strict=False))
                record = {
                    "source": SOURCE,
                    "fetched_at": fetched_at,
                    "window": window,
                    "provenance": url,
                    "domain": domain,
                    "timestamp": entry.get("timestamp"),
                    "original": entry.get("original"),
                    "statuscode": entry.get("statuscode"),
                    "mimetype": entry.get("mimetype"),
                    "digest": entry.get("digest"),
                    "wayback_url": (
                        f"https://web.archive.org/web/{entry.get('timestamp')}/"
                        f"{entry.get('original')}"
                        if entry.get("timestamp") and entry.get("original")
                        else None
                    ),
                }
                handle.write(json.dumps(record) + "\n")
                kept += 1
            time.sleep(1)  # Polite pacing between domains.
    print(f"wrote {kept} records to {out_path}")
    return 2 if degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
