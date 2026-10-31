# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Fetch recent Hacker News stories and comments via the Algolia API.

Uses the keyless ``search_by_date`` endpoint with ``numericFilters`` on
``created_at_i`` bounding the requested window. An empty ``--query``
returns recent stories and comments; a query restricts to matches.
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
SOURCE = "hn-algolia"
BASE_URL = "https://hn.algolia.com/api/v1/search_by_date"


class DegradedError(Exception):
    """Raised when the Algolia API rate-limits or blocks the request."""


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
        default="",
        help="Search text (default: empty = recent items)",
    )
    parser.add_argument(
        "--tags",
        default="story,comment",
        help="Comma-separated tags (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max items to request (default: %(default)s)",
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


def fetch_json(url: str, timeout: float) -> dict:
    """GET a JSON document with retries and backoff.

    Args:
        url: URL to fetch.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JSON object.

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
                        "HN Algolia returned a non-JSON (block) page"
                    ) from exc
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise DegradedError(
                    f"HN Algolia returned HTTP {exc.code}; "
                    "degraded, not retrying"
                ) from exc
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def main(argv: list[str] | None = None) -> int:
    """Run the HN Algolia scanner.

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
    tags = (
        "("
        + ",".join(t.strip() for t in args.tags.split(",") if t.strip())
        + ")"
    )
    numeric = (
        f"created_at_i>{int(start.timestamp())},"
        f"created_at_i<={int(now.timestamp())}"
    )
    params = {
        "tags": tags,
        "numericFilters": numeric,
        "hitsPerPage": str(args.limit),
    }
    if args.query:
        params["query"] = args.query
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    try:
        payload = fetch_json(url, args.timeout)
    except DegradedError as exc:
        print(f"degraded: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Network/parse failure after retries.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hn_algolia-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    kept = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for hit in payload.get("hits", []):
            record = {
                "source": SOURCE,
                "fetched_at": fetched_at,
                "window": window,
                "provenance": url,
                "object_id": hit.get("objectID"),
                "type": hit.get("_tags", [None])[0]
                if hit.get("_tags")
                else None,
                "title": hit.get("title"),
                "url": hit.get("url"),
                "author": hit.get("author"),
                "created_at": hit.get("created_at"),
                "created_at_i": hit.get("created_at_i"),
                "points": hit.get("points"),
                "num_comments": hit.get("num_comments"),
                "story_id": hit.get("story_id"),
                "comment_text": hit.get("comment_text"),
            }
            handle.write(json.dumps(record) + "\n")
            kept += 1
    print(f"wrote {kept} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
