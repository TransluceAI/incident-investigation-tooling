# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Search public Reddit posts via the keyless JSON search endpoint.

Uses ``https://www.reddit.com/search.json`` with ``sort=new`` and
filters results client-side to the requested window by ``created_utc``.
Reddit may serve HTTP 403/429 to unauthenticated or automated clients;
on either status the scanner exits 2 (degraded) with a clear message,
never crashes and never retries in a loop. Keyless only: no OAuth, no
login.
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
SOURCE = "reddit-public"
BASE_URL = "https://www.reddit.com/search.json"


class DegradedError(Exception):
    """Raised when Reddit blocks or rate-limits the request."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query", default="agent", help="Search text (default: %(default)s)"
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
    """GET a JSON document with one retry for transient failures only.

    Args:
        url: URL to fetch.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JSON object.

    Raises:
        DegradedError: On HTTP 403/429, immediately, without looping.
        urllib.error.URLError: On transport failure after the retry.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT}, method="GET"
    )
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                try:
                    return json.loads(response.read().decode("utf-8"))
                except ValueError as exc:
                    raise DegradedError(
                        "Reddit returned a non-JSON (block) page"
                    ) from exc
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise DegradedError(
                    f"Reddit returned HTTP {exc.code}; degraded, not retrying"
                ) from exc
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def main(argv: list[str] | None = None) -> int:
    """Run the public Reddit scanner.

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
    params = {"q": args.query, "sort": "new", "limit": str(args.limit)}
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
    out_path = out_dir / f"reddit_public-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    kept = 0
    with out_path.open("w", encoding="utf-8") as handle:
        children = (payload.get("data") or {}).get("children", [])
        for child in children:
            data = child.get("data", {}) if isinstance(child, dict) else {}
            created = data.get("created_utc")
            if created is not None:
                created_dt = datetime.fromtimestamp(created, tz=UTC)
                if not (start <= created_dt <= now):
                    continue
            record = {
                "source": SOURCE,
                "fetched_at": fetched_at,
                "window": window,
                "provenance": url,
                "post_id": data.get("id"),
                "title": data.get("title"),
                "permalink": data.get("permalink"),
                "post_url": (
                    f"https://www.reddit.com{data['permalink']}"
                    if data.get("permalink")
                    else None
                ),
                "subreddit": data.get("subreddit"),
                "author": data.get("author"),
                "created_utc": created,
                "score": data.get("score"),
                "num_comments": data.get("num_comments"),
            }
            handle.write(json.dumps(record) + "\n")
            kept += 1
    print(f"wrote {kept} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
