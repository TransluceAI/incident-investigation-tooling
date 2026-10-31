# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Search GitHub issues and commits for recent activity.

Uses the public GitHub Search API, keyless by default (60 requests/hour
unauthenticated bucket). An optional ``GITHUB_TOKEN`` raises the limit.
On rate-limit exhaustion the scanner stops cleanly with exit code 2
and never loops. The default query is the word ``agent`` restricted to
items created in the window; that default is a scanner default, not a
finding.
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
SOURCE = "github-search"
ENDPOINTS = {
    "issues": "https://api.github.com/search/issues",
    "commits": "https://api.github.com/search/commits",
}


class RateLimitedError(Exception):
    """Raised when the GitHub rate limit is exhausted."""


class DegradedError(Exception):
    """Raised when GitHub blocks the request (403/429, non-rate-limit)."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query", default="agent", help="Search term (default: %(default)s)"
    )
    parser.add_argument(
        "--endpoint",
        choices=["issues", "commits", "both"],
        default="both",
        help="Which search API to query",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max items per endpoint (default: %(default)s)",
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


def fetch_json(url: str, token: str | None, timeout: float) -> dict:
    """GET a JSON document with retries and backoff.

    Args:
        url: URL to fetch.
        token: Optional GitHub token for the Authorization header.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JSON object.

    Raises:
        RateLimitedError: When the rate limit is exhausted.
        DegradedError: On other HTTP 403/429 responses.
        urllib.error.URLError: On transport failure after retries.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                try:
                    return json.loads(response.read().decode("utf-8"))
                except ValueError as exc:
                    raise DegradedError(
                        "GitHub returned a non-JSON (block) page"
                    ) from exc
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                if (
                    exc.code == 403
                    and exc.headers.get("X-RateLimit-Remaining") == "0"
                ):
                    raise RateLimitedError(
                        "GitHub rate limit exhausted"
                    ) from exc
                raise DegradedError(
                    f"GitHub returned HTTP {exc.code}; degraded, not retrying"
                ) from exc
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def parse_time(value: str | None) -> datetime | None:
    """Parse a GitHub API timestamp to an aware datetime.

    Args:
        value: ISO-8601 timestamp string, or None/empty.

    Returns:
        Aware datetime in UTC, or None if missing or unparsable.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    """Run the GitHub search scanner.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Exit code: 0 ok, 2 degraded/rate-limited, 1 error.
    """
    args = parse_args(argv)
    now = datetime.now(UTC)
    start = now - timedelta(hours=args.window_hours)
    window = f"{args.window_hours:g}h"
    fetched_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    date = start.strftime("%Y-%m-%d")
    endpoints = (
        ["issues", "commits"] if args.endpoint == "both" else [args.endpoint]
    )
    token = os.environ.get("GITHUB_TOKEN")
    out_dir = Path(args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"github_search-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    kept = 0
    try:
        with out_path.open("w", encoding="utf-8") as handle:
            for name in endpoints:
                qualifier = (
                    f"created:>{date}"
                    if name == "issues"
                    else f"committer-date:>{date}"
                )
                query = f"{args.query} {qualifier}"
                params = {
                    "q": query,
                    "per_page": str(args.limit),
                }
                if name == "issues":
                    params["sort"] = "created"
                    params["order"] = "desc"
                # Commits are deliberately NOT sorted by committer-date:
                # far-future (faked) commit dates dominate the top of
                # that ordering, so best match plus the client-side
                # window filter below is what surfaces real commits.
                url = ENDPOINTS[name] + "?" + urllib.parse.urlencode(params)
                payload = fetch_json(url, token, args.timeout)
                for item in payload.get("items", []):
                    if name == "issues":
                        stamp = item.get("created_at")
                    else:
                        commit = item.get("commit") or {}
                        stamp = (commit.get("committer") or {}).get("date")
                    # GitHub date qualifiers are day-granular, so the
                    # exact window is enforced client-side on the item's
                    # own timestamp. Items without a parsable timestamp
                    # are kept rather than dropped silently.
                    seen = parse_time(stamp)
                    if seen is not None and not (start <= seen <= now):
                        continue
                    record = {
                        "source": SOURCE,
                        "fetched_at": fetched_at,
                        "window": window,
                        "provenance": url,
                        "endpoint": name,
                        "id": item.get("id"),
                        "html_url": item.get("html_url"),
                        "title": item.get("title"),
                        "message": (item.get("commit") or {}).get("message"),
                        "created_at": item.get("created_at"),
                        "updated_at": item.get("updated_at"),
                        "repository": (item.get("repository") or {}).get(
                            "full_name"
                        ),
                        "user": (item.get("user") or {}).get("login"),
                    }
                    handle.write(json.dumps(record) + "\n")
                    kept += 1
    except (RateLimitedError, DegradedError) as exc:
        print(f"degraded: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Network/parse failure after retries.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {kept} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
