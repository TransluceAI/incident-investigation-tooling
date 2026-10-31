"""MediaWiki recent-changes scanner (keyless).

Queries ``https://<wiki>/w/api.php?action=query&list=recentchanges``
for each wiki in ``--wikis``. Verified live 2026-10-31:
``en.wikipedia.org`` and ``publictestwiki.com`` return recentchanges
JSON; ``collusion.wiki`` returns HTTP 403 (Cloudflare "Forbidden"),
so that wiki degrades individually while the others still emit.
Per-wiki failure degrades only that wiki; the run exits 2 if any
wiki degraded, 0 if all succeeded.

The API is asked for ``rcstart``=now, ``rcdir``=older and
``rcend``=window start, so the server filters to the window.

Usage:
    uv run mediawiki.py --out-dir ./out --window-hours 24
    uv run mediawiki.py --wikis en.wikipedia.org,publictestwiki.com
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
DEFAULT_WIKIS = "en.wikipedia.org,publictestwiki.com,collusion.wiki"


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in ``Z``.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_json(url: str, timeout: float) -> tuple[int, Any]:
    """GET a URL with retries and return the parsed JSON payload.

    Args:
        url: URL to fetch.
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
    parser = argparse.ArgumentParser(
        description="Fetch MediaWiki recent changes."
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (else SCANNER_OUT_DIR, else ./out).",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Window in hours for recent changes.",
    )
    parser.add_argument(
        "--wikis", default=DEFAULT_WIKIS, help="Comma-separated wiki hosts."
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Max recent changes per wiki."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the MediaWiki scanner.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        0 if all wikis succeeded, 2 if any wiki degraded, 1 on error.
    """
    args = parse_args(argv)
    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    now = datetime.now(UTC)
    fetched_at = utcnow_iso()
    start = now - timedelta(hours=args.window_hours)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    records: list[dict[str, Any]] = []
    degraded: list[str] = []
    for wiki in [w.strip() for w in args.wikis.split(",") if w.strip()]:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "recentchanges",
                # NOTE: "type" is not a valid rcprop token (the API
                # warns about it); every recentchange item carries its
                # "type" unconditionally, so it is not requested here.
                "rcprop": "title|ids|timestamp|user|comment",
                "rcstart": now.strftime(fmt),
                "rcend": start.strftime(fmt),
                "rcdir": "older",
                "rclimit": args.limit,
                "format": "json",
                "formatversion": "2",
            }
        )
        url = f"https://{wiki}/w/api.php?{params}"
        try:
            status, payload = fetch_json(url, args.timeout)
        except urllib.error.URLError as exc:
            print(
                f"degraded: wiki {wiki} request failed: {exc}", file=sys.stderr
            )
            degraded.append(wiki)
            continue
        if status != 200 or not isinstance(payload, dict):
            print(
                f"degraded: wiki {wiki} returned HTTP {status}.",
                file=sys.stderr,
            )
            degraded.append(wiki)
            continue
        changes = payload.get("query", {}).get("recentchanges", [])
        for change in changes:
            if not isinstance(change, dict):
                continue
            records.append(
                {
                    "source": "mediawiki",
                    "fetched_at": fetched_at,
                    "window": f"{args.window_hours:g}h",
                    "provenance": url,
                    "wiki": wiki,
                    "type": change.get("type"),
                    "title": change.get("title"),
                    "pageid": change.get("pageid"),
                    "revid": change.get("revid"),
                    "old_revid": change.get("old_revid"),
                    "user": change.get("user"),
                    "timestamp": change.get("timestamp"),
                    "comment": change.get("comment"),
                }
            )
    path = write_jsonl(records, out_dir, "mediawiki")
    print(
        f"mediawiki: wrote {len(records)} records to {path}; "
        f"degraded wikis: {degraded or 'none'}"
    )
    return 2 if degraded else 0


if __name__ == "__main__":
    sys.exit(main())
