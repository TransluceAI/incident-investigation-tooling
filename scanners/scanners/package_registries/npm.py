# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Fetch recent npm registry changes from the replicate changes feed.

Reads the keyless CouchDB changes feed at replicate.npmjs.com
(newest first) and, for each changed package, reads its public
registry metadata to get a real modification timestamp, because the
changes feed itself carries no timestamps. Only packages whose
``time.modified`` falls inside the requested window are emitted.
Deleted packages and packages whose metadata cannot be read are
counted and skipped, not fabricated. No API key is needed.
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
SOURCE = "npmjs.com"
CHANGES_URL = "https://replicate.npmjs.com/_changes"
REGISTRY_URL = "https://registry.npmjs.org"


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
        "--limit",
        type=int,
        default=50,
        help="Changes-feed entries to inspect (default: %(default)s)",
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
        DegradedError: On HTTP 403/429 (rate limit or block).
        urllib.error.URLError: On transport failure after retries.
        urllib.error.HTTPError: On other HTTP failures after retries,
            except that callers may catch 404 for missing packages.
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
                    f"npm registry returned HTTP {exc.code}; "
                    "degraded, not retrying"
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
        value: Timestamp string from registry metadata.

    Returns:
        Aware datetime in UTC, or None if unparsable.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def registry_url_for(package: str) -> str:
    """Build the registry metadata URL for a package name.

    Args:
        package: Package name, possibly scoped (``@scope/name``).

    Returns:
        Metadata URL with the name path-escaped (slash kept for scopes
        is escaped too, as the registry expects ``@scope%2fname``).
    """
    return REGISTRY_URL + "/" + urllib.parse.quote(package, safe="")


def main(argv: list[str] | None = None) -> int:
    """Run the npm scanner.

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
    changes_url = (
        CHANGES_URL
        + "?"
        + urllib.parse.urlencode(
            {"descending": "true", "limit": str(args.limit)}
        )
    )
    try:
        feed = fetch_json(changes_url, args.timeout)
    except DegradedError as exc:
        print(f"degraded: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Network/parse failure after retries.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    records: list[dict] = []
    skipped = 0
    degraded = False
    for entry in feed.get("results", []):
        package = entry.get("id", "")
        if entry.get("deleted") or not package:
            skipped += 1
            continue
        meta_url = registry_url_for(package)
        try:
            meta = fetch_json(meta_url, args.timeout)
        except DegradedError as exc:
            print(f"degraded: {exc}", file=sys.stderr)
            degraded = True
            break
        except Exception:  # Deleted/unreadable package: skip, do not fake.
            skipped += 1
            continue
        times = meta.get("time") or {}
        modified = parse_time(times.get("modified") or "")
        if modified is None or not (start <= modified <= now):
            continue
        dist_tags = meta.get("dist-tags") or {}
        records.append(
            {
                "source": SOURCE,
                "fetched_at": fetched_at,
                "window": window,
                "provenance": changes_url,
                "metadata_provenance": meta_url,
                "seq": entry.get("seq"),
                "name": package,
                "latest_version": dist_tags.get("latest"),
                "modified_at": times.get("modified"),
                "created_at": times.get("created"),
                "description": meta.get("description"),
            }
        )
    out_dir = Path(args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"npm-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(
        f"wrote {len(records)} records to {out_path} "
        f"({skipped} changed entries skipped: deleted or unreadable)"
    )
    return 2 if degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
