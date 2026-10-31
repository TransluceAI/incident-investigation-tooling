# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Check latest versions of a watchlist of Go modules via the proxy.

HONEST CONSTRAINT: proxy.golang.org has no global "latest modules"
feed, so this scanner cannot enumerate new modules on its own. It
checks only the modules the operator supplies with ``--modules`` or
``--modules-file`` and reports each module's latest version and its
timestamp. With an empty watchlist it exits degraded (2) and explains
why, rather than faking a global feed. No API key is needed.
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
SOURCE = "proxy.golang.org"
PROXY_URL = "https://proxy.golang.org"


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
        "--modules", default="", help="Comma-separated module paths to check"
    )
    parser.add_argument(
        "--modules-file",
        default=None,
        help="File with one module path per line",
    )
    parser.add_argument(
        "--only-in-window",
        action="store_true",
        help="Drop modules whose latest version predates the window",
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


def collect_modules(args: argparse.Namespace) -> list[str]:
    """Collect the watchlist from flags and/or file, deduplicated.

    Args:
        args: Parsed arguments.

    Returns:
        Ordered, deduplicated list of module paths.
    """
    modules: list[str] = []
    if args.modules:
        modules.extend(part.strip() for part in args.modules.split(","))
    if args.modules_file:
        text = Path(args.modules_file).read_text(encoding="utf-8")
        modules.extend(line.strip() for line in text.splitlines())
    seen: set[str] = set()
    ordered: list[str] = []
    for module in modules:
        if module and not module.startswith("#") and module not in seen:
            seen.add(module)
            ordered.append(module)
    return ordered


def escape_module(module: str) -> str:
    """Escape a module path for the Go proxy protocol.

    Args:
        module: Module path, e.g. ``github.com/Example/Mod``.

    Returns:
        Path with each uppercase letter replaced by ``!`` plus its
        lowercase form, then URL-quoted with ``/`` and ``!`` kept.
    """
    escaped = "".join(f"!{ch.lower()}" if ch.isupper() else ch for ch in module)
    return urllib.parse.quote(escaped, safe="/!")


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
        urllib.error.HTTPError: On HTTP 4xx (e.g. 404 for an unknown
            module) without retry — a client error is not transient —
            and on 5xx after retries.
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
                    f"Go proxy returned HTTP {exc.code}; degraded, not retrying"
                ) from exc
            if exc.code < 500:
                raise  # Client error (e.g. 404): not transient.
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
        value: Timestamp string from the proxy.

    Returns:
        Aware datetime in UTC, or None if unparsable.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    """Run the Go module proxy scanner.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Exit code: 0 ok, 2 degraded, 1 error.
    """
    args = parse_args(argv)
    try:
        modules = collect_modules(args)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"error: cannot read modules file: {exc}", file=sys.stderr)
        return 1
    if not modules:
        print(
            "degraded: no watchlist supplied and proxy.golang.org has "
            "no global latest feed, so there is nothing to check. "
            "Supply --modules (comma list) and/or --modules-file "
            "(one module path per line). This scanner never fakes a "
            "global feed.",
            file=sys.stderr,
        )
        return 2
    now = datetime.now(UTC)
    start = now - timedelta(hours=args.window_hours)
    window = f"{args.window_hours:g}h"
    fetched_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    records: list[dict] = []
    unavailable = 0
    degraded = False
    for module in modules:
        url = f"{PROXY_URL}/{escape_module(module)}/@latest"
        try:
            payload = fetch_json(url, args.timeout)
        except DegradedError as exc:
            print(f"degraded: {exc}", file=sys.stderr)
            degraded = True
            continue
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(f"not found: {module} (HTTP 404)", file=sys.stderr)
            else:
                print(
                    f"unavailable: {module} (HTTP {exc.code})",
                    file=sys.stderr,
                )
            unavailable += 1
            continue
        except Exception as exc:  # Network failure after retries.
            print(f"error: {exc}", file=sys.stderr)
            return 1
        version_time = parse_time(payload.get("Time") or "")
        in_window = version_time is not None and start <= version_time <= now
        if args.only_in_window and not in_window:
            continue
        origin = payload.get("Origin") or {}
        records.append(
            {
                "source": SOURCE,
                "fetched_at": fetched_at,
                "window": window,
                "provenance": url,
                "module": module,
                "latest_version": payload.get("Version"),
                "latest_version_time": payload.get("Time"),
                "in_window": in_window,
                "origin_vcs": origin.get("VCS"),
                "origin_url": origin.get("URL"),
                "origin_ref": origin.get("Ref"),
            }
        )
    out_dir = Path(args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"go-proxy-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(
        f"wrote {len(records)} records to {out_path} "
        f"({unavailable} modules not found or unavailable)"
    )
    if not records:
        return 2
    return 2 if (degraded or unavailable) else 0


if __name__ == "__main__":
    raise SystemExit(main())
