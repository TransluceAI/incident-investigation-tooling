# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Search crt.sh certificate transparency logs for watched domains.

Queries the keyless crt.sh JSON endpoint for wildcard certificates
(``%.<domain>``) and keeps certificates whose ``not_before`` date falls
inside the requested window. crt.sh is an issuance search, not a
real-time feed: the window is a client-side filter on ``not_before``,
and completeness is not guaranteed when a domain has a very large
certificate history (results are capped by ``--limit`` per domain, most
recent first).
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
SOURCE = "crt.sh"

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
    """Raised when crt.sh rate-limits or blocks the request."""


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
        default=100,
        help="Max certs kept per domain, most recent first",
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
                        "crt.sh returned a non-JSON (error) page"
                    ) from exc
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise DegradedError(
                    f"crt.sh returned HTTP {exc.code}; degraded, not retrying"
                ) from exc
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def parse_cert_time(value: str) -> datetime | None:
    """Parse a crt.sh timestamp to an aware datetime.

    Args:
        value: Timestamp string from crt.sh.

    Returns:
        Aware datetime in UTC, or None if unparsable.
    """
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except (ValueError, TypeError):
            continue
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the crt.sh scanner.

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
    out_path = out_dir / f"crtsh-{now.strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    kept = 0
    degraded = False
    with out_path.open("w", encoding="utf-8") as handle:
        for domain in domains:
            params = {"q": f"%.{domain}", "output": "json"}
            url = "https://crt.sh/?" + urllib.parse.urlencode(params)
            try:
                certs = fetch_json(url, args.timeout)
            except DegradedError as exc:
                print(f"degraded: {exc} (domain {domain})", file=sys.stderr)
                degraded = True
                continue
            except Exception as exc:  # Network/parse failure after retries.
                print(f"error: {domain}: {exc}", file=sys.stderr)
                degraded = True
                continue
            certs.sort(
                key=lambda c: str(c.get("not_before") or ""), reverse=True
            )
            for cert in certs[: args.limit]:
                issued = parse_cert_time(cert.get("not_before", ""))
                if issued is not None and not (start <= issued <= now):
                    continue
                record = {
                    "source": SOURCE,
                    "fetched_at": fetched_at,
                    "window": window,
                    "provenance": url,
                    "domain": domain,
                    "cert_id": cert.get("id"),
                    "common_name": cert.get("common_name"),
                    "name_value": cert.get("name_value"),
                    "not_before": cert.get("not_before"),
                    "not_after": cert.get("not_after"),
                    "issuer_name": cert.get("issuer_name"),
                }
                handle.write(json.dumps(record) + "\n")
                kept += 1
            time.sleep(1)  # Polite pacing between domains.
    print(f"wrote {kept} records to {out_path}")
    return 2 if degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
