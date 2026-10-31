"""ThreatFox IOC scanner (keyed).

Queries ``https://threatfox-api.abuse.ch/api/v1/`` with the documented
``get_iocs`` query. Live probe 2026-10-31: without a key the API
returns ``{"error": "Unauthorized"}`` (HTTP 401), and the ThreatFox
community-API documentation requires an ``Auth-Key`` header. Without
``ABUSECH_API_KEY`` this scanner exits 2 (degraded).

NOTE ON METHOD: ThreatFox offers no GET endpoint for this query; its
documented method is a POST carrying a JSON query body. This scanner
therefore issues a single read-only POST query (no submission, no
login, no session, no state change). That is a deliberate, documented
deviation from the project-wide GET-only preference, because the
only alternative would be a permanently-degraded stub that can never
fetch ThreatFox data. If strict GET-only is ever required, this
scanner must be retired rather than silently degraded.

The API accepts ``days`` 1-7 only (based on ``first_seen``, per the
community-API documentation). ``--window-hours`` is converted to
whole days (ceil, clamped 1-7) and the record window stamps the actual
days used, never a 24h claim when more days were fetched. Records are
further filtered by ``first_seen`` when it parses (the documented
format is ``YYYY-MM-DD HH:MM:SS UTC``).

Usage:
    ABUSECH_API_KEY=... uv run threatfox.py --out-dir ./out --window-hours 24
"""

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

import argparse
import http.client
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any

USER_AGENT = (
    "url-scanners/1.0 "
    "(+https://github.com/TransluceAI/incident-investigation-tooling)"
)
API_URL = "https://threatfox-api.abuse.ch/api/v1/"


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in ``Z``.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: Any) -> datetime | None:
    """Parse a ThreatFox timestamp into an aware UTC datetime.

    Args:
        value: Timestamp string in the documented ``first_seen`` /
            ``last_seen`` format ``YYYY-MM-DD HH:MM:SS UTC`` (see the
            ThreatFox community-API documentation), or another
            ISO-8601 form.

    Returns:
        Aware datetime in UTC, or None if unparseable.
    """
    if not value:
        return None
    # The API's documented format carries a literal " UTC" suffix.
    text = str(value).strip().replace(" UTC", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def post_json(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
) -> tuple[int, Any]:
    """POST a JSON body with retries and return the parsed response.

    Args:
        url: URL to query.
        body: JSON-serialisable query body.
        headers: HTTP headers to send.
        timeout: Per-attempt timeout in seconds.

    Returns:
        Tuple of (HTTP status code, parsed JSON or None).

    Raises:
        urllib.error.URLError: If all attempts fail at the transport level.
    """
    data = json.dumps(body).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url, data=data, headers=headers, method="POST"
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(
                    response.read().decode("utf-8")
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                return exc.code, json.loads(raw)
            except ValueError:
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
    parser = argparse.ArgumentParser(description="Fetch ThreatFox recent IOCs.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (else SCANNER_OUT_DIR, else ./out).",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Window in hours; converted to whole API days (1-7).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the ThreatFox scanner.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        0 on success, 2 degraded (no key / key rejected), 1 on error.
    """
    args = parse_args(argv)
    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    api_key = os.environ.get("ABUSECH_API_KEY")
    if not api_key:
        print(
            "degraded: ABUSECH_API_KEY is not set; ThreatFox requires "
            "an abuse.ch Auth-Key (keyless probe returned HTTP 401). "
            "Get a free key at auth.abuse.ch, set ABUSECH_API_KEY, "
            "and re-run.",
            file=sys.stderr,
        )
        return 2
    days = max(1, min(7, math.ceil(args.window_hours / 24.0)))
    fetched_at = utcnow_iso()
    cutoff = datetime.now(UTC) - timedelta(days=days)
    window_label = f"{days}d"
    headers = {
        "User-Agent": USER_AGENT,
        "Auth-Key": api_key,
        "Content-Type": "application/json",
    }
    try:
        status, payload = post_json(
            API_URL, {"query": "get_iocs", "days": days}, headers, args.timeout
        )
    except urllib.error.URLError as exc:
        print(f"error: ThreatFox request failed: {exc}", file=sys.stderr)
        return 1
    if status in (401, 403, 429):
        print(
            "degraded: ThreatFox rejected or rate-limited the request "
            f"(HTTP {status}); check ABUSECH_API_KEY or retry later.",
            file=sys.stderr,
        )
        return 2
    if status != 200 or not isinstance(payload, dict):
        print(
            f"error: ThreatFox returned unexpected HTTP {status}.",
            file=sys.stderr,
        )
        return 1
    if payload.get("query_status") not in ("ok", None):
        print(
            f"degraded: ThreatFox query_status={payload.get('query_status')}.",
            file=sys.stderr,
        )
        return 2
    records: list[dict[str, Any]] = []
    for entry in payload.get("data", []):
        if not isinstance(entry, dict):
            continue
        first_seen = parse_time(entry.get("first_seen"))
        if first_seen is not None and first_seen < cutoff:
            continue
        records.append(
            {
                "source": "threatfox",
                "fetched_at": fetched_at,
                "window": window_label,
                "provenance": API_URL,
                "ioc": entry.get("ioc"),
                "ioc_type": entry.get("ioc_type"),
                "threat_type": entry.get("threat_type"),
                "malware": entry.get("malware"),
                "first_seen": entry.get("first_seen"),
                "last_seen": entry.get("last_seen"),
                "confidence_level": entry.get("confidence_level"),
                "tags": entry.get("tags", []),
                "reporter": entry.get("reporter"),
            }
        )
    path = write_jsonl(records, out_dir, "threatfox")
    print(
        f"threatfox: wrote {len(records)} records to {path} "
        f"(window {window_label})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
