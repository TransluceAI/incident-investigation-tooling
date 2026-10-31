# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""AlienVault OTX subscribed-pulses scanner.

Fetches pulses the OTX account is subscribed to. This endpoint is keyed:
live probe 2026-10-31 returned HTTP 403 ``{"detail": "Authentication
required"}`` without a key, and the OTX documentation requires the
``X-OTX-API-KEY`` header. Without ``OTX_API_KEY`` this scanner exits 2
(degraded) and emits no records; it never fabricates pulses.

Usage:
    uv run otx.py --out-dir ./out --window-hours 24 --limit 20
"""

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
BASE_URL = "https://otx.alienvault.com/api/v1/pulses/subscribed"


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in ``Z``.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_otx_time(value: str) -> datetime | None:
    """Parse an OTX timestamp into an aware datetime.

    Args:
        value: Timestamp string, e.g. ``2026-10-31T12:00:00.000000``.

    Returns:
        Aware datetime in UTC, or None if unparseable.
    """
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def fetch_json(
    url: str, headers: dict[str, str], timeout: float
) -> tuple[int, Any]:
    """GET a URL with retries and return the parsed JSON payload.

    Args:
        url: URL to fetch.
        headers: HTTP headers to send.
        timeout: Per-attempt timeout in seconds.

    Returns:
        Tuple of (HTTP status code, parsed JSON or None).

    Raises:
        urllib.error.URLError: If all attempts fail at the transport level.
    """
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(
                    response.read().decode("utf-8")
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code >= 500 and attempt < 2:
                # Transient server error: back off and retry.
                last_error = exc
                time.sleep(2**attempt)
                continue
            try:
                return exc.code, json.loads(body)
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


def build_records(
    payload: Any,
    cutoff: datetime,
    fetched_at: str,
    window_label: str,
    provenance: str,
) -> list[dict[str, Any]]:
    """Convert an OTX subscribed-pulses payload into JSONL records.

    Args:
        payload: Parsed JSON from the OTX API.
        cutoff: Only pulses modified at or after this time are kept.
        fetched_at: ISO-8601 fetch timestamp for every record.
        window_label: Window string stamped on every record.
        provenance: URL queried, stamped on every record.

    Returns:
        List of record dicts (possibly empty).
    """
    records: list[dict[str, Any]] = []
    pulses = payload.get("results", []) if isinstance(payload, dict) else []
    for pulse in pulses:
        if not isinstance(pulse, dict):
            continue
        modified = parse_otx_time(str(pulse.get("modified", "")))
        if modified is not None and modified < cutoff:
            continue
        records.append(
            {
                "source": "otx",
                "fetched_at": fetched_at,
                "window": window_label,
                "provenance": provenance,
                "pulse_id": pulse.get("id"),
                "pulse_name": pulse.get("name"),
                "author": pulse.get("author_name"),
                "modified": pulse.get("modified"),
                "created": pulse.get("created"),
                "tags": pulse.get("tags", []),
                "indicator_count": len(pulse.get("indicators", [])),
                "references": pulse.get("references", []),
            }
        )
    return records


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
    parser = argparse.ArgumentParser(description="Fetch OTX subscribed pulses.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (else SCANNER_OUT_DIR, else ./out).",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Only keep pulses modified within this many hours.",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Max pulses to request from OTX."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the OTX scanner.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        0 on success, 2 degraded (no key or API refused), 1 on error.
    """
    args = parse_args(argv)
    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    api_key = os.environ.get("OTX_API_KEY")
    if not api_key:
        print(
            "degraded: OTX_API_KEY is not set; subscribed pulses require "
            "an OTX API key (keyless probe returned HTTP 403). Set "
            "OTX_API_KEY to upgrade this scanner.",
            file=sys.stderr,
        )
        return 2
    fetched_at = utcnow_iso()
    cutoff = datetime.now(UTC) - timedelta(hours=args.window_hours)
    window_label = f"{args.window_hours:g}h"
    query = urllib.parse.urlencode({"limit": args.limit})
    url = f"{BASE_URL}?{query}"
    try:
        status, payload = fetch_json(
            url,
            {"User-Agent": USER_AGENT, "X-OTX-API-KEY": api_key},
            args.timeout,
        )
    except urllib.error.URLError as exc:
        print(f"error: OTX request failed: {exc}", file=sys.stderr)
        return 1
    if status in (401, 403, 429):
        print(
            f"degraded: OTX refused the request (HTTP {status}); "
            "check OTX_API_KEY and rate limits.",
            file=sys.stderr,
        )
        return 2
    if status != 200 or not isinstance(payload, dict):
        print(f"error: OTX returned unexpected HTTP {status}.", file=sys.stderr)
        return 1
    records = build_records(payload, cutoff, fetched_at, window_label, url)
    path = write_jsonl(records, out_dir, "otx")
    print(f"otx: wrote {len(records)} records to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
