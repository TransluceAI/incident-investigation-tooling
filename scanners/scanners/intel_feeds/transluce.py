# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Transluce public-findings scanner.

Probes a public Transluce findings-list URL for a JSON list of
findings. Verified live 2026-10-31: Transluce publishes its findings
as HTML at ``https://transluce.org/investigations``, and no public
JSON findings-list endpoint exists (``/investigations.json``,
``/api/findings``, ``/findings.json``, ``/rss.xml`` and ``/feed.xml``
all returned HTTP 404). The default list URL therefore points at the
public HTML investigations page; because it is not a JSON list, this
scanner exits 2 (degraded) and emits no records rather than
fabricating findings. No endpoint is invented: if a JSON list is later
served at the default URL or at a URL passed via ``--list-url``, the
parser below handles it: records are filtered by any date field
present (``created``, ``modified``, ``date``, ``published``,
``updated``), and if no usable dates exist the latest N are emitted
with the window stamped honestly as ``latest-N``.

Usage:
    uv run transluce.py --out-dir ./out --window-hours 24 --limit 50
"""

import argparse
import http.client
import json
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
LIST_URL = "https://transluce.org/investigations"
DATE_FIELDS = ("created", "modified", "date", "published", "updated")


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in ``Z``.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: Any) -> datetime | None:
    """Parse a date value into an aware UTC datetime.

    Args:
        value: A date string or epoch number.

    Returns:
        Aware datetime in UTC, or None if unparseable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (ValueError, OverflowError, OSError):
            return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            parsed = datetime.fromisoformat(candidate)
            break
        except ValueError:
            parsed = None
    if parsed is None:
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d %b %Y"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def fetch_text(url: str, timeout: float) -> tuple[int, str]:
    """GET a URL with retries and return its body text.

    Args:
        url: URL to fetch.
        timeout: Per-attempt timeout in seconds.

    Returns:
        Tuple of (HTTP status code, body text).

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
                return response.status, response.read().decode(
                    "utf-8", errors="replace"
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code >= 500 and attempt < 2:
                # Transient server error: back off and retry.
                last_error = exc
                time.sleep(2**attempt)
                continue
            return exc.code, body
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise urllib.error.URLError(str(last_error))


def extract_items(payload: Any) -> list[dict[str, Any]] | None:
    """Extract a list of finding dicts from a decoded payload.

    Args:
        payload: Decoded JSON from the findings list URL.

    Returns:
        List of finding dicts, or None if the payload has no list shape.
    """
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("findings", "results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return None


def build_records(
    items: list[dict[str, Any]],
    cutoff: datetime,
    limit: int,
    fetched_at: str,
    window_label: str,
    provenance: str,
) -> tuple[list[dict[str, Any]], str]:
    """Build records from findings, filtering by date when possible.

    Args:
        items: Finding dicts from the list endpoint.
        cutoff: Window start; findings dated before it are dropped.
        limit: Max records when no usable dates exist.
        fetched_at: ISO-8601 fetch timestamp for every record.
        window_label: Window label stamped on dated records.
        provenance: URL queried, stamped on every record.

    Returns:
        Tuple of (records, window label actually used).
    """
    dated: list[tuple[datetime, dict[str, Any]]] = []
    for item in items:
        for field in DATE_FIELDS:
            parsed = parse_time(item.get(field))
            if parsed is not None:
                dated.append((parsed, item))
                break
    if dated:
        records: list[dict[str, Any]] = []
        for when, item in dated:
            if when < cutoff:
                continue
            records.append(
                {
                    "source": "transluce",
                    "fetched_at": fetched_at,
                    "window": window_label,
                    "provenance": provenance,
                    "finding_id": item.get("id"),
                    "title": item.get("title") or item.get("name"),
                    "date": item.get("date")
                    or item.get("created")
                    or item.get("published"),
                    "url": item.get("url"),
                    "summary": item.get("summary") or item.get("description"),
                }
            )
        return records, window_label
    chosen = items[:limit]
    records = [
        {
            "source": "transluce",
            "fetched_at": fetched_at,
            "window": f"latest-{len(chosen)}",
            "provenance": provenance,
            "finding_id": item.get("id"),
            "title": item.get("title") or item.get("name"),
            "url": item.get("url"),
            "summary": item.get("summary") or item.get("description"),
        }
        for item in chosen
    ]
    return records, f"latest-{len(chosen)}"


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
        description="Fetch Transluce public findings."
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
        help="Window in hours for date-filtered findings.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max findings to emit when no usable dates exist.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--list-url",
        default=LIST_URL,
        help="Findings list URL to probe for a JSON list "
        "(default: Transluce public investigations page).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Transluce scanner.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        0 on success, 2 degraded (no usable public list), 1 on error.
    """
    args = parse_args(argv)
    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    fetched_at = utcnow_iso()
    cutoff = datetime.now(UTC) - timedelta(hours=args.window_hours)
    try:
        status, body = fetch_text(args.list_url, args.timeout)
    except urllib.error.URLError as exc:
        print(f"error: Transluce request failed: {exc}", file=sys.stderr)
        return 1
    if status != 200:
        print(
            f"degraded: Transluce findings list returned HTTP {status}; "
            "no verified public JSON findings-list endpoint exists. "
            "No records fabricated.",
            file=sys.stderr,
        )
        return 2
    try:
        payload = json.loads(body)
    except ValueError:
        print(
            "degraded: Transluce findings list did not return JSON; "
            "no verified public JSON findings-list endpoint exists. "
            "No records fabricated.",
            file=sys.stderr,
        )
        return 2
    items = extract_items(payload)
    if items is None:
        print(
            "degraded: Transluce findings list JSON has no findings list "
            "shape. No records fabricated.",
            file=sys.stderr,
        )
        return 2
    records, window_label = build_records(
        items,
        cutoff,
        args.limit,
        fetched_at,
        f"{args.window_hours:g}h",
        args.list_url,
    )
    path = write_jsonl(records, out_dir, "transluce")
    print(
        f"transluce: wrote {len(records)} records to {path} "
        f"(window {window_label})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
