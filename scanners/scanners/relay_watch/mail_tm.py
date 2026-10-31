# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""mail.tm API watch (keyed, read-only, metadata only).

Reads message metadata from the operator's own mail.tm inbox via::

    GET https://api.mail.tm/messages?page=N

with the operator-supplied JWT in the ``Authorization: Bearer <jwt>``
header, taken from the ``MAIL_TM_TOKEN`` environment variable. The
endpoint is documented at https://docs.mail.tm/api/messages
(paginated with the ``page`` parameter, up to 30 messages per page,
Hydra ``hydra:member`` envelope) and confirmed live: the base URL
entrypoint returns 200 and lists ``/messages``, while unauthenticated
``GET /messages`` returns 401
``{"code": 401, "message": "JWT Token not found"}``.

The API does not document message ordering, so pages are read until
an empty page or ``--max-pages`` instead of stopping at the first
out-of-window message; the ``createdAt`` filter does the windowing.

Without a token this scanner exits degraded (2) with a clear keyed-
upgrade message. Creating accounts is FORBIDDEN: this scanner never
POSTs to ``/accounts`` or ``/token`` and never creates an inbox; it
only reads with a token the operator supplied.

Privacy: only message metadata is emitted (sender, subject, intro,
timestamps, seen flag). Message bodies are NOT fetched or stored, and
the token itself is never logged or emitted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

USER_AGENT = (
    "url-scanners/1.0 "
    "(+https://github.com/TransluceAI/incident-investigation-tooling)"
)
BASE_URL = "https://api.mail.tm"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_PAGES = 5


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in "Z".
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_created(created_at: str) -> datetime | None:
    """Parse a mail.tm createdAt timestamp.

    Args:
        created_at: ISO-8601 timestamp string.

    Returns:
        Aware datetime, or None when unparseable.
    """
    try:
        stamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp


def api_get(
    url: str, token: str, timeout: float, retries: int = 2
) -> tuple[int | None, Any, str | None]:
    """GET a mail.tm API URL with a Bearer JWT.

    Args:
        url: API URL to fetch.
        token: Operator-supplied mail.tm JWT (never logged/emitted).
        timeout: Per-request timeout in seconds.
        retries: Number of retries after the first attempt.

    Returns:
        Tuple of (HTTP status or None, parsed JSON or None, error
        string or None).
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    last_error: str | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(response.read()), None
        except urllib.error.HTTPError as exc:
            return exc.code, None, f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        except json.JSONDecodeError as exc:
            return None, None, f"invalid JSON response: {exc}"
        if attempt < retries:
            time.sleep(1.0 * (attempt + 1))
    return None, None, last_error


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Watch the operator's own mail.tm inbox metadata "
        "(keyed, read-only; no account creation)."
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (else SCANNER_OUT_DIR, else ./out).",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=24,
        help="Only emit messages with createdAt inside this lookback "
        "(default 24).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Maximum /messages pages to fetch (default 5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds (default 30).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: read message metadata, write JSONL.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Process exit code: 0 ok, 2 degraded (no token, or the keyed
        API call failed), 1 on unexpected error.
    """
    args = build_parser().parse_args(argv)
    token = os.environ.get("MAIL_TM_TOKEN")
    if not token:
        print(
            "degraded: no mail.tm token configured. Set MAIL_TM_TOKEN "
            "to an operator-supplied mail.tm JWT (Bearer token for "
            "your own inbox). Keyed upgrade: with a token this "
            "scanner reads message metadata from "
            "https://api.mail.tm/messages. Without a token it fetches "
            "nothing. It never creates an account: no POST to "
            "/accounts or /token, ever.",
            file=sys.stderr,
        )
        return 2

    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create out dir {out_dir}: {exc}", file=sys.stderr)
        return 1

    cutoff = datetime.now(UTC).timestamp() - args.window_hours * 3600
    records: list[dict[str, Any]] = []
    for page in range(1, args.max_pages + 1):
        url = f"{BASE_URL}/messages?page={page}"
        status, payload, error = api_get(url, token, args.timeout)
        if status is None or status >= 400 or not isinstance(payload, dict):
            print(
                f"degraded: mail.tm /messages failed (status={status}, "
                f"error={error}). If 401, the MAIL_TM_TOKEN JWT is "
                f"missing, expired, or invalid.",
                file=sys.stderr,
            )
            return 2
        members = payload.get("hydra:member", [])
        if not members:
            break
        for msg in members:
            if not isinstance(msg, dict):
                continue
            created = parse_created(str(msg.get("createdAt", "")))
            if created is not None and created.timestamp() < cutoff:
                # Out of window; the API does not document ordering,
                # so keep scanning rather than stopping early.
                continue
            sender = msg.get("from", {})
            records.append(
                {
                    "source": "mail.tm",
                    "message_id": msg.get("id"),
                    "from_address": sender.get("address")
                    if isinstance(sender, dict)
                    else None,
                    "subject": msg.get("subject"),
                    "intro": msg.get("intro"),
                    "created_at": msg.get("createdAt"),
                    "seen": msg.get("seen"),
                    "fetched_at": utc_now_iso(),
                    "window": f"{args.window_hours}h",
                    "provenance": url,
                }
            )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"mail_tm-{stamp}.jsonl")
    try:
        with open(out_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"error: cannot write {out_path}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(records)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
