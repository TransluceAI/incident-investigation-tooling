# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""webhook.site watch (keyed path only).

Reads requests captured by the operator's own webhook.site tokens via
the documented API::

    GET https://webhook.site/token            (list account tokens)
    GET https://webhook.site/token/<id>/requests?sorting=newest

Authentication is the ``Api-Key`` header, taken from the
``WEBHOOKSITE_TOKEN`` environment variable. Per webhook.site docs, the
token list requires authentication, and tokens owned by an account
always require an API key; an unauthenticated probe of
``GET https://webhook.site/token`` returns 401.

Without a token this scanner exits degraded (2) with a clear keyed-
upgrade message. It never creates a token (creating one is a POST and
is forbidden by the read-only contract), never guesses token IDs, and
never fabricates request data.

Privacy: only request metadata is emitted (method, timestamp, URL,
body size). Request bodies and headers are NOT copied into output,
so captured secrets stay out of the JSONL.
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
BASE_URL = "https://webhook.site"
DEFAULT_TIMEOUT = 30.0


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in "Z".
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_window_cutoff(window_hours: int) -> str:
    """Return the ISO-8601 cutoff for the requested window.

    Args:
        window_hours: Lookback in hours.

    Returns:
        ISO-8601 UTC timestamp of (now minus window_hours).
    """
    cutoff = datetime.now(UTC).timestamp() - window_hours * 3600
    return datetime.fromtimestamp(cutoff, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def api_get(
    url: str, api_key: str, timeout: float, retries: int = 2
) -> tuple[int | None, Any, str | None]:
    """GET a webhook.site API URL with the Api-Key header.

    Args:
        url: API URL to fetch.
        api_key: webhook.site API key (never logged or emitted).
        timeout: Per-request timeout in seconds.
        retries: Number of retries after the first attempt.

    Returns:
        Tuple of (HTTP status or None, parsed JSON or None, error
        string or None).
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Api-Key": api_key,
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


def within_window(created_at: str, cutoff_iso: str) -> bool:
    """Check whether an ISO-8601 timestamp is at or after a cutoff.

    webhook.site returns ``created_at`` as a naive UTC timestamp
    (e.g. ``2019-10-03 19:06:35``, per the provider docs); naive
    stamps are read as UTC so the comparison never mixes offset-naive
    and offset-aware datetimes.

    Args:
        created_at: Timestamp from the API (ISO-8601 with offset, or
            naive UTC in ``yyyy-MM-dd HH:mm:ss`` form).
        cutoff_iso: Cutoff timestamp in "Z" format.

    Returns:
        True when created_at parses and is >= cutoff; False otherwise
        (unparseable timestamps are excluded, never assumed
        in-window).
    """
    try:
        stamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        cutoff = datetime.fromisoformat(cutoff_iso.replace("Z", "+00:00"))
    except ValueError:
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp >= cutoff


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Watch the operator's own webhook.site tokens for "
        "captured requests (keyed; metadata only)."
    )
    parser.add_argument(
        "--token-ids",
        default=None,
        help="Comma-separated webhook.site token UUIDs to read "
        "(else WEBHOOKSITE_TOKEN_IDS env). Without this, the token "
        "list for the account is listed instead.",
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
        help="Only emit requests with created_at inside this lookback "
        "(default 24).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds (default 30).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: list tokens or read token requests, write JSONL.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Process exit code: 0 ok, 2 degraded (no token configured, or
        the keyed API call failed), 1 on unexpected error.
    """
    args = build_parser().parse_args(argv)
    api_key = os.environ.get("WEBHOOKSITE_TOKEN")
    if not api_key:
        print(
            "degraded: no webhook.site API key configured. Set "
            "WEBHOOKSITE_TOKEN to your webhook.site API key (account "
            "settings) and optionally WEBHOOKSITE_TOKEN_IDS or "
            "--token-ids for specific token UUIDs. Keyed upgrade: "
            "with a key this scanner lists your account tokens or "
            "reads requests for the given tokens. Without a key it "
            "fetches nothing and fabricates nothing.",
            file=sys.stderr,
        )
        return 2

    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create out dir {out_dir}: {exc}", file=sys.stderr)
        return 1

    raw_ids = args.token_ids or os.environ.get("WEBHOOKSITE_TOKEN_IDS")
    token_ids = [t.strip() for t in raw_ids.split(",")] if raw_ids else []
    token_ids = [t for t in token_ids if t]
    cutoff = parse_window_cutoff(args.window_hours)
    records: list[dict[str, Any]] = []

    if not token_ids:
        status, payload, error = api_get(
            f"{BASE_URL}/token", api_key, args.timeout
        )
        if status is None or status >= 400 or not isinstance(payload, dict):
            print(
                f"degraded: webhook.site token list failed "
                f"(status={status}, error={error}). Check that "
                f"WEBHOOKSITE_TOKEN is a valid API key.",
                file=sys.stderr,
            )
            return 2
        for token in payload.get("data", []):
            if not isinstance(token, dict):
                continue
            records.append(
                {
                    "source": "webhook.site",
                    "record_type": "token",
                    "token_uuid": token.get("uuid"),
                    "alias": token.get("alias"),
                    "created_at": token.get("created_at"),
                    "fetched_at": utc_now_iso(),
                    # Token listing, not a time feed: the list call is
                    # never window-filtered, so no hour window is claimed.
                    "window": "current-state",
                    "provenance": f"{BASE_URL}/token",
                }
            )
    else:
        failed = 0
        for token_id in token_ids:
            url = (
                f"{BASE_URL}/token/{token_id}/requests"
                f"?sorting=newest&per_page=50"
            )
            status, payload, error = api_get(url, api_key, args.timeout)
            if status is None or status >= 400 or not isinstance(payload, dict):
                failed += 1
                print(
                    f"token {token_id}: failed (status={status}, "
                    f"error={error})",
                    file=sys.stderr,
                )
                continue
            for req in payload.get("data", []):
                if not isinstance(req, dict):
                    continue
                created = str(req.get("created_at", ""))
                if not within_window(created, cutoff):
                    continue
                # The API reports body size in bytes ("size"); fall
                # back to the content string length when absent.
                size = req.get("size")
                if not isinstance(size, int):
                    size = len(str(req.get("content") or ""))
                records.append(
                    {
                        "source": "webhook.site",
                        "record_type": "request",
                        "token_uuid": token_id,
                        "request_uuid": req.get("uuid"),
                        "method": req.get("method"),
                        "request_url": req.get("url"),
                        "created_at": created,
                        "content_size": size,
                        "fetched_at": utc_now_iso(),
                        "window": f"{args.window_hours}h",
                        "provenance": url,
                    }
                )
        if failed == len(token_ids):
            print(
                "degraded: every token request listing failed; check "
                "WEBHOOKSITE_TOKEN and the token UUIDs.",
                file=sys.stderr,
            )
            return 2

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"webhook_watch-{stamp}.jsonl")
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
