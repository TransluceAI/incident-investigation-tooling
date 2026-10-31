# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""ntfy.sh public-topic watch.

Polls operator-configured public ntfy.sh topics for cached messages
published since a Unix timestamp ``window_hours`` ago, using the
documented poll form::

    GET https://ntfy.sh/<topic>/json?poll=1&since=<unix timestamp>

``poll=1`` is required: without it the JSON endpoint is an open-ended
stream that never terminates. ``since`` accepts a Unix timestamp (also
a duration, a message ID, or ``all``); this scanner always uses a Unix
timestamp so the requested window is explicit.

No key is needed for public topics. Topics are operator-configured:
with no topics supplied the scanner exits degraded (2) and never
invents or ships a default topic. Private topics return 401/403 and
are recorded as per-topic errors, not crashes.

Actual-window honesty: ntfy.sh caches messages for a limited time
(about 12 hours on the default server), so a 24-hour request may
return less than 24 hours of messages. Each record's ``window`` is
``requested <N>h, cache-limited`` (never a bare hour claim), and
``window_requested`` plus a cache-limit ``window_note`` carry the
details; the scanner never claims coverage the server did not
return.
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
from datetime import UTC, datetime
from typing import Any

USER_AGENT = (
    "url-scanners/1.0 "
    "(+https://github.com/TransluceAI/incident-investigation-tooling)"
)
DEFAULT_SERVER = "https://ntfy.sh"
DEFAULT_TIMEOUT = 30.0


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in "Z".
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def unix_to_iso(unix_time: int) -> str:
    """Convert a Unix timestamp to an ISO-8601 UTC string.

    Args:
        unix_time: Seconds since the Unix epoch.

    Returns:
        ISO-8601 timestamp ending in "Z".
    """
    return datetime.fromtimestamp(unix_time, tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def fetch_topic(
    server: str, topic: str, since: int, timeout: float, retries: int = 2
) -> tuple[int | None, bytes, str | None]:
    """Poll one ntfy topic for cached messages.

    Args:
        server: ntfy server base URL, e.g. https://ntfy.sh.
        topic: Topic name (URL-encoded before use).
        since: Unix timestamp to request messages from.
        timeout: Per-request timeout in seconds.
        retries: Number of retries after the first attempt.

    Returns:
        Tuple of (HTTP status or None, body bytes, error string or None).
    """
    encoded = urllib.parse.quote(topic, safe="")
    url = f"{server.rstrip('/')}/{encoded}/json?poll=1&since={since}"
    last_error: str | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read(), None
        except urllib.error.HTTPError as exc:
            # 4xx (unknown/private topic) will not change on retry.
            return exc.code, b"", f"HTTP {exc.code}"
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
        except (TimeoutError, OSError) as exc:
            last_error = str(exc)
        if attempt < retries:
            time.sleep(1.0 * (attempt + 1))
    return None, b"", last_error


def parse_messages(
    body: bytes, topic: str, url: str, window: str
) -> list[dict[str, Any]]:
    """Parse an ntfy JSON stream body into records.

    Args:
        body: Raw response body, one JSON object per line.
        topic: Topic the messages came from.
        url: Provenance URL that was queried.
        window: Window stamp carried by every record (actual-coverage
            note, e.g. "requested 24h, cache-limited").

    Returns:
        One record per message event. Non-message lines (e.g. open
        events) are skipped. Message text is capped at 2000 chars and
        never executed.
    """
    records: list[dict[str, Any]] = []
    for line in body.decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("event") != "message":
            continue
        stamp = obj.get("time")
        records.append(
            {
                "source": f"ntfy.sh/{topic}",
                "topic": topic,
                "message_id": obj.get("id"),
                "message_time": unix_to_iso(int(stamp))
                if isinstance(stamp, int)
                else None,
                "message_time_unix": stamp,
                "title": obj.get("title"),
                "message": str(obj.get("message", ""))[:2000],
                "tags": obj.get("tags"),
                "fetched_at": utc_now_iso(),
                "window": window,
                "provenance": url,
            }
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Watch operator-configured public ntfy.sh topics "
        "for cached messages in the requested window."
    )
    parser.add_argument(
        "--topics",
        default=None,
        help="Comma-separated topic list (else NTFY_TOPICS env). "
        "Required: with no topics the scanner exits degraded (2).",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help="ntfy server base URL (default https://ntfy.sh).",
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
        help="Requested lookback in hours (default 24). The server "
        "cache may hold less; see README.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds (default 30).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: poll topics and write one JSONL record per message.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Process exit code: 0 ok, 2 degraded (no topics configured, or
        every topic failed), 1 on unexpected error.
    """
    args = build_parser().parse_args(argv)
    raw_topics = args.topics or os.environ.get("NTFY_TOPICS")
    if not raw_topics:
        print(
            "degraded: no ntfy topics configured. Pass --topics "
            "(comma-separated) or set NTFY_TOPICS. This scanner does "
            "not invent or ship a default topic; topics are "
            "operator-configured.",
            file=sys.stderr,
        )
        return 2
    topics = [t.strip() for t in raw_topics.split(",") if t.strip()]
    if not topics:
        print(
            "degraded: --topics was empty. Pass at least one "
            "operator-configured topic.",
            file=sys.stderr,
        )
        return 2

    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create out dir {out_dir}: {exc}", file=sys.stderr)
        return 1

    since = int(time.time()) - args.window_hours * 3600
    window_stamp = f"requested {args.window_hours}h, cache-limited"
    records: list[dict[str, Any]] = []
    failed = 0
    for topic in topics:
        url = (
            f"{args.server.rstrip('/')}/"
            f"{urllib.parse.quote(topic, safe='')}/json"
            f"?poll=1&since={since}"
        )
        status, body, error = fetch_topic(
            args.server, topic, since, args.timeout
        )
        if status is None or status >= 400:
            failed += 1
            print(
                f"topic {topic}: failed (status={status}, error={error})",
                file=sys.stderr,
            )
            continue
        topic_records = parse_messages(body, topic, url, window_stamp)
        for record in topic_records:
            record["window_requested"] = f"{args.window_hours}h"
            record["window_note"] = (
                "ntfy.sh caches messages for a limited time "
                "(about 12h on the default server); returned messages "
                "cover at most the cache, even when a longer window "
                "was requested."
            )
            records.append(record)
        print(f"topic {topic}: {len(topic_records)} message(s)")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"ntfy_watch-{stamp}.jsonl")
    try:
        with open(out_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"error: cannot write {out_path}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(records)} records to {out_path}")
    if failed == len(topics):
        print(
            "degraded: every configured topic failed; no messages "
            "were retrieved.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
