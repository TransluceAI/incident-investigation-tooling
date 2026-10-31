# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Paste/temp-file host watch (presence/listing watch, not a feed).

Covers a curated default set of 22 public paste and temp-file hosts
(the ``HOSTS`` table below; the set is fixed in this file, not
loaded from any external file).

Honest constraint: most paste hosts publish NO recent-activity
listing. For each host this scanner probes candidate listing URLs live
(homepage first, then a host-appropriate archive path where one is a
common pattern). A host is reported as:

* ``listing`` -- a candidate URL responded 200 and its page contained
  paste-like links; those links are listed (capped at 20) and the
  responding URL is recorded. Only responding URLs are shipped.
* ``no_public_listing`` -- the host responded, but no candidate page
  contained paste-like links. This is the normal case for upload-form
  homepages, raw-only hosts, and file hosts.
* ``unreachable`` -- every probe for the host failed or returned an
  error status.

This is a presence/listing watch, not an activity feed, wherever feeds
do not exist. Activity is never fabricated: no timestamps are claimed
for listed links, because listing pages generally do not expose them
in a uniform way.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
DEFAULT_TIMEOUT = 30.0
DEFAULT_DELAY = 0.2
MAX_ITEMS = 20

# Host -> candidate listing URLs, tried in order. Only a candidate that
# responds 200 AND contains paste-like links is ever reported as a
# listing; the rest are runtime probes, not shipped claims.
HOSTS: dict[str, tuple[str, ...]] = {
    "tmpfiles.org": ("https://tmpfiles.org/",),
    "anna.fyi": ("https://anna.fyi/", "https://anna.fyi/lists"),
    "paste.linuxiarz.pl": (
        "https://paste.linuxiarz.pl/",
        "https://paste.linuxiarz.pl/archive",
    ),
    "pastebin.k4be.pl": (
        "https://pastebin.k4be.pl/",
        "https://pastebin.k4be.pl/archive",
    ),
    "pastebin.tarcseh.me": (
        "https://pastebin.tarcseh.me/",
        "https://pastebin.tarcseh.me/archive",
    ),
    "paste.rs": ("https://paste.rs/",),
    "nicepaste.com": (
        "https://nicepaste.com/",
        "https://nicepaste.com/archive",
    ),
    "paste.ubuntu.org.cn": ("https://paste.ubuntu.org.cn/",),
    "pastebox.to": (
        "https://pastebox.to/",
        "https://pastebox.to/archive",
    ),
    "infinitypaste.club": (
        "https://infinitypaste.club/",
        "https://infinitypaste.club/archive",
    ),
    "porkpad.com": ("https://porkpad.com/",),
    "pasteview.com": ("https://pasteview.com/",),
    "gritpost.com": ("https://gritpost.com/",),
    "gotit.pub": ("https://gotit.pub/",),
    "mcbin.dev": ("https://mcbin.dev/",),
    "bytebin.lucko.me": ("https://bytebin.lucko.me/",),
    "bytebin.rkslot.nl": ("https://bytebin.rkslot.nl/",),
    "paste.ee": ("https://paste.ee/",),
    "paste.ubuntu.com": ("https://paste.ubuntu.com/",),
    "rentry.co": ("https://rentry.co/",),
    "mdshare.org": ("https://mdshare.org/",),
    "friendpaste.com": ("https://friendpaste.com/",),
}

HREF_RE = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)
# First path segments that are navigation, not pastes (also covers
# their sub-paths, e.g. /lists/105 pagination on anna.fyi).
SKIP_FIRST_SEGMENTS = {
    "login",
    "register",
    "about",
    "faq",
    "privacy",
    "terms",
    "tos",
    "api",
    "contact",
    "help",
    "archive",
    "trending",
    "recent",
    "lists",
    "trends",
    "rss",
}
SKIP_EXTENSIONS = (
    ".css",
    ".js",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".map",
)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in "Z".
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(
    url: str, timeout: float, retries: int = 1
) -> tuple[int | None, bytes, str | None]:
    """GET a URL, retrying transport errors with backoff.

    HTTP status codes are the server's verdict (a 403 or 404 will not
    change on retry) and are returned immediately; only connection-
    level failures are retried.

    Args:
        url: URL to fetch.
        timeout: Per-request timeout in seconds.
        retries: Number of retries after the first attempt.

    Returns:
        Tuple of (HTTP status or None, body bytes, error string or
        None).
    """
    last_error: str | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read(), None
        except urllib.error.HTTPError as exc:
            return exc.code, b"", None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        if attempt < retries:
            time.sleep(1.0 * (attempt + 1))
    return None, b"", last_error


def extract_paste_links(page_url: str, body: bytes) -> list[str]:
    """Extract paste-like links from a page body.

    Args:
        page_url: URL of the page, used to resolve relative links.
        body: Raw page body.

    Returns:
        Up to MAX_ITEMS distinct same-host links whose paths look like
        paste IDs (not navigation, assets, or known non-paste pages).
        Empty when the page shows no such links.
    """
    text = body.decode("utf-8", errors="ignore")
    found: list[str] = []
    seen: set[str] = set()
    parsed_page = urllib.parse.urlparse(page_url)
    for raw in HREF_RE.findall(text):
        resolved = urllib.parse.urljoin(page_url, raw)
        parsed = urllib.parse.urlparse(resolved)
        if parsed.netloc != parsed_page.netloc:
            continue
        path = parsed.path.rstrip("/")
        if not path:
            continue
        if path.lower().endswith(SKIP_EXTENSIONS):
            continue
        segments = [s for s in path.split("/") if s]
        if not segments or len(segments[-1]) < 2:
            continue
        if segments[0].lower() in SKIP_FIRST_SEGMENTS:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        found.append(resolved)
        if len(found) >= MAX_ITEMS:
            break
    return found


def check_host(host: str, timeout: float, delay: float) -> dict[str, Any]:
    """Probe one host for a public recent listing.

    Args:
        host: Hostname to probe.
        timeout: Per-request timeout in seconds.
        delay: Politeness delay between probes.

    Returns:
        One record for the host with status ``listing``,
        ``no_public_listing``, or ``unreachable``.
    """
    probed: list[str] = []
    any_response = False
    last_error: str | None = None
    for candidate in HOSTS[host]:
        probed.append(candidate)
        status, body, error = fetch(candidate, timeout)
        if status is not None:
            any_response = True
        if status == 200 and body:
            items = extract_paste_links(candidate, body)
            if items:
                return {
                    "source": host,
                    "status": "listing",
                    "listing_url": candidate,
                    "items_found": len(items),
                    "items": items,
                    "error": None,
                    "fetched_at": utc_now_iso(),
                    "window": "current-state",
                    "provenance": probed,
                }
        elif error:
            last_error = error
        time.sleep(delay)
    if any_response:
        status_value = "no_public_listing"
    else:
        status_value = "unreachable"
    return {
        "source": host,
        "status": status_value,
        "listing_url": None,
        "items_found": 0,
        "items": [],
        "error": last_error,
        "fetched_at": utc_now_iso(),
        "window": "current-state",
        "provenance": probed,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Watch paste/temp-file hosts for a public recent "
        "listing (presence watch; most hosts publish none)."
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
        help="Accepted for CLI uniformity; listing checks are not "
        "time-filtered because most hosts publish no timestamps.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds (default 30).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Politeness delay in seconds between requests (default 0.2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Check at most this many hosts (0 = all). For testing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: probe hosts, write one JSONL record per host.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Process exit code: 0 ok, 2 degraded (every host unreachable),
        1 on unexpected error.
    """
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create out dir {out_dir}: {exc}", file=sys.stderr)
        return 1
    hosts = list(HOSTS)
    if args.limit > 0:
        hosts = hosts[: args.limit]
    records = [check_host(host, args.timeout, args.delay) for host in hosts]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"paste_watch-{stamp}.jsonl")
    try:
        with open(out_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"error: cannot write {out_path}: {exc}", file=sys.stderr)
        return 1
    counts = {s: 0 for s in ("listing", "no_public_listing", "unreachable")}
    for record in records:
        counts[record["status"]] += 1
    print(f"wrote {len(records)} records to {out_path} ({counts})")
    if records and all(r["status"] == "unreachable" for r in records):
        print(
            "degraded: every host was unreachable.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
