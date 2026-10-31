"""Intel-blogs scanner: swarmcha.se and norforge.com (keyless).

Per-site method, verified live 2026-10-31:

* norforge.com publishes a real Atom feed, advertised on its homepage
  as ``<link rel="alternate" type="application/atom+xml"
  href="/blog/feed.xml">``. Entries are parsed with the standard
  library XML parser and filtered to the window by ``published``.
* swarmcha.se publishes NO feed: ``/atom.xml``, ``/feed``,
  ``/rss.xml``, ``/index.xml``, ``/posts.xml`` and ``/atom`` all
  return 404. Fallback is a homepage change-watch: fetch
  ``https://swarmcha.se/``, hash the body (SHA-256), and extract the
  dated post links (``<time datetime="...">`` + ``<a href>``) from
  the post list. Post records are filtered to the window.

Every run also emits one ``site_snapshot`` record per site (content
hash, latest post found, post count) so a quiet window is still an
auditable observation, not silence. Snapshot records stamp
``window`` as ``snapshot``; post records stamp the real window.

Usage:
    uv run intel_blogs.py --out-dir ./out --window-hours 24
"""

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

import argparse
import hashlib
import html
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from typing import Any

USER_AGENT = (
    "url-scanners/1.0 "
    "(+https://github.com/TransluceAI/incident-investigation-tooling)"
)
NORFORGE_FEED = "https://norforge.com/blog/feed.xml"
SWARMCHA_HOME = "https://swarmcha.se/"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
POST_RE = re.compile(
    r'<time datetime="([^"]+)">[^<]*</time>\s*<a href="([^"]+)">([^<]+)</a>'
)


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in ``Z``.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: Any) -> datetime | None:
    """Parse a date value into an aware UTC datetime.

    Args:
        value: Date string (ISO-8601 or YYYY-MM-DD) or None.

    Returns:
        Aware datetime in UTC, or None if unparseable.
    """
    if not value:
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
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
            return exc.code, exc.read().decode("utf-8", errors="replace")
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise urllib.error.URLError(str(last_error))


def atom_text(entry: ET.Element, tag: str) -> str:
    """Return the text of a child Atom element, or an empty string.

    Args:
        entry: Atom ``<entry>`` element.
        tag: Child tag name without namespace.

    Returns:
        Element text or "".
    """
    child = entry.find(f"{ATOM_NS}{tag}")
    return (child.text or "").strip() if child is not None else ""


def atom_link(entry: ET.Element) -> str:
    """Return the alternate link href of an Atom entry, or "".

    Args:
        entry: Atom ``<entry>`` element.

    Returns:
        Href string or "".
    """
    for link in entry.findall(f"{ATOM_NS}link"):
        if link.get("rel", "alternate") == "alternate":
            return link.get("href", "")
    return ""


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
        description="Watch swarmcha.se and norforge.com."
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
        help="Window in hours for dated posts.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the intel-blogs scanner.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        0 on success, 2 if any site degraded, 1 on error.
    """
    args = parse_args(argv)
    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    fetched_at = utcnow_iso()
    cutoff = datetime.now(UTC) - timedelta(hours=args.window_hours)
    window_label = f"{args.window_hours:g}h"
    records: list[dict[str, Any]] = []
    degraded: list[str] = []

    # norforge.com: Atom feed.
    try:
        status, body = fetch_text(NORFORGE_FEED, args.timeout)
    except urllib.error.URLError as exc:
        print(f"degraded: norforge feed failed: {exc}", file=sys.stderr)
        degraded.append("norforge.com")
        body, status = "", 0
    if status == 200:
        try:
            root = ET.fromstring(body)
            entries = root.findall(f"{ATOM_NS}entry")
        except ET.ParseError as exc:
            print(
                f"degraded: norforge feed did not parse: {exc}", file=sys.stderr
            )
            degraded.append("norforge.com")
            entries = []
        # Feeds list newest first, so the first entry is the latest post.
        if entries:
            latest_title = atom_text(entries[0], "title")
            latest_date = atom_text(entries[0], "published") or atom_text(
                entries[0], "updated"
            )
        else:
            latest_title, latest_date = None, None
        for entry in entries:
            title = atom_text(entry, "title")
            published = atom_text(entry, "published") or atom_text(
                entry, "updated"
            )
            when = parse_time(published)
            if when is not None and when >= cutoff:
                records.append(
                    {
                        "source": "intel_blogs",
                        "fetched_at": fetched_at,
                        "window": window_label,
                        "provenance": NORFORGE_FEED,
                        "kind": "post",
                        "site": "norforge.com",
                        "title": title,
                        "url": atom_link(entry),
                        "date": published,
                    }
                )
        records.append(
            {
                "source": "intel_blogs",
                "fetched_at": fetched_at,
                "window": "snapshot",
                "provenance": NORFORGE_FEED,
                "kind": "site_snapshot",
                "site": "norforge.com",
                "method": "atom_feed",
                "content_sha256": hashlib.sha256(
                    body.encode("utf-8")
                ).hexdigest(),
                "latest_post_title": latest_title,
                "latest_post_date": latest_date,
                "post_count": len(entries),
            }
        )
    elif status != 0:
        print(
            f"degraded: norforge feed returned HTTP {status}.", file=sys.stderr
        )
        degraded.append("norforge.com")

    # swarmcha.se: homepage change-watch (no feed exists).
    try:
        status, body = fetch_text(SWARMCHA_HOME, args.timeout)
    except urllib.error.URLError as exc:
        print(f"degraded: swarmcha homepage failed: {exc}", file=sys.stderr)
        degraded.append("swarmcha.se")
        body, status = "", 0
    if status == 200:
        posts = [
            (d, h, html.unescape(t.strip()))
            for d, h, t in POST_RE.findall(body)
        ]
        latest_title = posts[0][2] if posts else None
        latest_date = posts[0][0] if posts else None
        for date_text, href, title in posts:
            when = parse_time(date_text)
            if when is not None and when >= cutoff:
                records.append(
                    {
                        "source": "intel_blogs",
                        "fetched_at": fetched_at,
                        "window": window_label,
                        "provenance": SWARMCHA_HOME,
                        "kind": "post",
                        "site": "swarmcha.se",
                        "title": title,
                        "url": f"https://swarmcha.se{href}"
                        if href.startswith("/")
                        else href,
                        "date": date_text,
                    }
                )
        records.append(
            {
                "source": "intel_blogs",
                "fetched_at": fetched_at,
                "window": "snapshot",
                "provenance": SWARMCHA_HOME,
                "kind": "site_snapshot",
                "site": "swarmcha.se",
                "method": "homepage_change_watch",
                "content_sha256": hashlib.sha256(
                    body.encode("utf-8")
                ).hexdigest(),
                "latest_post_title": latest_title,
                "latest_post_date": latest_date,
                "post_count": len(posts),
            }
        )
    elif status != 0:
        print(
            f"degraded: swarmcha homepage returned HTTP {status}.",
            file=sys.stderr,
        )
        degraded.append("swarmcha.se")

    path = write_jsonl(records, out_dir, "intel_blogs")
    print(
        f"intel_blogs: wrote {len(records)} records to {path}; "
        f"degraded sites: {degraded or 'none'}"
    )
    return 2 if degraded else 0


if __name__ == "__main__":
    sys.exit(main())
