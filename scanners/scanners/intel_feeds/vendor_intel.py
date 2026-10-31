"""Vendor-intel index watch: OpenAI and Anthropic (keyless).

Fetches the index pages ``https://openai.com/index/``,
``https://openai.com/news/`` and ``https://www.anthropic.com/research``,
extracts post entries (link + nearby date where the page provides
one), and filters dated entries to the window.

REACHABILITY (verified live 2026-10-31): ``anthropic.com/research``
returns HTML and yields post entries; the two OpenAI pages sit
behind an intermittent Cloudflare bot wall — the same day produced
both HTTP 403 to this scanner's non-browser user agent and, later,
HTTP 200 with real post entries. When walled, those pages degrade
per-page while Anthropic still emits.

METHOD HONESTY: this is HTML parsing and is brittle by nature. Both
sites are JavaScript-heavy; dates come from page content where
present (``datetime`` attributes, ``<time>`` elements, or visible
``Month D, YYYY`` / ``YYYY-MM-DD`` text near a link). Entries with no
parseable date are emitted with ``date`` null and ``window``
``undated`` rather than being silently dropped or dated by guesswork.
Per-page failure degrades only that page; exit 2 if any page degraded.

Usage:
    uv run vendor_intel.py --out-dir ./out --window-hours 24
"""

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

import argparse
import html
import http.client
import json
import os
import re
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
PAGES = [
    ("openai.com", "https://openai.com/index/"),
    ("openai.com", "https://openai.com/news/"),
    ("anthropic.com", "https://www.anthropic.com/research"),
]
LINK_RE = re.compile(
    r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE
)
TAG_RE = re.compile(r"<[^>]+>")
DATE_RE = re.compile(
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2})"
)
DATETIME_ATTR_RE = re.compile(r'datetime="([^"]+)"')


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in ``Z``.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_date(text: str | None) -> datetime | None:
    """Parse a human or ISO date string into an aware UTC datetime.

    Args:
        text: Date string or None.

    Returns:
        Aware datetime in UTC, or None if unparseable.
    """
    if not text:
        return None
    value = text.strip()
    for candidate in (value, value.replace("Z", "+00:00")):
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            continue
    for fmt in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def clean_text(raw: str) -> str:
    """Strip HTML tags and unescape entities from link text.

    Args:
        raw: Raw inner HTML of a link.

    Returns:
        Cleaned plain text, whitespace-collapsed.
    """
    return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub(" ", raw))).strip()


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


def extract_entries(body: str, page_url: str) -> list[dict[str, Any]]:
    """Extract candidate post entries from an index page's HTML.

    Args:
        body: Page HTML.
        page_url: Page URL, used to resolve relative links.

    Returns:
        Deduplicated list of dicts with url, title, date (or None).
    """
    seen: dict[str, dict[str, Any]] = {}
    for match in LINK_RE.finditer(body):
        href, inner = match.group(1), match.group(2)
        if not href.startswith("/") and not href.startswith("http"):
            continue
        if any(
            skip in href
            for skip in (
                "#",
                "javascript:",
                "/static/",
                ".css",
                ".js",
                ".png",
                ".svg",
                ".woff",
            )
        ):
            continue
        title = clean_text(inner)
        if len(title) < 8 or len(title) > 200:
            continue
        # Drop navigation/section links: keep only post-shaped paths
        # (OpenAI /index/<slug>/, Anthropic /research/<slug>) or links
        # with a date nearby (a dated entry is a post by evidence).
        post_match = re.match(r"^/(index|research)/([^/]+)/?$", href)
        # A bare section index (e.g. /research/index/) is
        # navigation, not a post.
        post_path = (
            post_match
            if post_match and post_match.group(2) != "index"
            else None
        )
        date_nearby = DATE_RE.search(
            body[max(0, match.start() - 150) : match.end() + 150]
        )
        if post_path is None and date_nearby is None:
            continue
        # Date: datetime attribute inside the link, else a date string
        # in the 300 chars surrounding the link.
        date_text: str | None = None
        attr = DATETIME_ATTR_RE.search(inner)
        if attr:
            date_text = attr.group(1)
        else:
            window = body[max(0, match.start() - 150) : match.end() + 150]
            found = DATE_RE.search(window)
            if found:
                date_text = found.group(1)
        url = href
        # Resolve relative links against the page origin.
        if href.startswith("/"):
            origin = "/".join(page_url.split("/")[:3])
            url = origin + href
        if url in seen:
            continue
        seen[url] = {"url": url, "title": title, "date": date_text}
    return list(seen.values())


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
        description="Watch OpenAI/Anthropic index pages."
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
        help="Window in hours for dated entries.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the vendor-intel scanner.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        0 on success, 2 if any page degraded, 1 on error.
    """
    args = parse_args(argv)
    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    fetched_at = utcnow_iso()
    cutoff = datetime.now(UTC) - timedelta(hours=args.window_hours)
    window_label = f"{args.window_hours:g}h"
    records: list[dict[str, Any]] = []
    degraded: list[str] = []
    for vendor, page_url in PAGES:
        try:
            status, body = fetch_text(page_url, args.timeout)
        except urllib.error.URLError as exc:
            print(f"degraded: {page_url} failed: {exc}", file=sys.stderr)
            degraded.append(page_url)
            continue
        if status != 200:
            print(
                f"degraded: {page_url} returned HTTP {status}.", file=sys.stderr
            )
            degraded.append(page_url)
            continue
        for entry in extract_entries(body, page_url):
            when = parse_date(entry["date"])
            if when is not None and when < cutoff:
                continue
            records.append(
                {
                    "source": "vendor_intel",
                    "fetched_at": fetched_at,
                    "window": window_label if when is not None else "undated",
                    "provenance": page_url,
                    "vendor": vendor,
                    "title": entry["title"],
                    "url": entry["url"],
                    "date": entry["date"],
                }
            )
    path = write_jsonl(records, out_dir, "vendor_intel")
    print(
        f"vendor_intel: wrote {len(records)} records to {path}; "
        f"degraded pages: {degraded or 'none'}"
    )
    return 2 if degraded else 0


if __name__ == "__main__":
    sys.exit(main())
