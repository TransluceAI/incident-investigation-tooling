# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Agent-board endpoint watch.

Polls a curated default list of agent-board surfaces for their
agent-facing endpoint files (``/llms.txt``, ``/skill.md``,
``/mcp.json``) and the site root, and records the current state of
each endpoint: HTTP status, body size, and SHA-256 of the body.

This is an endpoint-state watch, not an activity feed. ``llms.txt`` has
no timestamps, so "latest activity" here means the current endpoint
state at fetch time, not a 24-hour feed. See ``agent_boards.README.md``.

Read-only GET only. No key is needed. Fetched payloads are hashed and
scanned for date strings only; they are never executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
DEFAULT_TIMEOUT = 30.0
DEFAULT_DELAY = 0.2

# Curated default targets: the 39 established (BASELINED) agent-board
# surfaces plus 6 CANDIDATE surfaces verified live 2026-10-31.
TARGETS: tuple[tuple[str, str], ...] = (
    ("swarmsay.com", "BASELINED"),
    ("commons.dooza.ai", "BASELINED"),
    ("robots.wiki", "BASELINED"),
    ("agenthaven.org", "BASELINED"),
    ("agentcolony.one", "BASELINED"),
    ("agent-message-board.kyle-hatcher.workers.dev", "BASELINED"),
    ("smokesignal-arianaram.pythonanywhere.com", "BASELINED"),
    ("msgboard.dev", "BASELINED"),
    ("agentsboard.org", "BASELINED"),
    ("agents-agents-agents.com", "BASELINED"),
    ("bboard.ai", "BASELINED"),
    ("agentgateway.pythonanywhere.com", "BASELINED"),
    ("tantive.space", "BASELINED"),
    ("swarmmemo.com", "BASELINED"),
    ("messageboardforaiagents.com", "BASELINED"),
    ("swarm.termina.digital", "BASELINED"),
    ("thecolony.ai", "BASELINED"),
    ("aiforum.grok.me", "BASELINED"),
    ("moltbook.com", "BASELINED"),
    ("commonlog.ai", "BASELINED"),
    ("jotspot.io", "BASELINED"),
    ("nervesocket.com", "BASELINED"),
    ("public-board.com", "BASELINED"),
    ("openagentforum.com", "BASELINED"),
    ("1f916.ai", "BASELINED"),
    ("agentchan.org", "BASELINED"),
    ("agents-relay.com", "BASELINED"),
    ("universalagentforum.com", "BASELINED"),
    ("rentahuman.ai", "BASELINED"),
    ("bookofbots.com", "BASELINED"),
    ("openwitness.net", "BASELINED"),
    ("openagentchat.net", "BASELINED"),
    ("agent-community.com", "BASELINED"),
    ("agent-board.juleskreuer.eu", "BASELINED"),
    ("botbook.space", "BASELINED"),
    ("signpost.public-agents.ai", "BASELINED"),
    ("messages.directory", "BASELINED"),
    ("pursekeeper.dev", "BASELINED"),
    ("hermes-agent.nousresearch.com", "BASELINED"),
    ("swarmbar.com", "CANDIDATE"),
    ("agentboard.ai", "CANDIDATE"),
    ("agentdm.ai", "CANDIDATE"),
    ("anansi-haven.anansidata.workers.dev", "CANDIDATE"),
    ("fallback.chat", "CANDIDATE"),
    ("agentbadge.xyz", "CANDIDATE"),
)

ENDPOINTS: tuple[str, ...] = ("/llms.txt", "/skill.md", "/mcp.json", "/")

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MONTH_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December) \d{1,2}, \d{4}\b"
)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 timestamp ending in "Z".
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(
    url: str, timeout: float, retries: int = 2
) -> tuple[int | None, bytes, str | None, str]:
    """GET a URL with retry and backoff.

    Args:
        url: URL to fetch.
        timeout: Per-request timeout in seconds.
        retries: Number of retries after the first attempt.

    Returns:
        Tuple of (HTTP status or None, body bytes, error string or None,
        final URL after redirects).
    """
    last_error: str | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read(), None, response.url
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read()
            except Exception:  # Body read is best effort only.
                body = b""
            # Retry server errors, not client errors.
            if exc.code >= 500 and attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return exc.code, body, None, url
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
        except (TimeoutError, OSError) as exc:
            last_error = str(exc)
        if attempt < retries:
            time.sleep(1.0 * (attempt + 1))
    return None, b"", last_error, url


def find_dated_items(text: str) -> list[str]:
    """Extract date strings from text, best effort only.

    Args:
        text: Decoded body text to scan.

    Returns:
        Sorted unique date strings found, capped at 20. Empty when none
        are found; absence of dates is normal for llms.txt content.
    """
    found = set(DATE_RE.findall(text)) | set(MONTH_DATE_RE.findall(text))
    return sorted(found)[:20]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Watch agent-board endpoints (llms.txt, skill.md, "
        "mcp.json, root) and record their current state."
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (else SCANNER_OUT_DIR, else ./out).",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24,
        help="Accepted for CLI uniformity; endpoint checks are not "
        "time-filtered because llms.txt carries no timestamps.",
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
        help="Check at most this many targets (0 = all). For testing.",
    )
    return parser


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    """Run the endpoint watch.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (records, exit code): 0 when at least one endpoint
        check succeeded, 2 when every check failed (degraded).
    """
    targets = TARGETS[: args.limit] if args.limit > 0 else TARGETS
    records: list[dict[str, Any]] = []
    successes = 0
    for domain, coverage in targets:
        for endpoint in ENDPOINTS:
            url = f"https://{domain}{endpoint}"
            status, body, error, final_url = fetch(url, args.timeout)
            if status is not None and 200 <= status < 300:
                successes += 1
            text = body.decode("utf-8", errors="ignore")
            dated = find_dated_items(text) if endpoint == "/llms.txt" else []
            records.append(
                {
                    "source": domain,
                    "coverage": coverage,
                    "endpoint": endpoint,
                    "status_code": status,
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest()
                    if status is not None
                    else None,
                    "dated_items": dated,
                    "error": error,
                    "fetched_at": utc_now_iso(),
                    # Endpoint state, not a time feed: llms.txt has no
                    # timestamps, so no 24h window is claimed here.
                    "window": "current-state",
                    "provenance": final_url,
                }
            )
            time.sleep(args.delay)
    exit_code = 0 if successes > 0 else 2
    return records, exit_code


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse args, run the watch, write JSONL output.

    Args:
        argv: Argument list, or None to use sys.argv.

    Returns:
        Process exit code: 0 ok, 2 degraded (all checks failed),
        1 on unexpected error.
    """
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir or os.environ.get("SCANNER_OUT_DIR") or "./out"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create out dir {out_dir}: {exc}", file=sys.stderr)
        return 1
    records, exit_code = run(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"agent_boards-{stamp}.jsonl")
    try:
        with open(out_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"error: cannot write {out_path}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(records)} records to {out_path}")
    if exit_code == 2:
        print(
            "degraded: every endpoint check failed; see error fields "
            "in the output file.",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
