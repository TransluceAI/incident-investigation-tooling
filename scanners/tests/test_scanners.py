"""Mocked-HTTP tests for every scanner in ``scanners/``.

No test touches the network: ``urllib.request.urlopen`` is patched in
every test and served by a small in-memory router returning realistic
payloads. Realistic means the payload shape is taken from the scanner's
own parsing code (field names, timestamp formats, envelope keys).

SHAPE NOTES (payload inferred from scanner code, not a live probe):
- ``transluce``: the public list endpoint is unverified by the scanner
  itself, so the list-of-findings payload is the shape its
  ``extract_items`` accepts, not a live-verified response.
- ``paste_watch``: the listing HTML is the shape its link extractor
  accepts (same-host ``/abc123``-style links), not a live listing page.
- ``vendor_intel`` / ``intel_blogs``: index-page HTML follows the
  scanners' own regexes (post-shaped paths, ``<time>`` rows).
All other tests use the documented public shape for the source
(RSS/Atom, CDX JSON, registry JSON, search-API JSON).

Happy-path payloads carry two items: one in-window ("fresh", now) and
one 30 days stale, so every client-side window filter is exercised:
the stale sibling must never become a record. Scanners that delegate
windowing to the server (query parameters) instead get an assertion
that the request URL actually carries the window bounds.

Every test asserts the contract fields ``source``, ``fetched_at``,
``window`` and ``provenance`` on every emitted record. Exit codes are
asserted exactly per scanner contract: 0 ok, 2 degraded (rate limit,
missing key, missing watchlist, unreachable source), 1 error; a
traceback fails the test by propagating out of ``main``.

Google Python Style Guide: docstrings, type hints, no wildcard imports,
no bare except.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import pytest

SCANNERS_DIR = Path(__file__).resolve().parent.parent / "scanners"

# Env keys the scanners read. Tests clear them by default so a real key
# on the machine can never leak into a mocked run; keyed-scanner tests
# set an obviously dummy value explicitly.
KNOWN_KEYS = (
    "OTX_API_KEY",
    "SHODAN_API_KEY",
    "ABUSECH_API_KEY",
    "GITHUB_TOKEN",
    "HF_TOKEN",
    "URLSCAN_API_KEY",
    "MAIL_TM_TOKEN",
    "NTFY_TOPICS",
    "WEBHOOKSITE_TOKEN",
    "WEBHOOKSITE_TOKEN_IDS",
    "SCANNER_OUT_DIR",
)

Router = Callable[[str], tuple[int, bytes]]

DUMMY_TOKEN = "dummy-test-token"

WEBHOOK_TOKEN_ID = "11111111-2222-3333-4444-555555555555"


def load_scanner(relative_path: str) -> ModuleType:
    """Load one scanner file as a module.

    Args:
        relative_path: Path relative to ``scanners/``, e.g.
            ``"package_registries/pypi.py"``.

    Returns:
        The loaded module.
    """
    path = SCANNERS_DIR / relative_path
    name = "scanner_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    """Minimal ``urlopen`` response: status, bytes body, URL."""

    def __init__(self, body: bytes, status: int = 200, url: str = "") -> None:
        """Store the fake body, status and URL.

        Args:
            body: Response body bytes.
            status: HTTP status code.
            url: Final URL (used by scanners that record redirects).
        """
        self._body = body
        self.status = status
        self.url = url
        self.headers: dict[str, str] = {}

    def read(self) -> bytes:
        """Return the body bytes."""
        return self._body

    def getcode(self) -> int:
        """Return the HTTP status code."""
        return self.status

    def geturl(self) -> str:
        """Return the final URL."""
        return self.url

    def info(self) -> dict[str, str]:
        """Return the (empty) headers mapping."""
        return self.headers

    def __enter__(self) -> FakeResponse:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc: Any) -> bool:
        """Exit the context manager without suppressing errors."""
        return False


def fake_urlopen(router: Router) -> Callable[..., FakeResponse]:
    """Build a ``urlopen`` replacement served by ``router``.

    Args:
        router: Maps a URL to ``(status, body bytes)``. Statuses >= 400
            raise ``HTTPError`` exactly like real ``urlopen``; raising
            ``URLError`` from the router simulates a transport failure.

    Returns:
        A function usable as ``urlopen`` side effect.
    """

    def _fake(
        request: Any, timeout: float | None = None, **kwargs: Any
    ) -> FakeResponse:
        url = (
            request.full_url
            if isinstance(request, urllib.request.Request)
            else str(request)
        )
        status, body = router(url)
        if status >= 400:
            raise urllib.error.HTTPError(
                url, status, "mock error", {}, io.BytesIO(body)
            )
        return FakeResponse(body, status, url)

    return _fake


def json_body(payload: Any) -> bytes:
    """Encode a JSON payload as UTF-8 bytes.

    Args:
        payload: JSON-serialisable value.

    Returns:
        UTF-8 encoded JSON bytes.
    """
    return json.dumps(payload).encode("utf-8")


def run_scanner_raw(
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    argv: list[str],
    router: Router,
    env: dict[str, str] | None = None,
) -> int:
    """Run one scanner's ``main`` with mocked HTTP and a raw argv.

    Args:
        monkeypatch: Pytest fixture for env patching.
        relative_path: Scanner path relative to ``scanners/``.
        argv: Full argument list (no ``--out-dir`` is injected).
        router: URL-to-payload mapping for the mocked ``urlopen``.
        env: Dummy env vars to set (all known keys are cleared first).

    Returns:
        The scanner's exit code.
    """
    module = load_scanner(relative_path)
    for key in KNOWN_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    with (
        mock.patch("urllib.request.urlopen", side_effect=fake_urlopen(router)),
        mock.patch("time.sleep", return_value=None),
    ):
        return module.main(argv)


def read_records(out_dir: Path) -> list[dict[str, Any]]:
    """Read every JSONL record written under ``out_dir``.

    Args:
        out_dir: Output directory to scan for ``*.jsonl`` files.

    Returns:
        Parsed records from all files, sorted by file name.
    """
    records: list[dict[str, Any]] = []
    for out_file in sorted(out_dir.glob("*.jsonl")):
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def run_scanner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: str,
    args: list[str],
    router: Router,
    env: dict[str, str] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Run one scanner with ``--out-dir tmp_path`` and mocked HTTP.

    Args:
        monkeypatch: Pytest fixture for env patching.
        tmp_path: Output directory passed as ``--out-dir``.
        relative_path: Scanner path relative to ``scanners/``.
        args: Extra CLI args (appended after ``--out-dir``).
        router: URL-to-payload mapping for the mocked ``urlopen``.
        env: Dummy env vars to set (all known keys are cleared first).

    Returns:
        Tuple of (exit code, parsed JSONL records from ``tmp_path``).
    """
    code = run_scanner_raw(
        monkeypatch,
        relative_path,
        ["--out-dir", str(tmp_path)] + args,
        router,
        env=env,
    )
    return code, read_records(tmp_path)


def assert_contract(records: list[dict[str, Any]]) -> None:
    """Assert records are non-empty and carry the contract fields.

    Args:
        records: Parsed JSONL records from a scanner run.
    """
    assert records, "expected at least one record"
    check_contract_fields(records)


def check_contract_fields(records: list[dict[str, Any]]) -> None:
    """Assert every record carries the four contract fields.

    Args:
        records: Parsed JSONL records from a scanner run; may be
            empty (degraded runs legitimately emit none).
    """
    for record in records:
        for field in ("source", "fetched_at", "window", "provenance"):
            assert field in record, f"record missing {field}: {record}"


def window_timestamps() -> dict[str, dict[str, Any]]:
    """Build fresh and stale timestamps in every scanner format.

    Returns:
        Dict with ``fresh`` (now) and ``stale`` (30 days ago) sub-dicts
        holding ISO-Z, naive ISO, CDX, Unix, RFC 822, human and
        space-separated date forms.
    """

    def forms(value: datetime) -> dict[str, Any]:
        return {
            "iso_z": value.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "iso_naive": value.strftime("%Y-%m-%dT%H:%M:%S"),
            "cdx": value.strftime("%Y%m%d%H%M%S"),
            "unix": int(value.timestamp()),
            "rfc822": format_datetime(value),
            "human": value.strftime("%b %d, %Y"),
            "space": value.strftime("%Y-%m-%d %H:%M:%S"),
            "space_utc": value.strftime("%Y-%m-%d %H:%M:%S") + " UTC",
        }

    now = datetime.now(UTC)
    return {"fresh": forms(now), "stale": forms(now - timedelta(days=30))}


def single_payload(payload: bytes) -> Router:
    """Return a router that serves one payload for every URL.

    Args:
        payload: Body bytes served for any URL with status 200.

    Returns:
        A router function.
    """

    def _router(url: str) -> tuple[int, bytes]:
        return 200, payload

    return _router


# ---------------------------------------------------------------------------
# Package registries (shape-realistic: RSS / registry JSON)
# ---------------------------------------------------------------------------


def test_pypi_rss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """PyPI: in-window RSS item becomes a record; stale one is dropped."""
    ts = window_timestamps()
    rss = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item><title>demo-pkg 1.2.3</title>"
        "<link>https://pypi.org/project/demo-pkg/</link>"
        "<description>demo package</description>"
        f"<pubDate>{ts['fresh']['rfc822']}</pubDate></item>"
        "<item><title>stale-pkg 0.0.1</title>"
        "<link>https://pypi.org/project/stale-pkg/</link>"
        "<description>old package</description>"
        f"<pubDate>{ts['stale']['rfc822']}</pubDate></item>"
        "</channel></rss>"
    ).encode()
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "package_registries/pypi.py",
        ["--feed", "updates"],
        single_payload(rss),
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["name"] == "demo-pkg"
    assert record["version"] == "1.2.3"
    assert record["event_type"] == "update"
    assert record["published_at"] == ts["fresh"]["iso_z"]
    assert record["link"] == "https://pypi.org/project/demo-pkg/"
    assert record["window"] == "24h"
    assert record["provenance"].endswith("updates.xml")
    assert all(r["name"] != "stale-pkg" for r in records)


def test_rubygems_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """RubyGems: in-window version becomes a record; stale one dropped."""
    ts = window_timestamps()
    payload = json_body(
        [
            {
                "name": "demo-gem",
                "version": "0.1.0",
                "version_created_at": ts["fresh"]["iso_z"],
                "info": "demo gem",
            },
            {
                "name": "stale-gem",
                "version": "9.9.9",
                "version_created_at": ts["stale"]["iso_z"],
                "info": "old gem",
            },
        ]
    )
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "package_registries/rubygems.py",
        ["--activity", "latest"],
        single_payload(payload),
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["name"] == "demo-gem"
    assert record["version"] == "0.1.0"
    assert record["activity"] == "latest"
    assert record["version_created_at"] == ts["fresh"]["iso_z"]
    assert record["gem_url"] == "https://rubygems.org/gems/demo-gem"
    assert all(r["name"] != "stale-gem" for r in records)


def test_npm_changes_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """npm: changes feed plus metadata becomes a record; stale dropped.

    Deleted changes-feed entries and packages whose metadata cannot be
    read are skipped, not fabricated.
    """
    ts = window_timestamps()

    def router(url: str) -> tuple[int, bytes]:
        if "changes" in url or "replicate" in url:
            return 200, json_body(
                {
                    "results": [
                        {"seq": 7, "id": "demo-pkg"},
                        {"seq": 8, "id": "stale-pkg"},
                        {"seq": 9, "id": "deleted-pkg", "deleted": True},
                    ]
                }
            )
        if url.endswith("/stale-pkg"):
            return 200, json_body(
                {
                    "time": {
                        "modified": ts["stale"]["iso_z"],
                        "created": ts["stale"]["iso_z"],
                    },
                    "dist-tags": {"latest": "0.0.1"},
                    "description": "old package",
                }
            )
        return 200, json_body(
            {
                "time": {
                    "modified": ts["fresh"]["iso_z"],
                    "created": ts["fresh"]["iso_z"],
                },
                "dist-tags": {"latest": "1.0.0"},
                "description": "demo package",
            }
        )

    code, records = run_scanner(
        monkeypatch, tmp_path, "package_registries/npm.py", [], router
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["name"] == "demo-pkg"
    assert record["latest_version"] == "1.0.0"
    assert record["modified_at"] == ts["fresh"]["iso_z"]
    assert record["seq"] == 7
    assert (
        record["metadata_provenance"] == "https://registry.npmjs.org/demo-pkg"
    )
    assert all("stale-pkg" != r["name"] for r in records)


def test_huggingface_models_datasets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hugging Face: fresh models/datasets become records; stale dropped."""
    ts = window_timestamps()
    payload = json_body(
        [
            {"id": "demo/test-item", "createdAt": ts["fresh"]["iso_z"]},
            {"id": "demo/stale-item", "createdAt": ts["stale"]["iso_z"]},
        ]
    )
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "package_registries/huggingface.py",
        [],
        single_payload(payload),
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 2  # one model record + one dataset record
    assert {r["kind"] for r in records} == {"model", "dataset"}
    for record in records:
        assert record["repo_id"] == "demo/test-item"
        assert record["created_at"] == ts["fresh"]["iso_z"]
    model = next(r for r in records if r["kind"] == "model")
    dataset = next(r for r in records if r["kind"] == "dataset")
    assert model["url"] == "https://huggingface.co/demo/test-item"
    assert dataset["url"] == ("https://huggingface.co/datasets/demo/test-item")


def test_go_proxy_latest_and_window_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Go proxy: both watchlist modules recorded, ``in_window`` honest.

    By default every watchlist module is reported with an ``in_window``
    flag; ``--only-in-window`` drops modules whose latest version
    predates the window.
    """
    ts = window_timestamps()
    fresh = json_body(
        {
            "Version": "v1.2.3",
            "Time": ts["fresh"]["iso_z"],
            "Origin": {
                "VCS": "git",
                "URL": "https://example.com/repo",
                "Ref": "v1.2.3",
            },
        }
    )
    stale = json_body({"Version": "v0.0.1", "Time": ts["stale"]["iso_z"]})

    def router(url: str) -> tuple[int, bytes]:
        return 200, stale if "stale-mod" in url else fresh

    args = ["--modules", "example.com/fresh,example.com/stale-mod"]
    code, records = run_scanner(
        monkeypatch,
        tmp_path / "default",
        "package_registries/go_proxy.py",
        args,
        router,
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 2
    by_module = {r["module"]: r for r in records}
    assert by_module["example.com/fresh"]["in_window"] is True
    assert by_module["example.com/fresh"]["latest_version"] == "v1.2.3"
    assert (
        by_module["example.com/fresh"]["latest_version_time"]
        == (ts["fresh"]["iso_z"])
    )
    assert by_module["example.com/stale-mod"]["in_window"] is False

    code, records = run_scanner(
        monkeypatch,
        tmp_path / "filtered",
        "package_registries/go_proxy.py",
        args + ["--only-in-window"],
        router,
    )
    assert code == 0
    assert [r["module"] for r in records] == ["example.com/fresh"]


def test_go_proxy_no_modules_degraded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Go proxy: no watchlist exits degraded (2) without fetching."""
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "package_registries/go_proxy.py",
        [],
        single_payload(b"{}"),
    )
    assert code == 2
    assert records == []


# ---------------------------------------------------------------------------
# Intel feeds (shape-realistic: search / feed / list JSON and HTML)
# ---------------------------------------------------------------------------


def test_crtsh_certs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """crt.sh: in-window certificate becomes a record; stale dropped."""
    ts = window_timestamps()
    payload = json_body(
        [
            {
                "id": 123,
                "common_name": "example.com",
                "name_value": "example.com",
                "not_before": ts["fresh"]["iso_naive"],
                "not_after": ts["fresh"]["iso_naive"],
                "issuer_name": "Test CA",
            },
            {
                "id": 456,
                "common_name": "stale.com",
                "name_value": "stale.com",
                "not_before": ts["stale"]["iso_naive"],
                "not_after": ts["stale"]["iso_naive"],
                "issuer_name": "Test CA",
            },
        ]
    )
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/crtsh.py",
        ["--domains", "example.com"],
        single_payload(payload),
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["common_name"] == "example.com"
    assert record["cert_id"] == 123
    assert record["not_before"] == ts["fresh"]["iso_naive"]
    assert record["domain"] == "example.com"


def test_github_search_issues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GitHub search: issue items become records, window is requested.

    Window filtering is delegated to the GitHub search qualifier, so
    the test asserts the request URL carries ``created:>`` bounds.
    """
    ts = window_timestamps()
    payload = json_body(
        {
            "items": [
                {
                    "id": 1,
                    "html_url": "https://github.com/x/y/issues/1",
                    "title": "demo",
                    "created_at": ts["fresh"]["iso_z"],
                    "updated_at": ts["fresh"]["iso_z"],
                    "repository": {"full_name": "x/y"},
                    "user": {"login": "tester"},
                }
            ]
        }
    )
    requested: list[str] = []

    def router(url: str) -> tuple[int, bytes]:
        requested.append(url)
        return 200, payload

    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/github_search.py",
        ["--endpoint", "issues"],
        router,
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["title"] == "demo"
    assert record["repository"] == "x/y"
    assert record["user"] == "tester"
    assert record["created_at"] == ts["fresh"]["iso_z"]
    assert record["endpoint"] == "issues"
    unquoted = urllib.parse.unquote("".join(requested))
    assert "created:>" in unquoted


def test_hn_algolia_hits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HN Algolia: hit fields are extracted and the window requested.

    Window filtering is delegated to the API's ``numericFilters``, so
    the test asserts the request URL carries ``created_at_i`` bounds.
    """
    ts = window_timestamps()
    payload = json_body(
        {
            "hits": [
                {
                    "objectID": "1",
                    "_tags": ["story"],
                    "title": "demo story",
                    "url": "https://example.com/",
                    "author": "tester",
                    "created_at": ts["fresh"]["iso_z"],
                    "created_at_i": ts["fresh"]["unix"],
                    "points": 5,
                    "num_comments": 2,
                }
            ]
        }
    )
    requested: list[str] = []

    def router(url: str) -> tuple[int, bytes]:
        requested.append(url)
        return 200, payload

    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/hn_algolia.py",
        ["--query", "demo"],
        router,
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["object_id"] == "1"
    assert record["type"] == "story"
    assert record["created_at"] == ts["fresh"]["iso_z"]
    assert record["created_at_i"] == ts["fresh"]["unix"]
    unquoted = urllib.parse.unquote("".join(requested))
    assert "created_at_i>" in unquoted


def test_reddit_public_posts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Public Reddit: in-window post becomes a record; stale dropped."""
    ts = window_timestamps()

    def child(post_id: str, created: int, title: str) -> dict[str, Any]:
        return {
            "data": {
                "id": post_id,
                "title": title,
                "permalink": f"/r/test/comments/{post_id}/{title}/",
                "subreddit": "test",
                "author": "tester",
                "created_utc": created,
                "score": 3,
                "num_comments": 1,
            }
        }

    payload = json_body(
        {
            "data": {
                "children": [
                    child("abc", ts["fresh"]["unix"], "demo post"),
                    child("old", ts["stale"]["unix"], "stale post"),
                ]
            }
        }
    )
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/reddit_public.py",
        [],
        single_payload(payload),
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["post_id"] == "abc"
    assert record["title"] == "demo post"
    assert record["created_utc"] == ts["fresh"]["unix"]
    assert record["post_url"] == "https://www.reddit.com" + (
        "/r/test/comments/abc/demo post/"
    )
    assert all(r["post_id"] != "old" for r in records)


def test_urlquery_home_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """urlquery.net: in-window report row becomes a record; stale out."""
    ts = window_timestamps()

    def row(report_id: str, when: str) -> str:
        return (
            "<tr><td>"
            f'<time datetime="{when}">now</time></td><td>'
            f'<a href="/report/{report_id}">'
            "https://example.com/</a></td><td>"
            '<span class="mono ip-literal">93.184.216.34</span>'
            "</td></tr>"
        )

    html = (
        "<table>"
        + row("12345678-1234-1234-1234-123456789abc", ts["fresh"]["iso_z"])
        + row("99999999-9999-9999-9999-999999999999", ts["stale"]["iso_z"])
        + "</table>"
    ).encode()
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/urlquery.py",
        [],
        single_payload(html),
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["report_id"] == "12345678-1234-1234-1234-123456789abc"
    assert record["ip"] == "93.184.216.34"
    assert record["timestamp"] == ts["fresh"]["iso_z"]
    assert record["submitted_url"] == "https://example.com/"
    assert record["report_url"] == (
        "https://urlquery.net/report/12345678-1234-1234-1234-123456789abc"
    )


def test_urlscan_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """urlscan.io: in-window search result becomes a record; stale out."""
    ts = window_timestamps()

    def result(scan_id: str, when: str) -> dict[str, Any]:
        return {
            "_id": scan_id,
            "result": f"https://urlscan.io/result/{scan_id}/",
            "task": {
                "time": when,
                "url": "https://example.com/",
            },
            "page": {
                "url": "https://example.com/",
                "domain": "example.com",
                "ip": "93.184.216.34",
                "country": "US",
            },
        }

    payload = json_body(
        {
            "results": [
                result("abc", ts["fresh"]["iso_z"]),
                result("old", ts["stale"]["iso_z"]),
            ]
        }
    )
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/urlscan.py",
        [],
        single_payload(payload),
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["scan_id"] == "abc"
    assert record["page_domain"] == "example.com"
    assert record["page_ip"] == "93.184.216.34"
    assert record["task_time"] == ts["fresh"]["iso_z"]
    assert record["scan_url"] == "https://urlscan.io/result/abc/"


def test_wayback_cdx_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Wayback CDX: capture fields extracted; window bounds requested."""
    ts = window_timestamps()
    payload = json_body(
        [
            ["timestamp", "original", "statuscode", "mimetype", "digest"],
            [
                ts["fresh"]["cdx"],
                "https://example.com/",
                "200",
                "text/html",
                "ABCDEF",
            ],
        ]
    )
    requested: list[str] = []

    def router(url: str) -> tuple[int, bytes]:
        requested.append(url)
        return 200, payload

    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/wayback_cdx.py",
        ["--domains", "example.com"],
        router,
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["original"] == "https://example.com/"
    assert record["timestamp"] == ts["fresh"]["cdx"]
    assert record["statuscode"] == "200"
    assert record["wayback_url"] == (
        f"https://web.archive.org/web/{ts['fresh']['cdx']}/https://example.com/"
    )
    assert record["domain"] == "example.com"
    unquoted = urllib.parse.unquote("".join(requested))
    assert "from=" in unquoted and "to=" in unquoted


def test_commoncrawl_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Common Crawl: collinfo plus one capture becomes a record."""
    ts = window_timestamps()

    def router(url: str) -> tuple[int, bytes]:
        if "collinfo" in url:
            return 200, json_body(
                [
                    {
                        "id": "CC-MAIN-2025-30",
                        "cdx-api": (
                            "https://index.commoncrawl.org/"
                            "CC-MAIN-2025-30-index"
                        ),
                        "from": "2025-07-01",
                        "to": "2025-07-31",
                    }
                ]
            )
        line = json.dumps(
            {
                "url": "https://example.com/",
                "timestamp": ts["fresh"]["cdx"],
                "mime": "text/html",
                "status": "200",
                "digest": "ABCDEF",
                "filename": "crawl.warc.gz",
            }
        )
        return 200, line.encode("utf-8")

    code, records = run_scanner(
        monkeypatch, tmp_path, "intel_feeds/commoncrawl.py", [], router
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["capture_url"] == "https://example.com/"
    assert record["capture_time"] == ts["fresh"]["iso_z"]
    assert record["crawl_index"] == "CC-MAIN-2025-30"
    assert record["status"] == "200"


def test_commoncrawl_index_degraded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Common Crawl: index-query HTTP failure exits degraded (2)."""

    def router(url: str) -> tuple[int, bytes]:
        if "collinfo" in url:
            return 200, json_body(
                [{"id": "CC-MAIN-2025-30", "from": "2025-07-01"}]
            )
        return 503, b"slow"

    code, records = run_scanner(
        monkeypatch, tmp_path, "intel_feeds/commoncrawl.py", [], router
    )
    assert code == 2
    assert records == []


def test_otx_pulses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OTX (keyed, dummy key): in-window pulse kept; stale dropped."""
    ts = window_timestamps()
    payload = json_body(
        {
            "results": [
                {
                    "id": "pulse-1",
                    "name": "demo pulse",
                    "author_name": "tester",
                    "modified": ts["fresh"]["iso_z"],
                    "created": ts["fresh"]["iso_z"],
                    "tags": ["demo"],
                    "indicators": [{}],
                    "references": [],
                },
                {
                    "id": "pulse-old",
                    "name": "stale pulse",
                    "author_name": "tester",
                    "modified": ts["stale"]["iso_z"],
                    "created": ts["stale"]["iso_z"],
                    "tags": [],
                    "indicators": [],
                    "references": [],
                },
            ]
        }
    )
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/otx.py",
        [],
        single_payload(payload),
        env={"OTX_API_KEY": DUMMY_TOKEN},
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["pulse_id"] == "pulse-1"
    assert record["pulse_name"] == "demo pulse"
    assert record["author"] == "tester"
    assert record["indicator_count"] == 1
    assert record["modified"] == ts["fresh"]["iso_z"]


def test_shodan_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Shodan (keyed, dummy key): in-window match kept, stale dropped.

    The API key must never leak into the record provenance.
    """
    ts = window_timestamps()

    def match(ip: str, when: str) -> dict[str, Any]:
        return {
            "ip_str": ip,
            "port": 80,
            "org": "Demo",
            "hostnames": [],
            "product": "http",
            "timestamp": when,
        }

    payload = json_body(
        {
            "total": 2,
            "matches": [
                match("93.184.216.34", ts["fresh"]["iso_z"]),
                match("198.51.100.7", ts["stale"]["iso_z"]),
            ],
        }
    )
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/shodan.py",
        [],
        single_payload(payload),
        env={"SHODAN_API_KEY": DUMMY_TOKEN},
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["ip"] == "93.184.216.34"
    assert record["scanned_at"] == ts["fresh"]["iso_z"]
    assert record["total_results"] == 2
    assert DUMMY_TOKEN not in str(record["provenance"])
    assert all(r["ip"] != "198.51.100.7" for r in records)


def test_threatfox_iocs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ThreatFox (keyed, dummy key): in-window IOC kept; stale out."""
    ts = window_timestamps()

    def entry(ioc: str, first_seen: str) -> dict[str, Any]:
        return {
            "ioc": ioc,
            "ioc_type": "url",
            "threat_type": "payload_delivery",
            "malware": "Demo",
            "first_seen": first_seen,
            "last_seen": first_seen,
            "confidence_level": 80,
            "tags": ["demo"],
            "reporter": "tester",
        }

    payload = json_body(
        {
            "query_status": "ok",
            "data": [
                entry("https://example.com/bad", ts["fresh"]["space"]),
                entry("https://example.com/old", ts["stale"]["space"]),
            ],
        }
    )
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/threatfox.py",
        [],
        single_payload(payload),
        env={"ABUSECH_API_KEY": DUMMY_TOKEN},
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["ioc"] == "https://example.com/bad"
    assert record["threat_type"] == "payload_delivery"
    assert record["first_seen"] == ts["fresh"]["space"]
    assert record["window"] == "1d"


def test_urlhaus_recent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """URLhaus (keyed, dummy key): in-window URL kept; stale dropped."""
    ts = window_timestamps()

    def url_entry(url_id: str, url: str, added: str) -> dict[str, Any]:
        return {
            "id": url_id,
            "url": url,
            "threat": "malware_download",
            "tags": ["demo"],
            "date_added": added,
            "reporter": "tester",
        }

    payload = json_body(
        {
            "query_status": "ok",
            "urls": [
                url_entry(
                    "1", "https://example.com/bad", ts["fresh"]["space_utc"]
                ),
                url_entry(
                    "2", "https://example.com/old", ts["stale"]["space_utc"]
                ),
            ],
        }
    )
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/urlhaus.py",
        [],
        single_payload(payload),
        env={"ABUSECH_API_KEY": DUMMY_TOKEN},
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["urlhaus_id"] == "1"
    assert record["url"] == "https://example.com/bad"
    assert record["threat"] == "malware_download"
    assert record["date_added"] == ts["fresh"]["space_utc"]


def test_transluce_findings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Transluce: in-window finding kept; stale finding dropped.

    Shape note: the public list endpoint is unverified, so this payload
    is the list shape the scanner's ``extract_items`` accepts.
    """
    ts = window_timestamps()
    payload = json_body(
        [
            {
                "id": "209",
                "title": "demo finding",
                "date": ts["fresh"]["iso_z"],
                "url": "https://example.com/finding",
                "summary": "demo summary",
            },
            {
                "id": "1",
                "title": "stale finding",
                "date": ts["stale"]["iso_z"],
                "url": "https://example.com/old",
                "summary": "old summary",
            },
        ]
    )
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/transluce.py",
        [],
        single_payload(payload),
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["finding_id"] == "209"
    assert record["title"] == "demo finding"
    assert record["window"] == "24h"


def test_mediawiki_recentchanges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MediaWiki: change fields extracted; window bounds requested."""
    ts = window_timestamps()
    payload = json_body(
        {
            "query": {
                "recentchanges": [
                    {
                        "type": "edit",
                        "title": "Demo",
                        "pageid": 1,
                        "revid": 2,
                        "old_revid": 1,
                        "user": "Tester",
                        "timestamp": ts["fresh"]["iso_z"],
                        "comment": "demo edit",
                    }
                ]
            }
        }
    )
    requested: list[str] = []

    def router(url: str) -> tuple[int, bytes]:
        requested.append(url)
        return 200, payload

    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/mediawiki.py",
        ["--wikis", "en.wikipedia.org"],
        router,
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["title"] == "Demo"
    assert record["revid"] == 2
    assert record["user"] == "Tester"
    assert record["timestamp"] == ts["fresh"]["iso_z"]
    assert record["wiki"] == "en.wikipedia.org"
    unquoted = urllib.parse.unquote("".join(requested))
    assert "rcstart=" in unquoted and "rcend=" in unquoted


def test_intel_blogs_atom_and_homepage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Intel blogs: fresh posts kept, stale dropped, snapshots hashed.

    Both a norforge.com Atom feed and a swarmcha homepage are served;
    only in-window posts become ``post`` records, and each site also
    emits a ``site_snapshot`` record whose hash covers the body.
    """
    ts = window_timestamps()
    atom = (
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<entry><title>Demo norforge post</title>"
        f"<published>{ts['fresh']['iso_z']}</published>"
        '<link href="https://norforge.com/blog/demo"/></entry>'
        "<entry><title>Stale norforge post</title>"
        f"<published>{ts['stale']['iso_z']}</published>"
        '<link href="https://norforge.com/blog/old"/></entry></feed>'
    ).encode()
    home = (
        f'<time datetime="{ts["fresh"]["iso_z"]}">now</time> '
        '<a href="/demo-post">Demo swarmcha post</a> '
        f'<time datetime="{ts["stale"]["iso_z"]}">old</time> '
        '<a href="/old-post">Stale swarmcha post</a>'
    ).encode()

    def router(url: str) -> tuple[int, bytes]:
        if "norforge.com" in url:
            return 200, atom
        return 200, home

    code, records = run_scanner(
        monkeypatch, tmp_path, "intel_feeds/intel_blogs.py", [], router
    )
    assert code == 0
    assert_contract(records)
    posts = [r for r in records if r.get("kind") == "post"]
    snapshots = [r for r in records if r.get("kind") == "site_snapshot"]
    assert len(posts) == 2
    assert {p["site"] for p in posts} == {"norforge.com", "swarmcha.se"}
    for post in posts:
        assert post["window"] == "24h"
    assert not any("Stale" in p["title"] for p in posts)
    assert len(snapshots) == 2
    for snapshot in snapshots:
        body = atom if snapshot["site"] == "norforge.com" else home
        assert snapshot["content_sha256"] == hashlib.sha256(body).hexdigest()
    norforge_snapshot = next(
        s for s in snapshots if s["site"] == "norforge.com"
    )
    assert norforge_snapshot["latest_post_title"] == "Demo norforge post"
    swarmcha_snapshot = next(s for s in snapshots if s["site"] == "swarmcha.se")
    assert swarmcha_snapshot["latest_post_title"] == "Demo swarmcha post"


def test_vendor_intel_index_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Vendor intel: dated post links kept; stale links dropped.

    Every configured vendor index page is served the same HTML with
    one fresh and one stale post link (dated via ``datetime``
    attributes so the dates cannot bleed between links).
    """
    ts = window_timestamps()
    html = (
        '<a href="/index/fresh-vendor-post/">'
        f'<time datetime="{ts["fresh"]["iso_z"]}">Fresh vendor post'
        "</time></a>" + "x" * 300 + '<a href="/index/stale-vendor-post/">'
        f'<time datetime="{ts["stale"]["iso_z"]}">Stale vendor post'
        "</time></a>"
    ).encode()
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "intel_feeds/vendor_intel.py",
        [],
        single_payload(html),
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 3  # one fresh record per configured page
    assert {r["title"] for r in records} == {"Fresh vendor post"}
    assert {r["vendor"] for r in records} == {"openai.com", "anthropic.com"}
    for record in records:
        assert record["window"] == "24h"
        assert record["date"] == ts["fresh"]["iso_z"]


# ---------------------------------------------------------------------------
# Agent boards and relay watch (shape-realistic: endpoint / API payloads)
# ---------------------------------------------------------------------------


def test_agent_boards_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Agent boards: one record per endpoint with hash and dates.

    This is an endpoint-state watch: every endpoint gets a record with
    the HTTP status, body size, SHA-256 of the body and any date
    strings found in ``llms.txt``.
    """
    ts = window_timestamps()
    today = ts["fresh"]["iso_z"][:10]
    llms = f"# Demo board\n\nUpdated {today}.\n".encode()

    def router(url: str) -> tuple[int, bytes]:
        if url.endswith("/llms.txt"):
            return 200, llms
        return 200, b"ok"

    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "agent_boards/agent_boards.py",
        ["--limit", "1", "--delay", "0"],
        router,
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 4  # llms.txt, skill.md, mcp.json, /
    assert {r["endpoint"] for r in records} == {
        "/llms.txt",
        "/skill.md",
        "/mcp.json",
        "/",
    }
    llms_record = next(r for r in records if r["endpoint"] == "/llms.txt")
    assert llms_record["status_code"] == 200
    assert llms_record["bytes"] == len(llms)
    assert llms_record["sha256"] == hashlib.sha256(llms).hexdigest()
    assert llms_record["dated_items"] == [today]
    for record in records:
        assert record["window"] == "current-state"
        assert record["status_code"] == 200


def test_mail_tm_messages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """mail.tm (keyed, dummy token): in-window message kept, stale out.

    The API does not document ordering, so a stale message does not
    stop pagination: the scanner keeps scanning until an empty page.
    """
    ts = window_timestamps()

    def message(msg_id: str, created: str) -> dict[str, Any]:
        return {
            "id": msg_id,
            "createdAt": created,
            "from": {"address": f"{msg_id}@example.com"},
            "subject": f"demo {msg_id}",
            "intro": "hello",
            "seen": False,
        }

    requested: list[str] = []

    def router(url: str) -> tuple[int, bytes]:
        requested.append(url)
        if "page=1" in url:
            return 200, json_body(
                {
                    "hydra:member": [
                        message("msg-1", ts["fresh"]["iso_z"]),
                        message("msg-old", ts["stale"]["iso_z"]),
                    ]
                }
            )
        return 200, json_body({"hydra:member": []})

    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "relay_watch/mail_tm.py",
        [],
        router,
        env={"MAIL_TM_TOKEN": DUMMY_TOKEN},
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["message_id"] == "msg-1"
    assert record["from_address"] == "msg-1@example.com"
    assert record["created_at"] == ts["fresh"]["iso_z"]
    assert record["window"] == "24h"
    # Pagination continues past a stale message, then stops on the
    # first empty page.
    assert len([u for u in requested if "page=2" in u]) == 1
    assert not any("page=3" in u for u in requested)


def test_ntfy_watch_messages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ntfy.sh: message events become records; non-messages skipped.

    The poll request must carry ``poll=1`` and a ``since`` bound.
    """
    ts = window_timestamps()
    body = (
        json.dumps(
            {
                "event": "message",
                "id": "ntfy-1",
                "time": ts["fresh"]["unix"],
                "title": "demo",
                "message": "hello world",
                "tags": ["demo"],
            }
        ).encode("utf-8")
        + b"\n"
        + json.dumps({"event": "open"}).encode("utf-8")
        + b"\n"
    )
    requested: list[str] = []

    def router(url: str) -> tuple[int, bytes]:
        requested.append(url)
        return 200, body

    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "relay_watch/ntfy_watch.py",
        ["--topics", "testtopic"],
        router,
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    expected_time = datetime.fromtimestamp(
        ts["fresh"]["unix"], tz=UTC
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert record["message"] == "hello world"
    assert record["message_time"] == expected_time
    assert record["message_time_unix"] == ts["fresh"]["unix"]
    assert record["window_requested"] == "24h"
    unquoted = urllib.parse.unquote("".join(requested))
    assert "poll=1" in unquoted and "since=" in unquoted


def test_ntfy_watch_no_topics_degraded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ntfy.sh: with no topics it exits degraded (2) and fetches nothing."""
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "relay_watch/ntfy_watch.py",
        [],
        single_payload(b""),
    )
    assert code == 2
    assert records == []


def test_paste_watch_listing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Paste watch: a listing page reports its links and listing URL.

    Shape note: the listing HTML is the shape the link extractor
    accepts, not a live listing page.
    """
    html = b'<html><body><a href="/abc123">paste abc123</a></body></html>'
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "relay_watch/paste_watch.py",
        ["--limit", "1", "--delay", "0"],
        single_payload(html),
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "listing"
    assert record["items_found"] == 1
    assert record["items"] == ["https://tmpfiles.org/abc123"]
    assert record["listing_url"] == "https://tmpfiles.org/"
    assert record["window"] == "current-state"


def test_webhook_watch_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """webhook.site (keyed, dummy key): in-window request kept, stale out."""
    ts = window_timestamps()

    def request(req_id: str, created: str) -> dict[str, Any]:
        return {
            "uuid": req_id,
            "method": "POST",
            "url": "/hook",
            "created_at": created,
            "content": "{}",
        }

    payload = json_body(
        {
            "data": [
                request("req-1", ts["fresh"]["iso_z"]),
                request("req-old", ts["stale"]["iso_z"]),
            ]
        }
    )
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        "relay_watch/webhook_watch.py",
        ["--token-ids", WEBHOOK_TOKEN_ID],
        single_payload(payload),
        env={"WEBHOOKSITE_TOKEN": DUMMY_TOKEN},
    )
    assert code == 0
    assert_contract(records)
    assert len(records) == 1
    record = records[0]
    assert record["record_type"] == "request"
    assert record["method"] == "POST"
    assert record["request_uuid"] == "req-1"
    assert record["token_uuid"] == WEBHOOK_TOKEN_ID
    assert record["created_at"] == ts["fresh"]["iso_z"]
    assert record["content_size"] == 2
    assert record["window"] == "24h"


# ---------------------------------------------------------------------------
# Output-directory selection: --out-dir beats SCANNER_OUT_DIR beats ./out
# ---------------------------------------------------------------------------


def _demo_rss() -> bytes:
    """Return a one-item in-window RSS body for the out-dir tests.

    Returns:
        UTF-8 encoded RSS XML with a single in-window item.
    """
    now = format_datetime(datetime.now(UTC))
    return (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item><title>demo-pkg 1.2.3</title>"
        "<link>https://pypi.org/project/demo-pkg/</link>"
        "<description>demo package</description>"
        f"<pubDate>{now}</pubDate></item>"
        "</channel></rss>"
    ).encode()


def test_out_dir_env_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without ``--out-dir`` the scanner writes to ``SCANNER_OUT_DIR``."""
    env_dir = tmp_path / "env-out"
    code = run_scanner_raw(
        monkeypatch,
        "package_registries/pypi.py",
        ["--feed", "updates"],
        single_payload(_demo_rss()),
        env={"SCANNER_OUT_DIR": str(env_dir)},
    )
    assert code == 0
    records = read_records(env_dir)
    assert len(records) == 1
    assert records[0]["name"] == "demo-pkg"


def test_out_dir_flag_beats_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--out-dir`` wins when both it and the env var are set."""
    cli_dir = tmp_path / "cli-out"
    env_dir = tmp_path / "env-out"
    code = run_scanner_raw(
        monkeypatch,
        "package_registries/pypi.py",
        ["--feed", "updates", "--out-dir", str(cli_dir)],
        single_payload(_demo_rss()),
        env={"SCANNER_OUT_DIR": str(env_dir)},
    )
    assert code == 0
    assert len(read_records(cli_dir)) == 1
    assert read_records(env_dir) == []


def test_out_dir_default_is_dot_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no flag and no env var the scanner writes to ``./out``."""
    monkeypatch.chdir(tmp_path)
    code = run_scanner_raw(
        monkeypatch,
        "package_registries/pypi.py",
        ["--feed", "updates"],
        single_payload(_demo_rss()),
    )
    assert code == 0
    records = read_records(tmp_path / "out")
    assert len(records) == 1
    assert records[0]["name"] == "demo-pkg"


# ---------------------------------------------------------------------------
# Keyless degraded path: key-required scanners fetch nothing without a key
# ---------------------------------------------------------------------------

KEYLESS_CASES: list[tuple[str, list[str]]] = [
    ("intel_feeds/otx.py", []),
    ("intel_feeds/shodan.py", []),
    ("intel_feeds/threatfox.py", []),
    ("intel_feeds/urlhaus.py", []),
    ("relay_watch/mail_tm.py", []),
    ("relay_watch/webhook_watch.py", ["--token-ids", WEBHOOK_TOKEN_ID]),
]


@pytest.mark.parametrize("relative_path,args", KEYLESS_CASES)
def test_keyless_degraded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: str,
    args: list[str],
) -> None:
    """Key-required scanners exit degraded (2) without fetching."""
    requested: list[str] = []

    def router(url: str) -> tuple[int, bytes]:
        requested.append(url)
        return 200, b"{}"

    code, records = run_scanner(
        monkeypatch, tmp_path, relative_path, args, router
    )
    assert code == 2
    assert records == []
    assert requested == [], "keyless run must not touch the network"


# ---------------------------------------------------------------------------
# Empty results: feeds with nothing in-window exit 0 with no records
# ---------------------------------------------------------------------------

EMPTY_RSS = b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'


def _empty_npm(url: str) -> tuple[int, bytes]:
    if "changes" in url or "replicate" in url:
        return 200, json_body({"results": []})
    return 200, json_body({"time": {}, "dist-tags": {}})


def _empty_commoncrawl(url: str) -> tuple[int, bytes]:
    if "collinfo" in url:
        return 200, json_body([{"id": "CC-MAIN-2025-30", "from": "2025-07-01"}])
    return 200, b""


def _not_found_go_proxy(url: str) -> tuple[int, bytes]:
    return 404, b"not found"


EMPTY_CASES: list[tuple[str, list[str], dict[str, str], Router, int]] = [
    (
        "package_registries/pypi.py",
        ["--feed", "updates"],
        {},
        single_payload(EMPTY_RSS),
        0,
    ),
    (
        "package_registries/rubygems.py",
        ["--activity", "latest"],
        {},
        single_payload(json_body([])),
        0,
    ),
    ("package_registries/npm.py", [], {}, _empty_npm, 0),
    (
        "package_registries/huggingface.py",
        [],
        {},
        single_payload(json_body([])),
        0,
    ),
    (
        "package_registries/go_proxy.py",
        ["--modules", "example.com/demo"],
        {},
        _not_found_go_proxy,
        2,
    ),
    (
        "intel_feeds/crtsh.py",
        ["--domains", "example.com"],
        {},
        single_payload(json_body([])),
        0,
    ),
    (
        "intel_feeds/github_search.py",
        ["--endpoint", "issues"],
        {},
        single_payload(json_body({"items": []})),
        0,
    ),
    (
        "intel_feeds/hn_algolia.py",
        ["--query", "demo"],
        {},
        single_payload(json_body({"hits": []})),
        0,
    ),
    (
        "intel_feeds/reddit_public.py",
        [],
        {},
        single_payload(json_body({"data": {"children": []}})),
        0,
    ),
    (
        "intel_feeds/urlquery.py",
        [],
        {},
        single_payload(b"<html><body></body></html>"),
        0,
    ),
    (
        "intel_feeds/urlscan.py",
        [],
        {},
        single_payload(json_body({"results": []})),
        0,
    ),
    (
        "intel_feeds/wayback_cdx.py",
        ["--domains", "example.com"],
        {},
        single_payload(json_body([])),
        0,
    ),
    ("intel_feeds/commoncrawl.py", [], {}, _empty_commoncrawl, 0),
    (
        "intel_feeds/otx.py",
        [],
        {"OTX_API_KEY": DUMMY_TOKEN},
        single_payload(json_body({"results": []})),
        0,
    ),
    (
        "intel_feeds/shodan.py",
        [],
        {"SHODAN_API_KEY": DUMMY_TOKEN},
        single_payload(json_body({"total": 0, "matches": []})),
        0,
    ),
    (
        "intel_feeds/threatfox.py",
        [],
        {"ABUSECH_API_KEY": DUMMY_TOKEN},
        single_payload(json_body({"query_status": "ok", "data": []})),
        0,
    ),
    (
        "intel_feeds/urlhaus.py",
        [],
        {"ABUSECH_API_KEY": DUMMY_TOKEN},
        single_payload(json_body({"query_status": "ok", "urls": []})),
        0,
    ),
    (
        "intel_feeds/transluce.py",
        [],
        {},
        single_payload(json_body([])),
        0,
    ),
    (
        "intel_feeds/mediawiki.py",
        ["--wikis", "en.wikipedia.org"],
        {},
        single_payload(json_body({"query": {"recentchanges": []}})),
        0,
    ),
    (
        "relay_watch/mail_tm.py",
        [],
        {"MAIL_TM_TOKEN": DUMMY_TOKEN},
        single_payload(json_body({"hydra:member": []})),
        0,
    ),
    (
        "relay_watch/ntfy_watch.py",
        ["--topics", "testtopic"],
        {},
        single_payload(b""),
        0,
    ),
    (
        "relay_watch/webhook_watch.py",
        ["--token-ids", WEBHOOK_TOKEN_ID],
        {"WEBHOOKSITE_TOKEN": DUMMY_TOKEN},
        single_payload(json_body({"data": []})),
        0,
    ),
]


@pytest.mark.parametrize(
    "relative_path,args,env,router,expected_code", EMPTY_CASES
)
def test_empty_results_exit_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: str,
    args: list[str],
    env: dict[str, str],
    router: Router,
    expected_code: int,
) -> None:
    """Empty feeds exit cleanly with no records (0, or 2 for go_proxy).

    A 404 watchlist miss is the Go proxy's "nothing to report" case:
    it exits degraded rather than claiming an empty global feed.
    """
    code, records = run_scanner(
        monkeypatch, tmp_path, relative_path, args, router, env=env
    )
    assert code == expected_code
    assert records == []


# ---------------------------------------------------------------------------
# Degraded path: HTTP 429 everywhere, exact per-scanner contract
# ---------------------------------------------------------------------------

# (relative_path, args, env, expected_exit, expected_record_count,
#  expected stderr marker or None, expected field values on records)
RATE_LIMIT_CASES: list[
    tuple[str, list[str], dict[str, str], int, int, str | None, dict[str, Any]]
] = [
    (
        "package_registries/pypi.py",
        ["--feed", "updates"],
        {},
        2,
        0,
        "degraded",
        {},
    ),
    ("package_registries/rubygems.py", [], {}, 2, 0, "degraded", {}),
    ("package_registries/npm.py", [], {}, 2, 0, "degraded", {}),
    (
        "package_registries/huggingface.py",
        [],
        {},
        2,
        0,
        "degraded",
        {},
    ),
    (
        "package_registries/go_proxy.py",
        ["--modules", "example.com/demo"],
        {},
        2,
        0,
        "degraded",
        {},
    ),
    (
        "intel_feeds/crtsh.py",
        ["--domains", "example.com"],
        {},
        2,
        0,
        "degraded",
        {},
    ),
    (
        "intel_feeds/github_search.py",
        ["--endpoint", "issues"],
        {},
        2,
        0,
        "degraded",
        {},
    ),
    ("intel_feeds/hn_algolia.py", [], {}, 2, 0, "degraded", {}),
    ("intel_feeds/reddit_public.py", [], {}, 2, 0, "degraded", {}),
    ("intel_feeds/urlquery.py", [], {}, 2, 0, "degraded", {}),
    ("intel_feeds/urlscan.py", [], {}, 2, 0, "degraded", {}),
    (
        "intel_feeds/wayback_cdx.py",
        ["--domains", "example.com"],
        {},
        2,
        0,
        "degraded",
        {},
    ),
    # A blocked or rate-limited source is degraded (exit 2), not an
    # error, for every scanner.
    ("intel_feeds/commoncrawl.py", [], {}, 2, 0, "degraded", {}),
    (
        "intel_feeds/otx.py",
        [],
        {"OTX_API_KEY": DUMMY_TOKEN},
        2,
        0,
        "degraded",
        {},
    ),
    (
        "intel_feeds/shodan.py",
        [],
        {"SHODAN_API_KEY": DUMMY_TOKEN},
        2,
        0,
        "degraded",
        {},
    ),
    (
        "intel_feeds/threatfox.py",
        [],
        {"ABUSECH_API_KEY": DUMMY_TOKEN},
        2,
        0,
        "degraded",
        {},
    ),
    (
        "intel_feeds/urlhaus.py",
        [],
        {"ABUSECH_API_KEY": DUMMY_TOKEN},
        2,
        0,
        "degraded",
        {},
    ),
    ("intel_feeds/transluce.py", [], {}, 2, 0, "degraded", {}),
    (
        "intel_feeds/mediawiki.py",
        ["--wikis", "en.wikipedia.org"],
        {},
        2,
        0,
        "degraded",
        {},
    ),
    ("intel_feeds/intel_blogs.py", [], {}, 2, 0, "degraded", {}),
    ("intel_feeds/vendor_intel.py", [], {}, 2, 0, "degraded", {}),
    # Endpoint-state watchers still record the 429 responses.
    (
        "agent_boards/agent_boards.py",
        ["--limit", "1", "--delay", "0"],
        {},
        2,
        4,
        "degraded",
        {"status_code": 429},
    ),
    (
        "relay_watch/mail_tm.py",
        [],
        {"MAIL_TM_TOKEN": DUMMY_TOKEN},
        2,
        0,
        "degraded",
        {},
    ),
    (
        "relay_watch/ntfy_watch.py",
        ["--topics", "testtopic"],
        {},
        2,
        0,
        "degraded",
        {},
    ),
    # A 429 response proves the paste host is reachable: reported as
    # no_public_listing, exit 0.
    (
        "relay_watch/paste_watch.py",
        ["--limit", "1", "--delay", "0"],
        {},
        0,
        1,
        None,
        {"status": "no_public_listing"},
    ),
    (
        "relay_watch/webhook_watch.py",
        ["--token-ids", WEBHOOK_TOKEN_ID],
        {"WEBHOOKSITE_TOKEN": DUMMY_TOKEN},
        2,
        0,
        "degraded",
        {},
    ),
]


@pytest.mark.parametrize(
    "relative_path,args,env,expected_code,"
    "expected_count,stderr_marker,expected_fields",
    RATE_LIMIT_CASES,
)
def test_rate_limit_degraded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    relative_path: str,
    args: list[str],
    env: dict[str, str],
    expected_code: int,
    expected_count: int,
    stderr_marker: str | None,
    expected_fields: dict[str, Any],
) -> None:
    """Every scanner meets its documented 429 contract, no traceback.

    Feed scanners exit degraded (2) with no records; the keyed
    OTX/Shodan/ThreatFox/URLhaus family and Common Crawl's collinfo
    call treat 429 as an error (1); endpoint-state watchers keep
    emitting records that carry the 429 status.
    """

    def rate_limited(url: str) -> tuple[int, bytes]:
        return 429, b'{"error": "rate limited"}'

    code, records = run_scanner(
        monkeypatch, tmp_path, relative_path, args, rate_limited, env=env
    )
    assert code == expected_code
    check_contract_fields(records)
    assert len(records) == expected_count
    for record in records:
        for field, value in expected_fields.items():
            assert record[field] == value
    captured = capsys.readouterr()
    if stderr_marker is None:
        assert captured.err == ""
    else:
        assert stderr_marker in captured.err


# ---------------------------------------------------------------------------
# Transport failure: URLError everywhere, exact per-scanner contract
# ---------------------------------------------------------------------------

# (relative_path, args, env, expected_exit, expected_record_count,
#  expected field values on records)
TRANSPORT_CASES: list[
    tuple[str, list[str], dict[str, str], int, int, dict[str, Any]]
] = [
    ("package_registries/pypi.py", ["--feed", "updates"], {}, 1, 0, {}),
    (
        "package_registries/rubygems.py",
        ["--activity", "latest"],
        {},
        1,
        0,
        {},
    ),
    ("package_registries/npm.py", [], {}, 1, 0, {}),
    ("package_registries/huggingface.py", [], {}, 1, 0, {}),
    (
        "package_registries/go_proxy.py",
        ["--modules", "example.com/demo"],
        {},
        1,
        0,
        {},
    ),
    (
        "intel_feeds/crtsh.py",
        ["--domains", "example.com"],
        {},
        2,
        0,
        {},
    ),
    (
        "intel_feeds/github_search.py",
        ["--endpoint", "issues"],
        {},
        1,
        0,
        {},
    ),
    ("intel_feeds/hn_algolia.py", ["--query", "demo"], {}, 1, 0, {}),
    ("intel_feeds/reddit_public.py", [], {}, 1, 0, {}),
    ("intel_feeds/urlquery.py", [], {}, 1, 0, {}),
    ("intel_feeds/urlscan.py", [], {}, 1, 0, {}),
    (
        "intel_feeds/wayback_cdx.py",
        ["--domains", "example.com"],
        {},
        2,
        0,
        {},
    ),
    ("intel_feeds/commoncrawl.py", [], {}, 1, 0, {}),
    ("intel_feeds/otx.py", [], {"OTX_API_KEY": DUMMY_TOKEN}, 1, 0, {}),
    (
        "intel_feeds/shodan.py",
        [],
        {"SHODAN_API_KEY": DUMMY_TOKEN},
        1,
        0,
        {},
    ),
    (
        "intel_feeds/threatfox.py",
        [],
        {"ABUSECH_API_KEY": DUMMY_TOKEN},
        1,
        0,
        {},
    ),
    (
        "intel_feeds/urlhaus.py",
        [],
        {"ABUSECH_API_KEY": DUMMY_TOKEN},
        1,
        0,
        {},
    ),
    ("intel_feeds/transluce.py", [], {}, 1, 0, {}),
    (
        "intel_feeds/mediawiki.py",
        ["--wikis", "en.wikipedia.org"],
        {},
        2,
        0,
        {},
    ),
    ("intel_feeds/intel_blogs.py", [], {}, 2, 0, {}),
    ("intel_feeds/vendor_intel.py", [], {}, 2, 0, {}),
    # Endpoint-state watchers record the failure and degrade.
    (
        "agent_boards/agent_boards.py",
        ["--limit", "1", "--delay", "0"],
        {},
        2,
        4,
        {"status_code": None},
    ),
    (
        "relay_watch/mail_tm.py",
        [],
        {"MAIL_TM_TOKEN": DUMMY_TOKEN},
        2,
        0,
        {},
    ),
    (
        "relay_watch/ntfy_watch.py",
        ["--topics", "testtopic"],
        {},
        2,
        0,
        {},
    ),
    (
        "relay_watch/paste_watch.py",
        ["--limit", "1", "--delay", "0"],
        {},
        2,
        1,
        {"status": "unreachable"},
    ),
    (
        "relay_watch/webhook_watch.py",
        ["--token-ids", WEBHOOK_TOKEN_ID],
        {"WEBHOOKSITE_TOKEN": DUMMY_TOKEN},
        2,
        0,
        {},
    ),
]


def _transport_failure(url: str) -> tuple[int, bytes]:
    raise urllib.error.URLError("mock transport failure")


@pytest.mark.parametrize(
    "relative_path,args,env,expected_code,expected_count,expected_fields",
    TRANSPORT_CASES,
)
def test_transport_failure_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: str,
    args: list[str],
    env: dict[str, str],
    expected_code: int,
    expected_count: int,
    expected_fields: dict[str, Any],
) -> None:
    """A dead network exits per contract: 1 error or 2 degraded.

    Watchers that record endpoint state still emit records carrying
    the transport error.
    """
    code, records = run_scanner(
        monkeypatch, tmp_path, relative_path, args, _transport_failure, env=env
    )
    assert code == expected_code
    check_contract_fields(records)
    assert len(records) == expected_count
    for record in records:
        for field, value in expected_fields.items():
            assert record[field] == value
        assert record.get("error"), "failure record must carry the error"


RETRY_CASES: list[tuple[str, list[str], dict[str, str], int]] = [
    # Single-feed scanners retry three times before giving up.
    ("package_registries/pypi.py", ["--feed", "updates"], {}, 3),
    ("intel_feeds/hn_algolia.py", ["--query", "demo"], {}, 3),
    ("intel_feeds/shodan.py", [], {"SHODAN_API_KEY": DUMMY_TOKEN}, 3),
    # agent_boards retries each of the 4 endpoints three times.
    ("agent_boards/agent_boards.py", ["--limit", "1", "--delay", "0"], {}, 12),
    # paste_watch retries each candidate URL once after the first try.
    ("relay_watch/paste_watch.py", ["--limit", "1", "--delay", "0"], {}, 2),
]


@pytest.mark.parametrize(
    "relative_path,args,env,expected_attempts", RETRY_CASES
)
def test_transport_failure_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: str,
    args: list[str],
    env: dict[str, str],
    expected_attempts: int,
) -> None:
    """Transport failures are retried the documented number of times."""

    def flaky(url: str) -> tuple[int, bytes]:
        raise urllib.error.URLError("mock transport failure")

    module = load_scanner(relative_path)
    for key in KNOWN_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    counter = mock.Mock(side_effect=fake_urlopen(flaky))
    with (
        mock.patch("urllib.request.urlopen", side_effect=counter),
        mock.patch("time.sleep", return_value=None),
    ):
        code = module.main(["--out-dir", str(tmp_path)] + args)
    assert code in (1, 2)
    assert counter.call_count == expected_attempts


# ---------------------------------------------------------------------------
# Unparseable payloads: bad bodies exit 1 (error) or 2 (degraded)
# ---------------------------------------------------------------------------

# (relative_path, args, env, body bytes, expected_code, stderr marker)
UNPARSEABLE_CASES: list[
    tuple[str, list[str], dict[str, str], bytes, int, str]
] = [
    (
        "package_registries/pypi.py",
        ["--feed", "updates"],
        {},
        b"this is not xml at all",
        1,
        "unparsable RSS",
    ),
    (
        "package_registries/npm.py",
        [],
        {},
        b"this is not json",
        1,
        "error",
    ),
    (
        "intel_feeds/commoncrawl.py",
        [],
        {},
        b"this is not json",
        1,
        "did not return JSON",
    ),
    (
        "intel_feeds/github_search.py",
        ["--endpoint", "issues"],
        {},
        b"this is not json",
        2,
        "non-JSON",
    ),
    (
        "intel_feeds/transluce.py",
        [],
        {},
        b"this is not json",
        2,
        "did not return JSON",
    ),
    (
        "relay_watch/mail_tm.py",
        [],
        {"MAIL_TM_TOKEN": DUMMY_TOKEN},
        b"this is not json",
        2,
        "invalid JSON",
    ),
]


@pytest.mark.parametrize(
    "relative_path,args,env,body,expected_code,stderr_marker",
    UNPARSEABLE_CASES,
)
def test_unparseable_payloads_exit_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    relative_path: str,
    args: list[str],
    env: dict[str, str],
    body: bytes,
    expected_code: int,
    stderr_marker: str,
) -> None:
    """Unparseable bodies exit 1 or 2 with a message, never a traceback."""
    code, records = run_scanner(
        monkeypatch,
        tmp_path,
        relative_path,
        args,
        single_payload(body),
        env=env,
    )
    assert code == expected_code
    assert records == []
    assert stderr_marker in capsys.readouterr().err
