# wayback_cdx.py — Wayback Machine CDX index

## Purpose

List recent Wayback captures for a curated set of watched domains,
keyless.

## Source and verification (2026-10-31, live probe)

`https://web.archive.org/cdx/search/cdx` (HTTPS) — HTTP 200, JSON
array whose first row is the header (`urlkey`, `timestamp`,
`original`, `statuscode`, `mimetype`, `digest`). Params: `url=<domain>/*`,
`from`/`to` (YYYYMMDDHHMMSS), `output=json`, `limit`. The backend is
flaky: the API can time out at 30s with zero bytes, and under load
it returns a "Temporarily Offline" HTML page instead of JSON. Both
failure modes are handled as degraded (exit 2), never a crash.

## Actual window

The requested `--window-hours` window (default 24h), passed to the CDX
API itself as `from`/`to`, so the emitted `window` is the window
actually requested from the index. Note: this is the *capture* window,
not a content-change feed — a capture only exists if the crawler
visited the URL in the window.

## Curated default domains (8)

`swarmsay.com`, `commons.dooza.ai`, `robots.wiki`, `agenthaven.org`,
`agentcolony.one`, `msgboard.dev`, `moltbook.com`, `commonlog.ai`.
These are curated defaults only — not a claim about which domains
matter. Override with `--domains` (comma list) for any other targets.

## Env vars

None. Keyless only.

## Usage

```sh
uv run wayback_cdx.py --out-dir ./out --window-hours 24
uv run wayback_cdx.py --domains example.com --limit 50
```

## Output schema (JSONL, one object per capture)

`source`, `fetched_at` (UTC ISO-8601), `window`, `provenance` (URL
queried), `domain`, `timestamp`, `original`, `statuscode`, `mimetype`,
`digest`, `wayback_url`.

## Limits

- Wayback CDX is flaky (timeouts, "Temporarily Offline" pages). Retry
  later; do not treat an empty/degraded run as "no captures".
- `--limit` caps captures per domain.

## Degraded

Exit 2 if any domain ends degraded (403/429, non-JSON page, timeout
after retries). Exit 0 only when every domain succeeded.
