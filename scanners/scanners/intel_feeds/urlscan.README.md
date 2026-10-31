# urlscan.py — urlscan.io search

## Purpose

Fetch recent urlscan.io scan results for a query, keyless by default.

## Source and verification (2026-10-31, live probe)

`https://urlscan.io/api/v1/search/?q=<query>&size=<n>` — HTTP 200,
JSON with a `results` list; each result carries `task.time`,
`task.url`, `page.url`, `page.domain`, `page.ip`, `page.country`.
Confirmed with `q=page.url:example.com`.

## Actual window

The requested `--window-hours` window (default 24h), enforced client-side
on `task.time`. The API's own date qualifier behaviour was not relied on,
so the window in each record is the filter actually applied here.

## Env vars

- `URLSCAN_API_KEY` (optional): higher rate limits / detail access.
  Absent, the keyless search path works.

## Usage

```sh
uv run urlscan.py --query "page.url:example.com" --limit 50
uv run urlscan.py --query "*" --window-hours 24 --out-dir ./out
```

## Output schema (JSONL, one object per scan)

`source`, `fetched_at` (UTC ISO-8601), `window`, `provenance` (URL
queried), `scan_id`, `scan_url`, `task_time`, `task_url`, `page_url`,
`page_domain`, `page_ip`, `page_country`.

## Limits

- Keyless rate limits apply; large `--limit` values may be truncated
  server-side. Results with an unparsable task time are kept (no time
  to filter on); results outside the window are dropped.

## Degraded

Exit 2 on HTTP 403/429 or a non-JSON block page. Exit 1 on transport
failure after retries.
