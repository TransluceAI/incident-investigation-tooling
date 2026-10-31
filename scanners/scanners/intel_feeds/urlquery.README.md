# urlquery.py — urlquery.net latest reports

## Purpose

Fetch the latest public urlquery.net scan reports, keyless.

## Source and verification (2026-10-31, live probe)

A keyless path **exists** and is what this scanner uses; no keyed JSON
endpoint was verified, so none is implemented and no key is required.

- `https://urlquery.net/` ("Recent reports") — HTTP 200, server-rendered
  HTML table, 28 report rows with `<time datetime="...">` UTC timestamps.
- `https://urlquery.net/search/reports?q=<query>` — HTTP 200, same table
  (24 rows per page, pagination via `offset=24`, `offset=48`, …).
  `/search` alone 307-redirects here. A query-less search page renders
  zero rows, so "latest reports" uses the home page, not an empty search.
- A `limit` parameter is **not** accepted (HTTP 400); paging uses `offset`.

## Actual window

The requested `--window-hours` window (default 24h), enforced client-side
on each row's UTC timestamp. Rows outside the window are dropped.
A row whose timestamp does not parse is kept, with its raw timestamp
in the record, rather than dropped silently.

## Env vars

None. No key is used or needed.

## Usage

```sh
uv run urlquery.py --out-dir ./out --window-hours 24 --limit 50
uv run urlquery.py --query agent --limit 50
```

## Output schema (JSONL, one object per report)

`source`, `fetched_at` (UTC ISO-8601), `window`, `provenance` (URL
queried), `report_id`, `report_url`, `timestamp`, `submitted_url`, `ip`.

## Limits

- HTML scraping of a server-rendered page: if urlquery.net changes its
  table markup, rows parse to zero and the run writes an empty file
  with exit 0 — check record counts.
- Home-page mode returns one page only; search mode paginates to `--limit`.

## Degraded

Exit 2 on HTTP 403/429 (blocked/rate-limited). Exit 1 on transport
failure after retries. Never a traceback for a missing key: there is none.
