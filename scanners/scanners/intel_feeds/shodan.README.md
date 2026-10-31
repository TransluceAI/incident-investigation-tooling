# shodan.py — Shodan host search (keyed only)

## Purpose
Search Shodan's indexed host banners for agent-harness-style services.

## Source
- API base: `https://api.shodan.io` (per developer.shodan.io)
- Endpoint: `GET /shodan/host/search?key=<key>&query=<query>&minify=true`
- Default query (documented example, an agent-harness facet-style
  query): `http.component:mcp OR http.title:"agent board"`
- Override with `--query`.

## Actual window
`--window-hours` (default 24). Shodan matches carry scan timestamps;
matches scanned before the window are dropped. Matches with no
parseable timestamp are kept but stamped `window: "undated"` — they
are search results, not window-filtered findings, and must not be
reported as in-window.

## Env vars
- `SHODAN_API_KEY` (required) — from your Shodan account page. Keyed
  only; there is no keyless search path (a keyless probe of the
  endpoint returns HTTP 401). The key is sent
  only to api.shodan.io and is never written to output or logs; the
  `provenance` URL in records omits the key.

## Usage
```sh
SHODAN_API_KEY=... uv run shodan.py --out-dir ./out
SHODAN_API_KEY=... uv run shodan.py --query 'product:elastic port:9200'
```

## Output schema (JSONL, one object per host match)
`source` (`"shodan"`), `fetched_at` (UTC ISO-8601), `window`,
`provenance` (search URL without the key), `query`, `ip`, `port`,
`org`, `hostnames`, `product`, `scanned_at`, `total_results`.

## Limits
- Search consumes Shodan query credits (plan-dependent); one page per
  run, `minify=true` to keep responses small.
- Free-tier search filters are limited; some filters need a paid plan.
- This scanner reads the index only; it never asks Shodan to scan
  anything and never touches a host directly.

## What "degraded" means
Exit 2, no records: `SHODAN_API_KEY` unset, or Shodan rejects the key
(HTTP 401/403). Exit 1: transport error or unexpected HTTP status.
Exit 0: records written.
