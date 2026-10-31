# urlhaus.py — URLhaus recent URLs

## Purpose
Fetch recently added malicious URLs from URLhaus (abuse.ch).

## Source
- API host: `https://urlhaus-api.abuse.ch/v1/` (docs:
  https://urlhaus-api.abuse.ch/)
- Endpoint: `GET /v1/urls/recent/limit/<n>/` — both the GET method
  (the docs' example is `curl -X GET .../v1/urls/recent/` and the
  response status includes `http_get_expected` for non-GET requests)
  and the `/limit/<n>/` path form are documented
  (`https://urlhaus-api.abuse.ch/v1/urls/recent/limit/3/`)
- Auth header when a key is set: `Auth-Key`

## Actual window
`--window-hours` (default 24). Records are filtered by `date_added`
when it parses. Entries whose `date_added` does not parse are kept
but stamped `window: "undated"` — they must not be reported as
in-window. Note the feed itself covers roughly the last 3 days,
so a 24h filter is a subset of the feed, not the feed's own window.

## Env vars
- `ABUSECH_API_KEY` (required in practice) — free abuse.ch Auth-Key
  from auth.abuse.ch. One key covers URLhaus, ThreatFox and
  MalwareBazaar.

## Usage
```sh
ABUSECH_API_KEY=... uv run urlhaus.py --out-dir ./out --limit 100
```

## Output schema (JSONL, one object per URL)
`source` (`"urlhaus"`), `fetched_at` (UTC ISO-8601), `window`,
`provenance` (URL queried), `urlhaus_id`, `url`, `threat`, `tags`,
`date_added`, `reporter`.

## Limits — key requirement verified live
Live probe 2026-10-31 without a key: HTTP 401
`{"error": "Unauthorized"}` on both the plain and `/limit/1/` recent
endpoints. Older write-ups calling URLhaus keyless are out of date:
abuse.ch announced (2026-10-21) that non-authenticated exports
stopped working. This scanner therefore treats a missing key as
degraded, not as a keyless run.
- Only the recent-URLs GET endpoint is used. URL/host/payload
  lookups (POST endpoints) are out of scope for this scanner.

## What "degraded" means
Exit 2, no records: `ABUSECH_API_KEY` unset, or URLhaus rejects the
key (HTTP 401/403). Exit 1: transport error or unexpected HTTP
status. Exit 0: records written.
