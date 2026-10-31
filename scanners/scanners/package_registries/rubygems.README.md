# rubygems.py — RubyGems recent activity

## Purpose

Report recently published and recently updated Ruby gems.

## Source

Keyless public activity API, no account or key:

- Latest: `https://rubygems.org/api/v1/activity/latest.json`
- Just updated: `https://rubygems.org/api/v1/activity/just_updated.json`

Verified live 2026-10-31: both endpoints return HTTP 200 JSON lists
with `version_created_at` per gem.

## Actual window

`--window-hours` (default 24) is enforced client-side on
`version_created_at`. Each endpoint returns a fixed recent list (50
entries at probe time), so on a busy day the list may cover less than
the requested window; only the entries the API actually returns can be
reported. The `window` field in each record is the requested window.

## Env vars

None. No key is needed for these endpoints.

## Usage

```sh
uv run rubygems.py --out-dir ./out --window-hours 24
uv run rubygems.py --activity just_updated
uv run rubygems.py --activity both
```

`--out-dir` falls back to `SCANNER_OUT_DIR`, then `./out`.
`--activity` defaults to `both` (latest and just_updated).

## Output schema (one JSON object per line)

| Field | Meaning |
|---|---|
| `source` | `"rubygems.org"` |
| `fetched_at` | UTC ISO-8601 fetch time |
| `window` | Requested window, e.g. `"24h"` |
| `provenance` | API URL queried |
| `activity` | `"latest"` or `"just_updated"` |
| `name`, `version` | Gem name and version |
| `version_created_at` | Version creation time from the API |
| `authors`, `info`, `licenses` | Gem metadata |
| `yanked`, `sha` | Yank status and version SHA |
| `homepage_uri`, `gem_url` | Homepage and rubygems.org page |

## Limits

- Only the recent entries each endpoint returns are visible; there is
  no paging parameter on these endpoints.
- Entries without a parsable `version_created_at` are skipped.
- Fetched JSON is parsed as data only, never executed.

## Degraded (exit 2)

An endpoint returns HTTP 403/429; records from the other endpoint (if
any) are still written. Exit 1 is a transport failure on all requested
endpoints.
