# npm.py — npm registry changes

## Purpose

Report npm packages changed recently, using the replicate changes feed
plus per-package registry metadata for real timestamps.

## Source

Keyless, no account or key:

- Changes feed: `https://replicate.npmjs.com/_changes`
  (CouchDB feed; `descending=true&limit=N` returns newest first).
  Verified live 2026-10-31: HTTP 200, newest entries at sequence
  ~141M. Per npm's 2025 replication-API migration notice,
  `/registry/_changes` is the canonical path and `/_changes` a
  supported alias; `descending` and `limit` are supported
  parameters, and package metadata is served by
  `registry.npmjs.org` (which is what this scanner uses).
- Per-package metadata: `https://registry.npmjs.org/<package>`
  (scoped names escaped as `@scope%2fname`).

## Actual window

The changes feed carries **no timestamps** — only sequence numbers and
revision ids. The scanner therefore reads each changed package's
registry metadata and enforces `--window-hours` (default 24)
client-side on `time.modified`. Only packages whose modified time falls
in the window are emitted. The `window` field in each record is that
requested window, applied to metadata time, not to feed position.

## Env vars

None. No key is needed for either endpoint.

## Usage

```sh
uv run npm.py --out-dir ./out --window-hours 24 --limit 50
uv run npm.py --limit 5 --window-hours 1
```

`--out-dir` falls back to `SCANNER_OUT_DIR`, then `./out`.
`--limit` caps changes-feed entries inspected (each triggers one
metadata fetch, so keep it modest).

## Output schema (one JSON object per line)

| Field | Meaning |
|---|---|
| `source` | `"npmjs.com"` |
| `fetched_at` | UTC ISO-8601 fetch time |
| `window` | Requested window, e.g. `"24h"` |
| `provenance` | Changes-feed URL queried |
| `metadata_provenance` | Registry metadata URL for the package |
| `seq` | Changes-feed sequence number |
| `name` | Package name |
| `latest_version` | Current `dist-tags.latest` |
| `modified_at`, `created_at` | Registry `time` fields |
| `description` | Package description |

## Limits

- Deleted packages and packages whose metadata cannot be read are
  counted in the stdout summary and skipped, never dated by guess.
- One metadata request per changed package: `--limit` is the politeness
  control.
- Fetched JSON is parsed as data only, never executed.

## Degraded (exit 2)

The metadata fetch hits HTTP 403/429 partway through (records already
gathered are still written). Exit 1 is a changes-feed transport failure.
