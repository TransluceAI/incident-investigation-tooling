# huggingface.py — Hugging Face newest models and datasets

## Purpose

Report recently created public models and datasets on the Hugging Face
Hub.

## Source

Keyless Hub list APIs:

- Models: `https://huggingface.co/api/models?sort=createdAt&limit=N`
- Datasets: `https://huggingface.co/api/datasets?sort=createdAt&limit=N`

Verified live 2026-10-31: both return HTTP 200 JSON lists with
`createdAt` per entry, newest first under `sort=createdAt`.

## Actual window

`--window-hours` (default 24) is enforced client-side on `createdAt`.
`--limit` (default 50 per listing) caps how far back the scanner can
see: on a busy day 50 entries may cover less than the requested window,
and only the entries the API returns can be reported. The `window`
field in each record is the requested window.

## Env vars

| Var | Effect |
|---|---|
| `HF_TOKEN` | Optional. Sent as a Bearer token for higher rate limits only. The keyless path works without it; a missing token is never an error. Never hardcode a token. |

Keyed upgrade: set `HF_TOKEN` to a read-only Hub token if rate limits
(429) appear on the keyless path.

## Usage

```sh
uv run huggingface.py --out-dir ./out --window-hours 24
uv run huggingface.py --kind models --limit 100
HF_TOKEN=hf_... uv run huggingface.py --kind datasets
```

`--out-dir` falls back to `SCANNER_OUT_DIR`, then `./out`.

## Output schema (one JSON object per line)

| Field | Meaning |
|---|---|
| `source` | `"huggingface.co"` |
| `fetched_at` | UTC ISO-8601 fetch time |
| `window` | Requested window, e.g. `"24h"` |
| `provenance` | API URL queried |
| `kind` | `"model"` or `"dataset"` |
| `repo_id`, `author` | Repository id and author |
| `created_at`, `last_modified` | Hub timestamps |
| `likes`, `downloads`, `private`, `gated` | Hub counters/flags |
| `tags` | Hub tags |
| `url` | Canonical Hub page URL |

## Limits

- Private and gated repos are not visible keyless; with a token, only
  repos that token may see appear.
- The models listing does not return `author`, `lastModified`, or
  `gated`, so those record fields are `null` for models; the datasets
  listing carries all three (verified live 2026-10-31).
- Entries without a parsable `createdAt` are skipped.
- Fetched JSON is parsed as data only, never executed.

## Degraded (exit 2)

One listing returns HTTP 401/403/429 while the other still yields
records. Exit 1 is a transport failure on all requested listings.
