# otx.py — AlienVault OTX subscribed pulses

## Purpose
Fetch pulses the OTX account is subscribed to, as threat-intel findings.

## Source
- API: `https://otx.alienvault.com/api/v1/pulses/subscribed`
- Auth header: `X-OTX-API-KEY`

## Actual window
`--window-hours` (default 24). Records are kept only when their `modified`
timestamp falls inside the window. Pulses with no parseable `modified` date
are kept (they cannot be window-filtered, and dropping them silently would
hide data).

## Env vars
- `OTX_API_KEY` (required) — free OTX API key from your OTX account settings.
  No key is hardcoded anywhere.

## Usage
```sh
uv run otx.py --out-dir ./out --window-hours 24 --limit 20
OTX_API_KEY=... uv run otx.py --out-dir ./out
```

## Output schema (JSONL, one object per pulse)
`source` (`"otx"`), `fetched_at` (UTC ISO-8601), `window`, `provenance`
(URL queried), `pulse_id`, `pulse_name`, `author`, `modified`, `created`,
`tags`, `indicator_count`, `references`.

## Limits
- Keyed only. Live probe 2026-10-31 without a key: HTTP 403
  `{"detail": "Authentication required"}`. There is no verified keyless path.
- A generic `/api/v1/pulses` list endpoint returned 404 in the same probe;
  it is not used.
- Transient server errors (HTTP 5xx) are retried with backoff before
  the run is reported as an error.

## What "degraded" means
Exit 2, no records: `OTX_API_KEY` unset, or OTX refuses the request
(HTTP 401/403/429 — bad key or rate limit). Exit 1: transport error or
unexpected HTTP status. Exit 0: records written (possibly zero if no
pulse was modified in-window).
