# webhook_watch — webhook.site watch (keyed)

## Purpose

Read requests captured by the operator's own webhook.site tokens, via
the documented webhook.site API. This is the keyed path: without a key
the scanner fetches nothing and exits degraded.

## Endpoints (verified against docs and live probe)

```text
GET https://webhook.site/token                         # list account tokens
GET https://webhook.site/token/<id>/requests?sorting=newest&per_page=50
```

- Provider docs: https://docs.webhook.site/api/tokens.html and
  https://docs.webhook.site/api/requests.html. `GET /token` (list
  account tokens) requires authentication; `GET
  /token/<id>/requests` accepts `sorting` (`newest`/`oldest`),
  `per_page` (max 100) and `page`.
- Authentication is the `Api-Key` header. Per webhook.site docs, the
  token list requires authentication, and tokens owned by an account
  always require an API key.
- Timestamps: webhook.site returns `created_at` in UTC as
  `yyyy-MM-dd HH:mm:ss` (no offset); the scanner reads such naive
  stamps as UTC.
- Live probe (2026-10-31): unauthenticated
  `GET https://webhook.site/token` returns 401. A probe of
  `https://webhook.site/api/v1/tokens` is not a real endpoint (403,
  "belongs to an inactive account" page) and is not used.
- Some webhook.site endpoints work without auth for unowned tokens,
  but this scanner does not guess token IDs: it only reads tokens the
  operator's account owns (via the key) or token UUIDs the operator
  explicitly supplies.

## Keyed upgrade

Without `WEBHOOKSITE_TOKEN`, the scanner exits 2 (degraded) with a
clear message and fetches nothing — never a crash, never fake data:

```text
degraded: no webhook.site API key configured. Set WEBHOOKSITE_TOKEN
to your webhook.site API key (account settings) ...
```

Keyed upgrade path: create an API key in your webhook.site account
settings, then:

```sh
export WEBHOOKSITE_TOKEN='<your api key>'
```

## Usage

```sh
export PATH="$HOME/.local/bin:$PATH"
# List the account's tokens:
uv run scanners/relay_watch/webhook_watch.py --out-dir ./out
# Read requests for specific tokens (UUIDs from your webhook.site URLs):
uv run scanners/relay_watch/webhook_watch.py --token-ids <uuid1>,<uuid2> --out-dir ./out
```

Env vars: `WEBHOOKSITE_TOKEN` (API key, required),
`WEBHOOKSITE_TOKEN_IDS` (comma list, alternative to `--token-ids`),
`SCANNER_OUT_DIR` (fallback output directory). The key is sent only
as the `Api-Key` header and is never logged or written to output.

## Output

JSONL in `<out-dir>/webhook_watch-<UTC timestamp>.jsonl`. Two record
shapes:

Token records (`record_type: "token"`): `source`, `token_uuid`,
`alias`, `created_at`, `fetched_at`, `window`, `provenance`.
Token records list the account's tokens and are not window-filtered,
so their `window` is `"current-state"`, never an hour claim.

Request records (`record_type: "request"`, only when `--token-ids`
is given, filtered to `created_at` within `--window-hours`, default
24): `source`, `token_uuid`, `request_uuid`, `method`,
`request_url` (the captured request's URL as reported by the API),
`created_at`, `content_size` (API-reported body size in bytes;
falls back to the content string length when the API omits `size`),
`fetched_at`, `window`, `provenance`.

## Privacy limit

Only request metadata is emitted. Request bodies and headers are NOT
copied into output, so captured secrets (tokens, passwords, webhook
payloads) stay out of the JSONL.

## Hard limits

- GET only. This scanner never creates a token (that is a POST and is
  forbidden by the read-only contract) and never creates sessions.
- It never guesses or enumerates token UUIDs. Unknown UUIDs return an
  API error and are recorded as per-token failures.

## What "degraded" means

Exit 0: the keyed call succeeded. Exit 2 (degraded): no token
configured, or the keyed API call failed (e.g. 401 — invalid or
expired key — or every supplied token failed). Exit 1: local error
(output directory/file).
