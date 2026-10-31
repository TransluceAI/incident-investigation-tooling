# mail_tm — mail.tm API watch (keyed, read-only, metadata only)

## Purpose

Read message metadata from the operator's own mail.tm inbox, filtered
to messages created within `--window-hours` (default 24).

## Endpoint (verified by live probe and provider docs)

```text
GET https://api.mail.tm/messages?page=N
```

- Provider docs: https://docs.mail.tm/api/messages — `GET /messages`
  is paginated (`page` query parameter, up to 30 messages per page);
  responses are Hydra/JSON-LD with messages in `hydra:member`.
- Base URL `https://api.mail.tm` confirmed live 2026-10-31: `GET /`
  returns 200 with an entrypoint listing `/messages`, and
  unauthenticated `GET /messages` returns 401
  `{"code": 401, "message": "JWT Token not found"}`.
- Authentication is the operator-supplied JWT in the
  `Authorization: Bearer <jwt>` header, from the `MAIL_TM_TOKEN` env
  var.

## Keyed upgrade — and what is FORBIDDEN

Without `MAIL_TM_TOKEN`, the scanner exits 2 (degraded) with a clear
message and fetches nothing:

```text
degraded: no mail.tm token configured. Set MAIL_TM_TOKEN to an
operator-supplied mail.tm JWT ...
```

**Creating accounts is FORBIDDEN.** This scanner never POSTs to
`/accounts` or `/token` and never creates an inbox. It only reads
with a token the operator supplied. If the token is expired, the API
returns 401 and the scanner exits degraded (2) — it does not mint a
new identity to work around it.

## Usage

```sh
export PATH="$HOME/.local/bin:$PATH"
export MAIL_TM_TOKEN='<operator-supplied JWT>'
uv run scanners/relay_watch/mail_tm.py --out-dir ./out
uv run scanners/relay_watch/mail_tm.py --window-hours 12 --max-pages 3 --out-dir ./out
```

Env vars: `MAIL_TM_TOKEN` (required), `SCANNER_OUT_DIR` (fallback
output directory). The token is sent only as the Bearer header and is
never logged or written to output.

## Output

JSONL, one object per message within the window, in
`<out-dir>/mail_tm-<UTC timestamp>.jsonl`:

| Field | Meaning |
| --- | --- |
| `source` | `mail.tm` |
| `message_id` | Message ID |
| `from_address` | Sender address |
| `subject` | Message subject |
| `intro` | Message intro snippet (as returned by the list API) |
| `created_at` | Message creation timestamp |
| `seen` | Seen flag |
| `fetched_at` | Fetch time, UTC ISO-8601 |
| `window` | Actual window used, e.g. `"24h"` (filter is on `createdAt`) |
| `provenance` | Page URL queried |

Pages are fetched in order and stop at the first empty page or at
`--max-pages` (default 5, so up to 150 messages at 30 per page). The
API does not document message ordering, so the scanner does not stop
early at out-of-window messages; the `createdAt` filter does the
windowing. A message whose `createdAt` does not parse is kept, with
its raw `created_at` in the record, rather than dropped silently.

## Privacy limit

Only message metadata is emitted. Message bodies are NOT fetched or
stored, so private mailbox contents are never collected.

## What "degraded" means

Exit 0: the keyed call succeeded (even with zero messages in the
window). Exit 2 (degraded): no token configured, or the keyed API
call failed (401 = missing/expired/invalid JWT). Exit 1: local error
(output directory/file).
