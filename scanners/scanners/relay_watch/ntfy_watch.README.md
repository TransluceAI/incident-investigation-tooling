# ntfy_watch — ntfy.sh public-topic watch

## Purpose

Poll operator-configured public ntfy.sh topics for cached messages
published since a Unix timestamp `window_hours` ago.

## Endpoint (verified)

Per the ntfy docs (subscribe API) and a live probe:

```text
GET https://ntfy.sh/<topic>/json?poll=1&since=<unix timestamp>
```

- `since` accepts a Unix timestamp (also a duration, a message ID, or
  `all`). This scanner always uses a Unix timestamp so the requested
  window is explicit.
- `poll=1` is required: without it the JSON endpoint is an open-ended
  stream that never terminates. With `poll=1` the connection ends
  after the cached messages are read.
- Probe note (2026-10-31): the poll form above was verified live —
  `/<topic>/json?poll=1&since=<unix>` returned 200 with cached
  messages, a topic with no cached messages returned 200 with an
  empty body (not an error), and omitting `poll=1` left the stream
  open. Some networks block or proxy egress to ntfy.sh, so a
  per-topic connection failure can be a network property, not an
  ntfy API error; per-topic failures are recorded as degraded,
  never as crashes.

## Topics are operator-configured — no defaults

With no topics supplied, the scanner exits 2 (degraded) with this
message and fetches nothing:

```text
degraded: no ntfy topics configured. Pass --topics ...
This scanner does not invent or ship a default topic.
```

It must never invent or ship a default topic: a guessed topic would
be someone else's notification feed.

## Usage

```sh
export PATH="$HOME/.local/bin:$PATH"
uv run scanners/relay_watch/ntfy_watch.py --topics mytopic,other-topic --out-dir ./out
uv run scanners/relay_watch/ntfy_watch.py --topics mytopic --window-hours 12 --out-dir ./out
uv run scanners/relay_watch/ntfy_watch.py --server https://ntfy.example.com --topics mytopic
```

No key is needed for public topics. `NTFY_TOPICS` (comma list) is an
alternative to `--topics`. `SCANNER_OUT_DIR` is the fallback output
directory when `--out-dir` is absent.

## Output

JSONL, one object per message, in
`<out-dir>/ntfy_watch-<UTC timestamp>.jsonl`:

| Field | Meaning |
| --- | --- |
| `source` | `ntfy.sh/<topic>` |
| `topic` | Topic name |
| `message_id` | ntfy message ID |
| `message_time` / `message_time_unix` | Message timestamp (ISO-8601 / Unix) |
| `title` | Message title, if any |
| `message` | Message text, capped at 2000 chars |
| `tags` | Message tags, if any |
| `fetched_at` | Fetch time, UTC ISO-8601 |
| `window` | `"requested 24h, cache-limited"` — requested lookback; the server cache limits actual coverage |
| `window_requested` | Requested lookback, e.g. `"24h"` |
| `window_note` | Cache-limit note (below) |
| `provenance` | URL queried |

An empty topic (no cached messages) is normal and yields zero records
for that topic, not an error.

## Actual window — read this before claiming 24h

ntfy.sh caches messages for a limited time (about 12 hours on the
default server — the documented `cache-duration` default is `12h`,
and live message `expires` fields confirm it). A 24-hour request
therefore may return less than 24 hours of messages. Each record's
`window` is `requested <N>h, cache-limited` — never a bare hour
claim — and `window_requested` plus `window_note` carry the details;
the scanner never claims coverage the server did not return.

## Limits

- Public topics only. A private topic returns 401/403 and is recorded
  as a per-topic failure.
- Message text is recorded, never executed; no instructions inside
  messages are followed.
- GET only. This scanner never publishes (no POST).

## What "degraded" means

Exit 0: at least one topic polled successfully (even with zero
messages). Exit 2 (degraded): no topics configured, or every
configured topic failed. Exit 1: local error (output directory/file).
