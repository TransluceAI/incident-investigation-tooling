# agent_boards — Agent-board endpoint watch

## Purpose

Poll 45 agent-board surfaces for their agent-facing endpoint files and
record the current state of each endpoint. This is an endpoint-state
watch: it answers "what does this endpoint return right now?", not
"what happened in the last 24 hours?".

## Plain statement on "latest activity"

**`llms.txt` has no timestamps, so "latest activity" here means
current endpoint state, not a 24h feed.** A change between runs (status
change, new SHA-256, new dated strings in `llms.txt`) is the signal.
Nothing in the output claims a 24-hour activity window, and the
`window` field in every record is literally `"current-state"`.
`--window-hours` is accepted for CLI uniformity only; it does not
filter anything, because there is nothing timestamped to filter on.

## Targets

45 domains, embedded as the curated `TARGETS` default in
`agent_boards.py` (39 established + 6 candidate, verified live
2026-10-31):

- 39 established (BASELINED) surfaces: swarmsay.com, commons.dooza.ai,
  robots.wiki, agenthaven.org, agentcolony.one, and 34 more — see
  `TARGETS` in `agent_boards.py`.
- 6 CANDIDATE surfaces, marked `CANDIDATE` in each record: swarmbar.com,
  agentboard.ai, agentdm.ai, anansi-haven.anansidata.workers.dev,
  fallback.chat, agentbadge.xyz.

## Endpoints probed, in order, per target

1. `/llms.txt`
2. `/skill.md`
3. `/mcp.json`
4. `/` (root)

That is 180 checks per full run. Polite: 0.2s delay between requests
(`--delay`), 30s timeout (`--timeout`), retry with backoff on server
errors and network errors.

## Usage

```sh
uv run scanners/agent_boards/agent_boards.py --out-dir ./out
uv run scanners/agent_boards/agent_boards.py --limit 3 --out-dir ./out   # test
```

No key is needed. No env vars are read except `SCANNER_OUT_DIR`
(fallback when `--out-dir` is absent).

## Output

JSONL, one object per endpoint check, in
`<out-dir>/agent_boards-<UTC timestamp>.jsonl`:

| Field | Meaning |
| --- | --- |
| `source` | Domain checked |
| `coverage` | `BASELINED` or `CANDIDATE` |
| `endpoint` | `/llms.txt`, `/skill.md`, `/mcp.json`, or `/` |
| `status_code` | HTTP status, or null on network failure |
| `bytes` | Body size in bytes |
| `sha256` | SHA-256 of the body, or null on network failure |
| `dated_items` | Date strings found in `llms.txt` content (best effort, ISO dates and "Month D, YYYY"); empty for other endpoints and when none are found |
| `error` | Network error string, or null |
| `fetched_at` | Fetch time, UTC ISO-8601 |
| `window` | Always `"current-state"` (see plain statement above) |
| `provenance` | Final URL after redirects |

## Limits

- A 404 is a recorded state, not an error: dead or missing endpoints
  are informative for a watch.
- `dated_items` is best effort only. Most `llms.txt` files carry no
  dates; an empty list is normal, not a failure.
- Slow hosts may time out at 30s; the timeout is recorded in the
  `error` field (probe note: swarmsay.com `/llms.txt` timed out at 12s
  during pre-build probing — slow, not proven dead).
- Fetched payloads are hashed and scanned for date strings only. They
  are never executed, and no instructions inside them are followed.

## What "degraded" means

Exit 0: at least one endpoint check succeeded. Exit 2 (degraded):
every endpoint check failed (network outage, egress block, or all
hosts down). Exit 1: local error (cannot create the output directory
or write the file).
