# threatfox.py — ThreatFox recent IOCs (keyed)

## Purpose
Fetch recently reported indicators of compromise from ThreatFox
(abuse.ch).

## Source
- `https://threatfox-api.abuse.ch/api/v1/` (docs:
  https://threatfox.abuse.ch/api/)
- Query: `{"query": "get_iocs", "days": <1-7>}` (ThreatFox
  community-API documentation)
- Auth header: `Auth-Key`

## Actual window — days, not hours
The API accepts whole days only (1–7, filtered on `first_seen`; API
default 3). `--window-hours` (default 24) is converted with ceil and
clamped to 1–7, and every record stamps the actual days used (`1d`,
`2d`, …) — never a 24h claim when more than one day was fetched.
Records are further filtered by `first_seen` when it parses; the
documented format is `YYYY-MM-DD HH:MM:SS UTC`.

## Env vars
- `ABUSECH_API_KEY` (required) — free abuse.ch Auth-Key from
  auth.abuse.ch. One key covers ThreatFox, URLhaus and MalwareBazaar.

## Usage
```sh
ABUSECH_API_KEY=... uv run threatfox.py --out-dir ./out --window-hours 24
ABUSECH_API_KEY=... uv run threatfox.py --window-hours 72   # fetches days=3, stamps 3d
```

## Output schema (JSONL, one object per IOC)
`source` (`"threatfox"`), `fetched_at` (UTC ISO-8601), `window`
(actual days), `provenance` (API URL), `ioc`, `ioc_type`,
`threat_type`, `malware`, `first_seen`, `last_seen`,
`confidence_level`, `tags`, `reporter`.

## Limits — read the method note
ThreatFox offers no GET endpoint for this query; its documented
method is a POST carrying a JSON query body. This scanner therefore
issues a single read-only POST query (no submission, no login, no
session, no state change). That is a deliberate, documented deviation
from the project-wide GET-only preference: the only alternative
would be a permanently-degraded stub that can never fetch ThreatFox
data. If strict GET-only is ever required, retire this scanner
rather than silently degrading it.
- Live probe 2026-10-31 without a key: HTTP 401
  `{"error": "Unauthorized"}`. Older write-ups calling ThreatFox
  keyless are out of date.

## What "degraded" means
Exit 2, no records: `ABUSECH_API_KEY` unset, key rejected (HTTP
401/403), or a non-`ok` `query_status`. Exit 1: transport error or
unexpected HTTP status. Exit 0: records written.
