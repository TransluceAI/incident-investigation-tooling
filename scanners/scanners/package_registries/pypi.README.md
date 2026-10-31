# pypi.py — PyPI recent activity

## Purpose

Report recently updated and newly registered Python packages from PyPI.

## Source

Keyless public RSS feeds, no account or key:

- Updates: `https://pypi.org/rss/updates.xml`
- Newest packages: `https://pypi.org/rss/packages.xml`

Verified live 2026-10-31: both feeds return HTTP 200 RSS with
`pubDate` per item.

## Actual window

`--window-hours` (default 24) is enforced client-side on each item's
`pubDate`. The feeds are rolling recent-item lists, not a full archive:
if the feed's oldest item is newer than the window start, coverage is
shorter than the requested window, and only the items the feed actually
carries can be reported. The `window` field in each record is the
requested window; the feed limit is the coverage limit, stated here
rather than hidden.

## Env vars

None. No key exists for these feeds.

## Usage

```sh
uv run pypi.py --out-dir ./out --window-hours 24
uv run pypi.py --feed updates --window-hours 6
uv run pypi.py --feed packages
```

`--out-dir` falls back to `SCANNER_OUT_DIR`, then `./out`.

## Output schema (one JSON object per line)

| Field | Meaning |
|---|---|
| `source` | `"pypi.org"` |
| `fetched_at` | UTC ISO-8601 fetch time |
| `window` | Requested window, e.g. `"24h"` |
| `provenance` | Feed URL queried |
| `feed` | `"updates"` or `"packages"` |
| `event_type` | `"update"` or `"new_package"` |
| `name`, `version` | Package name and version (empty for new-package items, whose titles carry no version) |
| `title`, `link`, `summary` | Feed item title, project URL, description |
| `published_at` | Item `pubDate`, UTC ISO-8601 |

## Limits

- Feed length is set by PyPI; old items fall off and cannot be recovered
  through this scanner.
- Items without a parsable `pubDate` are skipped, never dated by guess.
- Fetched XML is parsed as data only, never executed.

## Degraded (exit 2)

One feed returns HTTP 403/429 while the other still yields records.
Exit 1 is a transport failure on all requested feeds or unparsable XML.
