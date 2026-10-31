# vendor_intel.py — OpenAI + Anthropic index watch (keyless)

## Purpose
Watch vendor research/index pages for new posts.

## Sources
- `https://openai.com/index/`
- `https://openai.com/news/`
- `https://www.anthropic.com/research`
- Reachability, verified live 2026-10-31 with this scanner's user
  agent: `anthropic.com/research` returns HTTP 200 HTML and the
  extractor pulls post entries from it. The two OpenAI pages sit
  behind a Cloudflare bot wall that is intermittent from this
  environment: the same day produced both HTTP 403 (curl and the
  scanner's own urllib path; a fresh real-browser session was
  challenged too) and, later, HTTP 200 with real post entries. When
  walled, those pages degrade per-page (exit 2) while Anthropic
  records are still written; when open, they emit again with no code
  change.

## Actual window
`--window-hours` (default 24) for entries with a parseable date.
Entries with no parseable date are emitted with `date: null` and
`window: "undated"` — they are page-snapshot observations, not
window-filtered findings, and must not be reported as in-window.

## Method — honest and brittle by nature
This is HTML parsing, not an API. Links are extracted from page
HTML; a date is taken from a `datetime` attribute inside the link
when present, otherwise from a `Month D, YYYY` / `YYYY-MM-DD`
string near the link. Both sites are JavaScript-heavy, so the
extracted set is a best-effort sample of what the server-rendered
HTML contains, not a guaranteed-complete post list. If either site
changes markup, entries may drop to undated or disappear; that is a
known limit, not a silent failure — the run still reports how many
records it wrote and which pages degraded.

## Env vars
None.

## Usage
```sh
uv run vendor_intel.py --out-dir ./out --window-hours 24
```

## Output schema (JSONL, one object per entry)
`source` (`"vendor_intel"`), `fetched_at` (UTC ISO-8601), `window`
(`"<hours>h"` or `"undated"`), `provenance` (index page URL),
`vendor`, `title`, `url`, `date`.

## Limits
- No feed or API is used: neither vendor publishes a usable index
  feed for these pages that was verified here.
- Titles are link text, cleaned of tags; long navigation links are
  skipped by a length filter (8–200 chars).

## What "degraded" means
Exit 2: at least one page failed (transport error or non-200) — the
other pages' records are still written. Exit 0: all pages
succeeded. Exit 1: unexpected failure.
