# mediawiki.py — MediaWiki recent changes (keyless)

## Purpose
Fetch recent changes from a list of public MediaWiki wikis.

## Source
- Pattern per wiki:
  `https://<wiki>/w/api.php?action=query&list=recentchanges&rcstart=<now>&rcdir=older&rcend=<window-start>&format=json`
- Default `--wikis`: `en.wikipedia.org,publictestwiki.com,collusion.wiki`

## Actual window
`--window-hours` (default 24). The API is asked for `rcstart`=now,
`rcdir`=older and `rcend`=window start, so the server itself filters
to the window; records stamp that window.

## Env vars
None.

## Usage
```sh
uv run mediawiki.py --out-dir ./out --window-hours 24
uv run mediawiki.py --wikis en.wikipedia.org,publictestwiki.com --limit 100
```

## Output schema (JSONL, one object per change)
`source` (`"mediawiki"`), `fetched_at` (UTC ISO-8601), `window`,
`provenance` (API URL queried), `wiki`, `type`, `title`, `pageid`,
`revid`, `old_revid`, `user`, `timestamp`, `comment`.

## Limits — per-wiki status verified live 2026-10-31
- `en.wikipedia.org`: works (recentchanges JSON returned).
- `publictestwiki.com`: works (recentchanges JSON returned).
- `collusion.wiki`: HTTP 403 "Forbidden" (Cloudflare) on its api.php
  from this environment. That wiki degrades individually; it does not
  block the other wikis. If collusion.wiki becomes reachable again,
  it will start emitting with no code change.

## What "degraded" means
Exit 2: at least one wiki failed (transport error or non-200) — the
other wikis' records are still written. Exit 0: all wikis succeeded.
Exit 1: reserved for unexpected failures outside per-wiki handling.
