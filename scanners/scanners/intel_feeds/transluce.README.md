# transluce.py — Transluce public findings

## Purpose
Fetch the public Transluce findings list, if one is served.

## Source
- Default list URL: `https://transluce.org/investigations` — Transluce's
  public findings page. Pass `--list-url` to probe another URL.
- No tracker-API endpoint is invented. No login, no session, no POST.

## Actual window — READ FIRST
Verified live 2026-10-31: Transluce publishes its findings as HTML at
`https://transluce.org/investigations`, and no public JSON
findings-list endpoint exists (`/investigations.json`,
`/api/findings`, `/findings.json`, `/rss.xml`, `/feed.xml` all
returned HTTP 404). The default list URL therefore serves HTML, not
a JSON list: **this scanner exits 2 (degraded) and emits no records
rather than fabricating findings.** That is the honest current state,
not a bug to work around by guessing endpoints.

If a JSON list is later served at the list URL, the scanner handles
it without code changes:
- If findings carry a usable date field (`created`, `modified`,
  `date`, `published`, `updated`), records are filtered to
  `--window-hours` (default 24) and stamped with that window.
- If no usable dates exist, the latest N (`--limit`, default 50) are
  emitted and every record stamps `window` as `latest-N` — never 24h.

## Env vars
None.

## Usage
```sh
uv run transluce.py --out-dir ./out --window-hours 24 --limit 50
```

## Output schema (JSONL, one object per finding, when a list is served)
`source` (`"transluce"`), `fetched_at` (UTC ISO-8601), `window`,
`provenance` (URL queried), `finding_id`, `title`, `date`, `url`,
`summary`.

## Limits
- Depends entirely on the list URL returning a JSON list. HTML (e.g.
  the public investigations page) is treated as degraded, not parsed
  for findings.
- Individual finding pages are not scraped: without a list there is no
  honest way to enumerate them.

## What "degraded" means
Exit 2, no records: list URL non-200, non-JSON, or JSON without a
findings-list shape. Exit 1: transport error. Exit 0: records written.
