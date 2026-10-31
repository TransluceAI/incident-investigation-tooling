# commoncrawl.py — Common Crawl index lookup (keyless)

## Purpose
Look a URL/domain up in the latest Common Crawl index and emit its
capture records.

## Source
- Index list: `https://index.commoncrawl.org/collinfo.json`
- Query: `https://index.commoncrawl.org/<index>-index?url=<url>&output=json`
- Both verified live 2026-10-31. Latest index at probe time:
  `CC-MAIN-2026-43` (October 2026 Index).

## Actual window — NOT 24h
Common Crawl indexes are periodic (roughly monthly), not a 24h feed.
Every record stamps the actual crawl window from collinfo, e.g.
`latest crawl index CC-MAIN-2026-43 (2026-10-05T10:56:23 to
2026-10-18T05:34:47)`, in both `window` and `crawl_window`, plus the
`crawl_index` name. `--window-hours` is accepted for CLI uniformity
but does NOT filter captures — a crawl index cannot answer a 24h
question, and claiming otherwise would be false.

## Env vars
None.

## Usage
```sh
uv run commoncrawl.py --out-dir ./out --url example.com --limit 20
```

## Output schema (JSONL, one object per capture)
`source` (`"commoncrawl"`), `fetched_at` (UTC ISO-8601), `window`
(actual crawl window, see above), `provenance` (query URL),
`crawl_index`, `crawl_window`, `queried_url`, `capture_url`,
`capture_time`, `mime`, `status`, `digest`, `warc_filename`.

## Limits
- The index API is intermittently slow: it returned HTTP 504 twice
  during probing, then HTTP 200 for the same style of query minutes
  later. The scanner retries with backoff and degrades (exit 2) if
  the query still fails.
- Captures are crawl-time snapshots, not live pages; WARC content
  itself is not downloaded by this scanner.

## What "degraded" means
Exit 2: the index query failed after retries (e.g. HTTP 504).
Exit 1: collinfo fetch failed, collinfo was not JSON, or no indexes
were listed. Exit 0: records written.
