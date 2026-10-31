# hn_algolia.py — Hacker News Algolia API

## Purpose

Fetch recent Hacker News stories and comments, keyless.

## Source and verification (2026-10-31, live probe)

`https://hn.algolia.com/api/v1/search_by_date?tags=(story,comment)&numericFilters=created_at_i>...,created_at_i<=...&hitsPerPage=<n>`
— HTTP 200, JSON with `hits` (verified: story hits with `objectID`,
`title`, `url`, `author`, `created_at`, `created_at_i`, `points`).

## Actual window

The requested `--window-hours` window (default 24h), enforced by the
API itself via `numericFilters` on `created_at_i` (Unix seconds), so
the emitted `window` is the window actually requested.

## Env vars

None. Keyless only.

## Usage

```sh
uv run hn_algolia.py --out-dir ./out --window-hours 24
uv run hn_algolia.py --query agent --tags story --limit 50
```

## Output schema (JSONL, one object per hit)

`source`, `fetched_at` (UTC ISO-8601), `window`, `provenance` (URL
queried), `object_id`, `type`, `title`, `url`, `author`, `created_at`,
`created_at_i`, `points`, `num_comments`, `story_id`, `comment_text`.

## Limits

- Algolia `hitsPerPage` caps at 1,000; the default `--limit` is 50.
- An empty `--query` returns the most recent items across stories and
  comments; it is a recency feed, not a topic filter.

## Degraded

Exit 2 on HTTP 403/429 or a non-JSON block page. Exit 1 on transport
failure after retries.
