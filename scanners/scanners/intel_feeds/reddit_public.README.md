# reddit_public.py — public Reddit JSON search

## Purpose

Search recent public Reddit posts via the keyless JSON endpoint.

## Source and verification (2026-10-31, live probe)

`https://www.reddit.com/search.json?q=<query>&sort=new&limit=<n>` —
the documented public JSON search path. Reddit may serve **HTTP 403
with an HTML block page** (a ~190 KB body was observed) to
unauthenticated or automated clients. When Reddit serves JSON, the
scanner works unchanged: it parses `data.children[].data` and filters
client-side on `created_utc`. A 403/429 run means "Reddit refused",
not "no posts".

## Actual window

The requested `--window-hours` window (default 24h), enforced
client-side on `created_utc` when JSON is returned.

## Default query

`agent`. Pass `--query` for anything else.

## Env vars

None. Keyless only — no OAuth, no login, by design.

## Usage

```sh
uv run reddit_public.py --out-dir ./out --window-hours 24
uv run reddit_public.py --query "AI agents" --limit 50
```

## Output schema (JSONL, one object per post)

`source`, `fetched_at` (UTC ISO-8601), `window`, `provenance` (URL
queried), `post_id`, `title`, `permalink`, `post_url`, `subreddit`,
`author`, `created_utc`, `score`, `num_comments`.

## Limits

- Reddit blocks automated clients unpredictably (403/429); a degraded
  run means "Reddit refused", not "no posts".

## Degraded

Exit 2 on HTTP 403/429 or a non-JSON block page, with a clear message.
Never a crash, never a retry loop (at most one transient retry).
Exit 1 on transport failure after retries.
