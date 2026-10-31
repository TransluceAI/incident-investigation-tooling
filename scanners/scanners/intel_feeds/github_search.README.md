# github_search.py — GitHub search API

## Purpose

Search recent GitHub issues and commits, keyless by default.

## Source and verification (2026-10-31, live probe)

- `https://api.github.com/search/issues?q=<query>&per_page=<n>` — HTTP
  200, JSON with `items` (verified: `agent created:>2026-10-30`
  returned 153,180 total, items with `html_url`, `title`, `created_at`).
- `https://api.github.com/search/commits` — same shape; commit search
  uses the `committer-date:>` qualifier instead of `created:>`
  (verified 200). Issues are sorted `created` desc; commits use best
  match (the documented default) because sorting commits by
  `committer-date` surfaces far-future-dated fakes first.

## Actual window

The requested `--window-hours` window (default 24h). GitHub search
date qualifiers are day-granular only, so the query uses
`created:>`/`committer-date:>` at date granularity and the exact
window is then enforced client-side on each item's own timestamp
(`created_at` for issues, `commit.committer.date` for commits).
Items dated outside the window are dropped — including the
far-future-dated commits that dominate `committer-date`-sorted
results. An item without a parseable timestamp is kept, with its raw
fields in the record, rather than dropped silently.

## Default query

`agent`, restricted to items created in the window as above. This
default is a scanner default, not a finding — pass `--query` for
anything else.

## Env vars

- `GITHUB_TOKEN` (optional): raises the limit above the 60/hr
  unauthenticated bucket. Absent, the keyless path works.

## Usage

```sh
uv run github_search.py --out-dir ./out --window-hours 24
uv run github_search.py --query "AI agents" --endpoint issues --limit 30
```

## Output schema (JSONL, one object per item)

`source`, `fetched_at` (UTC ISO-8601), `window`, `provenance` (URL
queried), `endpoint` (`issues`/`commits`), `id`, `html_url`, `title`
(issues), `message` (commits), `created_at`, `updated_at`, `repository`,
`user`.

## Limits

- Day-granular date qualifiers (see "Actual window").
- Search API caps results at 1,000 per query.

## Degraded

Exit 2 on rate-limit exhaustion (an HTTP 403 carrying
`X-RateLimit-Remaining: 0`) or other 403/429/block pages. The scanner
stops cleanly and never loops on a limit. Exit 1 on transport failure.
