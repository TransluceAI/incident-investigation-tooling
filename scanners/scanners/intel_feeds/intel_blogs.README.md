# intel_blogs.py — swarmcha.se + norforge.com (keyless)

## Purpose
Watch two independent intel blogs for new posts.

## Sources and per-site method (verified live 2026-10-31)
- **norforge.com — Atom feed.** The homepage advertises
  `<link rel="alternate" type="application/atom+xml"
  href="/blog/feed.xml">`, and that feed returns valid Atom
  (latest entry at probe time: 2026-10-26, "Automated Subversion and
  Attempted Takeover of AI Pentesting Harnesses"). Entries are
  parsed with the standard-library XML parser and filtered to the
  window by `published`.
- **swarmcha.se — homepage change-watch (NO feed).** Probed and all
  404: `/atom.xml`, `/feed`, `/rss.xml`, `/index.xml`, `/posts.xml`,
  `/atom`. Fallback: fetch `https://swarmcha.se/`, record the
  SHA-256 of the body, and extract the dated post list
  (`<time datetime="...">` + post link). Latest post at probe time:
  2026-10-14, "We found a Tencent agent fleet".

## Actual window
`--window-hours` (default 24) for dated post records. Every run also
emits one `site_snapshot` record per site (content hash, latest post
found, post count) stamped `window: "snapshot"` — so a quiet window
is an auditable observation, not silence. Snapshot records are not
posts and must not be counted as such.

## Env vars
None.

## Usage
```sh
uv run intel_blogs.py --out-dir ./out --window-hours 24
```

## Output schema (JSONL)
Post records: `source` (`"intel_blogs"`), `fetched_at` (UTC
ISO-8601), `window`, `provenance`, `kind` (`"post"`), `site`,
`title`, `url`, `date`.
Snapshot records: same `source`/`fetched_at`/`provenance`, `kind`
(`"site_snapshot"`), `site`, `method` (`"atom_feed"` or
`"homepage_change_watch"`), `content_sha256`, `latest_post_title`,
`latest_post_date`, `post_count`, `window` (`"snapshot"`).

## Limits
- swarmcha.se parsing depends on the homepage's post-list markup;
  if the markup changes, post extraction degrades to snapshots only
  (hash still recorded, so a change is still visible).
- Neither site is polled more than once per run.

## What "degraded" means
Exit 2: at least one site failed (transport error, non-200, or an
unparseable feed) — the other site's records are still written.
Exit 0: both sites succeeded. Exit 1: unexpected failure.
