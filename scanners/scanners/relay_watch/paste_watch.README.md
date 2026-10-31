# paste_watch — Paste/temp-file host watch

## Purpose

Check whether each covered paste/temp-file host publishes a public
recent listing, and record what it found. This is a presence/listing
watch, not an activity feed, wherever feeds do not exist.

## Plain statement on feeds

**Most paste hosts publish NO recent-activity listing.** Their
homepages are upload forms. A run that reports `no_public_listing`
for most hosts is an honest result, not a broken run. Activity is
never fabricated: no timestamps are claimed for listed links, and no
host is reported as having a listing unless a candidate URL responded
200 and its page actually contained paste-like links during that run.

## Hosts

A curated default set of 22 unique public hosts (the `HOSTS` table in
`paste_watch.py`; the set is fixed in the scanner itself, not loaded
from any external file):

tmpfiles.org, anna.fyi, paste.linuxiarz.pl, pastebin.k4be.pl,
pastebin.tarcseh.me, paste.rs, nicepaste.com, paste.ubuntu.org.cn,
pastebox.to, infinitypaste.club, porkpad.com, pasteview.com,
gritpost.com, gotit.pub, mcbin.dev, bytebin.lucko.me,
bytebin.rkslot.nl, paste.ee, paste.ubuntu.com, rentry.co,
mdshare.org, friendpaste.com.

## How each host is probed

For each host, candidate listing URLs are probed live in order
(homepage first, then a host-appropriate `/archive` path where that is
a common pattern for pastebin-style hosts). Transport-level failures
are retried once with backoff; HTTP statuses (e.g. 403 or 404) are
the host's verdict and are not retried. Only a candidate that
responds 200 AND contains paste-like links (same-host links that are
not navigation, assets, or known non-paste pages) is reported as a
listing, with that URL recorded as `listing_url`.

Probe notes (2026-10-31, live verification): anna.fyi's real listing
is `https://anna.fyi/lists` (200, contains `/view/<id>` paste links);
its homepage only carries "Recent Pastes" / "Trending Pastes" tab
labels with no static paste links, so the homepage alone is not a
listing. `/archive` candidates returned 404 on several hosts
(paste.linuxiarz.pl, pastebin.k4be.pl, pastebin.tarcseh.me,
pastebox.to, infinitypaste.club) — a failed candidate is a normal
probe result, not an error. paste.ee's homepage returns 301
(redirects are followed). friendpaste.com returned 503 (the site
appears down). paste.ubuntu.com timed out during probing (slow, not
proven dead). porkpad.com failed DNS resolution. All other homepages
returned 200 but showed upload-form pages with no paste listing.

## Usage

```sh
export PATH="$HOME/.local/bin:$PATH"
uv run scanners/relay_watch/paste_watch.py --out-dir ./out
uv run scanners/relay_watch/paste_watch.py --limit 3 --out-dir ./out   # test
```

No key is needed. No env vars are read except `SCANNER_OUT_DIR`
(fallback when `--out-dir` is absent).

## Output

JSONL, one object per host per run, in
`<out-dir>/paste_watch-<UTC timestamp>.jsonl`:

| Field | Meaning |
| --- | --- |
| `source` | Host checked |
| `status` | `listing`, `no_public_listing`, or `unreachable` |
| `listing_url` | Responding listing URL, or null |
| `items_found` | Number of paste-like links found (0 unless `listing`) |
| `items` | Up to 20 paste-like link URLs (only when `listing`) |
| `error` | Last network error string, or null |
| `fetched_at` | Fetch time, UTC ISO-8601 |
| `window` | Always `"current-state"` — a listing snapshot, not a time feed |
| `provenance` | Candidate URLs probed, in order |

## Limits

- `items` are links found on a listing page, not verified paste
  contents and not timestamped activity. The scanner does not fetch
  the linked pastes themselves.
- Link detection is heuristic (same-host, non-navigation links). A
  listing page with an unusual layout may be reported as
  `no_public_listing`; that is a detection limit, not proof no
  listing exists.
- `--window-hours` is accepted for CLI uniformity only; listing
  checks are not time-filtered.

## What "degraded" means

Exit 0: at least one host responded (any status except all
unreachable). Exit 2 (degraded): every host was unreachable. Exit 1:
local error (output directory/file).
