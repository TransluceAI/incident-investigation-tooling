# Scanner index

26 scanners. Status values: **ready** (passes its tests), **degraded**
(ships with a stated limitation; see the notes below the table). Window
values are the windows actually applied, not just the windows
requested. Every scanner writes JSON Lines with `source`, `fetched_at`,
`window`, and `provenance`.

Run any scanner with `uv run <path> --out-dir ./out --window-hours 24`.
Every scanner accepts `--help`. Output directory precedence:
`--out-dir`, then `SCANNER_OUT_DIR`, then `./out`. Exit codes: 0 ok,
2 partial/degraded, 1 error.

## Package registries

| Scanner | Source | Actual window | Auth | Status |
|---|---|---|---|---|
| `scanners/package_registries/pypi.py` | PyPI RSS (updates, packages) | Requested window over the feed; feed length is the coverage limit | None | ready |
| `scanners/package_registries/rubygems.py` | RubyGems latest + just_updated APIs | Requested window over returned entries; API cap is the limit | None | ready |
| `scanners/package_registries/npm.py` | npm replicate `_changes` + registry metadata | Requested window over metadata time | None | ready |
| `scanners/package_registries/huggingface.py` | HuggingFace Hub models + datasets APIs | Requested window over latest 50 entries; busy days cover less | `HF_TOKEN` optional | ready |
| `scanners/package_registries/go_proxy.py` | Go module proxy `@latest` | Requested window, reported per record as `in_window` (default emits all, flagged) | None | ready |

## Agent boards

| Scanner | Source | Actual window | Auth | Status |
|---|---|---|---|---|
| `scanners/agent_boards/agent_boards.py` | 45 agent-board targets: llms.txt, skill.md, mcp.json, and related public endpoints (39 established + 6 candidate) | `current-state` (endpoint state, not a time window); 404/500/timeout recorded as state | None | ready |

## Intel feeds

| Scanner | Source | Actual window | Auth | Status |
|---|---|---|---|---|
| `scanners/intel_feeds/urlquery.py` | urlquery.net home + search | Requested window, client-side on row timestamps; unparseable rows kept with raw timestamp (disclosed) | None | ready |
| `scanners/intel_feeds/urlscan.py` | urlscan.io search API | Requested window, client-side; unparseable rows kept (disclosed) | `URLSCAN_API_KEY` optional | ready |
| `scanners/intel_feeds/crtsh.py` | crt.sh certificate search | Requested window on `not_before`; unparseable rows kept with raw value (disclosed) | None | ready |
| `scanners/intel_feeds/wayback_cdx.py` | Wayback CDX API (HTTPS) | Requested window, server-side `from`/`to` on capture time | None | ready |
| `scanners/intel_feeds/github_search.py` | GitHub issues + commits search | Server-side day-rounded qualifier plus client-side exact window on item timestamps | `GITHUB_TOKEN` optional | ready |
| `scanners/intel_feeds/hn_algolia.py` | Hacker News Algolia search by date | Requested window, server-side numeric filters | None | ready |
| `scanners/intel_feeds/reddit_public.py` | Reddit public JSON search | Requested window (when served) | None | **degraded** (environment: Reddit serves HTTP 403 to some unauthenticated clients; scanner exits 2 by design) |
| `scanners/intel_feeds/otx.py` | AlienVault OTX pulses | Requested window; unparseable `modified` dates kept (disclosed) | `OTX_API_KEY` required | ready |
| `scanners/intel_feeds/shodan.py` | Shodan host search API | Requested window on scan timestamps; undated matches stamped "undated"; key stripped from provenance URLs | `SHODAN_API_KEY` required | ready |
| `scanners/intel_feeds/threatfox.py` | abuse.ch ThreatFox IOC query | Whole API days (1–7; default 24h = 1 day), records further filtered by first_seen | `ABUSECH_API_KEY` required | **degraded** (POST-only API; read-only query, documented exception — see below) |
| `scanners/intel_feeds/urlhaus.py` | abuse.ch URLhaus recent URLs | date_added filter over the ~3-day feed; unparseable entries stamped "undated" | `ABUSECH_API_KEY` required | ready |
| `scanners/intel_feeds/transluce.py` | Transluce investigations page (transluce.org; HTML, no public JSON list) | `latest-N` fallback | None | **degraded** (no verified public list endpoint; exits 2, zero records, no fabrication) |
| `scanners/intel_feeds/commoncrawl.py` | Common Crawl index (collinfo + index) | Actual crawl window from collinfo (e.g. CC-MAIN-2026-43), never 24h | None | ready |
| `scanners/intel_feeds/mediawiki.py` | MediaWiki recentchanges (en.wikipedia, publictestwiki, collusion.wiki) | Requested window, server-side rcstart/rcend | None | ready |
| `scanners/intel_feeds/intel_blogs.py` | Norforge + swarmcha.se blogs (Atom/HTML) | Requested window for dated posts; undated posts stamped `snapshot` | None | ready |
| `scanners/intel_feeds/vendor_intel.py` | OpenAI news/index, Anthropic research | Dated findings window-filtered; undated stamped `undated`, never reported as in-window | None | ready |

## Relay watch

| Scanner | Source | Actual window | Auth | Status |
|---|---|---|---|---|
| `scanners/relay_watch/mail_tm.py` | mail.tm messages API | Requested window; unparseable `createdAt` kept with raw value (disclosed) | `MAIL_TM_TOKEN` required (operator-supplied JWT) | ready |
| `scanners/relay_watch/ntfy_watch.py` | ntfy.sh topic JSON polls | `window_requested` plus server-cache limit (ntfy retains ~12h on the free tier) | `NTFY_TOPICS` topic list | **degraded** (environment: some networks block ntfy.sh:443; scanner exits 2 per topic there) |
| `scanners/relay_watch/paste_watch.py` | Paste-host listing pages (22 hosts probed) | `current-state` (listing state, not a time window) | None | ready |
| `scanners/relay_watch/webhook_watch.py` | webhook.site token + request APIs | Token listings `current-state`; request records the hour window | `WEBHOOKSITE_TOKEN` required for listing | ready |

## Degraded scanners: what "degraded" means here

- **transluce:** no public JSON findings-list endpoint exists — the
  investigations page at transluce.org is HTML and the probed JSON and
  feed variants all return 404. The scanner's default `--list-url`
  points at the public page; it exits 2 with zero records rather than
  fabricating, keeping the gap visible instead of silently dropping it.
- **threatfox:** the abuse.ch API accepts only POST for this query;
  no GET endpoint exists for it. The POST is a read-only query (no
  submission, login, or session) and is documented in the scanner
  README. If strict GET-only is ever required, retire this scanner
  rather than using it.
- **reddit_public:** the endpoint is real and documented, but Reddit
  serves HTTP 403 block pages to some unauthenticated clients and
  networks. The scanner exits 2 by design and works unchanged where
  Reddit serves JSON.
- **ntfy_watch:** the endpoint format is the documented ntfy subscribe
  API, but some networks block ntfy.sh:443. The scanner exits 2 per
  topic on such networks.

## Test results

Suite: `tests/` via `./run_tests.sh` (ruff lint and format check, then
pytest, mocked HTTP only, no live network, then a `--help` smoke test
per scanner). Final run: **150 passed, 0 failed.**

- Happy-path tests per scanner pinning exact record counts, field
  extraction, timestamp normalization, and window filtering with a
  fresh plus 30-day-stale sibling item (the stale sibling must never
  become a record); server-side-window scanners get request-URL
  assertions proving the window bounds are actually sent.
- Parametrized contract tests across the scanners: HTTP 429 degraded
  behavior (exit 2, no traceback), transport failure with retries
  exhausted (exit 1 or 2 per the scanner's contract), keyless degraded
  paths that never touch the network, empty results, and unparseable
  payloads.
- Output-directory precedence: `--out-dir` > `SCANNER_OUT_DIR` > `./out`.
- Help smoke for all 26 scanners plus a scanner-count guard.

webhook_watch token listings stamp `"window": "current-state"` rather
than a time window; the suite covers this.

## Coverage

Four families: package registries (5 scanners), agent boards (1),
intel feeds (16), and relay watch (4). `agent_boards` checks 45
agent-board surfaces (39 established + 6 candidate); `paste_watch`
lists paste activity across 22 hosts; the remaining scanners each
cover a single source. Five scanners cover general sources rather than
single surfaces: pypi, npm, hn_algolia, reddit_public, ntfy_watch.
Sources that are utilities rather than activity feeds (retrieval
proxies, shorteners, static reports), or that require login sessions
or unverifiable hosts, are deliberately not scanned.

## Secrets audit

Zero hardcoded secrets. Every key comes only from environment
variables (see README.md for the table). The only secret-looking
literals in the tree are obviously fake placeholders in README
examples (`<your api key>`, `<operator-supplied JWT>`, `...`
ellipses) and documented example query strings, none of which are
credentials. No key is interpolated into any output or provenance
record (Shodan builds a key-stripped provenance URL; all others send
keys header-only).
