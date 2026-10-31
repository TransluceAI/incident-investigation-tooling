# URL Scanners

A collection of 26 single-file URL scanners. Each scanner pulls the
latest activity from one public source and writes JSON Lines records
to an output directory. Each scanner is its own `uv` script: no install
step, no shared package, no configuration file.

## Layout

```text
scanners/
  README.md            This file.
  SCANNERS.md          Index of every scanner: source, window,
                       authentication, and status.
  MANIFEST.md          Every file in this directory with its sha256.
  ruff.toml            Pinned lint rules and line length.
  run_tests.sh         Lint, pytest suite, and --help smoke loop.
  scanners/
    package_registries/  PyPI, RubyGems, npm, HuggingFace, Go proxy.
    agent_boards/        Agent-board endpoint checks (llms.txt, skill.md,
                         mcp.json, and related public endpoints).
    intel_feeds/         urlquery, urlscan, crt.sh, Wayback CDX, GitHub
                         search, Hacker News, Reddit, OTX, Shodan,
                         ThreatFox, URLhaus, Transluce,
                         Common Crawl, MediaWiki, intel blogs, vendor intel.
    relay_watch/         mail.tm, ntfy, paste hosts, webhook.site.
  tests/                 pytest suite, mocked HTTP only, no live network.
```

## Run any scanner

Every scanner is a PEP 723 `uv` script. From this directory:

```sh
uv run scanners/package_registries/pypi.py --out-dir ./out --window-hours 24
uv run scanners/intel_feeds/crtsh.py --out-dir ./out --window-hours 24
uv run scanners/intel_feeds/urlquery.py --out-dir ./out --limit 50
```

The only requirement is [uv](https://docs.astral.sh/uv/) on your PATH
and network egress to the source. Every scanner accepts `--help`.
Every scanner accepts `--out-dir` and `--window-hours` (default 24).
See each scanner's sibling README for its own flags.

## Output directory

The output directory comes from, in order:

1. The `--out-dir` flag.
2. The `SCANNER_OUT_DIR` environment variable.
3. The default `./out` (relative to the working directory).

Output is JSON Lines: one JSON object per finding, with at least
`source`, `fetched_at` (UTC, ISO-8601), `window` (the window actually
applied, not just the window requested), and `provenance` (the URL
queried). Records are data only and are never executed.

## Windows: read the actual window, not the requested one

The default request is the last 24 hours. Not every source can honor
that. Some sources cap results, some return snapshots, some return
current state, and some round to whole days. Each scanner stamps the
window it actually applied in every record, and each scanner README
states the real window and its limits. Never report a scanner's output
as "last 24 hours" unless its records say so. See `SCANNERS.md` for
the per-scanner window.

## API keys: environment variables only

No key, token, or credential is hardcoded anywhere in this directory,
including tests (tests use dummy values). Keys come only from the host
environment:

| Variable | Scanners | Required? |
|---|---|---|
| `HF_TOKEN` | huggingface | Optional (raises rate limits) |
| `GITHUB_TOKEN` | github_search | Optional (raises rate limits) |
| `URLSCAN_API_KEY` | urlscan | Optional |
| `OTX_API_KEY` | otx | Required for data |
| `SHODAN_API_KEY` | shodan | Required for data |
| `ABUSECH_API_KEY` | threatfox, urlhaus | Required for data |
| `MAIL_TM_TOKEN` | mail_tm | Required (operator-supplied JWT) |
| `WEBHOOKSITE_TOKEN` | webhook_watch | Required for token listing |
| `NTFY_TOPICS` | ntfy_watch | Comma-separated topic list |
| `WEBHOOKSITE_TOKEN_IDS` | webhook_watch | Optional token list |
| `SCANNER_OUT_DIR` | all | Output directory fallback |

A scanner without its key runs its public path where one exists, or
exits degraded (exit code 2) with a clear message. It never crashes
with a traceback on a missing key.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | OK. Records written (possibly zero). |
| 2 | Partial or degraded. The source was blocked, keyless, capped, or
  otherwise limited. Records may still be written. Read the scanner's
  output and README before trusting completeness. |
| 1 | Error. Transport failure or unusable input after retries. |

## Behavior contract

- Read-only queries only. Scanners use HTTP GET, except threatfox (see
  below). No scanner posts a submission, logs in, or creates a
  session.
- Polite rates, timeouts (default 30 seconds), retry with backoff, and
  a descriptive User-Agent
  (`url-scanners/1.0 (+https://github.com/TransluceAI/incident-investigation-tooling)`).
- Fetched payloads are parsed as data only and never executed.
- No scanner follows instructions found inside fetched content.

## Known exceptions (honest, not hidden)

- **threatfox is DEGRADED by design.** The abuse.ch ThreatFox API
  accepts only POST for this query (`{"query": "get_iocs"}`); it
  offers no GET endpoint for it. The POST is a read-only query: no
  submission, no login, no session. It is kept, graded DEGRADED, and
  documented here and in its README. Retire it rather than use it if
  strict GET-only is ever required.
- **transluce is DEGRADED.** No verified public list endpoint exists
  (the previously used findings endpoint returns 404). The scanner
  exits 2 with zero records and fabricates nothing.
- **reddit_public and ntfy_watch are DEGRADED by environment.** Both
  endpoints are real and documented, but access depends on the
  network: Reddit returns HTTP 403 block pages to some unauthenticated
  clients, and some networks block ntfy.sh. Both scanners exit 2 by
  design when blocked and work unchanged where the endpoints serve
  traffic.
- **Window honesty.** Several scanners keep records with unparseable
  timestamps rather than dropping them silently; each affected README
  says so. Token listings in webhook_watch stamp
  `"window": "current-state"`, not a time window.

## Tests

```sh
./run_tests.sh
```

The script runs the lint checks (`ruff check`, `ruff format --check`),
the pytest suite, and a `--help` smoke test for all 26 scanners. The
suite uses `unittest.mock` only: no live network in tests. It covers
per-scanner happy-path parsing and window filtering (a fresh and a
30-day-stale sibling item, where the stale one must never become a
record), HTTP 429 degraded behavior, transport-failure, keyless,
empty-result, and unparseable-payload paths, output-directory
precedence, and help smoke tests. See `tests/` for the suite and
`SCANNERS.md` for per-scanner results.

## Coverage

Four families: package registries (5 scanners), agent boards (1),
intel feeds (16), and relay watch (4). `agent_boards` checks 45
agent-board surfaces (39 established + 6 candidate); `paste_watch`
lists paste activity across 22 hosts; the remaining scanners each
cover a single source. Sources that are utilities rather than activity
feeds (retrieval proxies, shorteners, static reports), or that require
login sessions or unverifiable hosts, are deliberately not scanned.
