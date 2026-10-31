# go_proxy.py — Go module proxy watchlist checker

## Purpose

Report the latest version of each Go module on an operator-supplied
watchlist, via the Go module proxy.

## Source

Keyless, no account or key:

- Per-module latest: `https://proxy.golang.org/<module>/@latest`
  (module paths escaped per the proxy protocol: uppercase letters
  become `!` + lowercase).

Verified live 2026-10-31: `github.com/spf13/cobra/@latest` returns
HTTP 200 with `Version`/`Time`/`Origin`; an unknown module returns
HTTP 404.

## Honest constraint — no global feed

**proxy.golang.org has no global "latest modules" feed.** This scanner
cannot discover new modules on its own and never pretends otherwise.
It checks only modules you name. With an empty watchlist it exits
degraded (2) with a message explaining why, and writes no records.

## Actual window

`--window-hours` (default 24) is compared against each module's latest
version time and reported per record in `in_window`. By default all
resolved modules are emitted with their true timestamp, so a stale
module is visible as `in_window: false` rather than silently dropped.
Pass `--only-in-window` to emit only modules whose latest version falls
in the window. The `window` field in each record is the requested
window.

## Env vars

None. No key is needed. (`SCANNER_OUT_DIR` is the shared out-dir
fallback, not a credential.)

## Usage

```sh
uv run go_proxy.py --modules github.com/spf13/cobra,golang.org/x/tools
uv run go_proxy.py --modules-file watchlist.txt --only-in-window --out-dir ./out
```

`watchlist.txt` holds one module path per line (`#` lines ignored).
`--out-dir` falls back to `SCANNER_OUT_DIR`, then `./out`.

## Output schema (one JSON object per line)

| Field | Meaning |
|---|---|
| `source` | `"proxy.golang.org"` |
| `fetched_at` | UTC ISO-8601 fetch time |
| `window` | Requested window, e.g. `"24h"` |
| `provenance` | `@latest` URL queried for the module |
| `module` | Module path as supplied |
| `latest_version`, `latest_version_time` | Proxy `Version`/`Time` |
| `in_window` | Whether the latest version time falls in the window |
| `origin_vcs`, `origin_url`, `origin_ref` | Proxy `Origin` fields |

## Limits

- Watchlist-only: no discovery, no enumeration, no global feed.
- Unknown modules return 404 and are counted in the stdout summary;
  they produce no record. A 404 is not retried — it is a definitive
  answer — while 5xx responses are retried with backoff and then
  counted the same way.
- Fetched JSON is parsed as data only, never executed.

## Degraded (exit 2)

Empty watchlist; any module not found or otherwise unavailable on the
proxy (records for resolved modules are still written); or a 403/429
during the watchlist. Exit 1 is a transport failure or an unreadable
`--modules-file`.
