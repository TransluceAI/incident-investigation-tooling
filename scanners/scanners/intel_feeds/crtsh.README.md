# crtsh.py — crt.sh certificate search

## Purpose

Find recently issued certificates for a curated set of watched
domains via Certificate Transparency logs, keyless.

## Source and verification (2026-10-31, live probe)

`https://crt.sh/?q=%25.<domain>&output=json` — confirmed working JSON
(fields `id`, `common_name`, `name_value`, `not_before`, `not_after`,
`issuer_name`). One probe returned a transient HTTP 502 from crt.sh's
front end; a retry seconds later succeeded. Treat 502s as transient.

## Actual window — read this honestly

This is an **issuance search, not a real-time feed**. The requested
`--window-hours` window (default 24h) is a client-side filter on
`not_before`: only certificates issued inside the window are emitted,
and the record's `window` field reports that filter. A certificate
whose `not_before` does not parse is kept, with its raw `not_before`
in the record, rather than dropped silently. Completeness is
**not** guaranteed: results are capped at `--limit` per domain (default
100, most recent first), and a domain with a huge history may have
window certs beyond the cap.

## Curated default domains (8)

`swarmsay.com`, `commons.dooza.ai`, `robots.wiki`, `agenthaven.org`,
`agentcolony.one`, `msgboard.dev`, `moltbook.com`, `commonlog.ai`.
These are curated defaults only — not a claim about which domains
matter. Override with `--domains` (comma list) for any other targets.

## Env vars

None. Keyless only.

## Usage

```sh
uv run crtsh.py --out-dir ./out --window-hours 24
uv run crtsh.py --domains example.com,example.org --limit 50
```

## Output schema (JSONL, one object per certificate)

`source`, `fetched_at` (UTC ISO-8601), `window`, `provenance` (URL
queried), `domain`, `cert_id`, `common_name`, `name_value`,
`not_before`, `not_after`, `issuer_name`.

## Limits

- crt.sh is slow and occasionally 502s; the scanner paces domains 1s
  apart and retries. A domain that fails after retries marks the run
  degraded (see below) rather than aborting the other domains.

## Degraded

Exit 2 if any domain ends degraded (403/429, non-JSON page, or transport
failure after retries) — including when other domains produced records.
Exit 0 only when every domain succeeded.
