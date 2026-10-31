#!/usr/bin/env bash
# Quality gate for the url-scanners tool directory: ruff lint + format
# check, the mocked-HTTP pytest suite, then a --help smoke test per
# scanner (all scanners are stdlib-only PEP 723 uv scripts). Extra
# args after the script name are passed through to pytest.
set -euo pipefail
cd "$(dirname "$0")"

# Prefer uv from PATH; fall back to the default user install location.
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

# Run ruff via uvx when available, else via the resolved uv
# ("uvx" and "uv tool run" are the same command).
run_ruff() {
    if command -v uvx >/dev/null 2>&1; then
        uvx "$@"
    else
        "$UV" tool run "$@"
    fi
}

echo "=== ruff check ==="
run_ruff ruff check .

echo "=== ruff format --check ==="
run_ruff ruff format --check .

echo "=== pytest (via uv run) ==="
"$UV" run --with pytest pytest tests/ -q "$@"

echo "=== --help smoke (via uv run, one per scanner) ==="
fail=0
for scanner in scanners/*/*.py; do
    if "$UV" run "$scanner" --help >/dev/null 2>&1; then
        echo "OK   $scanner"
    else
        echo "FAIL $scanner"
        fail=1
    fi
done
if [ "$fail" -ne 0 ]; then
    echo "help smoke: FAILED" >&2
    exit 1
fi
echo "help smoke: all scanners passed"
