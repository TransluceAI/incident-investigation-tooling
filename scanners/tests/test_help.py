"""Smoke test: every scanner's ``--help`` exits 0 and prints usage.

Runs each scanner with the current Python interpreter (all scanners are
stdlib-only, so this is the same entry point ``uv run`` executes; the
``uv run`` variant of this smoke lives in ``run_tests.sh``).

Google Python Style Guide: docstrings, type hints, no wildcard imports,
no bare except.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCANNERS_DIR = Path(__file__).resolve().parent.parent / "scanners"


def scanner_files() -> list[Path]:
    """Return every scanner file, sorted for stable test ids.

    Returns:
        Sorted list of ``*.py`` files under ``scanners/``.
    """
    return sorted(SCANNERS_DIR.rglob("*.py"))


@pytest.mark.parametrize(
    "scanner", scanner_files(), ids=lambda p: str(p.relative_to(SCANNERS_DIR))
)
def test_help_exits_zero(scanner: Path) -> None:
    """Scanner ``--help`` exits 0 and prints argparse usage.

    Args:
        scanner: Path to one scanner file.
    """
    result = subprocess.run(
        [sys.executable, str(scanner), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_scanner_count() -> None:
    """The suite covers exactly the 26 expected scanners."""
    assert len(scanner_files()) == 26
