"""Shared fixtures for the database test package.

These fixtures isolate every test to a fresh SQLite file under
``tmp_path`` and provide a deterministic :class:`FilingIdentifier`
seed for registration-based tests.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from sec_generative_search.core.types import FilingIdentifier


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    """Isolated SQLite database path inside pytest's tmp directory.

    Each test receives a unique temporary directory, so databases never
    collide or persist between runs.
    """
    return str(tmp_path / "test_metadata.sqlite")


@pytest.fixture
def sample_filing_id() -> FilingIdentifier:
    """A realistic AAPL 10-K :class:`FilingIdentifier` with a synthetic accession.

    Ticker and form type are upper-cased by ``FilingIdentifier.__post_init__``
    — tests rely on that invariant, so the literal strings here mirror the
    canonical form.
    """
    return FilingIdentifier(
        ticker="AAPL",
        form_type="10-K",
        filing_date=date(2023, 11, 3),
        accession_number="0000320193-23-000077",
    )
