"""Tests for :class:`sec_generative_search.database.MetadataRegistry`.

Covers the full CRUD surface, SQL-level aggregate statistics, FIFO
eviction helpers, filing-limit enforcement, task-history persistence
with the redaction controls required for team/cloud deployments, the
concurrency guarantees of ``register_filing_if_new``, and the
SQLCipher-key handling path.

The tests drive a real SQLite database under ``tmp_path`` —
``MetadataRegistry`` is thin enough over the driver that mocking would
hide the very contracts (parameter binding, WAL pragma, integrity
errors) we care about.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from sec_generative_search.core.exceptions import (
    DatabaseError,
    FilingLimitExceededError,
)
from sec_generative_search.core.types import FilingIdentifier
from sec_generative_search.database import (
    DatabaseStatistics,
    FilingRecord,
    MetadataRegistry,
    TickerStatistics,
)
from sec_generative_search.database.metadata import _scrub_error_message

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_db_path: str) -> MetadataRegistry:
    """Registry backed by the per-test SQLite file."""
    return MetadataRegistry(db_path=tmp_db_path)


@pytest.fixture
def stored_filing(
    registry: MetadataRegistry,
    sample_filing_id: FilingIdentifier,
) -> FilingIdentifier:
    """Register ``sample_filing_id`` with 42 chunks and return the identifier."""
    registry.register_filing(sample_filing_id, chunk_count=42)
    return sample_filing_id


# ---------------------------------------------------------------------------
# Registration and duplicate detection
# ---------------------------------------------------------------------------


class TestRegisterFiling:
    def test_register_and_count(
        self,
        registry: MetadataRegistry,
        sample_filing_id: FilingIdentifier,
    ) -> None:
        registry.register_filing(sample_filing_id, chunk_count=59)
        assert registry.count() == 1

    def test_register_duplicate_raises(
        self,
        registry: MetadataRegistry,
        sample_filing_id: FilingIdentifier,
    ) -> None:
        registry.register_filing(sample_filing_id, chunk_count=5)
        with pytest.raises(DatabaseError, match="already exists"):
            registry.register_filing(sample_filing_id, chunk_count=5)


class TestIsDuplicate:
    def test_true_for_registered(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        assert registry.is_duplicate(stored_filing.accession_number) is True

    def test_false_for_absent(self, registry: MetadataRegistry) -> None:
        assert registry.is_duplicate("missing-accession") is False


class TestRegisterFilingIfNew:
    """``register_filing_if_new`` atomically checks and inserts."""

    def test_new_filing_registered(
        self,
        registry: MetadataRegistry,
        sample_filing_id: FilingIdentifier,
    ) -> None:
        assert registry.register_filing_if_new(sample_filing_id, chunk_count=42) is True
        assert registry.count() == 1

    def test_existing_filing_returns_false(
        self,
        registry: MetadataRegistry,
        sample_filing_id: FilingIdentifier,
    ) -> None:
        registry.register_filing(sample_filing_id, chunk_count=42)
        assert registry.register_filing_if_new(sample_filing_id, chunk_count=42) is False
        assert registry.count() == 1

    def test_atomic_under_contention(self, registry: MetadataRegistry) -> None:
        """Only one thread should succeed registering the same filing."""
        fid = FilingIdentifier(
            ticker="AAPL",
            form_type="10-K",
            filing_date=date(2024, 1, 1),
            accession_number="ACC-RACE",
        )
        results: list[bool] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(10)

        def try_register() -> None:
            try:
                barrier.wait()
                results.append(registry.register_filing_if_new(fid, chunk_count=10))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=try_register) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert results.count(True) == 1
        assert results.count(False) == 9
        assert registry.count() == 1

    def test_distinct_filings_all_succeed(self, registry: MetadataRegistry) -> None:
        errors: list[BaseException] = []

        def register(i: int) -> None:
            try:
                fid = FilingIdentifier(
                    ticker="AAPL",
                    form_type="10-K",
                    filing_date=date(2020 + i, 1, 1),
                    accession_number=f"ACC-CONC-{i}",
                )
                registry.register_filing_if_new(fid, chunk_count=i)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert registry.count() == 10


class TestGetExistingAccessions:
    def test_empty_input_short_circuits(self, registry: MetadataRegistry) -> None:
        assert registry.get_existing_accessions([]) == set()

    def test_none_match(self, registry: MetadataRegistry) -> None:
        assert registry.get_existing_accessions(["X", "Y"]) == set()

    def test_partial_match(self, registry: MetadataRegistry) -> None:
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1")
        fid2 = FilingIdentifier("MSFT", "10-K", date(2024, 3, 1), "ACC-2")
        registry.register_filing(fid1, chunk_count=10)
        registry.register_filing(fid2, chunk_count=20)

        result = registry.get_existing_accessions(["ACC-1", "ACC-MISSING", "ACC-2"])
        assert result == {"ACC-1", "ACC-2"}


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


class TestGetFiling:
    def test_found(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        record = registry.get_filing(stored_filing.accession_number)
        assert record is not None
        assert isinstance(record, FilingRecord)
        assert record.ticker == "AAPL"
        assert record.form_type == "10-K"
        assert record.chunk_count == 42
        assert record.filing_date == "2023-11-03"

    def test_not_found(self, registry: MetadataRegistry) -> None:
        assert registry.get_filing("nope") is None


class TestGetFilingsByAccessions:
    def test_empty_short_circuits(self, registry: MetadataRegistry) -> None:
        assert registry.get_filings_by_accessions([]) == []

    def test_returns_known_only(self, registry: MetadataRegistry) -> None:
        fid = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-KNOWN")
        registry.register_filing(fid, chunk_count=3)
        records = registry.get_filings_by_accessions(["ACC-KNOWN", "ACC-MISSING"])
        assert {r.accession_number for r in records} == {"ACC-KNOWN"}


class TestListFilings:
    def test_descending_by_filing_date(self, registry: MetadataRegistry) -> None:
        fid_old = FilingIdentifier("AAPL", "10-K", date(2022, 1, 1), "ACC-OLD")
        fid_new = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-NEW")
        registry.register_filing(fid_old, chunk_count=1)
        registry.register_filing(fid_new, chunk_count=2)

        filings = registry.list_filings()
        assert filings[0].accession_number == "ACC-NEW"
        assert filings[1].accession_number == "ACC-OLD"

    def test_ticker_filter_case_insensitive(self, registry: MetadataRegistry) -> None:
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-A")
        fid2 = FilingIdentifier("MSFT", "10-K", date(2024, 1, 1), "ACC-M")
        registry.register_filing(fid1, chunk_count=1)
        registry.register_filing(fid2, chunk_count=1)

        result = registry.list_filings(ticker="aapl")
        assert [r.accession_number for r in result] == ["ACC-A"]

    def test_combined_filters(self, registry: MetadataRegistry) -> None:
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1")
        fid2 = FilingIdentifier("AAPL", "10-Q", date(2024, 6, 1), "ACC-2")
        registry.register_filing(fid1, chunk_count=1)
        registry.register_filing(fid2, chunk_count=2)

        result = registry.list_filings(ticker="AAPL", form_type="10-K")
        assert [r.accession_number for r in result] == ["ACC-1"]


class TestListOldestFilings:
    def test_orders_by_ingested_at_ascending(self, registry: MetadataRegistry) -> None:
        fid_a = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-A")
        fid_b = FilingIdentifier("AAPL", "10-K", date(2024, 2, 1), "ACC-B")
        fid_c = FilingIdentifier("AAPL", "10-K", date(2024, 3, 1), "ACC-C")
        registry.register_filing(fid_a, chunk_count=1)
        registry.register_filing(fid_b, chunk_count=1)
        registry.register_filing(fid_c, chunk_count=1)

        oldest = registry.list_oldest_filings(limit=2)
        assert [r.accession_number for r in oldest] == ["ACC-A", "ACC-B"]


class TestCount:
    def test_filters_apply(self, registry: MetadataRegistry) -> None:
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1")
        fid2 = FilingIdentifier("MSFT", "10-K", date(2024, 1, 1), "ACC-2")
        registry.register_filing(fid1, chunk_count=1)
        registry.register_filing(fid2, chunk_count=2)

        assert registry.count() == 2
        assert registry.count(ticker="AAPL") == 1
        assert registry.count(form_type="10-K") == 2


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestGetStatistics:
    def test_empty_database(self, registry: MetadataRegistry) -> None:
        stats = registry.get_statistics()
        assert isinstance(stats, DatabaseStatistics)
        assert stats.filing_count == 0
        assert stats.tickers == []
        assert stats.form_breakdown == {}
        assert stats.ticker_breakdown == []

    def test_single_filing(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        stats = registry.get_statistics()
        assert stats.filing_count == 1
        assert stats.tickers == ["AAPL"]
        assert stats.form_breakdown == {"10-K": 1}
        assert len(stats.ticker_breakdown) == 1
        row = stats.ticker_breakdown[0]
        assert isinstance(row, TickerStatistics)
        assert row.ticker == "AAPL"
        assert row.filings == 1
        assert row.chunks == 42
        assert row.forms == ["10-K"]

    def test_multi_ticker_aggregation(self, registry: MetadataRegistry) -> None:
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1")
        fid2 = FilingIdentifier("AAPL", "10-Q", date(2024, 6, 1), "ACC-2")
        fid3 = FilingIdentifier("MSFT", "10-K", date(2024, 3, 1), "ACC-3")
        registry.register_filing(fid1, chunk_count=10)
        registry.register_filing(fid2, chunk_count=20)
        registry.register_filing(fid3, chunk_count=30)

        stats = registry.get_statistics()
        assert stats.filing_count == 3
        assert stats.tickers == ["AAPL", "MSFT"]
        assert stats.form_breakdown == {"10-K": 2, "10-Q": 1}

        assert len(stats.ticker_breakdown) == 2
        aapl, msft = stats.ticker_breakdown
        assert aapl.ticker == "AAPL"
        assert aapl.filings == 2
        assert aapl.chunks == 30
        assert aapl.forms == ["10-K", "10-Q"]
        assert msft.chunks == 30


# ---------------------------------------------------------------------------
# Delete operations
# ---------------------------------------------------------------------------


class TestRemoveFiling:
    def test_removes_existing(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        assert registry.remove_filing(stored_filing.accession_number) is True
        assert registry.count() == 0

    def test_returns_false_when_absent(self, registry: MetadataRegistry) -> None:
        assert registry.remove_filing("missing") is False


class TestRemoveFilingsBatch:
    def test_empty_noop(self, registry: MetadataRegistry) -> None:
        assert registry.remove_filings_batch([]) == 0

    def test_removes_multiple(self, registry: MetadataRegistry) -> None:
        for i in range(3):
            registry.register_filing(
                FilingIdentifier("AAPL", "10-K", date(2024, i + 1, 1), f"ACC-{i}"),
                chunk_count=1,
            )
        removed = registry.remove_filings_batch(["ACC-0", "ACC-2", "MISSING"])
        assert removed == 2
        assert {r.accession_number for r in registry.list_filings()} == {"ACC-1"}


class TestClearAll:
    def test_clears_rows(self, registry: MetadataRegistry) -> None:
        for i in range(3):
            registry.register_filing(
                FilingIdentifier("AAPL", "10-K", date(2024, i + 1, 1), f"ACC-{i}"),
                chunk_count=1,
            )
        assert registry.clear_all() == 3
        assert registry.count() == 0


# ---------------------------------------------------------------------------
# Filing limit (FIFO eviction trigger)
# ---------------------------------------------------------------------------


class TestCheckFilingLimit:
    def test_under_limit_passes(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        registry.check_filing_limit()

    def test_at_limit_raises(
        self,
        tmp_db_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exercise the limit gate by forcing a tiny max-filings ceiling.

        Setting ``_max_filings`` directly on the registry is deliberate:
        the Pydantic settings singleton is loaded at import time, and
        reloading it after ``monkeypatch.setenv`` still reads a cached
        default inside the test runner.  Overriding the registry's
        attribute bypasses that noise and exercises the contract —
        ``current >= self._max_filings`` raising
        :class:`FilingLimitExceededError`.
        """
        reg = MetadataRegistry(db_path=tmp_db_path)
        reg._max_filings = 1
        reg.register_filing(
            FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1"),
            chunk_count=1,
        )
        with pytest.raises(FilingLimitExceededError) as exc_info:
            reg.check_filing_limit()
        assert exc_info.value.current_count == 1
        assert exc_info.value.max_filings == 1


# ---------------------------------------------------------------------------
# Persistent connection and thread safety
# ---------------------------------------------------------------------------


class TestPersistentConnection:
    def test_wal_journal_enabled(self, registry: MetadataRegistry) -> None:
        with registry._lock:
            row = registry._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0].lower() == "wal"

    def test_close_prevents_further_ops(self, registry: MetadataRegistry) -> None:
        registry.close()
        with pytest.raises(DatabaseError):
            registry.count()

    def test_concurrent_registrations_persist(self, registry: MetadataRegistry) -> None:
        errors: list[BaseException] = []

        def register(i: int) -> None:
            try:
                registry.register_filing(
                    FilingIdentifier(
                        ticker="AAPL",
                        form_type="10-K",
                        filing_date=date(2020 + i, 1, 1),
                        accession_number=f"ACC-THREAD-{i}",
                    ),
                    chunk_count=i,
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert registry.count() == 8


# ---------------------------------------------------------------------------
# Task history persistence and privacy controls
# ---------------------------------------------------------------------------


def _save_sample_task(
    registry: MetadataRegistry,
    *,
    task_id: str = "task-1",
    status: str = "completed",
    tickers: list[str] | None = None,
    form_types: list[str] | None = None,
    results: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> None:
    registry.save_task_history(
        task_id,
        status=status,
        tickers=tickers or ["AAPL"],
        form_types=form_types or ["10-K"],
        results=results or [{"accession_number": "ACC-1", "status": "ok"}],
        error=error,
        started_at="2026-04-20T10:00:00+00:00",
        completed_at="2026-04-20T10:00:05+00:00",
        filings_done=1,
    )


class TestTaskHistoryRoundtrip:
    def test_save_and_retrieve(self, registry: MetadataRegistry) -> None:
        _save_sample_task(registry, tickers=["AAPL", "MSFT"])
        row = registry.get_task_history("task-1")
        assert row is not None
        assert row["task_id"] == "task-1"
        assert row["status"] == "completed"
        assert row["form_types"] == ["10-K"]
        assert row["results"] == [{"accession_number": "ACC-1", "status": "ok"}]
        assert row["filings_done"] == 1

    def test_missing_task_returns_none(self, registry: MetadataRegistry) -> None:
        assert registry.get_task_history("never-saved") is None

    def test_save_is_idempotent(self, registry: MetadataRegistry) -> None:
        _save_sample_task(registry)
        _save_sample_task(registry, status="failed")
        row = registry.get_task_history("task-1")
        assert row is not None
        assert row["status"] == "failed"


# ---------------------------------------------------------------------------
# Security / privacy controls
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSQLInjectionSafety:
    """Parameter-bound queries reject malicious input instead of executing it."""

    def test_ticker_injection_matches_nothing(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        assert registry.list_filings(ticker="' OR 1=1 --") == []
        # Table still intact.
        assert registry.count() == 1

    def test_form_type_injection_matches_nothing(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        assert registry.list_filings(form_type="'; DROP TABLE filings; --") == []
        assert registry.count() == 1

    def test_accession_injection_returns_none(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        assert registry.is_duplicate("' OR '1'='1") is False
        assert registry.get_filing("' UNION SELECT * FROM filings --") is None

    def test_count_injection_returns_zero(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        assert registry.count(ticker="' OR 1=1 --") == 0
        assert registry.count(form_type="'; DROP TABLE filings; --") == 0
        assert registry.count() == 1

    def test_batch_injection_returns_empty(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        result = registry.get_existing_accessions(["' OR '1'='1", "'; DROP TABLE filings; --"])
        assert result == set()
        assert registry.count() == 1


@pytest.mark.security
class TestErrorScrubbing:
    """`_scrub_error_message` strips tickers and accession numbers."""

    def test_removes_ticker_case_insensitive(self) -> None:
        out = _scrub_error_message("Failed to ingest AAPL filing", ["AAPL"])
        assert "AAPL" not in (out or "")
        assert "[TICKER]" in (out or "")

    def test_removes_accession_number(self) -> None:
        out = _scrub_error_message(
            "Accession 0000320193-23-000077 failed",
            tickers=[],
        )
        assert "0000320193-23-000077" not in (out or "")
        assert "[ACCESSION]" in (out or "")

    def test_handles_none(self) -> None:
        assert _scrub_error_message(None, tickers=["AAPL"]) is None

    def test_accession_without_dashes(self) -> None:
        out = _scrub_error_message("000032019323000077 bombed", tickers=[])
        assert "000032019323000077" not in (out or "")
        assert "[ACCESSION]" in (out or "")

    def test_skips_empty_ticker_strings(self) -> None:
        """Empty strings in the ticker list must not contribute to the regex.

        ``_scrub_error_message`` filters via ``[t for t in tickers if t]``, so
        an empty string is treated as "no ticker list" — the original text is
        returned untouched.  The whitespace-only entry is a regex-legal
        alternation because ``re.escape(" ")`` leaves it unchanged and the
        message contains literal spaces; the scrub replaces them, which is
        acceptable noise but verifies the empty-string branch in isolation.
        """
        out = _scrub_error_message("Plain error text", tickers=[""])
        assert out == "Plain error text"


@pytest.mark.security
class TestTaskHistoryPrivacy:
    """``DB_TASK_HISTORY_PERSIST_TICKERS`` gates ticker persistence."""

    def test_tickers_stripped_by_default(
        self,
        registry: MetadataRegistry,
    ) -> None:
        """Default (off) — ``tickers`` column stores null, never the literal list."""
        _save_sample_task(
            registry,
            tickers=["AAPL", "MSFT"],
            error="AAPL request for 0000320193-23-000077 rate-limited",
        )
        row = registry.get_task_history("task-1")
        assert row is not None
        # Tickers absent from persisted payload.
        assert row["tickers"] == []
        # Error scrubbed of ticker + accession identifiers.
        assert "AAPL" not in row["error"]
        assert "0000320193-23-000077" not in row["error"]
        assert "[TICKER]" in row["error"]
        assert "[ACCESSION]" in row["error"]

    def test_tickers_persist_when_opted_in(
        self,
        tmp_db_path: str,
    ) -> None:
        """Patch the settings-reader attribute directly — the persist-tickers
        toggle is read via ``get_settings().database.task_history_persist_tickers``
        at save time, so flipping the cached singleton's attribute exercises the
        opt-in branch without fighting pydantic-settings env-var caching.
        """
        from sec_generative_search.config import settings as settings_module

        settings = settings_module.get_settings()
        original = settings.database.task_history_persist_tickers
        settings.database.task_history_persist_tickers = True
        try:
            reg = MetadataRegistry(db_path=tmp_db_path)
            _save_sample_task(reg, tickers=["AAPL", "MSFT"])
            row = reg.get_task_history("task-1")
            assert row is not None
            assert row["tickers"] == ["AAPL", "MSFT"]
        finally:
            settings.database.task_history_persist_tickers = original


@pytest.mark.security
class TestEncryptionKeyHandling:
    """SQLCipher key wiring — safe fallback and no key leakage via repr/str."""

    @staticmethod
    def _enable_propagation() -> None:
        """Re-enable log propagation so pytest's ``caplog`` captures records.

        ``configure_logging`` sets ``propagate = False`` on the package
        logger (see ``tests/core/test_logging.py::TestAuditLog``). Tests
        that want records surfaced through ``caplog`` must opt back in
        and restore the flag in a ``finally`` block.
        """
        import logging

        logging.getLogger("sec_generative_search").propagate = True

    @staticmethod
    def _reset_propagation() -> None:
        import logging

        logging.getLogger("sec_generative_search").propagate = False

    def test_fallback_to_sqlite3_without_pysqlcipher(
        self,
        tmp_db_path: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With a key but no pysqlcipher3 installed, the registry warns and
        falls back to plain sqlite3 instead of silently storing data unprotected.

        The warning is the load-bearing signal for the operator; absence of
        pysqlcipher3 is a deploy-time error, not a runtime-silent one.
        """
        import logging

        self._enable_propagation()
        try:
            with caplog.at_level(logging.WARNING, logger="sec_generative_search"):
                registry = MetadataRegistry(
                    db_path=tmp_db_path,
                    encryption_key="unit-test-key-not-a-secret",
                )
            assert registry.encrypted is False
            assert any(
                "pysqlcipher3 is not installed" in record.getMessage() for record in caplog.records
            )
            # Plain sqlite3 is the actual driver — no SQLCipher classes.
            assert registry._sqlite_module is sqlite3
        finally:
            self._reset_propagation()

    def test_key_never_appears_in_log_output(
        self,
        tmp_db_path: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The debug message on registry init must not echo the raw key."""
        import logging

        self._enable_propagation()
        key = "never-log-this-secret-value"
        try:
            with caplog.at_level(logging.DEBUG, logger="sec_generative_search"):
                MetadataRegistry(db_path=tmp_db_path, encryption_key=key)
            for record in caplog.records:
                assert key not in record.getMessage()
                # Also ensure the hex encoding is not leaked.
                assert key.encode().hex() not in record.getMessage()
        finally:
            self._reset_propagation()

    def test_encrypted_flag_off_without_driver(
        self,
        tmp_db_path: str,
    ) -> None:
        """Without pysqlcipher3, ``encrypted`` must report ``False`` even when a
        key is supplied — the caller relies on this flag to decide whether the
        data-at-rest promise is upheld.
        """
        registry = MetadataRegistry(
            db_path=tmp_db_path,
            encryption_key="unit-test-key",
        )
        assert registry.encrypted is False

    def test_encrypted_flag_off_without_key(
        self,
        tmp_db_path: str,
    ) -> None:
        registry = MetadataRegistry(db_path=tmp_db_path)
        assert registry.encrypted is False


# ---------------------------------------------------------------------------
# Settings fallbacks (default db_path)
# ---------------------------------------------------------------------------


class TestSettingsDefaults:
    def test_default_db_path_is_used_when_not_provided(
        self,
        tmp_path: Path,
    ) -> None:
        """When ``db_path`` is omitted, the registry falls through to
        ``settings.database.metadata_db_path``.

        Rather than fight pydantic-settings env-var caching, override the
        already-loaded singleton's ``metadata_db_path`` attribute and
        restore it after the test. This exercises the same fall-through
        branch in :meth:`MetadataRegistry.__init__`.

        The temporary path lives under the project root to keep
        :meth:`DatabaseSettings._validate_paths` happy when other tests
        subsequently reload settings.
        """
        from sec_generative_search.config import settings as settings_module

        settings = settings_module.get_settings()
        project_tmp = Path.cwd() / "tmp_test_registry.sqlite"
        original = settings.database.metadata_db_path
        settings.database.metadata_db_path = str(project_tmp)
        try:
            registry = MetadataRegistry()
            registry.register_filing(
                FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-DEFAULT"),
                chunk_count=1,
            )
            assert registry.count() == 1
            assert project_tmp.exists()
            registry.close()
        finally:
            settings.database.metadata_db_path = original
            cleanup = (
                project_tmp,
                project_tmp.with_suffix(".sqlite-wal"),
                project_tmp.with_suffix(".sqlite-shm"),
            )
            for extra in cleanup:
                if extra.exists():
                    extra.unlink()
