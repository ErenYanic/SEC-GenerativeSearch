"""Tests for forward-only SQLite schema versioning.

The bootstrap path is exercised against three database states (brand
new, v1-shaped unstamped, malformed unstamped) by writing real SQLite
files and opening them through :class:`MetadataRegistry`.  The apply
path is exercised by injecting an ordered fake-migration tuple into
:func:`apply_pending_migrations` directly — the module-level
:data:`MIGRATIONS` is empty for v1 by design.

Both surfaces are tested without mocks: ``MetadataRegistry`` is thin
enough that a mocked SQLite would hide the parameter binding,
transaction boundary, and idempotent-CREATE contracts that this phase
exists to formalise.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.database import MetadataRegistry
from sec_generative_search.database.migrations import (
    MIGRATIONS,
    apply_pending_migrations,
    bootstrap_schema_version,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _v1_filings_ddl() -> str:
    """The exact v1 CREATE TABLE statement for ``filings``.

    Mirrors :meth:`MetadataRegistry._initialise_schema` so the test can
    seed a v1-shaped database without going through the registry first.
    """
    return """
        CREATE TABLE filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            form_type TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            accession_number TEXT NOT NULL UNIQUE,
            chunk_count INTEGER NOT NULL,
            ingested_at TEXT NOT NULL,
            UNIQUE(ticker, form_type, filing_date)
        )
    """


def _v1_task_history_ddl() -> str:
    """The exact v1 CREATE TABLE statement for ``task_history``."""
    return """
        CREATE TABLE task_history (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            tickers TEXT,
            form_types TEXT NOT NULL,
            results TEXT NOT NULL,
            error TEXT,
            started_at TEXT,
            completed_at TEXT,
            filings_done INTEGER NOT NULL DEFAULT 0,
            filings_skipped INTEGER NOT NULL DEFAULT 0,
            filings_failed INTEGER NOT NULL DEFAULT 0
        )
    """


def _seed_v1_unstamped(
    db_path: str,
    *,
    include_filings: bool = True,
    include_task_history: bool = True,
    extra_filing_rows: int = 0,
) -> None:
    """Create a SQLite file in the v1 shape with **no** ``schema_version``.

    Used to simulate databases created before the migration landed.  Each
    flag toggles one v1 table — flipping one to ``False`` produces the
    malformed-unstamped scenario.
    """
    conn = sqlite3.connect(db_path)
    try:
        if include_filings:
            conn.execute(_v1_filings_ddl())
        if include_task_history:
            conn.execute(_v1_task_history_ddl())
        for i in range(extra_filing_rows):
            # Vary filing_date so the UNIQUE(ticker, form_type, filing_date)
            # constraint does not collide across the seeded rows.
            conn.execute(
                "INSERT INTO filings (ticker, form_type, filing_date, "
                "accession_number, chunk_count, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "AAPL",
                    "10-K",
                    f"2023-{(i % 12) + 1:02d}-03",
                    f"0000320193-23-{i:06d}",
                    42,
                    "2023-11-04T00:00:00+00:00",
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _read_schema_versions(db_path: str) -> list[tuple[int, str]]:
    """Return every row in ``schema_version`` ordered by version."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT version, applied_at FROM schema_version ORDER BY version"
        ).fetchall()
    finally:
        conn.close()
    return rows


def _table_exists(db_path: str, name: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


# ---------------------------------------------------------------------------
# Bootstrap — brand new
# ---------------------------------------------------------------------------


class TestBootstrapBrandNew:
    """A brand-new SQLite file is stamped at v1 on first open."""

    def test_stamps_v1(self, tmp_db_path: str) -> None:
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.close()

        rows = _read_schema_versions(tmp_db_path)
        assert len(rows) == 1
        assert rows[0][0] == 1

    def test_creates_v1_tables(self, tmp_db_path: str) -> None:
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.close()

        assert _table_exists(tmp_db_path, "filings")
        assert _table_exists(tmp_db_path, "task_history")
        assert _table_exists(tmp_db_path, "schema_version")

    def test_idx_filings_ingested_at_present(self, tmp_db_path: str) -> None:
        """The secondary index survives the new init flow."""
        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.close()

        conn = sqlite3.connect(tmp_db_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_filings_ingested_at'"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None

    @pytest.mark.security
    def test_no_migration_body_run(self, tmp_db_path: str) -> None:
        """v1 is the baseline — nothing in :data:`MIGRATIONS` to execute.

        The empty tuple is the contract for "no body run".
        """
        assert MIGRATIONS == ()

        registry = MetadataRegistry(db_path=tmp_db_path)
        registry.close()

        rows = _read_schema_versions(tmp_db_path)
        assert [row[0] for row in rows] == [1]


# ---------------------------------------------------------------------------
# Bootstrap — v1-shaped unstamped (the lossless upgrade path)
# ---------------------------------------------------------------------------


class TestBootstrapV1Shaped:
    """An existing v1-shaped DB is stamped without losing any data."""

    @pytest.mark.security
    def test_existing_rows_survive_stamp(self, tmp_db_path: str) -> None:
        _seed_v1_unstamped(tmp_db_path, extra_filing_rows=3)

        registry = MetadataRegistry(db_path=tmp_db_path)
        try:
            assert registry.count() == 3
        finally:
            registry.close()

        assert [row[0] for row in _read_schema_versions(tmp_db_path)] == [1]

    def test_idempotent_reopen(self, tmp_db_path: str) -> None:
        """Two consecutive opens leave exactly one schema_version row."""
        _seed_v1_unstamped(tmp_db_path)

        MetadataRegistry(db_path=tmp_db_path).close()
        MetadataRegistry(db_path=tmp_db_path).close()

        rows = _read_schema_versions(tmp_db_path)
        assert len(rows) == 1
        assert rows[0][0] == 1


# ---------------------------------------------------------------------------
# Bootstrap — malformed unstamped (refuse loudly)
# ---------------------------------------------------------------------------


class TestBootstrapMalformed:
    """A partially-populated unstamped DB is refused with operator hints."""

    @pytest.mark.security
    def test_missing_task_history_refused(self, tmp_db_path: str) -> None:
        _seed_v1_unstamped(tmp_db_path, include_task_history=False)

        with pytest.raises(DatabaseError, match=r"task_history"):
            MetadataRegistry(db_path=tmp_db_path)

    @pytest.mark.security
    def test_missing_filings_refused(self, tmp_db_path: str) -> None:
        _seed_v1_unstamped(tmp_db_path, include_filings=False)

        with pytest.raises(DatabaseError, match=r"filings"):
            MetadataRegistry(db_path=tmp_db_path)

    def test_error_names_recovery_path(self, tmp_db_path: str) -> None:
        _seed_v1_unstamped(tmp_db_path, include_task_history=False)

        with pytest.raises(DatabaseError) as exc_info:
            MetadataRegistry(db_path=tmp_db_path)
        # The operator hint mentions a backup-driven recovery, not an
        # automated fallback — keeps the contract aligned with the
        # ReindexService rollback-out-of-scope-by-design ethos.
        assert "backup" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# clear_all preserves schema_version (it is schema, not data)
# ---------------------------------------------------------------------------


class TestClearAllPreservesSchemaVersion:
    @pytest.mark.security
    def test_clear_all_does_not_truncate_schema_version(
        self,
        tmp_db_path: str,
    ) -> None:
        registry = MetadataRegistry(db_path=tmp_db_path)
        try:
            registry.clear_all()
        finally:
            registry.close()

        rows = _read_schema_versions(tmp_db_path)
        assert len(rows) == 1
        assert rows[0][0] == 1


# ---------------------------------------------------------------------------
# apply_pending_migrations — injected fake migrations
# ---------------------------------------------------------------------------


class TestApplyPendingMigrations:
    """The apply path is exercised through the *migrations* test seam.

    :data:`MIGRATIONS` is empty for v1, so the apply surface is
    exercised by injecting fake migrations that record their execution
    on a real, stamped SQLite database.
    """

    def _open_stamped(self, tmp_db_path: str) -> Any:
        """Return a connection to a brand-new DB stamped at v1."""
        MetadataRegistry(db_path=tmp_db_path).close()
        return sqlite3.connect(tmp_db_path)

    def test_runs_pending_in_order(self, tmp_db_path: str) -> None:
        conn = self._open_stamped(tmp_db_path)
        order: list[int] = []

        def _v2(c: Any) -> None:
            order.append(2)

        def _v3(c: Any) -> None:
            order.append(3)

        try:
            apply_pending_migrations(conn, migrations=((2, _v2), (3, _v3)))
            conn.commit()
        finally:
            conn.close()

        assert order == [2, 3]
        assert [row[0] for row in _read_schema_versions(tmp_db_path)] == [1, 2, 3]

    def test_skips_already_applied(self, tmp_db_path: str) -> None:
        conn = self._open_stamped(tmp_db_path)
        called: list[int] = []

        def _v1(c: Any) -> None:
            called.append(1)

        try:
            apply_pending_migrations(conn, migrations=((1, _v1),))
            conn.commit()
        finally:
            conn.close()

        # The brand-new bootstrap already stamped v1 — the injected
        # body must not run a second time.
        assert called == []
        assert [row[0] for row in _read_schema_versions(tmp_db_path)] == [1]

    @pytest.mark.security
    def test_body_failure_leaves_prior_versions_committed(
        self,
        tmp_db_path: str,
    ) -> None:
        """A migration that raises must not stamp its own version.

        Mid-loop crash recovery hinges on this: the next open should
        re-attempt the failed migration, not skip past it.
        """
        conn = self._open_stamped(tmp_db_path)

        def _v2_ok(c: Any) -> None:
            c.execute("CREATE TABLE _v2_marker (x INTEGER)")

        def _v3_raises(c: Any) -> None:
            raise RuntimeError("simulated migration failure")

        try:
            with pytest.raises(RuntimeError, match="simulated migration failure"):
                apply_pending_migrations(
                    conn,
                    migrations=((2, _v2_ok), (3, _v3_raises)),
                )
            # v2 ran and was stamped *before* v3 raised; commit the
            # transaction so the next open sees the stamped state.
            conn.commit()
        finally:
            conn.close()

        # v2 stamped, v3 not stamped — exactly the partial state the
        # forward-only design relies on.
        assert [row[0] for row in _read_schema_versions(tmp_db_path)] == [1, 2]

    @pytest.mark.security
    def test_descending_migrations_rejected(self, tmp_db_path: str) -> None:
        conn = self._open_stamped(tmp_db_path)

        def _noop(c: Any) -> None:
            pass

        try:
            with pytest.raises(DatabaseError, match=r"strictly ascending"):
                apply_pending_migrations(
                    conn,
                    migrations=((3, _noop), (2, _noop)),
                )
        finally:
            conn.close()

    @pytest.mark.security
    def test_duplicate_migrations_rejected(self, tmp_db_path: str) -> None:
        conn = self._open_stamped(tmp_db_path)

        def _noop(c: Any) -> None:
            pass

        try:
            with pytest.raises(DatabaseError, match=r"strictly ascending"):
                apply_pending_migrations(
                    conn,
                    migrations=((2, _noop), (2, _noop)),
                )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# bootstrap_schema_version — direct unit tests
# ---------------------------------------------------------------------------


class TestBootstrapDirectCalls:
    """Exercise :func:`bootstrap_schema_version` directly on a raw connection.

    These complement the registry-driven tests above by pinning the
    pure-function contract (idempotency, transaction-agnosticism) so a
    future refactor that moves the call site can rely on it.
    """

    def test_idempotent_when_already_stamped(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "raw.sqlite")
        conn = sqlite3.connect(db_path)
        try:
            bootstrap_schema_version(conn)
            bootstrap_schema_version(conn)
            bootstrap_schema_version(conn)
            conn.commit()
        finally:
            conn.close()

        assert [row[0] for row in _read_schema_versions(db_path)] == [1]


# ---------------------------------------------------------------------------
# Source-level safeguard: no f-string SQL inside migrations.py
# ---------------------------------------------------------------------------


class TestMigrationsSourceHygiene:
    """Static safeguard: every SQL string in migrations.py is a literal.

    When v2+ migrations land, this test gates them — any new migration
    body that reaches for f-string interpolation will be caught here
    before the security review.
    """

    @pytest.mark.security
    def test_no_fstring_sql_in_module_source(self) -> None:
        from sec_generative_search.database import migrations as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")

        offending = [
            line
            for line in source.splitlines()
            # Cheap heuristic: an f-string that mentions a SQL verb.
            # Not a parser, but tight enough to catch real mistakes
            # without false-positives on the existing source.
            if (
                ('f"' in line or "f'" in line)
                and any(
                    verb in line.upper()
                    for verb in (
                        "SELECT ",
                        "INSERT ",
                        "UPDATE ",
                        "DELETE ",
                        "CREATE ",
                        "DROP ",
                    )
                )
            )
        ]
        assert offending == [], (
            "f-string SQL detected in migrations.py — every migration "
            "body must use ?-bound parameter binding."
        )

    @pytest.mark.security
    def test_no_credential_shaped_public_names(self) -> None:
        """The module surface carries no credential-bearing names.

        Defence-in-depth alongside the parametrised security tests in
        ``tests/core/test_types.py`` and ``tests/database/test_store.py``.
        """
        from sec_generative_search.database import migrations as mod

        forbidden = {
            "api_key",
            "secret",
            "password",
            "bearer",
            "token",
            "authorization",
        }
        public = {name for name in dir(mod) if not name.startswith("_")}
        assert public.isdisjoint(forbidden)
