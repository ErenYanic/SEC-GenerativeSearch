"""Tests for ``sec-rag manage status`` / ``list`` / ``remove`` / ``clear``.

The two backing storage classes (:class:`ChromaDBClient`,
:class:`MetadataRegistry`) and the dual-store coordinator
(:class:`FilingStore`) are stubbed at the ``cli.manage`` import site —
stubbing at the import site (rather than at the source modules) proves
the CLI wires the right seams.

Goals:

- Confirm every **write** flows through :class:`FilingStore`, never
  through :class:`ChromaDBClient` / :class:`MetadataRegistry` mutation
  methods (the legacy free function ``delete_filings_batch`` no longer
  exists on the new package).
- Confirm **reads** go straight to :class:`MetadataRegistry`
  (``list_filings`` / ``get_statistics`` / ``get_filing``).
- Cover the operator-facing surface: success, empty database, foreign
  accession numbers, mutually-exclusive flags, ``--yes`` skip-prompt vs
  interactive confirm / decline, failure paths.
- Validate the security stance:
    - The CLI does NOT honour ``API_DEMO_MODE`` for ``clear`` — that
      flag is API-tier.  An operator with ``-y`` always proceeds.
    - API keys / EDGAR identity never leak into stdout.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

import sec_generative_search.cli.manage as manage_module
from sec_generative_search.cli.manage import manage_app
from sec_generative_search.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
)
from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.core.types import EmbedderStamp
from sec_generative_search.database import FilingRecord
from sec_generative_search.database.metadata import (
    DatabaseStatistics,
    TickerStatistics,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _make_record(
    *,
    rec_id: int = 1,
    ticker: str = "AAPL",
    form_type: str = "10-K",
    filing_date: str = "2024-01-15",
    accession: str = "0000320193-24-000001",
    chunk_count: int = 10,
    ingested_at: str = "2024-02-01T00:00:00+00:00",
) -> FilingRecord:
    return FilingRecord(
        id=rec_id,
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date,
        accession_number=accession,
        chunk_count=chunk_count,
        ingested_at=ingested_at,
    )


class _FakeRegistry:
    """Read-only :class:`MetadataRegistry` stand-in.

    Mutation methods (``register_filing``, ``remove_filing``,
    ``remove_filings_batch``, ``clear_all``) are deliberately absent —
    if the CLI tried to call any of them an :class:`AttributeError`
    would surface as a non-zero exit.  Writes are only allowed through
    the :class:`FilingStore` stub below.
    """

    instances: ClassVar[list[_FakeRegistry]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.filings: list[FilingRecord] = []
        self.statistics: DatabaseStatistics | None = None
        self.get_filing_returns: FilingRecord | None = None
        self.raises: BaseException | None = None
        self.list_calls: list[tuple[str | None, str | None]] = []
        self.get_filing_calls: list[str] = []
        self.statistics_calls = 0
        _FakeRegistry.instances.append(self)

    def list_filings(
        self,
        ticker: str | None = None,
        form_type: str | None = None,
    ) -> list[FilingRecord]:
        self.list_calls.append((ticker, form_type))
        if self.raises is not None:
            raise self.raises
        # Faithful in-memory filter so the tests can assert on the
        # listing path without dragging in a real sqlite connection.
        result = list(self.filings)
        if ticker:
            result = [f for f in result if f.ticker == ticker]
        if form_type:
            result = [f for f in result if f.form_type == form_type]
        return result

    def get_filing(self, accession_number: str) -> FilingRecord | None:
        self.get_filing_calls.append(accession_number)
        if self.raises is not None:
            raise self.raises
        return self.get_filing_returns

    def get_statistics(self) -> DatabaseStatistics:
        self.statistics_calls += 1
        if self.raises is not None:
            raise self.raises
        if self.statistics is not None:
            return self.statistics
        # Default: empty database.
        return DatabaseStatistics(
            filing_count=0,
            tickers=[],
            form_breakdown={},
            ticker_breakdown=[],
        )


class _FakeChroma:
    """Recording :class:`ChromaDBClient` stand-in.

    ``status`` calls ``collection_count`` on this stub.  Mutation
    methods (``delete_filing``, ``delete_filings_batch``,
    ``clear_collection``) are absent — every write must flow through
    the :class:`FilingStore` stub instead.
    """

    instances: ClassVar[list[_FakeChroma]] = []

    def __init__(self, stamp: EmbedderStamp, *, chroma_path: str | None = None) -> None:
        self.stamp = stamp
        self.chroma_path = chroma_path
        self.chunk_count = 0
        _FakeChroma.instances.append(self)

    def collection_count(self) -> int:
        return self.chunk_count


class _FakeStore:
    """:class:`FilingStore` stand-in that records every write.

    Tests assert on ``delete_calls`` / ``batch_calls`` / ``clear_calls``
    to prove the CLI routes writes through the store (and only through
    the store).
    """

    instances: ClassVar[list[_FakeStore]] = []

    def __init__(self, chroma: Any, registry: Any) -> None:
        self.chroma = chroma
        self.registry = registry
        self.delete_calls: list[str] = []
        self.batch_calls: list[list[str]] = []
        self.clear_calls = 0
        self.raises: BaseException | None = None
        self.batch_returns = 0
        self.clear_returns: tuple[int, int] = (0, 0)
        _FakeStore.instances.append(self)

    def delete_filing(self, accession_number: str) -> bool:
        self.delete_calls.append(accession_number)
        if self.raises is not None:
            raise self.raises
        return True

    def delete_filings_batch(self, accession_numbers: list[str]) -> int:
        self.batch_calls.append(list(accession_numbers))
        if self.raises is not None:
            raise self.raises
        return self.batch_returns

    def clear_all(self) -> tuple[int, int]:
        self.clear_calls += 1
        if self.raises is not None:
            raise self.raises
        return self.clear_returns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def app() -> typer.Typer:
    """Throw-away Typer app wrapping the manage sub-Typer."""
    test_app = typer.Typer()
    test_app.add_typer(manage_app, name="manage")
    return test_app


class _StubSettings:
    """Hand-built settings stand-in matching the seams the CLI reads.

    Pydantic-settings caches nested defaults at class-definition time,
    so an env-driven backdrop would not propagate through ``Settings()``
    in a unit-test context.  Stubbing ``get_settings`` is the cleanest
    seam.
    """

    def __init__(
        self,
        *,
        embedding_provider: str = "openai",
        embedding_model_name: str = "text-embedding-3-small",
        max_filings: int = 10000,
    ) -> None:
        self.embedding = EmbeddingSettings.model_construct(
            provider=embedding_provider,
            model_name=embedding_model_name,
            device="auto",
            batch_size=32,
            idle_timeout_minutes=0,
        )
        self.database = DatabaseSettings.model_construct(
            deployment_profile="local",
            retention_max_age_days=0,
            chroma_path="./data/chroma_db",
            metadata_db_path="./data/metadata.sqlite",
            max_filings=max_filings,
            encryption_key=None,
            encryption_key_file=None,
            task_history_retention_days=0,
            task_history_persist_tickers=False,
        )


@pytest.fixture
def patched_storage(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace every seam ``cli.manage`` reaches into with a recording stub."""
    _FakeRegistry.instances.clear()
    _FakeChroma.instances.clear()
    _FakeStore.instances.clear()

    monkeypatch.setattr(manage_module, "get_settings", _StubSettings)
    monkeypatch.setattr(manage_module, "ChromaDBClient", _FakeChroma)
    monkeypatch.setattr(manage_module, "MetadataRegistry", _FakeRegistry)
    monkeypatch.setattr(manage_module, "FilingStore", _FakeStore)

    return {
        "registries": _FakeRegistry.instances,
        "chromas": _FakeChroma.instances,
        "stores": _FakeStore.instances,
    }


def _seed_registry(filings: list[FilingRecord], *, stats: DatabaseStatistics | None = None) -> None:
    """Arm the next-constructed :class:`_FakeRegistry` with seeded reads."""
    original_init = _FakeRegistry.__init__

    def _seeded(self: _FakeRegistry, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.filings = list(filings)
        if stats is not None:
            self.statistics = stats

    _FakeRegistry.__init__ = _seeded  # type: ignore[method-assign]


def _restore_registry_init() -> None:
    """Re-attach the unseeded :class:`_FakeRegistry` initialiser."""

    def _default(self: _FakeRegistry, *args: Any, **kwargs: Any) -> None:
        self.filings = []
        self.statistics = None
        self.get_filing_returns = None
        self.raises = None
        self.list_calls = []
        self.get_filing_calls = []
        self.statistics_calls = 0
        _FakeRegistry.instances.append(self)

    _FakeRegistry.__init__ = _default  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _reset_registry_init() -> Any:
    """Each test gets a fresh registry constructor so seeded state never leaks."""
    yield
    _restore_registry_init()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_empty_database_renders_dashes(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        """Empty database → 0 filings, 0 chunks, em-dash placeholders."""
        result = runner.invoke(app, ["manage", "status"])
        assert result.exit_code == 0, result.output
        assert "Filings" in result.output
        assert "0/10000" in result.output
        # The chunk count is 0 (default fake-chroma chunk_count).
        # Stamp seal occurred — chroma was opened with the configured stamp.
        chromas = patched_storage["chromas"]
        assert len(chromas) == 1
        assert chromas[0].stamp.provider == "openai"
        assert chromas[0].stamp.model == "text-embedding-3-small"
        assert chromas[0].stamp.dimension == 1536

    def test_populated_database_renders_breakdown(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stats = DatabaseStatistics(
            filing_count=3,
            tickers=["AAPL", "MSFT"],
            form_breakdown={"10-K": 2, "10-Q": 1},
            ticker_breakdown=[
                TickerStatistics(ticker="AAPL", filings=2, chunks=20, forms=["10-K"]),
                TickerStatistics(ticker="MSFT", filings=1, chunks=15, forms=["10-Q"]),
            ],
        )
        _seed_registry([], stats=stats)

        # Arm the chunk count on the next-constructed chroma stub.
        original = _FakeChroma.__init__

        def _seeded(self: _FakeChroma, *args: Any, **kwargs: Any) -> None:
            original(self, *args, **kwargs)
            self.chunk_count = 35

        monkeypatch.setattr(_FakeChroma, "__init__", _seeded)

        result = runner.invoke(app, ["manage", "status"])
        assert result.exit_code == 0, result.output
        assert "3/10000" in result.output
        assert "35" in result.output
        # Ticker breakdown is escaped but content survives.
        assert "AAPL" in result.output
        assert "MSFT" in result.output
        assert "10-K" in result.output

    def test_chroma_open_failure_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _BoomChroma:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise DatabaseError("stamp mismatch")

        monkeypatch.setattr(manage_module, "ChromaDBClient", _BoomChroma)

        result = runner.invoke(app, ["manage", "status"])
        assert result.exit_code == 1
        assert "Storage initialisation failed" in result.output


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    def test_empty_database_renders_no_filings(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["manage", "list"])
        assert result.exit_code == 0
        assert "No filings found" in result.output
        # Read path never opens ChromaDB.
        assert patched_storage["chromas"] == []

    def test_lists_filings_with_filters(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        _seed_registry(
            [
                _make_record(ticker="AAPL", form_type="10-K"),
                _make_record(
                    ticker="MSFT",
                    form_type="10-Q",
                    accession="0000789019-24-000002",
                ),
            ]
        )
        result = runner.invoke(app, ["manage", "list", "-k", "aapl"])
        assert result.exit_code == 0
        # Filter is uppercased before the registry query.
        registry = _FakeRegistry.instances[-1]
        assert registry.list_calls == [("AAPL", None)]
        assert "AAPL" in result.output
        assert "MSFT" not in result.output

    def test_list_database_error_exit_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        original_init = _FakeRegistry.__init__

        def _seeded(self: _FakeRegistry, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            self.raises = DatabaseError("disk error")

        _FakeRegistry.__init__ = _seeded  # type: ignore[method-assign]
        result = runner.invoke(app, ["manage", "list"])
        assert result.exit_code == 1
        assert "List failed" in result.output


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


class TestRemoveSingle:
    def test_happy_path_writes_through_store(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        record = _make_record()
        original_init = _FakeRegistry.__init__

        def _seeded(self: _FakeRegistry, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            self.get_filing_returns = record

        _FakeRegistry.__init__ = _seeded  # type: ignore[method-assign]

        result = runner.invoke(app, ["manage", "remove", record.accession_number, "-y"])
        assert result.exit_code == 0, result.output

        store = _FakeStore.instances[-1]
        assert store.delete_calls == [record.accession_number]
        # Bulk-delete and clear surfaces are untouched.
        assert store.batch_calls == []
        assert store.clear_calls == 0
        assert "Removed:" in result.output

    def test_missing_accession_exit_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        # Registry returns None for the lookup (no seeding needed —
        # ``get_filing_returns`` defaults to None).
        result = runner.invoke(app, ["manage", "remove", "9999-99-999999", "-y"])
        assert result.exit_code == 1
        assert "Filing not found" in result.output
        # Store never invoked.
        store = _FakeStore.instances[-1]
        assert store.delete_calls == []

    def test_interactive_decline_aborts_clean(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        record = _make_record()
        original_init = _FakeRegistry.__init__

        def _seeded(self: _FakeRegistry, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            self.get_filing_returns = record

        _FakeRegistry.__init__ = _seeded  # type: ignore[method-assign]

        result = runner.invoke(app, ["manage", "remove", record.accession_number], input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        store = _FakeStore.instances[-1]
        assert store.delete_calls == []


class TestRemoveBulk:
    def test_filter_bulk_writes_through_store(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        _seed_registry(
            [
                _make_record(ticker="AAPL", accession="0000320193-24-000001"),
                _make_record(
                    ticker="AAPL",
                    form_type="10-Q",
                    accession="0000320193-24-000002",
                ),
                _make_record(
                    ticker="MSFT",
                    form_type="10-K",
                    accession="0000789019-24-000003",
                ),
            ]
        )
        original_store_init = _FakeStore.__init__

        def _seeded_store(self: _FakeStore, *args: Any, **kwargs: Any) -> None:
            original_store_init(self, *args, **kwargs)
            self.batch_returns = 2

        _FakeStore.__init__ = _seeded_store  # type: ignore[method-assign]
        try:
            result = runner.invoke(app, ["manage", "remove", "--ticker", "aapl", "-y"])
        finally:
            _FakeStore.__init__ = original_store_init  # type: ignore[method-assign]

        assert result.exit_code == 0, result.output
        store = _FakeStore.instances[-1]
        # Only the two AAPL accessions are forwarded — MSFT survives.
        assert len(store.batch_calls) == 1
        assert sorted(store.batch_calls[0]) == [
            "0000320193-24-000001",
            "0000320193-24-000002",
        ]
        assert "2 filing(s) removed" in result.output

    def test_no_matches_renders_quiet_message(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        # Empty registry → empty list_filings result → quiet message.
        result = runner.invoke(app, ["manage", "remove", "--ticker", "ZZZZ", "-y"])
        assert result.exit_code == 0
        assert "No filings found matching" in result.output
        store = _FakeStore.instances[-1]
        assert store.batch_calls == []


class TestRemoveFlagValidation:
    def test_no_target_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["manage", "remove"])
        assert result.exit_code == 1
        assert "Missing target" in result.output
        # No storage opened — invalid flag combination short-circuits before
        # we reach ``_open_store``.
        assert patched_storage["stores"] == []

    def test_accession_plus_filters_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            [
                "manage",
                "remove",
                "0000320193-24-000001",
                "--ticker",
                "AAPL",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid flag combination" in result.output
        assert patched_storage["stores"] == []


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_empty_database_short_circuits(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["manage", "clear", "-y"])
        assert result.exit_code == 0
        assert "already empty" in result.output
        store = _FakeStore.instances[-1]
        assert store.clear_calls == 0

    def test_clear_with_yes_routes_through_store(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        _seed_registry(
            [
                _make_record(ticker="AAPL", accession="0000320193-24-000001"),
                _make_record(ticker="MSFT", accession="0000789019-24-000002"),
            ]
        )
        original_store_init = _FakeStore.__init__

        def _seeded_store(self: _FakeStore, *args: Any, **kwargs: Any) -> None:
            original_store_init(self, *args, **kwargs)
            self.clear_returns = (20, 2)

        _FakeStore.__init__ = _seeded_store  # type: ignore[method-assign]
        try:
            result = runner.invoke(app, ["manage", "clear", "-y"])
        finally:
            _FakeStore.__init__ = original_store_init  # type: ignore[method-assign]

        assert result.exit_code == 0, result.output
        store = _FakeStore.instances[-1]
        assert store.clear_calls == 1
        # Direct ChromaDB / registry mutation paths were untouched.
        assert store.batch_calls == []
        assert store.delete_calls == []
        assert "2 filing(s) removed" in result.output

    def test_interactive_decline_aborts(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        _seed_registry([_make_record()])
        result = runner.invoke(app, ["manage", "clear"], input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        store = _FakeStore.instances[-1]
        assert store.clear_calls == 0

    def test_clear_failure_surfaces_database_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        _seed_registry([_make_record()])
        original_store_init = _FakeStore.__init__

        def _seeded_store(self: _FakeStore, *args: Any, **kwargs: Any) -> None:
            original_store_init(self, *args, **kwargs)
            self.raises = DatabaseError("disk exploded")

        _FakeStore.__init__ = _seeded_store  # type: ignore[method-assign]
        try:
            result = runner.invoke(app, ["manage", "clear", "-y"])
        finally:
            _FakeStore.__init__ = original_store_init  # type: ignore[method-assign]

        assert result.exit_code == 1
        assert "Clear failed" in result.output


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecurity:
    def test_cli_does_not_honour_api_demo_mode(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``API_DEMO_MODE=true`` MUST NOT block ``manage clear -y``.

        That flag locks the API surface against web users in cloud
        deployments; it has no operator-tier effect.  An admin running
        ``sec-rag manage clear -y`` on the host must always proceed —
        anything else would surprise an operator trying to reset a
        demo environment from the box.
        """
        monkeypatch.setenv("API_DEMO_MODE", "true")
        _seed_registry([_make_record()])

        result = runner.invoke(app, ["manage", "clear", "-y"])
        assert result.exit_code == 0, result.output
        store = _FakeStore.instances[-1]
        assert store.clear_calls == 1

    def test_writes_never_touch_registry_or_chroma_mutation(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
    ) -> None:
        """The CLI MUST route every write through :class:`FilingStore`.

        The chroma + registry stubs do not implement mutation methods.
        If the CLI tried to call any of them an ``AttributeError`` would
        surface as a non-zero exit — this test fails loudly if a future
        refactor reintroduces direct ChromaDB / registry mutation from
        this surface.
        """
        record = _make_record()
        original_init = _FakeRegistry.__init__

        def _seeded(self: _FakeRegistry, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            self.get_filing_returns = record

        _FakeRegistry.__init__ = _seeded  # type: ignore[method-assign]

        result = runner.invoke(app, ["manage", "remove", record.accession_number, "-y"])
        assert result.exit_code == 0, result.output
        store = _FakeStore.instances[-1]
        assert store.delete_calls == [record.accession_number]

    def test_api_key_never_appears_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An OPENAI_API_KEY in the environment must never leak — including
        on error paths.  We arm a storage failure to exercise the
        loudest path."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-canary-token-must-not-appear-in-output")
        original_store_init = _FakeStore.__init__
        _seed_registry([_make_record()])

        def _seeded_store(self: _FakeStore, *args: Any, **kwargs: Any) -> None:
            original_store_init(self, *args, **kwargs)
            self.raises = DatabaseError("storage exploded")

        _FakeStore.__init__ = _seeded_store  # type: ignore[method-assign]
        try:
            result = runner.invoke(app, ["manage", "clear", "-y"])
        finally:
            _FakeStore.__init__ = original_store_init  # type: ignore[method-assign]

        assert "sk-canary-token" not in result.output

    def test_edgar_identity_never_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_storage: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """EDGAR identity is PII — it must not appear in any CLI output."""
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Test User")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "canary-edgar@example.test")
        _seed_registry([_make_record()])

        result = runner.invoke(app, ["manage", "status"])
        assert "canary-edgar@example.test" not in result.output
        assert "Test User" not in result.output
