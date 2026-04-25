"""Tests for ``sec-rag manage evict``.

Drive the command through ``typer.testing.CliRunner``.  The two
backing stores (:class:`ChromaDBClient`, :class:`MetadataRegistry`)
and the dual-store coordinator (:class:`FilingStore`) are stubbed at
the ``cli.evict`` import site — stubbing at the import site (rather
than at the source modules) proves the CLI wires the right seams.

The settings layer is exercised live via ``reload_settings`` so the
profile-defaults precedence the operator interacts with is validated
end-to-end.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

import sec_generative_search.cli.evict as evict_module
from sec_generative_search.cli.evict import evict
from sec_generative_search.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
)
from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.core.types import EmbedderStamp, EvictionReport

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeChroma:
    """Stand-in for :class:`ChromaDBClient`.

    Records the stamp it was opened with so a test can assert the
    CLI composed it from settings + registry without a factory call.
    """

    instances: ClassVar[list[_FakeChroma]] = []

    def __init__(self, stamp: EmbedderStamp, *, chroma_path: str | None = None) -> None:
        self.stamp = stamp
        self.chroma_path = chroma_path
        _FakeChroma.instances.append(self)


class _FakeRegistry:
    """Stand-in for :class:`MetadataRegistry`.

    The CLI never touches the registry directly — the seam is the
    :class:`FilingStore` that wraps both — but the constructor must
    succeed for the CLI to reach the eviction call.
    """

    instances: ClassVar[list[_FakeRegistry]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _FakeRegistry.instances.append(self)


class _FakeStore:
    """Stand-in for :class:`FilingStore`.

    Captures every ``evict_expired`` call so tests can assert on the
    cutoff actually delivered to the storage layer; a ``raises`` knob
    drives the CLI's error paths without touching real backing stores.
    """

    instances: ClassVar[list[_FakeStore]] = []

    def __init__(self, chroma: Any, registry: Any) -> None:
        self.chroma = chroma
        self.registry = registry
        self.calls: list[int] = []
        self.report = EvictionReport(
            filings_evicted=2,
            chunks_evicted=15,
            max_age_days=0,
        )
        self.raises: BaseException | None = None
        _FakeStore.instances.append(self)

    def evict_expired(self, max_age_days: int) -> EvictionReport:
        self.calls.append(max_age_days)
        if self.raises is not None:
            raise self.raises
        # Echo the actual cutoff in the returned report so the CLI's
        # success line can match the input it received.
        return EvictionReport(
            filings_evicted=self.report.filings_evicted,
            chunks_evicted=self.report.chunks_evicted,
            max_age_days=max_age_days,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def app() -> typer.Typer:
    """Throw-away Typer app exposing ``evict`` as a named subcommand.

    A second hidden command keeps Typer dispatching ``evict`` as a
    subcommand rather than auto-unwrapping a single-command app
    (matches the pattern in :mod:`tests.cli.test_reindex`).
    """
    test_app = typer.Typer()
    test_app.command(name="evict")(evict)

    @test_app.command(name="_noop", hidden=True)
    def _noop() -> None:
        """Forces Typer to treat ``evict`` as a subcommand."""

    return test_app


@pytest.fixture
def patch_storage(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the storage names on ``cli.evict`` with recording stubs.

    Returns a dict so individual tests can mutate the latest stub's
    ``raises`` / ``report`` attributes after construction (eviction
    happens inside the CLI body, after the stubs were created).
    """
    _FakeChroma.instances.clear()
    _FakeRegistry.instances.clear()
    _FakeStore.instances.clear()

    monkeypatch.setattr(evict_module, "ChromaDBClient", _FakeChroma)
    monkeypatch.setattr(evict_module, "MetadataRegistry", _FakeRegistry)
    monkeypatch.setattr(evict_module, "FilingStore", _FakeStore)

    return {
        "chroma_instances": _FakeChroma.instances,
        "registry_instances": _FakeRegistry.instances,
        "store_instances": _FakeStore.instances,
    }


class _StubSettings:
    """Hand-built settings stand-in for the CLI body.

    Pydantic-settings caches nested settings field defaults at class
    definition time, so env-var-driven test backdrops do not propagate
    through ``Settings()``.  Patching the CLI's ``get_settings`` symbol
    bypasses that gotcha and keeps the tests focused on the CLI surface
    rather than on the settings-loader plumbing.

    Only the fields the CLI actually reads are populated; an
    `AttributeError` on anything else is the loud signal we want.
    """

    def __init__(
        self,
        *,
        deployment_profile: str = "team",
        retention_max_age_days: int = 90,
        embedding_provider: str = "openai",
        embedding_model_name: str = "text-embedding-3-small",
    ) -> None:
        self.database = DatabaseSettings.model_construct(
            deployment_profile=deployment_profile,
            retention_max_age_days=retention_max_age_days,
            chroma_path="./data/chroma_db",
            metadata_db_path="./data/metadata.sqlite",
            max_filings=10000,
            encryption_key=None,
            encryption_key_file=None,
            task_history_retention_days=0,
            task_history_persist_tickers=False,
        )
        self.embedding = EmbeddingSettings.model_construct(
            provider=embedding_provider,
            model_name=embedding_model_name,
            device="auto",
            batch_size=32,
            idle_timeout_minutes=0,
        )


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    deployment_profile: str = "team",
    retention_max_age_days: int = 90,
    embedding_provider: str = "openai",
    embedding_model_name: str = "text-embedding-3-small",
) -> _StubSettings:
    """Replace ``cli.evict.get_settings`` with a closure over a stub."""
    stub = _StubSettings(
        deployment_profile=deployment_profile,
        retention_max_age_days=retention_max_age_days,
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
    )
    monkeypatch.setattr(evict_module, "get_settings", lambda: stub)
    return stub


@pytest.fixture
def env_team(monkeypatch: pytest.MonkeyPatch) -> _StubSettings:
    """Team profile, default 90-day retention, hosted embedder."""
    return _patch_settings(monkeypatch)


@pytest.fixture
def env_local_disabled(monkeypatch: pytest.MonkeyPatch) -> _StubSettings:
    """Local profile, eviction disabled (cutoff=0), hosted embedder."""
    return _patch_settings(
        monkeypatch,
        deployment_profile="local",
        retention_max_age_days=0,
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_explicit_max_age_days_drives_evict(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_storage: dict[str, Any],
        env_team: pytest.MonkeyPatch,
    ) -> None:
        result = runner.invoke(app, ["evict", "-d", "30", "-y"])
        assert result.exit_code == 0, result.output

        stores = patch_storage["store_instances"]
        assert len(stores) == 1
        assert stores[0].calls == [30]
        # The composed stamp matches the configured embedder + the
        # registry-resolved dimension (no factory call needed).
        chromas = patch_storage["chroma_instances"]
        assert len(chromas) == 1
        assert chromas[0].stamp.provider == "openai"
        assert chromas[0].stamp.model == "text-embedding-3-small"
        assert chromas[0].stamp.dimension == 1536
        # Success line rendered.
        assert "Eviction complete" in result.output
        assert "30 day(s)" in result.output

    def test_cutoff_falls_through_to_settings(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_storage: dict[str, Any],
        env_team: pytest.MonkeyPatch,
    ) -> None:
        """No --max-age-days flag → cutoff comes from
        ``DB_RETENTION_MAX_AGE_DAYS`` (team profile default = 90)."""
        result = runner.invoke(app, ["evict", "-y"])
        assert result.exit_code == 0, result.output
        stores = patch_storage["store_instances"]
        assert stores[0].calls == [90]

    def test_zero_eviction_renders_nothing_to_evict(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_storage: dict[str, Any],
        env_team: pytest.MonkeyPatch,
    ) -> None:
        """A successful sweep that found no expired rows must render
        as a quiet ``Nothing to evict`` rather than as an error."""
        # Pre-stage the store stub to return zero filings.
        # We know patch_storage will produce exactly one _FakeStore;
        # capture it before the CLI runs by sneaking a class-level
        # default down into _FakeStore.report.
        # Easier: set the next constructed store's report inside the
        # invocation by patching after run.  Simpler still: monkeypatch
        # the class default for this test only.

        original = _FakeStore.__init__

        def _patched_init(self: _FakeStore, *args: Any, **kwargs: Any) -> None:
            original(self, *args, **kwargs)
            self.report = EvictionReport(
                filings_evicted=0,
                chunks_evicted=0,
                max_age_days=0,
            )

        _FakeStore.__init__ = _patched_init  # type: ignore[method-assign]
        try:
            result = runner.invoke(app, ["evict", "-d", "365", "-y"])
        finally:
            _FakeStore.__init__ = original  # type: ignore[method-assign]
        assert result.exit_code == 0, result.output
        assert "Nothing to evict" in result.output
        assert "365 day(s)" in result.output

    def test_interactive_confirm_yes_runs(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_storage: dict[str, Any],
        env_team: pytest.MonkeyPatch,
    ) -> None:
        result = runner.invoke(app, ["evict", "-d", "30"], input="y\n")
        assert result.exit_code == 0, result.output
        assert patch_storage["store_instances"][0].calls == [30]

    def test_interactive_confirm_no_aborts_without_calling_store(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_storage: dict[str, Any],
        env_team: pytest.MonkeyPatch,
    ) -> None:
        result = runner.invoke(app, ["evict", "-d", "30"], input="n\n")
        assert result.exit_code == 0, result.output
        assert "Cancelled" in result.output
        # Confirmation aborts before the store is constructed.
        assert patch_storage["store_instances"] == []


# ---------------------------------------------------------------------------
# Disabled-cutoff refusal
# ---------------------------------------------------------------------------


class TestDisabledCutoffRefuses:
    """When the resolved cutoff is non-positive the CLI refuses with
    exit code 1 and a hint — never silently no-ops."""

    def test_local_default_refuses(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_storage: dict[str, Any],
        env_local_disabled: pytest.MonkeyPatch,
    ) -> None:
        result = runner.invoke(app, ["evict", "-y"])
        assert result.exit_code == 1
        assert "Eviction disabled" in result.output
        # Storage was never opened.
        assert patch_storage["store_instances"] == []

    def test_explicit_zero_refuses(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_storage: dict[str, Any],
        env_team: pytest.MonkeyPatch,
    ) -> None:
        """An explicit --max-age-days 0 also refuses; the operator
        gets the same hint regardless of which path supplied the 0."""
        result = runner.invoke(app, ["evict", "-d", "0", "-y"])
        assert result.exit_code == 1
        assert "Eviction disabled" in result.output
        assert patch_storage["store_instances"] == []


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_database_error_returns_exit_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_storage: dict[str, Any],
        env_team: pytest.MonkeyPatch,
    ) -> None:
        original = _FakeStore.__init__

        def _patched_init(self: _FakeStore, *args: Any, **kwargs: Any) -> None:
            original(self, *args, **kwargs)
            self.raises = DatabaseError("simulated", details="boom")

        _FakeStore.__init__ = _patched_init  # type: ignore[method-assign]
        try:
            result = runner.invoke(app, ["evict", "-d", "30", "-y"])
        finally:
            _FakeStore.__init__ = original  # type: ignore[method-assign]
        assert result.exit_code == 1
        assert "Eviction failed" in result.output
        assert "simulated" in result.output

    def test_keyboard_interrupt_returns_130(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_storage: dict[str, Any],
        env_team: pytest.MonkeyPatch,
    ) -> None:
        original = _FakeStore.__init__

        def _patched_init(self: _FakeStore, *args: Any, **kwargs: Any) -> None:
            original(self, *args, **kwargs)
            self.raises = KeyboardInterrupt()

        _FakeStore.__init__ = _patched_init  # type: ignore[method-assign]
        try:
            result = runner.invoke(app, ["evict", "-d", "30", "-y"])
        finally:
            _FakeStore.__init__ = original  # type: ignore[method-assign]
        assert result.exit_code == 130
        assert "Interrupted" in result.output

    def test_unknown_embedder_model_returns_exit_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_storage: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A misconfigured EMBEDDING_MODEL_NAME surfaces as a clean
        operator-facing error, not a stack trace."""
        _patch_settings(monkeypatch, embedding_model_name="no-such-model")

        result = runner.invoke(app, ["evict", "-d", "30", "-y"])
        assert result.exit_code == 1
        assert "Embedder configuration invalid" in result.output


# ---------------------------------------------------------------------------
# Security / operator surface guarantees
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecurity:
    def test_no_dimension_flag_on_surface(
        self,
        runner: CliRunner,
        app: typer.Typer,
    ) -> None:
        """``--dimension`` must not exist on the evict CLI for the
        same reason it is not on the reindex CLI: the registry is the
        single source of truth, and a CLI-supplied dimension would let
        a misconfigured stamp slip past the storage seal."""
        result = runner.invoke(app, ["evict", "--help"])
        assert result.exit_code == 0
        assert "--dimension" not in result.output

    def test_credential_env_does_not_appear_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_storage: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No path through the CLI may echo an API key into stdout —
        including error paths.  The CLI never reads the embedder's
        credential surface (it does no embedding), but this fails
        loudly if a future refactor changes that."""
        _patch_settings(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-canary-token-must-not-appear-in-output")

        result = runner.invoke(app, ["evict", "-d", "30", "-y"])
        assert "sk-canary-token" not in result.output
