"""Tests for ``sec-rag manage export`` and ``sec-rag manage import``.

Drive the commands through ``typer.testing.CliRunner``.  The portable
services and the storage collaborators are stubbed at the import site
on :mod:`sec_generative_search.cli.portable` so the tests focus on
the CLI seams: settings → registry probe → factory call → store
composition → service invocation → error rendering.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

import sec_generative_search.cli.portable as portable_module
from sec_generative_search.cli.portable import export, import_
from sec_generative_search.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
)
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    DatabaseError,
)
from sec_generative_search.core.types import (
    EmbedderStamp,
    ExportReport,
    ImportReport,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeExportService:
    instances: ClassVar[list[_FakeExportService]] = []

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.raises: BaseException | None = None
        _FakeExportService.instances.append(self)

    def export(
        self,
        output: Any,
        *,
        force: bool = False,
        tickers: list[str] | None = None,
        form_types: list[str] | None = None,
        accessions: list[str] | None = None,
        progress_callback: Any = None,
    ) -> ExportReport:
        self.calls.append(
            {
                "output": str(output),
                "force": force,
                "tickers": tickers,
                "form_types": form_types,
                "accessions": accessions,
            }
        )
        if self.raises is not None:
            raise self.raises
        return ExportReport(
            output_dir=str(output),
            source_embedder_stamp=EmbedderStamp(
                provider="local",
                model="m",
                dimension=4,
            ),
            filing_count=2,
            chunk_count=6,
            duration_seconds=0.3,
        )


class _FakeImportService:
    instances: ClassVar[list[_FakeImportService]] = []

    def __init__(self, store: Any, embedder: Any) -> None:
        self.store = store
        self.embedder = embedder
        self.calls: list[dict[str, Any]] = []
        self.raises: BaseException | None = None
        _FakeImportService.instances.append(self)

    def import_(
        self,
        input_path: Any,
        *,
        tickers: list[str] | None = None,
        form_types: list[str] | None = None,
        accessions: list[str] | None = None,
        progress_callback: Any = None,
    ) -> ImportReport:
        self.calls.append(
            {
                "input_path": str(input_path),
                "tickers": tickers,
                "form_types": form_types,
                "accessions": accessions,
            }
        )
        if self.raises is not None:
            raise self.raises
        return ImportReport(
            input_dir=str(input_path),
            source_embedder_stamp=EmbedderStamp(
                provider="other",
                model="m",
                dimension=4,
            ),
            filings_imported=2,
            filings_skipped=1,
            chunks_imported=6,
            duration_seconds=0.4,
        )


class _FakeChroma:
    instances: ClassVar[list[_FakeChroma]] = []

    def __init__(self, stamp: EmbedderStamp, *, chroma_path: str | None = None) -> None:
        self.stamp = stamp
        _FakeChroma.instances.append(self)


class _FakeRegistry:
    instances: ClassVar[list[_FakeRegistry]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _FakeRegistry.instances.append(self)


class _FakeStore:
    instances: ClassVar[list[_FakeStore]] = []

    def __init__(self, chroma: Any, registry: Any) -> None:
        self.chroma = chroma
        self.registry = registry
        _FakeStore.instances.append(self)


class _FakeEmbedder:
    def get_dimension(self) -> int:
        return 4


class _StubSettings:
    def __init__(
        self,
        *,
        provider: str = "openai",
        model_name: str = "text-embedding-3-small",
    ) -> None:
        self.database = DatabaseSettings.model_construct(
            chroma_path="./data/chroma_db",
            metadata_db_path="./data/metadata.sqlite",
            max_filings=10000,
            encryption_key=None,
            encryption_key_file=None,
            task_history_retention_days=0,
            task_history_persist_tickers=False,
            deployment_profile="local",
            retention_max_age_days=0,
        )
        self.embedding = EmbeddingSettings.model_construct(
            provider=provider,
            model_name=model_name,
            device="auto",
            batch_size=32,
            idle_timeout_minutes=0,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def app() -> typer.Typer:
    test_app = typer.Typer()
    test_app.command(name="export")(export)
    test_app.command(name="import")(import_)

    @test_app.command(name="_noop", hidden=True)
    def _noop() -> None:
        """Forces Typer to dispatch the others as subcommands."""

    return test_app


@pytest.fixture
def patch_export(monkeypatch: pytest.MonkeyPatch) -> type[_FakeExportService]:
    _FakeExportService.instances.clear()
    monkeypatch.setattr(portable_module, "PortableExportService", _FakeExportService)
    return _FakeExportService


@pytest.fixture
def patch_import(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    _FakeImportService.instances.clear()
    _FakeChroma.instances.clear()
    _FakeRegistry.instances.clear()
    _FakeStore.instances.clear()

    monkeypatch.setattr(portable_module, "PortableImportService", _FakeImportService)
    monkeypatch.setattr(portable_module, "ChromaDBClient", _FakeChroma)
    monkeypatch.setattr(portable_module, "MetadataRegistry", _FakeRegistry)
    monkeypatch.setattr(portable_module, "FilingStore", _FakeStore)
    # Default builder returns the fake embedder; tests mutate as needed.
    monkeypatch.setattr(
        portable_module,
        "build_embedder",
        lambda settings: _FakeEmbedder(),
    )
    return {
        "import_instances": _FakeImportService.instances,
        "chroma_instances": _FakeChroma.instances,
        "registry_instances": _FakeRegistry.instances,
        "store_instances": _FakeStore.instances,
    }


@pytest.fixture
def patch_settings(monkeypatch: pytest.MonkeyPatch) -> _StubSettings:
    stub = _StubSettings()
    monkeypatch.setattr(portable_module, "get_settings", lambda: stub)
    return stub


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_happy_path_invokes_service_with_no_filters(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_export: type[_FakeExportService],
        patch_settings: _StubSettings,
    ) -> None:
        result = runner.invoke(app, ["export", "-o", "out_dir"])
        assert result.exit_code == 0, result.output
        assert len(patch_export.instances) == 1
        call = patch_export.instances[0].calls[0]
        assert call["output"] == "out_dir"
        assert call["force"] is False
        assert call["tickers"] is None
        assert call["form_types"] is None
        assert call["accessions"] is None
        assert "Export complete" in result.output

    def test_filters_propagate(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_export: type[_FakeExportService],
        patch_settings: _StubSettings,
    ) -> None:
        result = runner.invoke(
            app,
            [
                "export",
                "-o",
                "out_dir",
                "--force",
                "-t",
                "AAPL",
                "-t",
                "MSFT",
                "-f",
                "10-K",
                "-a",
                "0000320193-23-000077",
            ],
        )
        assert result.exit_code == 0, result.output
        call = patch_export.instances[0].calls[0]
        assert call["force"] is True
        assert call["tickers"] == ["AAPL", "MSFT"]
        assert call["form_types"] == ["10-K"]
        assert call["accessions"] == ["0000320193-23-000077"]

    def test_database_error_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_settings: _StubSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        failing = _FakeExportService()
        failing.raises = DatabaseError("ChromaDB path not found", details="…")
        monkeypatch.setattr(
            portable_module,
            "PortableExportService",
            lambda: failing,
        )
        result = runner.invoke(app, ["export", "-o", "out_dir"])
        assert result.exit_code == 1, result.output
        assert "Export failed" in result.output

    def test_keyboard_interrupt_exits_130(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_settings: _StubSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        failing = _FakeExportService()
        failing.raises = KeyboardInterrupt()
        monkeypatch.setattr(
            portable_module,
            "PortableExportService",
            lambda: failing,
        )
        result = runner.invoke(app, ["export", "-o", "out_dir"])
        assert result.exit_code == 130, result.output
        assert "Interrupted" in result.output


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


class TestImport:
    def test_happy_path_composes_store_via_factory_seam(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_import: dict[str, Any],
        patch_settings: _StubSettings,
    ) -> None:
        result = runner.invoke(app, ["import", "-i", "in_dir", "-y"])
        assert result.exit_code == 0, result.output

        # ChromaDBClient was constructed with the stamp built from
        # settings + registry probe (no factory call needed for the
        # stamp; embedder construction is a separate concern).
        chromas = patch_import["chroma_instances"]
        assert len(chromas) == 1
        assert chromas[0].stamp == EmbedderStamp(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
        )
        # FilingStore was composed over both collaborators.
        stores = patch_import["store_instances"]
        assert len(stores) == 1
        # Import service received the composed store + the fake embedder.
        imports = patch_import["import_instances"]
        assert len(imports) == 1
        assert imports[0].store is stores[0]
        assert isinstance(imports[0].embedder, _FakeEmbedder)
        assert "Import complete" in result.output

    def test_filters_propagate(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_import: dict[str, Any],
        patch_settings: _StubSettings,
    ) -> None:
        result = runner.invoke(
            app,
            [
                "import",
                "-i",
                "in_dir",
                "-t",
                "AAPL",
                "-f",
                "10-K",
                "-a",
                "0000320193-23-000077",
                "-y",
            ],
        )
        assert result.exit_code == 0, result.output
        call = patch_import["import_instances"][0].calls[0]
        assert call["tickers"] == ["AAPL"]
        assert call["form_types"] == ["10-K"]
        assert call["accessions"] == ["0000320193-23-000077"]

    def test_invalid_embedder_settings_exit_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_import: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bad_stub = _StubSettings(model_name="absent-model")
        monkeypatch.setattr(portable_module, "get_settings", lambda: bad_stub)
        result = runner.invoke(app, ["import", "-i", "in_dir", "-y"])
        assert result.exit_code == 1, result.output
        assert "Embedder configuration invalid" in result.output

    def test_factory_configuration_error_exit_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_import: dict[str, Any],
        patch_settings: _StubSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raises(_settings: Any) -> Any:
            raise ConfigurationError("OPENAI_API_KEY required for openai embeddings")

        monkeypatch.setattr(portable_module, "build_embedder", _raises)
        result = runner.invoke(app, ["import", "-i", "in_dir", "-y"])
        assert result.exit_code == 1, result.output
        assert "Embedder construction failed" in result.output

    def test_factory_keyerror_for_missing_extra(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_import: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bad_stub = _StubSettings(provider="local", model_name="local-model")
        monkeypatch.setattr(portable_module, "get_settings", lambda: bad_stub)
        # ProviderRegistry.get_dimension would raise here for an
        # unknown local model — short-circuits the same path.
        result = runner.invoke(app, ["import", "-i", "in_dir", "-y"])
        assert result.exit_code == 1, result.output
        # Either path lands in an actionable error.
        assert (
            "Embedder configuration invalid" in result.output
            or "Embedder unavailable" in result.output
        )

    def test_storage_open_failure_exit_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_import: dict[str, Any],
        patch_settings: _StubSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _bad_chroma(stamp: Any, *args: Any, **kwargs: Any) -> Any:
            raise DatabaseError("ChromaDB stamp mismatch")

        monkeypatch.setattr(portable_module, "ChromaDBClient", _bad_chroma)
        result = runner.invoke(app, ["import", "-i", "in_dir", "-y"])
        assert result.exit_code == 1, result.output
        assert "Storage open failed" in result.output

    def test_database_error_during_import_exit_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_import: dict[str, Any],
        patch_settings: _StubSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Override the import service factory so the constructed
        # instance raises during import_().
        def _factory(store: Any, embedder: Any) -> _FakeImportService:
            inst = _FakeImportService(store, embedder)
            inst.raises = DatabaseError("Embedder failed on accession 'X'")
            return inst

        monkeypatch.setattr(portable_module, "PortableImportService", _factory)
        result = runner.invoke(app, ["import", "-i", "in_dir", "-y"])
        assert result.exit_code == 1, result.output
        assert "Import failed" in result.output

    def test_keyboard_interrupt_exits_130(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_import: dict[str, Any],
        patch_settings: _StubSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _factory(store: Any, embedder: Any) -> _FakeImportService:
            inst = _FakeImportService(store, embedder)
            inst.raises = KeyboardInterrupt()
            return inst

        monkeypatch.setattr(portable_module, "PortableImportService", _factory)
        result = runner.invoke(app, ["import", "-i", "in_dir", "-y"])
        assert result.exit_code == 130, result.output
        assert "Interrupted" in result.output
