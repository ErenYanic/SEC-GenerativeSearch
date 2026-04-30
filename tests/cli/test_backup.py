"""Tests for ``sec-rag manage backup`` and ``sec-rag manage restore``.

Drive the commands through ``typer.testing.CliRunner``.  The
:class:`BackupService` is stubbed at the import site so the tests
exercise the CLI's seams (settings → registry probe → service call →
error rendering) rather than re-running the round-trip covered by
``tests/database/test_backup.py``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

import sec_generative_search.cli.backup as backup_module
from sec_generative_search.cli.backup import backup, restore
from sec_generative_search.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
)
from sec_generative_search.core.exceptions import (
    DatabaseError,
    EmbeddingCollectionMismatchError,
)
from sec_generative_search.core.types import (
    BackupReport,
    EmbedderStamp,
    RestoreReport,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeBackupService:
    """Stand-in for :class:`BackupService`.

    Records every call to ``backup`` and ``restore`` with the kwargs
    so the tests can verify the CLI delivered the right arguments.
    A ``raises`` knob drives the error paths without touching real
    storage.
    """

    instances: ClassVar[list[_FakeBackupService]] = []

    def __init__(self) -> None:
        self.backup_calls: list[dict[str, Any]] = []
        self.restore_calls: list[dict[str, Any]] = []
        self.backup_raises: BaseException | None = None
        self.restore_raises: BaseException | None = None
        self.backup_report: BackupReport | None = None
        self.restore_report: RestoreReport | None = None
        _FakeBackupService.instances.append(self)

    def backup(
        self,
        output: Any,
        *,
        force: bool = False,
        progress_callback: Any = None,
    ) -> BackupReport:
        self.backup_calls.append(
            {
                "output": str(output),
                "force": force,
                "progress_callback": progress_callback,
            }
        )
        if self.backup_raises is not None:
            raise self.backup_raises
        return self.backup_report or BackupReport(
            output_path=str(output),
            embedder_stamp=EmbedderStamp(
                provider="openai",
                model="text-embedding-3-small",
                dimension=1536,
            ),
            schema_version=1,
            sqlcipher_encrypted=False,
            size_bytes=4096,
            duration_seconds=0.5,
        )

    def restore(
        self,
        input_path: Any,
        *,
        expected_stamp: EmbedderStamp,
        progress_callback: Any = None,
    ) -> RestoreReport:
        self.restore_calls.append(
            {
                "input_path": str(input_path),
                "expected_stamp": expected_stamp,
                "progress_callback": progress_callback,
            }
        )
        if self.restore_raises is not None:
            raise self.restore_raises
        return self.restore_report or RestoreReport(
            input_path=str(input_path),
            embedder_stamp=expected_stamp,
            schema_version=1,
            sqlcipher_encrypted=False,
            duration_seconds=0.4,
        )


class _StubSettings:
    """Hand-built settings stand-in matching the CLI's read surface."""

    def __init__(
        self,
        *,
        embedding_provider: str = "openai",
        embedding_model_name: str = "text-embedding-3-small",
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
            provider=embedding_provider,
            model_name=embedding_model_name,
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
    """Throw-away Typer app exposing both commands as named subcommands."""
    test_app = typer.Typer()
    test_app.command(name="backup")(backup)
    test_app.command(name="restore")(restore)

    @test_app.command(name="_noop", hidden=True)
    def _noop() -> None:
        """Forces Typer to dispatch the others as subcommands."""

    return test_app


@pytest.fixture
def patch_service(monkeypatch: pytest.MonkeyPatch) -> _FakeBackupService:
    """Replace ``BackupService`` on the CLI module with the fake."""
    _FakeBackupService.instances.clear()
    monkeypatch.setattr(backup_module, "BackupService", _FakeBackupService)
    # The CLI builds the instance inside the function body; we return
    # a sentinel that tests can use to grab it after the invocation.
    return _FakeBackupService  # type: ignore[return-value]


@pytest.fixture
def patch_settings(monkeypatch: pytest.MonkeyPatch) -> _StubSettings:
    """Replace ``cli.backup.get_settings`` with a closure over a stub."""
    stub = _StubSettings()
    monkeypatch.setattr(backup_module, "get_settings", lambda: stub)
    return stub


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------


class TestBackup:
    def test_happy_path_invokes_service_and_prints_summary(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_service: type[_FakeBackupService],
        patch_settings: _StubSettings,
    ) -> None:
        result = runner.invoke(app, ["backup", "-o", "out.tar.gz", "-y"])
        assert result.exit_code == 0, result.output

        services = patch_service.instances
        assert len(services) == 1
        assert services[0].backup_calls == [
            {
                "output": "out.tar.gz",
                "force": False,
                "progress_callback": services[0].backup_calls[0]["progress_callback"],
            }
        ]
        assert "Backup complete" in result.output

    def test_force_flag_propagates(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_service: type[_FakeBackupService],
        patch_settings: _StubSettings,
    ) -> None:
        result = runner.invoke(app, ["backup", "-o", "out.tar.gz", "--force", "-y"])
        assert result.exit_code == 0, result.output
        assert patch_service.instances[0].backup_calls[0]["force"] is True

    def test_database_error_exits_with_code_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_service: type[_FakeBackupService],
        patch_settings: _StubSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pre-populate a failing instance the CLI will reach.
        failing = _FakeBackupService()
        failing.backup_raises = DatabaseError(
            "ChromaDB path not found",
            details="…",
        )
        # Make ``BackupService()`` return the pre-built failing instance.
        monkeypatch.setattr(backup_module, "BackupService", lambda: failing)

        result = runner.invoke(app, ["backup", "-o", "out.tar.gz", "-y"])
        assert result.exit_code == 1, result.output
        assert "Backup failed" in result.output
        assert "ChromaDB path not found" in result.output

    def test_keyboard_interrupt_exits_130(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_service: type[_FakeBackupService],
        patch_settings: _StubSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        failing = _FakeBackupService()
        failing.backup_raises = KeyboardInterrupt()
        monkeypatch.setattr(backup_module, "BackupService", lambda: failing)

        result = runner.invoke(app, ["backup", "-o", "out.tar.gz", "-y"])
        assert result.exit_code == 130, result.output
        assert "Interrupted" in result.output
        assert "delete it" in result.output


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


class TestRestore:
    def test_happy_path_builds_expected_stamp_from_settings(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_service: type[_FakeBackupService],
        patch_settings: _StubSettings,
    ) -> None:
        result = runner.invoke(app, ["restore", "-i", "in.tar.gz", "-y"])
        assert result.exit_code == 0, result.output

        service = patch_service.instances[0]
        assert len(service.restore_calls) == 1
        call = service.restore_calls[0]
        assert call["input_path"] == "in.tar.gz"
        # Stamp is built from settings + registry probe.
        assert call["expected_stamp"] == EmbedderStamp(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
        )
        assert "Restore complete" in result.output

    def test_stamp_mismatch_exits_with_actionable_hint(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_service: type[_FakeBackupService],
        patch_settings: _StubSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected = EmbedderStamp(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
        )
        actual = EmbedderStamp(
            provider="local",
            model="google/embeddinggemma-300m",
            dimension=768,
        )
        failing = _FakeBackupService()
        failing.restore_raises = EmbeddingCollectionMismatchError(
            expected=expected,
            actual=actual,
        )
        monkeypatch.setattr(backup_module, "BackupService", lambda: failing)

        result = runner.invoke(app, ["restore", "-i", "in.tar.gz", "-y"])
        assert result.exit_code == 1, result.output
        assert "embedder stamp mismatch" in result.output
        # The mismatch error's uniform hint is rendered.
        assert "sec-rag manage reindex" in result.output

    def test_database_error_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_service: type[_FakeBackupService],
        patch_settings: _StubSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        failing = _FakeBackupService()
        failing.restore_raises = DatabaseError("Backup file not found")
        monkeypatch.setattr(backup_module, "BackupService", lambda: failing)

        result = runner.invoke(app, ["restore", "-i", "missing.tar.gz", "-y"])
        assert result.exit_code == 1, result.output
        assert "Restore failed" in result.output

    def test_keyboard_interrupt_exits_130(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_service: type[_FakeBackupService],
        patch_settings: _StubSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        failing = _FakeBackupService()
        failing.restore_raises = KeyboardInterrupt()
        monkeypatch.setattr(backup_module, "BackupService", lambda: failing)

        result = runner.invoke(app, ["restore", "-i", "in.tar.gz", "-y"])
        assert result.exit_code == 130, result.output
        assert "Interrupted" in result.output

    def test_invalid_embedder_settings_exit_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_service: type[_FakeBackupService],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Settings point at a model not in the registry.
        bad_stub = _StubSettings(
            embedding_provider="openai",
            embedding_model_name="absent-model",
        )
        monkeypatch.setattr(backup_module, "get_settings", lambda: bad_stub)

        result = runner.invoke(app, ["restore", "-i", "in.tar.gz", "-y"])
        assert result.exit_code == 1, result.output
        assert "Embedder configuration invalid" in result.output
