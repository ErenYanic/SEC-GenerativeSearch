"""Tests for the ``sec-rag`` CLI root shell.

Covers the shell invariants:

- ``--version`` short-circuits before any bootstrap side effect runs.
- ``--verbose`` switches the package logger to DEBUG via
    :func:`configure_logging`.
- ``KeyboardInterrupt`` from anywhere inside ``app()`` maps to exit 130
    instead of Click's default exit 1.
- EDGAR identity bootstrap fires exactly when both
    ``EDGAR_IDENTITY_NAME`` and ``EDGAR_IDENTITY_EMAIL`` are set, and never
    leaks the resolved name or email into any log record.

Tests drive the Typer app through ``CliRunner`` and stub the local
``edgar.set_identity`` import so we do not depend on edgartools' real
process-global mutation across tests.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from io import StringIO

import pytest
from rich.console import Console
from typer.testing import CliRunner

import sec_generative_search.cli.main as main_module
from sec_generative_search.cli.main import (
    _configure_edgar_identity,
    _resolve_version,
    app,
    main,
)
from sec_generative_search.config.settings import reload_settings

runner = CliRunner()


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


class TestVersion:
    def test_prints_version_and_exits_zero(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "sec-rag" in result.stdout

    @pytest.mark.security
    def test_version_skips_root_bootstrap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``--version`` is eager — the root callback (and therefore the
        EDGAR identity bootstrap) MUST NOT run.  An eagerly evaluated flag
        that still triggers identity mutation would be surprising and
        defeats the whole point of ``--version`` being a fast probe.
        """
        called: list[str] = []
        monkeypatch.setattr(
            main_module,
            "_configure_edgar_identity",
            lambda: called.append("bootstrap"),
        )
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert called == []

    def test_resolve_version_handles_missing_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(_name: str) -> str:
            raise PackageNotFoundError(_name)

        monkeypatch.setattr(main_module, "version", _raise)
        assert _resolve_version() == "unknown"

    @pytest.mark.security
    def test_version_output_passes_through_rich_escape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the version string ever contains Rich markup-like brackets,
        ``rich.markup.escape`` must keep them literal — never interpret
        them as colour codes or hyperlink tags.
        """
        buffer = StringIO()
        test_console = Console(
            file=buffer,
            force_terminal=False,
            no_color=True,
            highlight=False,
            width=200,
        )
        monkeypatch.setattr(main_module, "console", test_console)
        monkeypatch.setattr(main_module, "_resolve_version", lambda: "[red]EVIL[/red]")

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        # The literal brackets survive into the rendered output — they were
        # escaped, not interpreted as a markup tag.
        assert "[red]EVIL[/red]" in buffer.getvalue()


# ---------------------------------------------------------------------------
# --verbose
# ---------------------------------------------------------------------------


class TestVerbose:
    def test_switches_to_debug(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_configure(level: int | None = None) -> None:
            captured["level"] = level

        def fake_suppress() -> None:
            captured["suppress"] = True

        monkeypatch.setattr(main_module, "configure_logging", fake_configure)
        monkeypatch.setattr(main_module, "suppress_third_party_loggers", fake_suppress)

        # ``--verbose --help`` exercises the eager verbose callback without
        # needing a real subcommand.  Click short-circuits on --help.
        result = runner.invoke(app, ["--verbose", "--help"])

        assert result.exit_code == 0
        assert captured["level"] == logging.DEBUG
        assert captured["suppress"] is True

    def test_no_verbose_means_no_configure_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without ``--verbose`` we must not re-invoke configure_logging —
        a duplicate call would clobber operator-supplied log config from
        the environment.
        """
        captured: list[int] = []
        monkeypatch.setattr(
            main_module,
            "configure_logging",
            lambda level=None: captured.append(level),
        )
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert captured == []


# ---------------------------------------------------------------------------
# EDGAR identity bootstrap
# ---------------------------------------------------------------------------


class TestEdgarBootstrap:
    def _patch_set_identity(self, monkeypatch: pytest.MonkeyPatch, sink: list[str]) -> None:
        import edgar

        monkeypatch.setattr(edgar, "set_identity", lambda raw: sink.append(raw))

    def test_fires_when_both_env_vars_set(
        self,
        clean_env: pytest.MonkeyPatch,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clean_env.setenv("EDGAR_IDENTITY_NAME", "Test User")
        clean_env.setenv("EDGAR_IDENTITY_EMAIL", "test@example.com")
        reload_settings()

        captured: list[str] = []
        self._patch_set_identity(monkeypatch, captured)

        _configure_edgar_identity()
        assert captured == ["Test User test@example.com"]

    def test_skipped_when_name_missing(
        self,
        clean_env: pytest.MonkeyPatch,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clean_env.setenv("EDGAR_IDENTITY_EMAIL", "test@example.com")
        reload_settings()

        captured: list[str] = []
        self._patch_set_identity(monkeypatch, captured)

        _configure_edgar_identity()
        assert captured == []

    def test_skipped_when_email_missing(
        self,
        clean_env: pytest.MonkeyPatch,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        clean_env.setenv("EDGAR_IDENTITY_NAME", "Test User")
        reload_settings()

        captured: list[str] = []
        self._patch_set_identity(monkeypatch, captured)

        _configure_edgar_identity()
        assert captured == []

    def test_skipped_when_both_missing(self, clean_env: pytest.MonkeyPatch) -> None:
        """No env, no monkeypatch — the function must be a clean no-op,
        not raise on the missing identity (CLI must still run commands
        that do not touch EDGAR, e.g. ``manage list``).
        """
        reload_settings()
        _configure_edgar_identity()  # must not raise

    @pytest.mark.security
    def test_bootstrap_never_logs_identity(
        self,
        clean_env: pytest.MonkeyPatch,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """EDGAR identity is PII.  No record at any level may contain
        either the resolved name or the resolved email.
        """
        clean_env.setenv("EDGAR_IDENTITY_NAME", "Secret User")
        clean_env.setenv("EDGAR_IDENTITY_EMAIL", "secret@private.org")
        reload_settings()

        captured: list[str] = []
        self._patch_set_identity(monkeypatch, captured)

        # Capture EVERYTHING — root + package logger — to be sure no
        # bootstrap helper sneaks an identity into a log record.
        with (
            caplog.at_level(logging.DEBUG, logger="sec_generative_search"),
            caplog.at_level(logging.DEBUG, logger="root"),
        ):
            _configure_edgar_identity()

        all_records = "\n".join(record.getMessage() for record in caplog.records)
        assert "Secret User" not in all_records
        assert "secret@private.org" not in all_records
        # Make sure the test stub itself observed the call — otherwise the
        # privacy assertion is meaningless.
        assert captured == ["Secret User secret@private.org"]


# ---------------------------------------------------------------------------
# main() — signal handling
# ---------------------------------------------------------------------------


class TestSignalHandling:
    @pytest.mark.security
    def test_keyboard_interrupt_exits_130(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """POSIX SIGINT semantics: a Ctrl-C operator interrupt must exit
        with code 130, not Click's default 1.  This is the operator
        cancel-the-running-job contract.
        """

        def _raise() -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(main_module, "app", _raise)
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 130

    def test_broken_pipe_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``sec-rag ... | head`` closes the pipe early — that is
        normal operator usage, not a failure.
        """

        def _raise() -> None:
            raise BrokenPipeError

        monkeypatch.setattr(main_module, "app", _raise)
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_main_passes_through_normal_return(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[int] = []
        monkeypatch.setattr(main_module, "app", lambda: called.append(1))
        main()
        assert called == [1]

    def test_main_propagates_typer_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Typer translates command exits into ``SystemExit`` via Click —
        the wrapper must not swallow it.
        """

        def _raise() -> None:
            raise SystemExit(2)

        monkeypatch.setattr(main_module, "app", _raise)
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Command registration surface
# ---------------------------------------------------------------------------


class TestCommandRegistration:
    def test_adapted_operator_commands_registered(self) -> None:
        names = {c.name for c in app.registered_commands}
        assert {
            "reindex",
            "evict",
            "backup",
            "restore",
            "export",
            "import",
            "search",
        } <= names

    def test_ingest_sub_typer_registered(self) -> None:
        """The ingest shell must surface ``ingest`` as a group carrying
        ``add`` and ``batch`` subcommands, not as a top-level command."""
        group_names = {g.name for g in app.registered_groups}
        assert "ingest" in group_names

    def test_manage_sub_typer_registered(self) -> None:
        """``manage`` must be registered as a sub-Typer carrying ``status`` /
        ``list`` / ``remove`` / ``clear``."""
        group_names = {g.name for g in app.registered_groups}
        assert "manage" in group_names

    def test_rag_sub_typer_registered(self) -> None:
        """``rag`` must be registered as a sub-Typer carrying ``query``
        (and later ``chat``)."""
        group_names = {g.name for g in app.registered_groups}
        assert "rag" in group_names

    def test_unfinished_groups_not_yet_registered(self) -> None:
        """The shell must not expose unfinished groups.

        Registering a half-finished surface would be misleading to
        operators.
        """
        command_names = {c.name for c in app.registered_commands}
        group_names = {g.name for g in app.registered_groups}
        forbidden = {"provider"}
        assert command_names.isdisjoint(forbidden)
        assert group_names.isdisjoint(forbidden)
