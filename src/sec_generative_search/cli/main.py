"""Typer application root for the ``sec-rag`` CLI.

The CLI is an operator-on-host tool that bypasses API auth, rate limiting,
per-session EDGAR identity, and access-log redaction. It only registers the
operator commands that are already adapted in their owning modules.
"""

from __future__ import annotations

import logging
import sys
from importlib.metadata import PackageNotFoundError, version

import typer
from rich.console import Console
from rich.markup import escape

from sec_generative_search.cli.backup import backup, restore
from sec_generative_search.cli.evict import evict
from sec_generative_search.cli.ingest import ingest_app
from sec_generative_search.cli.manage import manage_app
from sec_generative_search.cli.portable import export, import_
from sec_generative_search.cli.rag import rag_app
from sec_generative_search.cli.reindex import reindex
from sec_generative_search.cli.search import search
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.logging import configure_logging, suppress_third_party_loggers

# Shared console — every CLI module imports this for consistent output.
console = Console()

app = typer.Typer(
    name="sec-rag",
    help="Security-first RAG over SEC filings (10-K, 10-Q, 8-K and amendments).",
    no_args_is_help=True,
)


def _resolve_version() -> str:
    """Return the installed package version, or ``"unknown"`` outside a wheel."""
    try:
        return version("sec-generative-search")
    except PackageNotFoundError:
        return "unknown"


def _version_callback(value: bool) -> None:
    """Print the version and exit cleanly.

    Eager — fires before the root callback so ``sec-rag --version`` never
    touches settings, EDGAR identity, or any side-effecting bootstrap.
    """
    if value:
        console.print(escape(f"sec-rag {_resolve_version()}"))
        raise typer.Exit()


def _verbose_callback(value: bool) -> None:
    """Switch the package logger to DEBUG when ``--verbose`` is passed.

    Eager — applies before any subcommand executes, so subcommand logs are
    captured at DEBUG from their very first line.
    """
    if value:
        configure_logging(level=logging.DEBUG)
        suppress_third_party_loggers()


def _configure_edgar_identity() -> None:
    """Apply ``edgar.set_identity`` once at CLI bootstrap.

    ``edgar.set_identity`` mutates *process-global* state.  Calling it once
    here gives every downstream command (``reindex``, future ``ingest``,
    etc.) a consistent EDGAR identity without each command re-discovering
    the env vars.

    The call fires **only** when both ``EDGAR_IDENTITY_NAME`` and
    ``EDGAR_IDENTITY_EMAIL`` are set — Scenarios B/C deliberately leave
    them unset and supply identity per-session via the HTTP layer.

    Privacy: the resolved identity is **never** logged.  EDGAR name + email
    are PII and must not appear in any log record at any level.
    """
    settings = get_settings()
    name = settings.edgar.identity_name
    email = settings.edgar.identity_email
    if name and email:
        # Local import — edgartools pulls a heavy dependency tree we do not
        # want loaded on ``sec-rag --version`` / ``--help``.
        from edgar import set_identity

        set_identity(f"{name} {email}")


@app.callback()
def _root(
    _version_flag: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    _verbose_flag: bool = typer.Option(
        False,
        "--verbose",
        help="Enable detailed DEBUG-level logging.",
        callback=_verbose_callback,
        is_eager=True,
    ),
) -> None:
    """Apply CLI-wide bootstrap before any subcommand runs.

    ``--version`` exits before this callback (eager).  ``--help`` is
    short-circuited by Click before this callback runs.  So bootstrap
    side effects fire only for an actual subcommand invocation.
    """
    _configure_edgar_identity()


# Register adapted operator commands as top-level entries.  The ``import``
# function is renamed in Python (``import_``) to dodge the keyword clash,
# but the CLI surface keeps the natural name.
app.command(name="reindex")(reindex)
app.command(name="evict")(evict)
app.command(name="backup")(backup)
app.command(name="restore")(restore)
app.command(name="export")(export)
app.command(name="import")(import_)
app.command(name="search")(search)

# ``ingest`` is a sub-Typer (``ingest add`` / ``ingest batch``) — it is
# wired via ``add_typer`` so the two subcommands surface as ``sec-rag
# ingest add ...`` / ``sec-rag ingest batch ...``.
app.add_typer(ingest_app, name="ingest")

# ``manage`` is a sub-Typer (``manage status`` / ``list`` / ``remove`` /
# ``clear``) — wired the same way so the four subcommands surface as
# ``sec-rag manage <verb> ...``.
app.add_typer(manage_app, name="manage")

# ``rag`` is a sub-Typer (``rag query`` now; ``rag chat`` is wired later)
# — wired so the verbs surface as ``sec-rag rag query ...``.
app.add_typer(rag_app, name="rag")


def main() -> None:
    """Entry point — wraps ``app()`` so ``KeyboardInterrupt`` maps to exit 130.

    Click's default ``KeyboardInterrupt`` handling prints ``Aborted!`` to
    stderr and exits 1.  Operator convention (and POSIX SIGINT semantics)
    is exit 130, so we intercept here.  ``BrokenPipeError`` (``sec-rag ... |
    head``) is silenced for the same operator-ergonomics reason.
    """
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
    except BrokenPipeError:
        sys.exit(0)
