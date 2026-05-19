"""Synchronous CLI wrapper over :class:`RetrievalService`.

Single operator-facing subcommand:

- ``sec-rag search QUERY`` — run a single-query retrieval and render the
  ranked hits in a Rich table.

Trust model and load-bearing rules:

1. The embedder is built via :func:`build_embedder` (the sole
   construction seam).  Direct adapter instantiation is forbidden here
   for the same reason it is forbidden in :mod:`cli.ingest` — every
   wiring site must route through the resolver so the credential seam
   stays consistent across surfaces.
2. The :class:`ChromaDBClient` is sealed with the same
   ``(provider, model, dimension)`` triple resolved from
   :class:`ProviderRegistry`.  Opening a stamped collection is the only
   way :class:`RetrievalService` will accept the client — the seal is
   the embedding-identity invariant.
3. The CLI runs at operator scope: no auth gate, no per-session EDGAR
    identity, no rate limit, no access-log redaction.  Distributing CLI
    access to team users is unsupported.
4. ISO date filters validate at the boundary so a malformed value
   (e.g. ``"2024-13-99"``) does not slip into ChromaDB's integer
   ``filing_date_int`` coercion path.  :class:`RetrievalService`
   re-validates defensively; failing fast here keeps the error
   attributable to the caller.

Output discipline mirrors the rest of the adapted CLI surface:
operator-facing strings flow through :func:`rich.markup.escape` so
incidental square brackets (section paths like ``"[redacted]"``,
ticker punctuation, install hints) survive Rich's markup parser.

Logging discipline: the query is Tier 3 user-generated data and never
reaches the operator log unredacted — :class:`RetrievalService` honours
``LOG_REDACT_QUERIES`` via :func:`redact_for_log`.  This module never
calls :func:`logger.info` / :func:`audit_log` directly with the query.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    DatabaseError,
    ProviderError,
    SearchError,
)
from sec_generative_search.core.types import EmbedderStamp, RetrievalResult
from sec_generative_search.database import ChromaDBClient
from sec_generative_search.providers.factory import build_embedder
from sec_generative_search.providers.registry import ProviderRegistry
from sec_generative_search.search import RetrievalService

__all__ = ["search"]


console = Console()

# Display caps for the rendered table.  Mirrors the legacy ergonomics —
# section paths and content previews are heavily truncated so a single
# wide chunk never blows up the operator terminal.
_CONTENT_PREVIEW_LIMIT = 1000
_SECTION_PATH_LIMIT = 500


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _print_error(
    label: str,
    message: str,
    *,
    details: str | None = None,
    hint: str | None = None,
) -> None:
    """Render an error with optional details and a single hint line.

    Shape matches the helper in :mod:`cli.ingest` / :mod:`cli.manage`
    so a future consolidation can lift it verbatim.  Every
    operator-facing string passes through :func:`rich.markup.escape` so
    accession numbers / install hints with literal square brackets
    render verbatim instead of being silently stripped by Rich's markup
    parser.
    """
    console.print(f"[red]{escape(label)}:[/red] {escape(message)}")
    if details:
        console.print(f"  [dim]{escape(details)}[/dim]")
    if hint:
        console.print(f"  [dim italic]Hint: {escape(hint)}[/dim italic]")


def _similarity_text(similarity: float) -> Text:
    """Colour-coded similarity percentage.

    Green for >= 40%, yellow for >= 25%, dim otherwise.  These bands
    reflect typical cosine ranges from sentence-level embeddings on SEC
    text — exact tuning is operator-visual; absolute scores are not
    commensurable across embedders so do not over-fit.
    """
    pct = f"{similarity:.1%}"
    if similarity >= 0.40:
        return Text(pct, style="bold green")
    if similarity >= 0.25:
        return Text(pct, style="yellow")
    return Text(pct, style="dim")


def _validate_date(value: str | None, param_name: str) -> str | None:
    """Validate ``YYYY-MM-DD`` strings at the CLI boundary.

    :class:`RetrievalService` will also reject malformed dates, but
    failing here surfaces a :class:`typer.BadParameter` so the error
    renders consistently with the rest of the CLI.  This is the same
    pattern :mod:`cli.ingest` uses for its date flags.
    """
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")  # noqa: DTZ007 — naive ISO date is intentional
    except ValueError:
        raise typer.BadParameter(
            f"Invalid date format for {param_name}: {value!r}. Expected YYYY-MM-DD."
        ) from None
    return value


# ---------------------------------------------------------------------------
# Service construction
# ---------------------------------------------------------------------------


def _build_service() -> RetrievalService:
    """Construct a stamp-sealed :class:`RetrievalService`.

    Failures surface as single operator-facing envelopes — the caller
    never sees a stack trace for a misconfigured environment.  The
    chain is identical to :mod:`cli.ingest`'s ``_build_pipeline``:

    1. Resolve ``(provider, model, dimension)`` from the registry.
    2. Build the embedder through :func:`build_embedder` (sole seam).
    3. Open :class:`ChromaDBClient` under the resulting stamp; the
       client's ``_verify_stamp`` refuses any mismatch.

    The :class:`MetadataRegistry` is not opened — search is a read-only
    path over ChromaDB and never touches SQLite directly.
    """
    settings = get_settings()
    embedding = settings.embedding

    try:
        target_dim = ProviderRegistry.get_dimension(embedding.provider, embedding.model_name)
    except (KeyError, ValueError) as exc:
        _print_error(
            "Embedder configuration invalid",
            f"Cannot resolve dimension for {embedding.provider}/{embedding.model_name}.",
            details=str(exc),
            hint=(
                "Check EMBEDDING_PROVIDER and EMBEDDING_MODEL_NAME against "
                "the registry — defaults live in providers/registry.py."
            ),
        )
        raise typer.Exit(code=1) from None

    try:
        embedder = build_embedder(embedding)
    except ConfigurationError as exc:
        _print_error(
            "Embedder construction failed",
            exc.message,
            hint="Set the expected API-key env var for this provider.",
        )
        raise typer.Exit(code=1) from None
    except KeyError as exc:
        _print_error(
            "Embedder unavailable",
            f"Provider {embedding.provider!r} requires additional packages.",
            details=str(exc),
            hint=(
                "Install the matching extra, e.g. "
                "`uv pip install -e '.[local-embeddings]'` for the local provider."
            ),
        )
        raise typer.Exit(code=1) from None

    stamp = EmbedderStamp(
        provider=embedding.provider,
        model=embedding.model_name,
        dimension=target_dim,
    )

    try:
        chroma = ChromaDBClient(stamp)
    except DatabaseError as exc:
        _print_error(
            "Storage initialisation failed",
            exc.message,
            details=exc.details,
            hint=(
                "Check DB_CHROMA_PATH is readable and that the existing "
                "collection's embedder stamp matches EMBEDDING_*."
            ),
        )
        raise typer.Exit(code=1) from None

    return RetrievalService(embedder=embedder, chroma_client=chroma)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_results(results: list[RetrievalResult], query: str) -> None:
    """Render the ranked hits in a Rich table.

    Mirrors the legacy presentation (similarity badge, source column,
    section path, truncated preview) so the operator visual contract is
    preserved across the rewrite.  ``rerank_score`` is intentionally
    omitted from the table — no reranker is bound in this CLI surface
    and surfacing a permanently-empty column would mislead operators.
    """
    console.print(
        f"\n[bold]Found {len(results)} result(s)[/bold] for: "
        f"[italic]{escape(query)}[/italic]\n"
    )

    table = Table(show_lines=True, expand=True, border_style="dim")
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Similarity", width=10, justify="right")
    table.add_column("Source", style="cyan", width=20, no_wrap=True)
    table.add_column("Section", style="dim", max_width=30)
    table.add_column("Content")

    for i, result in enumerate(results, 1):
        source = f"{result.ticker} {result.form_type}"
        if result.filing_date:
            source += f"\n{result.filing_date}"

        content = result.content
        if len(content) > _CONTENT_PREVIEW_LIMIT:
            content = content[:_CONTENT_PREVIEW_LIMIT] + "..."

        section = result.path
        if len(section) > _SECTION_PATH_LIMIT:
            section = section[:_SECTION_PATH_LIMIT] + "..."

        table.add_row(
            str(i),
            _similarity_text(result.similarity),
            escape(source),
            escape(section),
            escape(content),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Public command
# ---------------------------------------------------------------------------


def search(
    query: Annotated[str, typer.Argument(help="Natural language search query.")],
    top: Annotated[
        int | None,
        typer.Option("--top", "-t", help="Number of results to return.", min=1),
    ] = None,
    ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker", "-k", help="Filter by ticker symbol(s). Repeat for multiple."
        ),
    ] = None,
    form: Annotated[
        list[str] | None,
        typer.Option(
            "--form", "-f", help="Filter by form type(s). Repeat for multiple."
        ),
    ] = None,
    accession: Annotated[
        list[str] | None,
        typer.Option(
            "--accession",
            "-a",
            help="Restrict search to specific filing(s) by accession number.",
        ),
    ] = None,
    start_date: Annotated[
        str | None,
        typer.Option(
            "--start-date",
            help="Filter results to filings on or after this date (YYYY-MM-DD).",
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(
            "--end-date",
            help="Filter results to filings on or before this date (YYYY-MM-DD).",
        ),
    ] = None,
) -> None:
    """Search ingested SEC filings with a natural-language query.

    Examples:

        sec-rag search "risk factors related to supply chain"

        sec-rag search "revenue recognition" -t 10 -k AAPL

        sec-rag search "revenue" -k AAPL -k MSFT

        sec-rag search "liquidity" -f 10-Q

        sec-rag search "debt covenants" -a 0000320193-23-000106

        sec-rag search "revenue" --start-date 2023-01-01 --end-date 2023-12-31
    """
    # Reject empty / whitespace-only queries before constructing any
    # storage — the service would raise SearchError anyway, but failing
    # here saves an embedder build for a known-bad invocation.
    if not query or not query.strip():
        _print_error(
            "Invalid query",
            "Query must not be empty.",
        )
        raise typer.Exit(code=1)

    _validate_date(start_date, "--start-date")
    _validate_date(end_date, "--end-date")

    # Normalise filters to uppercase to match the metadata stored on
    # the ChromaDB collection (tickers and form types are uppercased at
    # ingest time).  Accession numbers are case-stable and pass through.
    ticker_filter = [t.upper() for t in ticker] if ticker else None
    form_filter = [f.upper() for f in form] if form else None
    accession_filter = list(accession) if accession else None

    service = _build_service()

    with console.status("Searching..."):
        try:
            results = service.retrieve(
                query,
                top_k=top,
                ticker=ticker_filter,
                form_type=form_filter,
                accession_number=accession_filter,
                start_date=start_date,
                end_date=end_date,
            )
        except SearchError as exc:
            _print_error(
                "Search failed",
                exc.message,
                details=exc.details,
                hint=(
                    "Ensure filings have been ingested with "
                    "'sec-rag ingest add' and that --start-date / --end-date "
                    "are valid YYYY-MM-DD values."
                ),
            )
            raise typer.Exit(code=1) from None
        except ProviderError as exc:
            # Embedding-side failure — the corpus and storage layer are
            # fine, the embedder upstream is not.  Mirror the
            # ``/api/search`` mapping (502 there) with a single
            # operator-facing line here.
            _print_error(
                "Embedding provider failure",
                "The embedding provider failed while processing the query.",
                details=exc.message,
                hint=(
                    "Retry after a short backoff; if the failure persists, "
                    "check the embedder API-key env var."
                ),
            )
            raise typer.Exit(code=1) from None
        except DatabaseError as exc:
            _print_error(
                "Database failure",
                exc.message,
                details=exc.details,
                hint="Check DB_CHROMA_PATH is readable and the collection is intact.",
            )
            raise typer.Exit(code=1) from None

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        console.print(
            "[dim italic]Hint: Try a broader query, or check that filings are "
            "ingested with 'sec-rag manage status'.[/dim italic]"
        )
        return

    _render_results(results, query)
