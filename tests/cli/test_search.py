"""Tests for ``sec-rag search``.

The retrieval primitives (:class:`RetrievalService`, the
``build_embedder`` factory seam, :class:`ChromaDBClient`) are stubbed
at the ``cli.search`` import site — stubbing at the import site (rather
than at the source modules) proves the CLI wires the right seams.

Goals:

- The embedder is built via :func:`build_embedder` and never through
  direct adapter instantiation.
- The :class:`ChromaDBClient` is opened under the
  ``(provider, model, dimension)`` stamp resolved from the registry.
- Filters are normalised at the boundary (ticker / form uppercased,
  accession passed through, ISO dates validated).
- Failure paths map to single operator-facing envelopes
  (:class:`SearchError`, :class:`ProviderError`, :class:`DatabaseError`,
  :class:`ConfigurationError`).
- Security: the raw query never lands in stdout as part of an error
  envelope; API keys / EDGAR identity never surface in any output path.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

import sec_generative_search.cli.search as search_module
from sec_generative_search.cli.search import search
from sec_generative_search.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
)
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    DatabaseError,
    ProviderError,
    SearchError,
)
from sec_generative_search.core.types import (
    ContentType,
    EmbedderStamp,
    RetrievalResult,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _make_hit(
    *,
    ticker: str = "AAPL",
    form_type: str = "10-K",
    filing_date: str = "2024-01-15",
    accession: str = "0000320193-24-000001",
    path: str = "Part I > Item 1A > Risk Factors",
    content: str = "Sample chunk content describing risk factors.",
    similarity: float = 0.55,
    chunk_id: str = "0000320193-24-000001:42",
) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        path=path,
        content_type=ContentType.TEXT,
        ticker=ticker,
        form_type=form_type,
        similarity=similarity,
        filing_date=filing_date,
        accession_number=accession,
        chunk_id=chunk_id,
        token_count=12,
    )


class _FakeEmbedder:
    """Stand-in for a hosted embedder.

    Construction is recorded; the CLI never calls ``embed_query``
    directly — that path runs inside the fake service.
    """

    instances: ClassVar[list[_FakeEmbedder]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _FakeEmbedder.instances.append(self)


class _FakeChroma:
    """Recording :class:`ChromaDBClient` stand-in.

    The CLI only opens the client to seal it under the resolved stamp;
    the retrieval path runs through the fake service that the CLI
    constructs around this stub.
    """

    instances: ClassVar[list[_FakeChroma]] = []

    def __init__(self, stamp: EmbedderStamp, *, chroma_path: str | None = None) -> None:
        self.stamp = stamp
        self.chroma_path = chroma_path
        _FakeChroma.instances.append(self)


class _FakeRetrievalService:
    """Recording :class:`RetrievalService` stand-in.

    ``retrieve`` records every call (so tests can assert on the
    normalised filters that reach the service) and returns
    ``returns``.  ``raises`` lets a test arm the failure path.
    """

    instances: ClassVar[list[_FakeRetrievalService]] = []

    def __init__(self, *, embedder: Any, chroma_client: Any) -> None:
        self.embedder = embedder
        self.chroma_client = chroma_client
        self.calls: list[dict[str, Any]] = []
        self.returns: list[RetrievalResult] = []
        self.raises: BaseException | None = None
        _FakeRetrievalService.instances.append(self)

    def retrieve(self, query: str, **kwargs: Any) -> list[RetrievalResult]:
        self.calls.append({"query": query, **kwargs})
        if self.raises is not None:
            raise self.raises
        return list(self.returns)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def app() -> typer.Typer:
    """Throw-away Typer app wrapping the ``search`` command.

    Mirrors the test harness in ``tests/cli/test_manage.py`` — wrapping
    keeps each test independent of ``sec-rag``'s global registration
    order.

    A second no-op command is registered alongside ``search`` so the
    Typer app stays in *group* mode.  With a single command Typer drops
    the command name from ``argv``, which would invert the
    ``["search", QUERY]`` invocation shape used by the rest of the
    CLI and across every other adapted CLI test file.
    """
    test_app = typer.Typer()
    test_app.command(name="search")(search)

    @test_app.command(name="_noop", hidden=True)
    def _noop() -> None:  # pragma: no cover — keeps Typer in group mode
        ...

    return test_app


class _StubSettings:
    """Hand-built settings stand-in matching the seams the CLI reads."""

    def __init__(
        self,
        *,
        embedding_provider: str = "openai",
        embedding_model_name: str = "text-embedding-3-small",
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
            max_filings=10000,
            encryption_key=None,
            encryption_key_file=None,
            task_history_retention_days=0,
            task_history_persist_tickers=False,
        )


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace every seam ``cli.search`` reaches into with a recording stub.

    Returning the instance lists lets each test assert on:

    - ``build_embedder`` was the one called (direct adapter
      instantiation would never appear in ``_FakeEmbedder.instances``).
    - The :class:`ChromaDBClient` stamp matches the
      ``(provider, model, dimension)`` resolved from the registry.
    - The :class:`RetrievalService` saw the post-normalised filters.
    """
    _FakeEmbedder.instances.clear()
    _FakeChroma.instances.clear()
    _FakeRetrievalService.instances.clear()

    monkeypatch.setattr(search_module, "get_settings", _StubSettings)
    monkeypatch.setattr(
        search_module,
        "build_embedder",
        lambda _settings: _FakeEmbedder(),
    )
    monkeypatch.setattr(search_module, "ChromaDBClient", _FakeChroma)
    monkeypatch.setattr(search_module, "RetrievalService", _FakeRetrievalService)

    return {
        "embedders": _FakeEmbedder.instances,
        "chromas": _FakeChroma.instances,
        "services": _FakeRetrievalService.instances,
    }


def _arm_returns(results: list[RetrievalResult]) -> None:
    """Arm the next-constructed :class:`_FakeRetrievalService` with results.

    Mirrors the ``_seed_registry`` helper in ``test_manage.py`` — we
    cannot pre-create the service instance because the CLI constructs
    it inside ``_build_service``.
    """
    original = _FakeRetrievalService.__init__

    def _seeded(self: _FakeRetrievalService, **kwargs: Any) -> None:
        original(self, **kwargs)
        self.returns = list(results)

    _FakeRetrievalService.__init__ = _seeded  # type: ignore[method-assign]


def _arm_raises(exc: BaseException) -> None:
    """Arm the next-constructed service to raise on ``retrieve``."""
    original = _FakeRetrievalService.__init__

    def _seeded(self: _FakeRetrievalService, **kwargs: Any) -> None:
        original(self, **kwargs)
        self.raises = exc

    _FakeRetrievalService.__init__ = _seeded  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _reset_service_init() -> Any:
    """Each test gets a fresh service constructor so armed state never leaks."""
    yield

    def _default(self: _FakeRetrievalService, **kwargs: Any) -> None:
        self.embedder = kwargs.get("embedder")
        self.chroma_client = kwargs.get("chroma_client")
        self.calls = []
        self.returns = []
        self.raises = None
        _FakeRetrievalService.instances.append(self)

    _FakeRetrievalService.__init__ = _default  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_results_render_in_a_table(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_returns(
            [
                _make_hit(content="Apple risk factor #1"),
                _make_hit(
                    ticker="MSFT",
                    accession="0000789019-24-000002",
                    content="Microsoft risk factor #2",
                    similarity=0.31,
                ),
            ]
        )
        result = runner.invoke(app, ["search", "supply chain risk"])
        assert result.exit_code == 0, result.output
        assert "Found 2 result(s)" in result.output
        assert "AAPL" in result.output
        assert "MSFT" in result.output
        # Service was constructed with the stamp-sealed chroma client.
        svc = patched["services"][-1]
        assert svc.chroma_client is patched["chromas"][-1]
        # Embedder was built through ``build_embedder`` (sole seam).
        assert len(patched["embedders"]) == 1

    def test_no_results_renders_yellow_message(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_returns([])
        result = runner.invoke(app, ["search", "no matches"])
        assert result.exit_code == 0, result.output
        assert "No results found" in result.output

    def test_stamp_matches_registry_dimension(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``ChromaDBClient`` MUST open under the dimension the registry
        prescribes for ``(provider, model)`` — never an operator
        override.  Mirrors :mod:`cli.manage`'s ``_resolve_stamp``
        invariant.
        """
        _arm_returns([])
        result = runner.invoke(app, ["search", "hello"])
        assert result.exit_code == 0
        chroma = patched["chromas"][-1]
        # text-embedding-3-small is 1536-dim in the registry.
        assert chroma.stamp.provider == "openai"
        assert chroma.stamp.model == "text-embedding-3-small"
        assert chroma.stamp.dimension == 1536


# ---------------------------------------------------------------------------
# Filter normalisation
# ---------------------------------------------------------------------------


class TestFilterNormalisation:
    def test_ticker_and_form_uppercased(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_returns([])
        result = runner.invoke(
            app,
            ["search", "revenue", "-k", "aapl", "-k", "msft", "-f", "10-k"],
        )
        assert result.exit_code == 0
        call = patched["services"][-1].calls[-1]
        assert call["ticker"] == ["AAPL", "MSFT"]
        assert call["form_type"] == ["10-K"]

    def test_accession_passed_through(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_returns([])
        result = runner.invoke(
            app,
            [
                "search",
                "debt",
                "-a",
                "0000320193-23-000106",
                "-a",
                "0000789019-24-000002",
            ],
        )
        assert result.exit_code == 0
        call = patched["services"][-1].calls[-1]
        assert call["accession_number"] == [
            "0000320193-23-000106",
            "0000789019-24-000002",
        ]

    def test_top_flag_propagates(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_returns([])
        result = runner.invoke(app, ["search", "query", "-t", "3"])
        assert result.exit_code == 0
        assert patched["services"][-1].calls[-1]["top_k"] == 3

    def test_date_range_propagates(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_returns([])
        result = runner.invoke(
            app,
            [
                "search",
                "liquidity",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
            ],
        )
        assert result.exit_code == 0
        call = patched["services"][-1].calls[-1]
        assert call["start_date"] == "2023-01-01"
        assert call["end_date"] == "2023-12-31"

    def test_retrieval_tuning_flags_propagate(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        # The diversity-cap + over-fetch flags reach the service
        # verbatim. Omitted flags pass None so the service falls back to
        # settings.search.
        _arm_returns([])
        result = runner.invoke(
            app,
            [
                "search",
                "liquidity",
                "--max-per-section",
                "3",
                "--max-per-filing",
                "2",
                "--rerank-over-fetch",
                "6",
            ],
        )
        assert result.exit_code == 0, result.output
        call = patched["services"][-1].calls[-1]
        assert call["max_per_section"] == 3
        assert call["max_per_filing"] == 2
        assert call["rerank_over_fetch_factor"] == 6

    def test_retrieval_tuning_flags_default_to_none(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_returns([])
        result = runner.invoke(app, ["search", "liquidity"])
        assert result.exit_code == 0, result.output
        call = patched["services"][-1].calls[-1]
        assert call["max_per_section"] is None
        assert call["max_per_filing"] is None
        assert call["rerank_over_fetch_factor"] is None


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


class TestBoundaryValidation:
    def test_empty_query_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["search", "   "])
        assert result.exit_code == 1
        assert "Invalid query" in result.output
        # Boundary rejection fires before any storage / embedder build.
        assert patched["embedders"] == []
        assert patched["chromas"] == []

    def test_malformed_start_date_exits_2(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Typer translates :class:`typer.BadParameter` into exit 2 with
        the standard Click error envelope.  We do not normalise away
        from that — it is the consistent CLI shape.
        """
        result = runner.invoke(
            app,
            ["search", "query", "--start-date", "2024-13-99"],
        )
        assert result.exit_code == 2
        assert "Invalid date format" in result.output

    def test_malformed_end_date_exits_2(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            ["search", "query", "--end-date", "not-a-date"],
        )
        assert result.exit_code == 2
        assert "Invalid date format" in result.output

    def test_negative_diversity_cap_exits_2(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        # Typer's ``min=0`` rejects a negative cap with the standard
        # Click error envelope (exit 2) before any storage is opened.
        result = runner.invoke(app, ["search", "query", "--max-per-section", "-1"])
        assert result.exit_code == 2
        assert patched["embedders"] == []

    def test_over_fetch_below_one_exits_2(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        # ``min=1`` on --rerank-over-fetch rejects a 0 multiplier.
        result = runner.invoke(app, ["search", "query", "--rerank-over-fetch", "0"])
        assert result.exit_code == 2
        assert patched["embedders"] == []


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_search_error_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_raises(SearchError("Bad filter", details="ticker has no value"))
        result = runner.invoke(app, ["search", "query"])
        assert result.exit_code == 1
        assert "Search failed" in result.output

    def test_provider_error_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_raises(ProviderError("embed broken"))
        result = runner.invoke(app, ["search", "query"])
        assert result.exit_code == 1
        assert "Embedding provider failure" in result.output

    def test_database_error_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_raises(DatabaseError("disk error"))
        result = runner.invoke(app, ["search", "query"])
        assert result.exit_code == 1
        assert "Database failure" in result.output

    def test_chroma_init_failure_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _BoomChroma:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise DatabaseError("stamp mismatch")

        monkeypatch.setattr(search_module, "ChromaDBClient", _BoomChroma)
        result = runner.invoke(app, ["search", "query"])
        assert result.exit_code == 1
        assert "Storage initialisation failed" in result.output

    def test_embedder_configuration_error_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(_settings: Any) -> None:
            raise ConfigurationError("missing OPENAI_API_KEY")

        monkeypatch.setattr(search_module, "build_embedder", _raise)
        result = runner.invoke(app, ["search", "query"])
        assert result.exit_code == 1
        assert "Embedder construction failed" in result.output

    def test_embedder_missing_extra_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(_settings: Any) -> None:
            raise KeyError("local")

        monkeypatch.setattr(search_module, "build_embedder", _raise)
        result = runner.invoke(app, ["search", "query"])
        assert result.exit_code == 1
        assert "Embedder unavailable" in result.output
        # Install hint with literal brackets must survive Rich's markup parser.
        assert ".[local-embeddings]" in result.output

    def test_unknown_embedding_dimension_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sec_generative_search.providers.registry import ProviderRegistry

        def _raise(_provider: str, _model: str) -> int:
            raise ValueError("Unknown (provider, model) pair")

        monkeypatch.setattr(ProviderRegistry, "get_dimension", staticmethod(_raise))
        result = runner.invoke(app, ["search", "query"])
        assert result.exit_code == 1
        assert "Embedder configuration invalid" in result.output


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecurity:
    def test_api_key_never_appears_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An OPENAI_API_KEY in the environment must never leak —
        including on the loudest error paths.  The CLI never logs the
        admin-env credential in any branch.
        """
        canary = "sk-canary-token-must-not-appear-in-output"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        _arm_raises(SearchError("boom", details="broken"))
        result = runner.invoke(app, ["search", "query with sensitive data"])
        assert canary not in result.output

    def test_edgar_identity_never_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """EDGAR identity is PII — it must not appear in any CLI output."""
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Secret User")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "canary-edgar@example.test")
        _arm_returns([])
        result = runner.invoke(app, ["search", "query"])
        assert "Secret User" not in result.output
        assert "canary-edgar@example.test" not in result.output

    def test_embedder_built_through_factory_seam(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """The CLI MUST construct the embedder through
        :func:`build_embedder` — not via direct adapter instantiation.
        The fixture replaces ``build_embedder`` with a recorder so this
        is the only construction path observable to the test; if the
        CLI ever side-stepped the seam the recorder would be empty.
        """
        _arm_returns([])
        result = runner.invoke(app, ["search", "query"])
        assert result.exit_code == 0
        assert len(patched["embedders"]) == 1

    def test_query_string_not_echoed_on_error_envelope(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Error envelopes carry the message + hint, not the raw query.

        Mirrors the ``/api/search`` discipline: the response NEVER
        echoes the raw query.  A different rule on the CLI would be
        surprising — operators piping CLI output into logs should not
        find queries inlined into error lines.
        """
        sentinel = "PII-LADEN-QUERY-DO-NOT-ECHO"
        _arm_raises(SearchError("boom"))
        result = runner.invoke(app, ["search", sentinel])
        assert result.exit_code == 1
        assert sentinel not in result.output


# ---------------------------------------------------------------------------
# Rendering markup safety
# ---------------------------------------------------------------------------


class TestMarkupSafety:
    def test_rich_markup_in_content_renders_verbatim(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """A retrieved chunk that happens to contain ``[red]...[/red]``
        MUST render literally — Rich must not interpret retrieved text
        as markup.  Otherwise an attacker who manages to land hostile
        text into an SEC filing could repaint operator output.
        """
        _arm_returns([_make_hit(content="Risk factor: [red]EVIL[/red]")])
        result = runner.invoke(app, ["search", "query"])
        assert result.exit_code == 0, result.output
        # The brackets survive into the rendered output.
        assert "[red]EVIL[/red]" in result.output
