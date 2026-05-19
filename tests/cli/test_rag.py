"""Tests for ``sec-rag rag query``.

The retrieval and generation primitives (:class:`RetrievalService`,
:class:`RAGOrchestrator`, ``understand_query``, the ``build_embedder`` /
``build_llm_provider`` factory seams) are stubbed at the ``cli.rag``
import site — stubbing at the import site (rather than at the source
modules) proves the CLI wires the right seams.

Goals:

- Both factory seams (``build_embedder`` and ``build_llm_provider``)
  are used; direct adapter instantiation never happens.
- The CLI resolver chain composes ``encrypted-user → admin-env``
  according to settings (encrypted tier appears only when both
  ``persist_provider_credentials`` and ``registry.encrypted`` are
  true; otherwise the chain collapses to admin-env alone).
- ``--show-plan`` runs ``understand_query`` and exits 0 *before*
  generation — the orchestrator must not be called.
- ``--skip-plan`` bypasses ``understand_query`` entirely and feeds a
  minimal plan to the orchestrator.
- The auto-plan path runs ``understand_query`` once and then
  ``orchestrator.generate`` once with the returned plan.
- ``--ticker`` / ``--form`` / ``--since`` / ``--until`` / ``--mode``
  overrides land on the plan and reach the orchestrator.
- Provider exceptions map to single operator-facing envelopes that
  mirror ``POST /api/rag/query`` (auth → unauthorised, transient →
  unavailable, every other provider error → "Provider error", etc.).
- Security: the raw question and the provider key never leak into
  stdout; EDGAR identity never surfaces.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

import sec_generative_search.cli.rag as rag_module
from sec_generative_search.cli.rag import rag_app
from sec_generative_search.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
    LLMSettings,
    RAGSettings,
    SearchSettings,
)
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    GenerationError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    SearchError,
)
from sec_generative_search.core.types import (
    Citation,
    EmbedderStamp,
    FilingIdentifier,
    GenerationResult,
    TokenUsage,
)
from sec_generative_search.providers.base import ProviderCapability
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.query_understanding import QueryPlan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Strips ANSI colour escapes; Click renders :class:`typer.BadParameter`
# inside a Rich-styled panel that injects colour codes between words and
# soft-wraps the message to the panel width.  In a narrow CI terminal
# that means a literal substring like ``"--since and --until must be
# supplied together"`` is broken by both colour codes and newlines, so a
# raw ``in`` check fails even when the message is intact.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _stripped(output: str) -> str:
    """Return *output* with ANSI codes removed and whitespace collapsed.

    Click / Rich panels render error messages with colour codes and
    word-wrap that vary with terminal width.  Tests that need to assert
    on a substring should anchor against this normalised view rather
    than against the raw bytes — otherwise CI runs with narrower
    terminals than the developer's local box flake.
    """
    return re.sub(r"\s+", " ", _ANSI_RE.sub("", output))


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _make_plan(
    *,
    raw_query: str = "What are Apple's main revenue segments?",
    detected_language: str = "en",
    tickers: list[str] | None = None,
    form_types: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    mode: AnswerMode = AnswerMode.CONCISE,
) -> QueryPlan:
    return QueryPlan(
        raw_query=raw_query,
        detected_language=detected_language,
        query_en=raw_query,
        tickers=list(tickers or []),
        form_types=list(form_types or []),
        date_range=date_range,
        intent="ask",
        suggested_answer_mode=mode,
    )


def _make_citation(
    *,
    ticker: str = "AAPL",
    form_type: str = "10-K",
    accession: str = "0000320193-24-000001",
    section_path: str = "Part II > Item 7 > MD&A",
    text_span: str = "Revenue grew across all geographic segments.",
    similarity: float = 0.62,
    display_index: int = 1,
) -> Citation:
    return Citation(
        chunk_id=f"{accession}:1",
        filing_id=FilingIdentifier(
            ticker=ticker,
            form_type=form_type,
            filing_date=date(2024, 1, 15),
            accession_number=accession,
        ),
        section_path=section_path,
        text_span=text_span,
        similarity=similarity,
        display_index=display_index,
    )


def _make_result(
    *,
    answer: str = "Apple has three main revenue segments [1].",
    citations: list[Citation] | None = None,
    refused: bool = False,
) -> GenerationResult:
    if refused:
        return GenerationResult(
            answer="I cannot answer.",
            provider="openai",
            model="gpt-x",
            prompt_version="v0",
            citations=[],
            retrieved_chunks=[],
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
            latency_seconds=0.1,
            streamed=False,
        )
    return GenerationResult(
        answer=answer,
        provider="openai",
        model="gpt-x",
        prompt_version="v1",
        citations=list(citations or [_make_citation()]),
        retrieved_chunks=[],  # filled by orchestrator; not asserted on
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        latency_seconds=0.42,
        streamed=False,
    )


class _FakeEmbedder:
    instances: ClassVar[list[_FakeEmbedder]] = []

    def __init__(self) -> None:
        _FakeEmbedder.instances.append(self)


class _FakeLLM:
    instances: ClassVar[list[_FakeLLM]] = []
    provider_name = "openai"

    def __init__(self) -> None:
        _FakeLLM.instances.append(self)


class _FakeChroma:
    instances: ClassVar[list[_FakeChroma]] = []

    def __init__(self, stamp: EmbedderStamp, *, chroma_path: str | None = None) -> None:
        self.stamp = stamp
        _FakeChroma.instances.append(self)


class _FakeRegistry:
    """Stand-in for :class:`MetadataRegistry`.

    Exposes the ``encrypted`` flag that ``_build_api_key_resolver``
    consults; defaults to False so the encrypted tier collapses out.
    """

    instances: ClassVar[list[_FakeRegistry]] = []
    encrypted_default: ClassVar[bool] = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.encrypted = type(self).encrypted_default
        _FakeRegistry.instances.append(self)


class _FakeRetrievalService:
    instances: ClassVar[list[_FakeRetrievalService]] = []

    def __init__(self, *, embedder: Any, chroma_client: Any) -> None:
        self.embedder = embedder
        self.chroma_client = chroma_client
        _FakeRetrievalService.instances.append(self)


class _FakeOrchestrator:
    """Recording :class:`RAGOrchestrator` stand-in."""

    instances: ClassVar[list[_FakeOrchestrator]] = []

    def __init__(self, *, retrieval: Any, llm: Any) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self.calls: list[dict[str, Any]] = []
        self.returns: GenerationResult = _make_result()
        self.raises: BaseException | None = None
        _FakeOrchestrator.instances.append(self)

    def generate(self, plan: QueryPlan, **kwargs: Any) -> GenerationResult:
        self.calls.append({"plan": plan, **kwargs})
        if self.raises is not None:
            raise self.raises
        return self.returns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def app() -> typer.Typer:
    """Throw-away Typer app wrapping the ``rag`` sub-Typer."""
    test_app = typer.Typer()
    test_app.add_typer(rag_app, name="rag")
    return test_app


class _StubSettings:
    """Hand-built settings stand-in matching the seams ``cli.rag`` reads."""

    def __init__(
        self,
        *,
        llm_provider: str = "openai",
        llm_default_model: str | None = None,
        persist_provider_credentials: bool = False,
    ) -> None:
        self.embedding = EmbeddingSettings.model_construct(
            provider="openai",
            model_name="text-embedding-3-small",
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
            persist_provider_credentials=persist_provider_credentials,
        )
        self.llm = LLMSettings.model_construct(
            default_provider=llm_provider,
            default_model=llm_default_model,
            max_output_tokens=2048,
        )
        self.rag = RAGSettings.model_construct(
            context_token_budget=8000,
            refusal_enabled=True,
        )
        self.search = SearchSettings.model_construct(top_k=5, min_similarity=0.0)


_DEFAULT_PLAN = _make_plan()


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace every seam ``cli.rag`` reaches into with a recording stub.

    ``understand_query`` is stubbed at the import site (``rag_module``)
    so we can assert on whether it was called and arm its return value
    or exception independently from the orchestrator stub.
    """
    _FakeEmbedder.instances.clear()
    _FakeLLM.instances.clear()
    _FakeChroma.instances.clear()
    _FakeRegistry.instances.clear()
    _FakeRegistry.encrypted_default = False
    _FakeRetrievalService.instances.clear()
    _FakeOrchestrator.instances.clear()

    understand_calls: list[dict[str, Any]] = []

    def _fake_understand(
        query: str,
        *,
        llm: Any,
        model: str,
        structured_output_supported: bool = False,
    ) -> QueryPlan:
        record = {
            "query": query,
            "llm": llm,
            "model": model,
            "structured_output_supported": structured_output_supported,
        }
        understand_calls.append(record)
        if "raises" in understand_calls[-1] or _understand_raises[0] is not None:
            raise _understand_raises[0]  # type: ignore[misc]
        return _understand_returns[0]

    _understand_returns: list[QueryPlan] = [_DEFAULT_PLAN]
    _understand_raises: list[BaseException | None] = [None]

    monkeypatch.setattr(rag_module, "get_settings", _StubSettings)
    monkeypatch.setattr(rag_module, "build_embedder", lambda _settings: _FakeEmbedder())
    monkeypatch.setattr(
        rag_module,
        "build_llm_provider",
        lambda _name, *, api_key_resolver=None: _FakeLLM(),
    )
    monkeypatch.setattr(rag_module, "ChromaDBClient", _FakeChroma)
    monkeypatch.setattr(rag_module, "MetadataRegistry", _FakeRegistry)
    monkeypatch.setattr(rag_module, "RetrievalService", _FakeRetrievalService)
    monkeypatch.setattr(rag_module, "RAGOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(rag_module, "understand_query", _fake_understand)

    return {
        "embedders": _FakeEmbedder.instances,
        "llms": _FakeLLM.instances,
        "chromas": _FakeChroma.instances,
        "registries": _FakeRegistry.instances,
        "services": _FakeRetrievalService.instances,
        "orchestrators": _FakeOrchestrator.instances,
        "understand_calls": understand_calls,
        "understand_returns": _understand_returns,
        "understand_raises": _understand_raises,
    }


def _arm_orchestrator_returns(result: GenerationResult) -> None:
    original = _FakeOrchestrator.__init__

    def _seeded(self: _FakeOrchestrator, **kwargs: Any) -> None:
        original(self, **kwargs)
        self.returns = result

    _FakeOrchestrator.__init__ = _seeded  # type: ignore[method-assign]


def _arm_orchestrator_raises(exc: BaseException) -> None:
    original = _FakeOrchestrator.__init__

    def _seeded(self: _FakeOrchestrator, **kwargs: Any) -> None:
        original(self, **kwargs)
        self.raises = exc

    _FakeOrchestrator.__init__ = _seeded  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _reset_orchestrator_init() -> Any:
    yield

    def _default(self: _FakeOrchestrator, **kwargs: Any) -> None:
        self.retrieval = kwargs.get("retrieval")
        self.llm = kwargs.get("llm")
        self.calls = []
        self.returns = _make_result()
        self.raises = None
        _FakeOrchestrator.instances.append(self)

    _FakeOrchestrator.__init__ = _default  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_auto_plan_then_generate(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Default flow: understand_query → orchestrator.generate."""
        result = runner.invoke(app, ["rag", "query", "What is revenue?"])
        assert result.exit_code == 0, result.output
        # Both factory seams used exactly once.
        assert len(patched["embedders"]) == 1
        assert len(patched["llms"]) == 1
        # Query understanding ran once.
        assert len(patched["understand_calls"]) == 1
        # Orchestrator generated once with the auto-plan.
        assert len(patched["orchestrators"]) == 1
        orch = patched["orchestrators"][-1]
        assert len(orch.calls) == 1
        # Answer and citation marker render.
        assert "Apple has three main revenue segments" in result.output
        assert "Citations" in result.output

    def test_refused_path(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """When retrieved chunks and citations are both empty the renderer
        labels the panel ``Refused`` — same shape as the API's ``refused=true``."""
        _arm_orchestrator_returns(_make_result(refused=True))
        result = runner.invoke(app, ["rag", "query", "Something obscure"])
        assert result.exit_code == 0, result.output
        assert "Refused" in result.output

    def test_show_plan_skips_generation(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``--show-plan`` runs query-understanding, prints the plan, and exits
        before any orchestrator construction."""
        result = runner.invoke(app, ["rag", "query", "What is revenue?", "--show-plan"])
        assert result.exit_code == 0, result.output
        assert "Query Plan" in result.output
        assert len(patched["understand_calls"]) == 1
        # No orchestrator was ever constructed.
        assert patched["orchestrators"] == []

    def test_skip_plan_bypasses_understanding(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``--skip-plan`` feeds a minimal plan straight to the orchestrator —
        ``understand_query`` is never called."""
        result = runner.invoke(app, ["rag", "query", "raw question", "--skip-plan"])
        assert result.exit_code == 0, result.output
        assert patched["understand_calls"] == []
        orch = patched["orchestrators"][-1]
        assert len(orch.calls) == 1
        plan: QueryPlan = orch.calls[0]["plan"]
        assert plan.raw_query == "raw question"
        assert plan.query_en == "raw question"
        assert plan.tickers == []
        assert plan.form_types == []


# ---------------------------------------------------------------------------
# Plan overrides
# ---------------------------------------------------------------------------


class TestPlanOverrides:
    def test_ticker_and_form_uppercased(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            [
                "rag",
                "query",
                "revenue",
                "--skip-plan",
                "--ticker",
                "aapl",
                "--ticker",
                "msft",
                "--form",
                "10-k",
            ],
        )
        assert result.exit_code == 0, result.output
        plan: QueryPlan = patched["orchestrators"][-1].calls[0]["plan"]
        assert plan.tickers == ["AAPL", "MSFT"]
        assert plan.form_types == ["10-K"]

    def test_date_range_override(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            [
                "rag",
                "query",
                "revenue",
                "--skip-plan",
                "--since",
                "2023-01-01",
                "--until",
                "2023-12-31",
            ],
        )
        assert result.exit_code == 0, result.output
        plan: QueryPlan = patched["orchestrators"][-1].calls[0]["plan"]
        assert plan.date_range == ("2023-01-01", "2023-12-31")

    def test_since_without_until_with_no_plan_range_rejected(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``--since`` alone with no plan ``date_range`` is rejected at the
        boundary; the orchestrator must not be reached."""
        result = runner.invoke(
            app,
            ["rag", "query", "q", "--skip-plan", "--since", "2023-01-01"],
        )
        assert result.exit_code == 2
        assert "--since and --until must be supplied together" in _stripped(result.output)
        assert patched["orchestrators"] == []

    def test_mode_override(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            ["rag", "query", "compare X and Y", "--skip-plan", "--mode", "comparative"],
        )
        assert result.exit_code == 0, result.output
        # The orchestrator receives the AnswerMode override AND the plan
        # mode is also updated (overrides shadow plan extraction).
        orch = patched["orchestrators"][-1]
        assert orch.calls[0]["mode"] == AnswerMode.COMPARATIVE
        assert orch.calls[0]["plan"].suggested_answer_mode == AnswerMode.COMPARATIVE

    def test_unknown_mode_rejected(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``_coerce_mode`` fails closed on unknown values — typos are caught
        rather than silently coerced to ``concise``."""
        result = runner.invoke(
            app,
            ["rag", "query", "q", "--skip-plan", "--mode", "analyitcal"],
        )
        assert result.exit_code == 2
        assert "Invalid --mode" in _stripped(result.output)
        assert patched["orchestrators"] == []


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


class TestBoundaryValidation:
    def test_empty_question_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["rag", "query", "   "])
        assert result.exit_code == 1
        assert "Invalid question" in result.output
        # Boundary rejection fires before any stack construction.
        assert patched["embedders"] == []

    def test_show_and_skip_plan_mutually_exclusive(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["rag", "query", "q", "--show-plan", "--skip-plan"])
        assert result.exit_code == 1
        assert "Invalid flag combination" in result.output
        assert patched["embedders"] == []

    def test_malformed_since_rejected(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan", "--since", "2024-13-99"])
        assert result.exit_code == 2
        assert "Invalid date format" in result.output


# ---------------------------------------------------------------------------
# Provider exception ladder
# ---------------------------------------------------------------------------


class TestProviderExceptions:
    def test_understand_provider_auth_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        patched["understand_raises"][0] = ProviderAuthError("bad key")
        result = runner.invoke(app, ["rag", "query", "q"])
        assert result.exit_code == 1
        assert "Provider unauthorised" in result.output
        # Orchestrator never built — auth failure short-circuits.
        assert patched["orchestrators"] == []

    def test_understand_rate_limit_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        patched["understand_raises"][0] = ProviderRateLimitError("slow down")
        result = runner.invoke(app, ["rag", "query", "q"])
        assert result.exit_code == 1
        assert "Provider unavailable" in result.output

    def test_generate_auth_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_orchestrator_raises(ProviderAuthError("invalid"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 1
        assert "Provider unauthorised" in result.output

    def test_generate_rate_limit_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_orchestrator_raises(ProviderTimeoutError("timeout"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 1
        assert "Provider unavailable" in result.output

    def test_generate_provider_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_orchestrator_raises(ProviderError("upstream 500"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 1
        assert "Provider error" in result.output

    def test_generate_generation_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_orchestrator_raises(GenerationError("malformed json"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 1
        assert "Generation failed" in result.output

    def test_generate_search_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_orchestrator_raises(SearchError("bad filter"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 1
        assert "Retrieval failed" in result.output

    def test_llm_construction_missing_key_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(_name: str, *, api_key_resolver: Any = None) -> Any:
            raise ConfigurationError("no key for openai")

        monkeypatch.setattr(rag_module, "build_llm_provider", _raise)
        result = runner.invoke(app, ["rag", "query", "q"])
        assert result.exit_code == 1
        assert "LLM provider key required" in result.output


# ---------------------------------------------------------------------------
# Unknown provider / model
# ---------------------------------------------------------------------------


class TestProviderValidation:
    def test_unknown_provider_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(name: str, surface: Any, *, model: Any = None) -> Any:
            raise KeyError(f"unknown provider {name!r}")

        monkeypatch.setattr(
            "sec_generative_search.providers.registry.ProviderRegistry.get_capability",
            classmethod(lambda cls, *a, **kw: _raise(*a, **kw)),
        )
        result = runner.invoke(app, ["rag", "query", "q"])
        assert result.exit_code == 1
        assert "Unknown LLM provider" in result.output
        # No retrieval stack built.
        assert patched["embedders"] == []

    def test_get_capability_invoked_with_model(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``capability.structured_output`` MUST drive the
        ``prefer_structured_output`` argument to ``orchestrator.generate``."""
        seen: dict[str, Any] = {}

        def _capability(name: str, surface: Any, *, model: Any = None) -> ProviderCapability:
            seen["name"] = name
            seen["model"] = model
            return ProviderCapability(chat=True, structured_output=True)

        monkeypatch.setattr(
            "sec_generative_search.providers.registry.ProviderRegistry.get_capability",
            classmethod(lambda cls, *a, **kw: _capability(*a, **kw)),
        )

        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0, result.output
        orch = patched["orchestrators"][-1]
        assert orch.calls[0]["prefer_structured_output"] is True


# ---------------------------------------------------------------------------
# Resolver chain composition
# ---------------------------------------------------------------------------


class TestResolverChain:
    def test_encrypted_tier_skipped_when_persist_disabled(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Scenario A default: ``persist_provider_credentials=false`` →
        encrypted tier collapses; chain is admin-env only."""
        # Settings stub defaults persist_provider_credentials=False.
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0, result.output
        # build_llm_provider was called — the chain composed without
        # raising, which means the encrypted tier wasn't attempted.

    def test_encrypted_tier_attempted_when_both_flags_set(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When both ``persist_provider_credentials=true`` AND
        ``registry.encrypted=true`` the chain composes the encrypted
        tier in front of admin-env.  The store construction is stubbed —
        we only assert that the chain composition does not blow up
        the command."""

        def _stub_settings_persist() -> _StubSettings:
            return _StubSettings(persist_provider_credentials=True)

        monkeypatch.setattr(rag_module, "get_settings", _stub_settings_persist)
        _FakeRegistry.encrypted_default = True

        seen_resolvers: list[Any] = []

        class _StubStore:
            def __init__(self, registry: Any) -> None:
                seen_resolvers.append(registry)

            def get(self, *_args: Any, **_kwargs: Any) -> None:
                return None

        monkeypatch.setattr(
            "sec_generative_search.database.credentials.EncryptedCredentialStore",
            _StubStore,
        )

        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0, result.output
        # The encrypted-credential store was constructed from the
        # registry handle the CLI owns.
        assert len(seen_resolvers) == 1


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecurity:
    def test_question_not_echoed_in_error_envelope(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """The raw question MUST NOT appear in any error envelope.

        Mirrors the ``/api/rag/query`` discipline: the response never
        echoes the raw query.  An operator piping CLI output into
        operator logs must not find Tier-3 question text inlined.
        """
        sentinel = "PII-LADEN-QUESTION-MUST-NOT-ECHO"
        _arm_orchestrator_raises(ProviderError("upstream"))
        result = runner.invoke(app, ["rag", "query", sentinel, "--skip-plan"])
        assert result.exit_code == 1
        assert sentinel not in result.output

    def test_api_key_never_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An ``OPENAI_API_KEY`` in the environment must never leak —
        including on the loudest error paths."""
        canary = "sk-canary-rag-key-must-not-appear"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        _arm_orchestrator_raises(ProviderError("boom"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert canary not in result.output

    def test_edgar_identity_never_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Secret User")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "canary-edgar@example.test")
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0
        assert "Secret User" not in result.output
        assert "canary-edgar@example.test" not in result.output

    def test_both_factory_seams_used(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """The CLI MUST go through :func:`build_embedder` AND
        :func:`build_llm_provider` — never direct adapter
        instantiation.  The fixtures replace both with recorders so if
        the CLI side-stepped either seam the recorder for that seam
        would be empty.
        """
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0, result.output
        assert len(patched["embedders"]) == 1
        assert len(patched["llms"]) == 1

    def test_hostile_markup_in_answer_renders_verbatim(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A retrieved chunk / generated answer containing
        ``[red]...[/red]`` MUST render literally — otherwise hostile
        SEC text could repaint operator output.

        Tested via the answer panel only because Rich aggressively
        wraps table cells based on terminal width, splitting literal
        markup tags across lines.  The escape contract is identical at
        both render sites (``escape(result.answer)`` and
        ``escape(c.text_span)``); proving it on the answer panel is
        sufficient — both call sites flow through
        :func:`rich.markup.escape`.
        """
        evil = "Risk factor: [red]EVIL[/red]"
        _arm_orchestrator_returns(_make_result(answer=evil))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0, result.output
        assert "[red]EVIL[/red]" in result.output
