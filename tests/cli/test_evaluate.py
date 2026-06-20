"""Tests for ``sec-rag evaluate retrieval``.

The retrieval primitives (:class:`RetrievalService`, ``build_embedder``,
:class:`ChromaDBClient`) are stubbed at the ``cli.evaluate`` import site
— stubbing at the import site (rather than at the source modules) proves
the CLI wires the right seams and is independently testable from
``cli.search``.

Goals:

- The embedder is built via :func:`build_embedder` (sole seam).
- :class:`ChromaDBClient` opens under the stamp from the registry.
- Evaluation cases are loaded from the path supplied to ``--cases``.
- Filters (ticker / form / date / diversity) are normalised and forwarded
  to every ``retrieve`` call.
- Failure paths map to single operator-facing envelopes.
- Security: raw case queries never appear in stdout; API keys and EDGAR
  identity never surface in any output path.  Report is content-free —
  only case IDs + numeric metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

import sec_generative_search.cli.evaluate as evaluate_module
from sec_generative_search.cli.evaluate import evaluate_app
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
    SearchResult,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "retrieval_eval_cases.json"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _make_hit(
    *,
    chunk_id: str = "AAPL_10-K_2023-11-03_000",
    accession_number: str = "0000320193-23-000077",
    path: str = "Part I > Item 1 > Business > Customers",
    ticker: str = "AAPL",
    similarity: float = 0.55,
) -> RetrievalResult:
    base = SearchResult(
        content="Sample chunk.",
        path=path,
        content_type=ContentType.TEXT,
        ticker=ticker,
        form_type="10-K",
        similarity=similarity,
        filing_date="2023-11-03",
        accession_number=accession_number,
        chunk_id=chunk_id,
    )
    return RetrievalResult.from_search_result(base)


class _FakeEmbedder:
    instances: ClassVar[list[_FakeEmbedder]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _FakeEmbedder.instances.append(self)


class _FakeChroma:
    instances: ClassVar[list[_FakeChroma]] = []

    def __init__(self, stamp: EmbedderStamp, *, chroma_path: str | None = None) -> None:
        self.stamp = stamp
        _FakeChroma.instances.append(self)


class _FakeRetrievalService:
    """Records every ``retrieve`` call; returns ``returns`` or raises ``raises``."""

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
    """Throw-away Typer app wrapping the ``evaluate`` sub-Typer.

    A second no-op command keeps the test app in *group* mode so
    sub-Typer dispatch is not short-circuited by Click.
    """
    test_app = typer.Typer()
    test_app.add_typer(evaluate_app, name="evaluate")

    @test_app.command(name="_noop", hidden=True)
    def _noop() -> None:  # pragma: no cover
        ...

    return test_app


class _StubSettings:
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
    """Replace every seam that ``cli.evaluate`` reaches with a recording stub."""
    _FakeEmbedder.instances.clear()
    _FakeChroma.instances.clear()
    _FakeRetrievalService.instances.clear()

    monkeypatch.setattr(evaluate_module, "get_settings", _StubSettings)
    monkeypatch.setattr(
        evaluate_module,
        "build_embedder",
        lambda _settings: _FakeEmbedder(),
    )
    monkeypatch.setattr(evaluate_module, "ChromaDBClient", _FakeChroma)
    monkeypatch.setattr(evaluate_module, "RetrievalService", _FakeRetrievalService)

    return {
        "embedders": _FakeEmbedder.instances,
        "chromas": _FakeChroma.instances,
        "services": _FakeRetrievalService.instances,
    }


def _arm_returns(results: list[RetrievalResult]) -> None:
    """Arm the next-constructed service to return ``results``."""
    original = _FakeRetrievalService.__init__

    def _seeded(self: _FakeRetrievalService, **kwargs: Any) -> None:
        original(self, **kwargs)
        self.returns = list(results)

    _FakeRetrievalService.__init__ = _seeded  # type: ignore[method-assign]


def _arm_raises(exc: BaseException) -> None:
    """Arm the next-constructed service to raise ``exc`` on ``retrieve``."""
    original = _FakeRetrievalService.__init__

    def _seeded(self: _FakeRetrievalService, **kwargs: Any) -> None:
        original(self, **kwargs)
        self.raises = exc

    _FakeRetrievalService.__init__ = _seeded  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _reset_service_init() -> Any:
    """Restore the default constructor after each test so armed state never leaks."""
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
    def test_fixture_cases_produce_text_report(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Fixture evaluation-case file drives a successful run."""
        _arm_returns([])  # zero hits → precision=0, recall=0 but exits 0
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert result.exit_code == 0, result.output
        assert "Evaluation Report" in result.output
        assert "Precision" in result.output
        assert "Recall" in result.output
        # Per-case case IDs appear in the table.
        assert "q-revenue-concentration-001" in result.output
        assert "q-supply-chain-risk-002" in result.output
        assert "q-cash-flow-summary-003" in result.output

    def test_json_output_shape(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """JSON output carries the expected content-free fields."""
        # One case with one expected chunk_id; service returns a matching hit.
        cases_file = tmp_path / "cases.json"
        chunk_id = "AAPL_10-K_2023-11-03_000"
        cases_file.write_text(
            f'[{{"case_id": "c1", "query": "revenue", "expected_chunk_ids": ["{chunk_id}"]}}]'
        )
        _arm_returns([_make_hit(chunk_id=chunk_id)])
        result = runner.invoke(
            app,
            [
                "evaluate",
                "retrieval",
                "--cases",
                str(cases_file),
                "--top-k",
                "1",
                "--output",
                "json",
            ],
        )
        assert result.exit_code == 0, result.output
        doc = json.loads(result.output.strip())
        assert doc["top_k"] == 1
        assert doc["case_count"] == 1
        assert doc["precision_at_k"] == pytest.approx(1.0)
        assert doc["recall_at_k"] == pytest.approx(1.0)
        assert len(doc["per_case"]) == 1
        assert doc["per_case"][0]["case_id"] == "c1"
        assert doc["per_case"][0]["hits"] == 1

    def test_stamp_matches_registry_dimension(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """ChromaDBClient MUST open under the dimension the registry prescribes."""
        _arm_returns([])
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert result.exit_code == 0
        chroma = patched["chromas"][-1]
        assert chroma.stamp.provider == "openai"
        assert chroma.stamp.model == "text-embedding-3-small"
        assert chroma.stamp.dimension == 1536

    def test_embedder_built_through_factory_seam(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Embedder MUST be constructed through build_embedder, never directly."""
        _arm_returns([])
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert result.exit_code == 0
        assert len(patched["embedders"]) == 1

    def test_top_k_forwarded_to_evaluate_retrieval(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """The --top-k value must reach every retrieve call."""
        _arm_returns([])
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH), "--top-k", "10"],
        )
        assert result.exit_code == 0
        svc = patched["services"][-1]
        for call in svc.calls:
            assert call["top_k"] == 10

    def test_perfect_recall_reported_correctly(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """When every expected item is hit, recall@k = 1.0."""
        cases_file = tmp_path / "cases.json"
        cases_file.write_text(
            '[{"case_id": "c1", "query": "q", "expected_section_paths": '
            '["Part I > Item 1 > Business > Customers"]}]'
        )
        _arm_returns([_make_hit(path="Part I > Item 1 > Business > Customers")])
        result = runner.invoke(
            app,
            [
                "evaluate",
                "retrieval",
                "--cases",
                str(cases_file),
                "--top-k",
                "1",
                "--output",
                "json",
            ],
        )
        assert result.exit_code == 0
        doc = json.loads(result.output.strip())
        assert doc["recall_at_k"] == pytest.approx(1.0)
        assert doc["precision_at_k"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Filter normalisation
# ---------------------------------------------------------------------------


class TestFilterNormalisation:
    def test_ticker_uppercased_and_forwarded(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_returns([])
        result = runner.invoke(
            app,
            [
                "evaluate",
                "retrieval",
                "--cases",
                str(_FIXTURE_PATH),
                "--ticker",
                "aapl",
                "--ticker",
                "msft",
            ],
        )
        assert result.exit_code == 0
        svc = patched["services"][-1]
        for call in svc.calls:
            assert call["ticker"] == ["AAPL", "MSFT"]

    def test_form_uppercased_and_forwarded(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_returns([])
        result = runner.invoke(
            app,
            [
                "evaluate",
                "retrieval",
                "--cases",
                str(_FIXTURE_PATH),
                "--form",
                "10-k",
            ],
        )
        assert result.exit_code == 0
        for call in patched["services"][-1].calls:
            assert call["form_type"] == ["10-K"]

    def test_date_range_forwarded(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_returns([])
        result = runner.invoke(
            app,
            [
                "evaluate",
                "retrieval",
                "--cases",
                str(_FIXTURE_PATH),
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
            ],
        )
        assert result.exit_code == 0
        for call in patched["services"][-1].calls:
            assert call["start_date"] == "2023-01-01"
            assert call["end_date"] == "2023-12-31"

    def test_diversity_caps_forwarded(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_returns([])
        result = runner.invoke(
            app,
            [
                "evaluate",
                "retrieval",
                "--cases",
                str(_FIXTURE_PATH),
                "--max-per-section",
                "3",
                "--max-per-filing",
                "2",
            ],
        )
        assert result.exit_code == 0
        for call in patched["services"][-1].calls:
            assert call["max_per_section"] == 3
            assert call["max_per_filing"] == 2

    def test_diversity_caps_default_to_none(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Omitted diversity caps pass None → service uses settings default."""
        _arm_returns([])
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert result.exit_code == 0
        for call in patched["services"][-1].calls:
            assert call["max_per_section"] is None
            assert call["max_per_filing"] is None

    def test_no_filters_passes_none(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_returns([])
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert result.exit_code == 0
        for call in patched["services"][-1].calls:
            assert call["ticker"] is None
            assert call["form_type"] is None
            assert call.get("start_date") is None
            assert call.get("end_date") is None


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


class TestBoundaryValidation:
    def test_missing_cases_file_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        missing = tmp_path / "nonexistent.json"
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(missing)],
        )
        assert result.exit_code == 1
        assert "Cases file not found" in result.output
        assert patched["embedders"] == []

    def test_invalid_json_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json at all")
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(bad)],
        )
        assert result.exit_code == 1
        assert "Cases file invalid" in result.output
        assert patched["embedders"] == []

    def test_unknown_field_in_case_file_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(
            '[{"case_id": "c1", "query": "q", "expected_chunk_ids": ["a"], "bad_field": 1}]'
        )
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(bad)],
        )
        assert result.exit_code == 1
        assert "Cases file invalid" in result.output

    def test_malformed_start_date_exits_2(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            [
                "evaluate",
                "retrieval",
                "--cases",
                str(_FIXTURE_PATH),
                "--start-date",
                "not-a-date",
            ],
        )
        assert result.exit_code == 2

    def test_top_k_below_one_exits_2(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH), "--top-k", "0"],
        )
        assert result.exit_code == 2
        assert patched["embedders"] == []

    def test_negative_max_per_section_exits_2(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            [
                "evaluate",
                "retrieval",
                "--cases",
                str(_FIXTURE_PATH),
                "--max-per-section",
                "-1",
            ],
        )
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
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert result.exit_code == 1
        assert "Search failed" in result.output

    def test_provider_error_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_raises(ProviderError("embed broken"))
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert result.exit_code == 1
        assert "Embedding provider failure" in result.output

    def test_database_error_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_raises(DatabaseError("disk error"))
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
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

        monkeypatch.setattr(evaluate_module, "ChromaDBClient", _BoomChroma)
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
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

        monkeypatch.setattr(evaluate_module, "build_embedder", _raise)
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
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

        monkeypatch.setattr(evaluate_module, "build_embedder", _raise)
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert result.exit_code == 1
        assert "Embedder unavailable" in result.output
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
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert result.exit_code == 1
        assert "Embedder configuration invalid" in result.output

    def test_search_error_json_envelope(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """JSON-mode failure paths emit parseable error envelopes."""
        _arm_raises(SearchError("Bad filter"))
        result = runner.invoke(
            app,
            [
                "evaluate",
                "retrieval",
                "--cases",
                str(_FIXTURE_PATH),
                "--output",
                "json",
            ],
        )
        assert result.exit_code == 1
        doc = json.loads(result.output.strip())
        assert doc["error"] == "search_failed"
        assert "message" in doc

    def test_missing_cases_file_json_envelope(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        missing = tmp_path / "gone.json"
        result = runner.invoke(
            app,
            [
                "evaluate",
                "retrieval",
                "--cases",
                str(missing),
                "--output",
                "json",
            ],
        )
        assert result.exit_code == 1
        doc = json.loads(result.output.strip())
        assert doc["error"] == "cases_file_not_found"


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
        """An OPENAI_API_KEY in the environment must never leak in any output path."""
        canary = "sk-canary-key-must-not-appear"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        _arm_raises(SearchError("boom"))
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert canary not in result.output

    def test_edgar_identity_never_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """EDGAR identity is PII — it must not appear in any evaluate output."""
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Private Operator")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "canary-edgar-eval@example.test")
        _arm_returns([])
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert "Private Operator" not in result.output
        assert "canary-edgar-eval@example.test" not in result.output

    def test_case_queries_never_echoed_in_text_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Case queries are Tier-3 data and must never appear in stdout.

        The report is content-free: only case IDs and numeric metrics.
        Query text could be PII or competitive-intelligence material and
        must not land in operator logs via CLI output.
        """
        _arm_returns([])
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert result.exit_code == 0
        # Known query text from the fixture file — must NOT appear.
        assert "Which customers represent more than 10% of revenue?" not in result.output
        assert "What supply chain risks does the company disclose?" not in result.output
        assert "Summarise operating cash flow trends" not in result.output

    def test_case_queries_never_echoed_in_json_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """JSON output is content-free: no query text, no chunk content."""
        _arm_returns([])
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH), "--output", "json"],
        )
        assert result.exit_code == 0
        raw = result.output
        assert "Which customers represent more than 10% of revenue?" not in raw
        assert "What supply chain risks does the company disclose?" not in raw
        # Verify the JSON is valid and content-free.
        doc = json.loads(raw.strip())
        allowed_keys = {"top_k", "case_count", "precision_at_k", "recall_at_k", "per_case"}
        assert set(doc.keys()) == allowed_keys
        per_case_keys = {"case_id", "precision", "recall", "hits", "expected"}
        for row in doc["per_case"]:
            assert set(row.keys()) == per_case_keys

    def test_embedder_built_through_factory_seam(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """CLI MUST construct the embedder through build_embedder (sole seam)."""
        _arm_returns([])
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert result.exit_code == 0
        assert len(patched["embedders"]) == 1

    def test_chunk_content_never_echoed_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Retrieved chunk content must never appear in the evaluate output.

        The report is metric-only; chunk text could be proprietary filing
        content and must not surface in operator-visible output.
        """
        sentinel_content = "PROPRIETARY-CHUNK-CONTENT-MUST-NOT-APPEAR"
        hit = _make_hit()
        _ = hit.content  # access the field to verify its value is irrelevant
        _arm_returns([hit])
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(_FIXTURE_PATH)],
        )
        assert result.exit_code == 0
        assert sentinel_content not in result.output
        # The generic "Sample chunk." content also must not appear.
        assert "Sample chunk." not in result.output

    def test_markup_injection_in_case_id_rendered_verbatim(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """A case ID containing Rich markup brackets renders literally.

        An operator who authors case IDs with ``[...]`` notation must not
        trigger Rich's markup parser — the output must show the raw
        string, not a colour change.
        """
        cases_file = tmp_path / "cases.json"
        cases_file.write_text(
            '[{"case_id": "[red]EVIL[/red]", "query": "q", "expected_chunk_ids": ["a"]}]'
        )
        _arm_returns([])
        result = runner.invoke(
            app,
            ["evaluate", "retrieval", "--cases", str(cases_file)],
        )
        assert result.exit_code == 0
        assert "[red]EVIL[/red]" in result.output
