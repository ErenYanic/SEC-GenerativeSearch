"""Metrics instrumentation tests for :mod:`sec_generative_search.rag.orchestrator`.

Confirms the orchestrator feeds the process-global metrics facade:

    - generation latency + token counters on the non-streaming and
      streaming success paths;
    - the provider-failure counter (keyed by provider + exception class
      name) when the LLM raises a :class:`ProviderError`, and that the
      error still propagates;
    - the refusal short-circuit records **no** generation series (no
      provider call was made).

Retrieval latency is exercised in :mod:`tests.search` against the real
:class:`RetrievalService`; the fan-out double here does not touch it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from sec_generative_search.core.exceptions import ProviderAuthError
from sec_generative_search.core.metrics import Metrics, get_metrics, reset_metrics
from sec_generative_search.core.types import TokenUsage
from sec_generative_search.providers.base import GenerationRequest, GenerationResponse
from sec_generative_search.rag.orchestrator import RAGOrchestrator
from sec_generative_search.rag.query_understanding import QueryPlan
from tests.rag.conftest import FakeLLMProvider, FakeRetrievalService


@pytest.fixture(autouse=True)
def _reset_metrics_singleton() -> Iterator[None]:
    reset_metrics()
    yield
    reset_metrics()


def _build_orchestrator(
    *, retrieval: FakeRetrievalService, llm: FakeLLMProvider
) -> RAGOrchestrator:
    def counter(text: str) -> int:
        return max(1, len(text) // 4)

    return RAGOrchestrator(retrieval=retrieval, llm=llm, token_counter=counter)


def _sample(metrics: Metrics, name: str, **labels: str) -> float | None:
    return metrics._registry.get_sample_value(name, labels)  # type: ignore[union-attr]


class _RaisingLLM(FakeLLMProvider):
    """LLM double whose generate / stream paths raise a provider error."""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise ProviderAuthError("bad key", provider=self.provider_name)

    def generate_stream(self, request: GenerationRequest) -> Iterator[GenerationResponse]:
        raise ProviderAuthError("bad key", provider=self.provider_name)
        yield  # pragma: no cover - unreachable, makes this a generator


class TestGenerationMetrics:
    def test_generate_records_latency_and_tokens(self, fake_retrieval, sample_chunks) -> None:
        pytest.importorskip("prometheus_client")
        llm = FakeLLMProvider(
            reply="Answer [1].", usage=TokenUsage(input_tokens=30, output_tokens=12)
        )
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=llm)

        orch.generate(QueryPlan(raw_query="What about revenue?"))

        metrics = get_metrics()
        assert _sample(metrics, "sec_generation_duration_seconds_count", provider="fake-llm") == 1.0
        # Token counters land under (provider, model, kind). Pricing tier
        # resolves to "unknown" — the fake provider is not in any
        # catalogue.
        assert (
            _sample(
                metrics,
                "sec_llm_tokens_total",
                provider="fake-llm",
                model="fake-model",
                kind="input",
                pricing_tier="unknown",
            )
            == 30.0
        )
        assert (
            _sample(
                metrics,
                "sec_llm_tokens_total",
                provider="fake-llm",
                model="fake-model",
                kind="output",
                pricing_tier="unknown",
            )
            == 12.0
        )

    def test_stream_records_latency_and_tokens(self, fake_retrieval, sample_chunks) -> None:
        pytest.importorskip("prometheus_client")
        llm = FakeLLMProvider(
            reply="Answer [1].", usage=TokenUsage(input_tokens=20, output_tokens=9)
        )
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=llm)

        list(orch.generate_stream(QueryPlan(raw_query="Q")))

        metrics = get_metrics()
        assert _sample(metrics, "sec_generation_duration_seconds_count", provider="fake-llm") == 1.0
        assert (
            _sample(
                metrics,
                "sec_llm_tokens_total",
                provider="fake-llm",
                model="fake-model",
                kind="output",
                pricing_tier="unknown",
            )
            == 9.0
        )

    def test_refusal_records_no_generation_series(self) -> None:
        pytest.importorskip("prometheus_client")
        # Empty retrieval → refusal short-circuit; the LLM is never
        # called, so no latency / token series may appear.
        orch = _build_orchestrator(
            retrieval=FakeRetrievalService(results=[]), llm=FakeLLMProvider()
        )
        result = orch.generate(QueryPlan(raw_query="Q"))
        assert result.answer  # refusal text

        metrics = get_metrics()
        assert (
            _sample(metrics, "sec_generation_duration_seconds_count", provider="fake-llm") is None
        )


@pytest.mark.security
class TestProviderFailureMetrics:
    def test_generate_failure_counts_and_propagates(self, fake_retrieval) -> None:
        pytest.importorskip("prometheus_client")
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=_RaisingLLM())

        with pytest.raises(ProviderAuthError):
            orch.generate(QueryPlan(raw_query="Q"))

        metrics = get_metrics()
        # Counter keyed by provider + exception CLASS NAME — never the
        # message (which could echo upstream text).
        assert (
            _sample(
                metrics,
                "sec_provider_failures_total",
                provider="fake-llm",
                error_type="ProviderAuthError",
            )
            == 1.0
        )

    def test_stream_failure_counts_and_propagates(self, fake_retrieval) -> None:
        pytest.importorskip("prometheus_client")
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=_RaisingLLM())

        with pytest.raises(ProviderAuthError):
            list(orch.generate_stream(QueryPlan(raw_query="Q")))

        metrics = get_metrics()
        assert (
            _sample(
                metrics,
                "sec_provider_failures_total",
                provider="fake-llm",
                error_type="ProviderAuthError",
            )
            == 1.0
        )
