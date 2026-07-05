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
from unittest.mock import MagicMock

import pytest

from sec_generative_search.core.exceptions import ProviderAuthError, ProviderConnectionError
from sec_generative_search.core.metrics import Metrics, get_metrics, reset_metrics
from sec_generative_search.core.provider_health import get_provider_health, reset_provider_health
from sec_generative_search.core.types import TokenUsage
from sec_generative_search.providers import openai_compat
from sec_generative_search.providers.base import GenerationRequest, GenerationResponse
from sec_generative_search.providers.openai_compat import OpenAICompatibleLLMProvider
from sec_generative_search.rag.orchestrator import RAGOrchestrator
from sec_generative_search.rag.query_understanding import QueryPlan
from tests.rag.conftest import FakeLLMProvider, FakeRetrievalService


@pytest.fixture(autouse=True)
def _reset_metrics_singleton() -> Iterator[None]:
    reset_metrics()
    reset_provider_health()
    yield
    reset_metrics()
    reset_provider_health()


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


class _FakeOpenAIConnectionError(openai_compat.APIConnectionError):
    """SDK connection error that skips the heavy ``httpx.Response`` ctor."""

    def __init__(self, message: str = "connection dropped mid-stream") -> None:
        Exception.__init__(self, message)


class _MidStreamDropAdapter(OpenAICompatibleLLMProvider):
    """A *real* OpenAI-compatible adapter used to exercise the F2 seam.

    Its ``generate_stream`` is entirely inherited — the failure is
    injected purely at the SDK-client boundary, so the test drives the
    genuine mid-stream normalisation path, not a hand-rolled double.
    """

    provider_name = "midstream-vendor"
    default_model = "demo-chat"


def _one_delta_then_drop() -> Iterator[MagicMock]:
    """SDK stream: one usable delta chunk, then a mid-stream connection drop."""
    delta = MagicMock()
    delta.content = "partial"
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.model = "demo-chat"
    chunk.usage = None
    yield chunk
    raise _FakeOpenAIConnectionError()


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

    def test_real_adapter_midstream_drop_feeds_failure_telemetry(
        self, fake_retrieval, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end F2 lock: a *real* adapter whose SDK stream drops
        mid-flight surfaces a normalised ``ProviderConnectionError`` — so
        the orchestrator's ``except ProviderError`` fires and feeds both
        the provider-failure metric and the passive-health circuit,
        instead of the raw SDK type escaping as an unclassified error.
        """
        pytest.importorskip("prometheus_client")

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _one_delta_then_drop()
        monkeypatch.setattr(openai_compat, "OpenAI", lambda **_kwargs: fake_client)
        adapter = _MidStreamDropAdapter("sk-demo-key-ABCDEFGH")  # pragma: allowlist secret
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=adapter)

        # The mid-stream SDK drop propagates as a normalised ProviderError,
        # not the raw ``openai.APIConnectionError``.
        with pytest.raises(ProviderConnectionError):
            list(orch.generate_stream(QueryPlan(raw_query="Q")))

        # Telemetry blind spot closed: the failure counter is keyed by the
        # normalised exception class name.
        metrics = get_metrics()
        assert (
            _sample(
                metrics,
                "sec_provider_failures_total",
                provider="midstream-vendor",
                error_type="ProviderConnectionError",
            )
            == 1.0
        )

        # Passive-health circuit fed the same content-free outcome.
        health = {s.provider: s for s in get_provider_health().snapshot()}
        assert health["midstream-vendor"].last_error_type == "ProviderConnectionError"
        assert health["midstream-vendor"].total_failures == 1
