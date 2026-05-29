"""Tests for :mod:`sec_generative_search.core.metrics`.

The metrics facade is a thin, security-sensitive wrapper over the
optional ``prometheus-client`` extra. Coverage focuses on:

    - the **no-op fallback** when the extra is absent (every recorder is
      inert; :meth:`Metrics.render_latest` returns ``None``);
    - counter / histogram values landing on the right series;
    - the **content-free, cardinality-bounded** label contract —
      especially the per-provider ``model`` cap that defends the metric
      backend against an unbounded slug spray (OpenRouter accepts any
      slug);
    - label hygiene (empty values collapse to ``unknown``);
    - the process-global singleton + :func:`reset_metrics` test seam.

When ``prometheus-client`` is not installed the "available path" tests
skip via :func:`pytest.importorskip`; the fallback test runs regardless.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

from sec_generative_search.core.metrics import (
    _MAX_MODELS_PER_PROVIDER,
    Metrics,
    get_metrics,
    metrics_available,
    reset_metrics,
)


@pytest.fixture(autouse=True)
def _reset_metrics_singleton() -> Iterator[None]:
    """Drop the process-global metrics instance around every test.

    The facade is a lazy singleton shared across the whole package;
    without a reset, counters recorded by one test would leak into the
    next (and into the unrelated route / orchestrator suites).
    """
    reset_metrics()
    yield
    reset_metrics()


def _sample(metrics: Metrics, name: str, **labels: str) -> float | None:
    """Read one sample value straight off the owned registry."""
    return metrics._registry.get_sample_value(name, labels)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# No-op fallback — the extra is not installed
# ---------------------------------------------------------------------------


class TestNoOpFallback:
    def test_inert_when_prometheus_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A ``None`` entry in ``sys.modules`` makes ``import
        # prometheus_client`` raise ImportError — exactly the shape a
        # deployment without the ``[metrics]`` extra sees.
        monkeypatch.setitem(sys.modules, "prometheus_client", None)

        metrics = Metrics()
        assert metrics.available is False

        # Every recorder is a no-op and MUST NOT raise.
        metrics.observe_ingestion(1.0)
        metrics.observe_retrieval(0.05)
        metrics.observe_generation("openai", 1.2)
        metrics.record_tokens("openai", "gpt", input_tokens=10, output_tokens=5, pricing_tier="low")
        metrics.record_provider_failure("openai", "ProviderAuthError")

        # The endpoint distinguishes "no extra" (None → 503) from "no
        # samples yet" (empty 200 body).
        assert metrics.render_latest() is None


# ---------------------------------------------------------------------------
# Available path — recording + exposition
# ---------------------------------------------------------------------------


class TestRecording:
    def test_render_returns_content_type_and_bytes(self) -> None:
        pytest.importorskip("prometheus_client")
        metrics = get_metrics()
        assert metrics.available is True

        metrics.observe_ingestion(3.0)
        rendered = metrics.render_latest()
        assert rendered is not None
        content_type, payload = rendered
        assert "text/plain" in content_type
        assert isinstance(payload, bytes)
        assert b"sec_ingestion_duration_seconds" in payload

    def test_latency_histograms_observe(self) -> None:
        pytest.importorskip("prometheus_client")
        metrics = get_metrics()

        metrics.observe_ingestion(3.0)
        metrics.observe_retrieval(0.05)
        metrics.observe_generation("anthropic", 1.5)

        assert _sample(metrics, "sec_ingestion_duration_seconds_count") == 1.0
        assert _sample(metrics, "sec_ingestion_duration_seconds_sum") == 3.0
        assert _sample(metrics, "sec_retrieval_duration_seconds_count") == 1.0
        assert (
            _sample(metrics, "sec_generation_duration_seconds_count", provider="anthropic") == 1.0
        )

    def test_negative_durations_clamped_to_zero(self) -> None:
        pytest.importorskip("prometheus_client")
        metrics = get_metrics()
        # An injected / backwards-running clock must never push a
        # negative observation into the histogram sum.
        metrics.observe_retrieval(-5.0)
        assert _sample(metrics, "sec_retrieval_duration_seconds_sum") == 0.0

    def test_token_counter_splits_input_and_output(self) -> None:
        pytest.importorskip("prometheus_client")
        metrics = get_metrics()

        metrics.record_tokens(
            "openai", "gpt-5.4-mini", input_tokens=100, output_tokens=40, pricing_tier="low"
        )

        in_val = _sample(
            metrics,
            "sec_llm_tokens_total",
            provider="openai",
            model="gpt-5.4-mini",
            kind="input",
            pricing_tier="low",
        )
        out_val = _sample(
            metrics,
            "sec_llm_tokens_total",
            provider="openai",
            model="gpt-5.4-mini",
            kind="output",
            pricing_tier="low",
        )
        assert in_val == 100.0
        assert out_val == 40.0

    def test_zero_token_counts_create_no_series(self) -> None:
        pytest.importorskip("prometheus_client")
        metrics = get_metrics()
        # Only output tokens; the input series must never materialise.
        metrics.record_tokens(
            "openai", "gpt", input_tokens=0, output_tokens=7, pricing_tier="unknown"
        )
        assert (
            _sample(
                metrics,
                "sec_llm_tokens_total",
                provider="openai",
                model="gpt",
                kind="input",
                pricing_tier="unknown",
            )
            is None
        )
        assert (
            _sample(
                metrics,
                "sec_llm_tokens_total",
                provider="openai",
                model="gpt",
                kind="output",
                pricing_tier="unknown",
            )
            == 7.0
        )

    def test_provider_failure_counter(self) -> None:
        pytest.importorskip("prometheus_client")
        metrics = get_metrics()
        metrics.record_provider_failure("anthropic", "ProviderAuthError")
        metrics.record_provider_failure("anthropic", "ProviderAuthError")
        assert (
            _sample(
                metrics,
                "sec_provider_failures_total",
                provider="anthropic",
                error_type="ProviderAuthError",
            )
            == 2.0
        )


# ---------------------------------------------------------------------------
# Singleton + reset
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_metrics_is_stable_within_a_process(self) -> None:
        assert get_metrics() is get_metrics()

    def test_reset_rebuilds_a_fresh_registry(self) -> None:
        pytest.importorskip("prometheus_client")
        first = get_metrics()
        first.observe_ingestion(2.0)
        assert _sample(first, "sec_ingestion_duration_seconds_count") == 1.0

        reset_metrics()
        second = get_metrics()
        assert second is not first
        # Fresh registry → the histogram count is back to zero (the
        # unlabelled histogram materialises its _count at definition),
        # proving no cross-instance series leakage.
        assert _sample(second, "sec_ingestion_duration_seconds_count") == 0.0

    def test_metrics_available_reflects_extra(self) -> None:
        # When prometheus_client is importable, availability is True;
        # the no-op test covers the inverse.
        pytest.importorskip("prometheus_client")
        assert metrics_available() is True


# ---------------------------------------------------------------------------
# Security: cardinality bound + label hygiene
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestLabelHygiene:
    def test_empty_labels_collapse_to_unknown(self) -> None:
        pytest.importorskip("prometheus_client")
        metrics = get_metrics()

        metrics.observe_generation("", 1.0)
        metrics.record_provider_failure("", "")
        metrics.record_tokens("", "", input_tokens=3, output_tokens=0, pricing_tier="")

        assert _sample(metrics, "sec_generation_duration_seconds_count", provider="unknown") == 1.0
        assert (
            _sample(
                metrics,
                "sec_provider_failures_total",
                provider="unknown",
                error_type="unknown",
            )
            == 1.0
        )
        assert (
            _sample(
                metrics,
                "sec_llm_tokens_total",
                provider="unknown",
                model="unknown",
                kind="input",
                pricing_tier="unknown",
            )
            == 3.0
        )

    def test_model_cardinality_is_bounded_per_provider(self) -> None:
        """An unbounded slug spray (OpenRouter accepts any slug) MUST NOT
        mint unbounded time series — overflow collapses to ``other``.
        """
        pytest.importorskip("prometheus_client")
        metrics = get_metrics()

        # Spray well past the per-provider cap.
        total = _MAX_MODELS_PER_PROVIDER + 50
        for i in range(total):
            metrics.record_tokens(
                "openrouter",
                f"vendor/model-{i}",
                input_tokens=1,
                output_tokens=0,
                pricing_tier="unknown",
            )

        # Count distinct ``model`` label values that actually landed on
        # the input-kind series for this provider.
        models_seen = set()
        for metric in metrics._registry.collect():  # type: ignore[union-attr]
            if metric.name != "sec_llm_tokens":
                continue
            for sample in metric.samples:
                if (
                    sample.name == "sec_llm_tokens_total"
                    and sample.labels.get("provider") == "openrouter"
                    and sample.labels.get("kind") == "input"
                ):
                    models_seen.add(sample.labels["model"])

        # At most the cap distinct slugs + the single ``other`` overflow
        # sentinel — never one series per sprayed slug.
        assert len(models_seen) <= _MAX_MODELS_PER_PROVIDER + 1
        assert "other" in models_seen

    def test_cap_is_per_provider_not_global(self) -> None:
        pytest.importorskip("prometheus_client")
        metrics = get_metrics()
        # A second provider gets its own budget — a busy OpenRouter must
        # not starve OpenAI's catalogue out of the metrics.
        metrics.record_tokens(
            "openai", "gpt-5.4-mini", input_tokens=1, output_tokens=0, pricing_tier="low"
        )
        assert (
            _sample(
                metrics,
                "sec_llm_tokens_total",
                provider="openai",
                model="gpt-5.4-mini",
                kind="input",
                pricing_tier="low",
            )
            == 1.0
        )
