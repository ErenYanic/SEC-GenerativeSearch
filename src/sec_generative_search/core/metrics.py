"""In-process metrics registry.

A tiny facade over `prometheus-client` that records the three metric
families the operator dashboards consume:

    - ingestion / retrieval / generation latency **histograms**;
    - LLM token-usage **counters** keyed ``(provider, model, kind)``
      with an additional coarse ``pricing_tier`` label;
    - provider-failure **counters** keyed ``(provider, error_type)``.

Why a facade rather than touching ``prometheus_client`` directly at the
call sites:

    - **Optional dependency.** ``prometheus-client`` ships behind the
      ``[metrics]`` extra. When it is not installed every record helper
      is a cheap no-op and :meth:`Metrics.render_latest` reports
      unavailability, so the instrumentation calls peppered through the
      retrieval / generation / ingestion paths never raise on a
      Scenario-A install that does not scrape.
    - **One owned registry.** The facade holds its *own*
      :class:`~prometheus_client.CollectorRegistry` rather than the
      process-global default. Tests reset it cleanly via
      :func:`reset_metrics`, and the exposition endpoint renders exactly
      this registry — never whatever a stray library registered globally.

Security contract (load-bearing):

    - **Every label value is content-free and low-cardinality.** The
      only labels in use are ``provider`` (curated registry name),
      ``model`` (a model slug — never user text), ``kind``
      (``input`` / ``output``), ``pricing_tier`` (a coarse bucket), and
      ``error_type`` (an exception class name). A ticker, query,
      accession number, ``user_id``, or ``session_id`` MUST NEVER reach
      a label — they would both leak Tier-3 activity into an aggregator
      and blow up time-series cardinality.
    - **The ``model`` label is the one unbounded axis.** OpenRouter (and
      any future arbitrary-slug provider) accepts free-form model names;
      a caller — or an attacker probing the RAG surface — could mint an
      unbounded set of slugs and explode the metric backend. The facade
      defends itself: it admits at most :data:`_MAX_MODELS_PER_PROVIDER`
      distinct model slugs per provider and collapses the overflow to
      the ``"other"`` sentinel. This cap is independent of any caller
      and is exercised by a dedicated security test.
    - **No dollar math.** Cost is expressed only as token counts plus
      the coarse ``pricing_tier`` label. There is deliberately no price
      gauge and no per-token rate.

This module imports only the standard library at import time;
``prometheus_client`` is imported lazily inside the constructor so the
package import graph does not hard-depend on the optional extra.
"""

from __future__ import annotations

import threading

from sec_generative_search.core.logging import get_logger

__all__ = [
    "Metrics",
    "get_metrics",
    "metrics_available",
    "reset_metrics",
]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cardinality + label hygiene constants
# ---------------------------------------------------------------------------

# Maximum distinct ``model`` slugs admitted per provider before the
# overflow collapses to ``_OVERFLOW_LABEL``. Bounds the one label axis a
# caller can drive to unbounded cardinality (OpenRouter accepts any
# slug). 64 comfortably covers every curated catalogue while capping a
# hostile slug-spray at a single extra series per provider.
_MAX_MODELS_PER_PROVIDER = 64

# Sentinel substituted for an empty label value and for the model label
# once a provider exceeds the per-provider cap.
_UNKNOWN_LABEL = "unknown"
_OVERFLOW_LABEL = "other"

# Histogram bucket boundaries (seconds). Tuned per stage: retrieval is
# sub-second, generation spans the streaming long tail, ingestion is a
# multi-step fetch→parse→chunk→embed→store pipeline per filing.
_RETRIEVAL_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
_GENERATION_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
_INGESTION_BUCKETS = (1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)


class Metrics:
    """Owned metrics registry + content-free recording facade.

    Construct once per process via :func:`get_metrics`. When
    ``prometheus-client`` is not installed the instance is *inert*:
    :attr:`available` is ``False``, every ``record_*`` / ``observe_*``
    call returns immediately, and :meth:`render_latest` returns ``None``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Per-provider set of admitted model slugs, used to enforce the
        # cardinality cap on the otherwise-unbounded ``model`` axis.
        self._models_seen: dict[str, set[str]] = {}

        try:
            from prometheus_client import (
                CONTENT_TYPE_LATEST,
                CollectorRegistry,
                Counter,
                Histogram,
            )
        except ImportError:
            self.available = False
            self._registry = None
            self._content_type = ""
            return

        self.available = True
        self._content_type = CONTENT_TYPE_LATEST
        # A private registry — never the global default — so the
        # exposition endpoint renders exactly the series this facade
        # owns and tests can drop the whole thing in one call.
        self._registry = CollectorRegistry()

        self._ingestion_seconds = Histogram(
            "sec_ingestion_duration_seconds",
            "Per-filing ingestion pipeline latency (fetch→parse→chunk→embed→store).",
            buckets=_INGESTION_BUCKETS,
            registry=self._registry,
        )
        self._retrieval_seconds = Histogram(
            "sec_retrieval_duration_seconds",
            "Single-query retrieval latency (embed→vector search→rank→pack).",
            buckets=_RETRIEVAL_BUCKETS,
            registry=self._registry,
        )
        self._generation_seconds = Histogram(
            "sec_generation_duration_seconds",
            "LLM answer-generation latency, by provider.",
            labelnames=("provider",),
            buckets=_GENERATION_BUCKETS,
            registry=self._registry,
        )
        self._llm_tokens = Counter(
            "sec_llm_tokens_total",
            "LLM token usage by provider, model, kind (input/output), and pricing tier.",
            labelnames=("provider", "model", "kind", "pricing_tier"),
            registry=self._registry,
        )
        self._provider_failures = Counter(
            "sec_provider_failures_total",
            "Provider call failures by provider and error type (exception class name).",
            labelnames=("provider", "error_type"),
            registry=self._registry,
        )

    # ------------------------------------------------------------------
    # Recording — every method is a no-op when the extra is absent
    # ------------------------------------------------------------------

    def observe_ingestion(self, seconds: float) -> None:
        """Record one per-filing ingestion duration."""
        if not self.available:
            return
        self._ingestion_seconds.observe(max(0.0, seconds))

    def observe_retrieval(self, seconds: float) -> None:
        """Record one retrieval-call duration."""
        if not self.available:
            return
        self._retrieval_seconds.observe(max(0.0, seconds))

    def observe_generation(self, provider: str, seconds: float) -> None:
        """Record one generation-call duration, labelled by provider."""
        if not self.available:
            return
        self._generation_seconds.labels(provider=_clean(provider)).observe(max(0.0, seconds))

    def record_tokens(
        self,
        provider: str,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int,
        pricing_tier: str,
    ) -> None:
        """Add input/output token counts for a single generation call.

        The ``model`` label is cardinality-bounded (see
        :data:`_MAX_MODELS_PER_PROVIDER`); ``pricing_tier`` is a coarse
        bucket string (e.g. ``"low"`` / ``"unknown"``) — never a price.
        """
        if not self.available:
            return
        prov = _clean(provider)
        mdl = self._bounded_model(prov, model)
        tier = _clean(pricing_tier)
        if input_tokens > 0:
            self._llm_tokens.labels(provider=prov, model=mdl, kind="input", pricing_tier=tier).inc(
                input_tokens
            )
        if output_tokens > 0:
            self._llm_tokens.labels(provider=prov, model=mdl, kind="output", pricing_tier=tier).inc(
                output_tokens
            )

    def record_provider_failure(self, provider: str, error_type: str) -> None:
        """Increment the provider-failure counter for ``(provider, error_type)``.

        ``error_type`` MUST be a content-free classifier — the exception
        class name (e.g. ``"ProviderAuthError"``), never a message that
        could carry a ticker / query / accession number.
        """
        if not self.available:
            return
        self._provider_failures.labels(
            provider=_clean(provider), error_type=_clean(error_type)
        ).inc()

    # ------------------------------------------------------------------
    # Exposition
    # ------------------------------------------------------------------

    def render_latest(self) -> tuple[str, bytes] | None:
        """Return ``(content_type, payload)`` for the exposition endpoint.

        ``None`` when ``prometheus-client`` is not installed — the route
        translates that into ``503 metrics_unavailable`` with an install
        hint rather than exposing an empty body.
        """
        if not self.available or self._registry is None:
            return None
        from prometheus_client import generate_latest

        return self._content_type, generate_latest(self._registry)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bounded_model(self, provider: str, model: str) -> str:
        """Map ``model`` to itself or the overflow sentinel.

        Admits up to :data:`_MAX_MODELS_PER_PROVIDER` distinct slugs per
        provider; every slug beyond the cap (and every empty slug)
        collapses so a hostile or buggy caller cannot mint unbounded
        time series via the ``model`` label.
        """
        cleaned = _clean(model)
        with self._lock:
            seen = self._models_seen.setdefault(provider, set())
            if cleaned in seen:
                return cleaned
            if len(seen) < _MAX_MODELS_PER_PROVIDER:
                seen.add(cleaned)
                return cleaned
        return _OVERFLOW_LABEL


def _clean(value: str) -> str:
    """Coerce a label value to a non-empty, single-token string.

    Empty / whitespace-only values become :data:`_UNKNOWN_LABEL`. This
    is a hygiene guard, not a content filter: callers are contractually
    responsible for passing only content-free, low-cardinality values
    (provider name, model slug, ``kind``, pricing tier, exception class
    name). See the module docstring's security contract.
    """
    cleaned = (value or "").strip()
    return cleaned or _UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Process-global accessor
# ---------------------------------------------------------------------------

_metrics: Metrics | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> Metrics:
    """Return the process-global :class:`Metrics` instance (lazy singleton).

    Built once on first access. The instrumentation seams (retrieval,
    generation, ingestion) and the exposition route all share this one
    instance so recorded samples and the scraped output stay in sync.
    """
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = Metrics()
    return _metrics


def metrics_available() -> bool:
    """Return whether the ``[metrics]`` extra (``prometheus-client``) is installed."""
    return get_metrics().available


def reset_metrics() -> None:
    """Drop the process-global instance so the next access rebuilds it.

    Test-only seam: a fresh instance owns a fresh
    :class:`~prometheus_client.CollectorRegistry`, so counters and the
    per-provider model cap start from zero without leaking series across
    tests.
    """
    global _metrics
    with _metrics_lock:
        _metrics = None
