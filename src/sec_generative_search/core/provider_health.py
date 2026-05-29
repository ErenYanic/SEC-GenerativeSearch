"""Passive per-provider health registry.

A process-global, in-memory registry that tracks the liveness of each
LLM provider driven by the orchestrator. It gives the admin health
endpoint a small current snapshot per provider without making any
upstream calls.

The registry stays observational: it records success and failure
outcomes, but it never short-circuits a live request. It keeps the
surface content-free, bounded to curated provider names, and exposes a
derived elapsed value instead of raw timestamps.

This module imports only the standard library plus the dependency-free
:mod:`core.resilience` and the settings singleton; it adds no new runtime

"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.resilience import CircuitBreaker

__all__ = [
    "ProviderHealthRegistry",
    "ProviderHealthSnapshot",
    "get_provider_health",
    "reset_provider_health",
]


@dataclass(frozen=True, slots=True)
class ProviderHealthSnapshot:
    """Immutable, content-free health view of one provider.

    Every field is either a curated identifier, a breaker state string, an
    integer counter, or a derived scalar — nothing here can carry
    Tier-3 user input. ``last_error_type`` is an exception class name; the
    raw exception message never reaches this dataclass.
    """

    provider: str
    state: str  # CircuitState value: "closed" / "open" / "half_open"
    consecutive_failures: int
    total_failures: int
    total_successes: int
    last_error_type: str | None
    last_failure_seconds_ago: float | None
    last_latency_seconds: float | None


@dataclass
class _ProviderState:
    """Mutable per-provider record held inside the registry.

    Holds the provider's own breaker plus the supplementary counters the
    snapshot surfaces. ``last_failure_at`` is a monotonic timestamp used
    only to derive ``last_failure_seconds_ago`` — it is never exposed.
    """

    breaker: CircuitBreaker
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_error_type: str | None = None
    last_failure_at: float | None = None  # monotonic; derive-only, never exposed
    last_latency_seconds: float | None = None


class ProviderHealthRegistry:
    """Owned, thread-safe store of per-provider liveness health.

    Construct once per process via :func:`get_provider_health`. Breakers
    are created lazily on the first outcome reported for a provider, all
    seeded with the same ``threshold`` / ``reset_timeout`` captured at
    construction (the global :class:`ProviderSettings` defaults).

    The two ``record_*`` methods sit on the generation hot path; they do
    trivial dict / integer work plus a breaker transition and never raise.
    """

    def __init__(
        self,
        *,
        threshold: int,
        reset_timeout: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._threshold = threshold
        self._reset_timeout = reset_timeout
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._states: dict[str, _ProviderState] = {}

    # ------------------------------------------------------------------
    # Recording — driven by the orchestrator's generation seams
    # ------------------------------------------------------------------

    def record_success(self, provider: str, latency_seconds: float) -> None:
        """Record one successful provider call.

        Closes the breaker (a success during ``half_open`` probes the
        upstream back to healthy), resets the consecutive-failure run, and
        stores the latest call latency for the snapshot.
        """
        with self._lock:
            state = self._get_or_create_locked(provider)
            state.breaker.on_success()
            state.consecutive_failures = 0
            state.total_successes += 1
            state.last_latency_seconds = max(0.0, latency_seconds)

    def record_failure(self, provider: str, error_type: str) -> None:
        """Record one failed provider call.

        ``error_type`` MUST be a content-free classifier — the exception
        class name (e.g. ``"ProviderTimeoutError"``), never a message that
        could echo upstream text or user input. Marks the breaker, which
        opens once ``threshold`` consecutive failures accumulate.
        """
        with self._lock:
            state = self._get_or_create_locked(provider)
            state.breaker.on_failure()
            state.consecutive_failures += 1
            state.total_failures += 1
            state.last_error_type = error_type or None
            state.last_failure_at = self._clock()

    # ------------------------------------------------------------------
    # Read surface
    # ------------------------------------------------------------------

    def snapshot(self) -> list[ProviderHealthSnapshot]:
        """Return a content-free health view for every tracked provider.

        Ordered by provider name for a stable wire shape. Reading
        :attr:`CircuitBreaker.state` applies any pending time-based
        ``open`` → ``half_open`` transition, so the snapshot reflects the
        breaker's *current* state, not its state at the last outcome.
        """
        now = self._clock()
        with self._lock:
            return [
                self._snapshot_one_locked(name, self._states[name], now)
                for name in sorted(self._states)
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_locked(self, provider: str) -> _ProviderState:
        """Return the provider's state record, creating it on first touch.

        Must be called under :attr:`_lock`. A fresh breaker inherits the
        registry-wide threshold / reset and the injected clock so tests can
        drive the FSM deterministically.
        """
        state = self._states.get(provider)
        if state is None:
            state = _ProviderState(
                breaker=CircuitBreaker(
                    threshold=self._threshold,
                    reset_timeout=self._reset_timeout,
                    clock=self._clock,
                )
            )
            self._states[provider] = state
        return state

    def _snapshot_one_locked(
        self, provider: str, state: _ProviderState, now: float
    ) -> ProviderHealthSnapshot:
        """Project one mutable record onto the immutable snapshot.

        Must be called under :attr:`_lock`. ``last_failure_seconds_ago`` is
        derived from the monotonic timestamp here so the raw value never
        crosses the module boundary; it is clamped to ``>= 0`` against an
        injected clock that runs backwards in tests.
        """
        last_failure_seconds_ago: float | None = None
        if state.last_failure_at is not None:
            last_failure_seconds_ago = max(0.0, now - state.last_failure_at)
        return ProviderHealthSnapshot(
            provider=provider,
            state=state.breaker.state.value,
            consecutive_failures=state.consecutive_failures,
            total_failures=state.total_failures,
            total_successes=state.total_successes,
            last_error_type=state.last_error_type,
            last_failure_seconds_ago=last_failure_seconds_ago,
            last_latency_seconds=state.last_latency_seconds,
        )


# ---------------------------------------------------------------------------
# Process-global accessor
# ---------------------------------------------------------------------------

_registry: ProviderHealthRegistry | None = None
_registry_lock = threading.Lock()


def get_provider_health() -> ProviderHealthRegistry:
    """Return the process-global :class:`ProviderHealthRegistry` (lazy singleton).

    Built once on first access, seeded from the current
    :class:`ProviderSettings` circuit-breaker knobs. The orchestrator
    recording seams and the exposition route share this one instance so
    the snapshot reflects exactly the outcomes the generation path saw.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                provider_settings = get_settings().provider
                _registry = ProviderHealthRegistry(
                    threshold=provider_settings.circuit_breaker_threshold,
                    reset_timeout=provider_settings.circuit_breaker_reset,
                )
    return _registry


def reset_provider_health() -> None:
    """Drop the process-global instance so the next access rebuilds it.

    Test-only seam: a fresh instance owns empty per-provider state and
    re-reads the (possibly monkeypatched) circuit-breaker settings, so
    breaker thresholds and counters start clean without leaking across
    tests.
    """
    global _registry
    with _registry_lock:
        _registry = None
