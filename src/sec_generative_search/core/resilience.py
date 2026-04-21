"""Resilience primitives for provider calls and future retrieval work.

This module is deliberately dependency-free: it imports only from the
standard library and :mod:`core.exceptions` / :mod:`core.logging`.  The
provider adapters compose these primitives, and the retrieval service
reuses the same building blocks, so no piece of this file may reach
into any SDK.

Contents:

- :class:`RetryPolicy` — exponential-backoff retry configuration.
- :class:`CircuitBreaker` — thread-safe three-state circuit breaker
  (``CLOSED`` → ``OPEN`` → ``HALF_OPEN`` → ``CLOSED``/``OPEN``).
- :class:`ExceptionMapping` — declarative mapping from SDK-specific
  exception types to :class:`ProviderError` subclasses.
- :func:`normalise_exception` — consult a mapping and return the
  corresponding :class:`ProviderError` subclass.
- :func:`with_timeout` — run a callable with a wall-clock timeout using
  a one-shot thread pool (safety net for SDKs that ignore their own
  timeout arg).
- :func:`resilient_call` — the top-level composer: optional circuit
  breaker check, optional timeout, retry with exponential backoff, and
  exception normalisation, all driven by the policies above.

Design notes:

- No ``ProviderError`` carries the raw API key or any user-supplied
  prompt text.  The mapping only inspects exception *types* — the
  original exception message is copied into ``details`` after the caller
  has ensured it is safe to log.  Never pass a raw prompt string through
  this layer.
- ``resilient_call`` treats :class:`ProviderAuthError` and
  :class:`ProviderContentFilterError` as terminal: both indicate an
  input that will not change on retry, so retrying wastes quota and
  extends the blast radius of a bad key.  All other
  :class:`ProviderError` subclasses are retried within the policy.
- Timeout is a *safety net*, not the primary control.  Concrete
  providers pass their SDK's own ``timeout=`` argument in addition to
  this.  The Python thread we spawn for :func:`with_timeout` cannot be
  killed; a runaway SDK call will continue in the background until the
  process exits.  This is the standard trade-off for thread-based
  timeouts in CPython.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "ExceptionMapping",
    "RetryPolicy",
    "normalise_exception",
    "resilient_call",
    "with_timeout",
]

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff retry configuration.

    Construction is cheap; instances are frozen so they can be shared
    across threads without synchronisation.  Values align with the
    :class:`ProviderSettings` defaults so the usual instantiation is
    ``RetryPolicy.from_settings(settings.provider)``.

    Attributes:
        max_retries: Number of *retry* attempts after the initial call.
            ``max_retries=3`` means up to four total attempts.
        backoff_base: Exponential base.  Delay before attempt *n* (1-
            indexed, retries only) is
            ``min(initial_delay * backoff_base ** (n - 1), max_delay)``.
        initial_delay: Delay (seconds) before the first retry.
        max_delay: Upper bound on the computed delay — prevents the
            schedule from drifting into hour-long waits.
    """

    max_retries: int = 3
    backoff_base: float = 2.0
    initial_delay: float = 1.0
    max_delay: float = 30.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.backoff_base < 1.0:
            raise ValueError("backoff_base must be >= 1.0")
        if self.initial_delay < 0:
            raise ValueError("initial_delay must be >= 0")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the backoff delay (seconds) before retry *attempt*.

        ``attempt`` is 1-indexed, counting only retries — attempt 1 is
        the delay before the *first* retry (i.e. after the initial call
        failed).
        """
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        delay = self.initial_delay * (self.backoff_base ** (attempt - 1))
        return min(delay, self.max_delay)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    """States of the :class:`CircuitBreaker` finite-state machine."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe three-state circuit breaker.

    The breaker opens after ``threshold`` consecutive failures and
    refuses calls for ``reset_timeout`` seconds.  After the cool-down,
    the next call transitions the state to ``HALF_OPEN`` and is allowed
    through as a probe: success closes the breaker, failure re-opens it.

    The clock is injectable so unit tests can drive the FSM
    deterministically without :func:`time.sleep`.
    """

    def __init__(
        self,
        *,
        threshold: int,
        reset_timeout: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        if reset_timeout < 0:
            raise ValueError("reset_timeout must be >= 0")
        self._threshold = threshold
        self._reset_timeout = reset_timeout
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        """Current state, after applying any time-based transitions."""
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            return self._state

    def _maybe_transition_to_half_open_locked(self) -> None:
        """OPEN → HALF_OPEN once ``reset_timeout`` has elapsed.

        Must be called under ``self._lock``.
        """
        if (
            self._state is CircuitState.OPEN
            and self._opened_at is not None
            and self._clock() - self._opened_at >= self._reset_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            logger.info("Circuit breaker entering HALF_OPEN for probe")

    def before_call(self) -> None:
        """Raise :class:`ProviderError` if the breaker is open.

        Called before each attempt inside :func:`resilient_call`.  The
        raised error is *not* counted as a failure against the breaker
        — no upstream request was made.
        """
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            if self._state is CircuitState.OPEN:
                raise ProviderError(
                    "Circuit breaker is open; upstream service is failing",
                    hint=(
                        f"Retry after {self._reset_timeout:.1f}s or investigate upstream failures."
                    ),
                )

    def on_success(self) -> None:
        """Record a successful call — closes the breaker."""
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                logger.info("Circuit breaker CLOSED after successful probe")
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = None

    def on_failure(self) -> None:
        """Record a failed call — may open or re-open the breaker."""
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                logger.warning("Circuit breaker re-OPENED after failed probe")
                return
            self._failures += 1
            if self._failures >= self._threshold and self._state is CircuitState.CLOSED:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                logger.warning(
                    "Circuit breaker OPENED after %d consecutive failures",
                    self._failures,
                )


# ---------------------------------------------------------------------------
# Exception normalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExceptionMapping:
    """Declarative mapping from SDK-specific exception types to :class:`ProviderError`.

    Each concrete provider constructs one at module load
    using the SDK's public exception hierarchy.  The resilience layer
    consults the mapping to translate every failure into the small set
    of types that the RAG orchestrator and UI can reason about.

    ``timeout`` defaults to ``(TimeoutError,)`` so the thread-timeout
    raised by :func:`with_timeout` is always normalised, even when a
    caller forgets to extend the tuple.  Callers may override this
    default to disable or extend the mapping.

    Empty tuples are fine — ``isinstance(exc, ())`` returns ``False``
    and the default branch of :func:`normalise_exception` returns a
    generic :class:`ProviderError`.
    """

    auth: tuple[type[BaseException], ...] = ()
    rate_limit: tuple[type[BaseException], ...] = ()
    timeout: tuple[type[BaseException], ...] = (TimeoutError,)
    content_filter: tuple[type[BaseException], ...] = ()


def normalise_exception(
    exc: BaseException,
    *,
    provider: str,
    mapping: ExceptionMapping,
) -> ProviderError:
    """Translate an arbitrary exception into a :class:`ProviderError` subclass.

    If *exc* is already a :class:`ProviderError`, it is returned
    unchanged — concrete providers may raise these directly to bypass
    the mapping.  ``details`` on the returned error holds ``str(exc)``;
    callers are responsible for ensuring the exception message is safe
    to log (no prompt text, no key material).
    """
    if isinstance(exc, ProviderError):
        return exc
    detail = str(exc) or type(exc).__name__
    if mapping.auth and isinstance(exc, mapping.auth):
        return ProviderAuthError(
            f"Authentication failed against {provider}",
            provider=provider,
            hint="Verify the API key is correct and not expired.",
            details=detail,
        )
    if mapping.rate_limit and isinstance(exc, mapping.rate_limit):
        return ProviderRateLimitError(
            f"Rate limit exceeded for {provider}",
            provider=provider,
            hint="Retry after the Retry-After interval or lower request volume.",
            details=detail,
        )
    if mapping.timeout and isinstance(exc, mapping.timeout):
        return ProviderTimeoutError(
            f"{provider} call timed out",
            provider=provider,
            hint="Increase PROVIDER_TIMEOUT or retry with fewer tokens.",
            details=detail,
        )
    if mapping.content_filter and isinstance(exc, mapping.content_filter):
        return ProviderContentFilterError(
            f"{provider} safety filter blocked the request",
            provider=provider,
            hint="Reformulate the prompt or route to a different provider.",
            details=detail,
        )
    return ProviderError(
        f"{provider} call failed: {type(exc).__name__}",
        provider=provider,
        details=detail,
    )


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------


def with_timeout[T](fn: Callable[[], T], *, seconds: float) -> T:
    """Run *fn* with a wall-clock timeout.

    A ``seconds`` of 0 (or less) disables the guard and runs in the
    calling thread.  When enabled, *fn* is submitted to a single-worker
    thread pool and the current thread waits up to ``seconds`` for the
    result.  A :class:`TimeoutError` is raised if the wait expires; the
    worker thread keeps running in the background until *fn* returns
    (Python threads are not killable).

    Treat this as a defensive safety net, not a replacement for the
    SDK's own timeout argument.
    """
    if seconds <= 0:
        return fn()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=seconds)
        except concurrent.futures.TimeoutError as e:
            raise TimeoutError(f"Call exceeded {seconds:.1f}s timeout") from e


# ---------------------------------------------------------------------------
# Composite wrapper
# ---------------------------------------------------------------------------


# Terminal error types — never retried.  Auth and content-filter failures
# are deterministic given the input; retrying wastes quota and extends
# the blast radius of a bad key or a blocked prompt.
_TERMINAL_PROVIDER_ERRORS: tuple[type[ProviderError], ...] = (
    ProviderAuthError,
    ProviderContentFilterError,
)


@dataclass
class ResilientCallPolicy:
    """Bundle of resilience policies applied by :func:`resilient_call`.

    Pass-through container so that provider subclasses can configure
    retry/timeout/circuit-breaker behaviour in one place and forward it
    to :func:`resilient_call` without a long argument list.
    """

    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    exception_mapping: ExceptionMapping = field(default_factory=ExceptionMapping)
    timeout: float = 0.0
    circuit_breaker: CircuitBreaker | None = None


def resilient_call[T](
    fn: Callable[[], T],
    *,
    provider: str,
    policy: ResilientCallPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Execute *fn* under timeout, retry, circuit-breaker, and mapping policies.

    Flow per attempt:

    1. If a circuit breaker is configured and OPEN, raise immediately
       (no retries, no attempt increment).
    2. Run *fn* directly, or under :func:`with_timeout` when
       ``policy.timeout > 0``.
    3. On success: close the breaker and return the result.
    4. On failure: normalise the exception, mark the breaker.
       *Terminal* errors (auth, content-filter) raise immediately.
       *Retryable* errors either exhaust the retry budget and raise, or
       sleep and loop.

    The ``sleep`` parameter is injectable so tests can drive the FSM
    without real time passing.
    """
    last_exc: ProviderError | None = None
    max_attempts = policy.retry_policy.max_retries + 1
    for attempt in range(1, max_attempts + 1):
        if policy.circuit_breaker is not None:
            # Breaker raises a ProviderError when OPEN — propagate
            # without recording a new failure.  The caller sees the
            # breaker state in the error, which is enough context.
            policy.circuit_breaker.before_call()

        try:
            result = with_timeout(fn, seconds=policy.timeout) if policy.timeout > 0 else fn()
        except Exception as raw:
            normalised = normalise_exception(
                raw,
                provider=provider,
                mapping=policy.exception_mapping,
            )
            if policy.circuit_breaker is not None:
                policy.circuit_breaker.on_failure()
            if isinstance(normalised, _TERMINAL_PROVIDER_ERRORS):
                raise normalised from raw
            last_exc = normalised
            if attempt >= max_attempts:
                raise last_exc from raw
            sleep(policy.retry_policy.delay_for_attempt(attempt))
            continue

        if policy.circuit_breaker is not None:
            policy.circuit_breaker.on_success()
        return result

    # The loop always returns on success or raises on exhaustion; this
    # line is unreachable.  Present to reassure type checkers.
    assert last_exc is not None  # pragma: no cover
    raise last_exc  # pragma: no cover
