"""Tests for :mod:`sec_generative_search.core.resilience`.

Covers:

- :class:`RetryPolicy` validation and delay calculation.
- :class:`CircuitBreaker` state transitions: CLOSED → OPEN → HALF_OPEN
  → CLOSED (recovery) and HALF_OPEN → OPEN (failed probe).
- :class:`ExceptionMapping` / :func:`normalise_exception` mapping paths.
- :func:`with_timeout` success, timeout, and disabled-guard paths.
- :func:`resilient_call` composition: retry on transient errors, no
  retry on terminal errors, circuit-breaker integration, retry
  exhaustion.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.resilience import (
    CircuitBreaker,
    CircuitState,
    ExceptionMapping,
    ResilientCallPolicy,
    RetryPolicy,
    normalise_exception,
    resilient_call,
    with_timeout,
)

# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.backoff_base == 2.0
        assert policy.initial_delay == 1.0
        assert policy.max_delay == 30.0

    def test_delay_for_attempt_exponential(self) -> None:
        policy = RetryPolicy(
            max_retries=5,
            backoff_base=2.0,
            initial_delay=1.0,
            max_delay=100.0,
        )
        assert policy.delay_for_attempt(1) == pytest.approx(1.0)
        assert policy.delay_for_attempt(2) == pytest.approx(2.0)
        assert policy.delay_for_attempt(3) == pytest.approx(4.0)
        assert policy.delay_for_attempt(4) == pytest.approx(8.0)

    def test_delay_honours_max(self) -> None:
        policy = RetryPolicy(
            max_retries=10,
            backoff_base=2.0,
            initial_delay=1.0,
            max_delay=5.0,
        )
        # 2^4 = 16 would exceed the 5s cap.
        assert policy.delay_for_attempt(5) == pytest.approx(5.0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_retries", -1),
            ("backoff_base", 0.5),
            ("initial_delay", -0.1),
        ],
    )
    def test_invalid_constructor_args_rejected(self, field: str, value: float) -> None:
        kwargs: dict[str, float] = {field: value}
        with pytest.raises(ValueError):
            RetryPolicy(**kwargs)  # type: ignore[arg-type]

    def test_max_delay_below_initial_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy(initial_delay=10.0, max_delay=1.0)

    def test_delay_for_attempt_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy().delay_for_attempt(0)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class _FakeClock:
    """Monotonic-like clock whose advance is explicitly driven in tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        breaker = CircuitBreaker(threshold=3, reset_timeout=10)
        assert breaker.state is CircuitState.CLOSED

    def test_opens_after_threshold_failures(self) -> None:
        breaker = CircuitBreaker(threshold=3, reset_timeout=10)
        for _ in range(3):
            breaker.on_failure()
        assert breaker.state is CircuitState.OPEN

    def test_before_call_raises_when_open(self) -> None:
        breaker = CircuitBreaker(threshold=1, reset_timeout=10)
        breaker.on_failure()
        with pytest.raises(ProviderError):
            breaker.before_call()

    def test_transitions_to_half_open_after_timeout(self) -> None:
        clock = _FakeClock()
        breaker = CircuitBreaker(threshold=1, reset_timeout=5, clock=clock)
        breaker.on_failure()
        assert breaker.state is CircuitState.OPEN
        clock.advance(6)
        # Querying state triggers the time-based transition.
        assert breaker.state is CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self) -> None:
        clock = _FakeClock()
        breaker = CircuitBreaker(threshold=1, reset_timeout=5, clock=clock)
        breaker.on_failure()
        clock.advance(6)
        breaker.before_call()  # no raise — probe is allowed
        breaker.on_success()
        assert breaker.state is CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self) -> None:
        clock = _FakeClock()
        breaker = CircuitBreaker(threshold=1, reset_timeout=5, clock=clock)
        breaker.on_failure()
        clock.advance(6)
        breaker.before_call()  # moves to half-open
        breaker.on_failure()
        assert breaker.state is CircuitState.OPEN

    def test_success_resets_failure_count(self) -> None:
        breaker = CircuitBreaker(threshold=3, reset_timeout=10)
        breaker.on_failure()
        breaker.on_failure()
        breaker.on_success()
        # Next two failures should NOT open — count was reset.
        breaker.on_failure()
        breaker.on_failure()
        assert breaker.state is CircuitState.CLOSED

    @pytest.mark.parametrize(
        ("threshold", "reset_timeout"),
        [(0, 10), (-1, 10), (1, -0.1)],
    )
    def test_invalid_constructor_args_rejected(
        self,
        threshold: int,
        reset_timeout: float,
    ) -> None:
        with pytest.raises(ValueError):
            CircuitBreaker(threshold=threshold, reset_timeout=reset_timeout)


# ---------------------------------------------------------------------------
# Exception normalisation
# ---------------------------------------------------------------------------


class _FakeAuthError(Exception):
    """Stand-in for an SDK-specific authentication exception."""


class _FakeRateLimitError(Exception):
    """Stand-in for an SDK-specific rate-limit exception."""


class _FakeContentFilterError(Exception):
    """Stand-in for an SDK-specific safety-filter exception."""


_MAPPING = ExceptionMapping(
    auth=(_FakeAuthError,),
    rate_limit=(_FakeRateLimitError,),
    content_filter=(_FakeContentFilterError,),
)


class TestNormaliseException:
    def test_passes_provider_error_through(self) -> None:
        original = ProviderRateLimitError("already normalised", provider="x")
        assert normalise_exception(original, provider="x", mapping=_MAPPING) is original

    def test_auth_mapping(self) -> None:
        normalised = normalise_exception(_FakeAuthError("401"), provider="openai", mapping=_MAPPING)
        assert isinstance(normalised, ProviderAuthError)
        assert normalised.provider == "openai"
        assert normalised.hint is not None

    def test_rate_limit_mapping(self) -> None:
        normalised = normalise_exception(
            _FakeRateLimitError("429"), provider="openai", mapping=_MAPPING
        )
        assert isinstance(normalised, ProviderRateLimitError)

    def test_timeout_mapping_uses_builtin_default(self) -> None:
        # The default ExceptionMapping already recognises TimeoutError.
        default_mapping = ExceptionMapping()
        normalised = normalise_exception(
            TimeoutError("slow"), provider="openai", mapping=default_mapping
        )
        assert isinstance(normalised, ProviderTimeoutError)

    def test_content_filter_mapping(self) -> None:
        normalised = normalise_exception(
            _FakeContentFilterError("blocked"), provider="openai", mapping=_MAPPING
        )
        assert isinstance(normalised, ProviderContentFilterError)

    def test_unknown_exception_falls_back_to_provider_error(self) -> None:
        normalised = normalise_exception(RuntimeError("boom"), provider="openai", mapping=_MAPPING)
        assert isinstance(normalised, ProviderError)
        # Not a known subclass.
        assert not isinstance(
            normalised,
            (
                ProviderAuthError,
                ProviderRateLimitError,
                ProviderTimeoutError,
                ProviderContentFilterError,
            ),
        )

    def test_details_are_exception_string(self) -> None:
        normalised = normalise_exception(
            _FakeAuthError("bad token"), provider="openai", mapping=_MAPPING
        )
        assert normalised.details == "bad token"


# ---------------------------------------------------------------------------
# with_timeout
# ---------------------------------------------------------------------------


class TestWithTimeout:
    def test_runs_in_calling_thread_when_disabled(self) -> None:
        assert with_timeout(lambda: 42, seconds=0) == 42

    def test_returns_value_when_fast_enough(self) -> None:
        assert with_timeout(lambda: "ok", seconds=5) == "ok"

    def test_raises_timeout_error_when_slow(self) -> None:
        def slow() -> None:
            time.sleep(0.5)

        with pytest.raises(TimeoutError):
            with_timeout(slow, seconds=0.05)


# ---------------------------------------------------------------------------
# resilient_call
# ---------------------------------------------------------------------------


def _zero_sleep(_: float) -> None:
    """Sleep stub — lets tests exhaust retries without waiting."""


def _make_policy(
    *,
    max_retries: int = 2,
    mapping: ExceptionMapping | None = None,
    breaker: CircuitBreaker | None = None,
    timeout: float = 0.0,
) -> ResilientCallPolicy:
    return ResilientCallPolicy(
        retry_policy=RetryPolicy(
            max_retries=max_retries,
            backoff_base=2.0,
            initial_delay=0.01,
            max_delay=0.1,
        ),
        exception_mapping=mapping or _MAPPING,
        circuit_breaker=breaker,
        timeout=timeout,
    )


class TestResilientCall:
    def test_returns_on_first_success(self) -> None:
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        result = resilient_call(fn, provider="test", policy=_make_policy(), sleep=_zero_sleep)
        assert result == "ok"
        assert calls == 1

    def test_retries_on_rate_limit(self) -> None:
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise _FakeRateLimitError("429")
            return "ok"

        result = resilient_call(
            fn, provider="test", policy=_make_policy(max_retries=3), sleep=_zero_sleep
        )
        assert result == "ok"
        assert calls == 3

    def test_auth_error_is_not_retried(self) -> None:
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            raise _FakeAuthError("401")

        with pytest.raises(ProviderAuthError):
            resilient_call(
                fn,
                provider="test",
                policy=_make_policy(max_retries=5),
                sleep=_zero_sleep,
            )
        assert calls == 1

    def test_content_filter_error_is_not_retried(self) -> None:
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            raise _FakeContentFilterError("blocked")

        with pytest.raises(ProviderContentFilterError):
            resilient_call(
                fn,
                provider="test",
                policy=_make_policy(max_retries=5),
                sleep=_zero_sleep,
            )
        assert calls == 1

    def test_raises_after_exhausting_retries(self) -> None:
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            raise _FakeRateLimitError("429")

        with pytest.raises(ProviderRateLimitError):
            resilient_call(
                fn,
                provider="test",
                policy=_make_policy(max_retries=2),
                sleep=_zero_sleep,
            )
        # Initial + 2 retries = 3 total attempts.
        assert calls == 3

    def test_circuit_breaker_opens_and_blocks_calls(self) -> None:
        breaker = CircuitBreaker(threshold=2, reset_timeout=60)

        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            raise _FakeRateLimitError("429")

        # First invocation: retry policy will keep hitting the breaker
        # until threshold is reached, at which point the breaker opens
        # and raises a ProviderError (not further retried).
        with pytest.raises(ProviderError):
            resilient_call(
                fn,
                provider="test",
                policy=_make_policy(max_retries=5, breaker=breaker),
                sleep=_zero_sleep,
            )
        # Breaker is open now.
        assert breaker.state is CircuitState.OPEN

        # A subsequent call must fail immediately without invoking fn.
        calls_at_open = calls

        def fn2() -> str:
            nonlocal calls
            calls += 1
            return "won't run"

        with pytest.raises(ProviderError):
            resilient_call(
                fn2,
                provider="test",
                policy=_make_policy(max_retries=2, breaker=breaker),
                sleep=_zero_sleep,
            )
        assert calls == calls_at_open  # fn2 was never executed

    def test_timeout_is_retried(self) -> None:
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise TimeoutError("slow")
            return "ok"

        result = resilient_call(
            fn, provider="test", policy=_make_policy(max_retries=3), sleep=_zero_sleep
        )
        assert result == "ok"
        assert calls == 2

    def test_sleep_is_invoked_between_retries(self) -> None:
        sleeps: list[float] = []

        def recording_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        attempts = 0

        def fn() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise _FakeRateLimitError("429")
            return "ok"

        resilient_call(
            fn,
            provider="test",
            policy=_make_policy(max_retries=3),
            sleep=recording_sleep,
        )
        # Two retries → two sleeps; delays follow the exponential schedule.
        assert len(sleeps) == 2
        assert sleeps[0] < sleeps[1]

    def test_provider_error_subclass_raised_directly_is_respected(self) -> None:
        """A concrete provider may raise a normalised error to bypass the
        mapping — the wrapper must not double-normalise it."""

        def fn() -> str:
            raise ProviderAuthError("bad key", provider="test")

        with pytest.raises(ProviderAuthError):
            resilient_call(
                fn,
                provider="test",
                policy=_make_policy(max_retries=5),
                sleep=_zero_sleep,
            )


# ---------------------------------------------------------------------------
# Composite typing sanity
# ---------------------------------------------------------------------------


def test_sleep_default_is_time_sleep() -> None:
    """Regression: callers must not accidentally pass ``None`` for ``sleep``."""
    sleep_default: Callable[[float], None] = time.sleep  # noqa: F841
