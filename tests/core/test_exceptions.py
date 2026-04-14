"""Tests for the exception hierarchy (Phase 1.4)."""

from __future__ import annotations

import pytest

from sec_generative_search.core.exceptions import (
    CitationError,
    ConfigurationError,
    DatabaseError,
    FetchError,
    FilingLimitExceededError,
    GenerationError,
    PromptError,
    ProviderAuthError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    SearchError,
    SECGenerativeSearchError,
)


class TestBaseException:
    def test_message_only(self) -> None:
        err = SECGenerativeSearchError("something failed")
        assert str(err) == "something failed"
        assert err.message == "something failed"
        assert err.details is None

    def test_message_with_details(self) -> None:
        err = SECGenerativeSearchError("something failed", details="ticker=AAPL")
        assert str(err) == "something failed — ticker=AAPL"
        assert err.details == "ticker=AAPL"


class TestHierarchy:
    """Every subclass must be catchable via the base type."""

    @pytest.mark.parametrize(
        "cls",
        [
            ConfigurationError,
            FetchError,
            DatabaseError,
            SearchError,
            ProviderError,
            ProviderAuthError,
            ProviderRateLimitError,
            ProviderTimeoutError,
            ProviderContentFilterError,
            GenerationError,
            PromptError,
            CitationError,
        ],
    )
    def test_inherits_from_base(self, cls: type[Exception]) -> None:
        assert issubclass(cls, SECGenerativeSearchError)

    def test_provider_subtypes_inherit_from_provider_error(self) -> None:
        for cls in (
            ProviderAuthError,
            ProviderRateLimitError,
            ProviderTimeoutError,
            ProviderContentFilterError,
        ):
            assert issubclass(cls, ProviderError)


class TestFilingLimitExceededError:
    def test_message_format(self) -> None:
        err = FilingLimitExceededError(current_count=500, max_filings=500)
        assert "500/500" in str(err)
        assert err.current_count == 500
        assert err.max_filings == 500

    def test_still_a_database_error(self) -> None:
        err = FilingLimitExceededError(current_count=1, max_filings=1)
        assert isinstance(err, DatabaseError)
        assert isinstance(err, SECGenerativeSearchError)


class TestProviderError:
    def test_captures_provider_and_hint(self) -> None:
        err = ProviderError(
            "auth failed",
            provider="openai",
            hint="rotate your API key",
        )
        assert err.provider == "openai"
        assert err.hint == "rotate your API key"
        assert err.message == "auth failed"

    def test_subclass_captures_provider_metadata(self) -> None:
        err = ProviderRateLimitError(
            "429 from Anthropic",
            provider="anthropic",
            hint="retry after 60s",
        )
        assert err.provider == "anthropic"
        assert err.hint == "retry after 60s"

    def test_raiseable_and_catchable(self) -> None:
        with pytest.raises(ProviderError) as exc_info:
            raise ProviderAuthError("bad key", provider="gemini")
        assert exc_info.value.provider == "gemini"
