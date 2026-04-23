"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from sec_generative_search.core.exceptions import (
    CitationError,
    ConfigurationError,
    DatabaseError,
    EmbeddingCollectionMismatchError,
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
from sec_generative_search.core.types import EmbedderStamp


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
            EmbeddingCollectionMismatchError,
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


class TestEmbeddingCollectionMismatchError:
    def _stamps(self) -> tuple[EmbedderStamp, EmbedderStamp]:
        expected = EmbedderStamp(provider="openai", model="text-embedding-3-small", dimension=1536)
        actual = EmbedderStamp(provider="local", model="google/embeddinggemma-300m", dimension=768)
        return expected, actual

    def test_still_a_database_error(self) -> None:
        expected, actual = self._stamps()
        err = EmbeddingCollectionMismatchError(expected=expected, actual=actual)
        assert isinstance(err, DatabaseError)
        assert isinstance(err, SECGenerativeSearchError)

    def test_message_names_both_stamps(self) -> None:
        expected, actual = self._stamps()
        err = EmbeddingCollectionMismatchError(expected=expected, actual=actual)
        message = str(err)
        # Expected triple
        assert "openai" in message
        assert "text-embedding-3-small" in message
        assert "1536" in message
        # Actual triple
        assert "local" in message
        assert "google/embeddinggemma-300m" in message
        assert "768" in message

    def test_stamps_preserved_as_attributes(self) -> None:
        expected, actual = self._stamps()
        err = EmbeddingCollectionMismatchError(expected=expected, actual=actual)
        assert err.expected is expected
        assert err.actual is actual

    def test_hint_is_uniform_across_deployments(self) -> None:
        """The hint is deliberately scenario-unaware.

        Two instances with different stamps must share the same hint
        string — the storage layer does not branch on deployment
        profile, and any future change that introduces branching would
        trip this assertion.
        """
        expected_a, actual_a = self._stamps()
        err_a = EmbeddingCollectionMismatchError(expected=expected_a, actual=actual_a)

        expected_b = EmbedderStamp(provider="gemini", model="text-embedding-004", dimension=768)
        actual_b = EmbedderStamp(provider="mistral", model="mistral-embed", dimension=1024)
        err_b = EmbeddingCollectionMismatchError(expected=expected_b, actual=actual_b)
        assert err_a.hint == err_b.hint
        # Spot-check the hint is the "reindex" guidance, not e.g. a
        # sanitised empty string.
        assert "reindex" in err_a.hint.lower()


@pytest.mark.security
class TestEmbeddingCollectionMismatchErrorCredentialHygiene:
    """The mismatch error must never echo a secret-shaped string."""

    def test_no_credential_words_in_hint_or_message(self) -> None:
        expected = EmbedderStamp(provider="openai", model="text-embedding-3-small", dimension=1536)
        actual = EmbedderStamp(provider="gemini", model="text-embedding-004", dimension=768)
        err = EmbeddingCollectionMismatchError(expected=expected, actual=actual)
        rendered = f"{err.message} {err.hint}".lower()
        for needle in ("api_key", "apikey", "secret", "password", "bearer"):
            assert needle not in rendered
