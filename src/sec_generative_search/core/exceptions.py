"""
Custom exception hierarchy for SEC-GenerativeSearch.

All exceptions inherit from SECGenerativeSearchError, allowing callers to catch
all project-specific errors with a single except clause when desired.

``EmbeddingCollectionMismatchError`` references :class:`EmbedderStamp` only in
its signature; the ``TYPE_CHECKING`` guard keeps this module free of a runtime
dependency on :mod:`core.types`, preserving the exceptions module's position
low in the import graph.

Exception hierarchy:
    SECGenerativeSearchError (base)
    ├── ConfigurationError — Invalid or missing configuration
    ├── FetchError — SEC EDGAR API or network failures
    ├── ParseError — HTML parsing failures (doc2dict)
    ├── ChunkingError — Text chunking failures
    ├── EmbeddingError — Embedding generation failures
    ├── DatabaseError — ChromaDB or SQLite failures
    │   ├── FilingLimitExceededError — Maximum filing count reached
    │   └── EmbeddingCollectionMismatchError — Collection stamp disagrees with configured embedder
    ├── SearchError — Search operation failures
    ├── ProviderError — External LLM/embedding provider failures
    │   ├── ProviderAuthError — Invalid or expired API key
    │   ├── ProviderRateLimitError — Provider rate limit exceeded
    │   ├── ProviderTimeoutError — Provider call timed out
    │   └── ProviderContentFilterError — Content blocked by provider safety filter
    ├── GenerationError — RAG answer generation failures
    ├── PromptError — Prompt template or injection-detection failures
    └── CitationError — Citation extraction or validation failures
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sec_generative_search.core.types import EmbedderStamp


class SECGenerativeSearchError(Exception):
    """
    Base exception for all SEC-GenerativeSearch errors.

    Args:
        message: Human-readable error description.
        details: Optional additional context for debugging.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        self.message = message
        self.details = details
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.details:
            return f"{self.message} — {self.details}"
        return self.message


class ConfigurationError(SECGenerativeSearchError):
    """
    Raised when configuration is invalid or missing.

    Examples:
        - Missing required environment variables (EDGAR_IDENTITY_NAME)
        - Invalid configuration values (negative token limits)
        - Missing .env file when required
    """


class FetchError(SECGenerativeSearchError):
    """
    Raised when fetching SEC filings fails.

    Examples:
        - Network connectivity issues
        - SEC EDGAR API rate limiting
        - Invalid ticker symbol
        - Filing not found for specified form type
    """


class ParseError(SECGenerativeSearchError):
    """
    Raised when parsing filing HTML fails.

    Examples:
        - Malformed HTML content
        - Unexpected document structure
        - doc2dict library errors
    """


class ChunkingError(SECGenerativeSearchError):
    """
    Raised when text chunking fails.

    Examples:
        - Empty content after parsing
        - Chunking algorithm failures
    """


class EmbeddingError(SECGenerativeSearchError):
    """
    Raised when embedding generation fails.

    Examples:
        - Model loading failures
        - GPU memory exhaustion
        - Invalid input to embedding model
    """


class DatabaseError(SECGenerativeSearchError):
    """
    Raised when database operations fail.

    Examples:
        - ChromaDB connection failures
        - SQLite write errors
        - Collection not found
    """


class FilingLimitExceededError(DatabaseError):
    """
    Raised when the maximum filing limit is reached.

    This is a soft limit for portfolio project scope, configurable via
    DB_MAX_FILINGS environment variable.
    """

    def __init__(
        self,
        current_count: int,
        max_filings: int,
        details: str | None = None,
    ) -> None:
        self.current_count = current_count
        self.max_filings = max_filings
        message = (
            f"Filing limit exceeded: {current_count}/{max_filings} filings stored. "
            f"Remove existing filings or increase DB_MAX_FILINGS."
        )
        super().__init__(message, details)


class EmbeddingCollectionMismatchError(DatabaseError):
    """
    Raised when a ChromaDB collection's embedder stamp disagrees with
    the configured embedder.

    The storage layer stamps every collection with its ``(provider,
    model, dimension)`` triple at creation and verifies that the
    configured embedder matches at every subsequent initialisation.  A
    mismatch means the configured embedder would produce vectors that
    are incompatible with the stored vectors — retrieval would silently
    return garbage results, which is the failure mode this error exists
    to prevent.

    The ``hint`` text is deliberately a **single uniform string**
    applicable across local, team, and cloud deployments — the storage
    layer stays scenario-unaware.  The API lifespan hook is the only
    place that is scenario-aware; it translates this exception into a
    503 sentinel with a redacted body for team/cloud.

    Attributes:
        expected: The stamp the configured embedder would produce.
        actual: The stamp read from the collection's metadata.
        hint: Uniform operator guidance for all three deployment
            profiles.
    """

    _UNIFORM_HINT = (
        "Rebuild the collection with 'sec-rag manage reindex' so it matches the "
        "configured embedder. In shared deployments an operator runs the same "
        "command on the server; never point the application at a mismatched "
        "collection — retrieval results would be silently wrong."
    )

    def __init__(
        self,
        expected: EmbedderStamp,
        actual: EmbedderStamp,
        details: str | None = None,
    ) -> None:
        self.expected = expected
        self.actual = actual
        self.hint = self._UNIFORM_HINT
        message = (
            f"Embedding collection stamp mismatch: expected "
            f"({expected.provider}, {expected.model}, dim={expected.dimension}) "
            f"but collection is stamped "
            f"({actual.provider}, {actual.model}, dim={actual.dimension})."
        )
        super().__init__(message, details)


class SearchError(SECGenerativeSearchError):
    """
    Raised when search operations fail.

    Examples:
        - Empty query string
        - No filings ingested
        - ChromaDB query failures
    """


# ---------------------------------------------------------------------------
# Provider errors — external LLM / embedding API failures
# ---------------------------------------------------------------------------


class ProviderError(SECGenerativeSearchError):
    """
    Base exception for external LLM/embedding provider failures.

    All provider-specific errors (auth, rate limit, timeout, content filter)
    inherit from this class, enabling callers to catch any provider failure
    generically.

    Attributes:
        provider: Name of the provider that failed (e.g. "openai", "anthropic").
        hint: Optional actionable suggestion for the user.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        hint: str | None = None,
        details: str | None = None,
    ) -> None:
        self.provider = provider
        self.hint = hint
        super().__init__(message, details)


class ProviderAuthError(ProviderError):
    """Raised when a provider API key is invalid, expired, or missing."""


class ProviderRateLimitError(ProviderError):
    """Raised when a provider's rate limit is exceeded.

    The caller should respect ``Retry-After`` headers or apply
    exponential backoff before retrying.
    """


class ProviderTimeoutError(ProviderError):
    """Raised when a provider API call exceeds the configured timeout."""


class ProviderContentFilterError(ProviderError):
    """Raised when a provider's safety filter blocks the request or response."""


# ---------------------------------------------------------------------------
# RAG pipeline errors — generation, prompt, and citation failures
# ---------------------------------------------------------------------------


class GenerationError(SECGenerativeSearchError):
    """
    Raised when RAG answer generation fails.

    Examples:
        - Prompt assembly failure (context too large for model window)
        - Model output parsing failure (malformed structured output)
        - Unexpected provider response format
    """


class PromptError(SECGenerativeSearchError):
    """
    Raised for prompt template or injection-detection failures.

    Examples:
        - Missing template variables
        - Prompt injection detected in retrieved filing content
        - Template rendering failure
    """


class CitationError(SECGenerativeSearchError):
    """
    Raised when citation extraction or validation fails.

    Examples:
        - Citation references a chunk ID not present in retrieval results
        - Citation span does not match source text
        - Malformed citation format in model output
    """
