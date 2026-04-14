"""
Core data types for SEC-GenerativeSearch.

This module defines the domain objects used throughout the pipeline:
    - FilingIdentifier: Unique identifier for an SEC filing
    - Segment: Parsed content unit from a filing
    - Chunk: Embedding-ready text unit
    - SearchResult: Query result with similarity score
    - IngestResult: Ingestion pipeline outcome
    - RetrievalResult: Search result enriched with context-window metadata
    - Citation: Immutable record of a chunk referenced in a generated answer
    - TokenUsage: Input/output token counts for a provider call or session
    - GenerationResult: Outcome of a RAG generation request with traceability
    - ConversationTurn: Session-scoped audit record of a query/answer pair
    - ProviderCapability: Feature matrix for an LLM/embedding provider-model pair
    - PricingTier: Coarse pricing classification for providers

Design notes:
    - Dataclasses are used for simplicity and performance (no runtime validation)
    - FilingIdentifier, Citation, and ProviderCapability are frozen (immutable)
      because each serves as an identifier or audit record that must not mutate
      after construction
    - ContentType and PricingTier enums enforce type-safe classification
    - Pydantic is limited to settings and API schemas only — domain objects are
      dataclasses (see AGENT.md "Must-follow constraints")
    - No domain type stores a provider API key, the EDGAR identity, or the DB
      encryption key — secrets never travel through the core type surface
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class ContentType(Enum):
    """
    Content types extracted from SEC filings via doc2dict.

    Values:
        TEXT: Regular paragraph text
        TEXTSMALL: Smaller text elements (footnotes, captions)
        TABLE: Tabular data converted to text representation
    """

    TEXT = "text"
    TEXTSMALL = "textsmall"
    TABLE = "table"


@dataclass(frozen=True)
class FilingIdentifier:
    """
    Unique identifier for an SEC filing.

    This immutable identifier is used to track filings throughout the pipeline
    and serves as the primary key in the metadata registry.

    Attributes:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
        form_type: SEC form type (e.g., "10-K", "10-Q")
        filing_date: Date the filing was submitted to SEC
        accession_number: SEC-assigned unique identifier (e.g., "0000320193-23-000077")

    Example:
        >>> filing_id = FilingIdentifier(
        ...     ticker="AAPL",
        ...     form_type="10-K",
        ...     filing_date=date(2023, 11, 3),
        ...     accession_number="0000320193-23-000077",
        ... )
    """

    ticker: str
    form_type: str
    filing_date: date
    accession_number: str

    def __post_init__(self) -> None:
        """Validate and normalise field values."""
        # Use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "ticker", self.ticker.upper())
        object.__setattr__(self, "form_type", self.form_type.upper())

    @property
    def date_str(self) -> str:
        """Return filing date as ISO format string (YYYY-MM-DD)."""
        return self.filing_date.isoformat()


@dataclass
class Segment:
    """
    A semantically meaningful unit of content extracted from a filing.

    Segments are created by the parser from doc2dict output. Each segment
    represents a coherent piece of content (paragraph, table, footnote)
    with its hierarchical location in the document.

    Attributes:
        path: Hierarchical path (e.g., "Part I > Item 1A > Risk Factors")
        content_type: Type of content (text, textsmall, table)
        content: The actual text content
        filing_id: Reference to the source filing

    Example:
        >>> segment = Segment(
        ...     path="Part I > Item 1A > Risk Factors",
        ...     content_type=ContentType.TEXT,
        ...     content="Our business is subject to...",
        ...     filing_id=filing_id,
        ... )
    """

    path: str
    content_type: ContentType
    content: str
    filing_id: FilingIdentifier


@dataclass
class Chunk:
    """
    An embedding-ready text unit derived from a segment.

    Chunks are created by splitting long segments at sentence boundaries.
    Each chunk inherits metadata from its source segment and is assigned
    a unique index for ChromaDB storage.

    Attributes:
        content: The text content (respects token limit)
        path: Inherited hierarchical path from source segment
        content_type: Inherited content type from source segment
        filing_id: Reference to the source filing
        chunk_index: Zero-based index within the filing's chunks

    The chunk_id property generates the ChromaDB document ID in the format:
        {TICKER}_{FORM_TYPE}_{DATE}_{INDEX}
    """

    content: str
    path: str
    content_type: ContentType
    filing_id: FilingIdentifier
    chunk_index: int = field(default=0)
    token_count: int = field(default=0)

    @property
    def chunk_id(self) -> str:
        """
        Generate unique ChromaDB document ID.

        Format: {TICKER}_{FORM_TYPE}_{DATE}_{INDEX}
        Example: AAPL_10-K_2023-11-03_042
        """
        return (
            f"{self.filing_id.ticker}_"
            f"{self.filing_id.form_type}_"
            f"{self.filing_id.date_str}_"
            f"{self.chunk_index:03d}"
        )

    def to_metadata(self) -> dict:
        """
        Convert chunk metadata to ChromaDB-compatible dict.

        Returns:
            Dictionary suitable for ChromaDB metadata. Includes both
            ``filing_date`` (ISO string for display) and ``filing_date_int``
            (``YYYYMMDD`` integer for range queries with ``$gte``/``$lte``).
        """
        return {
            "path": self.path,
            "content_type": self.content_type.value,
            "ticker": self.filing_id.ticker,
            "form_type": self.filing_id.form_type,
            "filing_date": self.filing_id.date_str,
            "filing_date_int": int(self.filing_id.date_str.replace("-", "")),
            "accession_number": self.filing_id.accession_number,
        }


@dataclass
class SearchResult:
    """
    A single result from a semantic search query.

    Search results are returned by the search engine, ranked by similarity.
    Each result contains the matched chunk content along with its metadata
    and relevance score.

    Attributes:
        content: The matched chunk text
        path: Hierarchical path in the source document
        content_type: Type of content (text, textsmall, table)
        ticker: Stock ticker of the source filing
        form_type: SEC form type of the source filing
        similarity: Cosine similarity score (0.0 to 1.0, higher is better)
        filing_date: Date of the source filing (optional)
        accession_number: SEC accession number (optional)
        chunk_id: ChromaDB document ID (optional)
    """

    content: str
    path: str
    content_type: ContentType
    ticker: str
    form_type: str
    similarity: float
    filing_date: str | None = None
    accession_number: str | None = None
    chunk_id: str | None = None

    @classmethod
    def from_chromadb_result(
        cls,
        document: str,
        metadata: dict,
        distance: float,
        chunk_id: str | None = None,
    ) -> "SearchResult":
        """
        Create SearchResult from ChromaDB query output.

        ChromaDB returns cosine distance; this method converts it to
        similarity (1 - distance).

        Args:
            document: The chunk text content
            metadata: ChromaDB metadata dictionary
            distance: Cosine distance from ChromaDB (0.0 to 2.0)
            chunk_id: Optional document ID

        Returns:
            SearchResult instance with similarity score
        """
        return cls(
            content=document,
            path=metadata.get("path", "(unknown)"),
            content_type=ContentType(metadata.get("content_type", "text")),
            ticker=metadata.get("ticker", ""),
            form_type=metadata.get("form_type", ""),
            similarity=1.0 - distance,
            filing_date=metadata.get("filing_date"),
            accession_number=metadata.get("accession_number"),
            chunk_id=chunk_id,
        )


@dataclass
class IngestResult:
    """
    Result of a filing ingestion operation.

    Returned by the pipeline orchestrator after successfully ingesting
    a filing, providing statistics for CLI output and logging.

    Attributes:
        filing_id: Identifier of the ingested filing
        segment_count: Number of segments extracted from HTML
        chunk_count: Number of chunks after splitting
        duration_seconds: Time taken for the full pipeline
    """

    filing_id: FilingIdentifier
    segment_count: int
    chunk_count: int
    duration_seconds: float


# ---------------------------------------------------------------------------
# RAG domain types — retrieval, citations, and generation
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult(SearchResult):
    """
    A search result enriched with metadata for context-window packing.

    ``RetrievalResult`` extends :class:`SearchResult` with the bookkeeping the
    RAG orchestrator needs when assembling prompt context: per-chunk token
    count, a flag for whether the chunk was clipped to fit the budget, and
    the parsed hierarchical section path.  Subclassing (rather than
    composition) keeps the type assignable wherever a ``SearchResult`` is
    accepted, which matters for the public retrieval API in Phase 7.

    Attributes:
        token_count: Token count of ``content`` measured by the active
            tokenizer.  Zero means "not yet counted".
        truncated: ``True`` when ``content`` was shortened to honour the
            context-token budget.  Surfaces retrieval-overshoot cases to
            the caller and the UI without re-reading the source chunk.
        section_boundaries: Tuple of the parsed ``path`` components, e.g.
            ``("Part I", "Item 1A", "Risk Factors")``.  Used by diversity
            controls (Phase 7.5) and by citation display logic to avoid
            re-parsing ``path`` at generation time.
    """

    token_count: int = 0
    truncated: bool = False
    section_boundaries: tuple[str, ...] = ()

    @classmethod
    def from_search_result(
        cls,
        result: SearchResult,
        *,
        token_count: int = 0,
        truncated: bool = False,
        section_boundaries: tuple[str, ...] | None = None,
    ) -> "RetrievalResult":
        """Lift a :class:`SearchResult` into a :class:`RetrievalResult`.

        When ``section_boundaries`` is not provided it is derived from
        ``result.path`` by splitting on the `` > `` separator used by the
        filing parser.  Pass an explicit tuple to override (e.g. when the
        caller has already parsed the path).
        """
        if section_boundaries is None:
            section_boundaries = tuple(
                part.strip() for part in result.path.split(">") if part.strip()
            )
        return cls(
            content=result.content,
            path=result.path,
            content_type=result.content_type,
            ticker=result.ticker,
            form_type=result.form_type,
            similarity=result.similarity,
            filing_date=result.filing_date,
            accession_number=result.accession_number,
            chunk_id=result.chunk_id,
            token_count=token_count,
            truncated=truncated,
            section_boundaries=section_boundaries,
        )


@dataclass(frozen=True)
class Citation:
    """
    Immutable record of a chunk referenced by a generated answer.

    Citations are the audit trail of which retrieved sources the model
    actually used.  They are frozen because, once written into a
    :class:`GenerationResult`, they must not be rewritten — that would
    break answer traceability (Phase 8.7).

    Attributes:
        chunk_id: ChromaDB document ID of the cited chunk.  Matches the
            ``chunk_id`` of a :class:`RetrievalResult` in the same
            :class:`GenerationResult`.
        filing_id: Source filing identifier (ticker, form, date, accession).
        section_path: Hierarchical path, e.g. ``"Part I > Item 1A >
            Risk Factors"``.  Rendered in the UI's source panel.
        text_span: The excerpt the answer quoted or paraphrased.  This is
            Tier 1 (public filing) content — safe to persist and return.
        similarity: Cosine similarity score of the cited chunk, preserved
            for transparency in the UI.
        display_index: One-based ordinal used to render inline citation
            markers like ``[1]`` or ``[2]`` in the answer.  Default of
            ``0`` means "not yet assigned".
    """

    chunk_id: str
    filing_id: FilingIdentifier
    section_path: str
    text_span: str
    similarity: float
    display_index: int = 0


@dataclass
class TokenUsage:
    """
    Input/output token counts for a provider call or aggregated session.

    Promoted to its own type (rather than two ``int`` fields on
    :class:`GenerationResult`) because the same shape is produced by every
    provider call (Phase 5.16), aggregated across conversation turns
    (Phase 8.9), and surfaced in dashboards (Phase 14.2).

    Attributes:
        input_tokens: Tokens consumed by the prompt (system + context +
            user query).
        output_tokens: Tokens produced by the model's response.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Return the sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Combine two usage records — useful for aggregating turn totals."""
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass
class GenerationResult:
    """
    Outcome of a RAG answer-generation request.

    Carries both the chunks *fed to the model* (``retrieved_chunks``) and
    the chunks *referenced in the answer* (``citations``).  The
    distinction diagnoses two separate failure modes — retrieval
    overshoot vs. model-ignores-context — that look identical from the
    final answer alone (AGENT.md "RAG orchestrator" section).

    Attributes:
        answer: The generated answer text.  May contain inline citation
            markers (``[1]``) when :attr:`RAGSettings.citation_mode` is
            ``"inline"``.
        provider: Registered provider key used (e.g. ``"openai"``,
            ``"anthropic"``).  Never contains a key or credential.
        model: Provider-specific model slug used (e.g. ``"gpt-4o"``).
        prompt_version: Version string of the prompt template used,
            captured so that answer regressions can be traced to a
            template change (Phase 8.6).
        citations: Chunks the model actually referenced in the answer.
        retrieved_chunks: Chunks the orchestrator fed to the model.
            A superset of the chunk IDs referenced by ``citations``.
        token_usage: Input/output token accounting.
        latency_seconds: Wall-clock duration of the generation call.
        streamed: ``True`` when the answer was produced via streaming.
    """

    answer: str
    provider: str
    model: str
    prompt_version: str
    citations: list[Citation] = field(default_factory=list)
    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    latency_seconds: float = 0.0
    streamed: bool = False


@dataclass
class ConversationTurn:
    """
    Session-scoped audit record of a query/answer pair.

    Conversation history is **off by default** (Phase 3.3,
    ``RAG_CHAT_HISTORY_ENABLED=0``); when enabled it is encrypted at rest
    via SQLCipher.  Follow-up turns re-retrieve fresh context — the
    ``retrieval_results`` stored here are kept for audit and debugging,
    not for re-feeding into a later prompt (AGENT.md "RAG orchestrator").

    Attributes:
        query: The user's question.  Tier 3 data; must pass through
            ``redact_for_log()`` before any log emission.
        retrieval_results: Chunks retrieved for this turn.
        generation_result: The answer and its provenance.
        timestamp: UTC timestamp at turn creation time.
    """

    query: str
    retrieval_results: list[RetrievalResult]
    generation_result: GenerationResult
    timestamp: datetime


# ---------------------------------------------------------------------------
# Provider capability types
# ---------------------------------------------------------------------------


class PricingTier(Enum):
    """
    Coarse pricing classification for providers and models.

    Used by the UI to surface "cheap vs. premium" guidance and by the
    CLI to pick sensible defaults when the user has not specified a
    model.  Not a billing system — providers report their own true
    costs via :class:`TokenUsage` accounting downstream.
    """

    FREE = "free"
    LOW = "low"
    STANDARD = "standard"
    PREMIUM = "premium"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderCapability:
    """
    Feature matrix for a concrete provider + model pair.

    Populated by :class:`ProviderRegistry` validation (Phase 5.13, 9.3) —
    tiny dry-run calls probe which features actually work, since some
    providers advertise capabilities they mishandle under real load
    (AGENT.md "Providers" section).

    Frozen because capabilities are resolved once per provider/model
    registration; swapping the model means constructing a new instance
    rather than mutating an existing one.

    No API key or credential is stored here — validation consumes the
    key, verifies it, and only the resulting capability matrix is kept.

    Attributes:
        chat: Supports chat completions.  Providers that cannot do this
            are hidden from the generator UI entirely.
        embeddings: Supports text embeddings (applies to embedding-only
            providers and dual providers such as OpenAI).
        streaming: Supports token-by-token streaming of responses.
        tool_use: Supports function/tool calling.
        structured_output: Supports JSON-mode / schema-constrained
            output (used by the query-understanding step).
        prompt_caching: Supports prompt-prefix caching to reduce repeat-
            query cost.
        vision: Supports image inputs (not used in v1 but probed so the
            capability surface is stable).
        context_window_tokens: Maximum total tokens the model accepts
            (prompt + response).  ``0`` means unknown.
        max_output_tokens: Maximum response tokens the model will emit.
            ``0`` means unknown.
        pricing_tier: Coarse pricing bucket (see :class:`PricingTier`).
    """

    chat: bool = False
    embeddings: bool = False
    streaming: bool = False
    tool_use: bool = False
    structured_output: bool = False
    prompt_caching: bool = False
    vision: bool = False
    context_window_tokens: int = 0
    max_output_tokens: int = 0
    pricing_tier: PricingTier = PricingTier.UNKNOWN
