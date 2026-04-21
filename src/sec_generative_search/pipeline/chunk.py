"""
Text chunking for SEC filings.

This module splits long segments into smaller chunks suitable for embedding.
It uses sentence-boundary splitting to ensure chunks don't cut mid-sentence.

Section boundaries are preserved implicitly: the parser produces one
:class:`Segment` per section path (e.g. ``"Part I > Item 1A > Risk
Factors"``), and chunking is performed **per-segment** — chunks never
span across sections.

Usage:
    from sec_generative_search.pipeline import TextChunker

    chunker = TextChunker()
    chunks = chunker.chunk_segments(segments)
"""

import re

from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import ChunkingError
from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.types import Chunk, ContentType, Segment

logger = get_logger(__name__)


class TextChunker:
    """
    Splits segments into embedding-ready chunks.

    This class implements sentence-boundary aware chunking to ensure
    that text is split at natural boundaries rather than mid-sentence.

    The chunking algorithm:
        1. If segment fits within token limit, keep as-is
        2. Otherwise, split on sentence boundaries (. ! ?)
        3. Accumulate sentences until adding another would exceed limit
        4. Tolerance band allows slight overrun to avoid tiny final chunks
          5. Optional sentence-level overlap carries the tail of the
              previous chunk into the next one for context continuity

    Table segments (``ContentType.TABLE``) are never split — table
    structure is preserved as a single chunk even when the token count
    exceeds ``token_limit``.  Sentence-boundary splitting would
    otherwise corrupt row alignment.

    Attributes:
        token_limit: Maximum tokens per chunk (from settings).
        tolerance: Acceptable overrun tolerance (from settings).
        overlap_tokens: Target number of tokens of trailing context
            carried from one chunk into the next.  Zero disables
            overlap (from :class:`RAGSettings.chunk_overlap_tokens`).

    Example:
        >>> chunker = TextChunker()
        >>> chunks = chunker.chunk_segments(segments)
        >>> print(f"Created {len(chunks)} chunks")
    """

    # Sentence boundary pattern: split after . ! ? followed by whitespace
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        token_limit: int | None = None,
        tolerance: int | None = None,
        overlap_tokens: int | None = None,
    ) -> None:
        """
        Initialise the chunker with configurable limits.

        Args:
            token_limit: Max tokens per chunk. If None, uses settings.
            tolerance: Acceptable overrun. If None, uses settings.
            overlap_tokens: Sentence-level overlap tokens between
                adjacent chunks.  If ``None`` uses
                :attr:`RAGSettings.chunk_overlap_tokens`.  A negative
                value is rejected; zero disables overlap.
        """
        settings = get_settings()
        self.token_limit = token_limit or settings.chunking.token_limit
        self.tolerance = tolerance or settings.chunking.tolerance
        if overlap_tokens is None:
            overlap_tokens = settings.rag.chunk_overlap_tokens
        if overlap_tokens < 0:
            raise ValueError(f"overlap_tokens must be non-negative (got {overlap_tokens})")
        if overlap_tokens >= self.token_limit:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be strictly less than "
                f"token_limit ({self.token_limit}) to guarantee forward progress."
            )
        self.overlap_tokens = overlap_tokens

        logger.debug(
            "TextChunker initialised: limit=%d, tolerance=%d, overlap=%d",
            self.token_limit,
            self.tolerance,
            self.overlap_tokens,
        )

    def _count_tokens(self, text: str) -> int:
        """
        Approximate token count using whitespace splitting.

        This is a simple heuristic that works well for English text.
        More accurate tokenisation would require the actual model's
        tokeniser, but whitespace splitting is sufficient for chunking.

        Args:
            text: Text to count tokens in.

        Returns:
            Approximate token count.
        """
        return len(text.split())

    def _chunk_text(self, text: str) -> list[tuple[str, int]]:
        """
        Split text into chunks respecting sentence boundaries.

        When ``overlap_tokens > 0``, the trailing sentences of a
        finalised chunk (up to the overlap target) are retained as the
        seed of the next chunk — this preserves context across
        embedding boundaries without requiring the caller to
        reassemble the filing.

        Args:
            text: Text content to split.

        Returns:
            List of (chunk_text, token_count) tuples.
        """
        total_tokens = self._count_tokens(text)

        # If text already fits, return as single chunk
        if total_tokens <= self.token_limit:
            return [(text, total_tokens)]

        # Split on sentence boundaries
        sentences = self.SENTENCE_PATTERN.split(text)
        sentence_token_counts = [self._count_tokens(s) for s in sentences]

        chunks: list[tuple[str, int]] = []
        current_sentences: list[str] = []
        current_token_counts: list[int] = []
        current_tokens = 0

        for sentence, sentence_tokens in zip(sentences, sentence_token_counts, strict=True):
            # Check if adding this sentence would exceed limit + tolerance.
            # If so, finalise current chunk (unless it's empty) and seed
            # the next chunk with the trailing overlap sentences.
            if (
                current_tokens + sentence_tokens > self.token_limit + self.tolerance
                and current_sentences
            ):
                chunks.append((" ".join(current_sentences), current_tokens))
                if self.overlap_tokens > 0:
                    current_sentences, current_token_counts = self._tail_for_overlap(
                        current_sentences, current_token_counts
                    )
                    current_tokens = sum(current_token_counts)
                else:
                    current_sentences = []
                    current_token_counts = []
                    current_tokens = 0

            current_sentences.append(sentence)
            current_token_counts.append(sentence_tokens)
            current_tokens += sentence_tokens

        # Flush remaining sentences
        if current_sentences:
            chunks.append((" ".join(current_sentences), current_tokens))

        return chunks

    def _tail_for_overlap(
        self,
        sentences: list[str],
        token_counts: list[int],
    ) -> tuple[list[str], list[int]]:
        """Return the trailing sentences whose total tokens ≤ ``overlap_tokens``.

        Walks backward from the end of the just-finalised chunk,
        collecting whole sentences until the next sentence would push
        the running total over :attr:`overlap_tokens`.  Returning whole
        sentences (rather than an arbitrary word slice) keeps the
        carried-over context grammatically well-formed.
        """
        tail: list[str] = []
        tail_counts: list[int] = []
        running = 0
        for sentence, count in zip(reversed(sentences), reversed(token_counts), strict=True):
            # Skip pathological sentences that would single-handedly exceed
            # the overlap budget — including them would bloat the next
            # chunk and risk exceeding the hard token limit.
            if count > self.overlap_tokens:
                break
            if running + count > self.overlap_tokens:
                break
            tail.append(sentence)
            tail_counts.append(count)
            running += count
        tail.reverse()
        tail_counts.reverse()
        return tail, tail_counts

    def chunk_segment(self, segment: Segment, start_index: int = 0) -> list[Chunk]:
        """
        Split a single segment into chunks.

        Table segments are never split — their structure would be
        corrupted by sentence-boundary splitting, so they are emitted
        as a single chunk even when the token count exceeds the limit.
        Text segments are split per :meth:`_chunk_text`.

        Args:
            segment: Segment to chunk.
            start_index: Starting chunk index for this segment.

        Returns:
            List of Chunk objects with sequential indices.
        """
        if segment.content_type is ContentType.TABLE:
            token_count = self._count_tokens(segment.content)
            text_chunks: list[tuple[str, int]] = [(segment.content, token_count)]
        else:
            text_chunks = self._chunk_text(segment.content)

        return [
            Chunk(
                content=text,
                path=segment.path,
                content_type=segment.content_type,
                filing_id=segment.filing_id,
                chunk_index=start_index + i,
                token_count=tokens,
            )
            for i, (text, tokens) in enumerate(text_chunks)
        ]

    def chunk_segments(self, segments: list[Segment]) -> list[Chunk]:
        """
        Chunk all segments from a filing.

        This is the main entry point for chunking. It processes all
        segments and assigns sequential chunk indices across the
        entire filing.

        Args:
            segments: List of segments from FilingParser.

        Returns:
            List of Chunk objects ready for embedding.

        Raises:
            ChunkingError: If segments list is empty.

        Example:
            >>> chunks = chunker.chunk_segments(segments)
            >>> for chunk in chunks[:3]:
            ...     print(f"[{chunk.chunk_index}] {chunk.path[:50]}...")
        """
        if not segments:
            raise ChunkingError(
                "No segments to chunk",
                details="Received empty segments list.",
            )

        filing_id = segments[0].filing_id

        logger.info(
            "Chunking %d segments from %s %s",
            len(segments),
            filing_id.ticker,
            filing_id.form_type,
        )

        chunks: list[Chunk] = []
        current_index = 0

        for segment in segments:
            segment_chunks = self.chunk_segment(segment, start_index=current_index)
            chunks.extend(segment_chunks)
            current_index += len(segment_chunks)

        # Log statistics — token counts are retained from chunking, no recount
        token_counts = [c.token_count for c in chunks]
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        avg_tokens = sum(token_counts) / len(token_counts)
        over_limit = sum(1 for t in token_counts if t > self.token_limit)

        logger.info(
            "Created %d chunks from %d segments (tokens: %d-%d, avg %.0f, %d over limit)",
            len(chunks),
            len(segments),
            min_tokens,
            max_tokens,
            avg_tokens,
            over_limit,
        )

        return chunks
