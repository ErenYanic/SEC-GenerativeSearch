"""Tests for :mod:`sec_generative_search.pipeline.chunk`.

Covers the chunker's core guarantees:
    - Sentence-boundary aware splitting stays within the token budget
      (``token_limit + tolerance``) where possible.
    - ``ContentType.TABLE`` segments are emitted whole, even when they
      exceed ``token_limit`` — row alignment must survive chunking.
    - Sentence-level overlap carries the trailing sentences of a
      finalised chunk into the next chunk; ``overlap_tokens=0``
      disables the behaviour entirely.
    - Chunk boundaries never cross segments (section boundaries are
      preserved implicitly by chunking per segment).
    - Constructor rejects invalid overlap configuration eagerly rather
      than blowing up mid-chunk.
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import pytest

from sec_generative_search.core.exceptions import ChunkingError
from sec_generative_search.core.types import (
    ContentType,
    FilingIdentifier,
    Segment,
)
from sec_generative_search.pipeline.chunk import TextChunker


@pytest.fixture
def filing_id() -> FilingIdentifier:
    return FilingIdentifier(
        ticker="AAPL",
        form_type="10-K",
        filing_date=date(2023, 11, 3),
        accession_number="0000320193-23-000077",
    )


def _make_segment(
    content: str,
    filing_id: FilingIdentifier,
    path: str = "Part I > Item 1A",
    content_type: ContentType = ContentType.TEXT,
) -> Segment:
    return Segment(
        path=path,
        content_type=content_type,
        content=content,
        filing_id=filing_id,
    )


class TestChunkerInit:
    def test_defaults_from_settings(self, clean_env: pytest.MonkeyPatch) -> None:
        chunker = TextChunker()
        # Defaults match constants exposed via Settings — see settings.py.
        assert chunker.token_limit == 500
        assert chunker.tolerance == 50
        assert chunker.overlap_tokens == 50

    def test_explicit_arguments_override_settings(self, clean_env: pytest.MonkeyPatch) -> None:
        chunker = TextChunker(token_limit=100, tolerance=10, overlap_tokens=0)
        assert chunker.token_limit == 100
        assert chunker.tolerance == 10
        assert chunker.overlap_tokens == 0

    def test_negative_overlap_tokens_rejected(self, clean_env: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            TextChunker(overlap_tokens=-1)

    def test_overlap_not_less_than_limit_rejected(self, clean_env: pytest.MonkeyPatch) -> None:
        # Overlap must be strictly less than token_limit, otherwise the
        # chunker cannot make forward progress.
        with pytest.raises(ValueError, match="strictly less than"):
            TextChunker(token_limit=50, overlap_tokens=50)

    def test_overlap_equal_to_limit_rejected(self, clean_env: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            TextChunker(token_limit=100, overlap_tokens=100)


class TestChunkSegments:
    def test_empty_list_raises(self, clean_env: pytest.MonkeyPatch) -> None:
        chunker = TextChunker()
        with pytest.raises(ChunkingError, match="No segments"):
            chunker.chunk_segments([])

    def test_short_segment_kept_whole(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        chunker = TextChunker(token_limit=100, tolerance=10, overlap_tokens=0)
        seg = _make_segment("one two three four.", filing_id)
        chunks = chunker.chunk_segments([seg])

        assert len(chunks) == 1
        assert chunks[0].content == "one two three four."
        assert chunks[0].chunk_index == 0
        assert chunks[0].token_count == 4
        assert chunks[0].path == seg.path

    def test_long_segment_split_on_sentence_boundary(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        chunker = TextChunker(token_limit=10, tolerance=2, overlap_tokens=0)
        text = " ".join([f"Sentence {i} has five tokens." for i in range(6)])
        seg = _make_segment(text, filing_id)
        chunks = chunker.chunk_segments([seg])

        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk must end at a sentence boundary.
            assert chunk.content.rstrip().endswith(".")
            # Tokens stay within limit + tolerance (unless the single
            # sentence on its own exceeds it — not the case here).
            assert chunk.token_count <= chunker.token_limit + chunker.tolerance

    def test_sequential_chunk_indices(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        chunker = TextChunker(token_limit=10, tolerance=2, overlap_tokens=0)
        seg_a = _make_segment(
            " ".join([f"A{i} sentence filler here." for i in range(4)]),
            filing_id,
            path="Part I > Item 1",
        )
        seg_b = _make_segment(
            " ".join([f"B{i} sentence filler here." for i in range(4)]),
            filing_id,
            path="Part I > Item 2",
        )
        chunks = chunker.chunk_segments([seg_a, seg_b])

        # Indices must run 0..N-1 with no gaps — storage-layer IDs depend
        # on this.
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunks_do_not_span_segments(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        chunker = TextChunker(token_limit=500, tolerance=50, overlap_tokens=0)
        seg_a = _make_segment("Section A content.", filing_id, path="Part I > Item 1")
        seg_b = _make_segment("Section B content.", filing_id, path="Part II > Item 7")
        chunks = chunker.chunk_segments([seg_a, seg_b])

        # Each chunk's path must match exactly one segment — chunks must
        # never be stitched across section boundaries.
        paths = {c.path for c in chunks}
        assert paths == {"Part I > Item 1", "Part II > Item 7"}


class TestTableNeverSplit:
    def test_oversized_table_is_single_chunk(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        # 400 tokens — well over the 10-token limit.  Sentence-boundary
        # splitting would destroy row alignment; the chunker must emit
        # the whole table as one chunk.
        table_content = "\n".join(f"Row{i} | col_a | col_b | col_c" for i in range(100))
        chunker = TextChunker(token_limit=10, tolerance=2, overlap_tokens=0)
        seg = _make_segment(table_content, filing_id, content_type=ContentType.TABLE)

        chunks = chunker.chunk_segments([seg])

        assert len(chunks) == 1
        assert chunks[0].content == table_content
        assert chunks[0].content_type is ContentType.TABLE
        assert chunks[0].token_count > chunker.token_limit


class TestOverlap:
    def _build_segment(self, filing_id: FilingIdentifier) -> Segment:
        # 12 sentences * 5 tokens = 60 tokens — forces several splits at
        # token_limit=15.
        text = " ".join([f"Sentence {i} has five tokens." for i in range(12)])
        return _make_segment(text, filing_id)

    def test_overlap_zero_produces_disjoint_chunks(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        chunker = TextChunker(token_limit=15, tolerance=2, overlap_tokens=0)
        seg = self._build_segment(filing_id)
        chunks = chunker.chunk_segments([seg])

        assert len(chunks) >= 2
        total_tokens = sum(c.token_count for c in chunks)
        # No overlap → total tokens equals the segment's raw token count.
        assert total_tokens == len(seg.content.split())

    def test_overlap_repeats_trailing_sentences(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        chunker = TextChunker(token_limit=15, tolerance=2, overlap_tokens=10)
        seg = self._build_segment(filing_id)
        chunks = chunker.chunk_segments([seg])

        assert len(chunks) >= 2
        # The start of each chunk (except the first) must be a suffix of
        # the previous chunk — i.e. the overlap is real, not just padding.
        for prev, curr in pairwise(chunks):
            first_sentence = curr.content.split(".", 1)[0] + "."
            assert first_sentence in prev.content

    def test_overlap_respects_budget(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        # Sentences are 5 tokens each; overlap budget=8 → at most ONE
        # trailing sentence carries over.
        chunker = TextChunker(token_limit=15, tolerance=2, overlap_tokens=8)
        seg = self._build_segment(filing_id)
        chunks = chunker.chunk_segments([seg])

        for prev, curr in pairwise(chunks):
            carried = curr.content.split(".", 1)[0] + "."
            # Only the last sentence from prev should appear at the head
            # of curr — not the penultimate one.
            prev_sentences = [s.strip() + "." for s in prev.content.split(".") if s.strip()]
            assert carried.strip() == prev_sentences[-1]

    def test_overlap_skips_sentence_exceeding_budget(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        # One sentence alone exceeds the overlap budget — the chunker
        # must fall back rather than emitting a bloated next chunk.
        long_sentence = "word " * 40 + "tail."
        short_sentences = " ".join([f"Short sentence number {i}." for i in range(10)])
        seg = _make_segment(long_sentence + " " + short_sentences, filing_id)
        chunker = TextChunker(token_limit=30, tolerance=5, overlap_tokens=10)

        chunks = chunker.chunk_segments([seg])

        # Chunker must still make forward progress and respect the token
        # limit band for the short-sentence tail.
        assert len(chunks) >= 2
        for chunk in chunks[1:]:
            assert chunk.token_count <= chunker.token_limit + chunker.tolerance
