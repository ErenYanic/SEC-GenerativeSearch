"""Tests for :mod:`sec_generative_search.rag.citations`."""

from __future__ import annotations

import json

import pytest

from sec_generative_search.rag.citations import (
    extract_citations,
    extract_from_inline_markers,
    extract_from_json_envelope,
)


class TestInlineMarkerExtraction:
    def test_picks_up_simple_markers(self, sample_chunks) -> None:
        text = "Revenue grew [1] and margin compressed [2]."
        result = extract_from_inline_markers(text, sample_chunks)
        assert [c.chunk_id for c in result.citations] == [
            sample_chunks[0].chunk_id,
            sample_chunks[1].chunk_id,
        ]
        # Display indices are 1-based in mention order.
        assert [c.display_index for c in result.citations] == [1, 2]
        # The answer text is preserved verbatim — UI needs the markers.
        assert result.answer == text

    def test_first_mention_order_for_duplicates(self, sample_chunks) -> None:
        text = "X [2]. Y [2]. Z [1]."
        result = extract_from_inline_markers(text, sample_chunks)
        # First mention of [2] then [1].
        assert [c.chunk_id for c in result.citations] == [
            sample_chunks[1].chunk_id,
            sample_chunks[0].chunk_id,
        ]

    @pytest.mark.security
    def test_drops_out_of_range_markers(self, sample_chunks) -> None:
        """A model that fabricates [99] must not crash or produce a phantom citation."""
        text = "Real [1]. Fabricated [99]. Real [2]."
        result = extract_from_inline_markers(text, sample_chunks)
        assert len(result.citations) == 2
        valid_ids = {sample_chunks[0].chunk_id, sample_chunks[1].chunk_id}
        assert all(c.chunk_id in valid_ids for c in result.citations)

    def test_no_markers_yields_no_citations(self, sample_chunks) -> None:
        result = extract_from_inline_markers("Plain answer with no markers.", sample_chunks)
        assert result.citations == []
        assert result.answer == "Plain answer with no markers."

    def test_empty_chunks_returns_empty(self) -> None:
        result = extract_from_inline_markers("Answer [1].", [])
        assert result.citations == []


class TestJsonEnvelopeExtraction:
    def test_parses_minimal_envelope(self, sample_chunks) -> None:
        payload = json.dumps(
            {
                "answer": "Revenue grew.",
                "cited_chunk_ids": [sample_chunks[0].chunk_id],
            }
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert result.answer == "Revenue grew."
        assert [c.chunk_id for c in result.citations] == [sample_chunks[0].chunk_id]

    def test_strips_markdown_fence(self, sample_chunks) -> None:
        payload = (
            "```json\n"
            + json.dumps(
                {
                    "answer": "Fenced.",
                    "cited_chunk_ids": [sample_chunks[1].chunk_id],
                }
            )
            + "\n```"
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert result.answer == "Fenced."
        assert result.citations[0].chunk_id == sample_chunks[1].chunk_id

    def test_finds_object_amid_prose(self, sample_chunks) -> None:
        payload = (
            "Sure! Here is the JSON: "
            + json.dumps({"answer": "X.", "cited_chunk_ids": [sample_chunks[0].chunk_id]})
            + " Hope that helps."
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert result.answer == "X."

    @pytest.mark.security
    def test_drops_unknown_chunk_ids(self, sample_chunks) -> None:
        """Model fabrication of a chunk_id must not produce a phantom citation."""
        payload = json.dumps(
            {
                "answer": "Answer.",
                "cited_chunk_ids": [
                    sample_chunks[0].chunk_id,
                    "FABRICATED_CHUNK_ID",
                ],
            }
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == sample_chunks[0].chunk_id

    def test_raises_on_missing_answer(self, sample_chunks) -> None:
        from sec_generative_search.core.exceptions import CitationError

        payload = json.dumps({"cited_chunk_ids": []})
        with pytest.raises(CitationError):
            extract_from_json_envelope(payload, sample_chunks)

    def test_raises_on_no_object(self, sample_chunks) -> None:
        from sec_generative_search.core.exceptions import CitationError

        with pytest.raises(CitationError):
            extract_from_json_envelope("plain text no json", sample_chunks)


class TestHybridDispatcher:
    def test_prefers_json_when_flag_set(self, sample_chunks) -> None:
        payload = json.dumps(
            {"answer": "JSON path.", "cited_chunk_ids": [sample_chunks[0].chunk_id]}
        )
        result = extract_citations(payload, sample_chunks, prefer_json=True)
        assert result.answer == "JSON path."
        assert result.citations[0].chunk_id == sample_chunks[0].chunk_id

    def test_falls_back_to_inline_when_json_path_set_but_payload_invalid(
        self, sample_chunks
    ) -> None:
        """JSON parse failure must not blank out citations — try inline markers."""
        payload = "Not JSON, just text [1]."
        result = extract_citations(payload, sample_chunks, prefer_json=True)
        assert result.answer == payload
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == sample_chunks[0].chunk_id

    def test_uses_inline_when_flag_unset(self, sample_chunks) -> None:
        payload = "Inline [1] [2]."
        result = extract_citations(payload, sample_chunks, prefer_json=False)
        assert len(result.citations) == 2
