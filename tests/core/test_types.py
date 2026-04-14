"""Tests for core domain types (Phase 1.6)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from sec_generative_search.core.types import (
    Chunk,
    ContentType,
    FilingIdentifier,
    IngestResult,
    SearchResult,
    Segment,
)


@pytest.fixture
def filing_id() -> FilingIdentifier:
    return FilingIdentifier(
        ticker="aapl",
        form_type="10-k",
        filing_date=date(2023, 11, 3),
        accession_number="0000320193-23-000077",
    )


class TestFilingIdentifier:
    def test_auto_uppercases_ticker_and_form(self, filing_id: FilingIdentifier) -> None:
        assert filing_id.ticker == "AAPL"
        assert filing_id.form_type == "10-K"

    def test_frozen(self, filing_id: FilingIdentifier) -> None:
        with pytest.raises(FrozenInstanceError):
            filing_id.ticker = "MSFT"  # type: ignore[misc]

    def test_date_str_iso_format(self, filing_id: FilingIdentifier) -> None:
        assert filing_id.date_str == "2023-11-03"

    def test_equality_matches_field_tuple(self) -> None:
        a = FilingIdentifier("AAPL", "10-K", date(2023, 1, 1), "acc-1")
        b = FilingIdentifier("aapl", "10-k", date(2023, 1, 1), "acc-1")
        assert a == b
        assert hash(a) == hash(b)


class TestChunk:
    def test_chunk_id_format(self, filing_id: FilingIdentifier) -> None:
        chunk = Chunk(
            content="Our business is subject to...",
            path="Part I > Item 1A > Risk Factors",
            content_type=ContentType.TEXT,
            filing_id=filing_id,
            chunk_index=42,
        )
        assert chunk.chunk_id == "AAPL_10-K_2023-11-03_042"

    def test_to_metadata(self, filing_id: FilingIdentifier) -> None:
        chunk = Chunk(
            content="text",
            path="Part I > Item 1",
            content_type=ContentType.TABLE,
            filing_id=filing_id,
            chunk_index=0,
        )
        md = chunk.to_metadata()
        assert md["ticker"] == "AAPL"
        assert md["form_type"] == "10-K"
        assert md["filing_date"] == "2023-11-03"
        # Range-query-friendly integer form of the date.
        assert md["filing_date_int"] == 20231103
        assert md["content_type"] == "table"
        assert md["accession_number"] == "0000320193-23-000077"


class TestSegment:
    def test_roundtrip(self, filing_id: FilingIdentifier) -> None:
        seg = Segment(
            path="Part I > Item 1A",
            content_type=ContentType.TEXT,
            content="risk factors content",
            filing_id=filing_id,
        )
        assert seg.filing_id is filing_id
        assert seg.content_type is ContentType.TEXT


class TestSearchResult:
    def test_from_chromadb_result_converts_distance_to_similarity(self) -> None:
        result = SearchResult.from_chromadb_result(
            document="revenue grew 5%",
            metadata={
                "path": "Part II > Item 7 > MD&A",
                "content_type": "text",
                "ticker": "MSFT",
                "form_type": "10-K",
                "filing_date": "2023-07-27",
                "accession_number": "acc-1",
            },
            distance=0.2,
            chunk_id="MSFT_10-K_2023-07-27_001",
        )
        assert result.ticker == "MSFT"
        assert result.similarity == pytest.approx(0.8)
        assert result.chunk_id == "MSFT_10-K_2023-07-27_001"
        assert result.content_type is ContentType.TEXT


class TestIngestResult:
    def test_basic_fields(self, filing_id: FilingIdentifier) -> None:
        r = IngestResult(
            filing_id=filing_id,
            segment_count=5,
            chunk_count=12,
            duration_seconds=1.23,
        )
        assert r.filing_id.ticker == "AAPL"
        assert r.chunk_count == 12
