"""Retrieval-quality integration tests.

Ingests a small, topic-separated corpus through the real
:class:`FilingStore` into on-disk ChromaDB, then drives the real
:class:`RetrievalService` against it.  Because the
:class:`KeywordEmbedder` is deterministic and term-frequency based,
ranking is reproducible without a fixed random seed: a query dense in a
chunk's vocabulary ranks that chunk first.

Covers semantic ranking, metadata filters (ticker / form type / date
range) end-to-end against the real vector store, the ``min_similarity``
floor, and an aggregate precision@k / recall@k pass through the shipped
:func:`evaluate_retrieval` scorer.
"""

from __future__ import annotations

from datetime import date

import pytest

from sec_generative_search.database import FilingStore
from sec_generative_search.search.evaluation import EvaluationCase, evaluate_retrieval
from sec_generative_search.search.retrieval import RetrievalService

from .conftest import KeywordEmbedder, build_processed_filing, make_filing_id

pytestmark = pytest.mark.integration


# Deterministic chunk_ids (TICKER_FORM_DATE_INDEX) for the corpus below.
AAPL_RISK = "AAPL_10-K_2023-11-03_000"
AAPL_MDNA = "AAPL_10-K_2023-11-03_001"
AAPL_LIQUIDITY = "AAPL_10-K_2023-11-03_002"
MSFT_CYBER = "MSFT_10-Q_2022-07-27_000"


@pytest.fixture
def corpus(store: FilingStore, embedder: KeywordEmbedder) -> FilingStore:
    """Ingest a four-chunk, two-filing corpus with separated topics."""
    aapl = build_processed_filing(
        make_filing_id(),
        [
            (
                "Part I > Item 1A > Risk Factors",
                "Litigation and a lawsuit pose legal risk amid competition.",
            ),
            (
                "Part II > Item 7 > MD&A",
                "Revenue growth was strong and operating margin expanded.",
            ),
            (
                "Part II > Item 7 > Liquidity",
                "The board raised the dividend and repaid debt.",
            ),
        ],
        embedder,
    )
    msft = build_processed_filing(
        make_filing_id(
            ticker="MSFT",
            form_type="10-Q",
            filing_date=date(2022, 7, 27),
            accession_number="0000789019-22-000077",
        ),
        [("Part I > Item 1A > Risk Factors", "Cybersecurity risk threatens the supply chain.")],
        embedder,
    )
    store.store_filing(aapl, register_if_new=True)
    store.store_filing(msft, register_if_new=True)
    return store


class TestSemanticRanking:
    def test_topical_query_ranks_relevant_chunk_first(
        self, corpus: FilingStore, retrieval: RetrievalService
    ) -> None:
        results = retrieval.retrieve("revenue growth and operating margin", top_k=4)
        assert results, "expected at least one hit"
        assert results[0].chunk_id == AAPL_MDNA
        # The revenue chunk must out-rank the litigation chunk.
        ids = [r.chunk_id for r in results]
        assert ids.index(AAPL_MDNA) < ids.index(AAPL_RISK)

    def test_distinct_topic_query_ranks_its_own_chunk_first(
        self, corpus: FilingStore, retrieval: RetrievalService
    ) -> None:
        results = retrieval.retrieve("dividend and debt repayment", top_k=4)
        assert results[0].chunk_id == AAPL_LIQUIDITY


class TestMetadataFilters:
    def test_ticker_filter_scopes_results(
        self, corpus: FilingStore, retrieval: RetrievalService
    ) -> None:
        results = retrieval.retrieve("risk", top_k=10, ticker="MSFT")
        assert results
        assert {r.ticker for r in results} == {"MSFT"}
        assert all(r.chunk_id == MSFT_CYBER for r in results)

    def test_form_type_filter_scopes_results(
        self, corpus: FilingStore, retrieval: RetrievalService
    ) -> None:
        results = retrieval.retrieve("risk", top_k=10, form_type="10-Q")
        assert results
        assert {r.form_type for r in results} == {"10-Q"}

    def test_date_range_filter_scopes_results(
        self, corpus: FilingStore, retrieval: RetrievalService
    ) -> None:
        # Only the AAPL filing falls on/after 2023-01-01.
        results = retrieval.retrieve("risk", top_k=10, start_date="2023-01-01")
        assert results
        assert {r.ticker for r in results} == {"AAPL"}
        assert MSFT_CYBER not in {r.chunk_id for r in results}


class TestSimilarityFloor:
    def test_min_similarity_drops_unrelated_chunks(
        self, corpus: FilingStore, retrieval: RetrievalService
    ) -> None:
        # "acquisition tax currency" shares no vocabulary with any chunk,
        # so every cosine similarity is ~0 and a high floor returns none.
        results = retrieval.retrieve("acquisition tax currency", top_k=10, min_similarity=0.5)
        assert results == []


class TestPrecisionRecall:
    def test_aggregate_precision_recall_is_perfect_on_separated_corpus(
        self, corpus: FilingStore, retrieval: RetrievalService
    ) -> None:
        cases = [
            EvaluationCase(
                case_id="revenue",
                query="revenue growth and operating margin",
                expected_chunk_ids=(AAPL_MDNA,),
            ),
            EvaluationCase(
                case_id="liquidity",
                query="dividend and debt repayment",
                expected_chunk_ids=(AAPL_LIQUIDITY,),
            ),
            EvaluationCase(
                case_id="cyber",
                query="cybersecurity supply chain",
                expected_chunk_ids=(MSFT_CYBER,),
            ),
        ]
        report = evaluate_retrieval(
            cases,
            lambda query, k: retrieval.retrieve(query, top_k=k),
            top_k=1,
        )
        # top_k=1 and one expected chunk per case → a perfect retriever
        # scores 1.0 on both axes.
        assert report.recall_at_k == pytest.approx(1.0)
        assert report.precision_at_k == pytest.approx(1.0)
