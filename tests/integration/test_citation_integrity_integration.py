"""Citation-integrity integration tests.

Drives the real :class:`RAGOrchestrator` over the real retrieval stack
(on-disk ChromaDB + :class:`KeywordEmbedder`) with a :class:`ScriptedLLM`
standing in for the model.  These tests assert the load-bearing
anti-fabrication invariant end-to-end: every citation a
:class:`GenerationResult` carries must resolve to a chunk that was
actually retrieved across the inline ``[N]`` path, the JSON-envelope
path, and the streaming surface.
"""

from __future__ import annotations

import json

import pytest

from sec_generative_search.database import FilingStore
from sec_generative_search.rag.orchestrator import RAGOrchestrator
from sec_generative_search.rag.query_understanding import QueryPlan

from .conftest import KeywordEmbedder, ScriptedLLM, build_processed_filing, make_filing_id

pytestmark = pytest.mark.integration

AAPL_MDNA = "AAPL_10-K_2023-11-03_001"
FABRICATED_ID = "ZZZZ_10-K_1900-01-01_999"


@pytest.fixture
def corpus(store: FilingStore, embedder: KeywordEmbedder) -> FilingStore:
    aapl = build_processed_filing(
        make_filing_id(),
        [
            ("Part I > Item 1A > Risk Factors", "Litigation and lawsuit risk from competition."),
            ("Part II > Item 7 > MD&A", "Revenue growth was strong and margin expanded."),
            ("Part II > Item 7 > Liquidity", "The board raised the dividend and cut debt."),
        ],
        embedder,
    )
    store.store_filing(aapl, register_if_new=True)
    return store


def _orchestrator(retrieval, llm: ScriptedLLM) -> RAGOrchestrator:
    return RAGOrchestrator(retrieval=retrieval, llm=llm)


class TestInlineCitationPath:
    def test_inline_marker_resolves_to_retrieved_chunk(
        self, corpus: FilingStore, retrieval
    ) -> None:
        llm = ScriptedLLM(reply="Revenue growth was strong [1].")
        result = _orchestrator(retrieval, llm).generate(
            QueryPlan(raw_query="How did revenue grow?")
        )

        assert result.citations, "an inline [1] marker should yield a citation"
        retrieved_ids = {c.chunk_id for c in result.retrieved_chunks}
        # The cited chunk_id must be one that was actually retrieved.
        assert all(cite.chunk_id in retrieved_ids for cite in result.citations)
        # [1] is 1-based into the retrieved list.
        assert result.citations[0].chunk_id == result.retrieved_chunks[0].chunk_id

    @pytest.mark.security
    def test_out_of_range_inline_marker_is_dropped(self, corpus: FilingStore, retrieval) -> None:
        # The model emits a marker past the end of the retrieved set.
        llm = ScriptedLLM(reply="According to the filing [99], revenue fell.")
        result = _orchestrator(retrieval, llm).generate(
            QueryPlan(raw_query="How did revenue grow?")
        )

        # No fabricated citation may survive; the orchestrator drops the
        # unresolved marker rather than inventing a source.
        retrieved_ids = {c.chunk_id for c in result.retrieved_chunks}
        assert all(cite.chunk_id in retrieved_ids for cite in result.citations)
        assert result.citations == []


class TestJsonEnvelopePath:
    @pytest.mark.security
    def test_fabricated_chunk_id_dropped_real_kept(self, corpus: FilingStore, retrieval) -> None:
        # Structured-output model cites one real chunk_id and one
        # fabricated id that was never retrieved.
        payload = json.dumps(
            {
                "answer": "Revenue growth was strong.",
                "cited_chunk_ids": [AAPL_MDNA, FABRICATED_ID],
            }
        )
        llm = ScriptedLLM(reply=payload, structured_output=True)
        result = _orchestrator(retrieval, llm).generate(
            QueryPlan(raw_query="revenue growth and operating margin"),
            prefer_structured_output=True,
        )

        retrieved_ids = {c.chunk_id for c in result.retrieved_chunks}
        cited_ids = {cite.chunk_id for cite in result.citations}

        assert AAPL_MDNA in retrieved_ids, "test corpus must retrieve the MD&A chunk"
        assert FABRICATED_ID not in cited_ids, "fabricated id must never survive"
        assert cited_ids == {AAPL_MDNA}
        assert all(cite.chunk_id in retrieved_ids for cite in result.citations)


class TestRefusalShortCircuit:
    def test_empty_retrieval_refuses_without_calling_the_model(
        self, corpus: FilingStore, retrieval
    ) -> None:
        llm = ScriptedLLM(reply="This should never be returned.")
        # A ticker that matches nothing in the corpus → empty retrieval.
        result = _orchestrator(retrieval, llm).generate(
            QueryPlan(raw_query="revenue", tickers=["NOSUCHTICKER"])
        )

        assert result.retrieved_chunks == []
        assert result.citations == []
        # The refusal path must NOT call the LLM (no token spend, no leak).
        assert llm.last_request is None


class TestStreamingCitationIntegrity:
    @pytest.mark.security
    def test_streamed_final_citations_resolve_to_retrieved_chunks(
        self, corpus: FilingStore, retrieval
    ) -> None:
        llm = ScriptedLLM(reply="Margin expanded [1] and the dividend rose [2].")
        events = list(
            _orchestrator(retrieval, llm).generate_stream(
                QueryPlan(raw_query="margin and dividend")
            )
        )

        finals = [e.final for e in events if e.final is not None]
        assert len(finals) == 1
        final = finals[0]

        retrieved_ids = {c.chunk_id for c in final.retrieved_chunks}
        assert final.citations, "expected at least one citation on the stream"
        assert all(cite.chunk_id in retrieved_ids for cite in final.citations)
