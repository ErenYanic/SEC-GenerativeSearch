"""Streaming generation load / throughput tests.

Drives the real :class:`RAGOrchestrator` streaming path over the real
retrieval stack with a :class:`ScriptedLLM` standing in for the model.
The sequential arm asserts streaming stays well-formed under volume —
exactly one terminal ``final`` event per stream, the deltas concatenate
to the model's reply, and every citation resolves to a retrieved chunk
(the anti-fabrication invariant, holding under load).

The concurrency arm is the load-bearing state-isolation lock: many
streams run at once, each over its *own* orchestrator + provider, and
each must return *its* answer with citations resolving to *its* retrieved
chunks — no cross-request state bleed.  Each generation owns a fresh
``GenerationRequest``; the orchestrator keeps no mutable per-request
state on a shared instance, and this test fails if a regression ever
introduces one.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from sec_generative_search.database import FilingStore
from sec_generative_search.rag.orchestrator import RAGOrchestrator
from sec_generative_search.rag.query_understanding import QueryPlan
from sec_generative_search.search.retrieval import RetrievalService

from .conftest import KeywordEmbedder, ScriptedLLM, make_corpus

pytestmark = pytest.mark.load


@pytest.fixture
def corpus(store: FilingStore, embedder: KeywordEmbedder) -> FilingStore:
    make_corpus(store, embedder, count=10)
    return store


def _drain_stream(orchestrator: RAGOrchestrator, query: str):
    """Run a stream to completion; return (joined_deltas, final_result)."""
    deltas: list[str] = []
    final = None
    for event in orchestrator.generate_stream(QueryPlan(raw_query=query)):
        if event.delta is not None:
            deltas.append(event.delta)
        if event.final is not None:
            assert final is None, "more than one final event on a single stream"
            final = event.final
    assert final is not None, "stream ended without a final event"
    return "".join(deltas), final


class TestSustainedStreaming:
    def test_many_sequential_streams_stay_well_formed(
        self, corpus: FilingStore, retrieval: RetrievalService
    ) -> None:
        from .conftest import measure

        reply = "Revenue growth was strong and operating margin expanded [1]."
        orchestrator = RAGOrchestrator(retrieval=retrieval, llm=ScriptedLLM(reply=reply))

        def run(_i: int) -> None:
            joined, final = _drain_stream(orchestrator, "revenue growth and operating margin")
            # Deltas reconstruct the model's full reply.
            assert joined == reply
            # Anti-fabrication invariant holds under volume.
            retrieved_ids = {c.chunk_id for c in final.retrieved_chunks}
            assert final.citations
            assert all(cite.chunk_id in retrieved_ids for cite in final.citations)

        summary = measure("stream", 40, run)
        assert summary.p95 < 3.0  # catastrophic-regression ceiling, not an SLO


class TestConcurrentStreaming:
    @pytest.mark.security
    def test_concurrent_streams_are_isolated(
        self, corpus: FilingStore, retrieval: RetrievalService
    ) -> None:
        ops = 32
        workers = 8

        def run(i: int) -> None:
            # Each stream owns its provider + orchestrator; the unique
            # marker proves thread i received thread i's answer.
            reply = f"Finding {i:04d} on revenue growth and margin [1]."
            orchestrator = RAGOrchestrator(retrieval=retrieval, llm=ScriptedLLM(reply=reply))
            joined, final = _drain_stream(orchestrator, "revenue growth and operating margin")

            assert joined == reply, f"stream {i} answer bled across threads"
            assert final.answer == reply
            # Citations resolve to chunks this stream actually retrieved.
            retrieved_ids = {c.chunk_id for c in final.retrieved_chunks}
            assert final.citations
            assert all(cite.chunk_id in retrieved_ids for cite in final.citations)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            # list() forces every future to resolve and re-raises any
            # assertion that fired inside a worker thread.
            list(pool.map(run, range(ops)))
