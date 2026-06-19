"""Retrieval load / throughput tests.

Populates the real vector store with a multi-filing corpus, then drives
the real :class:`RetrievalService` under sustained sequential volume and
under concurrent fan-out.  Because :class:`KeywordEmbedder` is
deterministic and term-frequency based, the *section* the top hit belongs
to is reproducible without a random seed — every filing carries the same
six sections, so a topical query ranks the matching section's chunk first
regardless of which filing wins the tie.  We assert on the section index
suffix of ``chunk_id`` (``TICKER_FORM_DATE_INDEX``), which is stable under
tie-breaking.

The concurrency arm is the load-bearing read-consistency lock: a query
fanned out across threads must return the same top section as the serial
baseline — no torn reads against the shared :class:`ChromaDBClient`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from sec_generative_search.database import FilingStore
from sec_generative_search.search.retrieval import RetrievalService

from .conftest import KeywordEmbedder, make_corpus, measure

pytestmark = pytest.mark.load

# query → expected section-index suffix of the top hit's chunk_id.
# Indices map to _SECTION_TEMPLATE in conftest.py:
#   000 Risk Factors · 001 Cybersecurity · 002 MD&A · 003 Liquidity
_QUERY_SET: tuple[tuple[str, str], ...] = (
    ("revenue growth and operating margin", "_002"),
    ("dividend and debt repayment", "_003"),
    ("cybersecurity supply chain risk", "_001"),
    ("litigation lawsuit competition", "_000"),
)


@pytest.fixture
def corpus(store: FilingStore, embedder: KeywordEmbedder) -> FilingStore:
    make_corpus(store, embedder, count=20)
    return store


class TestSustainedRetrieval:
    def test_top_section_is_stable_across_many_queries(
        self, corpus: FilingStore, retrieval: RetrievalService
    ) -> None:
        # Warm up (first query loads the index) so timing is representative.
        retrieval.retrieve(_QUERY_SET[0][0], top_k=5)

        iterations = 300

        def run(i: int) -> None:
            query, suffix = _QUERY_SET[i % len(_QUERY_SET)]
            results = retrieval.retrieve(query, top_k=5)
            assert results, f"query {query!r} returned nothing"
            assert results[0].chunk_id.endswith(suffix), (
                f"query {query!r} top hit {results[0].chunk_id} drifted from {suffix}"
            )

        summary = measure("retrieve", iterations, run)
        # Catastrophic-regression ceiling, not an SLO (corpus is tiny and
        # in-memory).  Anything near this means something broke.
        assert summary.p95 < 2.0


class TestConcurrentRetrieval:
    @pytest.mark.security
    def test_concurrent_queries_match_serial_baseline(
        self, corpus: FilingStore, retrieval: RetrievalService
    ) -> None:
        # Serial baseline: the top section each query resolves to.
        baseline = {
            query: retrieval.retrieve(query, top_k=5)[0].chunk_id[-4:] for query, _ in _QUERY_SET
        }

        ops = 200
        workers = 8
        queries = [_QUERY_SET[i % len(_QUERY_SET)][0] for i in range(ops)]

        def run(query: str) -> tuple[str, str]:
            results = retrieval.retrieve(query, top_k=5)
            return query, results[0].chunk_id[-4:]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            observed = list(pool.map(run, queries))

        # Every concurrent observation matches the serial baseline — the
        # shared client served consistent reads under contention.
        for query, suffix in observed:
            assert suffix == baseline[query], (
                f"concurrent read for {query!r} ({suffix}) "
                f"diverged from baseline ({baseline[query]})"
            )
