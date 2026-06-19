"""Ingestion load / throughput tests.

Drives the real dual-store :class:`FilingStore` (on-disk
:class:`ChromaDBClient` + :class:`MetadataRegistry`) under volume and
under write contention. The load-bearing assertion is the cross-store
invariant the unit suites can only check in isolation: under load the
chunk count ChromaDB holds still equals the count SQLite registers, and
no filing is lost or orphaned.

The meaningful write-concurrency property is the atomic
``register_if_new`` duplicate claim, where :class:`FilingStore`
serialises the ChromaDB write to one winner. We do not fan out
concurrent distinct-accession writes against a single ``PersistentClient``
because that is outside the supported single-writer model.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from sec_generative_search.database import ChromaDBClient, FilingStore, MetadataRegistry

from .conftest import KeywordEmbedder, make_corpus, make_filing

pytestmark = pytest.mark.load


class TestHighVolumeIngestion:
    def test_cross_store_invariant_holds_under_volume(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        embedder: KeywordEmbedder,
    ) -> None:
        filing_count = 40

        total_chunks = make_corpus(store, embedder, count=filing_count)

        # Every filing registered exactly once …
        assert registry.count() == filing_count
        # … and the two stores agree on the total chunk population.
        assert chroma.collection_count() == total_chunks
        # No orphaned chunks: summing each registered filing's chunk_count
        # reproduces the ChromaDB population exactly.
        per_filing = [registry.get_filing(f"0000320193-23-{i:06d}") for i in range(filing_count)]
        assert all(record is not None for record in per_filing)
        assert sum(record.chunk_count for record in per_filing if record) == total_chunks

    def test_ingestion_throughput_is_reported(
        self, store: FilingStore, embedder: KeywordEmbedder
    ) -> None:
        # Time per-filing ingestion (build + dual-store write).  This is a
        # throughput smoke, not an SLO: the ceiling is ~100x the expected
        # in-memory cost and only catches a catastrophic regression.
        from .conftest import measure

        def ingest(i: int) -> None:
            filing = make_filing(embedder, index=i)
            assert store.store_filing(filing, register_if_new=True) is True

        summary = measure("ingest filing", 30, ingest)
        assert summary.p95 < 2.0, "per-filing ingestion regressed catastrophically"


class TestConcurrentDuplicateClaim:
    @pytest.mark.security
    def test_concurrent_same_accession_claims_exactly_once(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        embedder: KeywordEmbedder,
    ) -> None:
        # Many threads race to ingest the SAME accession with differing
        # content via the atomic check-and-claim path.  Exactly one may
        # win; the store must not double-register or orphan chunks.
        workers = 16
        # Distinct ProcessedFiling objects (different chunk counts) sharing
        # one accession — proves the claim keys on accession, not identity.
        filings = [make_filing(embedder, index=0, sections=(idx % 6) + 1) for idx in range(workers)]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda f: store.store_filing(f, register_if_new=True), filings))

        # Exactly one thread stored; every other saw the claim and skipped.
        assert results.count(True) == 1
        assert results.count(False) == workers - 1

        # The store reflects a single filing with a single coherent chunk
        # set — no partial/duplicate writes from the losing threads.
        assert registry.count() == 1
        record = registry.get_filing("0000320193-23-000000")
        assert record is not None
        assert chroma.collection_count() == record.chunk_count
