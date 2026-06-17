"""End-to-end ingestion integration tests.

Exercises the full ingestion path with real components:

* the real :class:`FilingParser` (over a controlled ``html2dict``
  fixture — doc2dict is the external HTML-parsing boundary),
* the real :class:`TextChunker`,
* the deterministic :class:`KeywordEmbedder`,
* the real :class:`PipelineOrchestrator`,
* and the real dual-store :class:`FilingStore` over on-disk
  :class:`ChromaDBClient` + :class:`MetadataRegistry`.

Asserts the cross-store invariant the unit suites can only check in
isolation: the chunk count ChromaDB holds equals the count SQLite
registers, and the atomic ``register_if_new`` path skips a duplicate
accession across *both* stores rather than orphaning chunks.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.database import ChromaDBClient, FilingStore, MetadataRegistry
from sec_generative_search.pipeline import parse as parse_module
from sec_generative_search.pipeline.orchestrator import PipelineOrchestrator

from .conftest import KeywordEmbedder, build_processed_filing, make_filing_id

pytestmark = pytest.mark.integration


# A realistic multi-section doc2dict tree.  Section text is written with
# the embedder vocabulary so the stored vectors are non-degenerate.
_FILING_TREE: dict[str, Any] = {
    "document": {
        "section_0": {
            "title": "Part I",
            "contents": {
                "item_1a": {
                    "title": "Item 1A. Risk Factors",
                    "text": (
                        "Our revenue is exposed to competition and supply chain "
                        "risk. A material lawsuit or litigation could harm growth."
                    ),
                    "contents": {
                        "cyber": {
                            "title": "Cybersecurity",
                            "textsmall": "Cybersecurity risk remains elevated.",
                        }
                    },
                }
            },
        },
        "section_1": {
            "title": "Part II",
            "contents": {
                "item_7": {
                    "title": "Item 7. MD&A",
                    "text": (
                        "Revenue growth accelerated while operating margin "
                        "compressed. The board approved a dividend and reduced debt."
                    ),
                }
            },
        },
    }
}


def _ingest_full_pipeline(
    store: FilingStore,
    embedder: KeywordEmbedder,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, str]:
    """Run fetch-less full pipeline and persist; return (chunk_count, accession)."""
    monkeypatch.setattr(parse_module, "html2dict", lambda _html: _FILING_TREE)

    orchestrator = PipelineOrchestrator(embedder=embedder)
    filing_id = make_filing_id()
    processed = orchestrator.process_filing(filing_id, "<html>ignored</html>")
    store.store_filing(processed)
    return processed.ingest_result.chunk_count, filing_id.accession_number


class TestFullPipelineIngestion:
    def test_chunk_counts_agree_across_both_stores(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        embedder: KeywordEmbedder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chunk_count, accession = _ingest_full_pipeline(store, embedder, monkeypatch)

        # The real parser produced at least the three section segments.
        assert chunk_count >= 3

        # ChromaDB holds exactly the chunks the pipeline produced …
        assert chroma.collection_count() == chunk_count
        # … and SQLite registered the same count for the filing.
        assert registry.count() == 1
        record = registry.get_filing(accession)
        assert record is not None
        assert record.chunk_count == chunk_count
        assert record.ticker == "AAPL"
        assert record.form_type == "10-K"

    def test_embedder_was_actually_invoked(
        self,
        store: FilingStore,
        embedder: KeywordEmbedder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Guards against a regression where the orchestrator silently
        # skips embedding and the storage layer accepts vectorless chunks.
        _ingest_full_pipeline(store, embedder, monkeypatch)
        assert embedder.embed_calls >= 1


class TestEmbedderlessPipelineRejected:
    def test_storage_refuses_none_embeddings(
        self,
        store: FilingStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # An orchestrator with no embedder yields embeddings=None; the
        # storage layer MUST refuse rather than write vectorless chunks.
        monkeypatch.setattr(parse_module, "html2dict", lambda _html: _FILING_TREE)
        orchestrator = PipelineOrchestrator(embedder=None)
        processed = orchestrator.process_filing(make_filing_id(), "<html>x</html>")

        with pytest.raises(DatabaseError, match="embeddings is None"):
            store.store_filing(processed)


class TestAtomicDuplicateHandling:
    def test_duplicate_accession_skipped_without_orphaning(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        embedder: KeywordEmbedder,
    ) -> None:
        filing_id = make_filing_id()
        first = build_processed_filing(
            filing_id,
            [
                ("Part I > Item 1A", "Revenue growth and competition risk."),
                ("Part II > Item 7", "Margin and dividend and debt."),
            ],
            embedder,
        )
        assert store.store_filing(first, register_if_new=True) is True
        first_count = chroma.collection_count()

        # Re-ingest the SAME accession with DIFFERENT chunk content.
        second = build_processed_filing(
            filing_id,
            [("Part I > Item 1A", "Entirely different revenue text.")],
            embedder,
        )
        assert store.store_filing(second, register_if_new=True) is False

        # The duplicate was a no-op: no new chunks landed in ChromaDB and
        # the registry still reflects the FIRST ingest's chunk count.
        assert chroma.collection_count() == first_count
        assert registry.count() == 1
        record = registry.get_filing(filing_id.accession_number)
        assert record is not None
        assert record.chunk_count == first.ingest_result.chunk_count

    def test_distinct_accessions_both_persist(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        embedder: KeywordEmbedder,
    ) -> None:
        aapl = build_processed_filing(
            make_filing_id(),
            [("Part I > Item 1A", "Revenue growth risk.")],
            embedder,
        )
        msft = build_processed_filing(
            make_filing_id(
                ticker="MSFT",
                filing_date=date(2023, 7, 27),
                accession_number="0000789019-23-000077",
            ),
            [("Part I > Item 1A", "Cloud revenue and margin.")],
            embedder,
        )
        assert store.store_filing(aapl, register_if_new=True) is True
        assert store.store_filing(msft, register_if_new=True) is True

        assert registry.count() == 2
        assert chroma.collection_count() == 2
