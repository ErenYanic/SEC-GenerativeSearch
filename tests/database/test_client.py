"""Tests for :class:`sec_generative_search.database.ChromaDBClient`.

Covers the embedder-stamp contract, the
store-refuse-on-None-embeddings guarantee (orchestrator contract), and
smoke coverage of the read/write surface.

Tests drive a real :class:`chromadb.PersistentClient` under
``tmp_path`` — Chroma's sqlite backend is fast and reliable enough that
mocking it would obscure real contract breakage without a meaningful
speed gain.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import chromadb
import numpy as np
import pytest

from sec_generative_search.config.constants import COLLECTION_NAME
from sec_generative_search.core.exceptions import (
    DatabaseError,
    EmbeddingCollectionMismatchError,
)
from sec_generative_search.core.types import (
    Chunk,
    ContentType,
    EmbedderStamp,
    FilingIdentifier,
    IngestResult,
    SearchResult,
)
from sec_generative_search.database import ChromaDBClient
from sec_generative_search.pipeline.orchestrator import ProcessedFiling

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma_path(tmp_path: Path) -> str:
    """A per-test Chroma persistence directory."""
    return str(tmp_path / "chroma")


@pytest.fixture
def openai_stamp() -> EmbedderStamp:
    return EmbedderStamp(
        provider="openai",
        model="text-embedding-3-small",
        dimension=4,
    )


@pytest.fixture
def local_stamp() -> EmbedderStamp:
    return EmbedderStamp(
        provider="local",
        model="google/embeddinggemma-300m",
        dimension=4,
    )


def _make_filing_id(
    ticker: str = "AAPL",
    form_type: str = "10-K",
    filing_date: date | None = None,
    accession_number: str = "0000320193-23-000077",
) -> FilingIdentifier:
    return FilingIdentifier(
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date or date(2023, 11, 3),
        accession_number=accession_number,
    )


def _make_chunks(filing_id: FilingIdentifier, n: int = 2) -> list[Chunk]:
    return [
        Chunk(
            content=f"Chunk {i} for {filing_id.ticker}",
            path="Part I > Item 1 > Business",
            content_type=ContentType.TEXT,
            filing_id=filing_id,
            chunk_index=i,
            token_count=5,
        )
        for i in range(n)
    ]


def _make_processed_filing(
    stamp: EmbedderStamp,
    *,
    filing_id: FilingIdentifier | None = None,
    n_chunks: int = 2,
    with_embeddings: bool = True,
) -> ProcessedFiling:
    filing_id = filing_id or _make_filing_id()
    chunks = _make_chunks(filing_id, n=n_chunks)
    # Match the stamp's declared dimension so ChromaDB accepts the
    # first insert and pins the collection dimension.
    if with_embeddings:
        embeddings: np.ndarray | None = np.arange(
            n_chunks * stamp.dimension,
            dtype=np.float32,
        ).reshape(n_chunks, stamp.dimension)
    else:
        embeddings = None
    return ProcessedFiling(
        filing_id=filing_id,
        chunks=chunks,
        embeddings=embeddings,
        ingest_result=IngestResult(
            filing_id=filing_id,
            segment_count=n_chunks,
            chunk_count=n_chunks,
            duration_seconds=0.0,
        ),
    )


# ---------------------------------------------------------------------------
# Stamp seal + verification
# ---------------------------------------------------------------------------


class TestStampSealing:
    def test_fresh_collection_is_stamped(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        assert client.stamp == openai_stamp
        meta = client._collection.metadata or {}
        assert meta["embedding_provider"] == "openai"
        assert meta["embedding_model"] == "text-embedding-3-small"
        # Dimension is serialised as str in Chroma metadata.
        assert meta["embedding_dimension"] == "4"

    def test_reopen_with_matching_stamp_is_noop(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        # Second open must succeed against the same store.
        second = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        assert second.stamp == openai_stamp

    def test_reopen_with_different_provider_raises_mismatch(
        self,
        chroma_path: str,
        openai_stamp: EmbedderStamp,
        local_stamp: EmbedderStamp,
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        with pytest.raises(EmbeddingCollectionMismatchError) as excinfo:
            ChromaDBClient(local_stamp, chroma_path=chroma_path)

        err = excinfo.value
        assert err.expected == local_stamp
        assert err.actual == openai_stamp

    def test_reopen_with_different_model_raises_mismatch(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        other = EmbedderStamp(
            provider=openai_stamp.provider,
            model="text-embedding-3-large",
            dimension=openai_stamp.dimension,
        )
        with pytest.raises(EmbeddingCollectionMismatchError) as excinfo:
            ChromaDBClient(other, chroma_path=chroma_path)
        assert excinfo.value.actual.model == openai_stamp.model
        assert excinfo.value.expected.model == "text-embedding-3-large"

    def test_reopen_with_different_dimension_raises_mismatch(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        other = EmbedderStamp(
            provider=openai_stamp.provider,
            model=openai_stamp.model,
            dimension=openai_stamp.dimension + 1,
        )
        with pytest.raises(EmbeddingCollectionMismatchError) as excinfo:
            ChromaDBClient(other, chroma_path=chroma_path)
        assert excinfo.value.actual.dimension == openai_stamp.dimension

    def test_corrupt_stamp_raises_database_error(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        # Corrupt the dimension key directly on the collection metadata.
        raw = chromadb.PersistentClient(path=chroma_path)
        collection = raw.get_collection(name=COLLECTION_NAME)
        corrupt = {
            k: v for k, v in (collection.metadata or {}).items() if not k.startswith("hnsw:")
        }
        corrupt["embedding_dimension"] = "not-an-integer"
        collection.modify(metadata=corrupt)

        with pytest.raises(DatabaseError) as excinfo:
            ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        # Not a mismatch — category is distinct.
        assert not isinstance(excinfo.value, EmbeddingCollectionMismatchError)
        assert "corrupt" in str(excinfo.value).lower()

    def test_partial_stamp_raises_database_error(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        raw = chromadb.PersistentClient(path=chroma_path)
        collection = raw.get_collection(name=COLLECTION_NAME)
        # Remove the model key while keeping provider + dimension.
        partial = {
            k: v
            for k, v in (collection.metadata or {}).items()
            if not k.startswith("hnsw:") and k != "embedding_model"
        }
        collection.modify(metadata=partial)

        with pytest.raises(DatabaseError) as excinfo:
            ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        assert not isinstance(excinfo.value, EmbeddingCollectionMismatchError)

    def test_empty_unstamped_collection_is_stamped_on_open(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        # Pre-create an empty collection without a stamp (legacy path).
        raw = chromadb.PersistentClient(path=chroma_path)
        raw.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        meta = client._collection.metadata or {}
        assert meta["embedding_provider"] == "openai"
        assert meta["embedding_model"] == "text-embedding-3-small"
        assert meta["embedding_dimension"] == "4"

    def test_populated_unstamped_collection_is_refused(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        # Simulate a populated legacy collection with no stamp.
        raw = chromadb.PersistentClient(path=chroma_path)
        collection = raw.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            ids=["legacy-0"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            documents=["legacy chunk"],
            metadatas=[{"ticker": "AAPL"}],
        )

        with pytest.raises(DatabaseError) as excinfo:
            ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        assert not isinstance(excinfo.value, EmbeddingCollectionMismatchError)
        assert "reindex" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Write + read surface
# ---------------------------------------------------------------------------


class TestStoreAndQuery:
    def test_store_filing_persists_chunks(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, n_chunks=3)

        client.store_filing(pf)

        assert client.collection_count() == 3

    def test_store_filing_refuses_none_embeddings(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, with_embeddings=False)

        with pytest.raises(DatabaseError) as excinfo:
            client.store_filing(pf)
        assert "embeddings is none" in str(excinfo.value).lower()
        # No partial write — the collection stays empty.
        assert client.collection_count() == 0

    def test_query_returns_search_results(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, n_chunks=2)
        client.store_filing(pf)

        # Use the first stored embedding as the query to guarantee a hit.
        query = pf.embeddings[0].tolist()  # type: ignore[union-attr]
        results = client.query([query], n_results=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].ticker == "AAPL"
        assert results[0].form_type == "10-K"
        # Highest similarity first (cosine sim = 1 - distance).
        assert results[0].similarity >= results[1].similarity

    def test_query_filters_by_ticker(self, chroma_path: str, openai_stamp: EmbedderStamp) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        aapl = _make_processed_filing(
            openai_stamp,
            filing_id=_make_filing_id(ticker="AAPL", accession_number="0000320193-23-000077"),
            n_chunks=2,
        )
        msft = _make_processed_filing(
            openai_stamp,
            filing_id=_make_filing_id(ticker="MSFT", accession_number="0000789019-23-000001"),
            n_chunks=2,
        )
        client.store_filing(aapl)
        client.store_filing(msft)

        results = client.query(
            [aapl.embeddings[0].tolist()],  # type: ignore[union-attr]
            n_results=10,
            ticker="MSFT",
        )
        assert all(r.ticker == "MSFT" for r in results)

    def test_query_filters_by_date_range(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        old = _make_processed_filing(
            openai_stamp,
            filing_id=_make_filing_id(
                filing_date=date(2020, 1, 15),
                accession_number="0000320193-20-000001",
            ),
            n_chunks=1,
        )
        new = _make_processed_filing(
            openai_stamp,
            filing_id=_make_filing_id(
                filing_date=date(2024, 6, 1),
                accession_number="0000320193-24-000002",
            ),
            n_chunks=1,
        )
        client.store_filing(old)
        client.store_filing(new)

        results = client.query(
            [old.embeddings[0].tolist()],  # type: ignore[union-attr]
            n_results=10,
            start_date="2024-01-01",
        )
        assert len(results) == 1
        assert results[0].filing_date == "2024-06-01"

    def test_delete_filing_removes_all_chunks(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, n_chunks=3)
        client.store_filing(pf)

        assert client.collection_count() == 3
        client.delete_filing(pf.filing_id.accession_number)
        assert client.collection_count() == 0

    def test_delete_filings_batch_uses_single_call(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        a = _make_processed_filing(
            openai_stamp,
            filing_id=_make_filing_id(ticker="AAPL", accession_number="0000320193-23-000077"),
            n_chunks=2,
        )
        b = _make_processed_filing(
            openai_stamp,
            filing_id=_make_filing_id(ticker="MSFT", accession_number="0000789019-23-000001"),
            n_chunks=2,
        )
        client.store_filing(a)
        client.store_filing(b)
        assert client.collection_count() == 4

        client.delete_filings_batch([a.filing_id.accession_number, b.filing_id.accession_number])
        assert client.collection_count() == 0

    def test_delete_filings_batch_empty_list_is_noop(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        client.delete_filings_batch([])
        assert client.collection_count() == 0


class TestClearCollection:
    def test_clear_empty_returns_zero(self, chroma_path: str, openai_stamp: EmbedderStamp) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        assert client.clear_collection() == 0

    def test_clear_populated_rebuilds_and_reseals(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, n_chunks=3)
        client.store_filing(pf)
        assert client.collection_count() == 3

        removed = client.clear_collection()
        assert removed == 3
        assert client.collection_count() == 0

        meta = client._collection.metadata or {}
        assert meta["embedding_provider"] == openai_stamp.provider
        assert meta["embedding_model"] == openai_stamp.model
        assert meta["embedding_dimension"] == str(openai_stamp.dimension)

        # Reopening with the same stamp must still succeed.
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)

    def test_clear_then_open_with_different_stamp_still_raises(
        self,
        chroma_path: str,
        openai_stamp: EmbedderStamp,
        local_stamp: EmbedderStamp,
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, n_chunks=1)
        client.store_filing(pf)
        client.clear_collection()

        with pytest.raises(EmbeddingCollectionMismatchError):
            ChromaDBClient(local_stamp, chroma_path=chroma_path)


# ---------------------------------------------------------------------------
# Migration flag preserved from SEC-SemanticSearch
# ---------------------------------------------------------------------------


class TestMigrationFlag:
    def test_migration_flag_set_on_fresh_collection(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        meta = client._collection.metadata or {}
        assert meta.get(ChromaDBClient._MIGRATION_FLAG) is True

    def test_migration_is_o1_after_first_run(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        # Second open must not re-scan — we can't easily observe that
        # from the outside, but we at least assert the flag survives.
        second = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        meta = second._collection.metadata or {}
        assert meta.get(ChromaDBClient._MIGRATION_FLAG) is True


# ---------------------------------------------------------------------------
# Security-focused assertions
# ---------------------------------------------------------------------------


class TestSecurity:
    @pytest.mark.security
    def test_stamp_metadata_carries_no_credential_field_names(
        self, openai_stamp: EmbedderStamp
    ) -> None:
        """The collection-metadata keys must never resemble secrets.

        Defence in depth against a future refactor that tries to stash
        credentials alongside the stamp.
        """
        rendered = openai_stamp.to_metadata()
        credential_hints = {
            "api_key",
            "apikey",
            "secret",
            "password",
            "bearer",
            "token",
            "authorization",
            "auth",
        }
        for key in rendered:
            lowered = key.lower()
            assert all(hint not in lowered for hint in credential_hints), (
                f"Stamp metadata key {key!r} resembles a credential field name"
            )

    @pytest.mark.security
    def test_mismatch_error_surfaces_uniform_hint(
        self,
        chroma_path: str,
        openai_stamp: EmbedderStamp,
        local_stamp: EmbedderStamp,
    ) -> None:
        """Mismatch hint must be the uniform, deployment-unaware string.

        The API lifespan hook is the only scenario-aware translator of
        this error.  Branching the hint per deployment inside the
        storage layer would split the refusal message across N surfaces
        and regress operator clarity.
        """
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        with pytest.raises(EmbeddingCollectionMismatchError) as excinfo:
            ChromaDBClient(local_stamp, chroma_path=chroma_path)

        hint = excinfo.value.hint
        assert "reindex" in hint.lower()
        assert "sec-rag manage reindex" in hint

    @pytest.mark.security
    def test_store_filing_refuses_none_embeddings(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        """Silent storage of chunks without vectors would corrupt retrieval.

        Honours the orchestrator docstring's contract that the storage
        layer refuses rather than quietly drops the embedding step.
        """
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, with_embeddings=False)

        with pytest.raises(DatabaseError):
            client.store_filing(pf)
        assert client.collection_count() == 0

    @pytest.mark.security
    def test_populated_unstamped_collection_refuses_with_reindex_hint(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        """Legacy populated collections without a stamp must refuse traffic.

        We cannot prove the stored vectors were produced by the
        configured embedder, so retrieval would be silently wrong.
        """
        raw = chromadb.PersistentClient(path=chroma_path)
        collection = raw.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            ids=["legacy-0"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            documents=["legacy chunk"],
            metadatas=[{"ticker": "AAPL"}],
        )

        with pytest.raises(DatabaseError) as excinfo:
            ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        assert "reindex" in str(excinfo.value).lower()

    @pytest.mark.security
    def test_client_constructor_does_not_retain_credentials(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        """The client never stores an API key or EDGAR identity.

        Stamp fields are non-secret (provider/model/dimension); every
        other credential lives in the provider layer.  A future
        refactor must not sneak a key-shaped attribute onto the client.
        """
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        credential_hints = {
            "api_key",
            "apikey",
            "secret",
            "password",
            "bearer",
            "token",
            "authorization",
        }
        for attr in vars(client):
            lowered = attr.lower()
            assert all(hint not in lowered for hint in credential_hints), (
                f"Client attribute {attr!r} resembles a credential field name"
            )
