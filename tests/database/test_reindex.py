"""Tests for :class:`sec_generative_search.database.ReindexService`.

Drives a real :class:`chromadb.PersistentClient` under ``tmp_path`` and
a deterministic in-memory embedder.  The embedder is the only test
double — paging, staging, and swap all run against Chroma itself so
the tests exercise the actual contracts the service depends on.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

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
    ReindexReport,
)
from sec_generative_search.database import ChromaDBClient, ReindexService
from sec_generative_search.pipeline.orchestrator import ProcessedFiling

# ---------------------------------------------------------------------------
# Fake embedder — deterministic, in-memory, dimension-configurable
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Minimal embedder double satisfying the surface ReindexService uses.

    Only ``embed_texts`` and ``get_dimension`` are called by the
    service; anything else would be a drift from the design and should
    break the test.  ``fail_after`` triggers a deterministic failure
    mid-embedding so the cleanup path is testable without provider SDKs.
    """

    provider_name = "fake"

    def __init__(
        self,
        dimension: int,
        *,
        fail_after: int | None = None,
    ) -> None:
        self._dimension = dimension
        self._fail_after = fail_after
        self._seen = 0
        self.embed_calls: list[int] = []

    def get_dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.embed_calls.append(len(texts))
        if self._fail_after is not None and self._seen >= self._fail_after:
            raise RuntimeError("simulated embedder failure")
        self._seen += len(texts)
        # Encode the text length into one component so distinct inputs
        # produce distinct vectors — enough for round-trip checks
        # without pulling a real model.
        rows = []
        for idx, text in enumerate(texts):
            base = [float(idx), float(len(text)), 0.0, 0.0]
            rows.append(base[: self._dimension])
        return np.asarray(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma_path(tmp_path: Path) -> str:
    """Per-test ChromaDB persistence directory."""
    return str(tmp_path / "chroma")


@pytest.fixture
def source_stamp() -> EmbedderStamp:
    return EmbedderStamp(
        provider="local",
        model="google/embeddinggemma-300m",
        dimension=4,
    )


@pytest.fixture
def target_stamp() -> EmbedderStamp:
    return EmbedderStamp(
        provider="openai",
        model="text-embedding-3-small",
        dimension=4,
    )


def _make_filing_id(
    *,
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


def _make_chunks(filing_id: FilingIdentifier, n: int) -> list[Chunk]:
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
) -> ProcessedFiling:
    filing_id = filing_id or _make_filing_id()
    chunks = _make_chunks(filing_id, n=n_chunks)
    embeddings = np.arange(
        n_chunks * stamp.dimension,
        dtype=np.float32,
    ).reshape(n_chunks, stamp.dimension)
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


def _seed_source_collection(
    chroma_path: str,
    source_stamp: EmbedderStamp,
    *,
    n_filings: int = 2,
    n_chunks_each: int = 3,
) -> int:
    """Seed the live collection and return its total chunk count."""
    client = ChromaDBClient(source_stamp, chroma_path=chroma_path)
    for i in range(n_filings):
        pf = _make_processed_filing(
            source_stamp,
            filing_id=_make_filing_id(
                ticker=f"TK{i}",
                accession_number=f"0000320193-23-00000{i}",
            ),
            n_chunks=n_chunks_each,
        )
        client.store_filing(pf)
    total = client.collection_count()
    # Drop the high-level reference so the Chroma client is not shared
    # across the test — the service reopens the persistent directory
    # through its own raw client.
    del client
    return total


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_rebuilds_collection_with_target_stamp(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        total = _seed_source_collection(chroma_path, source_stamp)

        service = ReindexService(chroma_path=chroma_path, batch_size=2)
        report = service.run(target_stamp, _FakeEmbedder(dimension=4))

        assert isinstance(report, ReindexReport)
        assert report.source_stamp == source_stamp
        assert report.target_stamp == target_stamp
        assert report.chunks_copied == total
        assert report.duration_seconds >= 0.0

        # Reopening with the target stamp must succeed — confirms the
        # swap rebuilt ``sec_filings`` with the correct seal.
        reopened = ChromaDBClient(target_stamp, chroma_path=chroma_path)
        assert reopened.collection_count() == total

    def test_run_drops_staging_on_success(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        _seed_source_collection(chroma_path, source_stamp)
        service = ReindexService(chroma_path=chroma_path, batch_size=2)

        service.run(target_stamp, _FakeEmbedder(dimension=4))

        raw = chromadb.PersistentClient(path=chroma_path)
        names = {c.name for c in raw.list_collections()}
        assert ReindexService._STAGING_COLLECTION_NAME not in names

    def test_run_emits_progress_callbacks(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        total = _seed_source_collection(chroma_path, source_stamp)
        service = ReindexService(chroma_path=chroma_path, batch_size=2)
        events: list[tuple[str, int, int]] = []

        service.run(
            target_stamp,
            _FakeEmbedder(dimension=4),
            progress_callback=lambda step, cur, tot: events.append((step, cur, tot)),
        )

        steps = {e[0] for e in events}
        # Embedding phase emits "reindex"; swap phase emits
        # "reindex-swap" — both must fire for a multi-batch run.
        assert "reindex" in steps
        assert "reindex-swap" in steps
        # Final "reindex" event must report completion of the embedding
        # pass at the full chunk count.
        embed_events = [e for e in events if e[0] == "reindex"]
        assert embed_events[-1] == ("reindex", total, total)

    def test_run_preserves_chunk_metadata(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        """Reindex must not alter chunk IDs or per-chunk metadata."""
        _seed_source_collection(chroma_path, source_stamp, n_filings=1, n_chunks_each=3)
        raw_before = chromadb.PersistentClient(path=chroma_path)
        before = raw_before.get_collection(name=COLLECTION_NAME).get(
            include=["documents", "metadatas"],
        )
        ids_before = sorted(before["ids"])
        docs_before = dict(zip(before["ids"], before["documents"], strict=True))
        del raw_before

        service = ReindexService(chroma_path=chroma_path, batch_size=2)
        service.run(target_stamp, _FakeEmbedder(dimension=4))

        raw_after = chromadb.PersistentClient(path=chroma_path)
        after = raw_after.get_collection(name=COLLECTION_NAME).get(
            include=["documents", "metadatas"],
        )
        assert sorted(after["ids"]) == ids_before
        docs_after = dict(zip(after["ids"], after["documents"], strict=True))
        assert docs_after == docs_before

    def test_run_is_idempotent_across_instances(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        """Second run is a no-op refuse once source stamp == target."""
        _seed_source_collection(chroma_path, source_stamp)

        service = ReindexService(chroma_path=chroma_path, batch_size=2)
        service.run(target_stamp, _FakeEmbedder(dimension=4))

        # Second run now sees target_stamp as the source; refusing to
        # reindex-to-self is the no-op contract.
        with pytest.raises(DatabaseError, match="nothing to reindex"):
            service.run(target_stamp, _FakeEmbedder(dimension=4))


# ---------------------------------------------------------------------------
# Refuse-early paths
# ---------------------------------------------------------------------------


class TestRefuseEarly:
    def test_dimension_mismatch_raises_before_io(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        _seed_source_collection(chroma_path, source_stamp)

        mismatched = _FakeEmbedder(dimension=8)  # target_stamp declares 4
        service = ReindexService(chroma_path=chroma_path)

        with pytest.raises(DatabaseError, match="does not match target stamp"):
            service.run(target_stamp, mismatched)

        # Nothing should have been created.
        raw = chromadb.PersistentClient(path=chroma_path)
        names = {c.name for c in raw.list_collections()}
        assert ReindexService._STAGING_COLLECTION_NAME not in names
        assert mismatched.embed_calls == []

    def test_missing_source_collection_raises(
        self,
        chroma_path: str,
        target_stamp: EmbedderStamp,
    ) -> None:
        # Nothing was ever ingested — source collection absent.
        service = ReindexService(chroma_path=chroma_path)

        with pytest.raises(DatabaseError, match="not found"):
            service.run(target_stamp, _FakeEmbedder(dimension=4))

    def test_empty_source_collection_raises(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        # Stamp a fresh collection via ChromaDBClient but leave it
        # empty — the source-count check rejects this as ambiguous.
        ChromaDBClient(source_stamp, chroma_path=chroma_path)

        service = ReindexService(chroma_path=chroma_path)

        with pytest.raises(DatabaseError, match="empty"):
            service.run(target_stamp, _FakeEmbedder(dimension=4))

    def test_unstamped_populated_source_raises(
        self,
        chroma_path: str,
        target_stamp: EmbedderStamp,
    ) -> None:
        """Legacy populated collection with no stamp cannot be reindexed safely."""
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
        del raw

        service = ReindexService(chroma_path=chroma_path)
        with pytest.raises(DatabaseError, match="unstamped"):
            service.run(target_stamp, _FakeEmbedder(dimension=4))

    def test_corrupt_source_stamp_raises(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        _seed_source_collection(chroma_path, source_stamp)

        # Corrupt the dimension key directly on the collection metadata.
        raw = chromadb.PersistentClient(path=chroma_path)
        col = raw.get_collection(name=COLLECTION_NAME)
        corrupt = {k: v for k, v in (col.metadata or {}).items() if not k.startswith("hnsw:")}
        corrupt["embedding_dimension"] = "not-an-integer"
        col.modify(metadata=corrupt)
        del raw

        service = ReindexService(chroma_path=chroma_path)
        with pytest.raises(DatabaseError, match="corrupt"):
            service.run(target_stamp, _FakeEmbedder(dimension=4))

    def test_noop_raises_when_stamps_match(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
    ) -> None:
        _seed_source_collection(chroma_path, source_stamp)
        service = ReindexService(chroma_path=chroma_path)

        with pytest.raises(DatabaseError, match="nothing to reindex"):
            service.run(source_stamp, _FakeEmbedder(dimension=4))

    def test_batch_size_must_be_positive(self, chroma_path: str) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            ReindexService(chroma_path=chroma_path, batch_size=0)


# ---------------------------------------------------------------------------
# Failure paths + cleanup
# ---------------------------------------------------------------------------


class TestRunFailures:
    def test_embedding_failure_drops_staging(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        _seed_source_collection(chroma_path, source_stamp, n_filings=2, n_chunks_each=3)
        service = ReindexService(chroma_path=chroma_path, batch_size=2)
        embedder = _FakeEmbedder(dimension=4, fail_after=2)

        with pytest.raises(DatabaseError, match="Embedder failed"):
            service.run(target_stamp, embedder)

        # Staging collection must have been dropped.
        raw = chromadb.PersistentClient(path=chroma_path)
        names = {c.name for c in raw.list_collections()}
        assert ReindexService._STAGING_COLLECTION_NAME not in names

    def test_embedding_failure_leaves_live_collection_intact(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        total = _seed_source_collection(chroma_path, source_stamp)
        service = ReindexService(chroma_path=chroma_path, batch_size=2)
        embedder = _FakeEmbedder(dimension=4, fail_after=1)

        with pytest.raises(DatabaseError):
            service.run(target_stamp, embedder)

        # Live collection must still open under the original stamp and
        # carry all pre-reindex chunks.
        reopened = ChromaDBClient(source_stamp, chroma_path=chroma_path)
        assert reopened.collection_count() == total

    def test_swap_read_failure_restores_nothing_but_drops_staging(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A swap-phase failure is operator-scope; staging still dropped.

        We do not restore the live collection automatically.  What we
        do guarantee is that staging does not survive, so a retry does
        not trip the "staging already present" refuse path.
        """
        _seed_source_collection(chroma_path, source_stamp)
        service = ReindexService(chroma_path=chroma_path, batch_size=2)

        original_delete = service._client.delete_collection
        delete_calls: list[str] = []

        def _delete_spy(*args: Any, **kwargs: Any) -> Any:
            name = kwargs.get("name", args[0] if args else "")
            delete_calls.append(name)
            if name == COLLECTION_NAME and "staging" not in name:
                raise RuntimeError("simulated Chroma delete failure")
            return original_delete(*args, **kwargs)

        monkeypatch.setattr(service._client, "delete_collection", _delete_spy)

        with pytest.raises(DatabaseError, match="drop live collection"):
            service.run(target_stamp, _FakeEmbedder(dimension=4))

        # After the cleanup path, staging is gone.
        raw = chromadb.PersistentClient(path=chroma_path)
        names = {c.name for c in raw.list_collections()}
        assert ReindexService._STAGING_COLLECTION_NAME not in names

    def test_cleanup_never_shadows_original_error(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Embedder error must reach the caller even if cleanup errors."""
        _seed_source_collection(chroma_path, source_stamp)
        service = ReindexService(chroma_path=chroma_path, batch_size=2)

        original_delete = service._client.delete_collection

        def _delete(*args: Any, **kwargs: Any) -> Any:
            name = kwargs.get("name", args[0] if args else "")
            if name == ReindexService._STAGING_COLLECTION_NAME:
                # Cleanup target — simulate a secondary failure.
                raise RuntimeError("simulated cleanup failure")
            return original_delete(*args, **kwargs)

        monkeypatch.setattr(service._client, "delete_collection", _delete)

        # Primary failure: embedder dies immediately.
        embedder = _FakeEmbedder(dimension=4, fail_after=0)

        with pytest.raises(DatabaseError) as excinfo:
            service.run(target_stamp, embedder)

        # The original embedder error must be the one surfaced.
        assert "Embedder failed" in str(excinfo.value)
        assert "cleanup failure" not in str(excinfo.value)

    def test_stale_staging_is_dropped_before_new_run(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        """A leftover staging collection from a crashed run must not block reindex."""
        _seed_source_collection(chroma_path, source_stamp)

        # Forge a stale staging collection.
        raw = chromadb.PersistentClient(path=chroma_path)
        raw.create_collection(
            name=ReindexService._STAGING_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "note": "stale"},
        )
        del raw

        service = ReindexService(chroma_path=chroma_path, batch_size=2)
        report = service.run(target_stamp, _FakeEmbedder(dimension=4))
        assert report.chunks_copied > 0


# ---------------------------------------------------------------------------
# Security / defence-in-depth
# ---------------------------------------------------------------------------


class TestSecurity:
    @pytest.mark.security
    def test_service_instance_holds_no_credentials(
        self,
        chroma_path: str,
    ) -> None:
        """ReindexService must not grow credential-bearing attributes."""
        service = ReindexService(chroma_path=chroma_path)
        forbidden = (
            "api_key",
            "apikey",
            "secret",
            "password",
            "bearer",
            "authorization",
            "credential",
            "encryption_key",
            "token",
            "api_token",
        )
        for attr in vars(service):
            lowered = attr.lower()
            for hint in forbidden:
                assert hint not in lowered, (
                    f"ReindexService grew a credential-shaped attribute: {attr}"
                )

    @pytest.mark.security
    def test_noop_is_refused_loudly(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
    ) -> None:
        """Reindex-to-self must raise — a silent early-return would mask CLI typos.

        Refusing loudly guards against an operator accidentally
        destroying and rebuilding the live collection against the same
        embedder.
        """
        _seed_source_collection(chroma_path, source_stamp)
        service = ReindexService(chroma_path=chroma_path)

        with pytest.raises(DatabaseError, match="identical"):
            service.run(source_stamp, _FakeEmbedder(dimension=4))

    @pytest.mark.security
    def test_dimension_mismatch_refuses_before_touching_state(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        """Stamp/embedder drift must be caught before any write.

        If we allowed a mismatched dimension to stamp a staging
        collection, the failure would surface much later — at the
        first ``staging.add()`` — and would leave a corrupt staging
        collection behind.  Early refuse keeps the state machine
        clean.
        """
        _seed_source_collection(chroma_path, source_stamp)
        service = ReindexService(chroma_path=chroma_path)

        with pytest.raises(DatabaseError, match="does not match target stamp"):
            service.run(target_stamp, _FakeEmbedder(dimension=8))

        raw = chromadb.PersistentClient(path=chroma_path)
        names = {c.name for c in raw.list_collections()}
        assert ReindexService._STAGING_COLLECTION_NAME not in names

    @pytest.mark.security
    def test_unstamped_source_refuses_with_clear_message(
        self,
        chroma_path: str,
        target_stamp: EmbedderStamp,
    ) -> None:
        """Unstamped populated source is defence in depth against data corruption.

        Re-embedding chunks produced by an unknown embedder would
        happily succeed but silently change retrieval semantics.  The
        refuse surfaces the problem to the operator instead.
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
        del raw

        service = ReindexService(chroma_path=chroma_path)
        with pytest.raises(DatabaseError, match="unstamped"):
            service.run(target_stamp, _FakeEmbedder(dimension=4))

    @pytest.mark.security
    def test_module_imports_no_pipeline_or_ui_dependencies(self) -> None:
        """Reindex must stay surface-agnostic — no ``rich`` / ``typer`` / ``edgartools``.

        The CLI wrapper drives progress through an injected callback;
        the service itself never pulls UI or network SDKs.  This test
        reads the source and asserts the imports match the expectation
        rather than relying on a side-effect-free probe.
        """
        reindex_src = Path("src/sec_generative_search/database/reindex.py").read_text(
            encoding="utf-8"
        )

        forbidden = ("rich", "typer", "edgartools", "edgar")
        for name in forbidden:
            assert f"import {name}" not in reindex_src, (
                f"reindex.py must not import {name!r} — it is surface-agnostic"
            )
            assert f"from {name}" not in reindex_src, (
                f"reindex.py must not import from {name!r} — it is surface-agnostic"
            )

        # Pipeline modules are a separate concern — reindex reads from
        # stored chunk text, never re-fetches.
        assert "sec_generative_search.pipeline" not in reindex_src, (
            "reindex.py must not depend on the ingestion pipeline"
        )

    @pytest.mark.security
    def test_module_migration_flag_matches_client(self) -> None:
        """``ReindexService._MIGRATION_FLAG`` must match ``ChromaDBClient``.

        The flag is re-declared locally to keep the module
        dependency-light, but the two must stay in lockstep or a
        rebuilt collection would lose the O(1) startup guarantee.
        """
        assert ReindexService._MIGRATION_FLAG == ChromaDBClient._MIGRATION_FLAG, (
            "Migration-flag constants drifted between reindex.py and client.py"
        )

    @pytest.mark.security
    def test_rebuilt_collection_carries_migration_flag(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        """Swap must set ``_MIGRATION_FLAG=True`` on the new live collection.

        Without it, the next ``ChromaDBClient`` open would re-run the
        ``filing_date_int`` migration on every startup — a regression
        of the startup invariant.
        """
        _seed_source_collection(chroma_path, source_stamp)
        service = ReindexService(chroma_path=chroma_path, batch_size=2)
        service.run(target_stamp, _FakeEmbedder(dimension=4))

        raw = chromadb.PersistentClient(path=chroma_path)
        col = raw.get_collection(name=COLLECTION_NAME)
        meta = col.metadata or {}
        assert meta.get(ChromaDBClient._MIGRATION_FLAG) is True


# ---------------------------------------------------------------------------
# End-to-end: reindex then retrieval through ChromaDBClient
# ---------------------------------------------------------------------------


class TestReindexEndToEnd:
    def test_chromadb_client_serves_rebuilt_collection(
        self,
        chroma_path: str,
        source_stamp: EmbedderStamp,
        target_stamp: EmbedderStamp,
    ) -> None:
        """After reindex, retrieval through ChromaDBClient honours the new stamp.

        The client must refuse the old stamp (mismatch) and accept
        the new stamp.  This is the load-bearing promise reindex
        exists to deliver.
        """
        _seed_source_collection(chroma_path, source_stamp)
        service = ReindexService(chroma_path=chroma_path, batch_size=2)
        service.run(target_stamp, _FakeEmbedder(dimension=4))

        # Old stamp now fails.
        with pytest.raises(EmbeddingCollectionMismatchError):
            ChromaDBClient(source_stamp, chroma_path=chroma_path)

        # New stamp opens cleanly and the collection is populated.
        client = ChromaDBClient(target_stamp, chroma_path=chroma_path)
        assert client.collection_count() > 0
