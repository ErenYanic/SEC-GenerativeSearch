"""Tests for PortableExportService / PortableImportService.

Drives a real ``chromadb.PersistentClient`` and a real
:class:`MetadataRegistry` under ``tmp_path`` so the export round-trip
exercises the actual filesystem and SQLite contracts.  A deterministic
in-memory embedder fakes the embedding step on the import side — the
tests pin the seam, not provider SDKs.
"""

from __future__ import annotations

import json
import stat
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.core.types import (
    Chunk,
    ContentType,
    EmbedderStamp,
    ExportReport,
    FilingIdentifier,
    ImportReport,
    IngestResult,
)
from sec_generative_search.database import (
    ChromaDBClient,
    FilingStore,
    MetadataRegistry,
    PortableExportService,
    PortableImportService,
)
from sec_generative_search.pipeline.orchestrator import ProcessedFiling

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic in-memory embedder satisfying the import service surface."""

    provider_name = "fake"

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self.embed_calls: list[int] = []

    def get_dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.embed_calls.append(len(texts))
        rows = []
        for idx, text in enumerate(texts):
            base = [float(idx), float(len(text)), 0.0, 0.0]
            rows.append(base[: self._dimension])
        return np.asarray(rows, dtype=np.float32)


def _filing_id(
    *,
    ticker: str = "AAPL",
    accession: str = "0000320193-23-000077",
    filing_date: date | None = None,
) -> FilingIdentifier:
    return FilingIdentifier(
        ticker=ticker,
        form_type="10-K",
        filing_date=filing_date or date(2023, 11, 3),
        accession_number=accession,
    )


def _processed(
    stamp: EmbedderStamp,
    *,
    filing_id: FilingIdentifier | None = None,
    n_chunks: int = 2,
) -> ProcessedFiling:
    fid = filing_id or _filing_id()
    chunks = [
        Chunk(
            content=f"Chunk {i} for {fid.ticker} {fid.form_type}",
            path=f"Part I > Item {i + 1} > Heading",
            content_type=ContentType.TEXT,
            filing_id=fid,
            chunk_index=i,
            token_count=4,
        )
        for i in range(n_chunks)
    ]
    embeddings = np.arange(n_chunks * stamp.dimension, dtype=np.float32).reshape(
        n_chunks, stamp.dimension
    )
    return ProcessedFiling(
        filing_id=fid,
        chunks=chunks,
        embeddings=embeddings,
        ingest_result=IngestResult(
            filing_id=fid,
            segment_count=n_chunks,
            chunk_count=n_chunks,
            duration_seconds=0.0,
        ),
    )


def _seed_two_filings(
    chroma_path: str,
    metadata_db_path: str,
    stamp: EmbedderStamp,
) -> None:
    chroma = ChromaDBClient(stamp, chroma_path=chroma_path)
    registry = MetadataRegistry(db_path=metadata_db_path)
    try:
        for ticker, accession in (
            ("AAPL", "0000320193-23-000077"),
            ("MSFT", "0000789019-23-000099"),
        ):
            pf = _processed(
                stamp,
                filing_id=_filing_id(ticker=ticker, accession=accession),
                n_chunks=3,
            )
            chroma.store_filing(pf)
            registry.register_filing(pf.filing_id, pf.ingest_result.chunk_count)
    finally:
        registry.close()
        del chroma


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma_path(tmp_path: Path) -> str:
    return str(tmp_path / "chroma")


@pytest.fixture
def metadata_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "metadata.sqlite")


@pytest.fixture
def stamp() -> EmbedderStamp:
    return EmbedderStamp(
        provider="local",
        model="google/embeddinggemma-300m",
        dimension=4,
    )


@pytest.fixture
def seeded_storage(
    chroma_path: str,
    metadata_db_path: str,
    stamp: EmbedderStamp,
) -> tuple[str, str, EmbedderStamp]:
    _seed_two_filings(chroma_path, metadata_db_path, stamp)
    return chroma_path, metadata_db_path, stamp


# ---------------------------------------------------------------------------
# Export — happy path
# ---------------------------------------------------------------------------


class TestExportHappyPath:
    def test_export_writes_manifest_and_jsonl(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        chroma_path, metadata_db_path, stamp = seeded_storage
        out = tmp_path / "export"

        service = PortableExportService(
            chroma_path=chroma_path,
            metadata_db_path=metadata_db_path,
        )
        report = service.export(out)

        assert isinstance(report, ExportReport)
        assert report.source_embedder_stamp == stamp
        assert report.filing_count == 2
        assert report.chunk_count == 6
        assert (out / "manifest.json").is_file()
        assert (out / "chunks.jsonl").is_file()

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["format_version"] == 1
        assert manifest["source_embedder_stamp"] == {
            "provider": stamp.provider,
            "model": stamp.model,
            "dimension": stamp.dimension,
        }
        assert manifest["filing_count"] == 2
        assert manifest["chunk_count"] == 6

        rows = [
            json.loads(line)
            for line in (out / "chunks.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 6
        for row in rows:
            assert set(row.keys()) >= {
                "accession",
                "ticker",
                "form_type",
                "filing_date",
                "chunk_index",
                "section_path",
                "content_type",
                "text",
            }
            assert row["form_type"] == "10-K"
            assert row["content_type"] == "text"

    def test_filter_by_ticker(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        chroma_path, metadata_db_path, _ = seeded_storage
        service = PortableExportService(
            chroma_path=chroma_path,
            metadata_db_path=metadata_db_path,
        )
        report = service.export(tmp_path / "export", tickers=["aapl"])
        assert report.filing_count == 1
        assert report.chunk_count == 3

        rows = [
            json.loads(line)
            for line in (tmp_path / "export" / "chunks.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert {row["ticker"] for row in rows} == {"AAPL"}

    def test_filter_by_accession(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        chroma_path, metadata_db_path, _ = seeded_storage
        service = PortableExportService(
            chroma_path=chroma_path,
            metadata_db_path=metadata_db_path,
        )
        report = service.export(
            tmp_path / "export",
            accessions=["0000789019-23-000099"],
        )
        assert report.filing_count == 1
        assert report.chunk_count == 3
        rows = [
            json.loads(line)
            for line in (tmp_path / "export" / "chunks.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert {row["ticker"] for row in rows} == {"MSFT"}

    def test_progress_callback_emits_steps(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        chroma_path, metadata_db_path, _ = seeded_storage
        events: list[tuple[str, int, int]] = []
        service = PortableExportService(
            chroma_path=chroma_path,
            metadata_db_path=metadata_db_path,
        )
        service.export(
            tmp_path / "export",
            progress_callback=lambda s, c, t: events.append((s, c, t)),
        )
        steps = {s for s, _, _ in events}
        assert {"export-list-filings", "export-chunks", "export-manifest"} <= steps

    def test_export_to_empty_storage_writes_zero_rows(
        self,
        chroma_path: str,
        metadata_db_path: str,
        stamp: EmbedderStamp,
        tmp_path: Path,
    ) -> None:
        # Bring up the storage but keep it empty.
        ChromaDBClient(stamp, chroma_path=chroma_path)
        registry = MetadataRegistry(db_path=metadata_db_path)
        registry.close()

        service = PortableExportService(
            chroma_path=chroma_path,
            metadata_db_path=metadata_db_path,
        )
        report = service.export(tmp_path / "export")
        assert report.filing_count == 0
        assert report.chunk_count == 0
        assert (tmp_path / "export" / "chunks.jsonl").read_bytes() == b""


# ---------------------------------------------------------------------------
# Export — refuse paths
# ---------------------------------------------------------------------------


class TestExportRefuse:
    def test_refuses_non_empty_existing_dir_without_force(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        chroma_path, metadata_db_path, _ = seeded_storage
        out = tmp_path / "export"
        out.mkdir()
        (out / "old.txt").write_text("residue")

        service = PortableExportService(
            chroma_path=chroma_path,
            metadata_db_path=metadata_db_path,
        )
        with pytest.raises(DatabaseError, match="not empty"):
            service.export(out)

    def test_force_overwrites_existing_dir(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        chroma_path, metadata_db_path, _ = seeded_storage
        out = tmp_path / "export"
        out.mkdir()
        (out / "stale.json").write_text("{}")

        service = PortableExportService(
            chroma_path=chroma_path,
            metadata_db_path=metadata_db_path,
        )
        report = service.export(out, force=True)
        assert isinstance(report, ExportReport)

    def test_refuses_when_chroma_missing(
        self,
        metadata_db_path: str,
        tmp_path: Path,
    ) -> None:
        registry = MetadataRegistry(db_path=metadata_db_path)
        registry.close()
        service = PortableExportService(
            chroma_path=str(tmp_path / "absent_chroma"),
            metadata_db_path=metadata_db_path,
        )
        with pytest.raises(DatabaseError, match="ChromaDB path not found"):
            service.export(tmp_path / "export")

    def test_refuses_when_sqlite_missing(
        self,
        chroma_path: str,
        stamp: EmbedderStamp,
        tmp_path: Path,
    ) -> None:
        ChromaDBClient(stamp, chroma_path=chroma_path)
        service = PortableExportService(
            chroma_path=chroma_path,
            metadata_db_path=str(tmp_path / "absent.sqlite"),
        )
        with pytest.raises(DatabaseError, match="SQLite file not found"):
            service.export(tmp_path / "export")


# ---------------------------------------------------------------------------
# Import — happy path / round-trip
# ---------------------------------------------------------------------------


class TestImportHappyPath:
    def test_round_trip_export_then_import(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        src_chroma, src_sqlite, stamp = seeded_storage

        # Export to a portable directory.
        out = tmp_path / "export"
        exporter = PortableExportService(
            chroma_path=src_chroma,
            metadata_db_path=src_sqlite,
        )
        exporter.export(out)

        # Import into a fresh, separate target deployment with the same
        # embedder dimension so ChromaDBClient can open the collection.
        target_chroma = str(tmp_path / "target_chroma")
        target_sqlite = str(tmp_path / "target.sqlite")
        target_chroma_client = ChromaDBClient(stamp, chroma_path=target_chroma)
        target_registry = MetadataRegistry(db_path=target_sqlite)
        target_store = FilingStore(target_chroma_client, target_registry)

        embedder = _FakeEmbedder(dimension=stamp.dimension)
        importer = PortableImportService(target_store, embedder)
        report = importer.import_(out)

        assert isinstance(report, ImportReport)
        assert report.filings_imported == 2
        assert report.filings_skipped == 0
        assert report.chunks_imported == 6

        # Target storage now has both filings.
        assert target_chroma_client.collection_count() == 6
        assert target_registry.count() == 2

        # Embedder was called once per filing.
        assert sum(embedder.embed_calls) == 6

        target_registry.close()
        del target_chroma_client

    def test_duplicate_accession_is_skipped(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        src_chroma, src_sqlite, stamp = seeded_storage
        out = tmp_path / "export"
        PortableExportService(
            chroma_path=src_chroma,
            metadata_db_path=src_sqlite,
        ).export(out)

        # Target already has one of the two accessions registered.
        target_chroma_path = str(tmp_path / "target_chroma")
        target_sqlite = str(tmp_path / "target.sqlite")
        chroma = ChromaDBClient(stamp, chroma_path=target_chroma_path)
        registry = MetadataRegistry(db_path=target_sqlite)

        # Pre-register the AAPL accession via a real ProcessedFiling
        # so both stores agree on the existing filing.
        pre_existing = _processed(
            stamp,
            filing_id=_filing_id(
                ticker="AAPL",
                accession="0000320193-23-000077",
            ),
            n_chunks=1,
        )
        chroma.store_filing(pre_existing)
        registry.register_filing(
            pre_existing.filing_id,
            pre_existing.ingest_result.chunk_count,
        )

        store = FilingStore(chroma, registry)
        importer = PortableImportService(store, _FakeEmbedder(dimension=4))
        report = importer.import_(out)

        # Only the MSFT row should be newly imported.
        assert report.filings_imported == 1
        assert report.filings_skipped == 1
        assert report.chunks_imported == 3
        assert registry.count() == 2

        registry.close()
        del chroma

    def test_filter_by_ticker_on_import(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        src_chroma, src_sqlite, stamp = seeded_storage
        out = tmp_path / "export"
        PortableExportService(
            chroma_path=src_chroma,
            metadata_db_path=src_sqlite,
        ).export(out)

        target_chroma_path = str(tmp_path / "target_chroma")
        target_sqlite = str(tmp_path / "target.sqlite")
        chroma = ChromaDBClient(stamp, chroma_path=target_chroma_path)
        registry = MetadataRegistry(db_path=target_sqlite)
        store = FilingStore(chroma, registry)

        importer = PortableImportService(store, _FakeEmbedder(dimension=4))
        report = importer.import_(out, tickers=["AAPL"])

        assert report.filings_imported == 1
        assert report.filings_skipped == 0
        assert report.chunks_imported == 3

        registry.close()
        del chroma


# ---------------------------------------------------------------------------
# Import — refuse paths
# ---------------------------------------------------------------------------


class TestImportRefuse:
    def test_refuses_non_directory_input(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        _, _, stamp = seeded_storage
        target_chroma_path = str(tmp_path / "target_chroma")
        target_sqlite = str(tmp_path / "target.sqlite")
        chroma = ChromaDBClient(stamp, chroma_path=target_chroma_path)
        registry = MetadataRegistry(db_path=target_sqlite)
        store = FilingStore(chroma, registry)

        importer = PortableImportService(store, _FakeEmbedder(dimension=4))
        with pytest.raises(DatabaseError, match="not a directory"):
            importer.import_(tmp_path / "absent_dir")

        registry.close()
        del chroma

    def test_refuses_missing_manifest(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        _, _, stamp = seeded_storage
        bad = tmp_path / "broken"
        bad.mkdir()
        (bad / "chunks.jsonl").write_text("")

        target_chroma_path = str(tmp_path / "target_chroma")
        target_sqlite = str(tmp_path / "target.sqlite")
        chroma = ChromaDBClient(stamp, chroma_path=target_chroma_path)
        registry = MetadataRegistry(db_path=target_sqlite)
        store = FilingStore(chroma, registry)

        importer = PortableImportService(store, _FakeEmbedder(dimension=4))
        with pytest.raises(DatabaseError, match=r"missing 'manifest\.json'"):
            importer.import_(bad)
        registry.close()
        del chroma

    def test_refuses_wrong_format_version(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        _, _, stamp = seeded_storage
        bad = tmp_path / "wrong"
        bad.mkdir()
        (bad / "manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 99,
                    "source_embedder_stamp": {
                        "provider": stamp.provider,
                        "model": stamp.model,
                        "dimension": stamp.dimension,
                    },
                }
            )
        )
        (bad / "chunks.jsonl").write_text("")

        target_chroma_path = str(tmp_path / "target_chroma")
        target_sqlite = str(tmp_path / "target.sqlite")
        chroma = ChromaDBClient(stamp, chroma_path=target_chroma_path)
        registry = MetadataRegistry(db_path=target_sqlite)
        store = FilingStore(chroma, registry)

        importer = PortableImportService(store, _FakeEmbedder(dimension=4))
        with pytest.raises(DatabaseError, match="format_version"):
            importer.import_(bad)
        registry.close()
        del chroma

    def test_refuses_malformed_jsonl(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        _, _, stamp = seeded_storage
        bad = tmp_path / "broken"
        bad.mkdir()
        (bad / "manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "source_embedder_stamp": {
                        "provider": stamp.provider,
                        "model": stamp.model,
                        "dimension": stamp.dimension,
                    },
                }
            )
        )
        (bad / "chunks.jsonl").write_text("not json\n")

        target_chroma_path = str(tmp_path / "target_chroma")
        target_sqlite = str(tmp_path / "target.sqlite")
        chroma = ChromaDBClient(stamp, chroma_path=target_chroma_path)
        registry = MetadataRegistry(db_path=target_sqlite)
        store = FilingStore(chroma, registry)

        importer = PortableImportService(store, _FakeEmbedder(dimension=4))
        with pytest.raises(DatabaseError, match="not valid JSON"):
            importer.import_(bad)
        registry.close()
        del chroma


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecurity:
    def test_export_files_are_owner_only(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        chroma_path, metadata_db_path, _ = seeded_storage
        out = tmp_path / "export"
        PortableExportService(
            chroma_path=chroma_path,
            metadata_db_path=metadata_db_path,
        ).export(out)

        for name in ("manifest.json", "chunks.jsonl"):
            mode = (out / name).stat().st_mode & 0o777
            assert mode == 0o600, (
                f"{name} mode is {oct(mode)}; must be 0o600 to keep the artefact owner-only."
            )
            assert ((out / name).stat().st_mode & stat.S_IRWXG) == 0
            assert ((out / name).stat().st_mode & stat.S_IRWXO) == 0

    def test_export_has_no_credential_attributes(
        self,
        chroma_path: str,
        metadata_db_path: str,
    ) -> None:
        bare = PortableExportService(
            chroma_path=chroma_path,
            metadata_db_path=metadata_db_path,
        )
        forbidden = (
            "api_key",
            "apikey",
            "secret",
            "password",
            "bearer",
            "authorization",
            "credential",
            "encryption_key",
        )
        attrs = {a.lower() for a in vars(bare)}
        for hint in forbidden:
            assert hint not in attrs, (
                f"PortableExportService grew a credential-shaped attribute: {hint}"
            )
        assert "token" not in attrs

    def test_import_has_no_credential_attributes(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        _, _, stamp = seeded_storage
        chroma_path = str(tmp_path / "x_chroma")
        sqlite_path = str(tmp_path / "x.sqlite")
        chroma = ChromaDBClient(stamp, chroma_path=chroma_path)
        registry = MetadataRegistry(db_path=sqlite_path)
        store = FilingStore(chroma, registry)
        try:
            importer = PortableImportService(store, _FakeEmbedder(dimension=4))
            forbidden = (
                "api_key",
                "apikey",
                "secret",
                "password",
                "bearer",
                "authorization",
                "credential",
                "encryption_key",
            )
            attrs = {a.lower() for a in vars(importer)}
            for hint in forbidden:
                assert hint not in attrs, (
                    f"PortableImportService grew a credential-shaped attribute: {hint}"
                )
            assert "token" not in attrs
        finally:
            registry.close()
            del chroma

    def test_module_imports_no_pipeline_or_ui_dependencies(self) -> None:
        """``database/portable.py`` is surface-agnostic.

        No ``rich`` / ``typer`` / ``edgartools`` / pipeline imports.
        The CLI wrapper drives progress through an injected callback.
        """
        src = Path("src/sec_generative_search/database/portable.py").read_text(encoding="utf-8")
        for name in ("rich", "typer", "edgartools", "edgar"):
            assert f"import {name}" not in src, (
                f"portable.py must not import {name!r} — it is surface-agnostic"
            )
            assert f"from {name}" not in src, (
                f"portable.py must not import from {name!r} — it is surface-agnostic"
            )
        assert "sec_generative_search.pipeline" not in src, (
            "portable.py must not depend on the ingestion pipeline"
        )

    def test_refuses_symlink_in_lexical_parent_of_export_output(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        chroma_path, metadata_db_path, _ = seeded_storage
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target)

        service = PortableExportService(
            chroma_path=chroma_path,
            metadata_db_path=metadata_db_path,
        )
        with pytest.raises(DatabaseError, match="symlink"):
            service.export(link / "export")

    def test_refuses_symlink_in_lexical_parent_of_import_input(
        self,
        seeded_storage: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        _, _, stamp = seeded_storage
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target)

        chroma = ChromaDBClient(stamp, chroma_path=str(tmp_path / "x_chroma"))
        registry = MetadataRegistry(db_path=str(tmp_path / "x.sqlite"))
        store = FilingStore(chroma, registry)
        try:
            importer = PortableImportService(store, _FakeEmbedder(dimension=4))
            with pytest.raises(DatabaseError, match="symlink"):
                importer.import_(link / "export")
        finally:
            registry.close()
            del chroma
