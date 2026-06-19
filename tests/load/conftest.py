"""Fixtures and helpers for the load / throughput suite.

The suite reuses the deterministic doubles from the integration tests
and keeps everything else on the production stack. It stays fully
offline and content-free: no network, no provider, no credential. It
measures throughput correctness under volume and concurrency safety, not
wall-clock SLOs.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path

import pytest

from sec_generative_search.core.types import EmbedderStamp
from sec_generative_search.database import ChromaDBClient, FilingStore, MetadataRegistry
from sec_generative_search.pipeline.orchestrator import ProcessedFiling
from sec_generative_search.search.retrieval import RetrievalService

# Reuse the integration suite's deterministic doubles verbatim.
from tests.integration.conftest import (
    EMBED_DIM,
    KeywordEmbedder,
    ScriptedLLM,
    build_processed_filing,
    make_filing_id,
)

__all__ = [
    "EMBED_DIM",
    "KeywordEmbedder",
    "LatencySummary",
    "ScriptedLLM",
    "build_processed_filing",
    "make_corpus",
    "make_filing",
    "make_filing_id",
    "measure",
]


# ---------------------------------------------------------------------------
# Latency measurement helpers
# ---------------------------------------------------------------------------


class LatencySummary:
    """Percentile summary over a list of per-operation durations (seconds)."""

    def __init__(self, label: str, samples: list[float]) -> None:
        self.label = label
        self.count = len(samples)
        ordered = sorted(samples)
        self.mean = statistics.fmean(ordered) if ordered else 0.0
        self.p50 = _percentile(ordered, 0.50)
        self.p95 = _percentile(ordered, 0.95)
        self.max = ordered[-1] if ordered else 0.0

    def report(self) -> None:
        """Print the summary (visible when pytest runs with ``-s``)."""
        print(
            f"\n[load] {self.label}: n={self.count} "
            f"mean={self.mean * 1e3:.2f}ms "
            f"p50={self.p50 * 1e3:.2f}ms "
            f"p95={self.p95 * 1e3:.2f}ms "
            f"max={self.max * 1e3:.2f}ms"
        )


def _percentile(ordered: list[float], q: float) -> float:
    """Nearest-rank percentile over an already-sorted list."""
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return ordered[index]


def measure(label: str, n: int, op: Callable[[int], object]) -> LatencySummary:
    """Run ``op(i)`` ``n`` times, timing each call, and summarise latency.

    ``op`` receives the zero-based iteration index so callers can vary the
    operation (e.g. cycle through a query set) without closure gymnastics.
    """
    samples: list[float] = []
    for i in range(n):
        start = time.perf_counter()
        op(i)
        samples.append(time.perf_counter() - start)
    summary = LatencySummary(label, samples)
    summary.report()
    return summary


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

# Per-filing sections.  Every line carries vocabulary terms so the stored
# vectors are non-degenerate (an all-zero vector NaNs under cosine).
_SECTION_TEMPLATE: tuple[tuple[str, str], ...] = (
    ("Part I > Item 1A > Risk Factors", "Litigation and lawsuit risk from competition."),
    ("Part I > Item 1A > Cybersecurity", "Cybersecurity risk threatens the supply chain."),
    ("Part II > Item 7 > MD&A", "Revenue growth was strong and operating margin expanded."),
    ("Part II > Item 7 > Liquidity", "The board raised the dividend and repaid debt."),
    ("Part II > Item 7 > Outlook", "Guidance reflects currency and tax headwinds."),
    ("Part II > Item 8 > M&A", "A pending acquisition could change the growth outlook."),
)


# chunk_ids are ``TICKER_FORM_DATE_INDEX`` — accession is NOT part of the
# id, so distinct filings need a distinct (ticker, form, date) triple or
# their chunk_ids collide in ChromaDB.  Varying the filing date per index
# keeps every chunk_id unique while leaving the section-index suffix (what
# the retrieval suite asserts on) stable.
_CORPUS_BASE_DATE = date(2020, 1, 1)


def _date_for(index: int) -> date:
    return _CORPUS_BASE_DATE + timedelta(days=index)


def make_corpus(
    store: FilingStore,
    embedder: KeywordEmbedder,
    *,
    count: int,
    sections: int = len(_SECTION_TEMPLATE),
) -> int:
    """Ingest ``count`` distinct filings sequentially; return total chunks.

    Each filing gets a unique, shape-valid accession AND a unique filing
    date so its chunk_ids do not collide.  Used by the retrieval /
    streaming suites that need a populated store; the ingestion suite
    times this path directly.
    """
    total_chunks = 0
    for i in range(count):
        filing = make_filing(embedder, index=i, sections=sections)
        assert store.store_filing(filing, register_if_new=True) is True
        total_chunks += filing.ingest_result.chunk_count
    return total_chunks


def make_filing(
    embedder: KeywordEmbedder,
    *,
    index: int,
    sections: int = len(_SECTION_TEMPLATE),
) -> ProcessedFiling:
    """Build (but do not store) one filing with a unique accession + date."""
    return build_processed_filing(
        make_filing_id(
            filing_date=_date_for(index),
            accession_number=f"0000320193-23-{index:06d}",
        ),
        list(_SECTION_TEMPLATE[:sections]),
        embedder,
    )


# ---------------------------------------------------------------------------
# Real-stack fixtures (mirrors tests/integration/conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def stamp() -> EmbedderStamp:
    return EmbedderStamp(provider="keyword-test", model="keyword-bag", dimension=EMBED_DIM)


@pytest.fixture
def embedder() -> KeywordEmbedder:
    return KeywordEmbedder()


@pytest.fixture
def chroma(stamp: EmbedderStamp, tmp_path: Path) -> ChromaDBClient:
    return ChromaDBClient(stamp, chroma_path=str(tmp_path / "chroma"))


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[MetadataRegistry]:
    reg = MetadataRegistry(db_path=str(tmp_path / "metadata.sqlite"))
    yield reg
    reg.close()


@pytest.fixture
def store(chroma: ChromaDBClient, registry: MetadataRegistry) -> FilingStore:
    return FilingStore(chroma, registry)


@pytest.fixture
def retrieval(embedder: KeywordEmbedder, chroma: ChromaDBClient) -> RetrievalService:
    return RetrievalService(embedder, chroma)


@pytest.fixture(autouse=True)
def _reset_settings_singleton() -> Iterator[None]:
    """Keep the settings singleton clean across load tests."""
    from sec_generative_search.config.settings import reload_settings

    reload_settings()
    yield
    reload_settings()
