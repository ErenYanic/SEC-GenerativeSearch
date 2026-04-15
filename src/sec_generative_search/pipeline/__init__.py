"""SEC filing ingestion pipeline — fetch, parse, chunk, and orchestrate.

Public API re-exported here so callers can do::

    from sec_generative_search.pipeline import FilingFetcher, FilingParser, TextChunker

The embedding step is deliberately absent — Phase 5 introduces a
provider-backed embedder that the orchestrator consumes via
:class:`ChunkEmbedder`.
"""

from sec_generative_search.pipeline.chunk import TextChunker
from sec_generative_search.pipeline.fetch import FilingFetcher, FilingInfo
from sec_generative_search.pipeline.orchestrator import (
    ChunkEmbedder,
    PipelineOrchestrator,
    ProcessedFiling,
    ProgressCallback,
)
from sec_generative_search.pipeline.parse import FilingParser

__all__ = [
    "ChunkEmbedder",
    "FilingFetcher",
    "FilingInfo",
    "FilingParser",
    "PipelineOrchestrator",
    "ProcessedFiling",
    "ProgressCallback",
    "TextChunker",
]
