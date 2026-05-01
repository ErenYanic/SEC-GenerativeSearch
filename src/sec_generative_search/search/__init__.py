"""Retrieval services — embedding-based similarity search over SEC filings.

Re-exports :class:`RetrievalService` plus the offline retrieval-evaluation
harness so callers can write
``from sec_generative_search.search import RetrievalService, evaluate_retrieval``.
"""

from sec_generative_search.search.evaluation import (
    EvaluationCase,
    EvaluationReport,
    RetrievalFn,
    evaluate_retrieval,
    load_cases_from_json,
)
from sec_generative_search.search.retrieval import RetrievalService, TokenCounter

__all__ = [
    "EvaluationCase",
    "EvaluationReport",
    "RetrievalFn",
    "RetrievalService",
    "TokenCounter",
    "evaluate_retrieval",
    "load_cases_from_json",
]
