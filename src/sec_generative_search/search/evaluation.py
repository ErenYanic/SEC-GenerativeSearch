"""Offline retrieval evaluation harness.

This module provides a small, pure scoring library for retrieval
quality. An :class:`EvaluationCase` is a hand-authored question paired
with the chunk IDs, section paths, or accession numbers that an ideal
retrieval would surface. :func:`evaluate_retrieval` runs each case
through a caller-supplied retrieval function and returns aggregate
precision@k and recall@k.

The case file is JSON so analysts can author cases without running
Python. :func:`load_cases_from_json` validates the shape strictly so a
typo surfaces at load time, not at evaluation time.

A retrieval result counts as a hit for a case when any of these match
an entry in the case's expected set:

        1. the chunk's exact ``chunk_id``,
        2. the chunk's ``accession_number``,
        3. the chunk's ``path`` (section path).

This is deliberately permissive: authoring a case by chunk ID requires
inspecting the live collection, while authoring by section path or
filing is feasible from the SEC document alone. Evaluators who want a
stricter match can pre-process their cases to use only ``chunk_id``
expectations.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sec_generative_search.core.types import RetrievalResult

__all__ = [
    "EvaluationCase",
    "EvaluationReport",
    "RetrievalFn",
    "evaluate_retrieval",
    "load_cases_from_json",
]


RetrievalFn = Callable[[str, int], list["RetrievalResult"]]
"""Callable signature accepted by :func:`evaluate_retrieval`.

Takes a query string and a ``top_k`` integer; returns the retrieval
results.  Wrap :meth:`RetrievalService.retrieve` in a small lambda when
calling so any filters / diversity caps you want under evaluation are
held constant across cases.
"""


@dataclass(frozen=True)
class EvaluationCase:
    """One question paired with the expected retrieval surface.

    Frozen so a case loaded once from disk cannot be mutated mid-
    evaluation — that would silently change the denominator of recall.

    Attributes:
        case_id: Stable identifier for the case (e.g.
            ``"q-revenue-concentration-001"``).  Used in the per-case
            breakdown of :class:`EvaluationReport`.
        query: The question to evaluate.  Free-form natural language.
        expected_chunk_ids: Chunk IDs an ideal retrieval should surface.
            Empty when the case author identified expectations only by
            section path or accession number.
        expected_accession_numbers: Filing accession numbers an ideal
            retrieval should surface.  Empty when not used.
        expected_section_paths: Section paths an ideal retrieval should
            surface (e.g. ``"Part I > Item 1A > Risk Factors"``).
            Empty when not used.
        notes: Free-form authoring notes.  Not consumed by the scorer.
    """

    case_id: str
    query: str
    expected_chunk_ids: tuple[str, ...] = ()
    expected_accession_numbers: tuple[str, ...] = ()
    expected_section_paths: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("EvaluationCase.case_id must not be empty.")
        if not self.query.strip():
            raise ValueError(f"EvaluationCase {self.case_id!r}: query must not be empty.")
        if not (
            self.expected_chunk_ids
            or self.expected_accession_numbers
            or self.expected_section_paths
        ):
            raise ValueError(
                f"EvaluationCase {self.case_id!r}: at least one of "
                f"expected_chunk_ids / expected_accession_numbers / "
                f"expected_section_paths must be non-empty."
            )

    @property
    def expected_count(self) -> int:
        """Number of distinct expected items across all match dimensions.

        Used as the denominator of recall.  Counts the union of
        chunk IDs, accession numbers, and section paths.
        """
        return (
            len(set(self.expected_chunk_ids))
            + len(set(self.expected_accession_numbers))
            + len(set(self.expected_section_paths))
        )


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate results from :func:`evaluate_retrieval`.

    Frozen — once a report is produced it represents the exact
    snapshot of evaluator behaviour at that moment, and rewriting
    it would break the audit trail when the report is logged or
    archived.

    Attributes:
        top_k: ``top_k`` the retrieval function was called with.
        case_count: Number of cases evaluated.
        precision_at_k: Mean precision over all cases.  ``hits / top_k``
            per case, then averaged.
        recall_at_k: Mean recall over all cases.  ``hits / expected``
            per case, then averaged.
        per_case: Per-case ``(case_id, precision, recall, hits,
            expected)`` tuples for drill-down.
    """

    top_k: int
    case_count: int
    precision_at_k: float
    recall_at_k: float
    per_case: tuple[tuple[str, float, float, int, int], ...] = field(default_factory=tuple)


def evaluate_retrieval(
    cases: list[EvaluationCase],
    retrieval_fn: RetrievalFn,
    *,
    top_k: int = 5,
) -> EvaluationReport:
    """Run ``cases`` through ``retrieval_fn`` and aggregate metrics.

    Args:
        cases: List of evaluation cases.  Empty list raises
            :class:`ValueError` — callers must provide at least one
            case so the report's averages are well-defined.
        retrieval_fn: Caller-supplied retrieval function.  Wrap
            :meth:`RetrievalService.retrieve` in a lambda that pins
            any filters or diversity caps you want held constant
            across the evaluation.
        top_k: ``top_k`` to pass into ``retrieval_fn`` and use as the
            precision denominator.  Must be positive.

    Returns:
        An :class:`EvaluationReport` with mean precision and recall.

    The scorer is deliberately strict on the denominator: precision
    uses ``top_k`` (not ``len(results)``), so a retrieval that returns
    fewer than ``top_k`` chunks is penalised proportionally.  This
    discourages a strategy that artificially shortens output to inflate
    precision.
    """
    if not cases:
        raise ValueError("evaluate_retrieval requires at least one case.")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}.")

    per_case: list[tuple[str, float, float, int, int]] = []
    precision_sum = 0.0
    recall_sum = 0.0

    for case in cases:
        results = retrieval_fn(case.query, top_k)
        hits = _count_hits(case, results)
        precision = hits / top_k
        recall = hits / case.expected_count if case.expected_count else 0.0
        precision_sum += precision
        recall_sum += recall
        per_case.append((case.case_id, precision, recall, hits, case.expected_count))

    n = len(cases)
    return EvaluationReport(
        top_k=top_k,
        case_count=n,
        precision_at_k=precision_sum / n,
        recall_at_k=recall_sum / n,
        per_case=tuple(per_case),
    )


def _count_hits(case: EvaluationCase, results: list[RetrievalResult]) -> int:
    """Count distinct expected items matched by *any* result.

    A single result that satisfies multiple expected items (e.g. its
    ``chunk_id`` is expected AND its accession number is expected)
    counts both — this matches the recall denominator definition in
    :attr:`EvaluationCase.expected_count`.  A given expected item is
    only credited once, even if multiple results match it.
    """
    matched_chunks: set[str] = set()
    matched_accessions: set[str] = set()
    matched_sections: set[str] = set()

    expected_chunks = set(case.expected_chunk_ids)
    expected_accessions = set(case.expected_accession_numbers)
    expected_sections = set(case.expected_section_paths)

    for r in results:
        if r.chunk_id and r.chunk_id in expected_chunks:
            matched_chunks.add(r.chunk_id)
        if r.accession_number and r.accession_number in expected_accessions:
            matched_accessions.add(r.accession_number)
        if r.path and r.path in expected_sections:
            matched_sections.add(r.path)

    return len(matched_chunks) + len(matched_accessions) + len(matched_sections)


def load_cases_from_json(path: Path | str) -> list[EvaluationCase]:
    """Load and validate evaluation cases from a JSON file.

    The file must be a JSON array of objects matching
    :class:`EvaluationCase`.  Tuple-typed fields accept JSON arrays of
    strings and are converted on load; missing optional fields default
    to empty.  Any unrecognised top-level keys raise :class:`ValueError`
    so a misspelled field name surfaces immediately rather than as a
    silent zero-recall mystery.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Evaluation case file {path} must contain a JSON array at top level.")

    allowed_keys = {
        "case_id",
        "query",
        "expected_chunk_ids",
        "expected_accession_numbers",
        "expected_section_paths",
        "notes",
    }

    cases: list[EvaluationCase] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"Evaluation case file {path}: every entry must be a JSON object.")
        unknown = set(entry.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Evaluation case file {path}: unknown field(s) {sorted(unknown)}.")
        cases.append(
            EvaluationCase(
                case_id=entry["case_id"],
                query=entry["query"],
                expected_chunk_ids=tuple(entry.get("expected_chunk_ids", ())),
                expected_accession_numbers=tuple(entry.get("expected_accession_numbers", ())),
                expected_section_paths=tuple(entry.get("expected_section_paths", ())),
                notes=entry.get("notes", ""),
            )
        )
    return cases
