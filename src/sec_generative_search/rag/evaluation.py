"""Offline answer-quality evaluation harness (Phase 16.6).

The companion to :mod:`sec_generative_search.search.evaluation`. Where the
retrieval harness scores *which chunks were surfaced*, this module scores
*what the model did with them* across three dimensions:

    1. **Answer faithfulness** — a pure, deterministic proxy for whether the
       answer's substantive claims are grounded in cited sources. The proxy
       measures the fraction of substantive sentences that carry an inline
       ``[N]`` marker resolving to a real citation, plus the count of
       *dangling* markers that point at a non-existent citation index.
    2. **Citation accuracy** — precision/recall of the chunks the answer
       cited against a hand-authored gold set, plus a hard **integrity**
    check that every citation resolves to a chunk that was actually
    retrieved (the anti-fabrication invariant the orchestrator must
    uphold).
    3. **Latency** — wall-clock duration of each answer call, aggregated to
       mean / p50 / p95 / max, timed with an injectable clock so the harness
       is deterministic under test.

Design constraints (mirroring the retrieval harness and the project's
content-free observability discipline):

- **Pure and offline.** No network call, no LLM judge, no credential touch.
  The faithfulness proxy is mechanical — it does *not* attempt natural-
  language entailment. A high grounding ratio means "well-cited", not
  "semantically true"; treat it as a regression tripwire, not a truth oracle.
- **Content-free reports.** Every score record carries the ``case_id`` and
  numeric metrics only — never the answer text, the query, or chunk content.
  A report is safe to log or archive.
- **Gold-set format is shared.** Cases reuse
  :class:`sec_generative_search.search.evaluation.EvaluationCase`, so a single
  case file drives both the retrieval and the answer harness.

The faithfulness proxy's known limitation: a refusal answer ("I cannot answer
from the provided filings") makes no factual claims yet carries no citations,
so it scores ``grounding_ratio == 0``. The harness deliberately does **not**
special-case refusals with a natural-language heuristic — callers evaluating a
mixed set should read the per-case breakdown rather than the mean alone.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sec_generative_search.search.evaluation import EvaluationCase, load_cases_from_json

if TYPE_CHECKING:
    from sec_generative_search.core.types import GenerationResult

__all__ = [
    "AnswerEvaluationReport",
    "AnswerFn",
    "CitationScore",
    "EvaluationCase",
    "FaithfulnessScore",
    "LatencyStats",
    "evaluate_answers",
    "load_cases_from_json",
    "score_citations",
    "score_faithfulness",
]


AnswerFn = Callable[[str], "GenerationResult"]
"""Callable signature accepted by :func:`evaluate_answers`.

Takes a query string and returns the :class:`GenerationResult` an ideal
caller would produce. Wrap :meth:`RAGOrchestrator.generate` in a small
lambda that pins the provider, model, and answer mode you want held
constant across the evaluation.
"""


# A sentence terminator followed by whitespace — deliberately simple so the
# split is deterministic and dependency-free. SEC prose rarely uses the
# decimal/abbreviation edge cases that trip naive splitters, and the proxy
# tolerates a little noise in the denominator.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Inline citation marker: a bracketed positive integer, e.g. ``[1]`` / ``[12]``.
# Matches the marker grammar the orchestrator emits and the SPA's
# ``splitAnswerByCitation`` consumes.
_INLINE_MARKER_RE = re.compile(r"\[(\d+)\]")

# A sentence counts as "substantive" (i.e. a candidate factual claim that
# ought to be grounded) when it carries at least one letter and clears a
# short length floor — this drops bare list bullets, headers, and stray
# punctuation from the faithfulness denominator without any NL heuristic.
_MIN_SUBSTANTIVE_CHARS = 12
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass(frozen=True)
class FaithfulnessScore:
    """Per-answer grounding proxy.

    Frozen so a computed score cannot be silently rewritten before it
    reaches a report.

    Attributes:
        substantive_sentences: Count of sentences that look like factual
            claims (clear the length floor and contain a letter).
        grounded_sentences: Substantive sentences carrying at least one
            inline ``[N]`` marker that resolves to a real citation.
        grounding_ratio: ``grounded_sentences / substantive_sentences``.
            Defined as ``1.0`` when there are no substantive sentences
            (vacuously grounded — e.g. an empty or non-prose answer), so
            the metric never divides by zero.
        dangling_marker_count: Inline ``[N]`` markers whose ``N`` is not a
            valid citation ``display_index`` on the result. A non-zero
            count is a faithfulness red flag: the answer references a
            source that does not exist.
    """

    substantive_sentences: int
    grounded_sentences: int
    grounding_ratio: float
    dangling_marker_count: int


@dataclass(frozen=True)
class CitationScore:
    """Per-answer citation accuracy + integrity.

    Attributes:
        cited_count: Number of citations the answer carried.
        fabricated_count: Citations whose ``chunk_id`` is absent from the
            result's ``retrieved_chunks``. MUST be zero — a fabricated
            citation means the orchestrator's "validate against retrieved
            chunks, never fabricate" invariant was violated.
        has_integrity: ``True`` when ``fabricated_count == 0``.
        precision: Fraction of cited chunks that match the gold set
            (``matched_citations / cited_count``); ``0.0`` when nothing was
            cited.
        recall: Fraction of gold items the citations covered
            (``matched_expected / expected_count``); ``0.0`` when the case
            declared no expectations.
    """

    cited_count: int
    fabricated_count: int
    has_integrity: bool
    precision: float
    recall: float


@dataclass(frozen=True)
class LatencyStats:
    """Aggregate wall-clock latency across an evaluation run (seconds)."""

    count: int
    mean_seconds: float
    p50_seconds: float
    p95_seconds: float
    max_seconds: float


@dataclass(frozen=True)
class AnswerEvaluationReport:
    """Aggregate results from :func:`evaluate_answers`.

    Frozen and content-free: ``per_case`` carries the ``case_id`` and the
    numeric scores only — never answer text, query text, or chunk content —
    so the whole report is safe to log or archive.

    Attributes:
        case_count: Number of cases evaluated.
        mean_grounding_ratio: Mean faithfulness grounding ratio.
        mean_citation_precision: Mean citation precision.
        mean_citation_recall: Mean citation recall.
        integrity_rate: Fraction of cases with zero fabricated citations.
            ``1.0`` is the only acceptable value in a healthy run.
        total_fabricated_citations: Sum of fabricated citations across all
            cases — a single non-zero value fails the integrity invariant.
        total_dangling_markers: Sum of dangling inline markers across all
            cases.
        latency: Aggregate :class:`LatencyStats`.
        per_case: Per-case ``(case_id, grounding_ratio, citation_precision,
            citation_recall, fabricated_count, dangling_marker_count,
            latency_seconds)`` tuples for drill-down.
    """

    case_count: int
    mean_grounding_ratio: float
    mean_citation_precision: float
    mean_citation_recall: float
    integrity_rate: float
    total_fabricated_citations: int
    total_dangling_markers: int
    latency: LatencyStats
    per_case: tuple[tuple[str, float, float, float, int, int, float], ...] = field(
        default_factory=tuple
    )


def score_faithfulness(result: GenerationResult) -> FaithfulnessScore:
    """Score how well ``result.answer`` is grounded in its citations.

    Pure and deterministic. The proxy splits the answer into sentences,
    counts the substantive ones, and credits a sentence as grounded when it
    carries an inline ``[N]`` marker whose ``N`` is a real citation
    ``display_index`` on the result. Markers pointing at a non-existent
    index are tallied separately as ``dangling_marker_count``.
    """
    valid_indices = {c.display_index for c in result.citations if c.display_index > 0}

    substantive = 0
    grounded = 0
    for sentence in _SENTENCE_SPLIT_RE.split(result.answer):
        if not _is_substantive(sentence):
            continue
        substantive += 1
        markers = {int(m) for m in _INLINE_MARKER_RE.findall(sentence)}
        if markers & valid_indices:
            grounded += 1

    all_markers = [int(m) for m in _INLINE_MARKER_RE.findall(result.answer)]
    dangling = sum(1 for m in all_markers if m not in valid_indices)

    ratio = grounded / substantive if substantive else 1.0
    return FaithfulnessScore(
        substantive_sentences=substantive,
        grounded_sentences=grounded,
        grounding_ratio=ratio,
        dangling_marker_count=dangling,
    )


def score_citations(case: EvaluationCase, result: GenerationResult) -> CitationScore:
    """Score the answer's citations against a gold set, with an integrity check.

    Integrity (anti-fabrication) is evaluated against the result itself:
    every citation must reference a chunk present in ``retrieved_chunks``.
    Accuracy (precision/recall) is evaluated against the case's expected
    chunk IDs / accession numbers / section paths, using the same permissive
    multi-dimension match as the retrieval harness.
    """
    cited = result.citations
    cited_count = len(cited)

    retrieved_ids = {chunk.chunk_id for chunk in result.retrieved_chunks if chunk.chunk_id}
    fabricated = sum(1 for c in cited if c.chunk_id not in retrieved_ids)

    expected_chunks = set(case.expected_chunk_ids)
    expected_accessions = set(case.expected_accession_numbers)
    expected_sections = set(case.expected_section_paths)

    matched_citations = 0
    matched_chunks: set[str] = set()
    matched_accessions: set[str] = set()
    matched_sections: set[str] = set()
    for c in cited:
        accession = c.filing_id.accession_number
        hit = False
        if c.chunk_id in expected_chunks:
            matched_chunks.add(c.chunk_id)
            hit = True
        if accession in expected_accessions:
            matched_accessions.add(accession)
            hit = True
        if c.section_path in expected_sections:
            matched_sections.add(c.section_path)
            hit = True
        if hit:
            matched_citations += 1

    matched_expected = len(matched_chunks) + len(matched_accessions) + len(matched_sections)
    precision = matched_citations / cited_count if cited_count else 0.0
    recall = matched_expected / case.expected_count if case.expected_count else 0.0

    return CitationScore(
        cited_count=cited_count,
        fabricated_count=fabricated,
        has_integrity=fabricated == 0,
        precision=precision,
        recall=recall,
    )


def evaluate_answers(
    cases: list[EvaluationCase],
    answer_fn: AnswerFn,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> AnswerEvaluationReport:
    """Run ``cases`` through ``answer_fn`` and aggregate answer-quality metrics.

    Args:
        cases: Evaluation cases. Empty list raises :class:`ValueError` so the
            report's averages are well-defined.
        answer_fn: Caller-supplied function mapping a query to a
            :class:`GenerationResult`. Wrap :meth:`RAGOrchestrator.generate`
            in a lambda that pins provider / model / mode.
        clock: Monotonic clock used to time each ``answer_fn`` call.
            Injectable so tests can assert deterministic latency stats; do
            not inline :func:`time.monotonic` at the call site (mirrors the
            ``core.resilience`` clock-injection convention).

    Returns:
        An :class:`AnswerEvaluationReport`. The ``integrity_rate`` and
        ``total_fabricated_citations`` fields surface the load-bearing
        anti-fabrication invariant; a healthy run reports ``1.0`` and ``0``.
    """
    if not cases:
        raise ValueError("evaluate_answers requires at least one case.")

    per_case: list[tuple[str, float, float, float, int, int, float]] = []
    latencies: list[float] = []
    grounding_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    integrity_count = 0
    total_fabricated = 0
    total_dangling = 0

    for case in cases:
        start = clock()
        result = answer_fn(case.query)
        latency = clock() - start
        latencies.append(latency)

        faith = score_faithfulness(result)
        cit = score_citations(case, result)

        grounding_sum += faith.grounding_ratio
        precision_sum += cit.precision
        recall_sum += cit.recall
        integrity_count += 1 if cit.has_integrity else 0
        total_fabricated += cit.fabricated_count
        total_dangling += faith.dangling_marker_count

        per_case.append(
            (
                case.case_id,
                faith.grounding_ratio,
                cit.precision,
                cit.recall,
                cit.fabricated_count,
                faith.dangling_marker_count,
                latency,
            )
        )

    n = len(cases)
    return AnswerEvaluationReport(
        case_count=n,
        mean_grounding_ratio=grounding_sum / n,
        mean_citation_precision=precision_sum / n,
        mean_citation_recall=recall_sum / n,
        integrity_rate=integrity_count / n,
        total_fabricated_citations=total_fabricated,
        total_dangling_markers=total_dangling,
        latency=_latency_stats(latencies),
        per_case=tuple(per_case),
    )


def _is_substantive(sentence: str) -> bool:
    """True when ``sentence`` looks like a factual claim worth grounding."""
    stripped = sentence.strip()
    return len(stripped) >= _MIN_SUBSTANTIVE_CHARS and _LETTER_RE.search(stripped) is not None


def _latency_stats(latencies: list[float]) -> LatencyStats:
    """Aggregate per-call latencies into mean / p50 / p95 / max (nearest-rank)."""
    count = len(latencies)
    if count == 0:
        return LatencyStats(
            count=0,
            mean_seconds=0.0,
            p50_seconds=0.0,
            p95_seconds=0.0,
            max_seconds=0.0,
        )
    return LatencyStats(
        count=count,
        mean_seconds=sum(latencies) / count,
        p50_seconds=_percentile(latencies, 50),
        p95_seconds=_percentile(latencies, 95),
        max_seconds=max(latencies),
    )


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile — deterministic and interpolation-free.

    Nearest-rank (rather than linear interpolation) keeps the result equal
    to an actually-observed latency, which is both easier to reason about
    and trivially assertable in tests.
    """
    ordered = sorted(values)
    rank = max(1, _ceil_int(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def _ceil_int(value: float) -> int:
    """Ceiling without importing ``math`` — keeps the dependency surface tiny."""
    as_int = int(value)
    return as_int if as_int == value else as_int + 1
