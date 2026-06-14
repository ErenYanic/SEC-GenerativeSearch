"""Tests for :mod:`sec_generative_search.rag.evaluation`.

Exercises the offline answer-quality harness — faithfulness proxy,
citation accuracy + integrity, and latency aggregation — entirely with
synthetic :class:`GenerationResult` objects. The harness is pure scoring:
no ChromaDB, no LLM, no network, no credential.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sec_generative_search.core.types import (
    Citation,
    ContentType,
    GenerationResult,
    RetrievalResult,
)
from sec_generative_search.rag.evaluation import (
    EvaluationCase,
    LatencyStats,
    evaluate_answers,
    load_cases_from_json,
    score_citations,
    score_faithfulness,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "retrieval_eval_cases.json"

_ACCESSION = "0000320193-23-000077"
_SECTION = "Part I > Item 1A > Risk Factors"


# ---------------------------------------------------------------------------
# Sample-data builders
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    index: int,
    accession_number: str = _ACCESSION,
    path: str = _SECTION,
    content: str = "Risk factor text.",
) -> RetrievalResult:
    chunk_id = f"AAPL_10-K_2023-11-03_{index:03d}"
    return RetrievalResult(
        content=content,
        path=path,
        content_type=ContentType.TEXT,
        ticker="AAPL",
        form_type="10-K",
        similarity=0.9,
        filing_date="2023-11-03",
        accession_number=accession_number,
        chunk_id=chunk_id,
        token_count=10,
    )


def _citation_from(chunk: RetrievalResult, display_index: int) -> Citation:
    return chunk.to_citation(display_index)


def _make_result(
    *,
    answer: str,
    citations: list[Citation],
    retrieved: list[RetrievalResult],
) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        provider="fake",
        model="fake-model",
        prompt_version="v-test",
        citations=citations,
        retrieved_chunks=retrieved,
    )


# ---------------------------------------------------------------------------
# Faithfulness proxy
# ---------------------------------------------------------------------------


class TestFaithfulness:
    def test_fully_grounded_answer_scores_one(self) -> None:
        c1, c2 = _make_chunk(index=1), _make_chunk(index=2)
        result = _make_result(
            answer="Revenue grew eight percent [1]. Margin compressed in the third quarter [2].",
            citations=[_citation_from(c1, 1), _citation_from(c2, 2)],
            retrieved=[c1, c2],
        )
        score = score_faithfulness(result)
        assert score.substantive_sentences == 2
        assert score.grounded_sentences == 2
        assert score.grounding_ratio == 1.0
        assert score.dangling_marker_count == 0

    def test_partial_grounding_ratio(self) -> None:
        c1 = _make_chunk(index=1)
        result = _make_result(
            answer="Revenue grew eight percent [1]. Margins are under pressure this year.",
            citations=[_citation_from(c1, 1)],
            retrieved=[c1],
        )
        score = score_faithfulness(result)
        assert score.substantive_sentences == 2
        assert score.grounded_sentences == 1
        assert score.grounding_ratio == 0.5

    def test_dangling_marker_counted(self) -> None:
        c1 = _make_chunk(index=1)
        # Answer cites [3] but only display_index 1 exists.
        result = _make_result(
            answer="Liquidity remained strong throughout the year [3].",
            citations=[_citation_from(c1, 1)],
            retrieved=[c1],
        )
        score = score_faithfulness(result)
        assert score.dangling_marker_count == 1
        assert score.grounded_sentences == 0
        assert score.grounding_ratio == 0.0

    def test_refusal_answer_is_zero_grounded_not_an_error(self) -> None:
        # A refusal makes substantive claims but carries no markers — the
        # documented proxy limitation. It must score, never raise.
        result = _make_result(
            answer="I cannot answer that from the provided filings.",
            citations=[],
            retrieved=[],
        )
        score = score_faithfulness(result)
        assert score.substantive_sentences == 1
        assert score.grounding_ratio == 0.0

    def test_empty_answer_is_vacuously_grounded(self) -> None:
        result = _make_result(answer="", citations=[], retrieved=[])
        score = score_faithfulness(result)
        assert score.substantive_sentences == 0
        assert score.grounding_ratio == 1.0  # no division by zero

    def test_zero_display_index_is_not_a_valid_target(self) -> None:
        # display_index 0 means "unassigned" — a [0] marker must not count
        # as grounded and the citation must not seed the valid-index set.
        c1 = _make_chunk(index=1)
        result = _make_result(
            answer="Operating cash flow rose materially this year [0].",
            citations=[_citation_from(c1, 0)],
            retrieved=[c1],
        )
        score = score_faithfulness(result)
        assert score.grounded_sentences == 0
        assert score.dangling_marker_count == 1


# ---------------------------------------------------------------------------
# Citation accuracy + integrity
# ---------------------------------------------------------------------------


class TestCitationAccuracy:
    def test_precision_recall_against_section_gold(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            query="What risks are disclosed?",
            expected_section_paths=(_SECTION,),
        )
        c1 = _make_chunk(index=1, path=_SECTION)
        c2 = _make_chunk(index=2, path="Part II > Item 7 > MD&A")
        result = _make_result(
            answer="Risk text [1]. Other text [2].",
            citations=[_citation_from(c1, 1), _citation_from(c2, 2)],
            retrieved=[c1, c2],
        )
        score = score_citations(case, result)
        assert score.cited_count == 2
        assert score.precision == 0.5  # 1 of 2 cited chunks in the gold section
        assert score.recall == 1.0  # the one expected section was covered

    def test_accession_dimension_match(self) -> None:
        case = EvaluationCase(
            case_id="c2",
            query="Cash flow?",
            expected_accession_numbers=(_ACCESSION,),
        )
        c1 = _make_chunk(index=1)
        result = _make_result(
            answer="Cash flow detail [1].",
            citations=[_citation_from(c1, 1)],
            retrieved=[c1],
        )
        score = score_citations(case, result)
        assert score.precision == 1.0
        assert score.recall == 1.0

    def test_no_citations_yields_zero_precision(self) -> None:
        case = EvaluationCase(case_id="c3", query="q", expected_section_paths=(_SECTION,))
        result = _make_result(answer="No sources.", citations=[], retrieved=[])
        score = score_citations(case, result)
        assert score.cited_count == 0
        assert score.precision == 0.0
        assert score.recall == 0.0
        assert score.has_integrity is True  # nothing cited → nothing fabricated

    @pytest.mark.security
    def test_fabricated_citation_breaks_integrity(self) -> None:
        # A citation whose chunk_id was never retrieved is the anti-
        # fabrication invariant the orchestrator must uphold; the scorer
        # MUST flag it.
        case = EvaluationCase(case_id="c4", query="q", expected_section_paths=(_SECTION,))
        retrieved = _make_chunk(index=1)
        ghost = _make_chunk(index=99)  # not in retrieved set
        result = _make_result(
            answer="Claim [1].",
            citations=[_citation_from(ghost, 1)],
            retrieved=[retrieved],
        )
        score = score_citations(case, result)
        assert score.fabricated_count == 1
        assert score.has_integrity is False

    @pytest.mark.security
    def test_all_citations_grounded_in_retrieved_keeps_integrity(self) -> None:
        case = EvaluationCase(case_id="c5", query="q", expected_section_paths=(_SECTION,))
        c1, c2 = _make_chunk(index=1), _make_chunk(index=2)
        result = _make_result(
            answer="A [1]. B [2].",
            citations=[_citation_from(c1, 1), _citation_from(c2, 2)],
            retrieved=[c1, c2],
        )
        score = score_citations(case, result)
        assert score.fabricated_count == 0
        assert score.has_integrity is True


# ---------------------------------------------------------------------------
# evaluate_answers — aggregation + latency
# ---------------------------------------------------------------------------


class _ScriptedClock:
    """Returns successive scripted timestamps so latency is deterministic."""

    def __init__(self, ticks: list[float]) -> None:
        self._ticks = ticks
        self._i = 0

    def __call__(self) -> float:
        value = self._ticks[self._i]
        self._i += 1
        return value


class TestEvaluateAnswers:
    def _cases(self) -> list[EvaluationCase]:
        return [
            EvaluationCase(case_id="a", query="qa", expected_section_paths=(_SECTION,)),
            EvaluationCase(case_id="b", query="qb", expected_section_paths=(_SECTION,)),
            EvaluationCase(case_id="c", query="qc", expected_section_paths=(_SECTION,)),
        ]

    def _answer_for(self, query: str) -> GenerationResult:
        c1 = _make_chunk(index=1, path=_SECTION)
        return _make_result(
            answer="Grounded claim [1].",
            citations=[_citation_from(c1, 1)],
            retrieved=[c1],
        )

    def test_empty_cases_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one case"):
            evaluate_answers([], self._answer_for)

    def test_aggregate_metrics_and_latency(self) -> None:
        # start/end pairs → latencies 0.1, 0.2, 0.3
        clock = _ScriptedClock([0.0, 0.1, 1.0, 1.2, 2.0, 2.3])
        report = evaluate_answers(self._cases(), self._answer_for, clock=clock)

        assert report.case_count == 3
        assert report.mean_grounding_ratio == 1.0
        assert report.mean_citation_precision == 1.0
        assert report.mean_citation_recall == 1.0
        assert report.integrity_rate == 1.0
        assert report.total_fabricated_citations == 0
        assert report.total_dangling_markers == 0

        lat: LatencyStats = report.latency
        assert lat.count == 3
        assert lat.mean_seconds == pytest.approx(0.2)
        assert lat.p50_seconds == pytest.approx(0.2)
        assert lat.p95_seconds == pytest.approx(0.3)
        assert lat.max_seconds == pytest.approx(0.3)

    def test_integrity_rate_reflects_fabrication(self) -> None:
        cases = self._cases()[:2]
        good_c = _make_chunk(index=1)
        ghost = _make_chunk(index=99)

        def answer_for(query: str) -> GenerationResult:
            if query == "qa":
                return _make_result(
                    answer="Good [1].",
                    citations=[_citation_from(good_c, 1)],
                    retrieved=[good_c],
                )
            return _make_result(
                answer="Fabricated [1].",
                citations=[_citation_from(ghost, 1)],
                retrieved=[good_c],
            )

        report = evaluate_answers(cases, answer_for)
        assert report.integrity_rate == 0.5
        assert report.total_fabricated_citations == 1

    @pytest.mark.security
    def test_report_is_content_free(self) -> None:
        # The report (and its per_case rows) must carry case_id + numbers
        # only — never the answer text, the query, or chunk content.
        secret_answer = "CONFIDENTIAL revenue figure 4.2 billion [1]."
        secret_query = "SENSITIVE-QUERY-STRING"

        def answer_for(query: str) -> GenerationResult:
            c1 = _make_chunk(index=1, content="CHUNK-BODY-TEXT")
            return _make_result(
                answer=secret_answer,
                citations=[_citation_from(c1, 1)],
                retrieved=[c1],
            )

        case = EvaluationCase(
            case_id="content-free",
            query=secret_query,
            expected_section_paths=(_SECTION,),
        )
        report = evaluate_answers([case], answer_for)
        blob = repr(report)
        assert "case_id" not in blob  # tuple positions, not dict keys
        assert "content-free" in blob  # the case_id is allowed
        assert "CONFIDENTIAL" not in blob
        assert secret_query not in blob
        assert "CHUNK-BODY-TEXT" not in blob


# ---------------------------------------------------------------------------
# Shared gold-set format — reuses the retrieval case file
# ---------------------------------------------------------------------------


class TestSharedCaseFormat:
    def test_loads_retrieval_fixture_for_answer_eval(self) -> None:
        # The same case file drives both harnesses — load_cases_from_json is
        # re-exported from rag.evaluation.
        cases = load_cases_from_json(_FIXTURE_PATH)
        assert len(cases) == 3
        assert all(isinstance(c, EvaluationCase) for c in cases)
