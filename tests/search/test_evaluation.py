"""Tests for :mod:`sec_generative_search.search.evaluation`.

Exercises the retrieval evaluation harness against a mocked retrieval
function. No real ChromaDB or embedder instance is required because the
harness is pure scoring and JSON I/O.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sec_generative_search.core.types import ContentType, RetrievalResult, SearchResult
from sec_generative_search.search.evaluation import (
    EvaluationCase,
    evaluate_retrieval,
    load_cases_from_json,
)

# ---------------------------------------------------------------------------
# Sample-data builders
# ---------------------------------------------------------------------------


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "retrieval_eval_cases.json"


def _make_result(
    *,
    chunk_id: str,
    accession_number: str = "0000000000-00-000000",
    path: str = "Part I > Item 1A > Risk Factors",
) -> RetrievalResult:
    base = SearchResult(
        content="example",
        path=path,
        content_type=ContentType.TEXT,
        ticker="AAPL",
        form_type="10-K",
        similarity=0.5,
        filing_date="2023-11-03",
        accession_number=accession_number,
        chunk_id=chunk_id,
    )
    return RetrievalResult.from_search_result(base)


# ---------------------------------------------------------------------------
# EvaluationCase invariants
# ---------------------------------------------------------------------------


class TestEvaluationCaseValidation:
    def test_empty_case_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="case_id"):
            EvaluationCase(case_id="", query="q", expected_chunk_ids=("a",))

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(ValueError, match="query"):
            EvaluationCase(case_id="c1", query="  ", expected_chunk_ids=("a",))

    def test_no_expectations_rejected(self) -> None:
        with pytest.raises(ValueError, match="expected_"):
            EvaluationCase(case_id="c1", query="q")

    def test_expected_count_sums_dimensions(self) -> None:
        case = EvaluationCase(
            case_id="c1",
            query="q",
            expected_chunk_ids=("a", "b"),
            expected_accession_numbers=("acc-1",),
            expected_section_paths=("sec-1", "sec-2"),
        )
        assert case.expected_count == 5


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestEvaluateRetrieval:
    def test_perfect_recall_and_precision(self) -> None:
        case = EvaluationCase(case_id="c1", query="q", expected_chunk_ids=("a", "b"))

        def fake_retrieve(_q: str, _k: int) -> list[RetrievalResult]:
            return [_make_result(chunk_id="a"), _make_result(chunk_id="b")]

        report = evaluate_retrieval([case], fake_retrieve, top_k=2)
        assert report.precision_at_k == pytest.approx(1.0)
        assert report.recall_at_k == pytest.approx(1.0)
        assert report.case_count == 1

    def test_partial_recall(self) -> None:
        case = EvaluationCase(case_id="c1", query="q", expected_chunk_ids=("a", "b", "c"))

        def fake_retrieve(_q: str, _k: int) -> list[RetrievalResult]:
            return [_make_result(chunk_id="a"), _make_result(chunk_id="b")]

        report = evaluate_retrieval([case], fake_retrieve, top_k=2)
        # 2 / 2 precision, 2 / 3 recall.
        assert report.precision_at_k == pytest.approx(1.0)
        assert report.recall_at_k == pytest.approx(2 / 3)

    def test_precision_uses_top_k_denominator(self) -> None:
        # Returning fewer than top_k results shrinks precision.
        case = EvaluationCase(case_id="c1", query="q", expected_chunk_ids=("a",))

        def fake_retrieve(_q: str, _k: int) -> list[RetrievalResult]:
            return [_make_result(chunk_id="a")]

        report = evaluate_retrieval([case], fake_retrieve, top_k=5)
        assert report.precision_at_k == pytest.approx(0.2)
        assert report.recall_at_k == pytest.approx(1.0)

    def test_section_path_match(self) -> None:
        path = "Part I > Item 1 > Business > Customers"
        case = EvaluationCase(case_id="c1", query="q", expected_section_paths=(path,))

        def fake_retrieve(_q: str, _k: int) -> list[RetrievalResult]:
            return [_make_result(chunk_id="x", path=path)]

        report = evaluate_retrieval([case], fake_retrieve, top_k=1)
        assert report.recall_at_k == pytest.approx(1.0)

    def test_accession_number_match(self) -> None:
        case = EvaluationCase(case_id="c1", query="q", expected_accession_numbers=("acc-1",))

        def fake_retrieve(_q: str, _k: int) -> list[RetrievalResult]:
            return [_make_result(chunk_id="x", accession_number="acc-1")]

        report = evaluate_retrieval([case], fake_retrieve, top_k=1)
        assert report.recall_at_k == pytest.approx(1.0)

    def test_zero_cases_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            evaluate_retrieval([], lambda _q, _k: [], top_k=1)

    def test_invalid_top_k_rejected(self) -> None:
        case = EvaluationCase(case_id="c1", query="q", expected_chunk_ids=("a",))
        with pytest.raises(ValueError, match="top_k"):
            evaluate_retrieval([case], lambda _q, _k: [], top_k=0)

    def test_per_case_breakdown(self) -> None:
        cases = [
            EvaluationCase(case_id="c1", query="q1", expected_chunk_ids=("a",)),
            EvaluationCase(case_id="c2", query="q2", expected_chunk_ids=("b",)),
        ]

        def fake_retrieve(query: str, _k: int) -> list[RetrievalResult]:
            return [_make_result(chunk_id="a")] if query == "q1" else []

        report = evaluate_retrieval(cases, fake_retrieve, top_k=1)
        ids = [row[0] for row in report.per_case]
        assert ids == ["c1", "c2"]
        assert report.per_case[0][1] == pytest.approx(1.0)  # c1 precision
        assert report.per_case[1][1] == pytest.approx(0.0)  # c2 precision


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------


class TestLoadCasesFromJson:
    def test_loads_fixture(self) -> None:
        cases = load_cases_from_json(_FIXTURE_PATH)
        assert len(cases) == 3
        case_ids = {c.case_id for c in cases}
        assert "q-revenue-concentration-001" in case_ids
        assert "q-supply-chain-risk-002" in case_ids
        assert "q-cash-flow-summary-003" in case_ids

    def test_unknown_field_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(
            '[{"case_id": "c1", "query": "q", "expected_chunk_ids": ["a"], "unexpected_field": 42}]'
        )
        with pytest.raises(ValueError, match="unknown field"):
            load_cases_from_json(path)

    def test_top_level_object_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"case_id": "c1"}')
        with pytest.raises(ValueError, match="JSON array"):
            load_cases_from_json(path)

    def test_non_object_entry_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('["not-an-object"]')
        with pytest.raises(ValueError, match="JSON object"):
            load_cases_from_json(path)
