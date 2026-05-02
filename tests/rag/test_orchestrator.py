"""Tests for :mod:`sec_generative_search.rag.orchestrator`.

Covers the full pipeline against fake retrieval and LLM doubles —
construction, single-query happy path, refusal short-circuit, streaming,
comparative fan-out, conversation history, structured-output preference,
multilingual handling, and traceability.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from sec_generative_search.core.types import (
    ConversationTurn,
    GenerationResult,
    ProviderCapability,
    TokenUsage,
)
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.orchestrator import RAGOrchestrator
from sec_generative_search.rag.query_understanding import QueryPlan
from tests.rag.conftest import FakeLLMProvider, FakeRetrievalService


def _build_orchestrator(
    *,
    retrieval: FakeRetrievalService,
    llm: FakeLLMProvider,
) -> RAGOrchestrator:
    """Construct an orchestrator with a stable token counter."""

    def counter(text: str) -> int:
        return max(1, len(text) // 4)

    return RAGOrchestrator(retrieval=retrieval, llm=llm, token_counter=counter)


class TestSingleQueryHappyPath:
    def test_returns_generation_result_with_inline_citations(
        self, fake_retrieval, fake_llm, sample_chunks
    ) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="What about revenue?")

        result = orch.generate(plan)

        assert isinstance(result, GenerationResult)
        assert result.provider == "fake-llm"
        assert result.prompt_version == "v1.0.0"
        assert result.streamed is False
        # Reply includes [1] [2] markers; both present in chunks.
        assert len(result.citations) == 2
        assert {c.chunk_id for c in result.citations} == {
            sample_chunks[0].chunk_id,
            sample_chunks[1].chunk_id,
        }
        # Retrieved chunks are recorded in full (superset of citations).
        assert len(result.retrieved_chunks) == 3
        # Token usage recorded.
        assert result.token_usage.total_tokens == 20

    def test_passes_query_en_to_retrieval(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="AAPL gelirleri nasil?",
            detected_language="tr",
            query_en="How is AAPL revenue?",
        )
        orch.generate(plan)
        # Retrieval saw the English rendering, not the raw Turkish query.
        assert fake_retrieval.calls[0]["query"] == "How is AAPL revenue?"

    def test_user_prompt_carries_raw_query(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="AAPL gelirleri nasil?",
            detected_language="tr",
            query_en="How is AAPL revenue?",
        )
        orch.generate(plan)
        # The model still sees the user's verbatim question — keeps the
        # answer faithful to the user's wording.
        assert "AAPL gelirleri nasil?" in fake_llm.last_request.prompt
        # And the system prompt asks for the user's language.
        assert "tr" in fake_llm.last_request.system

    def test_filters_from_plan_propagate_to_retrieval(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="Q",
            tickers=["AAPL"],
            form_types=["10-K"],
            date_range=("2023-01-01", "2023-12-31"),
        )
        orch.generate(plan)
        call = fake_retrieval.calls[0]
        assert call["ticker"] == "AAPL"
        assert call["form_type"] == "10-K"
        assert call["start_date"] == "2023-01-01"
        assert call["end_date"] == "2023-12-31"

    def test_extra_filters_override_plan(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q", tickers=["AAPL"])
        orch.generate(plan, extra_filters={"ticker": "MSFT"})
        assert fake_retrieval.calls[0]["ticker"] == "MSFT"


class TestRefusalPath:
    def test_short_circuits_when_no_chunks(self, fake_llm) -> None:
        empty_retrieval = FakeRetrievalService(results=[])
        orch = _build_orchestrator(retrieval=empty_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q")

        result = orch.generate(plan)

        assert "cannot answer" in result.answer.lower()
        assert result.citations == []
        assert result.retrieved_chunks == []
        # LLM was never called.
        assert fake_llm.call_count == 0
        # Token usage stays at zero — we did not pretend to consume any.
        assert result.token_usage.total_tokens == 0

    def test_refusal_disabled_calls_llm_with_empty_context(self, fake_llm) -> None:
        # Pydantic-settings v2 evaluates nested-settings defaults at class
        # definition time; ``reload_settings()`` does not flow env-var
        # overrides into nested ``settings.rag`` (see
        # tests/config/test_settings.py for the direct-RAGSettings()
        # pattern).  Patch the orchestrator's captured settings instead.
        empty_retrieval = FakeRetrievalService(results=[])
        orch = _build_orchestrator(retrieval=empty_retrieval, llm=fake_llm)
        orch._rag_settings.refusal_enabled = False  # type: ignore[attr-defined]

        plan = QueryPlan(raw_query="Q")
        result = orch.generate(plan)
        # Provider was called.
        assert fake_llm.call_count == 1
        # And the request shows an empty context block.
        assert "(no chunks retrieved)" in fake_llm.last_request.prompt
        # Result is the model's reply, not a refusal.
        assert "cannot answer" not in result.answer.lower()


class TestStreaming:
    def test_yields_deltas_then_final(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q")

        events = list(orch.generate_stream(plan))

        deltas = [e for e in events if e.delta is not None]
        finals = [e for e in events if e.final is not None]
        assert len(deltas) >= 1
        assert len(finals) == 1
        # Deltas must reconstruct the model's reply.
        assert "".join(d.delta or "" for d in deltas) == fake_llm.reply

    def test_final_carries_assembled_result(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q")

        events = list(orch.generate_stream(plan))
        final = next(e.final for e in events if e.final is not None)
        assert final.streamed is True
        assert len(final.citations) == 2  # [1] and [2] in the canned reply
        assert final.token_usage.total_tokens > 0

    def test_streams_refusal(self, fake_llm) -> None:
        empty_retrieval = FakeRetrievalService(results=[])
        orch = _build_orchestrator(retrieval=empty_retrieval, llm=fake_llm)

        events = list(orch.generate_stream(QueryPlan(raw_query="Q")))
        assert any(e.delta is not None and "cannot answer" in e.delta.lower() for e in events)
        final = next(e.final for e in events if e.final is not None)
        assert final.citations == []
        assert fake_llm.call_count == 0


class TestComparativeFanOut:
    def test_runs_one_retrieval_per_ticker(self, sample_chunks, fake_llm) -> None:
        # Configure per-call results so we can see fan-out happen.
        retrieval = FakeRetrievalService(
            per_call_results=[
                [sample_chunks[0]],
                [sample_chunks[1]],
            ]
        )
        orch = _build_orchestrator(retrieval=retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="Compare AAPL vs MSFT revenue",
            tickers=["AAPL", "MSFT"],
            suggested_answer_mode=AnswerMode.COMPARATIVE,
        )

        orch.generate(plan)

        assert len(retrieval.calls) == 2
        tickers_seen = {call["ticker"] for call in retrieval.calls}
        assert tickers_seen == {"AAPL", "MSFT"}

    def test_dedupes_overlapping_results(self, sample_chunks, fake_llm) -> None:
        # Same chunk returned twice — must collapse to one.
        retrieval = FakeRetrievalService(
            per_call_results=[
                [sample_chunks[0]],
                [sample_chunks[0]],
            ]
        )
        orch = _build_orchestrator(retrieval=retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="Compare",
            tickers=["AAPL", "MSFT"],
            suggested_answer_mode=AnswerMode.COMPARATIVE,
        )
        result = orch.generate(plan)
        assert len(result.retrieved_chunks) == 1

    def test_single_ticker_skips_fan_out(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="One ticker",
            tickers=["AAPL"],
            suggested_answer_mode=AnswerMode.COMPARATIVE,
        )
        orch.generate(plan)
        assert len(fake_retrieval.calls) == 1


class TestConversationHistory:
    def _make_turn(self, q: str, a: str) -> ConversationTurn:
        return ConversationTurn(
            query=q,
            retrieval_results=[],
            generation_result=GenerationResult(
                answer=a,
                provider="fake-llm",
                model="fake-model",
                prompt_version="v1.0.0",
                token_usage=TokenUsage(input_tokens=5, output_tokens=3),
            ),
            timestamp=datetime(2026, 5, 1, 12, 0, 0),
        )

    def test_history_disabled_by_default(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        history = [self._make_turn("Prior Q?", "Prior A.")]
        orch.generate(QueryPlan(raw_query="Q"), history=history)
        assert "Prior Q?" not in fake_llm.last_request.prompt

    def test_history_enabled_renders_q_a_pairs(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        orch._rag_settings.chat_history_enabled = True  # type: ignore[attr-defined]

        history = [self._make_turn("Prior Q?", "Prior A.")]
        orch.generate(QueryPlan(raw_query="Q"), history=history)
        assert "Prior Q?" in fake_llm.last_request.prompt
        assert "Prior A." in fake_llm.last_request.prompt

    @pytest.mark.security
    def test_history_does_not_leak_prior_chunks(self, fake_retrieval, fake_llm, make_chunk) -> None:
        """Prior chunks must never be carried into a follow-up prompt."""
        # Build a fresh chunk distinct from anything the current retrieval
        # returns — using ``sample_chunks[0]`` would let the sentinel leak
        # via the *current* turn's retrieval, masking the test.
        prior_chunk = make_chunk(
            index=99,
            ticker="TSLA",
            content="TIER1-CHUNK-SENTINEL-XYZ",
        )
        prior = ConversationTurn(
            query="Prior Q?",
            retrieval_results=[prior_chunk],
            generation_result=GenerationResult(
                answer="Prior A.",
                provider="fake-llm",
                model="fake-model",
                prompt_version="v1.0.0",
            ),
            timestamp=datetime(2026, 5, 1, 12, 0, 0),
        )

        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        orch._rag_settings.chat_history_enabled = True  # type: ignore[attr-defined]
        orch.generate(QueryPlan(raw_query="Q"), history=[prior])
        assert "TIER1-CHUNK-SENTINEL-XYZ" not in fake_llm.last_request.prompt


class TestStructuredOutputPreference:
    def test_sets_response_format_when_preferred(
        self, fake_retrieval, fake_llm, sample_chunks
    ) -> None:
        # Reply must be a JSON envelope so the JSON path succeeds.
        fake_llm.reply = json.dumps(
            {
                "answer": "Revenue grew.",
                "cited_chunk_ids": [sample_chunks[0].chunk_id],
            }
        )
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q")
        result = orch.generate(plan, prefer_structured_output=True)
        assert fake_llm.last_request.response_format == "json"
        assert fake_llm.last_request.response_schema is not None
        assert result.answer == "Revenue grew."
        assert result.citations[0].chunk_id == sample_chunks[0].chunk_id


class TestOutputLanguageOverride:
    def test_auto_uses_detected_language(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q", detected_language="de")
        orch.generate(plan)
        # German appears in the system prompt.
        assert "de" in fake_llm.last_request.system

    def test_explicit_override_locks_language(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        orch._rag_settings.output_language = "en"  # type: ignore[attr-defined]

        plan = QueryPlan(raw_query="Q", detected_language="tr")
        orch.generate(plan)
        # Operator override pins English regardless of detection.
        sys_prompt = fake_llm.last_request.system
        # The directive line includes the chosen language; we want
        # English explicitly mentioned, and we want Turkish NOT to be
        # the one passed.
        assert " en" in sys_prompt or "in en" in sys_prompt


class TestTraceability:
    def test_records_provider_model_prompt_version(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q")
        result = orch.generate(plan, model="explicit-model")
        assert result.provider == "fake-llm"
        # Provider echoes the model slug back via GenerationResponse.model.
        assert result.model == "explicit-model"
        assert result.prompt_version == "v1.0.0"
        assert result.latency_seconds >= 0.0
        # retrieved_chunks ⊇ citations — diagnoses overshoot vs. ignore-context.
        cited_ids = {c.chunk_id for c in result.citations}
        retrieved_ids = {c.chunk_id for c in result.retrieved_chunks}
        assert cited_ids.issubset(retrieved_ids)


class TestBudgetIntegration:
    def test_passes_chunks_token_budget_to_retrieval(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q")
        orch.generate(plan, max_output_tokens=500)
        budget_arg = fake_retrieval.calls[0]["context_token_budget"]
        # Capability says window=8000; default history fraction = 1200;
        # answer = 500; plus a small system slice — chunks must be the
        # remainder, well above zero.
        assert budget_arg > 1000

    def test_unknown_window_falls_back_to_settings_budget(self, fake_retrieval) -> None:
        # LLM that advertises an unknown window.
        unknown_window_llm = FakeLLMProvider(
            reply="Reply.",
            capability=ProviderCapability(chat=True, streaming=True, context_window_tokens=0),
        )
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=unknown_window_llm)
        orch._rag_settings.context_token_budget = 1234  # type: ignore[attr-defined]
        orch.generate(QueryPlan(raw_query="Q"))
        assert fake_retrieval.calls[0]["context_token_budget"] == 1234
