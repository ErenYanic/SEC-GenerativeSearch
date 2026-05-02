"""Tests for :mod:`sec_generative_search.rag.query_understanding`."""

from __future__ import annotations

import json

import pytest

from sec_generative_search.core.exceptions import GenerationError
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.query_understanding import (
    QUERY_PLAN_JSON_SCHEMA,
    QueryPlan,
    parse_query_plan,
    understand_query,
)


class TestQueryPlanDataclass:
    def test_query_en_defaults_to_raw_query(self) -> None:
        plan = QueryPlan(raw_query="What about AAPL?")
        assert plan.query_en == "What about AAPL?"

    def test_explicit_query_en_preserved(self) -> None:
        plan = QueryPlan(raw_query="AAPL gelirleri?", query_en="AAPL revenue?")
        assert plan.query_en == "AAPL revenue?"
        assert plan.detected_language == "en"  # default until populated

    def test_default_mode_is_concise(self) -> None:
        plan = QueryPlan(raw_query="X")
        assert plan.suggested_answer_mode is AnswerMode.CONCISE


class TestParseQueryPlan:
    def test_parses_minimal_envelope(self) -> None:
        payload = json.dumps(
            {
                "raw_query": "Q",
                "detected_language": "en",
                "query_en": "Q",
                "tickers": ["AAPL"],
                "form_types": ["10-K"],
                "date_range": ["2023-01-01", "2023-12-31"],
                "intent": "Get risk factors.",
                "suggested_answer_mode": "concise",
            }
        )
        plan = parse_query_plan(payload, raw_query="Q")
        assert plan.tickers == ["AAPL"]
        assert plan.form_types == ["10-K"]
        assert plan.date_range == ("2023-01-01", "2023-12-31")
        assert plan.intent == "Get risk factors."
        assert plan.suggested_answer_mode is AnswerMode.CONCISE

    def test_handles_lowercase_tickers_and_punctuation(self) -> None:
        payload = json.dumps(
            {
                "raw_query": "Q",
                "detected_language": "en",
                "query_en": "Q",
                "tickers": ["(aapl)", "msft"],
                "form_types": [],
                "date_range": None,
                "intent": "",
                "suggested_answer_mode": "concise",
            }
        )
        plan = parse_query_plan(payload, raw_query="Q")
        assert plan.tickers == ["AAPL", "MSFT"]

    def test_unknown_mode_falls_back_to_concise(self) -> None:
        payload = json.dumps(
            {
                "raw_query": "Q",
                "detected_language": "en",
                "query_en": "Q",
                "tickers": [],
                "form_types": [],
                "date_range": None,
                "intent": "",
                "suggested_answer_mode": "garbage_mode",
            }
        )
        plan = parse_query_plan(payload, raw_query="Q")
        assert plan.suggested_answer_mode is AnswerMode.CONCISE

    def test_isolates_json_inside_prose(self) -> None:
        wrapped = (
            "Here you go: "
            + json.dumps(
                {
                    "raw_query": "Q",
                    "detected_language": "tr",
                    "query_en": "Q in english",
                    "tickers": [],
                    "form_types": [],
                    "date_range": None,
                    "intent": "",
                    "suggested_answer_mode": "concise",
                }
            )
            + " — hope this helps."
        )
        plan = parse_query_plan(wrapped, raw_query="Q")
        assert plan.detected_language == "tr"
        assert plan.query_en == "Q in english"

    def test_raises_on_unparseable(self) -> None:
        with pytest.raises(GenerationError):
            parse_query_plan("no json here", raw_query="Q")

    def test_raises_on_non_object_root(self) -> None:
        with pytest.raises(GenerationError):
            parse_query_plan('["array", "root"]', raw_query="Q")

    def test_query_en_falls_back_to_raw_when_blank(self) -> None:
        payload = json.dumps(
            {
                "raw_query": "AAPL gelirleri?",
                "detected_language": "tr",
                "query_en": "",
                "tickers": ["AAPL"],
                "form_types": [],
                "date_range": None,
                "intent": "",
                "suggested_answer_mode": "concise",
            }
        )
        plan = parse_query_plan(payload, raw_query="AAPL gelirleri?")
        # When the model omits query_en we conservatively use raw_query
        # so the embedding step does not blow up.
        assert plan.query_en == "AAPL gelirleri?"


class TestUnderstandQuery:
    def test_uses_structured_output_when_supported(self, fake_llm) -> None:
        fake_llm.reply = json.dumps(
            {
                "raw_query": "Q",
                "detected_language": "en",
                "query_en": "Q",
                "tickers": ["AAPL"],
                "form_types": [],
                "date_range": None,
                "intent": "",
                "suggested_answer_mode": "concise",
            }
        )
        plan = understand_query(
            "Q",
            llm=fake_llm,
            model="fake-model",
            structured_output_supported=True,
        )
        assert plan.tickers == ["AAPL"]
        assert fake_llm.last_request is not None
        assert fake_llm.last_request.response_format == "json"
        assert fake_llm.last_request.response_schema == QUERY_PLAN_JSON_SCHEMA

    def test_uses_text_output_when_not_supported(self, fake_llm) -> None:
        fake_llm.reply = json.dumps(
            {
                "raw_query": "Q",
                "detected_language": "en",
                "query_en": "Q",
                "tickers": [],
                "form_types": [],
                "date_range": None,
                "intent": "",
                "suggested_answer_mode": "concise",
            }
        )
        plan = understand_query(
            "Q",
            llm=fake_llm,
            model="fake-model",
            structured_output_supported=False,
        )
        assert plan.raw_query == "Q"
        assert fake_llm.last_request.response_format == "text"
        assert fake_llm.last_request.response_schema is None

    def test_falls_back_to_minimal_plan_on_parse_failure(self, fake_llm) -> None:
        """A malformed model response should not raise — the orchestrator must keep going."""
        fake_llm.reply = "totally not json"
        plan = understand_query(
            "Original query",
            llm=fake_llm,
            model="fake-model",
            structured_output_supported=False,
        )
        assert plan.raw_query == "Original query"
        assert plan.query_en == "Original query"
        assert plan.detected_language == "en"
        assert plan.tickers == []

    def test_multilingual_query_carries_query_en(self, fake_llm) -> None:
        fake_llm.reply = json.dumps(
            {
                "raw_query": "AAPL'in gelirleri nasil?",
                "detected_language": "tr",
                "query_en": "How is AAPL revenue?",
                "tickers": ["AAPL"],
                "form_types": [],
                "date_range": None,
                "intent": "",
                "suggested_answer_mode": "concise",
            }
        )
        plan = understand_query(
            "AAPL'in gelirleri nasil?",
            llm=fake_llm,
            model="fake-model",
            structured_output_supported=True,
        )
        assert plan.detected_language == "tr"
        assert plan.query_en == "How is AAPL revenue?"
