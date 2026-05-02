"""Tests for :mod:`sec_generative_search.rag.modes`."""

from __future__ import annotations

import pytest

from sec_generative_search.rag.modes import AnswerMode


class TestAnswerMode:
    def test_values_match_settings_strings(self) -> None:
        """The string values must round-trip through ``RAG_DEFAULT_ANSWER_MODE``."""
        assert AnswerMode.CONCISE.value == "concise"
        assert AnswerMode.ANALYTICAL.value == "analytical"
        assert AnswerMode.EXTRACTIVE.value == "extractive"
        assert AnswerMode.COMPARATIVE.value == "comparative"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("concise", AnswerMode.CONCISE),
            ("ANALYTICAL", AnswerMode.ANALYTICAL),
            ("  extractive  ", AnswerMode.EXTRACTIVE),
            ("Comparative", AnswerMode.COMPARATIVE),
        ],
    )
    def test_from_string_normalises_input(self, raw: str, expected: AnswerMode) -> None:
        assert AnswerMode.from_string(raw, default=AnswerMode.CONCISE) is expected

    @pytest.mark.parametrize("raw", [None, "", "wat", "compre", "json"])
    def test_from_string_falls_back_to_default(self, raw: str | None) -> None:
        """Unknown / blank values must not raise — the orchestrator picks the default."""
        assert AnswerMode.from_string(raw, default=AnswerMode.ANALYTICAL) is AnswerMode.ANALYTICAL
