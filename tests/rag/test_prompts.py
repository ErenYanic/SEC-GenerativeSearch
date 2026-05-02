"""Tests for :mod:`sec_generative_search.rag.prompts`."""

from __future__ import annotations

import pytest

from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.prompts import (
    ACTIVE_PROMPT_VERSION,
    QUERY_UNDERSTANDING_TEMPLATE,
    TEMPLATES,
    PromptTemplate,
    get_template,
)


class TestPromptTemplate:
    def test_every_mode_has_a_template(self) -> None:
        """Every :class:`AnswerMode` member must have a registered template."""
        for mode in AnswerMode:
            template = get_template(mode)
            assert isinstance(template, PromptTemplate)
            assert template.mode is mode

    def test_active_version_stamped_on_every_template(self) -> None:
        for template in TEMPLATES.values():
            assert template.version == ACTIVE_PROMPT_VERSION

    def test_render_system_includes_output_language(self) -> None:
        rendered = get_template(AnswerMode.CONCISE).render_system(output_language="tr")
        assert " tr" in rendered or "in tr" in rendered

    def test_render_system_mentions_untrusted_delimiter(self) -> None:
        """The system prompt must name the delimiter — that's the trust boundary."""
        rendered = get_template(AnswerMode.CONCISE).render_system(output_language="en")
        assert "<UNTRUSTED_FILING_CONTEXT>" in rendered
        assert "</UNTRUSTED_FILING_CONTEXT>" in rendered

    @pytest.mark.parametrize("mode", list(AnswerMode))
    def test_each_mode_directive_appears_in_render(self, mode: AnswerMode) -> None:
        """Mode-specific directive line must survive into the rendered prompt."""
        template = get_template(mode)
        rendered = template.render_system(output_language="en")
        assert template.mode_directive in rendered

    def test_template_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        template = get_template(AnswerMode.CONCISE)
        with pytest.raises(FrozenInstanceError):
            template.version = "v0"  # type: ignore[misc]


class TestQueryUnderstandingTemplate:
    def test_placeholder_present(self) -> None:
        """The template must contain a ``{query}`` placeholder for the user query."""
        assert "{query}" in QUERY_UNDERSTANDING_TEMPLATE

    def test_describes_required_json_shape(self) -> None:
        """All required QueryPlan fields must appear in the schema description."""
        for field_name in (
            "raw_query",
            "detected_language",
            "query_en",
            "tickers",
            "form_types",
            "date_range",
            "intent",
            "suggested_answer_mode",
        ):
            assert field_name in QUERY_UNDERSTANDING_TEMPLATE
