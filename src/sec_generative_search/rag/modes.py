"""Answer modes for the RAG orchestrator.

Each mode maps to a :class:`~sec_generative_search.rag.prompts.PromptTemplate`
variant that tunes the system prompt and the answer instructions for a
specific class of question.

Mode → prompt mapping is owned by :mod:`sec_generative_search.rag.prompts`;
this module only defines the enum so other modules can reference modes
without importing the prompt templates (and the dependency direction
stays one-way).

Mode descriptions:

- ``CONCISE`` — short, direct answer (default). 1-3 sentences, citations
  inline.
- ``ANALYTICAL`` — longer-form analysis with explicit reasoning steps,
  still grounded in the retrieved chunks only.
- ``EXTRACTIVE`` — minimally-rephrased extraction of the relevant text;
  preserves filing language. Useful when the user wants to quote the
  filing.
- ``COMPARATIVE`` — side-by-side comparison across two or more
  filings/tickers/date_ranges. Triggers multi-query fan-out in the
  orchestrator (split by ticker and/or date_range; form_type fan-out is
  deferred per the Phase 8 plan).

The string values match the ``RAG_DEFAULT_ANSWER_MODE`` env var values
already accepted by :class:`RAGSettings`.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["AnswerMode"]


class AnswerMode(StrEnum):
    """User-selectable answer-generation mode.

    ``StrEnum`` so values round-trip through env vars and JSON without
    explicit conversions; ``AnswerMode.CONCISE.value == "concise"``.
    """

    CONCISE = "concise"
    ANALYTICAL = "analytical"
    EXTRACTIVE = "extractive"
    COMPARATIVE = "comparative"

    @classmethod
    def from_string(cls, value: str | None, *, default: AnswerMode) -> AnswerMode:
        """Parse a string into an :class:`AnswerMode`, falling back to *default*.

        Used by the orchestrator and the query-understanding step where
        the source string may be operator-supplied (env var) or
        model-supplied (structured output) — both can be malformed.
        """
        if not value:
            return default
        try:
            return cls(value.strip().lower())
        except ValueError:
            return default
