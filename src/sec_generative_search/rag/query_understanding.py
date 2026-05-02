"""Query-understanding step for the RAG orchestrator.

Extracts a :class:`QueryPlan` from a user query via a single LLM call
with structured output. The plan surfaces as editable chips in the UI;
nothing else runs until the caller invokes :meth:`RAGOrchestrator.generate`.

Two-mode call shape:

- When the active provider's
  :attr:`ProviderCapability.structured_output` is True, the call uses
  ``response_format="json"`` with the schema defined here.
- Otherwise the call uses ``response_format="text"`` and we parse the
  model's free-form output through the same JSON envelope isolator used
    by citation extraction. This is the documented best-effort fallback
    used by the orchestrator.

Multilingual contract:

- ``detected_language`` — BCP-47 code, defaults to ``"en"`` on
  low-confidence detection.
- ``query_en`` — English rendering of the query, used for embedding.
  When ``detected_language == "en"`` it equals ``raw_query``.
- The orchestrator passes ``query_en`` to
  :meth:`RetrievalService.retrieve` regardless of the original query's
  language. This stabilises retrieval quality across embedder swaps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sec_generative_search.core.exceptions import GenerationError
from sec_generative_search.core.logging import get_logger, redact_for_log
from sec_generative_search.rag.citations import _isolate_json_object
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.prompts import QUERY_UNDERSTANDING_TEMPLATE

if TYPE_CHECKING:
    from sec_generative_search.providers.base import (
        BaseLLMProvider,
    )

__all__ = [
    "QUERY_PLAN_JSON_SCHEMA",
    "QueryPlan",
    "parse_query_plan",
    "understand_query",
]


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# QueryPlan dataclass + JSON schema
# ---------------------------------------------------------------------------


@dataclass
class QueryPlan:
    """Structured plan produced by the query-understanding step.

    Mutable on purpose — the UI's "editable chips" surface lets the user
    correct fields between :func:`understand_query` and
    :meth:`RAGOrchestrator.execute`. Frozen would make that flow awkward.

    Attributes:
        raw_query: The user's query, verbatim. Tier 3 user-generated
            content; passes through ``redact_for_log`` at log sites.
        detected_language: BCP-47 code (``"en"``, ``"tr"``, ``"de"``,
            ...).  ``"en"`` on low-confidence detection.
        query_en: English rendering used for embedding. Equal to
            ``raw_query`` when ``detected_language == "en"``.
        tickers: Uppercase stock symbols extracted from the query.
        form_types: SEC form types (``"10-K"``, ``"10-Q"``, ...). Empty
            when the user did not specify.
        date_range: Optional ``(start, end)`` ISO date pair. ``None``
            when the user did not specify or imply a window.
        intent: One short sentence describing what the user wants. Used
            for traceability and downstream UX (chip label, error
            messages).
        suggested_answer_mode: Mode the model judged appropriate.
            Defaults to :attr:`AnswerMode.CONCISE` on parse failure or
            unrecognised values — :meth:`AnswerMode.from_string` does
            the lift.
    """

    raw_query: str
    detected_language: str = "en"
    query_en: str = ""
    tickers: list[str] = field(default_factory=list)
    form_types: list[str] = field(default_factory=list)
    date_range: tuple[str, str] | None = None
    intent: str = ""
    suggested_answer_mode: AnswerMode = AnswerMode.CONCISE

    def __post_init__(self) -> None:
        """Fill ``query_en`` from ``raw_query`` when blank.

        The understanding step always populates ``query_en`` explicitly,
        but this default keeps the dataclass usable without going
        through :func:`understand_query` (e.g. for tests, or for
        callers that bypass query-understanding entirely).
        """
        if not self.query_en:
            self.query_en = self.raw_query


# JSON Schema for structured-output calls.  Kept inline so the schema
# version moves with the dataclass; if the schema diverges from
# :class:`QueryPlan`, :func:`parse_query_plan` is the single seam where
# the divergence is noticed.
QUERY_PLAN_JSON_SCHEMA: dict = {
    "$id": "rag_query_plan",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "raw_query",
        "detected_language",
        "query_en",
        "tickers",
        "form_types",
        "date_range",
        "intent",
        "suggested_answer_mode",
    ],
    "properties": {
        "raw_query": {"type": "string"},
        "detected_language": {"type": "string"},
        "query_en": {"type": "string"},
        "tickers": {"type": "array", "items": {"type": "string"}},
        "form_types": {"type": "array", "items": {"type": "string"}},
        "date_range": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "string"},
                },
            ]
        },
        "intent": {"type": "string"},
        "suggested_answer_mode": {
            "type": "string",
            "enum": ["concise", "analytical", "extractive", "comparative"],
        },
    },
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_query_plan(payload: str, *, raw_query: str) -> QueryPlan:
    """Parse a JSON envelope (or free-form text containing one) into a plan.

    Tolerant of:
    - markdown fences around the JSON block,
    - extra prose before/after the object,
    - unknown keys (silently dropped),
    - missing optional fields (filled with safe defaults).

    Strict on:
    - the payload must contain at least one balanced JSON object,
    - tickers and form_types must be lists of strings,
    - date_range, when present, must be a 2-tuple of ISO date strings.

    Raises :class:`GenerationError` when the payload is unparseable —
    the orchestrator catches this and either retries with a stricter
    prompt or falls back to a minimal plan derived from ``raw_query``.
    """
    isolated = _isolate_json_object(payload)
    if isolated is None:
        raise GenerationError(
            "Query-understanding output contains no JSON object",
            details=f"first 80 chars: {payload[:80]!r}",
        )

    try:
        parsed = json.loads(isolated)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            "Query-understanding JSON envelope is malformed",
            details=str(exc),
        ) from exc

    if not isinstance(parsed, dict):
        raise GenerationError(
            "Query-understanding JSON envelope is not an object",
            details=f"got {type(parsed).__name__}",
        )

    detected_language = _coerce_str(parsed.get("detected_language"), default="en")
    # The English rendering defaults to raw_query when the model omitted
    # it — best-effort even for non-English queries (the embedder's
    # cross-lingual fallback is the safety net there).  Keeps the
    # contract that ``query_en`` is always populated.
    query_en = _coerce_str(parsed.get("query_en"), default="")
    if not query_en:
        query_en = raw_query

    return QueryPlan(
        raw_query=raw_query,
        detected_language=detected_language,
        query_en=query_en,
        tickers=_coerce_uppercase_list(parsed.get("tickers")),
        form_types=_coerce_str_list(parsed.get("form_types")),
        date_range=_coerce_date_range(parsed.get("date_range")),
        intent=_coerce_str(parsed.get("intent"), default=""),
        suggested_answer_mode=AnswerMode.from_string(
            _coerce_str(parsed.get("suggested_answer_mode"), default=""),
            default=AnswerMode.CONCISE,
        ),
    )


# ---------------------------------------------------------------------------
# Understanding call
# ---------------------------------------------------------------------------


def understand_query(
    query: str,
    *,
    llm: BaseLLMProvider,
    model: str,
    max_output_tokens: int = 512,
    temperature: float = 0.0,
    structured_output_supported: bool = False,
) -> QueryPlan:
    """Call the LLM once to extract a :class:`QueryPlan` from *query*.

    Single-call by design — folding language detection / translation
    into the same structured-output call avoids a second round-trip on
    the user-facing path.

    Args:
        query: Tier 3 user query, possibly non-English.
        llm: Pre-built LLM provider; the orchestrator builds this once
            per request from the user-supplied provider key.
        model: Model slug to use. The orchestrator picks the same
            model used for generation by default; callers may swap to
            a cheaper model here.
        max_output_tokens: Cap for the structured output. Plans are
            small — 512 is generous.
        temperature: Default 0.0 because the plan is structural and
            deterministic across runs is what we want.
        structured_output_supported: Set to ``True`` when the active
            provider/model's ``ProviderCapability.structured_output``
            is True. When True, the call uses ``response_format="json"``
            with :data:`QUERY_PLAN_JSON_SCHEMA`; otherwise we ask for
            JSON via prompt and parse free-form output. Either path
            ends in :func:`parse_query_plan`.

    Returns:
        Validated :class:`QueryPlan`. On parse failure, falls back to
        a minimal plan derived from *query* (the field defaults yield
        an English-only, no-filter plan that still routes through the
        rest of the pipeline).

    Raises:
        :class:`GenerationError`: Never raised — the function catches
            parse failures and returns the fallback plan. Underlying
            provider errors propagate verbatim (they already carry
            their own type via :func:`resilient_call`).
    """
    # Local import keeps ``rag`` free of a top-level dependency on the
    # provider request type at import time; ``query_understanding`` is
    # lazy-imported by the orchestrator anyway.
    from sec_generative_search.providers.base import GenerationRequest

    rendered_prompt = QUERY_UNDERSTANDING_TEMPLATE.format(query=query)
    request = GenerationRequest(
        prompt=rendered_prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_format="json" if structured_output_supported else "text",
        response_schema=QUERY_PLAN_JSON_SCHEMA if structured_output_supported else None,
    )

    logger.debug(
        "Query-understanding call: query=%r structured=%s",
        redact_for_log(query[:80]),
        structured_output_supported,
    )

    response = llm.generate(request)
    try:
        return parse_query_plan(response.text, raw_query=query)
    except GenerationError as exc:
        logger.warning(
            "Query-understanding parse failed (%s); falling back to minimal plan",
            exc,
        )
        return QueryPlan(raw_query=query)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_str(value: object, *, default: str) -> str:
    """Return *value* as a stripped string, or *default* on bad input."""
    if isinstance(value, str):
        return value.strip() or default
    return default


def _coerce_str_list(value: object) -> list[str]:
    """Return a list of stripped non-empty strings; ignore non-string entries."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                out.append(stripped)
    return out


def _coerce_uppercase_list(value: object) -> list[str]:
    """Like :func:`_coerce_str_list` but uppercase + alnum-only.

    Tickers occasionally come back lower-cased from non-English queries
    or wrapped in parentheses ("(AAPL)"); we strip non-alphanumerics
    rather than fail. Empty entries after cleanup are dropped.
    """
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = "".join(ch for ch in item if ch.isalnum()).upper()
        if cleaned:
            out.append(cleaned)
    return out


def _coerce_date_range(value: object) -> tuple[str, str] | None:
    """Validate a ``[start, end]`` ISO-date pair; return ``None`` on bad input.

    Permissive on order — the retrieval layer validates ISO format
    independently and ChromaDB's range filter doesn't care which
    bound is larger; swapping them would silently match nothing, which
    is observable, so we don't second-guess the model here.
    """
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        return None
    start, end = value
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    return (start.strip(), end.strip())
