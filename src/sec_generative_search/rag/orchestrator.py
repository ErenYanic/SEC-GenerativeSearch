"""RAG orchestrator — composes retrieval, generation, and citation extraction.

:class:`RAGOrchestrator` is the single seam between the user-facing
surface (CLI, API, web UI) and the lower-level primitives (retrieval,
provider, types). It owns:

- the four-way context-budget allocation,
- prompt assembly with the untrusted-data delimiter contract,
- the user-supplied vs. orchestrator-supplied filter merge,
- comparative fan-out (multiple retrievals → merged candidate set),
- non-streaming and streaming generation,
- citation extraction (hybrid JSON / inline-marker),
- conversation-history rendering (Q/A pairs only — never chunks),
- :class:`GenerationResult` traceability.

Pipeline (single-query mode):

1. Resolve the active :class:`PromptTemplate` from the requested mode.
2. Render the system prompt (mode + output language).
3. Allocate the four-way budget against the model's context window.
4. Render the conversation-history block (newest-first packing).
5. Retrieve chunks (using ``query_en`` from the plan, the
   ``chunks_tokens`` budget, and any user-supplied filters merged with
   plan filters).
6. If retrieval returns nothing and ``refusal_enabled`` is True,
   short-circuit with a refusal :class:`GenerationResult` — the LLM is
   never called.
7. Render the context block with sanitisation + delimiters.
8. Build the user prompt: history (if any) + context block + question.
9. Call the provider (non-stream or stream).
10. Extract citations (JSON envelope when supported, inline markers
    fallback).
11. Return :class:`GenerationResult` with full traceability.

Comparative-mode fan-out (8.4):
    When ``mode == AnswerMode.COMPARATIVE`` and the plan carries
    multiple tickers OR a date_range that the plan says should be
    split, the orchestrator runs N retrieval calls (one per fan-out
    bucket), merges and deduplicates the candidate sets, then proceeds
    from step 7 with the merged set. v1 splits on ticker and
    date_range only; form_type fan-out is deferred per the Phase 8
    plan.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import GenerationError
from sec_generative_search.core.logging import get_logger, redact_for_log
from sec_generative_search.core.types import (
    GenerationResult,
    RetrievalResult,
    TokenUsage,
)
from sec_generative_search.rag.citations import extract_citations
from sec_generative_search.rag.context import (
    BudgetAllocation,
    ContextBudget,
    build_context_block,
    render_history_block,
)
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.prompts import get_template
from sec_generative_search.rag.query_understanding import QueryPlan

if TYPE_CHECKING:
    from sec_generative_search.core.types import ConversationTurn
    from sec_generative_search.providers.base import (
        BaseLLMProvider,
        GenerationRequest,
    )
    from sec_generative_search.search.retrieval import (
        RetrievalService,
        TokenCounter,
    )

__all__ = ["RAGOrchestrator", "StreamEvent"]


logger = get_logger(__name__)


# Refusal answer text used when retrieval returns no chunks and
# ``refusal_enabled`` is True. Kept short and honest — the UI's
# source-panel will be empty, which is the user-facing signal that the
# system gave up.
_REFUSAL_TEXT = (
    "I cannot answer this from the available filings. No relevant chunks "
    "were retrieved for this query. Try a different ticker, form type, or "
    "date range, or ingest more filings."
)


@dataclass(frozen=True)
class StreamEvent:
    """One event in a streaming generation.

    Streaming consumers iterate :class:`StreamEvent` instances. Most
    events carry a non-empty :attr:`delta` and ``final=None``; the
    final event always carries ``final`` populated and ``delta=None``.

    Frozen so consumers can fan events out to UI subscribers without
    defensive copying.

    Attributes:
        delta: Newly-generated text since the previous event.
            ``None`` on the final event.
        final: The fully-assembled :class:`GenerationResult` with
            citations parsed.  ``None`` on every event except the
            terminal one.
    """

    delta: str | None = None
    final: GenerationResult | None = None


class RAGOrchestrator:
    """Compose retrieval, generation, and citation extraction.

    Construction takes pre-built dependencies — same composition rule
    as :class:`RetrievalService`. The orchestrator never instantiates
    a provider or a retrieval service itself; that lets the API layer's
    ``Depends()`` providers wire credentials and lifespans without the
    orchestrator becoming aware of them.

    Attributes intentionally kept private — public surface is
    :meth:`generate`, :meth:`generate_stream`, and the construction
    arguments.
    """

    def __init__(
        self,
        *,
        retrieval: RetrievalService,
        llm: BaseLLMProvider,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Bind dependencies.

        Args:
            retrieval: Pre-built single-query retrieval primitive.
                Comparative fan-out lives in this class, not in the
                retrieval primitive.
            llm: The user's chosen LLM provider, already constructed
                with their API key.  Same instance is used for
                generation and (when called separately) query
                understanding.
            token_counter: Optional ``Callable[[str], int]``.  Defaults
                to the same ``cl100k_base`` counter as
                :class:`RetrievalService`.  Passing the same counter
                across both layers keeps token costs commensurable.
        """
        self._retrieval = retrieval
        self._llm = llm

        # Reuse the retrieval layer's default counter rather than
        # constructing a second one — the imports happen lazily so this
        # does not bloat the orchestrator's import time.
        if token_counter is None:
            from sec_generative_search.search.retrieval import (
                _get_default_token_counter,
            )

            token_counter = _get_default_token_counter()
        self._token_counter = token_counter

        settings = get_settings()
        self._rag_settings = settings.rag
        self._llm_settings = settings.llm

    # ------------------------------------------------------------------
    # Public surface — non-streaming
    # ------------------------------------------------------------------

    def generate(
        self,
        plan: QueryPlan,
        *,
        mode: AnswerMode | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        history: list[ConversationTurn] | None = None,
        prefer_structured_output: bool = False,
        extra_filters: dict | None = None,
    ) -> GenerationResult:
        """Run the full pipeline non-streaming and return the final result.

        Args:
            plan: Approved :class:`QueryPlan` from the
                query-understanding step (or a minimal plan
                constructed directly from a query).
            mode: Override the mode chosen by the plan. ``None`` means
                use ``plan.suggested_answer_mode``.
            model: Provider model slug. ``None`` means use
                ``settings.llm.default_model`` or the provider's own
                default.
            max_output_tokens: Reserved-for-answer slice. ``None``
                means use ``settings.llm.max_output_tokens``.
            history: Optional conversation history. Only the Q/A pairs
                are rendered; the prior turn's chunks are NEVER
                carried forward.
            prefer_structured_output: Set to ``True`` when the active
                provider's ``ProviderCapability.structured_output`` is
                True. Activates the JSON-envelope citation path.
            extra_filters: Optional caller-supplied overrides to merge
                on top of the plan's filters (e.g. UI-edited chips
                that the user wants to take effect even though
                ``plan`` was generated before the edit).

        Returns:
            :class:`GenerationResult` with citations, retrieved chunks,
            traceability, and token usage populated.
        """
        start = time.monotonic()

        effective_mode = mode if mode is not None else plan.suggested_answer_mode
        effective_model = model or self._llm_settings.default_model or ""
        effective_max_output = (
            max_output_tokens
            if max_output_tokens is not None
            else self._llm_settings.max_output_tokens
        )

        template = get_template(effective_mode)
        output_language = self._resolve_output_language(plan.detected_language)
        system_prompt = template.render_system(output_language=output_language)

        allocation = self._allocate_budget(
            system_prompt=system_prompt,
            max_output_tokens=effective_max_output,
        )

        history_block = self._render_history(history, allocation)
        retrieved = self._retrieve_for_plan(
            plan=plan,
            mode=effective_mode,
            chunks_token_budget=allocation.chunks_tokens,
            extra_filters=extra_filters,
        )

        if not retrieved and self._rag_settings.refusal_enabled:
            return self._build_refusal_result(
                plan=plan,
                template_version=template.version,
                model=effective_model,
                latency_seconds=time.monotonic() - start,
            )

        user_prompt = self._build_user_prompt(
            history_block=history_block,
            retrieved=retrieved,
            plan=plan,
            mode=effective_mode,
        )

        request = self._build_generation_request(
            system=system_prompt,
            user_prompt=user_prompt,
            model=effective_model,
            max_output_tokens=allocation.answer_tokens,
            prefer_structured_output=prefer_structured_output,
        )

        logger.info(
            "RAG generate: mode=%s model=%s lang=%s chunks=%d structured=%s",
            effective_mode.value,
            effective_model or "<provider default>",
            output_language,
            len(retrieved),
            prefer_structured_output,
        )

        response = self._llm.generate(request)

        extracted = extract_citations(
            response.text,
            retrieved,
            prefer_json=prefer_structured_output,
        )

        return GenerationResult(
            answer=extracted.answer,
            provider=self._llm.provider_name,
            model=response.model or effective_model,
            prompt_version=template.version,
            citations=extracted.citations,
            retrieved_chunks=retrieved,
            token_usage=response.token_usage,
            latency_seconds=time.monotonic() - start,
            streamed=False,
        )

    # ------------------------------------------------------------------
    # Public surface — streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        plan: QueryPlan,
        *,
        mode: AnswerMode | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        history: list[ConversationTurn] | None = None,
        prefer_structured_output: bool = False,
        extra_filters: dict | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream the answer and yield :class:`StreamEvent` deltas.

        Identical setup phase to :meth:`generate`. The only difference
        is the call to :meth:`BaseLLMProvider.generate_stream` plus
        accumulation of the streamed text so the final event can carry
        a complete :class:`GenerationResult` with parsed citations.

        Citation extraction runs **once**, on the fully-assembled
        text, because both the JSON-envelope path and the inline-
        marker path need the complete answer to validate references
        against the retrieved chunk set.
        """
        start = time.monotonic()

        effective_mode = mode if mode is not None else plan.suggested_answer_mode
        effective_model = model or self._llm_settings.default_model or ""
        effective_max_output = (
            max_output_tokens
            if max_output_tokens is not None
            else self._llm_settings.max_output_tokens
        )

        template = get_template(effective_mode)
        output_language = self._resolve_output_language(plan.detected_language)
        system_prompt = template.render_system(output_language=output_language)

        allocation = self._allocate_budget(
            system_prompt=system_prompt,
            max_output_tokens=effective_max_output,
        )
        history_block = self._render_history(history, allocation)
        retrieved = self._retrieve_for_plan(
            plan=plan,
            mode=effective_mode,
            chunks_token_budget=allocation.chunks_tokens,
            extra_filters=extra_filters,
        )

        if not retrieved and self._rag_settings.refusal_enabled:
            refusal = self._build_refusal_result(
                plan=plan,
                template_version=template.version,
                model=effective_model,
                latency_seconds=time.monotonic() - start,
            )
            yield StreamEvent(delta=refusal.answer)
            yield StreamEvent(final=refusal)
            return

        user_prompt = self._build_user_prompt(
            history_block=history_block,
            retrieved=retrieved,
            plan=plan,
            mode=effective_mode,
        )
        request = self._build_generation_request(
            system=system_prompt,
            user_prompt=user_prompt,
            model=effective_model,
            max_output_tokens=allocation.answer_tokens,
            prefer_structured_output=prefer_structured_output,
        )

        logger.info(
            "RAG stream: mode=%s model=%s lang=%s chunks=%d structured=%s",
            effective_mode.value,
            effective_model or "<provider default>",
            output_language,
            len(retrieved),
            prefer_structured_output,
        )

        accumulated: list[str] = []
        token_usage = TokenUsage()
        last_model_slug = effective_model
        for chunk in self._llm.generate_stream(request):
            if chunk.text:
                accumulated.append(chunk.text)
                yield StreamEvent(delta=chunk.text)
            if chunk.token_usage.total_tokens > 0:
                token_usage = chunk.token_usage
            if chunk.model:
                last_model_slug = chunk.model

        full_answer = "".join(accumulated)
        extracted = extract_citations(
            full_answer,
            retrieved,
            prefer_json=prefer_structured_output,
        )
        final = GenerationResult(
            answer=extracted.answer,
            provider=self._llm.provider_name,
            model=last_model_slug,
            prompt_version=template.version,
            citations=extracted.citations,
            retrieved_chunks=retrieved,
            token_usage=token_usage,
            latency_seconds=time.monotonic() - start,
            streamed=True,
        )
        yield StreamEvent(final=final)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_output_language(self, detected_language: str) -> str:
        """Pick the answer-output language from settings + plan.

        ``RAG_OUTPUT_LANGUAGE`` is the operator override:
        - ``"auto"`` (default) → use ``detected_language`` from the plan.
        - any BCP-47 code → lock the answer to that language regardless
          of the input language. Useful for shared-deployment scenarios
          where the operator wants every answer in (say) English even
          when the user types in another language.
        """
        configured = self._rag_settings.output_language
        if not configured or configured.lower() == "auto":
            return detected_language or "en"
        return configured

    def _allocate_budget(
        self,
        *,
        system_prompt: str,
        max_output_tokens: int,
    ) -> BudgetAllocation:
        """Build the four-way :class:`BudgetAllocation` for this call.

        Capability discovery is best-effort: a permissive model returns
        ``context_window_tokens=0`` and the allocator then falls back
        to ``RAG_CONTEXT_TOKEN_BUDGET``.  Calling
        :meth:`get_capabilities` is cheap (no network), so we do it on
        every generate call rather than caching — capability matrices
        are immutable per ``(provider, model)`` and the orchestrator
        does not memoise them.
        """
        try:
            capability = self._llm.get_capabilities()
            total_window = capability.context_window_tokens
        except Exception:
            total_window = 0

        budget = ContextBudget(
            total_window=total_window,
            max_output_tokens=max_output_tokens,
            history_token_budget=self._rag_settings.history_token_budget,
            default_chunks_budget_fallback=self._rag_settings.context_token_budget,
        )
        return budget.allocate(
            system_prompt=system_prompt,
            token_counter=self._token_counter,
        )

    def _render_history(
        self,
        history: list[ConversationTurn] | None,
        allocation: BudgetAllocation,
    ) -> str:
        """Render history under its budget; respect the chat-history flag."""
        if history is None or not self._rag_settings.chat_history_enabled:
            return ""
        # Cap by max_turns first to keep work bounded; the renderer
        # then packs newest-first under the token budget.
        max_turns = self._rag_settings.chat_history_max_turns
        if max_turns > 0 and len(history) > max_turns:
            history = history[-max_turns:]
        return render_history_block(
            history,
            max_tokens=allocation.history_tokens,
            token_counter=self._token_counter,
        )

    def _retrieve_for_plan(
        self,
        *,
        plan: QueryPlan,
        mode: AnswerMode,
        chunks_token_budget: int,
        extra_filters: dict | None,
    ) -> list[RetrievalResult]:
        """Run retrieval — single call, or comparative fan-out.

        Comparative fan-out splits by ticker first, then by date_range
        when more than one is implied (v1 takes the plan's
        ``date_range`` as a single bucket; multi-range comparison is
        still out of scope here).
        """
        merged_filters = self._merge_filters(plan, extra_filters)

        if mode != AnswerMode.COMPARATIVE or len(plan.tickers) <= 1:
            # Single-query path — the common case.
            return self._retrieval.retrieve(
                plan.query_en,
                context_token_budget=chunks_token_budget,
                **merged_filters,
            )

        # Comparative fan-out by ticker.  Per-ticker budget is the
        # total chunks budget divided by the ticker count, so the
        # final merged context still fits the model.
        per_ticker_budget = max(256, chunks_token_budget // len(plan.tickers))
        merged: list[RetrievalResult] = []
        seen_ids: set[str] = set()
        for ticker in plan.tickers:
            per_filters = dict(merged_filters)
            per_filters["ticker"] = ticker
            results = self._retrieval.retrieve(
                plan.query_en,
                context_token_budget=per_ticker_budget,
                **per_filters,
            )
            for hit in results:
                if hit.chunk_id and hit.chunk_id in seen_ids:
                    continue
                if hit.chunk_id:
                    seen_ids.add(hit.chunk_id)
                merged.append(hit)
        return merged

    @staticmethod
    def _merge_filters(plan: QueryPlan, extra: dict | None) -> dict:
        """Build the filter kwargs for :meth:`RetrievalService.retrieve`.

        The plan's filters are the baseline; *extra* (caller overrides
        from edited chips) takes precedence per-key. Empty plan fields
        are not forwarded — the retrieval primitive accepts ``None``
        for every filter and only adds the filter when it is non-None.
        """
        base: dict = {}
        if plan.tickers:
            base["ticker"] = plan.tickers if len(plan.tickers) > 1 else plan.tickers[0]
        if plan.form_types:
            base["form_type"] = plan.form_types if len(plan.form_types) > 1 else plan.form_types[0]
        if plan.date_range is not None:
            start, end = plan.date_range
            base["start_date"] = start
            base["end_date"] = end
        if extra:
            base.update(extra)
        return base

    @staticmethod
    def _build_user_prompt(
        *,
        history_block: str,
        retrieved: list[RetrievalResult],
        plan: QueryPlan,
        mode: AnswerMode,
    ) -> str:
        """Assemble the user-message body fed to the provider.

        Order matters:

        1. Conversation history (if any) — gives the model continuity.
        2. Retrieved-context block — the only ground truth.
        3. Question — last so the model's most recent attention is on
           the actual ask.

        For comparative mode we prepend a small "compare X across Y"
        directive that nudges the model toward a tabular answer; the
        per-mode template's ``mode_directive`` already says the same
        thing, but seeing it twice (system + user) reduces the chance
        the model collapses into a single-entity summary.
        """
        parts: list[str] = []
        if history_block:
            parts.append("Earlier turns in this session:")
            parts.append(history_block)
            parts.append("")
        parts.append(build_context_block(retrieved))
        parts.append("")
        if mode == AnswerMode.COMPARATIVE and len(plan.tickers) > 1:
            parts.append(f"Compare the following entities side by side: {', '.join(plan.tickers)}.")
        parts.append(f"Question: {plan.raw_query}")
        return "\n".join(parts)

    @staticmethod
    def _build_generation_request(
        *,
        system: str,
        user_prompt: str,
        model: str,
        max_output_tokens: int,
        prefer_structured_output: bool,
    ) -> GenerationRequest:
        """Build the :class:`GenerationRequest` passed to the provider."""
        # Local import keeps top-level imports light.
        from sec_generative_search.providers.base import GenerationRequest

        return GenerationRequest(
            prompt=user_prompt,
            model=model,
            system=system,
            max_output_tokens=max_output_tokens,
            response_format="json" if prefer_structured_output else "text",
            response_schema=_CITATION_ENVELOPE_SCHEMA if prefer_structured_output else None,
        )

    def _build_refusal_result(
        self,
        *,
        plan: QueryPlan,
        template_version: str,
        model: str,
        latency_seconds: float,
    ) -> GenerationResult:
        """Synthesise a refusal :class:`GenerationResult` without an LLM call.

        Refusal short-circuits before generation; we still populate the
        full traceability surface so logs and dashboards see a uniform
        shape across success/refusal paths. Token usage is genuinely
        zero — no provider call was made — so we leave it at the
        default rather than fabricating numbers.

        The query is logged via :func:`redact_for_log` because Tier 3
        user data must not land in operator logs without the redactor
        gate.
        """
        logger.info(
            "RAG refusal — no chunks retrieved for query=%r tickers=%s",
            redact_for_log(plan.raw_query[:80]),
            plan.tickers or "any",
        )
        return GenerationResult(
            answer=_REFUSAL_TEXT,
            provider=self._llm.provider_name,
            model=model,
            prompt_version=template_version,
            citations=[],
            retrieved_chunks=[],
            token_usage=TokenUsage(),
            latency_seconds=latency_seconds,
            streamed=False,
        )


# JSON Schema for the citation-envelope response_format.  The orchestrator
# attaches this to the request when ``prefer_structured_output=True``;
# providers that accept JSON Schema (OpenAI, Gemini) constrain output to
# match, providers that only accept generic JSON-mode ignore the schema.
_CITATION_ENVELOPE_SCHEMA: dict = {
    "$id": "rag_citation_envelope",
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "cited_chunk_ids"],
    "properties": {
        "answer": {"type": "string"},
        "cited_chunk_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


# Re-export the exception so callers can `except GenerationError` without
# reaching into core.exceptions.  Same pattern as other public packages.
__all__ = [*__all__, "GenerationError"]
