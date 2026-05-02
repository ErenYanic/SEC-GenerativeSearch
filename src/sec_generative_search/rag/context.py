"""Context-window budgeting and prompt-block assembly for the RAG orchestrator.

Two separate concerns share this module because both operate on token
budgets — splitting them across two files would force the orchestrator
to hold the same `TokenCounter` callable twice.

Concern 1 — :class:`ContextBudget`
    Allocates a model's total context window across four consumers:
    system prompt, retrieved chunks, conversation history,
    reserved-for-answer. The allocator returns concrete integer budgets
    so the rest of the orchestrator can pass them straight to
    :meth:`RetrievalService.retrieve` and to the history renderer.

Concern 2 — :func:`build_context_block` and :func:`render_history_block`
    Assemble the user-message body the LLM sees.  Retrieved chunks are
    sanitised per-chunk and wrapped in
    ``<UNTRUSTED_FILING_CONTEXT>`` delimiters; conversation history is
    rendered as ``Q: …\nA: …`` pairs only — the prior turn's
    ``retrieval_results`` are deliberately discarded (they exist for
    audit, never as future prompt fuel).

Token-budget defaults:
    - History slice defaults to 15% of the total context window. The
      number is empirical (a few prior Q/A pairs is enough for most
      follow-ups; more pushes retrieval out).
    - Operator override via ``RAG_HISTORY_TOKEN_BUDGET`` env var, read
      from :class:`RAGSettings`.
    - When the model's context window is unknown
      (``ProviderCapability.context_window_tokens == 0``), the
      allocator falls back to ``RAG_CONTEXT_TOKEN_BUDGET`` plus a
      conservative headroom for the system + answer slices.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sec_generative_search.core.security import sanitize_retrieved_context

if TYPE_CHECKING:
    from sec_generative_search.core.types import (
        ConversationTurn,
        RetrievalResult,
    )

__all__ = [
    "BudgetAllocation",
    "ContextBudget",
    "build_context_block",
    "render_history_block",
]


TokenCounter = Callable[[str], int]
"""Same signature as :data:`sec_generative_search.search.retrieval.TokenCounter`.

Re-declared here so ``rag.context`` does not pull in the search module
at import time — keeps the dependency graph one-way (rag → search,
never the reverse).
"""


# Default fraction of total context allocated to conversation history
# when no explicit operator budget is set.  Empirical sweet spot — see
# module docstring.  Operator override via ``RAG_HISTORY_TOKEN_BUDGET``.
DEFAULT_HISTORY_BUDGET_FRACTION = 0.15

# Conservative system-prompt headroom used when the model's full context
# window is unknown.  The actual system prompt usually clocks in well
# below this; the allocator measures the rendered string anyway.
_SYSTEM_HEADROOM_FALLBACK_TOKENS = 1024


@dataclass(frozen=True)
class BudgetAllocation:
    """Concrete per-consumer integer budgets for one generation call.

    Frozen — once allocated, the budget is the contract for the rest of
    the orchestrator's pipeline.  Mutating it mid-pipeline would silently
    overshoot the model's window.

    Attributes:
        total_window: The model's full context window in tokens, or the
            project-wide fallback when unknown.
        system_tokens: Measured token count of the rendered system
            prompt.
        history_tokens: Maximum tokens allocated to rendered
            conversation history.
        chunks_tokens: Maximum tokens allocated to retrieved-chunk
            context.  Passed straight to
            :meth:`RetrievalService.retrieve` as
            ``context_token_budget``.
        answer_tokens: Maximum response tokens (passed as
            ``max_output_tokens`` to the provider).
    """

    total_window: int
    system_tokens: int
    history_tokens: int
    chunks_tokens: int
    answer_tokens: int


@dataclass(frozen=True)
class ContextBudget:
    """Four-way context-window allocator for the RAG orchestrator.

    Frozen because allocation depends only on construction-time inputs;
    a per-call allocator makes the data flow explicit (settings →
    allocator → :class:`BudgetAllocation`) without mutable state.

    Construction inputs are deliberately small and explicit so tests can
    drive the allocator without booting the full settings tree.

    Attributes:
        total_window: Model's context window in tokens.  ``0`` triggers
            the project-wide fallback derived from
            ``RAG_CONTEXT_TOKEN_BUDGET``.
        max_output_tokens: Reserved-for-answer slice; matches the
            request's ``max_output_tokens``.
        history_token_budget: Operator override for the history slice;
            ``0`` means "use the default fraction".  Negative is
            rejected at :class:`RAGSettings` load.
        default_chunks_budget_fallback: Used when ``total_window == 0``;
            comes from ``RAG_CONTEXT_TOKEN_BUDGET``.
    """

    total_window: int
    max_output_tokens: int
    history_token_budget: int
    default_chunks_budget_fallback: int

    def allocate(
        self,
        *,
        system_prompt: str,
        token_counter: TokenCounter,
    ) -> BudgetAllocation:
        """Compute the per-consumer budgets for one generation call.

        Allocation rules:

        1. Measure the rendered system prompt with ``token_counter``.
        2. Reserve ``max_output_tokens`` for the answer.
        3. History slice is ``history_token_budget`` when set,
           otherwise ``int(total_window * DEFAULT_HISTORY_BUDGET_FRACTION)``.
        4. Chunks slice gets whatever remains after the three above.
           If the remainder would be non-positive, the chunks slice is
           set to a small floor (256 tokens) and the history slice is
           collapsed first; this is a defensive path — the caller
           should size the model and budget so it does not fire.

        When ``total_window == 0`` (model capability did not advertise
        a window), the allocator uses
        ``default_chunks_budget_fallback`` as the chunks slice
        directly and computes ``total_window`` as
        ``system + history + chunks + answer`` for traceability.

        Args:
            system_prompt: The fully-rendered system prompt that will
                be sent to the LLM.  Measured here, not estimated.
            token_counter: Same callable used by retrieval; counts must
                be commensurable across consumers.

        Returns:
            A frozen :class:`BudgetAllocation`.
        """
        system_tokens = max(0, token_counter(system_prompt))
        answer_tokens = max(0, self.max_output_tokens)

        if self.total_window <= 0:
            # Unknown-window path.  Use the configured chunks budget
            # verbatim; pick a fixed history slice from operator
            # override or default-fraction-of-chunks.
            chunks_tokens = max(0, self.default_chunks_budget_fallback)
            if self.history_token_budget > 0:
                history_tokens = self.history_token_budget
            else:
                history_tokens = int(chunks_tokens * DEFAULT_HISTORY_BUDGET_FRACTION)
            total = system_tokens + history_tokens + chunks_tokens + answer_tokens
            return BudgetAllocation(
                total_window=total,
                system_tokens=system_tokens,
                history_tokens=history_tokens,
                chunks_tokens=chunks_tokens,
                answer_tokens=answer_tokens,
            )

        # Known-window path.  Allocate from the top down.
        if self.history_token_budget > 0:
            history_tokens = self.history_token_budget
        else:
            history_tokens = int(self.total_window * DEFAULT_HISTORY_BUDGET_FRACTION)

        remaining = self.total_window - system_tokens - answer_tokens - history_tokens
        if remaining <= 0:
            # Defensive collapse: drop history first (least load-bearing
            # for grounded answers), then floor chunks at 256.
            history_tokens = 0
            remaining = self.total_window - system_tokens - answer_tokens
            chunks_tokens = max(256, remaining)
        else:
            chunks_tokens = remaining

        return BudgetAllocation(
            total_window=self.total_window,
            system_tokens=system_tokens,
            history_tokens=history_tokens,
            chunks_tokens=chunks_tokens,
            answer_tokens=answer_tokens,
        )


# ---------------------------------------------------------------------------
# Prompt-block assembly
# ---------------------------------------------------------------------------


# Delimiters wrapping the retrieved-context block.  The system prompt
# refers to these tags by name; changing them here is a breaking change
# to the trust boundary and requires a ``ACTIVE_PROMPT_VERSION`` bump.
_CONTEXT_OPEN = "<UNTRUSTED_FILING_CONTEXT>"
_CONTEXT_CLOSE = "</UNTRUSTED_FILING_CONTEXT>"


def build_context_block(chunks: list[RetrievalResult]) -> str:
    """Render retrieved chunks into the user-message context block.

    Layout::

        <UNTRUSTED_FILING_CONTEXT>
        [1] {ticker} {form_type} ({filing_date}) — {section_path}
        {sanitised_text_span}
        ---
        [2] ...
        </UNTRUSTED_FILING_CONTEXT>

    The ``[N]`` markers are 1-based and match the citation-extraction
    convention (see :mod:`sec_generative_search.rag.citations`).  Each
    chunk's text is run through
    :func:`sanitize_retrieved_context` before interpolation —
    defence-in-depth against chat-template control tokens — but the
    delimiter wrapping is what makes this safe; the system prompt
    instructs the model to treat the block as untrusted data.

    An empty *chunks* list is rendered as the empty block so the
    prompt structure stays uniform; the orchestrator's refusal path
    short-circuits before we get here, so this branch is only hit in
    tests.
    """
    if not chunks:
        return f"{_CONTEXT_OPEN}\n(no chunks retrieved)\n{_CONTEXT_CLOSE}"

    parts: list[str] = [_CONTEXT_OPEN]
    for index, chunk in enumerate(chunks, start=1):
        header = (
            f"[{index}] {chunk.ticker} {chunk.form_type} "
            f"({chunk.filing_date or 'unknown date'}) — {chunk.path}"
        )
        sanitised = sanitize_retrieved_context(chunk.content)
        parts.append(header)
        parts.append(sanitised)
        if index != len(chunks):
            parts.append("---")
    parts.append(_CONTEXT_CLOSE)
    return "\n".join(parts)


def render_history_block(
    history: list[ConversationTurn],
    *,
    max_tokens: int,
    token_counter: TokenCounter,
) -> str:
    """Render conversation history as ``Q: …\\nA: …`` pairs only.

    The prior turn's ``retrieval_results`` and ``citations`` are
    **never** carried into the next prompt. Only the raw user query and
    the model's plain-text answer survive into the follow-up. This both
    honours the privacy posture (no chunk text leaks into a follow-up
    the user might not have intended for that context) and keeps the
    history slice tiny.

    Greedy newest-first packing — the most recent turns are most
    relevant for follow-ups, so they get budget first.  Older turns are
    dropped when the budget is exhausted; the result preserves
    chronological order in the rendered string.

    Returns the empty string when *history* is empty or *max_tokens*
    is zero — the orchestrator splices this directly into the user
    message and an empty splice is a no-op.
    """
    if not history or max_tokens <= 0:
        return ""

    # Render each turn into its final form first, then pack greedily
    # from the newest backwards so the most recent context is preserved
    # under tight budgets.
    rendered_turns: list[tuple[int, str]] = []
    used = 0
    for turn in reversed(history):
        block = f"Q: {turn.query}\nA: {turn.generation_result.answer}"
        cost = token_counter(block)
        if used + cost > max_tokens:
            break
        used += cost
        rendered_turns.append((cost, block))

    if not rendered_turns:
        return ""

    # Reverse back to chronological order before joining.
    return "\n\n".join(block for _, block in reversed(rendered_turns))
