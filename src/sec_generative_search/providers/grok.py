"""xAI Grok provider adapter.

xAI exposes an OpenAI-compatible Chat Completions surface at
``https://api.x.ai/v1`` — the SDK works unchanged via a ``base_url``
override.  xAI ships no first-party embedding or reranker API, so this
module exports the chat adapter only; pair with one of the existing
embedding adapters when using Grok end-to-end.

The published catalogue uses dated slugs
(``grok-4.20-0309-reasoning``, ``grok-4-1-fast-reasoning``) rather than
moving aliases.  The dated form is preserved here because it is what
the xAI API actually accepts — a shorter alias like ``grok-4`` is not
authoritative across every endpoint and would surface as a 404 at call
time.  Reasoning vs. non-reasoning is expressed as a separate slug
(not a toggle on a single slug), matching xAI's dispatch model.
"""

from __future__ import annotations

from typing import ClassVar

from sec_generative_search.core.types import PricingTier, ProviderCapability
from sec_generative_search.providers.openai_compat import (
    ModelInfo,
    OpenAICompatibleLLMProvider,
)

__all__ = ["GrokProvider"]


class GrokProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting xAI's hosted Grok models."""

    provider_name = "grok"
    default_base_url = "https://api.x.ai/v1"
    default_model = "grok-4-1-fast-non-reasoning"

    # Slugs follow xAI's official ``docs.x.ai`` pricing table.  Every
    # chat-capable text model shares a 2 000 000-token context window
    # today; individual ``max_output_tokens`` values are not published,
    # so a conservative 8 192-token ceiling is used for the capability
    # matrix — call sites can override via :class:`GenerationRequest`
    # and the SDK will surface any upstream limit.
    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {
        "grok-4.20-0309-reasoning": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=2_000_000,
                max_output_tokens=2_000_000,
                pricing_tier=PricingTier.HIGH,
            ),
        ),
        "grok-4.20-0309-non-reasoning": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=2_000_000,
                max_output_tokens=2_000_000,
                pricing_tier=PricingTier.HIGH,
            ),
        ),
        "grok-4.20-multi-agent-0309": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=2_000_000,
                max_output_tokens=2_000_000,
                pricing_tier=PricingTier.HIGH,
            ),
        ),
        "grok-4-1-fast-reasoning": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=2_000_000,
                max_output_tokens=2_000_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "grok-4-1-fast-non-reasoning": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=2_000_000,
                max_output_tokens=2_000_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
    }
