"""Kimi (Moonshot AI) provider adapter (Phase 5D.2).

Moonshot AI ships an OpenAI-compatible Chat Completions surface at
``https://api.moonshot.ai/v1`` (international) — Mainland callers use the
``.cn`` mirror, which shares the wire protocol.  Only the chat surface
is exposed; Moonshot does not currently offer a first-party embedding
API, so pair with :class:`OpenAIEmbeddingProvider` /
:class:`MistralEmbeddingProvider` / :class:`GeminiEmbeddingProvider` when
using this adapter end-to-end.
"""

from __future__ import annotations

from typing import ClassVar

from sec_generative_search.core.types import PricingTier, ProviderCapability
from sec_generative_search.providers.openai_compat import (
    ModelInfo,
    OpenAICompatibleLLMProvider,
)

__all__ = ["KimiProvider"]


class KimiProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting Moonshot AI's Kimi endpoints."""

    provider_name = "kimi"
    default_base_url = "https://api.moonshot.ai/v1"
    default_model = "moonshot-v1-32k"

    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {
        "kimi-k2": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=128_000,
                max_output_tokens=8_192,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "moonshot-v1-8k": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=8_192,
                max_output_tokens=4_096,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "moonshot-v1-32k": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=32_768,
                max_output_tokens=4_096,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "moonshot-v1-128k": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=131_072,
                max_output_tokens=4_096,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
    }
