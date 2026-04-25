"""Kimi (Moonshot AI) provider adapter.

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
    default_model = "moonshot-v1-8k"

    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {
        "kimi-k2.6": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=262_144,
                max_output_tokens=262_144,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "kimi-k2.5": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=262_144,
                max_output_tokens=262_144,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "kimi-k2-0905-preview": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=262_144,
                max_output_tokens=262_144,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "kimi-k2-0711-preview": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=131_072,
                max_output_tokens=131_072,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "kimi-k2-turbo-preview": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=262_144,
                max_output_tokens=262_144,
                pricing_tier=PricingTier.HIGH,
            ),
        ),
        "kimi-k2-thinking": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=262_144,
                max_output_tokens=262_144,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "kimi-k2-thinking-turbo": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=262_144,
                max_output_tokens=262_144,
                pricing_tier=PricingTier.HIGH,
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
                max_output_tokens=8_192,
                pricing_tier=PricingTier.STANDARD,
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
                max_output_tokens=32_768,
                pricing_tier=PricingTier.STANDARD,
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
                max_output_tokens=131_072,
                pricing_tier=PricingTier.HIGH,
            ),
        ),
    }
