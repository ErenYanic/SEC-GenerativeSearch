"""Qwen (Alibaba DashScope) provider adapters.

Alibaba's DashScope exposes an OpenAI-compatible surface at
``https://dashscope-intl.aliyuncs.com/compatible-mode/v1`` for
international accounts.  Both chat and embedding surfaces are available
and are modelled separately here so each can be used independently.
"""

from __future__ import annotations

from typing import ClassVar

from sec_generative_search.core.types import PricingTier, ProviderCapability
from sec_generative_search.providers.openai_compat import (
    ModelInfo,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLLMProvider,
)

__all__ = [
    "QwenEmbeddingProvider",
    "QwenProvider",
]


_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting DashScope's Qwen models."""

    provider_name = "qwen"
    default_base_url = _QWEN_BASE_URL
    default_model = "qwen3.5-plus"

    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {
        "qwen3-max": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=262_144,
                max_output_tokens=65_536,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "qwen3.6-plus": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=1_000_000,
                max_output_tokens=65_536,
                pricing_tier=PricingTier.HIGH,
            ),
        ),
        "qwen3.5-plus": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=1_000_000,
                max_output_tokens=65_536,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "qwen3.5-flash": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=1_000_000,
                max_output_tokens=65_536,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "qwen3.5-omni-plus": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=262_144,
                max_output_tokens=65_536,
                pricing_tier=PricingTier.HIGH,
            ),
        ),
        "qwen3.5-omni-flash": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=262_144,
                max_output_tokens=65_536,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "qwen-max": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=32_768,
                max_output_tokens=8_192,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "qwen-plus": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=1_000_000,
                max_output_tokens=32_768,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "qwen-flash": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=1_000_000,
                max_output_tokens=32_768,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "qwen-turbo": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=131_072,
                max_output_tokens=8_192,
                pricing_tier=PricingTier.LOW,
            ),
        ),
    }


class QwenEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Embedding provider for DashScope's ``text-embedding-v*`` models."""

    provider_name = "qwen"
    default_base_url = _QWEN_BASE_URL
    default_model = "text-embedding-v4"

    MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "text-embedding-v4": 1024,
        "text-embedding-v3": 1024,
    }
