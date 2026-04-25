"""Mistral provider adapters.

Mistral's La Plateforme exposes an OpenAI-compatible Chat Completions
and Embeddings surface at ``https://api.mistral.ai/v1``.  Both the chat
and embedding surfaces subclass the shared OpenAI-compatible bases and
differ only in ``provider_name``, ``default_base_url``, ``default_model``,
and the static catalogue that drives the O(1) capability probe.

Pricing tiers are coarse buckets, not real cost figures (same pattern as
:mod:`providers.openai`); the tier exists so the UI can hint
"cheap vs. premium" without consulting a per-token cost table.
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
    "MistralEmbeddingProvider",
    "MistralProvider",
]


class MistralProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting ``api.mistral.ai``."""

    provider_name = "mistral"
    default_base_url = "https://api.mistral.ai/v1"
    default_model = "ministral-small-2603"

    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {
        "magistral-medium-2509": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=128_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.HIGH,
            ),
        ),
        "magistral-small-2509": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=128_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "mistral-large-2512": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=256_000,
                max_output_tokens=256_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "mistral-medium-2508": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=128_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "mistral-small-2603": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=256_000,
                max_output_tokens=256_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "mistral-small-2506": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=256_000,
                max_output_tokens=256_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "ministral-14b-2512": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=256_000,
                max_output_tokens=256_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "ministral-8b-2512": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=256_000,
                max_output_tokens=256_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "ministral-3b-2512": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=256_000,
                max_output_tokens=256_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "codestral-2508": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=128_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "devstral-2512": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=256_000,
                max_output_tokens=256_000,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
    }


class MistralEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Embedding provider for Mistral's hosted embedding models."""

    provider_name = "mistral"
    default_base_url = "https://api.mistral.ai/v1"
    default_model = "mistral-embed"

    MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "mistral-embed": 1024,
        "codestral-embed": 1536,
    }
