"""OpenAI provider adapters.

Concrete subclasses of the OpenAI-compatible bases that target
``api.openai.com`` directly.  No ``base_url`` override; relies on the
SDK's default endpoint.

The ``MODEL_CATALOGUE`` and ``MODEL_DIMENSIONS`` mappings serve as the
cheap capability probe: no extra network round-trip is
needed to know which models support streaming, structured output, and
prompt caching, which keeps registration startup fast and offline-
testable.

Pricing tiers are coarse buckets, not real cost figures.  The exact
spend is reported via :class:`TokenUsage` once a call returns; the tier
exists only so the UI can hint "cheap vs. premium" without consulting
a cost table.
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
    "OpenAIEmbeddingProvider",
    "OpenAIProvider",
]


class OpenAIProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider for OpenAI's hosted models."""

    provider_name = "openai"
    default_model = "gpt-5.4-mini"

    # The catalogue is intentionally narrow — only the families v1
    # supports.  Adding a model later is a one-line addition; the SDK
    # will gracefully refuse unknown slugs at call time so a missing
    # entry never silently degrades.
    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {
        "gpt-5.4-pro": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=1_000_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "gpt-5.4": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=1_000_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "gpt-5.4-mini": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=400_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "gpt-5.4-nano": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=400_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "gpt-5.2-pro": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=1_000_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "gpt-5.2": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=1_000_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "gpt-5.1": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=1_000_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "gpt-5-pro": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=1_000_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "gpt-5": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=1_000_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "gpt-5-mini": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=400_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "gpt-5-nano": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=400_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "gpt-4.1": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=1_000_000,
                max_output_tokens=32_768,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "gpt-4.1-mini": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=1_000_000,
                max_output_tokens=32_768,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "gpt-4.1-nano": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=1_000_000,
                max_output_tokens=32_768,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "gpt-4o": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=128_000,
                max_output_tokens=16_384,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "gpt-4o-mini": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=128_000,
                max_output_tokens=16_384,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "o3": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=200_000,
                max_output_tokens=100_000,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "o4-mini": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=200_000,
                max_output_tokens=100_000,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
    }


class OpenAIEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Embedding provider for OpenAI's hosted ``text-embedding-3-*`` models."""

    provider_name = "openai"
    default_model = "text-embedding-3-small"

    # Native dimensions (OpenAI also supports the ``dimensions`` request
    # parameter for shrinking; v1 sticks to the native sizes for
    # ChromaDB collection stability).
    MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
