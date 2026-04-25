"""DeepSeek provider adapter.

DeepSeek's API is OpenAI-compatible at ``https://api.deepseek.com/v1``.
Only the chat surface is exposed — DeepSeek does not ship a first-party
embedding API, so pair with a separate embedding provider when using
this adapter end-to-end.
"""

from __future__ import annotations

from typing import ClassVar

from sec_generative_search.core.types import PricingTier, ProviderCapability
from sec_generative_search.providers.openai_compat import (
    ModelInfo,
    OpenAICompatibleLLMProvider,
)

__all__ = ["DeepSeekProvider"]


class DeepSeekProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting DeepSeek's hosted models."""

    provider_name = "deepseek"
    default_base_url = "https://api.deepseek.com"
    default_model = "deepseek-v4-flash"

    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {
        "deepseek-v4-flash": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                context_window_tokens=1_000_000,
                max_output_tokens=384_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
        "deepseek-v4-pro": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=False,
                structured_output=False,
                prompt_caching=True,
                context_window_tokens=1_000_000,
                max_output_tokens=384_000,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
    }
