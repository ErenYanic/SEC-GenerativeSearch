"""Z.ai / Zhipu AI GLM provider adapter (Phase 5G.1).

Z.ai ships an OpenAI-compatible Chat Completions surface at
``https://api.z.ai/api/paas/v4`` (the general-purpose PaaS endpoint; a
separate ``/api/coding/paas/v4`` endpoint exists specifically for the
Coding plan and is intentionally not used here).  Mainland callers use
the ``open.bigmodel.cn`` mirror, which shares the wire protocol.

Only the chat surface is modelled.  Zhipu does ship first-party
embedding models (``embedding-3`` / ``embedding-2``) but the project's
curated embedding surface is intentionally kept narrow — the existing
OpenAI / Gemini / Mistral / Qwen adapters plus the optional
:class:`LocalEmbeddingProvider` already cover the cloud and on-device
dimensions the storage layer requires, and adding another vendor on the
embedding surface would multiply the ChromaDB collection-dimension
matrix without a matching use case.  Pair ``ZaiProvider`` with one of
the existing embedding adapters when using this vendor end-to-end.
"""

from __future__ import annotations

from typing import ClassVar

from sec_generative_search.core.types import PricingTier, ProviderCapability
from sec_generative_search.providers.openai_compat import (
    ModelInfo,
    OpenAICompatibleLLMProvider,
)

__all__ = ["ZaiProvider"]


class ZaiProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting Z.ai's GLM family."""

    provider_name = "zai"
    default_base_url = "https://api.z.ai/api/paas/v4"
    default_model = "glm-4.5-air"

    # Catalogue captures the slugs Z.ai currently serves on the general
    # PaaS endpoint.  ``glm-5.1`` is the post-training upgrade to
    # ``glm-5`` (same 754B-param / 40B-activated MoE, sharper coding);
    # ``glm-4.5-air`` is the cheap-fast default so omitting ``model``
    # on a request still reaches a live endpoint without surprising the
    # caller with premium pricing.
    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {
        "glm-5.1": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=False,
                context_window_tokens=200_000,
                max_output_tokens=98_304,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "glm-5": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=False,
                context_window_tokens=200_000,
                max_output_tokens=98_304,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "glm-4.7": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=False,
                context_window_tokens=200_000,
                max_output_tokens=98_304,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "glm-4.5-air": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=False,
                context_window_tokens=128_000,
                max_output_tokens=8_192,
                pricing_tier=PricingTier.LOW,
            ),
        ),
    }
