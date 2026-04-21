"""Xiaomi MiMo provider adapter.

Xiaomi ships an OpenAI-compatible Chat Completions surface at
``https://api.xiaomimimo.com/v1``.  The service sits behind the
``platform.xiaomimimo.com`` token-plan portal and uses standard OpenAI
``/chat/completions`` paths — the SDK works unchanged via a
``base_url`` override.

The catalogue covers the text-only MiMo-V2 slugs.  ``MiMo-V2-Omni``
(multimodal) and ``MiMo-V2-TTS`` (speech synthesis) are intentionally
excluded: the project's current surface is grounded SEC text analysis
and neither modality contributes to that pipeline today. If voice or
vision support is added to the orchestrator, those slugs can
be added with the appropriate capability flags.
"""

from __future__ import annotations

from typing import ClassVar

from sec_generative_search.core.types import PricingTier, ProviderCapability
from sec_generative_search.providers.openai_compat import (
    ModelInfo,
    OpenAICompatibleLLMProvider,
)

__all__ = ["MimoProvider"]


class MimoProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting Xiaomi's MiMo API."""

    provider_name = "mimo"
    default_base_url = "https://api.xiaomimimo.com/v1"
    default_model = "MiMo-V2-Pro"

    # Xiaomi's published slugs preserve the ``MiMo-V2-*`` casing — the
    # API is case-sensitive. ``MiMo-V2-Pro`` is the reasoning-capable
    # flagship with a 1M-token context window; ``MiMo-V2-Flash`` remains
    # available as the cheaper fast tier.
    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {
        "MiMo-V2-Pro": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=False,
                context_window_tokens=1_048_576,
                max_output_tokens=32_000,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "MiMo-V2-Flash": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=False,
                context_window_tokens=262_144,
                max_output_tokens=8_192,
                pricing_tier=PricingTier.LOW,
            ),
        ),
    }
