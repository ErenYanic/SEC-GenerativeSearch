"""OpenRouter meta-provider adapter (Phase 5D.5).

OpenRouter is a proxy that multiplexes across many upstream providers
using the OpenAI wire protocol at ``https://openrouter.ai/api/v1``.  The
catalogue is open-ended — OpenRouter can gain or lose models daily — so
this adapter deliberately keeps ``MODEL_CATALOGUE`` empty and relies on
the base class's permissive-default branch: unknown slugs yield
``ProviderCapability(chat=True, streaming=True)`` and the SDK rejects
unserviceable slugs at call time with a clear error.

Phase 5D.5 explicitly calls for "lazy validation against OpenRouter's
model list" — that validation is the ``models.list`` round-trip that
:meth:`validate_key` already performs via the inherited base behaviour.
A richer "does this slug exist right now?" probe belongs in the Phase 5F
registry, not in this adapter.

OpenRouter slugs use the ``vendor/model`` form (e.g. ``openai/gpt-4o``,
``anthropic/claude-sonnet-4-6``).  :data:`default_model` picks a cheap,
widely-available slug so omitting ``model`` on a request still reaches a
live endpoint.
"""

from __future__ import annotations

from typing import ClassVar

from sec_generative_search.providers.openai_compat import (
    ModelInfo,
    OpenAICompatibleLLMProvider,
)

__all__ = ["OpenRouterProvider"]


class OpenRouterProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting ``openrouter.ai``.

    Any model slug accepted by OpenRouter works here — the capability
    probe returns a permissive default for every slug not pre-declared
    in :attr:`MODEL_CATALOGUE`.  Callers that need an accurate capability
    matrix for a specific slug should consult the Phase 5F provider
    registry, which caches OpenRouter's own ``/models`` response.
    """

    provider_name = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    default_model = "openai/gpt-4o-mini"

    # Intentionally empty — see module docstring.  The base class
    # :meth:`get_capabilities` returns a permissive default for any slug
    # not found here, which is the correct semantics for a meta-provider.
    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {}
