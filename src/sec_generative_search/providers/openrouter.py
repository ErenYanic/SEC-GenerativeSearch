"""OpenRouter meta-provider adapter.

OpenRouter is a proxy that multiplexes across many upstream providers
using the OpenAI wire protocol at ``https://openrouter.ai/api/v1``.  The
catalogue is open-ended — OpenRouter can gain or lose models daily — so
this adapter deliberately keeps ``MODEL_CATALOGUE`` empty and relies on
the base class's permissive-default branch: unknown slugs yield
``ProviderCapability(chat=True, streaming=True)`` and the SDK rejects
unserviceable slugs at call time with a clear error.

Lazy validation against OpenRouter's model list is the ``models.list`` round-trip that
:meth:`validate_key` already performs via the inherited base behaviour.
A richer "does this slug exist right now?" probe belongs in the registry,
not in this adapter.

OpenRouter slugs use the ``vendor/model`` form (e.g. ``openai/gpt-4o``,
``anthropic/claude-sonnet-4-6``).  :data:`default_model` picks a cheap,
widely-available slug so omitting ``model`` on a request still reaches a
live endpoint.

Upstream-provider routing:

- OpenRouter accepts an optional ``provider`` block in the request body
  that pins, allowlists, or blocklists the upstream providers it routes
  to.  Callers pass :class:`OpenRouterRoutingHints` on
  :class:`~sec_generative_search.providers.base.GenerationRequest` and
  :class:`OpenRouterProvider` forwards the block via the SDK's
  ``extra_body`` kwarg.  Every non-OpenRouter adapter silently ignores
  the hint — the OpenAI-compatible base's :meth:`_extra_request_kwargs`
  hook returns an empty dict by default.
- The hint object is pass-through: no validation, no auth material.  The
  authoritative error surface is OpenRouter's API at call time.  The
  OpenRouter API key remains the sole credential in flight — security
  tests in ``tests/providers/test_openrouter.py`` enforce this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from sec_generative_search.providers.openai_compat import (
    ModelInfo,
    OpenAICompatibleLLMProvider,
)

if TYPE_CHECKING:
    from sec_generative_search.providers.base import GenerationRequest

__all__ = [
    "OpenRouterProvider",
    "OpenRouterRoutingHints",
]


@dataclass(frozen=True)
class OpenRouterRoutingHints:
    """Provider-neutral upstream-routing hints for OpenRouter.

    Forwarded to OpenRouter's ``provider`` request block when attached
    to a :class:`~sec_generative_search.providers.base.GenerationRequest`
    that hits :class:`OpenRouterProvider`.  Every other provider ignores
    the hint — see :meth:`OpenAICompatibleLLMProvider._extra_request_kwargs`.

    Frozen and tuple-valued so the hint object is hashable, immutable,
    and safe to share across threads without defensive copying.
    Values are **pass-through**: this adapter does not validate them.
    An unknown upstream slug or malformed ``data_collection`` value
    surfaces at call time with OpenRouter's own error message, which is
    the authoritative source of truth for what OpenRouter currently
    accepts.

    The field set mirrors OpenRouter's documented ``provider`` block keys:

    - ``order``: preferred upstream order, first match wins.
    - ``allow_fallbacks``: when ``False``, refuse to fall back to
      upstreams outside ``order`` / ``only``.
    - ``only``: allowlist of upstream slugs.
    - ``ignore``: blocklist of upstream slugs.
    - ``require_parameters``: when ``True``, only route to upstreams
      that honour every request parameter (e.g. reasoning, tools).
    - ``data_collection``: ``"allow"`` or ``"deny"`` — refuse upstreams
      that log prompt/response content.

    This object carries no credential-shaped fields by design.  A
    parametrised security test enforces the field set against the same
    credential-hint list used for :class:`GenerationRequest` and the
    provider registry rows.
    """

    order: tuple[str, ...] = ()
    allow_fallbacks: bool | None = None
    only: tuple[str, ...] = ()
    ignore: tuple[str, ...] = ()
    require_parameters: bool | None = None
    data_collection: str | None = None

    def to_provider_block(self) -> dict[str, Any]:
        """Render the hints as the ``provider`` block OpenRouter expects.

        Omitted / empty fields are dropped so OpenRouter's own defaults
        apply — a hint object with no fields set yields an empty dict and
        is equivalent to not attaching hints at all.  Tuples are coerced
        to lists because OpenRouter's JSON schema expects arrays; the
        conversion is safe because :class:`OpenRouterRoutingHints` is
        frozen — we never hand callers a mutable reference to the
        original storage.
        """
        block: dict[str, Any] = {}
        if self.order:
            block["order"] = list(self.order)
        if self.allow_fallbacks is not None:
            block["allow_fallbacks"] = self.allow_fallbacks
        if self.only:
            block["only"] = list(self.only)
        if self.ignore:
            block["ignore"] = list(self.ignore)
        if self.require_parameters is not None:
            block["require_parameters"] = self.require_parameters
        if self.data_collection is not None:
            block["data_collection"] = self.data_collection
        return block


class OpenRouterProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting ``openrouter.ai``.

    Any model slug accepted by OpenRouter works here — the capability
    probe returns a permissive default for every slug not pre-declared
    in :attr:`MODEL_CATALOGUE`.  Callers that need an accurate capability
    matrix for a specific slug should consult the provider registry,
    which caches OpenRouter's own ``/models`` response.

    When the :class:`~sec_generative_search.providers.base.GenerationRequest`
    carries :class:`OpenRouterRoutingHints`, the hints are forwarded into
    OpenRouter's ``provider`` request block via the SDK's ``extra_body``
    kwarg.  No other provider honours the hint — see
    :meth:`OpenAICompatibleLLMProvider._extra_request_kwargs`.
    """

    provider_name = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    default_model = "openai/gpt-5.4-mini"

    # Intentionally empty — see module docstring.  The base class
    # :meth:`get_capabilities` returns a permissive default for any slug
    # not found here, which is the correct semantics for a meta-provider.
    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {}

    def _extra_request_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        """Forward routing hints into OpenRouter's ``provider`` block.

        Returns an empty dict when no hints are attached, keeping the
        common path identical to every other OpenAI-compatible vendor.
        The hint object is rendered via
        :meth:`OpenRouterRoutingHints.to_provider_block` — a pass-through
        copy with empty / ``None`` fields dropped.  No credential is
        added here; the OpenRouter API key already flows through the SDK
        client's ``Authorization`` header and is the sole credential in
        flight.
        """
        hints = request.routing_hints
        if hints is None:
            return {}
        block = hints.to_provider_block()
        if not block:
            return {}
        return {"extra_body": {"provider": block}}
