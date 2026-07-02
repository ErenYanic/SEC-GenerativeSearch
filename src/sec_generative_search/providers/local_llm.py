"""Local-LLM provider adapter.

A self-hosted model server the operator runs on their own machine —
Ollama (the default), llama.cpp's ``llama-server``, vLLM, or LM Studio —
each of which exposes an OpenAI-compatible Chat Completions surface.  The
adapter is therefore a thin :class:`OpenAICompatibleLLMProvider` subclass:
it inherits the shared client construction, :func:`resilient_call`
plumbing, streaming, token accounting, and SDK exception mapping
unchanged, and differs only by ``provider_name`` and the loopback
``default_base_url`` / ``default_model`` an Ollama install ships with.

The provider ships with **no optional extra** — the ``openai`` SDK is a
core dependency, so ``local_llm`` is always available on the LLM surface.

Privacy framing: "local" means *a model server you control*, not "no
prompt ever leaves the process".  Generation still sends the assembled
prompt — including the user's question (Tier-3 data) and the retrieved
filing context — over the wire to that endpoint.  The privacy win is that
the endpoint is one the operator runs (loopback by default), so the
prompt need not reach a third-party hosted provider.  The base URL that
decides *where* the prompt goes is settings-driven (``LOCAL_LLM_BASE_URL``)
and host-policy-guarded: the adapter re-reads it at construction and fails
closed to the shipped loopback default, so a torn-down or off-policy config
never silently ships the prompt off-box.

The endpoint accepts any model slug the local server has pulled, so the
provider is registered OpenRouter-style — ``supports_arbitrary_models``
with an empty vendored catalogue — and (like every non-OpenRouter vendor)
silently drops upstream-routing hints via the OpenAI-compatible base's
empty default :meth:`_extra_request_kwargs` hook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.types import ProviderCapability
from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleLLMProvider,
)

if TYPE_CHECKING:
    from sec_generative_search.core.resilience import RetryPolicy

__all__ = ["LocalLLMProvider"]

logger = get_logger(__name__)


class LocalLLMProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting a self-hosted OpenAI-wire endpoint.

    Defaults target a stock Ollama install on the loopback interface;
    other backends (llama.cpp-server, vLLM, LM Studio) work by pointing the
    base URL at their own OpenAI-compatible port.  The capability probe
    returns a FREE (``0.0``-cost) capability for every slug because the
    served model set is whatever the local server has pulled — an endpoint
    the operator hosts costs nothing per token, and the SDK surfaces an
    unserviceable slug at call time with the endpoint's own error.

    The endpoint URL is operator-owned *deployment* configuration, not a
    per-request / per-user credential, so the provider re-reads
    :class:`~sec_generative_search.config.settings.LocalLLMSettings`
    standalone at construction (``LOCAL_LLM_BASE_URL`` /
    ``LOCAL_LLM_DEFAULT_MODEL``) rather than routing through the credential
    resolver chain.  Resolution **fails closed to the shipped loopback
    default**: any error reading or validating those settings (chiefly a
    non-loopback ``LOCAL_LLM_BASE_URL`` without ``LOCAL_LLM_ALLOW_NON_LOCAL``)
    falls back to the loopback endpoint so a torn-down config never silently
    sends the assembled prompt (Tier-3 data) off-box.  The host-policy guard
    that decides whether a non-loopback URL is permitted lives in
    ``LocalLLMSettings`` itself.
    """

    provider_name = "local_llm"
    default_base_url = "http://127.0.0.1:11434/v1"
    default_model = "llama3.2"

    # Intentionally absent from the vendored catalogue — the served model
    # set is operator-defined, so the registry lists it OpenRouter-style
    # (``supports_arbitrary_models``) and this adapter supplies the FREE
    # capability for any slug the local server accepts.

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        """Construct the adapter, resolving the endpoint from settings.

        An explicitly-passed *base_url* (the registry's credential-free
        ``validate_key`` construction, a test, or a future caller) always
        wins.  Otherwise the base URL and default model are read from
        :class:`LocalLLMSettings`, failing closed to the loopback default —
        see the class docstring.
        """
        resolved_base_url, resolved_model = self._resolve_endpoint()
        effective_base_url = base_url if base_url is not None else resolved_base_url
        super().__init__(
            api_key,
            base_url=effective_base_url,
            timeout=timeout,
            retry_policy=retry_policy,
        )
        # Per-instance default model from ``LOCAL_LLM_DEFAULT_MODEL``, used as
        # the fallback when a request carries no explicit slug.  The class
        # attribute deliberately stays the shipped ``"llama3.2"`` so the
        # registry's credential-free, class-level capability probe is
        # unaffected (it reads the class, never an instance).
        self.default_model = resolved_model

    @classmethod
    def _resolve_endpoint(cls) -> tuple[str, str]:
        """Return ``(base_url, default_model)`` from settings, fail-closed.

        Re-reads :class:`LocalLLMSettings` standalone so an operator's
        ``LOCAL_LLM_BASE_URL`` / ``LOCAL_LLM_DEFAULT_MODEL`` take effect at
        provider construction.  On **any** error — an invalid URL, a
        non-loopback host without the opt-in, a torn-down environment — the
        shipped loopback class defaults are returned instead of propagating,
        so the prompt is never sent to an unintended endpoint.  The
        settings-load validator remains the loud, fail-fast surface for a
        genuine misconfiguration; this fallback is defence-in-depth.
        """
        try:
            # Local import keeps the providers ⇄ settings edge one-directional
            # at module-import time and avoids any construction-order cycle.
            from sec_generative_search.config.settings import LocalLLMSettings

            settings = LocalLLMSettings()
        except Exception:
            logger.warning(
                "Could not load LOCAL_LLM_* endpoint settings; falling back to "
                "the loopback default. Check LOCAL_LLM_BASE_URL / "
                "LOCAL_LLM_ALLOW_NON_LOCAL."
            )
            return cls.default_base_url, cls.default_model
        return settings.base_url, settings.default_model

    def get_capabilities(self, model: str | None = None) -> ProviderCapability:
        """Return the FREE capability matrix for any served slug.

        The vendored catalogue is empty by design, so every slug is
        uncatalogued.  A self-hosted endpoint costs the operator nothing
        per API token, so both per-MTok costs are reported as ``0.0``:
        :class:`~sec_generative_search.core.types.ProviderCapability`
        derives the tier from that (→ :attr:`PricingTier.FREE`) rather than
        accepting a hand-assigned tier, and
        :func:`~sec_generative_search.core.types.estimate_cost` returns
        ``$0.00`` instead of the honest-UNKNOWN ``None``.  This mirrors the
        ``free_tier`` branch of :meth:`ProviderRegistry.get_capability`, so
        the credential-free registry probe and this instance probe never
        disagree.
        """
        del model  # served model set is operator-defined; cost is 0.0 for any slug
        return ProviderCapability(
            chat=True,
            streaming=True,
            input_cost_per_mtok=0.0,
            output_cost_per_mtok=0.0,
        )
