"""Factory for embedder construction — the sole seam.

Every wiring site that needs a concrete
:class:`~sec_generative_search.providers.base.BaseEmbeddingProvider`
(storage bootstrap, CLI, API lifespan, tests) routes through
:func:`build_embedder`.  Direct
``OpenAIEmbeddingProvider(...)`` / ``LocalEmbeddingProvider(...)``
instantiation outside this module defeats the credential-resolver seam
that lets callers swap the environment-backed resolver for a session /
admin credential store.

Design notes:

- The resolver is a ``Callable[[str], str | None]`` indexed by the
  registry's provider name (``"openai"``, ``"local"``, ...).  The default
    implementation reads provider-specific environment variables.  A
    store-backed resolver can substitute without changing this signature.
- ``local`` is the only embedding provider that tolerates a ``None`` key
  — its sentinel handles the un-gated model path.  Every other provider
  is hosted and requires a real key; the factory raises
  :class:`ConfigurationError` with an actionable hint when the resolver
  returns ``None`` for a hosted provider.
- Construction flows through
  :meth:`ProviderRegistry.get_class`, reusing the registry's
  optional-extras gate so a user who set ``EMBEDDING_PROVIDER=local``
  without installing ``[local-embeddings]`` gets a clear ``KeyError``
  with install instructions at ``build_embedder`` time (settings load
  is intentionally not where that error surfaces — see the validator
  docstring in :mod:`config.settings`).
- The factory deliberately does not cache the built embedder.  Callers
  are expected to hold the singleton at their own lifespan boundary
  (storage bootstrap, API ``app.state``, CLI process).  A hidden cache
  here would leak across tests and complicate credential rotation.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from sec_generative_search.config.settings import EmbeddingSettings
from sec_generative_search.core.exceptions import ConfigurationError
from sec_generative_search.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
)
from sec_generative_search.providers.registry import (
    ProviderRegistry,
    ProviderSurface,
)

__all__ = [
    "ApiKeyResolver",
    "build_embedder",
    "build_llm_provider",
    "default_api_key_resolver",
]


ApiKeyResolver = Callable[[str], str | None]


# Provider name → environment variable the default resolver reads.
# Unknown names fall through to ``None`` — the factory decides whether
# that is acceptable (only ``local`` is).  Embedding-only entries
# (``local``) and LLM-only entries (``anthropic``, ``deepseek``, …) coexist
# in one table because the registry name is the same on both surfaces; a
# provider that ships both surfaces (``openai``, ``gemini``, ``mistral``,
# ``qwen``) has one env var that the default resolver returns for both.
_DEFAULT_ENV_VAR_BY_PROVIDER: dict[str, str] = {
    # Both surfaces.
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    # Embedding-only.
    "local": "HF_TOKEN",
    # LLM-only.
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "zai": "ZAI_API_KEY",
    "grok": "XAI_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "mimo": "MIMO_API_KEY",
}


def default_api_key_resolver(provider: str) -> str | None:
    """Read the provider's credential from the process environment.

    This environment-backed default can be replaced by any session or
    admin credential store that implements the same
    ``Callable[[str], str | None]`` shape.

    Empty-string env values coerce to ``None`` so a blank
    ``OPENAI_API_KEY=`` never enables a zero-length credential (same
    rule :class:`~sec_generative_search.config.settings.ApiSettings`
    applies to ``API_KEY``).
    """
    env_var = _DEFAULT_ENV_VAR_BY_PROVIDER.get(provider)
    if env_var is None:
        return None
    value = os.environ.get(env_var)
    return value or None


def build_embedder(
    settings: EmbeddingSettings,
    *,
    api_key_resolver: ApiKeyResolver = default_api_key_resolver,
) -> BaseEmbeddingProvider:
    """Construct the embedder selected by *settings*.

    Args:
        settings: The resolved embedding settings.  ``settings.provider``
            is the registry key; ``settings.model_name`` is the model
            slug forwarded to the provider constructor.
        api_key_resolver: Callable that returns the credential for a
            given provider name, or ``None`` when no credential is
            configured.  Defaults to :func:`default_api_key_resolver`
            which reads process environment variables.

    Returns:
        A concrete :class:`BaseEmbeddingProvider` subclass instance with
        the provider's model set.

    Raises:
        KeyError: The provider's optional extras are not installed
            (raised by :meth:`ProviderRegistry.get_class`).
        ConfigurationError: The resolver returned ``None`` for a hosted
            provider — the user must set the expected env var or
            register a credential in the session store.
        ValueError: The model slug is unknown for this provider.

    The returned instance is held at the caller's lifespan — the factory
    never caches.  Direct adapter instantiation outside this function
    and its tests is forbidden; that rule is what makes the
    credential-resolver seam meaningful.
    """
    provider_cls = ProviderRegistry.get_class(settings.provider, ProviderSurface.EMBEDDING)

    api_key = api_key_resolver(settings.provider)
    if api_key is None and settings.provider != "local":
        env_var = _DEFAULT_ENV_VAR_BY_PROVIDER.get(settings.provider, "<unknown>")
        raise ConfigurationError(
            f"No API key resolved for embedding provider '{settings.provider}'. "
            f"Set {env_var} in the environment or provide a custom "
            f"api_key_resolver that returns a credential for this provider."
        )

    # ``local`` accepts ``None`` (uses its internal sentinel); every
    # other provider requires a real key per ``_ProviderBase``.
    return provider_cls(api_key, model=settings.model_name)


def build_llm_provider(
    provider_name: str,
    *,
    api_key_resolver: ApiKeyResolver = default_api_key_resolver,
) -> BaseLLMProvider:
    """Construct the LLM provider named *provider_name*.

    Parallel construction seam to :func:`build_embedder`, so every site
    that needs a concrete :class:`~sec_generative_search.providers.base.BaseLLMProvider`
    (RAG orchestrator wiring, future API lifespan, future CLI shell)
    routes through one resolver-aware factory.  Direct
    ``OpenAIProvider(...)`` / ``AnthropicProvider(...)`` instantiation
    outside this function and its tests is forbidden — that is what
    makes the resolver-chain seam meaningful for user-supplied
    credentials.

    Differences from :func:`build_embedder`:

    - LLM providers do not carry a model on the instance — model
      selection is per-request via ``GenerationRequest.model``.  The
      factory therefore takes only ``provider_name``.
    - There is no ``local`` LLM surface, and therefore no sentinel
      tolerance for a missing credential.  Every LLM provider requires
      a real key; a ``None`` resolver result is always a configuration
      error.

    Args:
        provider_name: Registry key (``"openai"``, ``"anthropic"``,
            ``"openrouter"``, …).
        api_key_resolver: Callable returning the credential for the
            named provider.  Defaults to :func:`default_api_key_resolver`
            (process environment); callers can supply a chained resolver
            via :func:`sec_generative_search.core.credentials.chain_resolvers`.

    Returns:
        A concrete :class:`BaseLLMProvider` instance.

    Raises:
        KeyError: The provider's optional extras are not installed
            (raised by :meth:`ProviderRegistry.get_class`).
        ConfigurationError: The resolver returned ``None`` — every LLM
            provider requires a real key.
    """
    provider_cls = ProviderRegistry.get_class(provider_name, ProviderSurface.LLM)

    api_key = api_key_resolver(provider_name)
    if api_key is None:
        env_var = _DEFAULT_ENV_VAR_BY_PROVIDER.get(provider_name, "<unknown>")
        raise ConfigurationError(
            f"No API key resolved for LLM provider '{provider_name}'. "
            f"Supply the key via the request-scoped resolver chain "
            f"(session / encrypted user store) or set {env_var} in the "
            f"server environment for the admin-default fallback."
        )

    return provider_cls(api_key)
