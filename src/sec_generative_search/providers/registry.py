"""Provider registry — curated discovery and capability lookup.

The registry is the single source of truth for "which providers ship
with this build, which surfaces (LLM / embedding / reranker) do they
expose, what models do they advertise, and how do I validate a user's
API key against them?".

Design notes:

- The list of providers is a **curated tuple** (:data:`_ENTRIES`), not a
  plug-in surface.  Allowing third-party classes to register themselves
  would weaken the security boundary the rest of the package relies on
  (every provider is reviewed for credential hygiene, exception mapping,
  and ``resilient_call`` use).  Adding a new vendor is a one-line change
  in this file plus the concrete adapter — that is the supported path.

- Indexed by :class:`(name, surface)` because the same provider name can
  back two surfaces (``"openai"`` ships both an LLM and an embedding
  adapter).  Callers ask for one surface at a time.

- The capability probe is **O(1) and credential-free**: it reads the
    static ``MODEL_CATALOGUE`` / ``MODEL_DIMENSIONS`` ClassVars on the
    provider class, never instantiates anything, and never touches the
    network.

- :meth:`ProviderRegistry.validate_key` is the only method that
  instantiates a provider.  It accepts the key as a positional argument
  so the same call signature works for every adapter (including
  :class:`LocalEmbeddingProvider`, whose first positional is the HF
  token).  The provider instance is local to the call and is dropped on
  return — the registry never stores or logs the key.  ``ProviderAuthError``
  collapses to ``False``; every other ``ProviderError`` propagates so a
  network blip is distinguishable from a bad key.

- Optional-extras gating uses :func:`importlib.util.find_spec` — only
  the ``LocalEmbeddingProvider`` entry currently declares an extra
  (``sentence_transformers``).  Probes are cached on the class so the
  cost is paid once per process.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from sec_generative_search.core.exceptions import ProviderAuthError
from sec_generative_search.core.types import ProviderCapability
from sec_generative_search.providers.anthropic import AnthropicProvider
from sec_generative_search.providers.deepseek import DeepSeekProvider
from sec_generative_search.providers.gemini import (
    GeminiEmbeddingProvider,
    GeminiProvider,
)
from sec_generative_search.providers.grok import GrokProvider
from sec_generative_search.providers.kimi import KimiProvider
from sec_generative_search.providers.local import LocalEmbeddingProvider
from sec_generative_search.providers.mimo import MimoProvider
from sec_generative_search.providers.minimax import MiniMaxProvider
from sec_generative_search.providers.mistral import (
    MistralEmbeddingProvider,
    MistralProvider,
)
from sec_generative_search.providers.openai import (
    OpenAIEmbeddingProvider,
    OpenAIProvider,
)
from sec_generative_search.providers.openrouter import OpenRouterProvider
from sec_generative_search.providers.qwen import (
    QwenEmbeddingProvider,
    QwenProvider,
)
from sec_generative_search.providers.zai import ZaiProvider

__all__ = [
    "ProviderEntry",
    "ProviderRegistry",
    "ProviderSurface",
]


# ---------------------------------------------------------------------------
# Surface enum — used to disambiguate ``(name, surface)`` look-ups
# ---------------------------------------------------------------------------


class ProviderSurface(StrEnum):
    """The three provider surfaces.

    Inheriting from :class:`StrEnum` keeps the values trivially
    serialisable for any future API or CLI surface that wants to expose
    them as strings without an extra coercion layer.
    """

    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


# ---------------------------------------------------------------------------
# Curated entry shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderEntry:
    """One curated row in the registry.

    ``provider_cls`` must be the concrete adapter class — the registry
    deliberately does not accept ABCs, mixins, or factories.

    Attributes:
        name: The vendor key, matching ``provider_cls.provider_name``.
            Lower-case by convention (``"openai"``, ``"anthropic"``).
        surface: Which of the three surfaces this entry exposes.
        provider_cls: The concrete provider class.  Stored as :class:`type`
            because the registry-level type system intentionally treats
            LLM / embedding / reranker classes uniformly — callers pick
            the surface up-front, so we do not need a generic.
        requires_extras: Tuple of importable module names that must be
            available for this entry to be usable.  Empty by default;
            populated for entries gated behind an optional-extras install
            (only :class:`LocalEmbeddingProvider` today, gated on
            ``sentence_transformers``).
        supports_arbitrary_models: ``True`` for meta-providers whose
            ``MODEL_CATALOGUE`` is intentionally empty
            (:class:`OpenRouterProvider`).  Tells UIs to render a free-text
            slug input rather than a closed dropdown.
    """

    name: str
    surface: ProviderSurface
    provider_cls: type
    requires_extras: tuple[str, ...] = ()
    supports_arbitrary_models: bool = False


# ---------------------------------------------------------------------------
# Registry — class-level, immutable, no instances
# ---------------------------------------------------------------------------


class ProviderRegistry:
    """Curated registry of every shipped provider.

    The class is intentionally **non-instantiable** in spirit — every
    method is a :func:`classmethod` and there is no per-instance state.
    Callers use it directly as ``ProviderRegistry.list_providers(...)``
    rather than holding an instance.  No ``get_registry()`` accessor; no
    module-level singleton.
    """

    # The single source of truth.  Order is presentation-friendly
    # (popular vendors first within each surface) so the CLI / web UI can
    # render the tuple verbatim if it likes.  Adding a vendor is a one-
    # line addition; removing one requires removing all references in the
    # rest of the package and is therefore intentionally noisy.
    _ENTRIES: ClassVar[tuple[ProviderEntry, ...]] = (
        # --- LLM surface ---
        ProviderEntry("openai", ProviderSurface.LLM, OpenAIProvider),
        ProviderEntry("anthropic", ProviderSurface.LLM, AnthropicProvider),
        ProviderEntry("gemini", ProviderSurface.LLM, GeminiProvider),
        ProviderEntry("deepseek", ProviderSurface.LLM, DeepSeekProvider),
        ProviderEntry("kimi", ProviderSurface.LLM, KimiProvider),
        ProviderEntry("mistral", ProviderSurface.LLM, MistralProvider),
        ProviderEntry("qwen", ProviderSurface.LLM, QwenProvider),
        ProviderEntry("zai", ProviderSurface.LLM, ZaiProvider),
        ProviderEntry("grok", ProviderSurface.LLM, GrokProvider),
        ProviderEntry("minimax", ProviderSurface.LLM, MiniMaxProvider),
        ProviderEntry("mimo", ProviderSurface.LLM, MimoProvider),
        ProviderEntry(
            "openrouter",
            ProviderSurface.LLM,
            OpenRouterProvider,
            supports_arbitrary_models=True,
        ),
        # --- Embedding surface ---
        ProviderEntry("openai", ProviderSurface.EMBEDDING, OpenAIEmbeddingProvider),
        ProviderEntry("gemini", ProviderSurface.EMBEDDING, GeminiEmbeddingProvider),
        ProviderEntry("mistral", ProviderSurface.EMBEDDING, MistralEmbeddingProvider),
        ProviderEntry("qwen", ProviderSurface.EMBEDDING, QwenEmbeddingProvider),
        ProviderEntry(
            "local",
            ProviderSurface.EMBEDDING,
            LocalEmbeddingProvider,
            requires_extras=("sentence_transformers",),
        ),
        # --- Reranker surface ---
        # No first-party reranker ships yet. An empty surface is a
        # valid registry state.
    )

    # Per-process cache of optional-extras availability.  ``find_spec``
    # is cheap (single sys.path scan) but we cache anyway to make
    # repeated UI listings noise-free.
    _availability_cache: ClassVar[dict[str, bool]] = {}

    # ------------------------------------------------------------------
    # Listings
    # ------------------------------------------------------------------

    @classmethod
    def all_entries(
        cls,
        surface: ProviderSurface | None = None,
        *,
        include_unavailable: bool = False,
    ) -> tuple[ProviderEntry, ...]:
        """Return every registered entry, optionally filtered by surface.

        ``include_unavailable=True`` includes entries whose optional
        extras are not installed — useful for the "available providers"
        view in the web UI that wants to display un-installed providers
        with a "requires extra" hint.
        """
        rows = cls._ENTRIES
        if surface is not None:
            rows = tuple(e for e in rows if e.surface == surface)
        if include_unavailable:
            return rows
        return tuple(e for e in rows if cls._is_available(e))

    @classmethod
    def list_providers(cls, surface: ProviderSurface) -> list[str]:
        """Return the names of providers currently usable on *surface*.

        "Currently usable" means the optional extras (if any) are
        importable in the running interpreter.  The list is ordered by
        the registry's curated order, with duplicates removed in case a
        future provider lists itself twice on the same surface (which
        would be a registry bug, but the de-dupe keeps callers safe).
        """
        seen: set[str] = set()
        names: list[str] = []
        for entry in cls.all_entries(surface):
            if entry.name in seen:
                continue
            seen.add(entry.name)
            names.append(entry.name)
        return names

    @classmethod
    def get_entry(cls, name: str, surface: ProviderSurface) -> ProviderEntry:
        """Return the entry for ``(name, surface)``.

        Raises :class:`KeyError` when no entry matches, or when the
        entry's optional extras are not importable in this process.  The
        latter is treated the same as "not registered" because callers
        cannot meaningfully use an entry whose backing SDK is missing.
        """
        for entry in cls._ENTRIES:
            if entry.name == name and entry.surface == surface:
                if not cls._is_available(entry):
                    raise KeyError(
                        f"Provider '{name}' (surface={surface.value}) requires "
                        f"optional extras: {', '.join(entry.requires_extras)}. "
                        f"Install with the appropriate '[extra]' to enable it."
                    )
                return entry
        raise KeyError(f"No provider registered for name='{name}', surface='{surface.value}'.")

    @classmethod
    def get_class(cls, name: str, surface: ProviderSurface) -> type:
        """Return the provider class for ``(name, surface)``.

        Thin wrapper over :meth:`get_entry` for callers that only need
        the class.
        """
        return cls.get_entry(name, surface).provider_cls

    # ------------------------------------------------------------------
    # Models and capabilities (O(1), credential-free)
    # ------------------------------------------------------------------

    @classmethod
    def list_models(cls, name: str, surface: ProviderSurface) -> list[str]:
        """Return the model slugs declared on the provider class.

        For LLM providers this is ``MODEL_CATALOGUE.keys()``; for
        embedding providers it is ``MODEL_DIMENSIONS.keys()``.  The list
        is intentionally **empty** for meta-providers
        (:class:`OpenRouterProvider`) — pair this call with
        :meth:`supports_arbitrary_models` to decide between a dropdown
        and a free-text input in the UI.
        """
        cls_obj = cls.get_class(name, surface)
        if surface is ProviderSurface.LLM:
            catalogue = getattr(cls_obj, "MODEL_CATALOGUE", {})
            return list(catalogue.keys())
        if surface is ProviderSurface.EMBEDDING:
            dimensions = getattr(cls_obj, "MODEL_DIMENSIONS", {})
            return list(dimensions.keys())
        # Reranker surface — no model surface yet, return empty.
        return []

    @classmethod
    def supports_arbitrary_models(cls, name: str, surface: ProviderSurface) -> bool:
        """Whether the provider accepts any slug, not just catalogued ones.

        ``True`` for :class:`OpenRouterProvider`; ``False`` for every
        provider that ships a closed catalogue.  UIs should render a
        free-text input when this returns ``True``.
        """
        return cls.get_entry(name, surface).supports_arbitrary_models

    @classmethod
    def get_capability(
        cls,
        name: str,
        surface: ProviderSurface,
        model: str | None = None,
    ) -> ProviderCapability:
        """Return the static capability matrix for ``(name, surface, model)``.

        Reads the provider class's static catalogue directly — never
        instantiates the class, never makes a network call, never needs
        an API key.  This is the canonical pre-flight probe used by
        routing code before generation.

        For LLM providers, unknown slugs receive the same permissive
        ``ProviderCapability(chat=True, streaming=True)`` default that
        the provider itself returns at call time — except for OpenRouter,
        where unknown slugs are *expected* (the catalogue is empty by
        design).  For embedding providers, an unknown slug is rejected
        with :class:`ValueError` because the collection dimension must be
        known up front (ChromaDB collections are dimension-locked on
        creation).
        """
        entry = cls.get_entry(name, surface)
        cls_obj = entry.provider_cls

        if surface is ProviderSurface.LLM:
            slug = model or getattr(cls_obj, "default_model", "") or ""
            catalogue: dict[str, Any] = getattr(cls_obj, "MODEL_CATALOGUE", {})
            info = catalogue.get(slug)
            if info is not None:
                return info.capability
            # OpenRouter (or any future arbitrary-models provider) is
            # *expected* to fall through here for almost every slug.
            # Other providers fall through only for unknown / freshly-
            # released slugs — same permissive default as their own
            # ``get_capabilities``.
            return ProviderCapability(chat=True, streaming=True)

        if surface is ProviderSurface.EMBEDDING:
            slug = model or getattr(cls_obj, "default_model", "") or ""
            dimensions: dict[str, int] = getattr(cls_obj, "MODEL_DIMENSIONS", {})
            if slug not in dimensions:
                raise ValueError(
                    f"Unknown embedding model '{slug}' for {name}. "
                    f"Known models: {sorted(dimensions.keys())}."
                )
            return ProviderCapability(embeddings=True)

        # Reranker — no first-party adapters yet. A future concrete
        # reranker class will surface its own capability.
        return ProviderCapability()

    @classmethod
    def get_dimension(cls, name: str, model: str | None = None) -> int:
        """Return the embedding dimension for ``(name, model)``.

        Convenience wrapper for the storage layer, which needs to stamp
        the collection dimension before any embed call. Always
        operates on the embedding surface; raises :class:`ValueError` for
        unknown slugs (same contract as :meth:`get_capability`).
        """
        entry = cls.get_entry(name, ProviderSurface.EMBEDDING)
        cls_obj = entry.provider_cls
        slug = model or getattr(cls_obj, "default_model", "") or ""
        dimensions: dict[str, int] = getattr(cls_obj, "MODEL_DIMENSIONS", {})
        if slug not in dimensions:
            raise ValueError(
                f"Unknown embedding model '{slug}' for {name}. "
                f"Known models: {sorted(dimensions.keys())}."
            )
        return dimensions[slug]

    # ------------------------------------------------------------------
    # Key validation — the only path that instantiates a provider
    # ------------------------------------------------------------------

    @classmethod
    def validate_key(
        cls,
        name: str,
        surface: ProviderSurface,
        api_key: str,
        *,
        model: str | None = None,
    ) -> bool:
        """Validate *api_key* against ``(name, surface)``.

        Returns ``True`` when the provider accepts the key.  Returns
        ``False`` specifically when the provider rejects the key
        (:class:`ProviderAuthError`) — that mapping lets callers render
        a "wrong key" UI without distinguishing the underlying SDK
        exception.

        Every other :class:`ProviderError` subclass propagates: a
        rate-limit, a timeout, or a content-filter event is *not* a
        verdict on the key, and forcing it into ``False`` would mislead
        the caller into rotating a working key.

        The provider instance is local to this call and is dropped on
        return.  The registry never logs the key, never stores it on a
        class attribute, and never re-raises an exception that carries
        the key in its message — see :class:`_ProviderBase`'s ``__repr__``
        contract for the underlying guarantee.
        """
        entry = cls.get_entry(name, surface)

        try:
            provider = cls._construct(entry, api_key, model=model)
        except ProviderAuthError:
            # Some providers fail-fast on a structurally invalid key at
            # construction time — count that as a failed validation
            # rather than letting it bubble.
            return False

        try:
            provider.validate_key()
        except ProviderAuthError:
            return False
        # Every other exception (ProviderRateLimitError, ProviderTimeoutError,
        # ProviderContentFilterError, generic ProviderError, transport
        # errors) propagates intentionally — see the docstring.
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _construct(
        cls,
        entry: ProviderEntry,
        api_key: str,
        *,
        model: str | None = None,
    ) -> Any:
        """Instantiate the provider for *entry* with *api_key*.

        Uses positional ``api_key`` so the same call works for every
        adapter, including :class:`LocalEmbeddingProvider` whose first
        positional parameter is ``hf_token`` rather than ``api_key``.
        Model is forwarded as a keyword for embedding providers (which
        require it at construction) and ignored for LLM / reranker
        providers (where the model is per-request, not per-instance).
        """
        if entry.surface is ProviderSurface.EMBEDDING:
            kwargs: dict[str, Any] = {}
            if model is not None:
                kwargs["model"] = model
            return entry.provider_cls(api_key, **kwargs)
        # LLM and reranker providers carry no per-instance model.
        return entry.provider_cls(api_key)

    @classmethod
    def _is_available(cls, entry: ProviderEntry) -> bool:
        """Whether every optional extra declared by *entry* is importable."""
        return all(cls._has_module(name) for name in entry.requires_extras)

    @classmethod
    def _has_module(cls, module_name: str) -> bool:
        """Cached :func:`importlib.util.find_spec` probe."""
        cached = cls._availability_cache.get(module_name)
        if cached is not None:
            return cached
        present = importlib.util.find_spec(module_name) is not None
        cls._availability_cache[module_name] = present
        return present

    @classmethod
    def _reset_availability_cache(cls) -> None:
        """Test-only seam: clear the cached :func:`find_spec` results.

        Public so the unit tests can monkeypatch ``find_spec`` and
        re-run the discovery logic deterministically.  Production code
        has no reason to call this.
        """
        cls._availability_cache.clear()
