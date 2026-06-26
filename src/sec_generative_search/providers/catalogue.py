"""Vendored model catalogue — single source of LLM capability + cost data.

The bundled JSON baseline (``data/model_catalogue.json``) is loaded once and
exposes O(1), credential-free, network-free lookups consumed by:

- :class:`~sec_generative_search.providers.registry.ProviderRegistry` — the
  classmethod capability probe used by the read routes / CLI, and
- the concrete LLM adapters' instance ``get_capabilities`` (e.g. the
  orchestrator's context-budgeting probe).

The pricing tier is **never** stored in the data file: it is derived from
exact cost by :class:`~sec_generative_search.core.types.ProviderCapability`
at construction (:func:`~sec_generative_search.core.types.derive_pricing_tier`).
``chat=True`` / ``embeddings=False`` are LLM-surface invariants applied by the
loader, not stored per row.

The active catalogue is a process-global swapped via :func:`set_catalogue` /
:func:`reset_catalogue`.  Tests use that seam to register fixture providers.
Nothing here touches the network or a credential — the refresh that *fetches*
an overlay is a separate, opt-in seam.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources
from typing import Any, Final

from sec_generative_search.core.types import ProviderCapability

__all__ = [
    "ModelCatalogue",
    "capability_from_row",
    "model_catalogue",
    "reset_catalogue",
    "set_catalogue",
]

# The baseline ships as package data under ``providers/data/``.  Referencing
# the parent package (rather than a ``data`` subpackage) keeps the data tree
# free of an ``__init__.py``; ``Traversable.joinpath`` descends into the plain
# directory.
_DATA_PACKAGE: Final = "sec_generative_search.providers"
_BASELINE_DIR: Final = "data"
_BASELINE_NAME: Final = "model_catalogue.json"

# Allow-listed capability fields a data-file row may carry.  ``chat`` /
# ``embeddings`` are LLM-surface invariants applied below, never read from the
# row, so an overlay payload cannot flip a row onto the embedding surface.
_BOOL_FIELDS: Final = (
    "streaming",
    "tool_use",
    "structured_output",
    "prompt_caching",
    "vision",
)
_INT_FIELDS: Final = ("context_window_tokens", "max_output_tokens")
_COST_FIELDS: Final = ("input_cost_per_mtok", "output_cost_per_mtok")


def capability_from_row(row: Mapping[str, Any]) -> ProviderCapability:
    """Map one catalogue row to a :class:`ProviderCapability` (LLM surface).

    ``chat=True`` / ``embeddings=False`` are applied here as surface
    invariants.  Cost validation (finite, non-negative) and pricing-tier
    derivation happen inside :meth:`ProviderCapability.__post_init__`; this
    function never assigns a tier.

    The lift is an explicit allow-list — keys outside
    ``_BOOL_FIELDS`` / ``_INT_FIELDS`` / ``_COST_FIELDS`` are ignored, so a
    future field in an overlay payload cannot smuggle state into the
    capability matrix.
    """
    kwargs: dict[str, Any] = {"chat": True}
    for field in _BOOL_FIELDS:
        kwargs[field] = bool(row.get(field, False))
    for field in _INT_FIELDS:
        value = row.get(field, 0)
        kwargs[field] = int(value) if value is not None else 0
    for field in _COST_FIELDS:
        value = row.get(field)
        kwargs[field] = float(value) if value is not None else None
    return ProviderCapability(**kwargs)


class ModelCatalogue:
    """Immutable in-memory map of LLM capabilities, keyed by (provider, slug).

    Built once from the bundled baseline.  Lookups are plain nested-dict
    access — O(1), no I/O, no credential.
    """

    __slots__ = ("_by_provider",)

    def __init__(self, by_provider: Mapping[str, Mapping[str, ProviderCapability]]) -> None:
        self._by_provider: dict[str, dict[str, ProviderCapability]] = {
            provider: dict(models) for provider, models in by_provider.items()
        }

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_llm_capability(self, provider: str, slug: str) -> ProviderCapability | None:
        """Return the capability for *(provider, slug)*, or ``None`` if absent."""
        return self._by_provider.get(provider, {}).get(slug)

    def list_llm_models(self, provider: str) -> list[str]:
        """Return *provider*'s catalogued slugs in declaration order."""
        return list(self._by_provider.get(provider, {}))

    def has_provider(self, provider: str) -> bool:
        """Whether *provider* has at least one catalogued model."""
        return provider in self._by_provider

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_rows(cls, providers: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> ModelCatalogue:
        """Build from raw data-file rows (``{provider: {slug: row}}``)."""
        built = {
            provider: {slug: capability_from_row(row) for slug, row in models.items()}
            for provider, models in providers.items()
        }
        return cls(built)

    @classmethod
    def load_baseline(cls) -> ModelCatalogue:
        """Load the bundled JSON baseline shipped as package data."""
        text = (
            resources.files(_DATA_PACKAGE)
            .joinpath(_BASELINE_DIR, _BASELINE_NAME)
            .read_text(encoding="utf-8")
        )
        document = json.loads(text)
        providers = document.get("providers", {})
        return cls.from_rows(providers)

    def with_provider(
        self, provider: str, models: Mapping[str, ProviderCapability]
    ) -> ModelCatalogue:
        """Return a new catalogue with *provider*'s models added / overridden.

        Used by the catalogue swap seam and by tests to layer additional
        providers onto the baseline without mutating the shared instance.
        """
        merged = {p: dict(m) for p, m in self._by_provider.items()}
        merged.setdefault(provider, {}).update(models)
        return ModelCatalogue(merged)


_active: ModelCatalogue | None = None


def model_catalogue() -> ModelCatalogue:
    """Return the active catalogue, lazily loading the baseline on first use.

    The lazy load keeps importing this module side-effect-free; the file is
    read once, at the first capability probe (effectively startup), and cached
    for the process.
    """
    global _active
    if _active is None:
        _active = ModelCatalogue.load_baseline()
    return _active


def set_catalogue(catalogue: ModelCatalogue) -> None:
    """Install *catalogue* as the active one (refresh-overlay swap / tests)."""
    global _active
    _active = catalogue


def reset_catalogue() -> None:
    """Drop the active catalogue so the next access reloads the baseline."""
    global _active
    _active = None
