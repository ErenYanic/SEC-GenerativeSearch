"""Tests for the vendored model catalogue.

Covers:

- Baseline load from package data (``providers/data/model_catalogue.json``).
- ``capability_from_row`` field mapping: surface invariants, allow-list,
  cost validation, cost-derived pricing tier.
- Lookups (``get_llm_capability`` / ``list_llm_models``) + the
  ``set_catalogue`` / ``reset_catalogue`` / ``with_provider`` seam.
- The load-bearing **baseline invariant** (``@pytest.mark.security``): every
    vendored model carries exact cost and a non-``UNKNOWN`` derived tier. This
    stays locked for the bundled baseline even if future overlay models are
    allowed to remain ``UNKNOWN``.
"""

from __future__ import annotations

import json
import math
from importlib import resources

import pytest

from sec_generative_search.core.types import PricingTier, ProviderCapability
from sec_generative_search.providers.catalogue import (
    ModelCatalogue,
    capability_from_row,
    model_catalogue,
    reset_catalogue,
    set_catalogue,
)


@pytest.fixture(autouse=True)
def _reset_active_catalogue() -> None:
    """Each test starts and ends on the bundled baseline."""
    reset_catalogue()
    yield
    reset_catalogue()


# ---------------------------------------------------------------------------
# Baseline load + package data
# ---------------------------------------------------------------------------


class TestBaselineLoad:
    def test_baseline_resource_is_packaged(self) -> None:
        # The data file must ship as package data so the wheel / Docker
        # image can load it; a missing carve-out would break this.
        resource = resources.files("sec_generative_search.providers").joinpath(
            "data", "model_catalogue.json"
        )
        assert resource.is_file()

    def test_meta_block_present(self) -> None:
        text = (
            resources.files("sec_generative_search.providers")
            .joinpath("data", "model_catalogue.json")
            .read_text(encoding="utf-8")
        )
        document = json.loads(text)
        assert document["_meta"]["schema_version"] == 1
        # The tier is derived, never stored — guard against a regression
        # that re-introduces a second pricing table in the data file.
        for models in document["providers"].values():
            for row in models.values():
                assert "pricing_tier" not in row

    def test_load_baseline_has_known_providers_and_models(self) -> None:
        cat = ModelCatalogue.load_baseline()
        assert cat.has_provider("openai")
        assert cat.has_provider("anthropic")
        assert "gpt-4o" in cat.list_llm_models("openai")
        # OpenRouter is intentionally absent (arbitrary-slug meta-provider).
        assert not cat.has_provider("openrouter")
        assert cat.list_llm_models("openrouter") == []

    def test_model_catalogue_is_cached(self) -> None:
        assert model_catalogue() is model_catalogue()


# ---------------------------------------------------------------------------
# capability_from_row
# ---------------------------------------------------------------------------


class TestCapabilityFromRow:
    def test_surface_invariants_applied(self) -> None:
        cap = capability_from_row({})
        assert cap.chat is True
        assert cap.embeddings is False

    def test_full_row_maps_every_field(self) -> None:
        cap = capability_from_row(
            {
                "streaming": True,
                "tool_use": True,
                "structured_output": True,
                "prompt_caching": True,
                "vision": True,
                "context_window_tokens": 400_000,
                "max_output_tokens": 128_000,
                "input_cost_per_mtok": 0.9,
                "output_cost_per_mtok": 3.6,
            }
        )
        assert cap.streaming and cap.tool_use and cap.structured_output
        assert cap.prompt_caching and cap.vision
        assert cap.context_window_tokens == 400_000
        assert cap.max_output_tokens == 128_000
        assert cap.input_cost_per_mtok == 0.9
        assert cap.output_cost_per_mtok == 3.6
        # blended (0.9 + 3.6) / 2 = 2.25 -> STANDARD
        assert cap.pricing_tier is PricingTier.STANDARD

    def test_missing_booleans_default_false(self) -> None:
        cap = capability_from_row({"input_cost_per_mtok": 0.1, "output_cost_per_mtok": 0.1})
        assert cap.tool_use is False
        assert cap.vision is False

    def test_absent_cost_is_unknown_tier(self) -> None:
        cap = capability_from_row({"context_window_tokens": 100})
        assert cap.input_cost_per_mtok is None
        assert cap.output_cost_per_mtok is None
        assert cap.pricing_tier is PricingTier.UNKNOWN

    def test_unknown_keys_are_ignored(self) -> None:
        # The lift is an explicit allow-list: an unexpected key (e.g. one a
        # future overlay payload might smuggle) must not reach the dataclass.
        cap = capability_from_row(
            {"input_cost_per_mtok": 0.1, "output_cost_per_mtok": 0.1, "embeddings": True, "evil": 1}
        )
        # ``embeddings`` is a surface invariant, never read from the row.
        assert cap.embeddings is False

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            capability_from_row({"input_cost_per_mtok": -1.0, "output_cost_per_mtok": 1.0})


# ---------------------------------------------------------------------------
# Lookups + swap seam
# ---------------------------------------------------------------------------


class TestLookupsAndSeam:
    def test_get_llm_capability_unknown_returns_none(self) -> None:
        assert model_catalogue().get_llm_capability("openai", "no-such-slug") is None
        assert model_catalogue().get_llm_capability("no-such-provider", "x") is None

    def test_with_provider_does_not_mutate_original(self) -> None:
        base = model_catalogue()
        extra = ProviderCapability(chat=True, streaming=True)
        layered = base.with_provider("demo-vendor", {"demo-chat": extra})
        assert layered.get_llm_capability("demo-vendor", "demo-chat") is extra
        # Original instance is untouched.
        assert base.get_llm_capability("demo-vendor", "demo-chat") is None

    def test_set_and_reset_catalogue(self) -> None:
        custom = ModelCatalogue({"demo-vendor": {"demo-chat": ProviderCapability(chat=True)}})
        set_catalogue(custom)
        assert model_catalogue() is custom
        reset_catalogue()
        # Reload yields the baseline again (a fresh instance, not ``custom``).
        assert model_catalogue() is not custom
        assert model_catalogue().has_provider("openai")


# ---------------------------------------------------------------------------
# Baseline invariant (security lock)
# ---------------------------------------------------------------------------


class TestBaselineInvariant:
    @pytest.mark.security
    def test_every_baseline_model_has_cost_and_non_unknown_tier(self) -> None:
        """Vendored baseline rows must carry exact cost + a real tier.

        Future overlay models may be ``UNKNOWN``, but the **bundled
        baseline** stays fully priced — it is the metrics ``pricing_tier``
        label + UI cheap-vs-premium source.
        """
        cat = ModelCatalogue.load_baseline()
        document = json.loads(
            resources.files("sec_generative_search.providers")
            .joinpath("data", "model_catalogue.json")
            .read_text(encoding="utf-8")
        )
        offenders: list[str] = []
        for provider, models in document["providers"].items():
            for slug in models:
                cap = cat.get_llm_capability(provider, slug)
                assert cap is not None
                if cap.input_cost_per_mtok is None or cap.output_cost_per_mtok is None:
                    offenders.append(f"{provider}/{slug}: missing cost")
                elif not (
                    math.isfinite(cap.input_cost_per_mtok)
                    and math.isfinite(cap.output_cost_per_mtok)
                ):
                    offenders.append(f"{provider}/{slug}: non-finite cost")
                elif cap.pricing_tier is PricingTier.UNKNOWN:
                    offenders.append(f"{provider}/{slug}: UNKNOWN tier")
        assert not offenders, offenders
