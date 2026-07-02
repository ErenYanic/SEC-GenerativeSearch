"""Tests for the local-LLM provider adapter.

``LocalLLMProvider`` is a thin :class:`OpenAICompatibleLLMProvider`
subclass — the shared generation / streaming / token-accounting /
exception-mapping behaviour is exhaustively covered in
``test_openai_compat``.  This suite pins the contract that is specific to
the local provider:

- the class declarations (name, loopback base URL, default model);
- it is registered on the LLM surface OpenRouter-style — an empty vendored
  catalogue with ``supports_arbitrary_models=True`` and
  ``supports_upstream_routing=False``;
- the O(1) capability probe accepts any slug as a FREE (``0.0``-cost)
  capability and never makes an SDK round-trip;
- FREE pricing is cost-derived (0.0 cost → ``PricingTier.FREE`` →
  ``estimate_cost`` returns ``$0.00``) and the registry's credential-free
  probe agrees with the adapter's instance probe;
- ``build_llm_provider`` tolerates a missing credential for this provider
  only, passing a non-secret sentinel;
- the loopback base URL is wired into the OpenAI client unchanged, with
  the SDK's own retry loop disabled (``resilient_call`` owns retry);
- security: the SDK key never appears in the provider's ``repr``.

Every test stubs the ``OpenAI`` SDK factory — no network call is issued
and no live endpoint is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from sec_generative_search.core.types import (
    PricingTier,
    ProviderCapability,
    TokenUsage,
    estimate_cost,
)
from sec_generative_search.providers import openai_compat
from sec_generative_search.providers.factory import build_llm_provider
from sec_generative_search.providers.local_llm import LocalLLMProvider
from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleLLMProvider,
)
from sec_generative_search.providers.registry import (
    ProviderRegistry,
    ProviderSurface,
)

# A realistic sentinel-shaped key; here we just need a non-empty string
# with a distinctive tail to assert ``repr`` redaction.
_LONG_KEY = "sk-local-ABCDEFGHIJKLMNOPQRSTUVWX"  # pragma: allowlist secret
_KEY_TAIL = _LONG_KEY[-4:]

_LOOPBACK_BASE_URL = "http://127.0.0.1:11434/v1"
_DEFAULT_MODEL = "llama3.2"


@pytest.fixture
def patched_openai(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the OpenAI client class with a kwargs-capturing MagicMock."""
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> MagicMock:
        client = MagicMock()
        captured["client"] = client
        captured["kwargs"] = kwargs
        return client

    monkeypatch.setattr(openai_compat, "OpenAI", factory)
    return captured


# ---------------------------------------------------------------------------
# Class declarations
# ---------------------------------------------------------------------------


class TestDeclarations:
    def test_is_openai_compatible_subclass(self) -> None:
        # The whole point of the adapter is to inherit the shared surface.
        assert issubclass(LocalLLMProvider, OpenAICompatibleLLMProvider)

    def test_provider_name(self) -> None:
        assert LocalLLMProvider.provider_name == "local_llm"

    def test_default_base_url_is_loopback(self) -> None:
        # The shipped default targets a stock Ollama install on loopback.
        assert LocalLLMProvider.default_base_url == _LOOPBACK_BASE_URL

    def test_default_model(self) -> None:
        assert LocalLLMProvider.default_model == _DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Registry registration contract (OpenRouter-style: arbitrary models)
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registered_on_llm_surface(self) -> None:
        entry = ProviderRegistry.get_entry("local_llm", ProviderSurface.LLM)
        assert entry.provider_cls is LocalLLMProvider

    def test_listed_among_llm_providers(self) -> None:
        assert "local_llm" in ProviderRegistry.list_providers(ProviderSurface.LLM)

    def test_supports_arbitrary_models(self) -> None:
        # The served model set is operator-defined, so the registry treats
        # it like OpenRouter — a free-text slug provider, not a dropdown.
        assert ProviderRegistry.supports_arbitrary_models("local_llm", ProviderSurface.LLM)

    def test_does_not_support_upstream_routing(self) -> None:
        # No meta-routing layer — a self-hosted endpoint serves one model
        # set, so the routing-hint UI must stay hidden for it.
        assert not ProviderRegistry.supports_upstream_routing("local_llm", ProviderSurface.LLM)

    def test_catalogue_is_empty_by_design(self) -> None:
        # Mirrors OpenRouter: an empty catalogue is load-bearing — adding
        # rows would start rejecting slugs the local server actually serves.
        assert ProviderRegistry.list_models("local_llm", ProviderSurface.LLM) == []


# ---------------------------------------------------------------------------
# Capability probe: O(1), accepts any slug as FREE, no SDK round-trip
# ---------------------------------------------------------------------------


class TestCapabilityProbe:
    def test_arbitrary_slug_returns_free_capability(self, patched_openai: dict[str, Any]) -> None:
        provider = LocalLLMProvider(_LONG_KEY)
        client = patched_openai["client"]
        client.reset_mock()
        caps = provider.get_capabilities("some-model-the-server-pulled")
        assert caps.chat is True
        assert caps.streaming is True
        # A self-hosted endpoint costs nothing per token — both per-MTok
        # costs are 0.0 and the tier is *derived* to FREE (never assigned).
        assert caps.input_cost_per_mtok == 0.0
        assert caps.output_cost_per_mtok == 0.0
        assert caps.pricing_tier is PricingTier.FREE
        # The FREE capability carries no context window / output budget —
        # the true values depend on the locally served model.
        assert not caps.context_window_tokens
        assert not caps.max_output_tokens
        # O(1): inspecting declared capabilities makes no SDK call.
        client.models.list.assert_not_called()
        client.chat.completions.create.assert_not_called()

    def test_registry_capability_probe_is_offline_and_free(self) -> None:
        # The registry probe reads the (empty) catalogue and falls through
        # to the FREE capability without instantiating the provider.
        cap = ProviderRegistry.get_capability("local_llm", ProviderSurface.LLM, "any-slug")
        assert cap.chat is True
        assert cap.streaming is True
        assert cap.pricing_tier is PricingTier.FREE
        assert cap.input_cost_per_mtok == 0.0
        assert cap.output_cost_per_mtok == 0.0

    def test_registry_and_adapter_probes_agree(self, patched_openai: dict[str, Any]) -> None:
        # The credential-free registry probe and the adapter's instance
        # probe must never disagree — both are the FREE capability.
        provider = LocalLLMProvider(_LONG_KEY)
        adapter_cap = provider.get_capabilities("mixtral")
        registry_cap = ProviderRegistry.get_capability("local_llm", ProviderSurface.LLM, "mixtral")
        assert adapter_cap == registry_cap


# ---------------------------------------------------------------------------
# FREE pricing is cost-derived and yields a $0.00 estimate
# ---------------------------------------------------------------------------


class TestFreeTierPricing:
    def test_free_tier_is_cost_derived_not_assigned(self, patched_openai: dict[str, Any]) -> None:
        # The tier must fall out of the 0.0 cost via ProviderCapability's
        # __post_init__, matching a freshly-derived capability — never a
        # hand-stamped FREE tier alongside a different cost.
        cap = LocalLLMProvider(_LONG_KEY).get_capabilities()
        assert cap == ProviderCapability(
            chat=True,
            streaming=True,
            input_cost_per_mtok=0.0,
            output_cost_per_mtok=0.0,
        )

    def test_estimate_cost_is_zero_for_any_usage(self, patched_openai: dict[str, Any]) -> None:
        cap = LocalLLMProvider(_LONG_KEY).get_capabilities("llama3.2")
        usage = TokenUsage(input_tokens=12_345, output_tokens=6_789)
        # 0.0 cost (not None) → a real $0.00 estimate, not the honest-UNKNOWN
        # ``None`` an arbitrary-slug OpenRouter row would report.
        assert estimate_cost(usage, cap) == 0.0

    def test_registry_probe_estimate_cost_is_zero(self) -> None:
        cap = ProviderRegistry.get_capability("local_llm", ProviderSurface.LLM, "phi3")
        usage = TokenUsage(input_tokens=1_000, output_tokens=2_000)
        assert estimate_cost(usage, cap) == 0.0


# ---------------------------------------------------------------------------
# build_llm_provider: sentinel-key tolerance for this provider only
# ---------------------------------------------------------------------------


class TestBuildWithoutCredential:
    def test_builds_without_a_resolved_key(self, patched_openai: dict[str, Any]) -> None:
        # A None resolver result is tolerated only for local_llm; the
        # factory passes a non-secret sentinel so construction succeeds.
        provider = build_llm_provider("local_llm", api_key_resolver=lambda _n: None)
        assert isinstance(provider, LocalLLMProvider)

    def test_default_env_resolver_needs_no_local_llm_var(
        self, patched_openai: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # There is no LOCAL_LLM_API_KEY env var; the default resolver
        # returns None for local_llm and the build still succeeds.
        provider = build_llm_provider("local_llm")
        assert isinstance(provider, LocalLLMProvider)

    def test_resolved_key_is_honoured_over_sentinel(self, patched_openai: dict[str, Any]) -> None:
        # A vLLM/LM-Studio server behind a bearer token: when the resolver
        # DOES return a key, it must be used verbatim, not the sentinel.
        provider = build_llm_provider(
            "local_llm",
            api_key_resolver=lambda _n: "sk-vllm-REALKEY-1234567890",
        )
        rendered = repr(provider)
        # repr is redacted, but the distinctive tail proves the real key
        # (not the 5-char sentinel) reached the instance.
        assert "1234567890"[-4:] in rendered


# ---------------------------------------------------------------------------
# Construction: loopback base URL reaches the SDK, retry stays disabled
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_loopback_base_url_wired_into_client(self, patched_openai: dict[str, Any]) -> None:
        LocalLLMProvider(_LONG_KEY)
        kwargs = patched_openai["kwargs"]
        assert kwargs["base_url"] == _LOOPBACK_BASE_URL
        # Resilience is owned by ``resilient_call`` — the SDK's own retry
        # loop must stay disabled, like every OpenAI-compatible subclass.
        assert kwargs["max_retries"] == 0

    def test_explicit_base_url_override_reaches_client(
        self, patched_openai: dict[str, Any]
    ) -> None:
        # Other backends (llama.cpp-server, vLLM, LM Studio) work by
        # pointing the base URL at their own OpenAI-compatible port.
        LocalLLMProvider(_LONG_KEY, base_url="http://127.0.0.1:8080/v1")
        assert patched_openai["kwargs"]["base_url"] == "http://127.0.0.1:8080/v1"


# ---------------------------------------------------------------------------
# Security — the SDK key is never rendered in the provider's repr
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_repr_never_exposes_key(patched_openai: dict[str, Any]) -> None:
    del patched_openai  # fixture only stubs the OpenAI client
    text = repr(LocalLLMProvider(_LONG_KEY))
    assert _LONG_KEY not in text
    assert _KEY_TAIL in text


# ---------------------------------------------------------------------------
# Security — the no-credential tolerance is scoped to local_llm alone
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestNoKeyToleranceIsScoped:
    """``requires_api_key=False`` is a per-provider opt-in — never a default.

    Relaxing the credential requirement for every provider would let a
    hosted vendor build silently against a sentinel and burn quota (or leak
    the sentinel as if it were a real bearer token).  The relaxation is
    locked to ``local_llm`` only.
    """

    def test_only_local_llm_relaxes_the_key_requirement(self) -> None:
        for name in ProviderRegistry.list_providers(ProviderSurface.LLM):
            entry = ProviderRegistry.get_entry(name, ProviderSurface.LLM)
            expected = name == "local_llm"
            assert entry.requires_api_key is (not expected), (
                f"{name}: requires_api_key must be True for every hosted "
                "provider and False only for the self-hosted local_llm"
            )

    def test_hosted_provider_still_fails_closed_without_a_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The default (requires_api_key=True) must stay fail-closed: a
        # None resolver result for a hosted provider is still a hard error.
        from sec_generative_search.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            build_llm_provider("openai", api_key_resolver=lambda _n: None)

    def test_sentinel_is_not_a_real_credential(self, patched_openai: dict[str, Any]) -> None:
        # The sentinel must be short enough that mask_secret fully redacts
        # it (no recognisable tail) and must not look key-shaped.
        from sec_generative_search.core.security import mask_secret
        from sec_generative_search.providers import factory

        sentinel = factory._LLM_NO_KEY_SENTINEL
        assert len(sentinel) < 8
        # A short value is masked in full — no recognisable tail leaks.
        assert mask_secret(sentinel) == "***"
        for needle in ("sk-", "bearer", "secret", "password"):
            assert needle not in sentinel.lower()

    def test_free_tier_relaxation_is_scoped_to_local_llm(self) -> None:
        # Only local_llm prices an uncatalogued slug as FREE; every other
        # LLM entry keeps its baseline pricing (and OpenRouter stays UNKNOWN
        # for arbitrary slugs — never silently free).
        for name in ProviderRegistry.list_providers(ProviderSurface.LLM):
            entry = ProviderRegistry.get_entry(name, ProviderSurface.LLM)
            assert entry.free_tier is (name == "local_llm")
