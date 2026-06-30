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
- the O(1) capability probe accepts any slug with the permissive default
  and never makes an SDK round-trip;
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

from sec_generative_search.providers import openai_compat
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
# Capability probe: O(1), accepts any slug, no SDK round-trip
# ---------------------------------------------------------------------------


class TestCapabilityProbe:
    def test_arbitrary_slug_returns_permissive_default(
        self, patched_openai: dict[str, Any]
    ) -> None:
        provider = LocalLLMProvider(_LONG_KEY)
        client = patched_openai["client"]
        client.reset_mock()
        caps = provider.get_capabilities("some-model-the-server-pulled")
        assert caps.chat is True
        assert caps.streaming is True
        # Permissive default carries no context window / output budget —
        # the true values depend on the locally served model.
        assert not caps.context_window_tokens
        assert not caps.max_output_tokens
        # O(1): inspecting declared capabilities makes no SDK call.
        client.models.list.assert_not_called()
        client.chat.completions.create.assert_not_called()

    def test_registry_capability_probe_is_offline(self) -> None:
        # The registry probe reads the (empty) catalogue and falls through
        # to the permissive default without instantiating the provider.
        cap = ProviderRegistry.get_capability("local_llm", ProviderSurface.LLM, "any-slug")
        assert cap.chat is True
        assert cap.streaming is True


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
