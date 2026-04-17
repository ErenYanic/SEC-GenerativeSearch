"""Parametrised vendor tests for the Phase 5D OpenAI-compatible subclasses.

Every vendor in Phase 5D ships by configuring the shared
:mod:`providers.openai_compat` surface with nothing more than a
``provider_name``, a ``default_base_url``, a ``default_model``, and a
static catalogue.  The heavy behaviour — retry/circuit-breaker, content-
filter handling, streaming, token accounting — is already exhaustively
covered in ``test_openai_compat``; this suite therefore focuses on the
Phase 5D-specific contract:

- Each provider declares a non-empty, reachable-looking ``base_url``.
- Each provider's ``default_model`` is present in its catalogue (or the
  catalogue is deliberately empty for the :class:`OpenRouterProvider`
  meta-provider).
- The O(1) capability probe returns the declared catalogue entry
  without any SDK round-trip.
- Unknown slugs on the meta-provider fall back to the permissive
  ``ProviderCapability(chat=True, streaming=True)`` default, preserving
  the "accepts any model slug" semantics of OpenRouter.
- Embedding variants expose their declared native dimension without a
  network call.
- Construction plumbing wires the vendor-specific ``base_url`` into the
  underlying ``OpenAI`` client exactly once.
- Security: the API key is never rendered in the provider's ``repr``.

Every test mocks the ``OpenAI`` SDK factory — no real network calls are
issued and no live API key is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from sec_generative_search.providers import openai_compat
from sec_generative_search.providers.deepseek import DeepSeekProvider
from sec_generative_search.providers.kimi import KimiProvider
from sec_generative_search.providers.mistral import (
    MistralEmbeddingProvider,
    MistralProvider,
)
from sec_generative_search.providers.openrouter import OpenRouterProvider
from sec_generative_search.providers.qwen import QwenEmbeddingProvider, QwenProvider

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


_LONG_KEY = "sk-test-VENDOR-ABCDEFGHIJKLMNOPQRSTUVWX"  # pragma: allowlist secret
_KEY_TAIL = _LONG_KEY[-4:]


@pytest.fixture
def patched_openai(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the OpenAI client class with a MagicMock factory.

    Captures the kwargs so each test can assert on the ``base_url``
    passed by the vendor subclass.
    """
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> MagicMock:
        client = MagicMock()
        captured["client"] = client
        captured["kwargs"] = kwargs
        return client

    monkeypatch.setattr(openai_compat, "OpenAI", factory)
    return captured


# ---------------------------------------------------------------------------
# LLM vendor table — (ProviderClass, expected provider_name, expected
# default_base_url, default_model, an extra catalogue slug to spot-check)
# ---------------------------------------------------------------------------


_LLM_VENDORS: list[tuple[type, str, str, str, str | None]] = [
    (
        MistralProvider,
        "mistral",
        "https://api.mistral.ai/v1",
        "mistral-small-latest",
        "mistral-large-latest",
    ),
    (
        KimiProvider,
        "kimi",
        "https://api.moonshot.ai/v1",
        "moonshot-v1-32k",
        "kimi-k2",
    ),
    (
        DeepSeekProvider,
        "deepseek",
        "https://api.deepseek.com/v1",
        "deepseek-chat",
        "deepseek-reasoner",
    ),
    (
        QwenProvider,
        "qwen",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "qwen-turbo",
        "qwen-max",
    ),
    (
        OpenRouterProvider,
        "openrouter",
        "https://openrouter.ai/api/v1",
        "openai/gpt-4o-mini",
        None,  # catalogue is deliberately empty
    ),
]


_EMBED_VENDORS: list[tuple[type, str, str, str, int]] = [
    (
        MistralEmbeddingProvider,
        "mistral",
        "https://api.mistral.ai/v1",
        "mistral-embed",
        1024,
    ),
    (
        QwenEmbeddingProvider,
        "qwen",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "text-embedding-v3",
        1024,
    ),
]


# ---------------------------------------------------------------------------
# Class-level contract: provider_name, base_url, default_model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider_cls", "provider_name", "base_url", "default_model", "extra_slug"),
    _LLM_VENDORS,
)
class TestLLMVendorDeclarations:
    def test_provider_name_matches(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        extra_slug: str | None,
    ) -> None:
        assert provider_cls.provider_name == provider_name

    def test_default_base_url_matches(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        extra_slug: str | None,
    ) -> None:
        assert provider_cls.default_base_url == base_url

    def test_default_model_is_declared(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        extra_slug: str | None,
    ) -> None:
        assert provider_cls.default_model == default_model

    def test_default_model_is_in_catalogue_or_meta(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        extra_slug: str | None,
    ) -> None:
        # Every non-meta provider pins its default_model to an entry in
        # the catalogue so the capability probe is truly O(1).  The
        # OpenRouter meta-provider deliberately ships an empty catalogue
        # — unknown slugs are the whole point.
        catalogue = provider_cls.MODEL_CATALOGUE
        if catalogue:
            assert default_model in catalogue, (
                f"{provider_cls.__name__} default_model '{default_model}' "
                "missing from MODEL_CATALOGUE"
            )
        else:
            assert provider_cls is OpenRouterProvider

    def test_extra_catalogue_slug_present(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        extra_slug: str | None,
    ) -> None:
        if extra_slug is None:
            pytest.skip("meta-provider — catalogue intentionally empty")
        assert extra_slug in provider_cls.MODEL_CATALOGUE


# ---------------------------------------------------------------------------
# Construction contract: base_url reaches the SDK client unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider_cls", "provider_name", "base_url", "default_model", "extra_slug"),
    _LLM_VENDORS,
)
def test_base_url_wired_into_client(
    provider_cls: type,
    provider_name: str,
    base_url: str,
    default_model: str,
    extra_slug: str | None,
    patched_openai: dict[str, Any],
) -> None:
    """Every LLM vendor subclass hands its ``default_base_url`` to the SDK."""
    provider_cls(_LONG_KEY)
    kwargs = patched_openai["kwargs"]
    assert kwargs["base_url"] == base_url
    # Resilience is owned by ``resilient_call`` — the SDK's own retry
    # loop must stay disabled on every OpenAI-compatible subclass.
    assert kwargs["max_retries"] == 0


# ---------------------------------------------------------------------------
# Capability probe contract: O(1), no SDK round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider_cls", "provider_name", "base_url", "default_model", "extra_slug"),
    _LLM_VENDORS,
)
def test_capability_probe_is_offline(
    provider_cls: type,
    provider_name: str,
    base_url: str,
    default_model: str,
    extra_slug: str | None,
    patched_openai: dict[str, Any],
) -> None:
    provider = provider_cls(_LONG_KEY)
    client = patched_openai["client"]
    client.reset_mock()
    caps = provider.get_capabilities(default_model)
    assert caps.chat is True
    assert caps.streaming is True
    # O(1) probe: no SDK call required to inspect declared capabilities.
    client.models.list.assert_not_called()
    client.chat.completions.create.assert_not_called()


def test_openrouter_accepts_any_slug(patched_openai: dict[str, Any]) -> None:
    """OpenRouter's meta-provider contract: unknown slug → permissive default.

    This is the Phase 5D.5 "accepts any model slug; lazy validation
    against OpenRouter's model list" contract.  Validation against the
    live model list is delegated to ``validate_key``/the Phase 5F
    registry; the capability probe deliberately stays permissive so the
    SDK call can proceed and the upstream API produces the authoritative
    error if the slug is unserviceable.
    """
    provider = OpenRouterProvider(_LONG_KEY)
    caps = provider.get_capabilities("some-vendor/model-that-may-not-exist")
    assert caps.chat is True
    assert caps.streaming is True
    # Permissive default carries no context window or output budget — the
    # true values depend on whichever upstream OpenRouter routes the slug
    # to, and the SDK call will surface any mismatch.
    assert not caps.context_window_tokens
    assert not caps.max_output_tokens


# ---------------------------------------------------------------------------
# Embedding providers: declared native dimensions, no network on empty input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider_cls", "provider_name", "base_url", "default_model", "dimension"),
    _EMBED_VENDORS,
)
class TestEmbeddingVendorDeclarations:
    def test_provider_name_matches(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        dimension: int,
    ) -> None:
        assert provider_cls.provider_name == provider_name

    def test_base_url_matches(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        dimension: int,
    ) -> None:
        assert provider_cls.default_base_url == base_url

    def test_default_model_has_dimension(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        dimension: int,
    ) -> None:
        # Dimension is declared statically; the constructor rejects any
        # slug missing from MODEL_DIMENSIONS, so we can rely on the
        # lookup without a network call.
        assert provider_cls.MODEL_DIMENSIONS[default_model] == dimension

    def test_get_dimension_matches(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        dimension: int,
        patched_openai: dict[str, Any],
    ) -> None:
        provider = provider_cls(_LONG_KEY)
        assert provider.get_dimension() == dimension


# ---------------------------------------------------------------------------
# Security — vendor subclasses inherit the base redaction, and we re-assert
# per-vendor so the regression fails loudly on any accidental override.
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.parametrize(
    ("provider_cls", "provider_name", "base_url", "default_model", "extra_slug"),
    _LLM_VENDORS,
)
def test_llm_repr_never_exposes_key(
    provider_cls: type,
    provider_name: str,
    base_url: str,
    default_model: str,
    extra_slug: str | None,
    patched_openai: dict[str, Any],
) -> None:
    text = repr(provider_cls(_LONG_KEY))
    assert _LONG_KEY not in text
    assert _KEY_TAIL in text


@pytest.mark.security
@pytest.mark.parametrize(
    ("provider_cls", "provider_name", "base_url", "default_model", "dimension"),
    _EMBED_VENDORS,
)
def test_embed_repr_never_exposes_key(
    provider_cls: type,
    provider_name: str,
    base_url: str,
    default_model: str,
    dimension: int,
    patched_openai: dict[str, Any],
) -> None:
    text = repr(provider_cls(_LONG_KEY))
    assert _LONG_KEY not in text
    assert _KEY_TAIL in text
