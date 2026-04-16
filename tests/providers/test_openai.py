"""Tests for :mod:`sec_generative_search.providers.openai` (Phase 5B.3/5B.4).

The OpenAI-compatible plumbing is exhaustively covered in
``test_openai_compat``; this suite focuses on the OpenAI-specific
surface:

- The model catalogue is well-formed and surfaces sensible capability
  matrices for each declared model.
- The default ``base_url`` is left unset so the SDK targets
  ``api.openai.com``.
- ``provider_name`` is consistent across LLM and embedding providers.
- ``OpenAIEmbeddingProvider`` exposes the documented native dimensions
  for ``text-embedding-3-{small,large}``.
- Capability probing populates ``ProviderCapability`` *without* a
  network call (Phase 5B.4 cheap-probe contract).
- The OpenAI key never leaks through repr/str/log emissions.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.core.types import PricingTier
from sec_generative_search.providers import openai_compat
from sec_generative_search.providers.openai import OpenAIEmbeddingProvider, OpenAIProvider

_LONG_KEY = "sk-test-OPENAI-ABCDEFGHIJKLMNOPQRSTUVWX"
_KEY_TAIL = _LONG_KEY[-4:]


@pytest.fixture
def _patch_openai_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the OpenAI client class with a MagicMock factory."""
    captured: dict[str, MagicMock] = {}

    def factory(**kwargs: object) -> MagicMock:
        client = MagicMock()
        captured["client"] = client
        captured["kwargs"] = kwargs  # type: ignore[assignment]
        return client

    monkeypatch.setattr(openai_compat, "OpenAI", factory)
    return captured  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Model catalogue and base URL contract
# ---------------------------------------------------------------------------


class TestProviderMetadata:
    def test_provider_name_matches_across_surfaces(self) -> None:
        assert OpenAIProvider.provider_name == "openai"
        assert OpenAIEmbeddingProvider.provider_name == "openai"

    def test_default_base_url_is_unset(self, _patch_openai_client: dict) -> None:
        OpenAIProvider(_LONG_KEY)
        assert _patch_openai_client["kwargs"]["base_url"] is None

    def test_default_model_is_gpt_4o_mini(self) -> None:
        assert OpenAIProvider.default_model == "gpt-4o-mini"

    def test_catalogue_includes_required_models(self) -> None:
        for slug in ("gpt-4o", "gpt-4o-mini", "o3", "o4-mini"):
            assert slug in OpenAIProvider.MODEL_CATALOGUE, (
                f"OpenAIProvider missing required model '{slug}' in MODEL_CATALOGUE"
            )

    def test_pricing_tiers_set(self) -> None:
        # gpt-4o-mini should be the cheapest production option
        assert OpenAIProvider.MODEL_CATALOGUE["gpt-4o-mini"].capability.pricing_tier == (
            PricingTier.LOW
        )
        assert OpenAIProvider.MODEL_CATALOGUE["o3"].capability.pricing_tier == PricingTier.PREMIUM


# ---------------------------------------------------------------------------
# Capability probe (Phase 5B.4)
# ---------------------------------------------------------------------------


class TestCapabilityProbe:
    def test_probe_does_not_call_network(self, _patch_openai_client: dict) -> None:
        provider = OpenAIProvider(_LONG_KEY)
        client = _patch_openai_client["client"]
        client.reset_mock()
        caps = provider.get_capabilities("gpt-4o")
        assert caps.streaming is True
        assert caps.tool_use is True
        assert caps.context_window_tokens == 128_000
        # Cheap probe contract: no SDK call required to read capabilities.
        client.models.list.assert_not_called()
        client.chat.completions.create.assert_not_called()

    def test_unknown_model_is_permissive(self, _patch_openai_client: dict) -> None:
        provider = OpenAIProvider(_LONG_KEY)
        caps = provider.get_capabilities("gpt-99")
        # Permissive default — let the SDK reject the slug at call time
        # rather than failing capability probe.
        assert caps.chat is True
        assert caps.streaming is True


# ---------------------------------------------------------------------------
# Embedding provider — declared native dimensions
# ---------------------------------------------------------------------------


class TestEmbeddingDimensions:
    def test_small_dimension(self, _patch_openai_client: dict) -> None:
        provider = OpenAIEmbeddingProvider(_LONG_KEY, model="text-embedding-3-small")
        assert provider.get_dimension() == 1536

    def test_large_dimension(self, _patch_openai_client: dict) -> None:
        provider = OpenAIEmbeddingProvider(_LONG_KEY, model="text-embedding-3-large")
        assert provider.get_dimension() == 3072

    def test_default_model_is_small(self, _patch_openai_client: dict) -> None:
        provider = OpenAIEmbeddingProvider(_LONG_KEY)
        assert provider.get_dimension() == 1536

    def test_unknown_model_rejected_at_construction(self, _patch_openai_client: dict) -> None:
        with pytest.raises(ValueError):
            OpenAIEmbeddingProvider(_LONG_KEY, model="text-embedding-2")


# ---------------------------------------------------------------------------
# Token counting — uses tiktoken with model-specific encoding
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_count_tokens_matches_tiktoken_for_known_model(
        self,
        _patch_openai_client: dict,
    ) -> None:
        import tiktoken

        provider = OpenAIProvider(_LONG_KEY)
        encoder = tiktoken.encoding_for_model("gpt-4o-mini")
        text = "Apple Inc. reported strong revenues in Q4 2023."
        assert provider.count_tokens(text, model="gpt-4o-mini") == len(encoder.encode(text))


# ---------------------------------------------------------------------------
# Security — keys never leak via repr or any log path
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecretSafety:
    def test_repr_never_exposes_key(self, _patch_openai_client: dict) -> None:
        text = repr(OpenAIProvider(_LONG_KEY))
        assert _LONG_KEY not in text
        assert _KEY_TAIL in text

    def test_embed_repr_never_exposes_key(self, _patch_openai_client: dict) -> None:
        text = repr(OpenAIEmbeddingProvider(_LONG_KEY))
        assert _LONG_KEY not in text
        assert _KEY_TAIL in text

    def test_logger_emissions_redact_key(
        self,
        caplog: pytest.LogCaptureFixture,
        _patch_openai_client: dict,
    ) -> None:
        package_logger = logging.getLogger(LOGGER_NAME)
        previous = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
                provider = OpenAIProvider(_LONG_KEY)
                package_logger.info("Constructed %s", provider)
        finally:
            package_logger.propagate = previous

        for record in caplog.records:
            assert _LONG_KEY not in record.getMessage()
            assert _LONG_KEY not in str(record.args)
