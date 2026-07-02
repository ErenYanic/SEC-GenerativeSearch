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

from sec_generative_search.core.exceptions import (
    ProviderConnectionError,
    ProviderTimeoutError,
)
from sec_generative_search.core.resilience import RetryPolicy
from sec_generative_search.core.types import (
    PricingTier,
    ProviderCapability,
    TokenUsage,
    estimate_cost,
)
from sec_generative_search.providers import openai_compat
from sec_generative_search.providers.base import GenerationRequest
from sec_generative_search.providers.factory import build_llm_provider
from sec_generative_search.providers.local_llm import LocalLLMProvider
from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleLLMProvider,
)
from sec_generative_search.providers.openrouter import OpenRouterRoutingHints
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


def _fast_retry() -> RetryPolicy:
    """Tight retry schedule so a retried failure stays in milliseconds."""
    return RetryPolicy(max_retries=2, backoff_base=2.0, initial_delay=0.0, max_delay=0.0)


class _FakeAPIConnection(openai_compat.APIConnectionError):
    """SDK-shaped connection error that skips the heavy httpx constructor."""

    def __init__(self, message: str = "connection refused") -> None:
        Exception.__init__(self, message)


class _FakeAPITimeout(openai_compat.APITimeoutError):
    """SDK-shaped timeout — subclasses the connection error in the real SDK."""

    def __init__(self, message: str = "timed out") -> None:
        Exception.__init__(self, message)


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


# ---------------------------------------------------------------------------
# Endpoint unreachable — the dominant local_llm failure
# ---------------------------------------------------------------------------


class TestConnectionFailure:
    """An unreachable endpoint maps onto the ProviderError contract.

    A stock Ollama install that is not running (or a mis-pointed
    ``LOCAL_LLM_BASE_URL``) surfaces as ``openai.APIConnectionError`` from
    the SDK.  The shared mapping normalises that to
    :class:`ProviderConnectionError` — a transient, retryable
    :class:`ProviderError` the orchestrator / API / CLI already reason
    about — and never lets the raw SDK/transport exception leak.
    """

    def test_unreachable_endpoint_maps_to_connection_error(
        self, patched_openai: dict[str, Any]
    ) -> None:
        provider = LocalLLMProvider(_LONG_KEY, retry_policy=_fast_retry())
        client = patched_openai["client"]
        client.chat.completions.create.side_effect = _FakeAPIConnection()
        with pytest.raises(ProviderConnectionError) as excinfo:
            provider.generate(GenerationRequest(prompt="hi", model="llama3.2"))
        # Transient → retried within the budget (initial + 2 retries).
        assert client.chat.completions.create.call_count == 3
        # The endpoint base URL must never leak into the normalised error.
        assert _LOOPBACK_BASE_URL not in str(excinfo.value)

    def test_timeout_is_not_reclassified_as_connection(
        self, patched_openai: dict[str, Any]
    ) -> None:
        # ``APITimeoutError`` subclasses ``APIConnectionError`` — a slow
        # local server must stay a ProviderTimeoutError, never a
        # ProviderConnectionError (ordering guard).
        provider = LocalLLMProvider(_LONG_KEY, retry_policy=_fast_retry())
        patched_openai["client"].chat.completions.create.side_effect = _FakeAPITimeout()
        with pytest.raises(ProviderTimeoutError) as excinfo:
            provider.generate(GenerationRequest(prompt="hi", model="llama3.2"))
        assert not isinstance(excinfo.value, ProviderConnectionError)

    @pytest.mark.security
    def test_connection_error_never_leaks_the_key(self, patched_openai: dict[str, Any]) -> None:
        # Whatever the SDK transport error carries, the normalised
        # connection error must not echo the API key.
        provider = LocalLLMProvider(_LONG_KEY, retry_policy=_fast_retry())
        patched_openai["client"].chat.completions.create.side_effect = _FakeAPIConnection(
            "refused talking to endpoint"
        )
        with pytest.raises(ProviderConnectionError) as excinfo:
            provider.generate(GenerationRequest(prompt="hi", model="llama3.2"))
        assert _LONG_KEY not in str(excinfo.value)
        assert _LONG_KEY not in str(excinfo.value.details or "")
        assert _LONG_KEY not in str(excinfo.value.hint or "")


# ---------------------------------------------------------------------------
# SDK response doubles — shaped like the OpenAI completion / stream chunk
# objects, bypassing the real SDK.  The exhaustive generation contract is
# covered in ``test_openai_compat``; these prove the local adapter drives
# that shared path end-to-end against a *faked local endpoint*.
# ---------------------------------------------------------------------------


def _make_completion(
    *,
    text: str = "The 10-K reports revenue of $1.2B.",
    finish_reason: str = "stop",
    prompt_tokens: int = 42,
    completion_tokens: int = 17,
    model: str = _DEFAULT_MODEL,
) -> MagicMock:
    """Build a non-streaming completion shape mirroring the SDK response."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    completion.model = model
    return completion


def _make_text_chunk(content: str, *, model: str = _DEFAULT_MODEL) -> MagicMock:
    """Build a streaming delta chunk carrying partial text."""
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.model = model
    chunk.usage = None
    return chunk


def _make_usage_chunk(
    prompt_tokens: int, completion_tokens: int, *, model: str = _DEFAULT_MODEL
) -> MagicMock:
    """Build the terminal usage-only chunk (empty ``choices``)."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    chunk = MagicMock()
    chunk.choices = []
    chunk.model = model
    chunk.usage = usage
    return chunk


# ---------------------------------------------------------------------------
# Generation / streaming against a faked local endpoint
# ---------------------------------------------------------------------------


class TestGeneration:
    """The adapter drives the shared OpenAI-wire path against a fake server.

    A stock Ollama / llama.cpp-server / vLLM / LM Studio install answers on
    an OpenAI-compatible Chat Completions surface; here the SDK client is a
    MagicMock, so no live endpoint is required.  These pin that the local
    adapter forwards the requested (arbitrary) slug and surfaces the
    endpoint's text + token accounting unchanged.
    """

    def test_generate_returns_text_and_usage(self, patched_openai: dict[str, Any]) -> None:
        provider = LocalLLMProvider(_LONG_KEY)
        client = patched_openai["client"]
        # The local server serves whatever slug it has pulled — an arbitrary
        # one here proves the empty catalogue never gates generation.
        client.chat.completions.create.return_value = _make_completion(model="mistral-nemo")
        response = provider.generate(
            GenerationRequest(prompt="What was revenue?", model="mistral-nemo")
        )
        assert response.text == "The 10-K reports revenue of $1.2B."
        assert response.model == "mistral-nemo"
        assert response.token_usage.input_tokens == 42
        assert response.token_usage.output_tokens == 17
        # The arbitrary slug was forwarded verbatim, non-streaming.
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "mistral-nemo"
        assert call_kwargs["stream"] is False

    def test_generate_stream_yields_deltas_then_usage(self, patched_openai: dict[str, Any]) -> None:
        provider = LocalLLMProvider(_LONG_KEY)
        client = patched_openai["client"]
        client.chat.completions.create.return_value = [
            _make_text_chunk("Revenue "),
            _make_text_chunk("grew 12%."),
            _make_usage_chunk(30, 8),
        ]
        chunks = list(
            provider.generate_stream(GenerationRequest(prompt="Summarise", model="llama3.2"))
        )
        # Two text deltas plus the terminal usage-only frame.
        assert [c.text for c in chunks[:2]] == ["Revenue ", "grew 12%."]
        assert chunks[-1].token_usage.input_tokens == 30
        assert chunks[-1].token_usage.output_tokens == 8
        # Streaming asks for usage in the final frame, like every vendor.
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# Endpoint settings wiring — the provider re-reads LocalLLMSettings,
# fails closed to the loopback default
# ---------------------------------------------------------------------------


class TestEndpointSettings:
    """The base URL + default model come from ``LOCAL_LLM_*`` settings.

    The endpoint is operator-owned deployment config, so the adapter
    re-reads :class:`LocalLLMSettings` standalone at construction rather
    than through the credential resolver chain — and an explicit
    constructor argument still wins.
    """

    def test_base_url_from_settings_reaches_client(
        self, clean_env: pytest.MonkeyPatch, patched_openai: dict[str, Any]
    ) -> None:
        # Other backends (llama.cpp-server, vLLM, LM Studio) work by pointing
        # the base URL at their own loopback OpenAI-compatible port.
        clean_env.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
        LocalLLMProvider(_LONG_KEY)
        assert patched_openai["kwargs"]["base_url"] == "http://127.0.0.1:8080/v1"

    def test_default_model_from_settings_is_used(
        self, clean_env: pytest.MonkeyPatch, patched_openai: dict[str, Any]
    ) -> None:
        clean_env.setenv("LOCAL_LLM_DEFAULT_MODEL", "qwen2.5")
        provider = LocalLLMProvider(_LONG_KEY)
        # The instance carries the configured default; the class attribute
        # stays the shipped default so the registry's class-level probe is
        # unaffected.
        assert provider.default_model == "qwen2.5"
        assert LocalLLMProvider.default_model == _DEFAULT_MODEL
        # A request with no explicit slug falls through to that default.
        client = patched_openai["client"]
        client.chat.completions.create.return_value = _make_completion(model="qwen2.5")
        provider.generate(GenerationRequest(prompt="hi", model=None))
        assert client.chat.completions.create.call_args.kwargs["model"] == "qwen2.5"

    def test_explicit_base_url_wins_over_settings(
        self, clean_env: pytest.MonkeyPatch, patched_openai: dict[str, Any]
    ) -> None:
        clean_env.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
        LocalLLMProvider(_LONG_KEY, base_url="http://127.0.0.1:9999/v1")
        assert patched_openai["kwargs"]["base_url"] == "http://127.0.0.1:9999/v1"

    def test_non_local_base_url_with_opt_in_is_honoured(
        self, clean_env: pytest.MonkeyPatch, patched_openai: dict[str, Any]
    ) -> None:
        # The operator explicitly acknowledged the prompt leaves the host.
        clean_env.setenv("LOCAL_LLM_BASE_URL", "http://192.168.1.10:11434/v1")
        clean_env.setenv("LOCAL_LLM_ALLOW_NON_LOCAL", "true")
        LocalLLMProvider(_LONG_KEY)
        assert patched_openai["kwargs"]["base_url"] == "http://192.168.1.10:11434/v1"


@pytest.mark.security
class TestEndpointFailsClosed:
    """The provider never sends the prompt to an off-policy endpoint.

    ``LocalLLMSettings`` rejects a non-loopback URL (without the opt-in) at
    settings load — but the provider's standalone re-read must *also* refuse
    to honour such a value, falling back to the loopback default rather than
    propagating or, worse, quietly dialling the off-box host.  The base URL
    decides where the assembled prompt (Tier-3 data) is sent, so this is a
    security control, not a convenience.
    """

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://203.0.113.7:11434/v1",  # public IP
            "http://192.168.1.10:11434/v1",  # private LAN IP
            "http://ollama.internal:11434/v1",  # bare hostname
            "https://api.example.com/v1",  # public hostname
        ],
    )
    def test_non_local_without_opt_in_falls_back_to_loopback(
        self,
        clean_env: pytest.MonkeyPatch,
        patched_openai: dict[str, Any],
        base_url: str,
    ) -> None:
        clean_env.setenv("LOCAL_LLM_BASE_URL", base_url)
        clean_env.delenv("LOCAL_LLM_ALLOW_NON_LOCAL", raising=False)
        LocalLLMProvider(_LONG_KEY)
        # The off-policy host never reaches the SDK client.
        assert patched_openai["kwargs"]["base_url"] == _LOOPBACK_BASE_URL

    @pytest.mark.parametrize(
        "base_url",
        [
            "ftp://127.0.0.1/v1",  # non-http(s) scheme
            "http:///v1",  # no host
            "not-a-url",  # unparseable
        ],
    )
    def test_malformed_base_url_falls_back_to_loopback(
        self,
        clean_env: pytest.MonkeyPatch,
        patched_openai: dict[str, Any],
        base_url: str,
    ) -> None:
        clean_env.setenv("LOCAL_LLM_BASE_URL", base_url)
        LocalLLMProvider(_LONG_KEY)
        assert patched_openai["kwargs"]["base_url"] == _LOOPBACK_BASE_URL


# ---------------------------------------------------------------------------
# Security — routing hints are silently dropped (non-routing provider)
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRoutingHintsDropped:
    """``local_llm`` does not advertise upstream routing, so it drops hints.

    The registry reports ``supports_upstream_routing=False`` for
    ``local_llm``, and the OpenAI-compatible base's empty default
    :meth:`_extra_request_kwargs` hook means a
    :class:`OpenRouterRoutingHints` supplied on the request must never reach
    the SDK call (mirrors the non-OpenRouter-vendor lock in
    ``test_openrouter``).  A self-hosted endpoint has no meta-router to
    honour them, so forwarding would be a no-op leak at best.
    """

    def test_generate_drops_routing_hints(self, patched_openai: dict[str, Any]) -> None:
        provider = LocalLLMProvider(_LONG_KEY)
        client = patched_openai["client"]
        client.chat.completions.create.return_value = _make_completion()
        request = GenerationRequest(
            prompt="hello",
            model="llama3.2",
            routing_hints=OpenRouterRoutingHints(order=("anthropic",)),
        )
        provider.generate(request)
        assert "extra_body" not in client.chat.completions.create.call_args.kwargs

    def test_generate_stream_drops_routing_hints(self, patched_openai: dict[str, Any]) -> None:
        provider = LocalLLMProvider(_LONG_KEY)
        client = patched_openai["client"]
        client.chat.completions.create.return_value = [_make_usage_chunk(5, 3)]
        request = GenerationRequest(
            prompt="hello",
            model="llama3.2",
            routing_hints=OpenRouterRoutingHints(order=("anthropic",)),
        )
        list(provider.generate_stream(request))
        assert "extra_body" not in client.chat.completions.create.call_args.kwargs
