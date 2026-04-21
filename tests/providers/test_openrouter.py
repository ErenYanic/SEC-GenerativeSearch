"""Tests for :mod:`sec_generative_search.providers.openrouter` — Phase 5H.

Covers the OpenRouter-specific routing-hint channel introduced in
Phase 5H.  Catalogue / base-url / repr-redaction behaviour already lives
in ``test_openai_compat_vendors`` — this file focuses on:

- :class:`OpenRouterRoutingHints` shape and pass-through serialisation.
- :class:`OpenRouterProvider` forwards hints into the SDK call via
  ``extra_body`` when they are attached, and makes the same call as
  without hints when they are absent.
- Every other OpenAI-compatible vendor ignores
  :attr:`GenerationRequest.routing_hints` — the OpenAI-compatible base's
  default ``_extra_request_kwargs`` hook returns an empty dict and the
  SDK kwargs stay unchanged.
- Security (``@pytest.mark.security``): the hint object carries no
  credential-shaped field names; the serialised ``provider`` block
  never adds an ``authorization`` / ``api_key`` / ``bearer`` key; the
  OpenRouter API key remains the sole credential in flight across the
  SDK call.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any
from unittest.mock import MagicMock

import pytest

from sec_generative_search.providers import openai_compat
from sec_generative_search.providers.base import GenerationRequest
from sec_generative_search.providers.deepseek import DeepSeekProvider
from sec_generative_search.providers.kimi import KimiProvider
from sec_generative_search.providers.mistral import MistralProvider
from sec_generative_search.providers.openrouter import (
    OpenRouterProvider,
    OpenRouterRoutingHints,
)
from sec_generative_search.providers.qwen import QwenProvider

_LONG_KEY = "sk-or-v1-ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # pragma: allowlist secret
_KEY_TAIL = _LONG_KEY[-4:]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_openai(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the OpenAI SDK client with a captured ``MagicMock``.

    The completion response is shaped minimally so ``generate`` can
    produce a :class:`GenerationResponse` without any real network
    call.  Each test reads ``captured["kwargs"]`` to assert on the SDK
    call shape.
    """
    captured: dict[str, Any] = {}

    fake_usage = MagicMock(prompt_tokens=1, completion_tokens=2)
    fake_message = MagicMock(content="ok")
    fake_choice = MagicMock(message=fake_message, finish_reason="stop")
    fake_completion = MagicMock(
        choices=[fake_choice],
        usage=fake_usage,
        model="openai/gpt-5.4-mini",
    )

    def factory(**kwargs: Any) -> MagicMock:
        client = MagicMock()
        client.chat.completions.create.return_value = fake_completion

        def create(**call_kwargs: Any) -> MagicMock:
            captured["kwargs"] = call_kwargs
            return fake_completion

        client.chat.completions.create.side_effect = create
        captured["client"] = client
        captured["client_kwargs"] = kwargs
        return client

    monkeypatch.setattr(openai_compat, "OpenAI", factory)
    return captured


# ---------------------------------------------------------------------------
# OpenRouterRoutingHints shape
# ---------------------------------------------------------------------------


class TestOpenRouterRoutingHintsShape:
    """Dataclass is frozen, hashable, and pass-through serialises."""

    def test_defaults_are_empty_pass_through(self) -> None:
        hints = OpenRouterRoutingHints()
        assert hints.order == ()
        assert hints.allow_fallbacks is None
        assert hints.only == ()
        assert hints.ignore == ()
        assert hints.require_parameters is None
        assert hints.data_collection is None

    def test_is_hashable(self) -> None:
        """Frozen dataclass with tuple / scalar fields must be hashable."""
        a = OpenRouterRoutingHints(order=("anthropic", "openai"))
        b = OpenRouterRoutingHints(order=("anthropic", "openai"))
        # Equality and hash round-trip through a ``set`` without error.
        assert {a, b} == {a}

    def test_to_provider_block_omits_unset_fields(self) -> None:
        # Empty hints produce an empty block — OpenRouter's own defaults
        # apply and the provider adapter treats this as "no hints".
        assert OpenRouterRoutingHints().to_provider_block() == {}

    def test_to_provider_block_renders_every_field(self) -> None:
        hints = OpenRouterRoutingHints(
            order=("anthropic", "openai"),
            allow_fallbacks=False,
            only=("anthropic",),
            ignore=("grok",),
            require_parameters=True,
            data_collection="deny",
        )
        assert hints.to_provider_block() == {
            "order": ["anthropic", "openai"],
            "allow_fallbacks": False,
            "only": ["anthropic"],
            "ignore": ["grok"],
            "require_parameters": True,
            "data_collection": "deny",
        }

    def test_allow_fallbacks_true_is_rendered(self) -> None:
        """``None`` is "unset"; ``True`` must still reach the block."""
        assert OpenRouterRoutingHints(allow_fallbacks=True).to_provider_block() == {
            "allow_fallbacks": True,
        }

    def test_block_lists_are_fresh_copies(self) -> None:
        """Mutating the rendered block must not alias the tuple storage."""
        hints = OpenRouterRoutingHints(order=("openai",))
        block = hints.to_provider_block()
        block["order"].append("mutated")
        # Rendering again returns the original — the tuple is immutable
        # and ``to_provider_block`` builds a fresh list each call.
        assert hints.to_provider_block()["order"] == ["openai"]


# ---------------------------------------------------------------------------
# Forwarding — OpenRouterProvider
# ---------------------------------------------------------------------------


class TestOpenRouterForwardsHints:
    """OpenRouter forwards routing hints via SDK ``extra_body``."""

    def test_generate_without_hints_omits_extra_body(
        self,
        patched_openai: dict[str, Any],
    ) -> None:
        provider = OpenRouterProvider(_LONG_KEY)
        request = GenerationRequest(prompt="hello", model="openai/gpt-5.4-mini")
        provider.generate(request)
        kwargs = patched_openai["kwargs"]
        # No routing hints means the call shape is identical to every
        # other OpenAI-compatible vendor's.
        assert "extra_body" not in kwargs

    def test_generate_with_hints_sets_provider_block(
        self,
        patched_openai: dict[str, Any],
    ) -> None:
        provider = OpenRouterProvider(_LONG_KEY)
        hints = OpenRouterRoutingHints(
            order=("anthropic", "openai"),
            allow_fallbacks=False,
            data_collection="deny",
        )
        request = GenerationRequest(
            prompt="hello",
            model="openai/gpt-5.4-mini",
            routing_hints=hints,
        )
        provider.generate(request)

        kwargs = patched_openai["kwargs"]
        assert kwargs["extra_body"] == {
            "provider": {
                "order": ["anthropic", "openai"],
                "allow_fallbacks": False,
                "data_collection": "deny",
            },
        }

    def test_empty_hints_object_behaves_like_no_hints(
        self,
        patched_openai: dict[str, Any],
    ) -> None:
        """An empty :class:`OpenRouterRoutingHints` must degrade cleanly.

        Attaching an "all defaults" hint object is semantically the same
        as attaching no hints — we do not want to send OpenRouter an
        empty ``provider: {}`` block, which would be interpreted as
        "use whatever you want" and is misleading noise on the wire.
        """
        provider = OpenRouterProvider(_LONG_KEY)
        request = GenerationRequest(
            prompt="hello",
            model="openai/gpt-5.4-mini",
            routing_hints=OpenRouterRoutingHints(),
        )
        provider.generate(request)
        assert "extra_body" not in patched_openai["kwargs"]

    def test_generate_stream_forwards_hints(
        self,
        patched_openai: dict[str, Any],
    ) -> None:
        """Streaming path must honour hints too.

        Only the *request* shape matters here — we stub a trivial empty
        iterator as the stream body; the test asserts on the kwargs the
        SDK would have been called with.
        """
        provider = OpenRouterProvider(_LONG_KEY)
        client = patched_openai["client"]
        client.chat.completions.create.side_effect = None
        client.chat.completions.create.return_value = iter(())

        def capture(**call_kwargs: Any) -> Any:
            patched_openai["kwargs"] = call_kwargs
            return iter(())

        client.chat.completions.create.side_effect = capture

        hints = OpenRouterRoutingHints(only=("anthropic",))
        request = GenerationRequest(
            prompt="hello",
            model="openai/gpt-5.4-mini",
            routing_hints=hints,
        )
        list(provider.generate_stream(request))

        kwargs = patched_openai["kwargs"]
        assert kwargs["stream"] is True
        assert kwargs["extra_body"] == {"provider": {"only": ["anthropic"]}}


# ---------------------------------------------------------------------------
# Every other OpenAI-compatible vendor ignores hints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls",
    [MistralProvider, KimiProvider, DeepSeekProvider, QwenProvider],
)
def test_other_openai_compatible_vendors_ignore_hints(
    provider_cls: type,
    patched_openai: dict[str, Any],
) -> None:
    """Non-OpenRouter providers must not forward ``routing_hints``.

    The OpenAI-compatible base's default :meth:`_extra_request_kwargs`
    returns an empty dict — adding routing hints to the request must
    therefore leave the SDK kwargs unchanged.  If a future subclass
    overrode the hook, this test would catch the leak.
    """
    provider = provider_cls(_LONG_KEY)
    hints = OpenRouterRoutingHints(order=("anthropic",))
    request = GenerationRequest(
        prompt="hello",
        model=provider.default_model,
        routing_hints=hints,
    )
    provider.generate(request)
    assert "extra_body" not in patched_openai["kwargs"]


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


_SECRET_FIELD_HINTS = (
    "api_key",
    "api-key",
    "apikey",
    "secret",
    "password",
    "credential",
    "private_key",
    "auth_token",
    "bearer",
    "authorization",
    "token",
)


@pytest.mark.security
class TestRoutingHintsCarryNoCredentials:
    """Routing hints are a *routing* channel, not an auth channel.

    The OpenRouter API key remains the sole credential in flight across
    every SDK call.  These assertions fail fast if a future refactor
    widens the hint object into a credential pathway.
    """

    def test_hint_object_has_no_credential_shaped_fields(self) -> None:
        for f in fields(OpenRouterRoutingHints):
            lowered = f.name.lower()
            for hint in _SECRET_FIELD_HINTS:
                assert hint not in lowered, (
                    f"OpenRouterRoutingHints.{f.name} looks credential-bearing; "
                    "routing hints must never carry secrets."
                )

    def test_rendered_block_has_no_credential_shaped_keys(self) -> None:
        # Build a maximally-populated hint object and assert that nothing
        # credential-looking appears in the rendered ``provider`` block.
        hints = OpenRouterRoutingHints(
            order=("anthropic", "openai"),
            allow_fallbacks=True,
            only=("anthropic",),
            ignore=("grok",),
            require_parameters=True,
            data_collection="allow",
        )
        block = hints.to_provider_block()
        for key in block:
            lowered = key.lower()
            for hint in _SECRET_FIELD_HINTS:
                assert hint not in lowered, (
                    f"OpenRouter provider-block key {key!r} looks credential-bearing; "
                    "routing hints must be pass-through only."
                )

    def test_extra_body_never_contains_credential_shaped_keys(
        self,
        patched_openai: dict[str, Any],
    ) -> None:
        provider = OpenRouterProvider(_LONG_KEY)
        hints = OpenRouterRoutingHints(
            order=("anthropic",),
            data_collection="deny",
        )
        request = GenerationRequest(
            prompt="hello",
            model="openai/gpt-5.4-mini",
            routing_hints=hints,
        )
        provider.generate(request)

        extra_body = patched_openai["kwargs"].get("extra_body", {})
        # Walk the full rendered payload — top level + nested provider
        # block — and assert no credential-shaped key appears.
        stack: list[Any] = [extra_body]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for k, v in current.items():
                    lowered = k.lower()
                    for hint in _SECRET_FIELD_HINTS:
                        assert hint not in lowered, (
                            f"extra_body key {k!r} looks credential-bearing; "
                            "routing hints are not an auth channel."
                        )
                    stack.append(v)

    def test_openrouter_api_key_is_sole_credential_in_flight(
        self,
        patched_openai: dict[str, Any],
    ) -> None:
        """The SDK client is constructed with the OpenRouter key once.

        Forwarding hints must not inject a second key into the SDK call
        kwargs, a second client, or any header override.  We assert that
        the only place the key reaches the SDK is the client's
        ``api_key`` kwarg, and the call-time kwargs carry no secondary
        credential.
        """
        provider = OpenRouterProvider(_LONG_KEY)
        hints = OpenRouterRoutingHints(order=("anthropic",))
        request = GenerationRequest(
            prompt="hello",
            model="openai/gpt-5.4-mini",
            routing_hints=hints,
        )
        provider.generate(request)

        # Client constructed exactly once, with the OpenRouter key.
        client_kwargs = patched_openai["client_kwargs"]
        assert client_kwargs["api_key"] == _LONG_KEY

        # Call-time kwargs carry no credential material at all.  No SDK
        # headers/auth override arguments allowed through the hint path.
        call_kwargs = patched_openai["kwargs"]
        forbidden = {
            "api_key",
            "authorization",
            "headers",
            "default_headers",
            "bearer",
            "token",
        }
        assert forbidden.isdisjoint(call_kwargs), (
            f"SDK call unexpectedly received credential-shaped kwarg(s): "
            f"{forbidden & call_kwargs.keys()}"
        )

    def test_repr_still_redacts_key_under_hints(
        self,
        patched_openai: dict[str, Any],
    ) -> None:
        """Adding the hint path must not regress the base-class redaction."""
        del patched_openai
        provider = OpenRouterProvider(_LONG_KEY)
        text = repr(provider)
        assert _LONG_KEY not in text
        assert _KEY_TAIL in text


@pytest.mark.security
class TestGenerationRequestRoutingHintsFieldIsSafe:
    """The new :attr:`GenerationRequest.routing_hints` field must not
    widen the credential-field surface on the provider request type."""

    def test_routing_hints_field_is_not_credential_shaped(self) -> None:
        names = {f.name for f in fields(GenerationRequest)}
        assert "routing_hints" in names
        # The parametrised check in ``test_base.py`` already covers the
        # full hint list; this test documents intent at the site of the
        # change so a deliberate rename is flagged here too.
        assert "api_key" not in names
        assert "secret" not in names
