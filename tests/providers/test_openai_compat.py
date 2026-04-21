"""Tests for :mod:`sec_generative_search.providers.openai_compat`.

Covers the shared OpenAI-compatible plumbing used by every vendor that
speaks the OpenAI Chat Completions / Embeddings wire protocol:

- Non-streaming generation: text + token-usage extraction.
- Streaming generation: per-chunk yields plus the final usage-only frame.
- Content-filter responses (``finish_reason="content_filter"``) raise
  :class:`ProviderContentFilterError` and are *not* retried.
- Auth errors (``AuthenticationError``, ``PermissionDeniedError``) are
  normalised to :class:`ProviderAuthError` and not retried.
- Rate-limit errors normalise to :class:`ProviderRateLimitError` and
  are retried.
- Timeout errors (``APITimeoutError`` and the stdlib ``TimeoutError``)
  normalise to :class:`ProviderTimeoutError` and are retried.
- Embeddings: empty input bypasses the network; non-empty input
  produces a ``(n, dimension)`` float32 array.
- Capability matrix: known model returns the catalogue entry; unknown
  model falls back to the permissive default.
- Security: API keys never appear in error messages, repr/str output,
  or any package log emission.

Every test mocks the OpenAI SDK client — no real network calls are
issued and no live API key is required.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest

from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.core.resilience import RetryPolicy
from sec_generative_search.core.types import PricingTier, ProviderCapability
from sec_generative_search.providers import openai_compat
from sec_generative_search.providers.base import GenerationRequest
from sec_generative_search.providers.openai_compat import (
    OPENAI_EXCEPTION_MAPPING,
    ModelInfo,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLLMProvider,
)

# ---------------------------------------------------------------------------
# SDK exception subclasses that bypass the heavy ``httpx.Response``
# constructors — isinstance() checks still hit the OpenAI base classes.
# ---------------------------------------------------------------------------


class _FakeAuthError(openai_compat.AuthenticationError):
    def __init__(self, message: str = "bad key") -> None:
        Exception.__init__(self, message)


class _FakePermissionDenied(openai_compat.PermissionDeniedError):
    def __init__(self, message: str = "no access") -> None:
        Exception.__init__(self, message)


class _FakeRateLimit(openai_compat.RateLimitError):
    def __init__(self, message: str = "429") -> None:
        Exception.__init__(self, message)


class _FakeAPITimeout(openai_compat.APITimeoutError):
    def __init__(self, message: str = "timed out") -> None:
        Exception.__init__(self, message)


# ---------------------------------------------------------------------------
# Concrete test doubles for the abstract OpenAI-compatible bases.
# ---------------------------------------------------------------------------


_LONG_KEY = "sk-test-ABCDEFGHIJKLMNOPQRSTUV"
_KEY_TAIL = _LONG_KEY[-4:]


class _DemoLLM(OpenAICompatibleLLMProvider):
    """Concrete subclass used only by the test suite."""

    provider_name = "demo-vendor"
    default_model = "demo-chat"
    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {
        "demo-chat": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                context_window_tokens=8192,
                max_output_tokens=2048,
                pricing_tier=PricingTier.LOW,
            ),
        ),
    }


class _DemoEmbed(OpenAICompatibleEmbeddingProvider):
    provider_name = "demo-vendor"
    default_model = "demo-embed"
    MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {"demo-embed": 8}


# ---------------------------------------------------------------------------
# Helpers — fast retry policy + chunk/completion fixtures
# ---------------------------------------------------------------------------


def _fast_retry() -> RetryPolicy:
    """Tight retry schedule so test runtime stays in milliseconds."""
    return RetryPolicy(max_retries=2, backoff_base=2.0, initial_delay=0.0, max_delay=0.0)


def _make_completion(
    *,
    text: str = "hello",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    model: str = "demo-chat",
) -> MagicMock:
    """Build a SimpleNamespace-shaped object mirroring the SDK's response."""
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


def _make_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    model: str = "demo-chat",
    usage: tuple[int, int] | None = None,
) -> MagicMock:
    """Build a streaming-chunk shape (``choices[0].delta`` + optional usage)."""
    chunk = MagicMock()
    chunk.model = model
    if usage is not None:
        usage_obj = MagicMock()
        usage_obj.prompt_tokens, usage_obj.completion_tokens = usage
        chunk.usage = usage_obj
    else:
        chunk.usage = None
    if content is None and finish_reason is None and usage is not None:
        chunk.choices = []
        return chunk
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    return chunk


# ---------------------------------------------------------------------------
# Fixture: build a provider whose underlying ``OpenAI`` client is a
# MagicMock.  We patch the constructor in :mod:`openai_compat` so the
# real network client is never instantiated.
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_provider(monkeypatch: pytest.MonkeyPatch) -> _DemoLLM:
    fake_client = MagicMock()
    monkeypatch.setattr(openai_compat, "OpenAI", lambda **_kwargs: fake_client)
    provider = _DemoLLM(_LONG_KEY, retry_policy=_fast_retry())
    provider._fake_client = fake_client  # type: ignore[attr-defined]
    return provider


@pytest.fixture
def embed_provider(monkeypatch: pytest.MonkeyPatch) -> _DemoEmbed:
    fake_client = MagicMock()
    monkeypatch.setattr(openai_compat, "OpenAI", lambda **_kwargs: fake_client)
    provider = _DemoEmbed(_LONG_KEY, retry_policy=_fast_retry())
    provider._fake_client = fake_client  # type: ignore[attr-defined]
    return provider


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_llm_init_stores_api_key_under_underscore(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(openai_compat, "OpenAI", lambda **_kwargs: MagicMock())
        provider = _DemoLLM(_LONG_KEY)
        assert provider._api_key == _LONG_KEY

    def test_llm_passes_key_to_sdk_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def factory(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(openai_compat, "OpenAI", factory)
        _DemoLLM(_LONG_KEY)
        assert captured["api_key"] == _LONG_KEY
        # Retries are owned by ``resilient_call`` — the SDK must be configured
        # with ``max_retries=0`` so the two layers never race.
        assert captured["max_retries"] == 0
        # default_base_url is None on the bases — SDK uses canonical endpoint.
        assert captured["base_url"] is None

    def test_explicit_base_url_overrides_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            openai_compat,
            "OpenAI",
            lambda **kwargs: captured.update(kwargs) or MagicMock(),
        )
        _DemoLLM(_LONG_KEY, base_url="https://api.demo-vendor.example/v1")
        assert captured["base_url"] == "https://api.demo-vendor.example/v1"

    def test_embed_rejects_unknown_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_compat, "OpenAI", lambda **_kwargs: MagicMock())
        with pytest.raises(ValueError, match="Unknown embedding model"):
            _DemoEmbed(_LONG_KEY, model="not-real")


# ---------------------------------------------------------------------------
# Generation — non-streaming
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_text_and_usage(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.chat.completions.create.return_value = _make_completion(
            text="42",
            prompt_tokens=12,
            completion_tokens=3,
        )
        response = llm_provider.generate(GenerationRequest(prompt="ultimate?", model="demo-chat"))
        assert response.text == "42"
        assert response.token_usage.input_tokens == 12
        assert response.token_usage.output_tokens == 3
        assert response.finish_reason == "stop"

    def test_uses_default_model_when_none_supplied(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.chat.completions.create.return_value = _make_completion()
        llm_provider.generate(GenerationRequest(prompt="hi", model=""))
        call_kwargs = llm_provider._fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "demo-chat"
        assert call_kwargs["stream"] is False

    def test_system_prompt_prepended(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.chat.completions.create.return_value = _make_completion()
        llm_provider.generate(
            GenerationRequest(prompt="user message", model="demo-chat", system="be terse")
        )
        messages = llm_provider._fake_client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "be terse"}
        assert messages[1] == {"role": "user", "content": "user message"}

    def test_content_filter_finish_reason_raises(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.chat.completions.create.return_value = _make_completion(
            finish_reason="content_filter",
        )
        with pytest.raises(ProviderContentFilterError):
            llm_provider.generate(GenerationRequest(prompt="bad", model="demo-chat"))

    def test_missing_message_content_yields_empty_string(self, llm_provider: _DemoLLM) -> None:
        completion = _make_completion()
        completion.choices[0].message.content = None
        llm_provider._fake_client.chat.completions.create.return_value = completion
        response = llm_provider.generate(GenerationRequest(prompt="x", model="demo-chat"))
        assert response.text == ""


# ---------------------------------------------------------------------------
# Generation — streaming
# ---------------------------------------------------------------------------


class TestGenerateStream:
    def test_yields_per_chunk_and_final_usage(self, llm_provider: _DemoLLM) -> None:
        chunks = [
            _make_chunk(content="hel"),
            _make_chunk(content="lo"),
            _make_chunk(content=None, usage=(20, 7)),  # final usage frame
        ]
        llm_provider._fake_client.chat.completions.create.return_value = iter(chunks)

        responses = list(
            llm_provider.generate_stream(GenerationRequest(prompt="hi", model="demo-chat"))
        )
        assert [r.text for r in responses[:2]] == ["hel", "lo"]
        # Earlier chunks carry empty usage; final frame carries totals.
        assert responses[-1].text == ""
        assert responses[-1].token_usage.input_tokens == 20
        assert responses[-1].token_usage.output_tokens == 7

    def test_stream_passes_include_usage_option(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.chat.completions.create.return_value = iter(
            [_make_chunk(content=None, usage=(1, 1))]
        )
        list(llm_provider.generate_stream(GenerationRequest(prompt="hi", model="demo-chat")))
        kwargs = llm_provider._fake_client.chat.completions.create.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}

    def test_content_filter_in_stream_raises(self, llm_provider: _DemoLLM) -> None:
        chunks = [
            _make_chunk(content="ok"),
            _make_chunk(content=None, finish_reason="content_filter"),
        ]
        llm_provider._fake_client.chat.completions.create.return_value = iter(chunks)
        gen = llm_provider.generate_stream(GenerationRequest(prompt="x", model="demo-chat"))
        # First chunk yields fine; second raises terminally.
        first = next(gen)
        assert first.text == "ok"
        with pytest.raises(ProviderContentFilterError):
            next(gen)


# ---------------------------------------------------------------------------
# Error normalisation through resilient_call
# ---------------------------------------------------------------------------


class TestErrorMapping:
    def test_auth_error_normalised_and_not_retried(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.chat.completions.create.side_effect = _FakeAuthError()
        with pytest.raises(ProviderAuthError):
            llm_provider.generate(GenerationRequest(prompt="x", model="demo-chat"))
        assert llm_provider._fake_client.chat.completions.create.call_count == 1

    def test_permission_denied_treated_as_auth(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.chat.completions.create.side_effect = _FakePermissionDenied()
        with pytest.raises(ProviderAuthError):
            llm_provider.generate(GenerationRequest(prompt="x", model="demo-chat"))
        assert llm_provider._fake_client.chat.completions.create.call_count == 1

    def test_rate_limit_normalised_and_retried(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.chat.completions.create.side_effect = _FakeRateLimit()
        with pytest.raises(ProviderRateLimitError):
            llm_provider.generate(GenerationRequest(prompt="x", model="demo-chat"))
        # Initial + 2 retries = 3 attempts under the fast retry policy.
        assert llm_provider._fake_client.chat.completions.create.call_count == 3

    def test_api_timeout_normalised_and_retried(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.chat.completions.create.side_effect = _FakeAPITimeout()
        with pytest.raises(ProviderTimeoutError):
            llm_provider.generate(GenerationRequest(prompt="x", model="demo-chat"))
        assert llm_provider._fake_client.chat.completions.create.call_count == 3

    def test_builtin_timeout_also_normalised(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.chat.completions.create.side_effect = TimeoutError("slow")
        with pytest.raises(ProviderTimeoutError):
            llm_provider.generate(GenerationRequest(prompt="x", model="demo-chat"))

    def test_eventual_success_after_retry(self, llm_provider: _DemoLLM) -> None:
        attempts: list[int] = []

        def side_effect(**_kwargs: Any) -> Any:
            attempts.append(1)
            if len(attempts) < 2:
                raise _FakeRateLimit()
            return _make_completion(text="ok")

        llm_provider._fake_client.chat.completions.create.side_effect = side_effect
        response = llm_provider.generate(GenerationRequest(prompt="x", model="demo-chat"))
        assert response.text == "ok"
        assert len(attempts) == 2

    def test_unknown_exception_falls_back_to_provider_error(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.chat.completions.create.side_effect = RuntimeError("boom")
        with pytest.raises(ProviderError):
            llm_provider.generate(GenerationRequest(prompt="x", model="demo-chat"))


# ---------------------------------------------------------------------------
# Validation and capability matrix
# ---------------------------------------------------------------------------


class TestValidationAndCapabilities:
    def test_validate_key_calls_models_list(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.models.list.return_value = MagicMock()
        assert llm_provider.validate_key() is True
        assert llm_provider._fake_client.models.list.call_count == 1

    def test_validate_key_propagates_auth_error(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.models.list.side_effect = _FakeAuthError()
        with pytest.raises(ProviderAuthError):
            llm_provider.validate_key()

    def test_known_model_returns_catalogue_capability(self, llm_provider: _DemoLLM) -> None:
        caps = llm_provider.get_capabilities("demo-chat")
        assert caps.streaming is True
        assert caps.context_window_tokens == 8192

    def test_unknown_model_returns_permissive_default(self, llm_provider: _DemoLLM) -> None:
        caps = llm_provider.get_capabilities("never-heard-of")
        assert caps.chat is True
        assert caps.streaming is True
        assert caps.context_window_tokens == 0

    def test_default_model_used_when_none_supplied(self, llm_provider: _DemoLLM) -> None:
        caps = llm_provider.get_capabilities()
        assert caps.context_window_tokens == 8192


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_count_tokens_returns_positive(self, llm_provider: _DemoLLM) -> None:
        n = llm_provider.count_tokens("Apple Inc. reported strong revenues.")
        assert n > 0

    def test_count_tokens_caches_encoder(self, llm_provider: _DemoLLM) -> None:
        llm_provider.count_tokens("first call")
        cached = dict(llm_provider._encoders)
        llm_provider.count_tokens("second call")
        assert dict(llm_provider._encoders) == cached  # no new encoder built


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_empty_input_avoids_network(self, embed_provider: _DemoEmbed) -> None:
        out = embed_provider.embed_texts([])
        assert out.shape == (0, 8)
        embed_provider._fake_client.embeddings.create.assert_not_called()

    def test_embed_texts_returns_array(self, embed_provider: _DemoEmbed) -> None:
        item_a = MagicMock()
        item_a.embedding = [0.1] * 8
        item_b = MagicMock()
        item_b.embedding = [0.2] * 8
        response = MagicMock()
        response.data = [item_a, item_b]
        embed_provider._fake_client.embeddings.create.return_value = response

        out = embed_provider.embed_texts(["one", "two"])
        assert out.shape == (2, 8)
        assert out.dtype == np.float32

    def test_embed_query_returns_1d_vector(self, embed_provider: _DemoEmbed) -> None:
        item = MagicMock()
        item.embedding = [0.3] * 8
        response = MagicMock()
        response.data = [item]
        embed_provider._fake_client.embeddings.create.return_value = response
        out = embed_provider.embed_query("hello")
        assert out.shape == (8,)

    def test_embed_passes_correct_model(self, embed_provider: _DemoEmbed) -> None:
        item = MagicMock()
        item.embedding = [0.0] * 8
        response = MagicMock()
        response.data = [item]
        embed_provider._fake_client.embeddings.create.return_value = response
        embed_provider.embed_texts(["x"])
        assert embed_provider._fake_client.embeddings.create.call_args.kwargs["model"] == (
            "demo-embed"
        )


# ---------------------------------------------------------------------------
# Mapping sanity
# ---------------------------------------------------------------------------


class TestExceptionMapping:
    def test_mapping_contains_expected_types(self) -> None:
        assert openai_compat.AuthenticationError in OPENAI_EXCEPTION_MAPPING.auth
        assert openai_compat.PermissionDeniedError in OPENAI_EXCEPTION_MAPPING.auth
        assert openai_compat.RateLimitError in OPENAI_EXCEPTION_MAPPING.rate_limit
        assert openai_compat.APITimeoutError in OPENAI_EXCEPTION_MAPPING.timeout

    def test_content_filter_is_response_signal_not_exception(self) -> None:
        # Content-filter is surfaced from the response body, not as an SDK
        # exception, so the mapping must not contain a content_filter type
        # (otherwise we would normalise an unrelated exception).
        assert OPENAI_EXCEPTION_MAPPING.content_filter == ()


# ---------------------------------------------------------------------------
# Security — keys never leak
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecretSafety:
    def test_repr_masks_key(self, llm_provider: _DemoLLM) -> None:
        text = repr(llm_provider)
        assert _LONG_KEY not in text
        assert _KEY_TAIL in text

    def test_provider_error_does_not_leak_key(self, llm_provider: _DemoLLM) -> None:
        llm_provider._fake_client.chat.completions.create.side_effect = _FakeAuthError(
            "tool error mentions key"
        )
        with pytest.raises(ProviderAuthError) as excinfo:
            llm_provider.generate(GenerationRequest(prompt="x", model="demo-chat"))
        # Whatever the SDK puts in the message, the normalised error
        # must not echo the API key.
        assert _LONG_KEY not in str(excinfo.value)
        assert _LONG_KEY not in str(excinfo.value.details or "")
        assert _LONG_KEY not in str(excinfo.value.hint or "")

    def test_provider_layer_logs_never_contain_key(
        self,
        caplog: pytest.LogCaptureFixture,
        llm_provider: _DemoLLM,
    ) -> None:
        package_logger = logging.getLogger(LOGGER_NAME)
        previous = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
                llm_provider._fake_client.chat.completions.create.return_value = _make_completion()
                llm_provider.generate(GenerationRequest(prompt="hi", model="demo-chat"))
        finally:
            package_logger.propagate = previous
        for record in caplog.records:
            assert _LONG_KEY not in record.getMessage()
            assert _LONG_KEY not in str(record.args)


# ---------------------------------------------------------------------------
# Streaming generator iteration sanity (signal: yields don't blow up
# when the SDK emits zero-content deltas)
# ---------------------------------------------------------------------------


class TestStreamingEdgeCases:
    def test_empty_delta_then_stop_does_not_emit_dummy_frame(
        self,
        llm_provider: _DemoLLM,
    ) -> None:
        chunks = [
            _make_chunk(content=None, finish_reason="stop"),
            _make_chunk(content=None, usage=(5, 1)),
        ]
        llm_provider._fake_client.chat.completions.create.return_value = iter(chunks)
        out = list(llm_provider.generate_stream(GenerationRequest(prompt="x", model="demo-chat")))
        # Only the usage frame should be emitted.
        assert len(out) == 1
        assert out[0].text == ""
        assert out[0].token_usage.total_tokens == 6


# ---------------------------------------------------------------------------
# Iterator typing — ``generate_stream`` must yield an Iterator
# ---------------------------------------------------------------------------


def test_generate_stream_returns_iterator(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.chat.completions.create.return_value = iter([_make_chunk(content="x")])
    monkeypatch.setattr(openai_compat, "OpenAI", lambda **_kwargs: fake)
    provider = _DemoLLM(_LONG_KEY, retry_policy=_fast_retry())
    stream = provider.generate_stream(GenerationRequest(prompt="hi", model="demo-chat"))
    assert isinstance(stream, Iterator)
