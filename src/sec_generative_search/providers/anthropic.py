"""Anthropic provider adapter.

Concrete :class:`BaseLLMProvider` that targets Anthropic's Messages API
via the first-party ``anthropic`` SDK.  The SDK's retry loop is disabled
(``max_retries=0``) so retry, timeout, and circuit-breaker behaviour
stay owned by :func:`resilient_call` — the same contract every other
provider in this package follows.

Key notes:

- Anthropic does not ship a hosted embedding model; the embedding path
  is out of scope.  Callers needing embeddings pair this with an
  embedding-only provider (OpenAI, Gemini) — the ``ProviderRegistry``
  in handles the dual-provider wiring.
- The Messages API returns a ``stop_reason`` of ``"refusal"`` when the
  safety system blocks the response.  We treat that as terminal and
  surface it as :class:`ProviderContentFilterError` — same shape as the
  OpenAI content-filter path.  Doing the check on the response body
  (rather than via the exception mapping) matches the SDK's own
  behaviour: refusals come back as valid HTTP 200 responses.
- ``count_tokens`` approximates with :mod:`tiktoken` ``cl100k_base``.
  The SDK ships an accurate ``messages.count_tokens`` but it is a
  network call; the context-window packer budgets prompts
  *before* the generation call, so it needs an offline counter.  The
  approximation is conservative (slightly over-counts), which is the
  right direction for a budget guard.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar

from anthropic import (
    Anthropic,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from sec_generative_search.core.exceptions import (
    ProviderContentFilterError,
)
from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.resilience import (
    ExceptionMapping,
    ResilientCallPolicy,
    RetryPolicy,
    resilient_call,
)
from sec_generative_search.core.types import (
    PricingTier,
    ProviderCapability,
    TokenUsage,
)
from sec_generative_search.providers.base import (
    BaseLLMProvider,
    GenerationRequest,
    GenerationResponse,
)
from sec_generative_search.providers.openai_compat import ModelInfo

if TYPE_CHECKING:
    pass


__all__ = [
    "ANTHROPIC_EXCEPTION_MAPPING",
    "AnthropicProvider",
]


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exception mapping
# ---------------------------------------------------------------------------

# The Anthropic SDK raises distinct typed exceptions, so a plain
# ``ExceptionMapping`` captures every non-terminal class cleanly.  Note
# that content-filter refusals are *not* represented here — they arrive
# as valid responses with ``stop_reason="refusal"`` and are raised from
# the response handler (terminal, bypassing the retry loop).
ANTHROPIC_EXCEPTION_MAPPING = ExceptionMapping(
    auth=(AuthenticationError, PermissionDeniedError),
    rate_limit=(RateLimitError,),
    timeout=(APITimeoutError, TimeoutError),
)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AnthropicProvider(BaseLLMProvider):
    """Chat-completion provider for Anthropic's hosted Claude models."""

    provider_name: ClassVar[str] = "anthropic"
    default_model: ClassVar[str] = "claude-haiku-4-5"
    default_timeout: ClassVar[float] = 60.0

    # Static capability probe — O(1) lookup, no network call at
    # registration.  Context windows and max-output figures come from
    # the Anthropic model cards; they change rarely and are surface-
    # only (the SDK will reject an over-long request at call time).
    MODEL_CATALOGUE: ClassVar[dict[str, ModelInfo]] = {
        "claude-opus-4-7": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=1_000_000,
                max_output_tokens=128_000,
                pricing_tier=PricingTier.PREMIUM,
            ),
        ),
        "claude-sonnet-4-6": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=1_000_000,
                max_output_tokens=64_000,
                pricing_tier=PricingTier.STANDARD,
            ),
        ),
        "claude-haiku-4-5": ModelInfo(
            capability=ProviderCapability(
                chat=True,
                streaming=True,
                tool_use=True,
                structured_output=True,
                prompt_caching=True,
                vision=True,
                context_window_tokens=200_000,
                max_output_tokens=64_000,
                pricing_tier=PricingTier.LOW,
            ),
        ),
    }

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(api_key)
        wall_clock = timeout if timeout is not None else self.default_timeout
        # Disable the SDK's own retry loop — ``resilient_call`` owns it.
        self._client = Anthropic(
            api_key=self._api_key,
            base_url=base_url,
            timeout=wall_clock,
            max_retries=0,
        )
        self._policy = ResilientCallPolicy(
            retry_policy=retry_policy or RetryPolicy(),
            exception_mapping=ANTHROPIC_EXCEPTION_MAPPING,
            timeout=0.0,
        )
        # Lazy tiktoken encoder — cl100k_base approximation (see
        # module docstring for why).
        self._encoder: Any | None = None

    def _call[T](self, fn: Callable[[], T]) -> T:
        return resilient_call(fn, provider=self.provider_name, policy=self._policy)

    # ------------------------------------------------------------------
    # Capability and validation
    # ------------------------------------------------------------------

    def validate_key(self) -> bool:
        """Probe the key with the cheapest authenticated call.

        ``client.models.list`` returns a small page behind the same
        authentication bucket as generation, so it is the right
        validation surface.  Any failure propagates as a normalised
        :class:`ProviderError` subclass via the resilience wrapper.
        """
        self._call(lambda: self._client.models.list())
        return True

    def get_capabilities(self, model: str | None = None) -> ProviderCapability:
        """Return the static capability matrix for *model*.

        Unknown slugs receive a permissive
        ``ProviderCapability(chat=True, streaming=True)`` — same
        semantics as the OpenAI-compatible bases, so the SDK rejects
        unsupported slugs at call time with a clear error rather than
        here.
        """
        slug = model or self.default_model
        info = self.MODEL_CATALOGUE.get(slug)
        if info is not None:
            return info.capability
        return ProviderCapability(chat=True, streaming=True)

    # ------------------------------------------------------------------
    # Generation — non-streaming
    # ------------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Non-streaming Messages API call."""
        kwargs = self._build_kwargs(request)

        def call() -> Any:
            return self._client.messages.create(stream=False, **kwargs)

        message = self._call(call)
        stop_reason = message.stop_reason or "end_turn"
        # Terminal safety refusal — surface before the caller ever sees
        # the empty content list.
        if stop_reason == "refusal":
            raise ProviderContentFilterError(
                f"{self.provider_name} safety filter refused to respond",
                provider=self.provider_name,
                hint="Reformulate the prompt or route to a different provider.",
            )

        text = self._concat_text_blocks(message.content)
        usage = message.usage
        token_usage = TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
        return GenerationResponse(
            text=text,
            model=getattr(message, "model", request.model or self.default_model),
            token_usage=token_usage,
            finish_reason=self._normalise_stop_reason(stop_reason),
        )

    # ------------------------------------------------------------------
    # Generation — streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationResponse]:
        """Streaming Messages API call.

        Emits one :class:`GenerationResponse` per text-delta event plus
        a final usage-only frame so callers can aggregate token counts
        consistently with the OpenAI-compatible path.
        """
        kwargs = self._build_kwargs(request)
        model_slug = request.model or self.default_model

        def call() -> Any:
            return self._client.messages.create(stream=True, **kwargs)

        stream = self._call(call)

        # Anthropic separates input and output token reporting: the
        # opening ``message_start`` event carries ``input_tokens`` but
        # ``output_tokens=0``; each subsequent ``message_delta`` event
        # replaces ``output_tokens`` with the running total.  Hold the
        # latest value and emit a final usage-only frame.
        input_tokens = 0
        output_tokens = 0
        final_stop_reason = "end_turn"

        for event in stream:
            event_type = getattr(event, "type", "")

            if event_type == "message_start":
                message = getattr(event, "message", None)
                if message is not None:
                    usage = getattr(message, "usage", None)
                    if usage is not None:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                continue

            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                delta_text = getattr(delta, "text", "") if delta is not None else ""
                if delta_text:
                    yield GenerationResponse(
                        text=delta_text,
                        model=model_slug,
                        token_usage=TokenUsage(),
                        finish_reason="stop",
                    )
                continue

            if event_type == "message_delta":
                delta = getattr(event, "delta", None)
                stop_reason = getattr(delta, "stop_reason", None) if delta is not None else None
                if stop_reason == "refusal":
                    raise ProviderContentFilterError(
                        f"{self.provider_name} safety filter refused mid-stream",
                        provider=self.provider_name,
                        hint="Reformulate the prompt or route to a different provider.",
                    )
                if stop_reason:
                    final_stop_reason = stop_reason
                usage = getattr(event, "usage", None)
                if usage is not None:
                    output_tokens = getattr(usage, "output_tokens", output_tokens) or output_tokens
                continue

            # ``message_stop`` (and any unknown event types) ends the
            # conversation — nothing to yield here.
            if event_type == "message_stop":
                break

        yield GenerationResponse(
            text="",
            model=model_slug,
            token_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            finish_reason=self._normalise_stop_reason(final_stop_reason),
        )

    # ------------------------------------------------------------------
    # Token counting — offline approximation
    # ------------------------------------------------------------------

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Return an offline token count for *text*.

        Uses :mod:`tiktoken`'s ``cl100k_base`` encoding as a conservative
        approximation.  The context-window packer only needs
        the count to stay *below* the model's window — a slight
        over-count biases towards safety.  The SDK's own
        ``messages.count_tokens`` is exact but a network call; we
        avoid it in this hot path.
        """
        del model  # approximation does not vary by Claude model
        if self._encoder is None:
            import tiktoken

            self._encoder = tiktoken.get_encoding("cl100k_base")
        return len(self._encoder.encode(text))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        """Render a :class:`GenerationRequest` into Messages API kwargs.

        The Anthropic API separates the ``system`` prompt from the
        ``messages`` array; we honour that split so system framing is
        not accidentally interpreted as a user turn.
        """
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            kwargs["system"] = request.system
        return kwargs

    @staticmethod
    def _concat_text_blocks(blocks: list[Any]) -> str:
        """Concatenate the text of every ``type=="text"`` block."""
        parts: list[str] = []
        for block in blocks or []:
            if getattr(block, "type", "") == "text":
                text = getattr(block, "text", "") or ""
                if text:
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _normalise_stop_reason(stop_reason: str) -> str:
        """Map Anthropic stop reasons onto the shared finish-reason vocabulary.

        ``GenerationResponse.finish_reason`` is provider-neutral; a
        ``"length"`` signal from the OpenAI path and Anthropic's
        ``"max_tokens"`` should read the same downstream.
        """
        return {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_use",
            "pause_turn": "pause",
            "refusal": "content_filter",
        }.get(stop_reason, stop_reason or "stop")
