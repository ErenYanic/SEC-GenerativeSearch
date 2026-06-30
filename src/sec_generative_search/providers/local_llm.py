"""Local-LLM provider adapter.

A self-hosted model server the operator runs on their own machine —
Ollama (the default), llama.cpp's ``llama-server``, vLLM, or LM Studio —
each of which exposes an OpenAI-compatible Chat Completions surface.  The
adapter is therefore a thin :class:`OpenAICompatibleLLMProvider` subclass:
it inherits the shared client construction, :func:`resilient_call`
plumbing, streaming, token accounting, and SDK exception mapping
unchanged, and differs only by ``provider_name`` and the loopback
``default_base_url`` / ``default_model`` an Ollama install ships with.

The provider ships with **no optional extra** — the ``openai`` SDK is a
core dependency, so ``local_llm`` is always available on the LLM surface.

Privacy framing: "local" means *a model server you control*, not "no
prompt ever leaves the process".  Generation still sends the assembled
prompt — including the user's question (Tier-3 data) and the retrieved
filing context — over the wire to that endpoint.  The privacy win is that
the endpoint is one the operator runs (loopback by default), so the
prompt need not reach a third-party hosted provider.  The base URL that
decides *where* the prompt goes is settings-driven and host-policy-guarded; 
this module pins only the shipped loopback default.

The endpoint accepts any model slug the local server has pulled, so the
provider is registered OpenRouter-style — ``supports_arbitrary_models``
with an empty vendored catalogue — and (like every non-OpenRouter vendor)
silently drops upstream-routing hints via the OpenAI-compatible base's
empty default :meth:`_extra_request_kwargs` hook.
"""

from __future__ import annotations

from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleLLMProvider,
)

__all__ = ["LocalLLMProvider"]


class LocalLLMProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting a self-hosted OpenAI-wire endpoint.

    Defaults target a stock Ollama install on the loopback interface;
    other backends (llama.cpp-server, vLLM, LM Studio) work by pointing the
    base URL at their own OpenAI-compatible port.  The capability probe
    returns the permissive default for every slug because the served model
    set is whatever the local server has pulled — the SDK surfaces an
    unserviceable slug at call time with the endpoint's own error.
    """

    provider_name = "local_llm"
    default_base_url = "http://127.0.0.1:11434/v1"
    default_model = "llama3.2"

    # Intentionally absent from the vendored catalogue — the served model
    # set is operator-defined, so the registry lists it OpenRouter-style
    # (``supports_arbitrary_models``) and the base class's permissive
    # capability default covers any slug the local server accepts.
