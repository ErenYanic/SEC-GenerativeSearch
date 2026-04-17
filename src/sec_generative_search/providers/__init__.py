"""Provider abstraction layer for SEC-GenerativeSearch.

Phase 5A lands the abstract base classes and supporting types; Phase 5B
introduces the OpenAI wire-compatible plumbing; Phase 5C adds first-
party Anthropic and Google Gemini adapters.  Further OpenAI-compatible
vendors and the local embedding provider arrive in Phase 5D-5E.
"""

from sec_generative_search.providers.anthropic import AnthropicProvider
from sec_generative_search.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    BaseRerankerProvider,
    GenerationRequest,
    GenerationResponse,
    RerankResult,
)
from sec_generative_search.providers.gemini import (
    GeminiEmbeddingProvider,
    GeminiProvider,
)
from sec_generative_search.providers.openai import (
    OpenAIEmbeddingProvider,
    OpenAIProvider,
)

__all__ = [
    "AnthropicProvider",
    "BaseEmbeddingProvider",
    "BaseLLMProvider",
    "BaseRerankerProvider",
    "GeminiEmbeddingProvider",
    "GeminiProvider",
    "GenerationRequest",
    "GenerationResponse",
    "OpenAIEmbeddingProvider",
    "OpenAIProvider",
    "RerankResult",
]
