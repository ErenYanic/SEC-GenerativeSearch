"""Provider abstraction layer for SEC-GenerativeSearch.

Phase 5A lands the abstract base classes and supporting types; Phase 5B
introduces the OpenAI wire-compatible plumbing; Phase 5C adds first-
party Anthropic and Google Gemini adapters; Phase 5D ships concrete
OpenAI-compatible subclasses for Mistral, Kimi (Moonshot), DeepSeek,
Qwen (DashScope), and OpenRouter.  The local embedding provider
arrives in Phase 5E.
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
from sec_generative_search.providers.deepseek import DeepSeekProvider
from sec_generative_search.providers.gemini import (
    GeminiEmbeddingProvider,
    GeminiProvider,
)
from sec_generative_search.providers.kimi import KimiProvider
from sec_generative_search.providers.mistral import (
    MistralEmbeddingProvider,
    MistralProvider,
)
from sec_generative_search.providers.openai import (
    OpenAIEmbeddingProvider,
    OpenAIProvider,
)
from sec_generative_search.providers.openrouter import OpenRouterProvider
from sec_generative_search.providers.qwen import (
    QwenEmbeddingProvider,
    QwenProvider,
)

__all__ = [
    "AnthropicProvider",
    "BaseEmbeddingProvider",
    "BaseLLMProvider",
    "BaseRerankerProvider",
    "DeepSeekProvider",
    "GeminiEmbeddingProvider",
    "GeminiProvider",
    "GenerationRequest",
    "GenerationResponse",
    "KimiProvider",
    "MistralEmbeddingProvider",
    "MistralProvider",
    "OpenAIEmbeddingProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "QwenEmbeddingProvider",
    "QwenProvider",
    "RerankResult",
]
