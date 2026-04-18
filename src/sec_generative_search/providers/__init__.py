"""Provider abstraction layer for SEC-GenerativeSearch.

Phase 5A lands the abstract base classes and supporting types; Phase 5B
introduces the OpenAI wire-compatible plumbing; Phase 5C adds first-
party Anthropic and Google Gemini adapters; Phase 5D ships concrete
OpenAI-compatible subclasses for Mistral, Kimi (Moonshot), DeepSeek,
Qwen (DashScope), and OpenRouter; Phase 5E adds the on-device
``LocalEmbeddingProvider`` backed by ``sentence-transformers``; Phase 5F
adds the curated :class:`ProviderRegistry` for capability lookup, model
listings, and key validation.
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
from sec_generative_search.providers.local import LocalEmbeddingProvider
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
from sec_generative_search.providers.registry import (
    ProviderEntry,
    ProviderRegistry,
    ProviderSurface,
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
    "LocalEmbeddingProvider",
    "MistralEmbeddingProvider",
    "MistralProvider",
    "OpenAIEmbeddingProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderEntry",
    "ProviderRegistry",
    "ProviderSurface",
    "QwenEmbeddingProvider",
    "QwenProvider",
    "RerankResult",
]
