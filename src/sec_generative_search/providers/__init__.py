"""Provider abstraction layer for SEC-GenerativeSearch.

Exports the abstract provider contracts, the concrete hosted-vendor
adapters, the optional on-device embedding provider, and the curated
registry used for capability lookup, model listings, and key
validation.
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
from sec_generative_search.providers.grok import GrokProvider
from sec_generative_search.providers.kimi import KimiProvider
from sec_generative_search.providers.local import LocalEmbeddingProvider
from sec_generative_search.providers.mimo import MimoProvider
from sec_generative_search.providers.minimax import MiniMaxProvider
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
from sec_generative_search.providers.zai import ZaiProvider

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
    "GrokProvider",
    "KimiProvider",
    "LocalEmbeddingProvider",
    "MimoProvider",
    "MiniMaxProvider",
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
    "ZaiProvider",
]
