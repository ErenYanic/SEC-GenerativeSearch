"""Provider abstraction layer for SEC-GenerativeSearch.

Phase 5A lands the abstract base classes and supporting types; concrete
adapters (OpenAI, Anthropic, Gemini, OpenAI-compat vendors, local
embeddings) arrive in Phase 5B+.
"""

from sec_generative_search.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    BaseRerankerProvider,
    GenerationRequest,
    GenerationResponse,
    RerankResult,
)

__all__ = [
    "BaseEmbeddingProvider",
    "BaseLLMProvider",
    "BaseRerankerProvider",
    "GenerationRequest",
    "GenerationResponse",
    "RerankResult",
]
