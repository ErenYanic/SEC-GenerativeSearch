"""Read-tier embedder resource introspection.

Single endpoint: ``GET /api/resources/gpu``. Surfaces the operator-facing
load state of the lifespan-managed embedder attached to ``app.state``.

The route is intentionally read-only — there is no ``DELETE`` /
admin-unload counterpart yet. The caller-driven idle-unload hook is the
sanctioned eviction path on the current package surface.

Privacy contract:

    - No file system paths. The embedder is identified by ``provider`` /
      ``model`` slugs only.
    - No credentials. The HF token / admin-env key is never read from
      this seam; :class:`_ProviderBase` keeps it off ``repr`` and the
      response schema has no field that could accept one.
    - For on-device loads, only the elapsed ``idle_seconds`` is surfaced
      — the underlying ``LocalEmbeddingProvider._last_used`` monotonic
      timestamp would leak process start-time information without
      operator value.

This route never triggers a model load: the read of
:attr:`LocalEmbeddingProvider.is_loaded` is a property over a private
field and does not call ``_ensure_model``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from sec_generative_search.api.dependencies import get_embedder, verify_api_key
from sec_generative_search.api.schemas import GpuStatusResponse
from sec_generative_search.config.settings import get_settings
from sec_generative_search.providers.base import BaseEmbeddingProvider
from sec_generative_search.providers.local import LocalEmbeddingProvider

__all__ = ["router"]


router = APIRouter(dependencies=[Depends(verify_api_key)])


def _build_status(embedder: BaseEmbeddingProvider) -> GpuStatusResponse:
    """Project the embedder onto the wire schema.

    Branches on :class:`LocalEmbeddingProvider` because the hosted
    embedders (OpenAI, Gemini, Mistral, Qwen, ...) are stateless HTTP
    clients with no load state to surface. Hosted embedders therefore
    collapse to ``is_local=False`` / ``is_loaded=True`` / no device /
    no idle data so the operator UI can render a single layout for
    both shapes without a separate "not applicable" code path.
    """
    settings = get_settings().embedding

    if isinstance(embedder, LocalEmbeddingProvider):
        is_loaded = embedder.is_loaded
        # ``_resolved_device`` is None until the first load. We only
        # surface it while the model is resident — once unloaded, the
        # device choice is just configuration, not state.
        device = embedder._resolved_device if is_loaded else None
        # Idle elapsed time is derived against the provider's injected
        # clock so the test suite (clock injection) sees consistent
        # values. ``_last_used`` is the monotonic timestamp recorded by
        # ``_mark_used``; we never surface the raw value.
        idle_seconds: float | None = None
        if is_loaded and embedder._last_used is not None:
            elapsed = embedder._clock() - embedder._last_used
            # Clamp to >= 0 against an injected clock that runs backwards
            # in tests; the API contract is "non-negative seconds".
            idle_seconds = max(0.0, elapsed)
        idle_timeout_minutes = embedder._idle_timeout_seconds // 60
        return GpuStatusResponse(
            provider=settings.provider,
            model=settings.model_name,
            is_local=True,
            is_loaded=is_loaded,
            device=device,
            idle_seconds=idle_seconds,
            idle_timeout_minutes=idle_timeout_minutes,
        )

    # Hosted embedder — no load state, no device, no idle window. We
    # report ``is_loaded=True`` because every successful call lands on a
    # ready endpoint; the field exists so the local case has somewhere
    # to surface load state without a separate route shape.
    return GpuStatusResponse(
        provider=settings.provider,
        model=settings.model_name,
        is_local=False,
        is_loaded=True,
        device=None,
        idle_seconds=None,
        idle_timeout_minutes=0,
    )


@router.get(
    "/gpu",
    response_model=GpuStatusResponse,
    tags=["resources"],
    summary="Embedder load-state snapshot",
)
async def gpu_status(
    embedder: BaseEmbeddingProvider = Depends(get_embedder),
) -> GpuStatusResponse:
    """Return the lifespan-managed embedder's load-state snapshot.

    The route never triggers a model load — reading ``is_loaded`` and
    the underlying private fields is a pure attribute access. Hosted
    embedders collapse to a static "ready" shape because they have no
    load state to surface.
    """
    return _build_status(embedder)
