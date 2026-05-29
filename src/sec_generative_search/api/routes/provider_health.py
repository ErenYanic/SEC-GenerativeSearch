"""Admin-gated passive provider-health introspection.

Single endpoint: ``GET /api/providers/health``. Surfaces the process-
global health snapshot so an operator can see, per LLM provider, the
breaker state and a handful of failure / latency counters recorded by the
generation path.

The handler is admin-gated, passive, content-free, and uncached. It reads
an in-memory snapshot only, never instantiates a provider, never touches a
credential, and never makes a network call.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from sec_generative_search.api.dependencies import admin_route_dependencies
from sec_generative_search.api.schemas import (
    ProviderHealthResponse,
    ProviderHealthSchema,
)
from sec_generative_search.core.provider_health import get_provider_health

__all__ = ["router"]


router = APIRouter(dependencies=admin_route_dependencies())


@router.get(
    "/health",
    response_model=ProviderHealthResponse,
    tags=["providers"],
    summary="Passive provider health snapshot (admin only)",
)
async def provider_health() -> JSONResponse:
    """Return the per-provider circuit-breaker + counter snapshot.

    Lifts each
    :class:`~sec_generative_search.core.provider_health.ProviderHealthSnapshot`
    onto the wire as an explicit :class:`ProviderHealthSchema` — never an
    ``**asdict()`` splat — so a future snapshot field cannot leak onto this
    surface without a deliberate schema change.

    The body is rendered through :class:`JSONResponse` so the
    ``Cache-Control: no-store`` header rides every response without a
    second middleware hop; the route itself makes no upstream call and
    reads no credential.
    """
    snapshots = get_provider_health().snapshot()
    payload = ProviderHealthResponse(
        providers=[
            ProviderHealthSchema(
                provider=snap.provider,
                state=snap.state,
                consecutive_failures=snap.consecutive_failures,
                total_failures=snap.total_failures,
                total_successes=snap.total_successes,
                last_error_type=snap.last_error_type,
                last_failure_seconds_ago=snap.last_failure_seconds_ago,
                last_latency_seconds=snap.last_latency_seconds,
            )
            for snap in snapshots
        ],
        total=len(snapshots),
    )
    return JSONResponse(
        content=payload.model_dump(mode="json"),
        headers={"Cache-Control": "no-store"},
    )
