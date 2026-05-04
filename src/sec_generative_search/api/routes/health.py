"""Health-check route.

Exposes a single unauthenticated endpoint used by reverse proxies and
container orchestrators to determine whether the process is reachable.
The response is intentionally minimal — no version, no build info, no
deployment profile — to avoid disclosing fingerprintable details to
anonymous callers.
"""

from __future__ import annotations

from fastapi import APIRouter

from sec_generative_search.api.schemas import HealthResponse

__all__ = ["router"]


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Liveness probe",
)
async def health() -> HealthResponse:
    return HealthResponse()
