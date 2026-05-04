"""Authenticated status route.

Returns a small operator-facing snapshot of the deployment. Fields are
deliberately limited: version, embedding identity, filings count,
deployment profile, the encrypted-credential-store toggle, and whether
the caller is admin. No paths, no host names, no environment-variable
values.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from sec_generative_search import __version__
from sec_generative_search.api.dependencies import (
    get_registry,
    is_admin_request,
    verify_api_key,
)
from sec_generative_search.api.schemas import StatusResponse
from sec_generative_search.config.settings import get_settings
from sec_generative_search.database import MetadataRegistry

__all__ = ["router"]


router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get(
    "/",
    response_model=StatusResponse,
    tags=["status"],
    summary="Deployment status snapshot",
)
async def status(
    request: Request,
    registry: MetadataRegistry = Depends(get_registry),
) -> StatusResponse:
    settings = get_settings()
    stats = registry.get_statistics()
    return StatusResponse(
        version=__version__,
        deployment_profile=settings.database.deployment_profile,
        embedding_provider=settings.embedding.provider,
        embedding_model=settings.embedding.model_name,
        storage_filings=stats.filing_count,
        is_admin=is_admin_request(request),
        persist_provider_credentials=settings.database.persist_provider_credentials,
    )
