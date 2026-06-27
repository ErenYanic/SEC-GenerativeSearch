"""Admin-gated model-catalogue refresh trigger.

Single endpoint: ``POST /api/providers/catalogue/refresh``.  It is the HTTP
trigger for the opt-in refresh seam
(:mod:`sec_generative_search.providers.refresh`) — the *only* network touch
in ``providers/`` and deliberately kept out of the request / credential path.
A CLI command and an external scheduler (cron / systemd timer / Cloud
Scheduler hitting this route) are the other two triggers; there is no
in-process scheduler.

Security contract:

    - **Admin-gated (rule A).** The router wires
      ``admin_route_dependencies()`` = ``[verify_api_key, verify_admin_key]``
      in that order, so an admin-key-only request surfaces ``401`` (read tier
      first), never ``403``.
    - **Content-free response (rule L).** The success body is an explicit
      allow-list lift of :class:`CatalogueRefreshReport` — source key, public
      metadata URL, and aggregate counts only.  The report's ``overlay_path``
      is **not** lifted (no filesystem path on the wire); no model slug, no
      per-token cost, and no credential ever reaches this surface.
    - **Fail-closed.** Every failure mode of the refresh collapses to a
      content-free :class:`CatalogueRefreshError`; on error the prior overlay
      and the bundled baseline keep serving untouched, and the route emits a
      ``502`` envelope whose message carries an exception *type*, never an
      upstream body.
    - **No event-loop block.** The handler is a plain ``def`` so Starlette
      runs the bounded (≤ :data:`~sec_generative_search.providers.refresh.FETCH_TIMEOUT_SECONDS`)
      network fetch in a worker thread rather than on the event loop.

On success the route calls :func:`reset_catalogue` so the worker that handled
the request serves the freshly-merged baseline-plus-overlay catalogue
immediately.  The overlay is durable on the data volume, so the rest of a
multi-worker fleet picks it up on its next start regardless.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from sec_generative_search.api.dependencies import admin_route_dependencies
from sec_generative_search.api.errors import http_error
from sec_generative_search.api.schemas import CatalogueRefreshResponse
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import CatalogueRefreshError
from sec_generative_search.core.logging import get_logger
from sec_generative_search.providers.catalogue import reset_catalogue
from sec_generative_search.providers.refresh import refresh_overlay

__all__ = ["router"]


logger = get_logger(__name__)


router = APIRouter(dependencies=admin_route_dependencies())


@router.post(
    "/catalogue/refresh",
    response_model=CatalogueRefreshResponse,
    tags=["providers"],
    summary="Refresh the model catalogue overlay (admin only)",
)
def refresh_catalogue() -> JSONResponse:
    """Fetch, validate, and write a model-catalogue overlay, then reload it.

    A plain ``def`` so the ≤15 s bounded fetch runs in a worker thread and
    never blocks the event loop.  Source, optional URL override, and overlay
    path all come from :class:`ProviderSettings`; the operator configures them
    via ``PROVIDER_CATALOGUE_REFRESH_SOURCE`` / ``_URL`` / ``_OVERLAY_PATH``.

    The handler takes no request input — there is nothing to reflect — and
    returns the content-free :class:`CatalogueRefreshReport` counts behind
    ``Cache-Control: no-store``.  Any failure surfaces as a ``502``
    ``catalogue_refresh_failed`` envelope; the active catalogue keeps serving
    the prior overlay + baseline.
    """
    provider_settings = get_settings().provider

    try:
        report = refresh_overlay(
            overlay_path=provider_settings.catalogue_overlay_path,
            source=provider_settings.catalogue_refresh_source,
            url=provider_settings.catalogue_refresh_url,
        )
    except CatalogueRefreshError as exc:
        # The exception message is content-free by construction (an exception
        # *type* or a bounded count, never an upstream body or slug), so it is
        # safe to relay verbatim as the envelope message.
        raise http_error(
            status_code=502,
            error="catalogue_refresh_failed",
            message=str(exc),
            hint=(
                "The refresh is fail-closed: the prior overlay and the bundled "
                "baseline keep serving.  Check the configured source / URL is "
                "reachable and returns the expected schema, then retry."
            ),
        ) from exc
    except OSError as exc:
        # Disk failure writing the overlay (permissions / full volume).  Map to
        # a generic 500 WITHOUT echoing ``str(exc)`` — an OS error message can
        # carry the configured filesystem path, which never belongs on the wire.
        logger.warning("Catalogue overlay write failed (%s).", type(exc).__name__)
        raise http_error(
            status_code=500,
            error="catalogue_write_failed",
            message="Could not write the catalogue overlay to the data volume.",
            hint="Verify the overlay path is writable and the volume has free space.",
        ) from exc

    # Drop the cached active catalogue so the next capability probe rebuilds
    # the merged baseline-plus-overlay from disk — making this refresh effective
    # in the handling worker without a restart.  Safe under concurrency: the
    # reassignment is atomic and the rebuild is fail-closed to the baseline.
    reset_catalogue()

    payload = CatalogueRefreshResponse(
        source=report.source,
        source_url=report.source_url,
        provider_count=report.provider_count,
        model_count=report.model_count,
    )
    return JSONResponse(
        content=payload.model_dump(mode="json"),
        headers={"Cache-Control": "no-store"},
    )
