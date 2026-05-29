"""Admin-gated metrics exposition.

Single endpoint: ``GET /api/metrics``. Renders the process-global
:class:`~sec_generative_search.core.metrics.Metrics` registry in the
OpenMetrics / Prometheus text exposition format so a B/C scraper can
pull ingestion / retrieval / generation latency, LLM token counters,
and provider-failure counters.

Security contract:

    - **Admin-gated.** Wired with :func:`admin_route_dependencies` —
      both ``X-API-Key`` and ``X-Admin-Key`` are required (read tier
      rejects first, so an admin-key-only probe surfaces ``401`` not
      ``403``). Observability is an operator surface, never public.
    - **Content-free body.** The exposition only ever contains the
      content-free, cardinality-bounded labels the metrics facade
      admits (provider / model / kind / pricing_tier / error_type).
      The route adds nothing of its own — no query echo, no path
      reflection.
    - **No caching.** ``Cache-Control: no-store`` so a shared cache /
      proxy never retains an operator-only snapshot.

When the optional ``[metrics]`` extra (``prometheus-client``) is not
installed the facade is inert and this route returns
``503 metrics_unavailable`` with an install hint rather than an empty
``200`` that a scraper would silently treat as "no samples".
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from sec_generative_search.api.dependencies import admin_route_dependencies
from sec_generative_search.api.errors import http_error
from sec_generative_search.core.metrics import get_metrics

__all__ = ["router"]


router = APIRouter(dependencies=admin_route_dependencies())


@router.get(
    "/",
    tags=["metrics"],
    summary="OpenMetrics exposition (admin only)",
    responses={
        200: {"content": {"text/plain": {}}, "description": "OpenMetrics exposition."},
        503: {"description": "Metrics extra not installed."},
    },
)
async def metrics() -> Response:
    """Render the in-process metrics registry in the exposition format.

    Returns the bytes verbatim from
    :meth:`~sec_generative_search.core.metrics.Metrics.render_latest`;
    the route never instantiates a provider, touches a credential, or
    makes a network call.
    """
    rendered = get_metrics().render_latest()
    if rendered is None:
        raise http_error(
            status_code=503,
            error="metrics_unavailable",
            message="Metrics collection is not enabled on this deployment.",
            hint="Install the optional extra: pip install '.[metrics]'.",
        )

    content_type, payload = rendered
    return Response(
        content=payload,
        media_type=content_type,
        headers={"Cache-Control": "no-store"},
    )
