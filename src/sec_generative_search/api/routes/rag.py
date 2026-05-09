"""RAG query-understanding route.

Single endpoint: ``POST /api/rag/plan``. Wraps
:func:`sec_generative_search.rag.query_understanding.understand_query`
behind the request-scoped resolver chain so the LLM call uses whichever
key the caller has provided (header → session → encrypted-user → admin
env).

Why a separate plan route exists:

        - The UI lets the user inspect and *edit* the plan before the
            orchestrator runs. Splitting plan / generate makes that edit point
            a hard contract: ``POST /api/rag/query`` and ``POST /api/rag/stream``
            accept a plan, never a raw query, so the human-in-the-loop step is
            enforced by the API shape rather than convention.
        - Generation is high-latency; understanding is short and cheap. A
            separate route lets the UI render chips while the user reviews
            filters, and only spend the generation budget on caller approval.

Security contract:

    - Query travels in the body, never the URL — same rule as
      :mod:`api.routes.search`.  Audit-log carries metadata only.
    - The route is read-tier (``Depends(verify_api_key)``).  Plan
      generation is a read-only side effect on the upstream LLM and
      every authenticated tenant must be able to run it.
    - Rate-limited under the ``rag`` category by
      :class:`RateLimitMiddleware._classify_path`.
    - LLM construction goes through
      :func:`request_scoped_resolver` → :func:`build_llm_provider`; a
      missing key is reported as 400 ``provider_key_required`` so the
      caller knows to register one rather than treating it as a server
      fault.
    - :class:`ProviderAuthError` → 401 ``provider_unauthorized``
      (caller's own key is invalid; not a server fault).
    - :class:`ProviderRateLimitError` / :class:`ProviderTimeoutError`
      → 503 ``provider_unavailable``.
    - Every other :class:`ProviderError` → 502 ``provider_error``.
    - The unified envelope's ``message`` / ``hint`` never echoes the raw
      query or the provider key.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from sec_generative_search.api.dependencies import (
    request_scoped_resolver,
    verify_api_key,
)
from sec_generative_search.api.errors import http_error
from sec_generative_search.api.schemas import (
    QueryPlanSchema,
    RagPlanRequest,
    RagPlanResponse,
)
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.providers.factory import build_llm_provider
from sec_generative_search.providers.registry import (
    ProviderRegistry,
    ProviderSurface,
)
from sec_generative_search.rag.query_understanding import (
    QueryPlan,
    understand_query,
)

__all__ = ["router"]


logger = get_logger(__name__)


router = APIRouter(dependencies=[Depends(verify_api_key)])


def _client_ip(request: Request) -> str:
    """Best-effort client IP for audit-log lines."""
    return request.client.host if request.client else "unknown"


def _resolve_provider_and_model(body: RagPlanRequest) -> tuple[str, str]:
    """Pick the provider / model for this call.

    Body fields override settings; settings defaults apply when both are
    omitted.  Returning a concrete model string (possibly empty) keeps
    the audit log uniform — the LLM provider's own default fills in the
    blank at call time when it stays empty.
    """
    settings = get_settings()
    provider = body.provider or settings.llm.default_provider
    model = body.model or (settings.llm.default_model or "")
    return provider, model


def _plan_to_schema(plan: QueryPlan) -> QueryPlanSchema:
    """Lift the dataclass plan onto the wire schema.

    Explicit field-by-field copy — no ``**asdict()`` splat — so a future
    field addition on :class:`QueryPlan` does not silently leak onto the
    API.  ``date_range`` is a tuple in the dataclass and a list on the
    wire; the conversion happens here.
    """
    return QueryPlanSchema(
        raw_query=plan.raw_query,
        detected_language=plan.detected_language,
        query_en=plan.query_en,
        tickers=list(plan.tickers),
        form_types=list(plan.form_types),
        date_range=list(plan.date_range) if plan.date_range is not None else None,
        intent=plan.intent,
        suggested_answer_mode=plan.suggested_answer_mode.value,
    )


@router.post(
    "/plan",
    response_model=RagPlanResponse,
    tags=["rag"],
    summary="Run query-understanding and return an editable plan",
)
async def plan_query(
    request: Request,
    body: RagPlanRequest,
) -> RagPlanResponse:
    """Run :func:`understand_query` and return the resulting plan.

    The route is a thin adapter — every business rule (multilingual
    handling, JSON-schema vs free-form fallback, parse-failure → minimal
    plan) lives in
    :func:`sec_generative_search.rag.query_understanding.understand_query`.
    Keeping the route this thin makes the resolver-chain seam explicit:
    a future change to credential resolution does not need to touch
    routes.
    """
    provider_name, model = _resolve_provider_and_model(body)

    # Reject unknown providers up front with a 400 — building the LLM
    # otherwise raises an obscure ``KeyError`` from the registry on the
    # same call site.  The capability probe is O(1) and credential-free.
    try:
        capability = ProviderRegistry.get_capability(
            provider_name,
            ProviderSurface.LLM,
            model=model or None,
        )
    except KeyError as exc:
        raise http_error(
            status_code=400,
            error="unknown_provider",
            message=str(exc),
            hint="Use ProviderRegistry.list_providers(LLM) to see registered slugs.",
        ) from None
    except ValueError as exc:
        # ``get_capability`` raises ValueError only for an unknown
        # *embedding* slug; reaching here on the LLM surface would be a
        # registry bug, but we surface 400 rather than 500 so a future
        # registry edit fails caller-visibly.
        raise http_error(
            status_code=400,
            error="unknown_model",
            message=str(exc),
            hint="Omit the model field to use the provider default.",
        ) from None

    # Build the LLM via the request-scoped resolver chain (header →
    # session → encrypted-user → admin-env).  ``ConfigurationError``
    # means no key resolved through any tier — surface as 400 so the
    # caller registers one rather than treating the absence as a server
    # fault.
    resolver = request_scoped_resolver(request)
    try:
        llm = build_llm_provider(provider_name, api_key_resolver=resolver)
    except ConfigurationError as exc:
        raise http_error(
            status_code=400,
            error="provider_key_required",
            message=f"No API key resolved for provider '{provider_name}'.",
            hint=(
                "Supply the key via the X-Provider-Key-{provider} header, "
                "register it in the active session, or configure the "
                "admin-env fallback for the named provider."
            ),
        ) from exc
    except KeyError as exc:
        # Optional-extras gating in ``ProviderRegistry.get_class``.
        raise http_error(
            status_code=400,
            error="provider_unavailable",
            message=str(exc),
            hint="Install the provider's optional extras and restart the API.",
        ) from None

    try:
        plan = understand_query(
            body.query,
            llm=llm,
            model=model,
            structured_output_supported=capability.structured_output,
        )
    except ProviderAuthError as exc:
        # The caller's own key was rejected by the upstream provider —
        # 401 so the UI can prompt them to rotate their key.  Distinct
        # from the route-level 401 ``unauthorised`` (which signals an
        # invalid ``X-API-Key``).
        raise http_error(
            status_code=401,
            error="provider_unauthorized",
            message=("The upstream provider rejected the supplied API key."),
            hint=("Verify or rotate the provider key (header / session / admin env)."),
        ) from exc
    except (ProviderRateLimitError, ProviderTimeoutError) as exc:
        # Transient — same redaction logic as the validate route.
        logger.warning(
            "rag plan upstream transient: provider=%s kind=%s",
            provider_name,
            type(exc).__name__,
        )
        raise http_error(
            status_code=503,
            error="provider_unavailable",
            message=("The upstream provider is rate-limited or timed out."),
            hint="Retry after a short backoff; do not rotate the key.",
        ) from exc
    except ProviderError as exc:
        logger.error(
            "rag plan provider error: provider=%s kind=%s",
            provider_name,
            type(exc).__name__,
        )
        raise http_error(
            status_code=502,
            error="provider_error",
            message=("The upstream provider returned an error during query understanding."),
            details={"provider": provider_name, "kind": type(exc).__name__},
            hint=("Inspect the audit log; do not rotate the key on a non-auth error."),
        ) from exc

    # Audit-log MUST NOT carry the raw query — the plan IS the query
    # reformulated, so the response body has it; logs are a different
    # surface and stay metadata-only.  The query is logged via
    # ``redact_for_log`` only at debug level (inside ``understand_query``).
    audit_log(
        "rag_plan",
        client_ip=_client_ip(request),
        endpoint="POST /api/rag/plan",
        detail=(
            f"provider={provider_name} model={model or '<provider default>'} "
            f"lang={plan.detected_language} tickers={len(plan.tickers)} "
            f"forms={len(plan.form_types)} mode={plan.suggested_answer_mode.value}"
        ),
    )

    return RagPlanResponse(
        plan=_plan_to_schema(plan),
        provider=provider_name,
        model=model,
    )
