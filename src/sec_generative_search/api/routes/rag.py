"""RAG query-understanding and generation routes.

Two endpoints live here:

        - ``POST /api/rag/plan`` wraps
            :func:`sec_generative_search.rag.query_understanding.understand_query`
            behind the request-scoped resolver chain and returns an editable
            :class:`QueryPlanSchema`.
        - ``POST /api/rag/query`` runs
            :meth:`RAGOrchestrator.generate` against an approved plan and
            returns the generated answer with citations and traceability.

Why two routes (plan vs. generate):

        - The UI lets the user inspect and edit the plan before the
            orchestrator runs. Splitting plan / generate makes that edit point
            a hard contract: generation accepts a plan, never a raw query, so
            the human-in-the-loop step is enforced by the API shape rather
            than convention.
        - Generation is high-latency; understanding is short and cheap. A
            separate route lets the UI render chips while the user reviews
            filters, and only spend the generation budget on caller approval.

Security contract (shared by both routes):

        - Query / plan travels in the body, never the URL. Audit-log carries
            metadata only; the raw query never lands in any non-redacted log
            line.
        - Both routes are read-tier (``Depends(verify_api_key)``). Generation
            is a read-only side effect on the upstream LLM (no server-state
            mutation) and every authenticated tenant must be able to run it.
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
            query, the plan body, or the provider key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request

from sec_generative_search.api.dependencies import (
    get_retrieval_service,
    request_scoped_resolver,
    verify_api_key,
)
from sec_generative_search.api.errors import http_error
from sec_generative_search.api.schemas import (
    CitationSchema,
    QueryPlanSchema,
    RagPlanRequest,
    RagPlanResponse,
    RagQueryRequest,
    RagQueryResponse,
    TokenUsageSchema,
)
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    GenerationError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.types import Citation, GenerationResult
from sec_generative_search.providers.factory import build_llm_provider
from sec_generative_search.providers.registry import (
    ProviderRegistry,
    ProviderSurface,
)
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.orchestrator import RAGOrchestrator
from sec_generative_search.rag.query_understanding import (
    QueryPlan,
    understand_query,
)

if TYPE_CHECKING:
    from sec_generative_search.providers.base import BaseLLMProvider
    from sec_generative_search.search import RetrievalService

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


# ---------------------------------------------------------------------------
# POST /api/rag/query — non-streaming generation
# ---------------------------------------------------------------------------


def _plan_from_schema(schema: QueryPlanSchema) -> QueryPlan:
    """Lift the wire schema back into the dataclass.

    Inverse of :func:`_plan_to_schema` — explicit field-by-field copy
    (no ``**dict()`` splat) so a future field on either side is a hard
    compile-time / runtime mismatch rather than a silent leak in either
    direction.  ``date_range`` becomes a tuple again; the wire schema
    bounds the list shape so we don't need to defend against an
    arbitrary length here.

    The orchestrator and citation extractor downstream re-validate every
    field they care about; this helper trusts the Pydantic shape and
    keeps the lift minimal.
    """
    date_range: tuple[str, str] | None
    if schema.date_range is None:
        date_range = None
    else:
        # The Pydantic schema does not bound the list length directly;
        # we defensively check here and reject any other shape because
        # ``QueryPlan.date_range`` is typed as a 2-tuple.
        if len(schema.date_range) != 2:
            raise http_error(
                status_code=400,
                error="invalid_plan",
                message="plan.date_range must contain exactly two ISO dates.",
                hint="Send [start, end] (YYYY-MM-DD) or null.",
            )
        date_range = (schema.date_range[0], schema.date_range[1])
    return QueryPlan(
        raw_query=schema.raw_query,
        detected_language=schema.detected_language,
        query_en=schema.query_en,
        tickers=list(schema.tickers),
        form_types=list(schema.form_types),
        date_range=date_range,
        intent=schema.intent,
        suggested_answer_mode=AnswerMode.from_string(
            schema.suggested_answer_mode,
            default=AnswerMode.CONCISE,
        ),
    )


def _citation_to_schema(citation: Citation) -> CitationSchema:
    """Lift one :class:`Citation` onto the wire schema.

    Flattens :attr:`Citation.filing_id` (a frozen :class:`FilingIdentifier`)
    into the four scalar fields the wire surface carries — keeping the
    JSON flat and saving the UI a nested traversal.  ``filing_date`` is a
    :class:`datetime.date` on the dataclass and renders as ISO on the
    wire.
    """
    filing_id = citation.filing_id
    return CitationSchema(
        chunk_id=citation.chunk_id,
        ticker=filing_id.ticker,
        form_type=filing_id.form_type,
        filing_date=filing_id.filing_date.isoformat(),
        accession_number=filing_id.accession_number,
        section_path=citation.section_path,
        text_span=citation.text_span,
        similarity=citation.similarity,
        display_index=citation.display_index,
    )


def _result_to_response(
    result: GenerationResult,
    *,
    refused: bool,
) -> RagQueryResponse:
    """Lift the orchestrator's :class:`GenerationResult` onto the wire.

    Explicit allow-list lift (no ``**asdict()`` splat) so a future field
    addition on :class:`GenerationResult` does not silently leak onto
    the API.  ``retrieved_chunks`` is intentionally dropped — citations
    are the user-visible audit trail; operators wanting the full
    candidate set use ``POST /api/search`` against the same retrieval
    primitive.
    """
    usage = result.token_usage
    return RagQueryResponse(
        answer=result.answer,
        citations=[_citation_to_schema(c) for c in result.citations],
        provider=result.provider,
        model=result.model,
        prompt_version=result.prompt_version,
        token_usage=TokenUsageSchema(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        ),
        latency_seconds=result.latency_seconds,
        streamed=result.streamed,
        refused=refused,
    )


def _build_llm_for_request(
    request: Request,
    provider_name: str,
) -> BaseLLMProvider:
    """Construct the user's LLM via the request-scoped resolver chain.

    Centralised here so the same error-mapping table covers both the
    plan and query routes — a divergence between the two surfaces would
    confuse callers who hit ``provider_key_required`` from one but
    ``unknown_provider`` from the other for the same misconfiguration.
    """
    resolver = request_scoped_resolver(request)
    try:
        return build_llm_provider(provider_name, api_key_resolver=resolver)
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
        raise http_error(
            status_code=400,
            error="provider_unavailable",
            message=str(exc),
            hint="Install the provider's optional extras and restart the API.",
        ) from None


def _resolve_provider_and_model_query(body: RagQueryRequest) -> tuple[str, str]:
    """Pick provider / model for the generation route.

    Distinct from :func:`_resolve_provider_and_model` only in the body
    type it accepts — a single helper would pull both schemas into the
    same callsite and obscure which route is being served.
    """
    settings = get_settings()
    provider = body.provider or settings.llm.default_provider
    model = body.model or (settings.llm.default_model or "")
    return provider, model


@router.post(
    "/query",
    response_model=RagQueryResponse,
    tags=["rag"],
    summary="Generate an answer for an approved QueryPlan (non-streaming)",
)
async def generate_answer(
    request: Request,
    body: RagQueryRequest,
    retrieval: RetrievalService = Depends(get_retrieval_service),
) -> RagQueryResponse:
    """Run :meth:`RAGOrchestrator.generate` against the supplied plan.

    The route is a thin adapter — every business rule (context budgeting,
    comparative fan-out, prompt assembly with the
    ``<UNTRUSTED_FILING_CONTEXT>`` delimiter contract, hybrid citation
    extraction) lives in :class:`RAGOrchestrator`.  The route's job is
    to (1) wire credentials through the resolver chain, (2) translate
    the wire schema to / from the dataclass surface, and (3) collapse
    the full provider exception ladder into the structured envelope.

    Why we accept a plan and not a raw query:

        - The UI's editable-chips review is the human-in-the-loop step
          that catches misparsed tickers, hallucinated date ranges, and
          mode misclassification *before* the generation budget is
          spent.  Accepting only a plan means a future client cannot
          accidentally bypass that review.
        - The plan carries ``query_en``; retrieval embeds the English
          rendering even when the original query is in another
          language.  A raw-query route would have to re-run
          query-understanding to get there, doubling the LLM round-trip
          on the user-facing path.
    """
    provider_name, model = _resolve_provider_and_model_query(body)

    # Reject unknown providers up front with a 400 — same contract as
    # the plan route.  The capability probe is O(1) and credential-free.
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
        raise http_error(
            status_code=400,
            error="unknown_model",
            message=str(exc),
            hint="Omit the model field to use the provider default.",
        ) from None

    plan = _plan_from_schema(body.plan)
    llm = _build_llm_for_request(request, provider_name)

    orchestrator = RAGOrchestrator(retrieval=retrieval, llm=llm)

    # ``body.mode`` is already pattern-validated at the schema layer
    # (one of the four AnswerMode values).  ``from_string`` is still
    # the canonical lift — handles the explicit ``None`` case (orchestrator
    # falls back to ``plan.suggested_answer_mode``) and stays consistent
    # with the orchestrator's own internal lifts.
    effective_mode = (
        AnswerMode.from_string(body.mode, default=plan.suggested_answer_mode)
        if body.mode is not None
        else None
    )

    try:
        result = orchestrator.generate(
            plan,
            mode=effective_mode,
            model=model or None,
            max_output_tokens=body.max_output_tokens,
            prefer_structured_output=capability.structured_output,
        )
    except ProviderAuthError as exc:
        raise http_error(
            status_code=401,
            error="provider_unauthorized",
            message="The upstream provider rejected the supplied API key.",
            hint="Verify or rotate the provider key (header / session / admin env).",
        ) from exc
    except (ProviderRateLimitError, ProviderTimeoutError) as exc:
        logger.warning(
            "rag query upstream transient: provider=%s kind=%s",
            provider_name,
            type(exc).__name__,
        )
        raise http_error(
            status_code=503,
            error="provider_unavailable",
            message="The upstream provider is rate-limited or timed out.",
            hint="Retry after a short backoff; do not rotate the key.",
        ) from exc
    except ProviderError as exc:
        logger.error(
            "rag query provider error: provider=%s kind=%s",
            provider_name,
            type(exc).__name__,
        )
        raise http_error(
            status_code=502,
            error="provider_error",
            message="The upstream provider returned an error during generation.",
            details={"provider": provider_name, "kind": type(exc).__name__},
            hint="Inspect the audit log; do not rotate the key on a non-auth error.",
        ) from exc
    except GenerationError as exc:
        # Citation parser / orchestrator hit an unrecoverable state
        # (e.g. malformed structured-output payload that even the inline
        # fallback cannot rescue).  Surface as 502 — the upstream
        # output, not the caller's input, is at fault.
        logger.error("rag query generation error: %s", type(exc).__name__)
        raise http_error(
            status_code=502,
            error="generation_error",
            message="The orchestrator could not assemble a valid answer.",
            hint="Retry the request; if the failure persists, switch model or provider.",
        ) from exc

    refused = not result.retrieved_chunks and not result.citations
    audit_mode = (
        effective_mode.value if effective_mode is not None else plan.suggested_answer_mode.value
    )

    audit_log(
        "rag_query",
        client_ip=_client_ip(request),
        endpoint="POST /api/rag/query",
        detail=(
            f"provider={provider_name} model={model or '<provider default>'} "
            f"lang={plan.detected_language} tickers={len(plan.tickers)} "
            f"forms={len(plan.form_types)} mode={audit_mode} "
            f"prompt_version={result.prompt_version} "
            f"chunks={len(result.retrieved_chunks)} citations={len(result.citations)} "
            f"refused={refused}"
        ),
    )

    return _result_to_response(result, refused=refused)
