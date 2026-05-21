"""Provider catalogue + key-validation routes.

Two endpoints, both read-tier (``Depends(verify_api_key)`` only — never
admin-gated):

- ``GET /api/providers/`` — lifts the curated :class:`ProviderRegistry`
  tuple onto the wire as ``(name, surface, supports_upstream_routing)``.
  No keys, no masked tails, no model catalogue; the schema is
  intentionally narrow so a future field addition is a deliberate
  security-reviewed change rather than an accidental drift.
- ``POST /api/providers/validate`` — wraps the audit-logged
  :func:`validate_credential` seam so the SDK round-trip that confirms a
  key is a thin route adapter, not a re-implementation.

Security contract (validate):

    - The body is parsed by :class:`ProviderValidateRequest` which
      bounds ``api_key`` length and restricts ``provider`` to the
      lower-case slug shape :class:`ProviderRegistry` advertises.  The
      key never appears in the response body or any non-audit log.
        - The route is rate-limited per-IP **and** per-``session_id`` —
            both sliding windows must allow the request (see
            :class:`RateLimitMiddleware`). Without the per-session window a
            single tenant could brute-force a stolen key list across many
            origins behind shared NAT.
    - :class:`ProviderAuthError` collapses to ``valid=False`` (200 OK).
      Every other :class:`ProviderError` propagates as a 502 / 503
      envelope so the caller does not interpret a network blip as
      a "wrong key" verdict and rotate a working credential.
    - The route is API-key gated (read-tier) when ``API_KEY`` is set;
      it is NOT admin-gated — validation is a read-only side effect on
      the upstream provider and tenants must be able to verify their
      own keys.

Security contract (list):

    - The handler reads the registry's static class attributes only —
      never instantiates a provider, never touches the network, never
      reads a credential.  Adding a vendor with a leaked
      ``MODEL_CATALOGUE`` shape cannot leak through this route because
      the explicit allow-list lift in the schema drops every attribute
      the registry might add in the future.
    - The route returns only providers whose optional extras are
      importable (``ProviderRegistry.all_entries()`` default).  An
      un-installed provider is indistinguishable from "not registered"
      on this surface — surfacing it would invite the UI to attempt a
      validation that would 400 at the very next step.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from sec_generative_search.api.dependencies import verify_api_key
from sec_generative_search.api.errors import http_error
from sec_generative_search.api.schemas import (
    ProviderInfoSchema,
    ProviderListResponse,
    ProviderValidateRequest,
    ProviderValidateResponse,
)
from sec_generative_search.core.credentials import validate_credential
from sec_generative_search.core.exceptions import (
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.logging import get_logger
from sec_generative_search.providers.registry import (
    ProviderRegistry,
    ProviderSurface,
)

__all__ = ["router"]


logger = get_logger(__name__)


router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get(
    "/",
    response_model=ProviderListResponse,
    tags=["providers"],
    summary="List registered providers",
)
async def list_providers() -> ProviderListResponse:
    """Return the curated provider catalogue.

    Lifts every available entry from :class:`ProviderRegistry` onto the
    wire as ``(name, surface, supports_upstream_routing)``.  The wire
    schema is an explicit allow-list — never ``**asdict()`` of the
    :class:`ProviderEntry` dataclass — so future additions to the
    registry shape do not leak through this surface.

    Entries gated behind an optional extra
    (``LocalEmbeddingProvider`` → ``sentence_transformers``) are filtered
    out when the extra is not importable.  That keeps the UI catalogue
    aligned with what the API can actually serve; an un-installed entry
    would 400 at validation time anyway.
    """
    entries = ProviderRegistry.all_entries()
    items = [
        ProviderInfoSchema(
            name=entry.name,
            surface=entry.surface.value,
            supports_upstream_routing=entry.supports_upstream_routing,
        )
        for entry in entries
    ]
    return ProviderListResponse(providers=items, total=len(items))


@router.post(
    "/validate",
    response_model=ProviderValidateResponse,
    tags=["providers"],
    summary="Validate a provider API key",
)
async def validate_provider(
    body: ProviderValidateRequest,
) -> ProviderValidateResponse:
    """Round-trip a candidate key against the upstream provider.

    Returns ``valid=True`` when the provider accepts the key.  Returns
    ``valid=False`` when the provider explicitly rejects it.  Any
    other failure (rate-limit, timeout, transport, content-filter)
    surfaces as a 502 / 503 envelope so the caller can distinguish
    a verdict from a transient failure.
    """
    surface = ProviderSurface(body.surface)

    # Reject unknown providers up front with a 400 — sending the key to
    # ``validate_credential`` would otherwise raise an obscure
    # :class:`KeyError` from the registry on the same call site.
    try:
        ProviderRegistry.get_entry(body.provider, surface)
    except KeyError as exc:
        raise http_error(
            status_code=400,
            error="unknown_provider",
            message=str(exc),
            hint="Use ProviderRegistry.list_providers() to see registered slugs.",
        ) from None

    try:
        ok = validate_credential(
            body.provider,
            surface=surface,
            api_key=body.api_key,
            model=body.model,
        )
    except (ProviderRateLimitError, ProviderTimeoutError) as exc:
        # Transient — caller should retry, not rotate the key. 503
        # rather than 502 because the key has not been refused; the
        # upstream provider is just unavailable to render a verdict.
        raise http_error(
            status_code=503,
            error="provider_unavailable",
            message=(
                "The upstream provider could not render a verdict on the key "
                "(rate-limited or timed out)."
            ),
            details={"provider": body.provider, "surface": body.surface},
            hint="Retry after a short backoff; do not rotate the key.",
        ) from exc
    except ProviderError as exc:
        # Network / transport / content-filter / unknown. Same logic:
        # we cannot honestly say "wrong key", so do not pretend.
        raise http_error(
            status_code=502,
            error="provider_error",
            message=("The upstream provider returned an error while validating the key."),
            details={
                "provider": body.provider,
                "surface": body.surface,
                "kind": type(exc).__name__,
            },
            hint="Inspect the audit log; do not rotate the key on a non-auth error.",
        ) from exc

    return ProviderValidateResponse(
        valid=ok,
        provider=body.provider,
        surface=body.surface,
    )
