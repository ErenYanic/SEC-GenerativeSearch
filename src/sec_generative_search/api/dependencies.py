"""FastAPI dependency providers.

All dependencies read pre-initialised singletons from
``request.app.state`` (set during the lifespan startup in
:mod:`sec_generative_search.api.app`). This keeps the API on a single
ChromaDB connection, a single SQLite registry, and a single embedding
model across the process.

Security contract:

        - ``API_KEY`` and ``API_ADMIN_KEY`` comparisons go through
            :func:`hmac.compare_digest` — never ``==``.
        - ``session_id`` extracted from the cookie is validated against the
            mint policy: a non-empty URL-safe base64 string of at least 32
            characters before it is accepted as a key into the session store.
            Browser-supplied or forged shapes round-trip as "no session".
        - ``X-Provider-Key-{provider}`` header values are parsed into a
            per-request map and consulted as the outermost tier of the
            resolver chain. The header-name suffix is shape-checked against
            the lower-case slug pattern; anything else is dropped so a
            malformed header cannot key a credential lookup. Headers are
            already redacted at the access-log layer
            (:data:`api.access_log.REDACTED_HEADER_PREFIXES`).
"""

from __future__ import annotations

import hmac
import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from fastapi import Depends, Request, Security
from fastapi.params import Depends as DependsParam
from fastapi.security import APIKeyHeader

from sec_generative_search.api.errors import http_error
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.credentials import (
    ApiKeyResolver,
    CredentialStore,
    chain_resolvers,
    encrypted_user_resolver,
    session_resolver,
)
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.security import mask_secret
from sec_generative_search.providers.factory import default_api_key_resolver

if TYPE_CHECKING:
    from sec_generative_search.database import (
        ChromaDBClient,
        FilingStore,
        MetadataRegistry,
    )
    from sec_generative_search.providers.base import BaseEmbeddingProvider
    from sec_generative_search.search import RetrievalService

__all__ = [
    "ADMIN_USER_ID",
    "PROVIDER_KEY_HEADER_PREFIX",
    "SESSION_COOKIE_NAME",
    "admin_route_dependencies",
    "extract_session_id",
    "get_chroma",
    "get_embedder",
    "get_encrypted_credential_store",
    "get_filing_store",
    "get_registry",
    "get_retrieval_service",
    "get_session_id",
    "get_session_store",
    "header_resolver",
    "is_admin_request",
    "parse_provider_key_headers",
    "request_scoped_resolver",
    "verify_admin_key",
    "verify_api_key",
]


logger = get_logger(__name__)


# Cookie name for the server-minted ``session_id``.
SESSION_COOKIE_NAME = "sec_rag_session"


# Until multi-user auth lands, the only writer/reader of the encrypted
# credential store is "the admin user" — a stable opaque string with
# no PII. Multi-user identifiers arrive with auth.
ADMIN_USER_ID = "__admin__"


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def _secrets_match(provided: str | None, expected: str) -> bool:
    """Constant-time secret comparison.  ``None`` always returns False."""
    if provided is None:
        return False
    return hmac.compare_digest(provided, expected)


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> None:
    """Validate the ``X-API-Key`` header when authentication is enabled.

    When ``API_KEY`` is unset (Scenario A — local dev) authentication is
    disabled and every caller passes through.  When it is set, the
    header MUST match (constant-time comparison) or the request is
    rejected with 401.

    Denials emit a ``SECURITY_AUDIT:`` line carrying the client IP and
    endpoint so unauthorised probing is greppable in operator logs. The
    key value itself is never logged.
    """
    expected = get_settings().api.key
    if expected is None:
        return
    if not _secrets_match(api_key, expected):
        client_ip = request.client.host if request.client else "unknown"
        audit_log(
            "api_key_denied",
            detail=(f"client_ip={client_ip} endpoint={request.method} {request.url.path}"),
        )
        raise http_error(
            status_code=401,
            error="unauthorised",
            message="Invalid or missing API key.",
            hint="Provide a valid key via the X-API-Key header.",
        )


async def verify_admin_key(
    request: Request,
    admin_key: str | None = Security(_admin_key_header),
) -> None:
    """Validate the ``X-Admin-Key`` header for destructive operations.

    Two-tier access control:

    - ``API_ADMIN_KEY`` unset → unrestricted (Scenario A).
    - ``API_ADMIN_KEY`` set → caller MUST supply a matching
      ``X-Admin-Key`` header; mismatches log an audit-log entry and
      return 403.

    Wire this with ``Depends(verify_admin_key)`` on routes that mutate
    or destroy filings, drop credentials, or unload provider caches.
    """
    expected = get_settings().api.admin_key
    if expected is None:
        return
    if not _secrets_match(admin_key, expected):
        client_ip = request.client.host if request.client else "unknown"
        audit_log(
            "admin_denied",
            detail=(f"client_ip={client_ip} endpoint={request.method} {request.url.path}"),
        )
        raise http_error(
            status_code=403,
            error="admin_required",
            message="Admin access is required for this operation.",
            hint="Provide a valid admin key via the X-Admin-Key header.",
        )


def is_admin_request(request: Request) -> bool:
    """Return whether the current request carries a valid admin key.

    Unlike :func:`verify_admin_key` this does *not* raise — used by the
    status endpoint to surface ``is_admin`` without blocking non-admin
    callers, and by the resolver factory to gate the encrypted-user
    tier of the chain.
    """
    expected = get_settings().api.admin_key
    if expected is None:
        # No admin key configured — every caller is effectively admin.
        return True
    return _secrets_match(request.headers.get("X-Admin-Key"), expected)


def admin_route_dependencies() -> list[DependsParam]:
    """Canonical dependency list for routes that mutate or destroy state.

    A destructive route must present both headers. Wiring both
    dependencies on the route - rather than only ``verify_admin_key`` -
    keeps the read tier load-bearing: when ``API_KEY`` is configured but
    ``API_ADMIN_KEY`` is not, a non-admin caller still cannot reach the
    destructive route because the read check rejects first; when both
    are unset, every caller passes through.

    Usage::

        router = APIRouter(dependencies=admin_route_dependencies())
        # or per-route: ``@router.delete("/x", dependencies=admin_route_dependencies())``

    Returned as a fresh list on each call so a router that mutates the
    list (FastAPI does not, but downstream code might) cannot poison the
    canonical wiring for other routers.
    """
    return [Depends(verify_api_key), Depends(verify_admin_key)]


# ---------------------------------------------------------------------------
# session_id extraction
# ---------------------------------------------------------------------------


# ``secrets.token_urlsafe(32)`` produces a 43-character base64-url string.
# We accept anything within a generous bound so the helper does not
# accidentally reject legitimate values produced by future mints with
# different lengths, but we DO reject obviously forged shapes (empty,
# whitespace-only, longer than any reasonable token, control characters).
_MIN_SESSION_ID_LEN = 32
_MAX_SESSION_ID_LEN = 256
_VALID_SESSION_ID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def extract_session_id(request: Request) -> str | None:
    """Return the validated ``session_id`` cookie value, or ``None``.

    Returns ``None`` (rather than raising) for absent or syntactically
    invalid cookies so the resolver chain can fall through to the next
    tier.  A *forged* cookie shape is treated identically to *no*
    cookie — neither resolves to a credential.

    The shape check is *not* a substitute for server-side minting.  It
    only rejects values that obviously did not come out of
    ``secrets.token_urlsafe(32)``; a sufficiently motivated attacker
    who guesses a real token still has every other layer (TTL, audit
    log, store contents) to bypass.
    """
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    if raw is None:
        return None
    if not (_MIN_SESSION_ID_LEN <= len(raw) <= _MAX_SESSION_ID_LEN):
        return None
    if any(ch not in _VALID_SESSION_ID_CHARS for ch in raw):
        return None
    return raw


def get_session_id(request: Request) -> str | None:
    """``Depends()``-friendly wrapper around :func:`extract_session_id`."""
    return extract_session_id(request)


# ---------------------------------------------------------------------------
# Singleton state providers
# ---------------------------------------------------------------------------


def get_registry(request: Request) -> MetadataRegistry:
    return request.app.state.registry


def get_chroma(request: Request) -> ChromaDBClient:
    return request.app.state.chroma


def get_filing_store(request: Request) -> FilingStore:
    return request.app.state.filing_store


def get_embedder(request: Request) -> BaseEmbeddingProvider:
    return request.app.state.embedder


def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval_service


def get_session_store(request: Request) -> CredentialStore:
    return request.app.state.session_store


def get_encrypted_credential_store(request: Request) -> CredentialStore | None:
    """Return the encrypted store when configured; ``None`` otherwise.

    The encrypted tier is a profile-driven optional dependency
    (``DB_PERSIST_PROVIDER_CREDENTIALS``); the resolver factory must
    treat its absence as a no-op rather than a hard error.
    """
    return getattr(request.app.state, "encrypted_credential_store", None)


# ---------------------------------------------------------------------------
# Per-request provider-key header parsing
# ---------------------------------------------------------------------------


# All ``X-Provider-Key-*`` headers are masked at the access-log layer
# (see ``api.access_log.REDACTED_HEADER_PREFIXES``); the prefix below is
# the canonical lowercase form the parser keys on.
PROVIDER_KEY_HEADER_PREFIX = "x-provider-key-"


# Provider names land in URLs / config / logs.  Restrict the alphabet to
# the same lower-case slug shape ``ProviderRegistry._ENTRIES`` advertises;
# this prevents header-injected weirdness (case mismatches, unicode
# look-alikes, accidental newlines) from being treated as a "valid"
# provider.  A bounded length is defence-in-depth against pathological
# header floods.
_PROVIDER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def parse_provider_key_headers(
    headers: Mapping[str, str],
) -> dict[str, str]:
    """Extract per-provider keys from ``X-Provider-Key-{provider}`` headers.

    Returns a ``{provider: api_key}`` mapping suitable for
    :func:`header_resolver`.  Headers whose suffix fails the
    :data:`_PROVIDER_NAME_RE` shape check or whose value is empty are
    silently ignored — the caller never receives a partial / unsafe
    parse.  No raw value ever appears in this function's logs (the
    header-redaction layer covers the access log; the resolver itself is
    the single audit-logged seam).
    """
    parsed: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if not lowered.startswith(PROVIDER_KEY_HEADER_PREFIX):
            continue
        provider = lowered[len(PROVIDER_KEY_HEADER_PREFIX) :]
        if not _PROVIDER_NAME_RE.match(provider):
            continue
        if not value:
            continue
        parsed[provider] = value
    return parsed


def header_resolver(headers_map: Mapping[str, str]) -> ApiKeyResolver:
    """Adapt a parsed header map to the ``ApiKeyResolver`` shape.

    Mirrors :func:`session_resolver` / :func:`encrypted_user_resolver`:
    on each non-``None`` hit emit a single ``credential_resolved`` audit
    line carrying only ``mask_secret``-tailed values, so a credential's
    actually-resolved-from-header lineage is greppable in operator logs.

    The map is consumed by reference; callers MUST treat it as
    short-lived (per-request) and never let it escape the request
    scope — keys live as plain strings in the dict and the dict's
    lifetime is the credential's exposure window.
    """

    def resolver(provider: str) -> str | None:
        value = headers_map.get(provider)
        if value is not None:
            audit_log(
                "credential_resolved",
                detail=(f"resolver=header provider={provider} key_tail={mask_secret(value)}"),
            )
        return value

    return resolver


# ---------------------------------------------------------------------------
# Per-request credential resolver chain
# ---------------------------------------------------------------------------


def request_scoped_resolver(
    request: Request,
    *,
    extra_resolvers: Callable[[str], str | None] | None = None,
) -> ApiKeyResolver:
    """Build a resolver chain for the current request.

    Composition (first hit wins):

    1. **Per-request headers** (``X-Provider-Key-{provider}``) —
       short-lived, opt-in, scoped to one request. The parser runs up
       front so the chain stays O(1) per lookup.
    2. ``extra_resolvers`` — caller-supplied front of the chain. Used
       by tests and by future surfaces that need a tier in front of the
       header layer; production routes pass nothing.
    3. **Session store** keyed by the validated ``session_id`` cookie,
       when present.
    4. **Encrypted user store** keyed by ``ADMIN_USER_ID`` when both
       (a) the store exists on ``app.state`` and (b) the request is
       admin-authenticated. Until multi-user auth lands the encrypted
       tier is admin-only by design.
    5. ``default_api_key_resolver`` — process-environment fallback.

    Returns a callable of the
    :data:`~sec_generative_search.providers.factory.ApiKeyResolver`
    shape that plugs straight into ``build_embedder`` /
    ``build_llm_provider``.
    """
    chain: list[ApiKeyResolver] = []

    header_keys = parse_provider_key_headers(request.headers)
    if header_keys:
        chain.append(header_resolver(header_keys))

    if extra_resolvers is not None:
        chain.append(extra_resolvers)

    session_id = extract_session_id(request)
    if session_id is not None:
        store = getattr(request.app.state, "session_store", None)
        if store is not None:
            chain.append(session_resolver(store, session_id))

    encrypted = getattr(request.app.state, "encrypted_credential_store", None)
    if encrypted is not None and is_admin_request(request):
        chain.append(encrypted_user_resolver(encrypted, ADMIN_USER_ID))

    chain.append(default_api_key_resolver)

    return chain_resolvers(*chain)
