"""FastAPI dependency providers.

All dependencies read pre-initialised singletons from
``request.app.state`` (set during the lifespan startup in
:mod:`sec_generative_search.api.app`).  This guarantees route handlers
share a single ChromaDB connection, a single SQLite registry, and a
single embedding model across the process.

Phase-10 split:

    - 10A surface here: API/admin authentication, state providers for
      the singletons currently wired in lifespan, ``session_id``
      extraction from the HTTP-only cookie, and the
      :func:`request_scoped_resolver` factory that builds a Phase-9
      resolver chain per request (header → session → encrypted-user →
      admin-env).
    - 10B will extend this module with EDGAR identity resolution,
      provider-key header parsing, and helpers around ``TaskManager``
      ownership.

Security contract:

    - ``API_KEY`` and ``API_ADMIN_KEY`` comparisons go through
      :func:`hmac.compare_digest` — never ``==``.
    - ``session_id`` extracted from the cookie is validated against the
      mint policy: a non-empty URL-safe base64 string of at least 32
      characters before it is accepted as a key into the session store.
      Browser-supplied / forged shapes round-trip as "no session".
    - Per-provider ``X-Provider-Key-{name}`` headers ARE NOT parsed in
      10A; the resolver chain falls back to session → encrypted →
      admin-env when the header tier is absent.  10B wires the header
      tier on top of the same factory.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import Request, Security
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
    "SESSION_COOKIE_NAME",
    "extract_session_id",
    "get_chroma",
    "get_embedder",
    "get_encrypted_credential_store",
    "get_filing_store",
    "get_registry",
    "get_retrieval_service",
    "get_session_id",
    "get_session_store",
    "is_admin_request",
    "request_scoped_resolver",
    "verify_admin_key",
    "verify_api_key",
]


logger = get_logger(__name__)


# Cookie name for the server-minted ``session_id`` (10A.6).
SESSION_COOKIE_NAME = "sec_rag_session"


# Until SSO/OAuth lands in Phase 13, the only writer/reader of the
# encrypted credential store is "the admin user" — a stable opaque
# string with no PII.  Multi-user identifiers arrive with auth.
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
    api_key: str | None = Security(_api_key_header),
) -> None:
    """Validate the ``X-API-Key`` header when authentication is enabled.

    When ``API_KEY`` is unset (Scenario A — local dev) authentication is
    disabled and every caller passes through.  When it is set, the
    header MUST match (constant-time comparison) or the request is
    rejected with 401.
    """
    expected = get_settings().api.key
    if expected is None:
        return
    if not _secrets_match(api_key, expected):
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
# Per-request credential resolver chain
# ---------------------------------------------------------------------------


def request_scoped_resolver(
    request: Request,
    *,
    extra_resolvers: Callable[[str], str | None] | None = None,
) -> ApiKeyResolver:
    """Build a Phase-9 resolver chain for the current request.

    Composition (first hit wins):

    1. ``extra_resolvers`` — caller-supplied front of the chain.  10B
       installs the per-provider header parser here so the factory
       seam needs no awareness of the header layer.
    2. **Session store** keyed by the validated ``session_id`` cookie,
       when present.
    3. **Encrypted user store** keyed by ``ADMIN_USER_ID`` when both
       (a) the store exists on ``app.state`` and (b) the request is
       admin-authenticated.  Until SSO lands the encrypted tier is
       admin-only by design.
    4. ``default_api_key_resolver`` — process-environment fallback.

    Returns a callable of the
    :data:`~sec_generative_search.providers.factory.ApiKeyResolver`
    shape that plugs straight into ``build_embedder`` /
    ``build_llm_provider``.
    """
    chain: list[ApiKeyResolver] = []

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
