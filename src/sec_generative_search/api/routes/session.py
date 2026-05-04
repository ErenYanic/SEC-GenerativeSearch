"""Server-side ``session_id`` minting and revocation.

Implements TODO 10A.6 / 10.9 — the load-bearing security control for
the Phase-9 :class:`InMemorySessionCredentialStore`.  A weak or
forgeable ``session_id`` collapses the entire credential isolation
model: one tenant's guess gives them another tenant's keys.

Cookie contract enforced here:

    - **Server-minted only** — ``secrets.token_urlsafe(32)`` (≥256 bits
      of entropy).  Any value supplied by a browser is ignored on
      ``POST /api/session``; mint always replaces.
    - **HTTP-only** — JavaScript cannot read the cookie.
        - **``Secure``** — browser refuses to send over plain HTTP except
            from ``localhost``/``127.0.0.1`` (modern browsers treat localhost
            as a secure context).
    - **``SameSite=Strict``** — eliminates the cross-site request
      forgery vector for credentialed endpoints.
    - **Rotated on auth-state change** — call ``POST /api/session``
      again on login/logout; the prior session is invalidated and a
      fresh ``session_id`` is issued.
    - **Never logged raw** — only ``mask_secret``-tailed forms hit the
      audit log (already enforced by the credentials module).
    - **Never in URLs / query strings / referrers** — the cookie path
      is the only transport surface.

Logout (``POST /api/session/logout``) clears the in-memory store entry
for the cookie's ``session_id`` and expires the cookie immediately.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Request, Response

from sec_generative_search.api.dependencies import (
    SESSION_COOKIE_NAME,
    extract_session_id,
    get_session_store,
)
from sec_generative_search.api.schemas import (
    SessionLogoutResponse,
    SessionResponse,
)
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.credentials import CredentialStore
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.security import mask_secret

__all__ = ["router"]


logger = get_logger(__name__)


router = APIRouter()


# ``secrets.token_urlsafe(32)`` produces a 43-char base64-url string
# carrying 256 bits of entropy.  This is the floor — never weaken.
_SESSION_ID_BYTES = 32


# Default browser-cookie sliding TTL.  Mirrors the in-memory store's
# ``_DEFAULT_SESSION_TTL_SECONDS`` so the cookie never outlives the
# server-side entry it points at.
_DEFAULT_COOKIE_MAX_AGE = 60 * 60  # one hour


def _mint_session_id() -> str:
    """Generate a fresh server-minted ``session_id``.

    Wrapped in a function so tests can monkeypatch it deterministically
    without touching every ``secrets.token_urlsafe`` call site.
    """
    return secrets.token_urlsafe(_SESSION_ID_BYTES)


def _set_session_cookie(response: Response, session_id: str, *, max_age: int) -> None:
    """Apply the cookie attributes required for session security.

    ``Secure`` and ``HttpOnly`` are unconditional.  ``SameSite=Strict``
    is the strongest mode FastAPI/Starlette expose — same-site form
    submissions and cross-origin XHR cannot send the cookie at all,
    which is exactly what we want for credentialed routes.
    """
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=max_age,
        path="/",
        httponly=True,
        secure=True,
        samesite="strict",
    )


def _clear_session_cookie(response: Response) -> None:
    """Issue an immediate-expiry ``Set-Cookie`` for the session cookie.

    ``delete_cookie`` requires the same path the cookie was set with;
    we mirror :func:`_set_session_cookie` exactly.
    """
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=True,
        httponly=True,
        samesite="strict",
    )


@router.post(
    "/session",
    response_model=SessionResponse,
    tags=["session"],
    summary="Mint a server-side session_id",
    status_code=201,
)
async def mint_session(
    request: Request,
    response: Response,
    store: CredentialStore = Depends(get_session_store),
) -> SessionResponse:
    """Issue a fresh ``session_id`` cookie.

    Idempotent rotation semantics: when the request already carries a
    ``session_id`` cookie, the prior session's stored credentials are
    cleared from the in-memory store before the new id is minted.  This
    an authenticating client sends the call once and the rotation
    happens transparently.

    The newly minted ``session_id`` is returned exclusively via the
    ``Set-Cookie`` header — the JSON body never carries it.
    """
    prior = extract_session_id(request)
    cleared = 0
    if prior is not None:
        cleared = store.clear(prior)
        if cleared:
            logger.info(
                "Rotated session %s — cleared %d credential(s)",
                mask_secret(prior),
                cleared,
            )

    session_id = _mint_session_id()
    audit_log(
        "session_mint",
        detail=(
            f"session_id_tail={mask_secret(session_id)} "
            f"client_ip={request.client.host if request.client else 'unknown'} "
            f"rotated={'yes' if prior is not None else 'no'}"
        ),
    )

    # Sliding TTL — the in-memory store evicts on idle; the cookie
    # ``Max-Age`` matches the operator-configurable session TTL.
    ttl_seconds = get_settings().api.session_ttl_seconds
    _set_session_cookie(response, session_id, max_age=ttl_seconds)

    return SessionResponse(
        issued=True,
        cookie_name=SESSION_COOKIE_NAME,
        expires_in_seconds=ttl_seconds,
    )


@router.post(
    "/session/logout",
    response_model=SessionLogoutResponse,
    tags=["session"],
    summary="Invalidate the current session_id",
)
async def logout_session(
    request: Request,
    response: Response,
    store: CredentialStore = Depends(get_session_store),
) -> SessionLogoutResponse:
    """Clear the session's credentials and expire the cookie.

    Idempotent — a request with no cookie returns ``cleared=0`` and the
    expired-cookie ``Set-Cookie`` is still emitted defensively.
    """
    session_id = extract_session_id(request)
    cleared = 0
    if session_id is not None:
        cleared = store.clear(session_id)
        audit_log(
            "session_logout",
            detail=(f"session_id_tail={mask_secret(session_id)} cleared={cleared}"),
        )

    _clear_session_cookie(response)
    return SessionLogoutResponse(cleared_credentials=cleared)
