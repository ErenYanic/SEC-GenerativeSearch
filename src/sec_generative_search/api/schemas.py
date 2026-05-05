"""Pydantic v2 request and response models for the API surface.

Discipline:

    - All models use ``ConfigDict(extra="forbid")`` so unexpected fields
      are rejected loudly rather than silently dropped.
    - Response models MUST NOT carry credential-shaped fields.  A
      :pytest.mark.security regression test asserts this for the
      module's exports.
    - Models meant to be returned to the client must have stable field
      names; rename via aliases, never break.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "HealthResponse",
    "ProviderValidateRequest",
    "ProviderValidateResponse",
    "SessionLogoutResponse",
    "SessionResponse",
    "StatusResponse",
]


class _BaseModel(BaseModel):
    """Project default — strict and case-sensitive."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(_BaseModel):
    """Liveness response — deliberately tiny.

    ``version`` is *not* surfaced here so an anonymous caller cannot
    fingerprint the deployment.  Authenticated callers obtain it via
    :class:`StatusResponse`.
    """

    status: str = Field(
        default="ok",
        description="Liveness marker; always 'ok' when the API is reachable.",
    )


class StatusResponse(_BaseModel):
    """Authenticated status snapshot."""

    version: str
    deployment_profile: str
    embedding_provider: str
    embedding_model: str
    storage_filings: int
    is_admin: bool
    persist_provider_credentials: bool


class SessionResponse(_BaseModel):
    """Result of a successful ``POST /api/session``.

    The ``session_id`` itself is NEVER returned in the body — it lives
    exclusively in the ``Set-Cookie`` header so it cannot be read from
    JavaScript (the cookie is HTTP-only).  This response carries the
    operator-visible metadata only.
    """

    issued: bool
    cookie_name: str = Field(
        description="Name of the HTTP-only cookie that holds the session_id.",
    )
    expires_in_seconds: int = Field(
        description="Sliding TTL of the session as configured on the server.",
    )


class SessionLogoutResponse(_BaseModel):
    """Result of a successful ``POST /api/session/logout``."""

    cleared_credentials: int = Field(
        description="Number of provider credentials dropped from the session store.",
    )


class ProviderValidateRequest(_BaseModel):
    """Body for ``POST /api/providers/validate``.

    ``api_key`` is required and never echoed back in any response or
    log; the validate route hands it directly to
    :func:`validate_credential` which audit-logs only the masked tail.
    The schema does NOT support a query-string fallback for the key —
    bodies are encrypted by TLS in transit and never end up in proxy
    access logs the way query strings do.
    """

    provider: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$",
        description="Lower-case provider slug; must be registered in ProviderRegistry.",
    )
    api_key: str = Field(
        min_length=1,
        max_length=4096,
        description="The provider key to validate.  Never echoed back.",
    )
    surface: str = Field(
        default="llm",
        pattern=r"^(llm|embedding|reranker)$",
        description="Provider surface to validate against.",
    )
    model: str | None = Field(
        default=None,
        max_length=128,
        description="Optional model slug for embedding-surface validation.",
    )


class ProviderValidateResponse(_BaseModel):
    """Verdict from a key-validation attempt.

    ``valid=True`` means the provider accepted the key.  ``valid=False``
    is reserved for the explicit ``ProviderAuthError`` case — every
    other ``ProviderError`` (rate limit, timeout, content filter,
    transport) propagates as a 502 / 503 envelope so the caller does
    not interpret a network blip as a "wrong key".
    """

    valid: bool
    provider: str
    surface: str
