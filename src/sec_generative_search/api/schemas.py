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

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "BulkDeleteRequest",
    "BulkDeleteResponse",
    "ClearAllResponse",
    "DeleteByIdsRequest",
    "DeleteByIdsResponse",
    "DeleteResponse",
    "EdgarIdentityClearResponse",
    "EdgarIdentityRegisterResponse",
    "EdgarIdentityRequest",
    "FilingListResponse",
    "FilingSchema",
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
    cleared_edgar_identity: bool = Field(
        default=False,
        description=("Whether a per-session EDGAR identity was cleared as part of logout."),
    )


class EdgarIdentityRequest(_BaseModel):
    """Body for ``POST /api/session/edgar``.

    Both fields are validated server-side before storage; the response
    NEVER echoes the values back.  The route requires an active
    server-minted ``session_id`` cookie — without one there is no scope
    in which to register the identity.
    """

    name: str = Field(
        min_length=1,
        max_length=128,
        description="Acting user's full name (sent to SEC as part of the User-Agent).",
    )
    email: str = Field(
        min_length=3,
        max_length=254,
        description="Acting user's email address (sent to SEC as part of the User-Agent).",
    )


class EdgarIdentityRegisterResponse(_BaseModel):
    """Result of a successful ``POST /api/session/edgar``.

    Body is deliberately minimal — neither name nor email is echoed back.
    """

    registered: bool


class EdgarIdentityClearResponse(_BaseModel):
    """Result of a successful ``DELETE /api/session/edgar``."""

    cleared: bool


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


# ---------------------------------------------------------------------------
# Filing management schemas
# ---------------------------------------------------------------------------


# SEC accession numbers are deterministic: 10-2-6 digits separated by
# hyphens (e.g. ``0000320193-23-000077``).  We pin the shape at the
# schema boundary so a stray identifier never reaches the registry.
_ACCESSION_PATTERN = r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$"

# Ticker symbols on US exchanges fit ``[A-Z][A-Z0-9.-]{0,9}`` but we
# upper-bound length defensively — a longer string is a sign of misuse,
# not an exotic ticker.
_TICKER_PATTERN = r"^[A-Z][A-Z0-9.\-]{0,15}$"

# Form types follow the SEC's published list; we keep validation
# permissive (any 10-character upper-case slug) so amended forms (10-K/A,
# 8-K/A) and future variants do not require a schema bump.
_FORM_TYPE_PATTERN = r"^[A-Z0-9][A-Z0-9/\-]{0,15}$"


class FilingSchema(_BaseModel):
    """Single filing row as exposed to the API.

    Strictly mirrors :class:`FilingRecord` minus the auto-increment
    ``id`` — the integer PK is an internal SQLite detail that does not
    belong on the wire.
    """

    ticker: str
    form_type: str
    filing_date: str
    accession_number: str
    chunk_count: int
    ingested_at: str


class FilingListResponse(_BaseModel):
    """Result of ``GET /api/filings/``.

    ``total`` is the count of returned rows, not the global filing
    count.  Operators that need the registry-wide total go to
    ``GET /api/status/``.
    """

    filings: list[FilingSchema]
    total: int


class DeleteResponse(_BaseModel):
    """Result of ``DELETE /api/filings/{accession}``."""

    accession_number: str
    chunks_deleted: int


class DeleteByIdsRequest(_BaseModel):
    """Body for ``POST /api/filings/delete-by-ids``.

    Bounded list length (1..500) keeps a single request from monopolising
    the SQLite write lock or generating a many-MB ChromaDB ``$in``
    clause.  Each accession is pinned to the SEC shape at schema time so
    the registry never sees a malformed identifier.
    """

    accession_numbers: list[str] = Field(
        min_length=1,
        max_length=500,
        description="Accession numbers to delete (1..500 per request).",
    )

    @field_validator("accession_numbers")
    @classmethod
    def _validate_accession_shape(cls, value: list[str]) -> list[str]:
        pattern = re.compile(_ACCESSION_PATTERN)
        for accession in value:
            if not pattern.fullmatch(accession):
                raise ValueError(
                    f"Invalid accession number shape: {accession!r}. Expected NNNNNNNNNN-NN-NNNNNN."
                )
        # Dedupe while preserving order — duplicates make the request
        # idempotent on the registry side, but waste a parameter slot in
        # the bounded ``IN (?, ?, …)`` clause.
        seen: set[str] = set()
        unique: list[str] = []
        for accession in value:
            if accession not in seen:
                seen.add(accession)
                unique.append(accession)
        return unique


class DeleteByIdsResponse(_BaseModel):
    """Result of ``POST /api/filings/delete-by-ids``."""

    filings_deleted: int
    chunks_deleted: int
    not_found: list[str]


class BulkDeleteRequest(_BaseModel):
    """Body for ``POST /api/filings/bulk-delete``.

    At least one of ``ticker`` / ``form_type`` MUST be set.  An empty
    body would expand to ``DELETE FROM filings`` and we have a separate
    confirm-gated route for that destructive shape.
    """

    ticker: str | None = Field(
        default=None,
        pattern=_TICKER_PATTERN,
        description="Filter by upper-case ticker symbol (e.g. AAPL).",
    )
    form_type: str | None = Field(
        default=None,
        pattern=_FORM_TYPE_PATTERN,
        description="Filter by upper-case form type (e.g. 10-K, 10-K/A).",
    )


class BulkDeleteResponse(_BaseModel):
    """Result of ``POST /api/filings/bulk-delete``."""

    filings_deleted: int
    chunks_deleted: int
    tickers_affected: list[str]


class ClearAllResponse(_BaseModel):
    """Result of ``DELETE /api/filings/?confirm=true``."""

    filings_deleted: int
    chunks_deleted: int


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
