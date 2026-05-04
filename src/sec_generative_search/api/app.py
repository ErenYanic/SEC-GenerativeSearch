"""FastAPI application factory and lifespan.

10A scope — what *this* file owns:

    - Build the FastAPI ``app`` instance via :func:`create_app`.
    - Initialise singletons in :func:`lifespan`: ``MetadataRegistry``,
      ``ChromaDBClient``, the embedder, ``FilingStore``,
      ``RetrievalService``, the Phase-9
      ``InMemorySessionCredentialStore``, and the optional
      ``EncryptedCredentialStore`` (only when the deployment profile
      enables persistent credential storage).
    - Wire the middleware stack and the structured-error handlers.
    - Mount 10A routes: ``/api/health``, ``/api/status/``,
      ``/api/session``, ``/api/session/logout``.
    - Toggle OpenAPI docs off when ``API_KEY`` is configured.

Out of scope for 10A — added by 10B:

    - LLM provider construction (per-request via
      :func:`request_scoped_resolver`).
    - ``TaskManager`` (ingestion progress).
    - Filings, ingest, retrieval, RAG, provider-validate routes.

Notes for the lifespan:

    - The embedder is constructed eagerly at startup using the
    *administrative* default-env-var resolver. A startup failure here
    is the right behaviour: a deployment that cannot embed will not
    serve correct retrievals; refuse early.
    - ``EncryptedCredentialStore`` is optional and gated by
      ``DB_PERSIST_PROVIDER_CREDENTIALS``.  When disabled, the resolver
      chain falls through to the session store and the admin-env
      fallback, which is the documented Scenario-A behaviour.
    - ``app.state`` is a Starlette-supplied attribute namespace; we
      attach typed names that ``api.dependencies`` reads back.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sec_generative_search import __version__
from sec_generative_search.api.errors import install_error_handlers
from sec_generative_search.api.middleware import (
    ContentSizeLimitMiddleware,
    InsecureTransportWarningMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from sec_generative_search.api.routes.health import router as health_router
from sec_generative_search.api.routes.session import router as session_router
from sec_generative_search.api.routes.status import router as status_router
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.types import EmbedderStamp
from sec_generative_search.database import ChromaDBClient, FilingStore, MetadataRegistry
from sec_generative_search.database.credentials import EncryptedCredentialStore
from sec_generative_search.providers.factory import build_embedder
from sec_generative_search.providers.registry import ProviderRegistry
from sec_generative_search.search import RetrievalService

__all__ = ["app", "create_app", "lifespan"]


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Boot every singleton, expose them on ``app.state``, and tear down.

    Order is load-bearing:

    1. Embedder — fails early on missing admin env var.
    2. ``EmbedderStamp`` from the registry — the storage seal.
    3. ``ChromaDBClient`` — opens (or creates) the sealed collection.
    4. ``MetadataRegistry`` — opens SQLite/SQLCipher and applies
       migrations including v2 (``provider_credentials``).
    5. ``FilingStore`` — coordinator over the two stores above.
    6. ``RetrievalService`` — pre-built single-query primitive.
    7. ``InMemorySessionCredentialStore`` — Phase-9 in-memory store.
    8. ``EncryptedCredentialStore`` — optional, gated by settings.
    """
    settings = get_settings()
    logger.info("SEC-GenerativeSearch API starting up (v%s)", __version__)

    # 1. Embedder — administrative selection, default-env resolver only.
    embedder = build_embedder(settings.embedding)

    # 2. Stamp via the registry — single source of truth for dimension.
    dimension = ProviderRegistry.get_dimension(
        settings.embedding.provider,
        settings.embedding.model_name,
    )
    stamp = EmbedderStamp(
        provider=settings.embedding.provider,
        model=settings.embedding.model_name,
        dimension=dimension,
    )

    # 3-5. Storage chain.
    chroma = ChromaDBClient(stamp)
    registry = MetadataRegistry()
    filing_store = FilingStore(chroma, registry)

    # 6. Retrieval — pre-built; provider-neutral by design.
    retrieval_service = RetrievalService(embedder=embedder, chroma_client=chroma)

    # 7. In-memory session credential store.  TTL mirrors the cookie's
    # ``Max-Age`` so a credential cannot outlive the cookie that points
    # at it (lazy eviction; no background thread).
    session_store = InMemorySessionCredentialStore(
        ttl_seconds=settings.api.session_ttl_seconds,
    )

    # 8. Optional encrypted store.  Settings validation already rejected
    # ``persist=true`` without SQLCipher at load time; we still defend
    # at this seam by letting the store's own constructor refuse loudly.
    encrypted_store: EncryptedCredentialStore | None = None
    if settings.database.persist_provider_credentials:
        try:
            encrypted_store = EncryptedCredentialStore(registry)
        except Exception:  # pragma: no cover - defensive log
            logger.exception(
                "Encrypted credential store failed to initialise — "
                "continuing without persistent credential tier."
            )
            encrypted_store = None

    # Expose every singleton on ``app.state``.
    app.state.settings = settings
    app.state.embedder = embedder
    app.state.chroma = chroma
    app.state.registry = registry
    app.state.filing_store = filing_store
    app.state.retrieval_service = retrieval_service
    app.state.session_store = session_store
    app.state.encrypted_credential_store = encrypted_store

    logger.info(
        "API ready: embedder=%s/%s, dim=%d, encrypted_store=%s",
        stamp.provider,
        stamp.model,
        stamp.dimension,
        "yes" if encrypted_store is not None else "no",
    )

    try:
        yield
    finally:
        logger.info("SEC-GenerativeSearch API shutting down.")
        # ``MetadataRegistry`` owns the SQLite/SQLCipher connection that
        # the encrypted store also uses — close it here, never inside
        # the encrypted store, so we don't double-close from two sites.
        registry.close()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and configure the ASGI application.

    The factory is the single seam every entry point goes through —
    uvicorn, the test client, and downstream embedders that wrap the
    app.  Settings are resolved here, not in module-level code, so a
    test fixture that patches the environment before calling
    :func:`create_app` gets fresh wiring.
    """
    settings = get_settings()

    # 10A.5 — disable OpenAPI / Swagger / Redoc when API_KEY is set.
    # In Scenario-A development the docs remain available; in
    # Scenarios B/C they are off so an unauthenticated probe cannot
    # discover the surface.
    is_protected = bool(settings.api.key)

    app = FastAPI(
        title="SEC-GenerativeSearch API",
        description=(
            "REST API for retrieval-augmented generation over SEC filings. "
            "10A surface: health, status, server-side session minting."
        ),
        version=__version__,
        docs_url=None if is_protected else "/docs",
        redoc_url=None if is_protected else "/redoc",
        openapi_url=None if is_protected else "/openapi.json",
        lifespan=lifespan,
    )

    install_error_handlers(app)

    # ------------------------------------------------------------------
    # Middleware stack — read top-to-bottom as outermost-to-innermost.
    # ASGI middleware is applied LIFO: the *last* ``add_middleware``
    # call is the *first* to see an inbound request.
    #
    # Order rationale (outer → inner):
    #
    #   1. CORSMiddleware           — first to see preflight; emits
    #                                 headers without invoking the app.
    #   2. SecurityHeadersMiddleware — touches every response, including
    #                                  rate-limit and 413 responses.
    #   3. RateLimitMiddleware       — rejects abusive callers before we
    #                                  pay the cost of body parsing.
    #   4. ContentSizeLimitMiddleware — cheap reject on the body stream.
    #   5. InsecureTransportWarning  — closest to the route layer; only
    #                                  needs to observe, not modify.
    # ------------------------------------------------------------------
    app.add_middleware(InsecureTransportWarningMiddleware)
    app.add_middleware(ContentSizeLimitMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        search_rpm=settings.api.rate_limit_search,
        ingest_rpm=settings.api.rate_limit_ingest,
        delete_rpm=settings.api.rate_limit_delete,
        general_rpm=settings.api.rate_limit_general,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "X-API-Key",
            "X-Admin-Key",
            "X-Edgar-Name",
            "X-Edgar-Email",
        ],
    )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(health_router, prefix="/api", tags=["meta"])
    app.include_router(status_router, prefix="/api/status", tags=["status"])
    app.include_router(session_router, prefix="/api", tags=["session"])

    return app


# ---------------------------------------------------------------------------
# Module-level ASGI app (used by uvicorn and the FastAPI test client).
# Cheap at import time: only the FastAPI instance + middleware + routes
# are wired here; storage / embedder / credential-store construction
# happens inside ``lifespan`` when uvicorn enters the lifespan context.
# ---------------------------------------------------------------------------


app = create_app()
