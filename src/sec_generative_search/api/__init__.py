"""FastAPI application surface for SEC-GenerativeSearch.

10A scope:

    - ``create_app()`` — application factory used by uvicorn, the
      FastAPI test client, and downstream embedders that wrap the app.
    - ``app`` — the module-level ASGI application built once at import.

Routes mounted in 10A: ``/api/health``, ``/api/status/``,
``/api/session``, ``/api/session/logout``.

10B will append ingestion, filing management, retrieval, RAG, and
provider-validate routes.
"""

from sec_generative_search.api.app import app, create_app

__all__ = ["app", "create_app"]
