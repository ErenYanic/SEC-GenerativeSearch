"""FastAPI application surface for SEC-GenerativeSearch.

Exports the application factory and the module-level ASGI application
built at import time.
"""

from sec_generative_search.api.app import app, create_app

__all__ = ["app", "create_app"]
