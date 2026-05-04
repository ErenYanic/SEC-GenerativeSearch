"""Shared fixtures for the API test surface.

Strategy
--------

The 10A tests exercise the route + middleware stack.  We deliberately
*skip* the production lifespan (which boots the embedder, ChromaDB,
SQLite, and the credential stores) and inject a minimal set of stubs on
``app.state`` instead.  This keeps the tests fast, hermetic, and free
of optional-extras dependencies (sentence-transformers, SQLCipher).

Pattern: build the app via :func:`create_app`, attach the singletons
the routes actually read, then return a :class:`TestClient` *without*
entering the ``with`` context — Starlette only runs ``lifespan`` inside
that context.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore


@pytest.fixture(autouse=True)
def _clean_api_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset API-relevant env vars so each test sees deterministic settings."""
    for key in list(os.environ.keys()):
        if key.startswith("API_"):
            monkeypatch.delenv(key, raising=False)
    reload_settings()
    yield
    reload_settings()


def _build_test_app(*, env: dict[str, str] | None = None):
    """Create a test app with a stubbed ``app.state``.

    Importing :mod:`sec_generative_search.api.app` builds a module-level
    ``app`` object via ``create_app()`` once at import time, but each
    test wants its own settings — so we re-import via ``create_app``
    after ``reload_settings()`` to pick up monkeypatched env vars.
    """
    if env:
        for key, value in env.items():
            os.environ[key] = value
    reload_settings()

    # Local import keeps the module fresh after env mutations.
    from sec_generative_search.api.app import create_app

    app = create_app()
    # Stub the singletons the 10A route surface reads.  Heavier wiring
    # is added per-test via the ``stub_state`` fixture when needed.
    app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=3600)
    app.state.encrypted_credential_store = None
    return app


@pytest.fixture
def api_app():
    """Build a fresh app per test (no lifespan)."""
    yield _build_test_app()


@pytest.fixture
def api_client(api_app) -> Iterator[TestClient]:
    """Test client without entering the lifespan context.

    ``base_url`` is ``https://`` so the ``Secure`` cookie set by the
    session route round-trips between requests inside the same client
    (the cookiejar refuses to send ``Secure`` cookies over plain HTTP).
    """
    yield TestClient(api_app, base_url="https://testserver")


@pytest.fixture
def api_client_factory(monkeypatch: pytest.MonkeyPatch):
    """Build a client with custom env vars (e.g. ``API_KEY=secret``)."""

    def factory(**env: str) -> TestClient:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        app = _build_test_app(env=env)
        return TestClient(app, base_url="https://testserver")

    return factory
