"""Regression locks for threadpool dispatch and correlation-ID propagation.

These tests verify that sync handlers still see the request-bound
correlation ID and that the converted search route logs with the same ID.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sec_generative_search.api.middleware import CorrelationIdMiddleware
from sec_generative_search.core.correlation import get_correlation_id
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import InMemorySessionEdgarIdentityStore
from sec_generative_search.core.logging import get_logger

# A shape-valid inbound X-Request-ID so the middleware adopts it verbatim
# rather than minting a fresh one — that lets the tests assert against a
# known value.
_INBOUND_ID = "f1-threadpool-lock-abcdef123456"


# ---------------------------------------------------------------------------
# 1 + 2. Mechanism: the ContextVar survives the sync-dispatch hop
# ---------------------------------------------------------------------------


def _mechanism_app() -> FastAPI:
    """Minimal app with one sync route and one async route."""
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/cid-sync")
    def cid_sync() -> dict[str, str | None]:
        return {"cid": get_correlation_id()}

    @app.get("/cid-async")
    async def cid_async() -> dict[str, str | None]:
        return {"cid": get_correlation_id()}

    return app


@pytest.mark.security
def test_sync_route_sees_bound_correlation_id() -> None:
    """A sync handler reads the ID the middleware bound."""
    client = TestClient(_mechanism_app(), base_url="https://testserver")
    resp = client.get("/cid-sync", headers={"X-Request-ID": _INBOUND_ID})
    assert resp.status_code == 200
    # Read inside the threadpool thread == the bound ID.
    assert resp.json()["cid"] == _INBOUND_ID
    # And the middleware still echoes it on the response.
    assert resp.headers["X-Request-ID"] == _INBOUND_ID


@pytest.mark.security
def test_async_route_sees_bound_correlation_id_control() -> None:
    """Control: an async handler binds the ID too."""
    client = TestClient(_mechanism_app(), base_url="https://testserver")
    resp = client.get("/cid-async", headers={"X-Request-ID": _INBOUND_ID})
    assert resp.status_code == 200
    assert resp.json()["cid"] == _INBOUND_ID
    assert resp.headers["X-Request-ID"] == _INBOUND_ID


# ---------------------------------------------------------------------------
# 3. Real surface: a converted route's audit line carries the bound ID
# ---------------------------------------------------------------------------


class _StubRetrieval:
    """Trivial retrieval double — the correlation lock does not need hits."""

    def retrieve(self, query: str, **kwargs: Any) -> list:
        return []


class _CorrelationCapture(logging.Handler):
    """Record the correlation ID visible *at emit time* for each record."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str | None] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.seen.append(get_correlation_id())


@pytest.fixture
def _search_app():
    """Build the real app with a stub retrieval service on ``app.state``."""
    from sec_generative_search.api.app import create_app

    app = create_app()
    app.state.retrieval_service = _StubRetrieval()
    app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
    app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
    app.state.encrypted_credential_store = None
    return app


@pytest.fixture
def _capture_audit() -> Iterator[_CorrelationCapture]:
    """Attach a capturing handler to the audit logger for the test's span."""
    audit_logger = get_logger("security.audit")
    handler = _CorrelationCapture()
    audit_logger.addHandler(handler)
    try:
        yield handler
    finally:
        audit_logger.removeHandler(handler)


@pytest.mark.security
def test_converted_search_route_audit_carries_correlation_id(
    _search_app,
    _capture_audit: _CorrelationCapture,
) -> None:
    """The converted ``/api/search`` audit line carries the bound ID."""
    client = TestClient(_search_app, base_url="https://testserver")
    resp = client.post(
        "/api/search",
        json={"query": "revenue growth"},
        headers={"X-Request-ID": _INBOUND_ID},
    )
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == _INBOUND_ID
    # The audit line emitted from inside the threadpooled handler saw the
    # bound ID.
    assert _INBOUND_ID in _capture_audit.seen
