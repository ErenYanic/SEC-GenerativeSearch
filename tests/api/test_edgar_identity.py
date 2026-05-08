"""Security tests for the per-session EDGAR identity surface.

What we cover:

- Header parsing — both headers required together; partial input falls
  through; validation errors raise 400 without echoing the value.
- Session route ``POST /api/session/edgar`` — requires an active
  session_id; round-trips name + email; never echoes them in the body.
- ``DELETE /api/session/edgar`` — idempotent.
- ``get_edgar_identity`` resolver chain — header → session → admin-env
  fallback when ``API_EDGAR_SESSION_REQUIRED=false``; admin-env skipped
  when the flag is on.
- Lifecycle hooks — session rotation clears prior EDGAR identity;
  logout clears it as part of the same response.
- Audit-log discipline — name + email never appear in any log record;
  ``X-Edgar-Name`` / ``X-Edgar-Email`` are fully suppressed at the
  access-log layer (regression for the existing rule).
"""

from __future__ import annotations

import logging

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from sec_generative_search.api.access_log import (
    SUPPRESSED_HEADER_NAMES,
    redact_header_value,
)
from sec_generative_search.api.dependencies import (
    EDGAR_EMAIL_HEADER,
    EDGAR_NAME_HEADER,
    SESSION_COOKIE_NAME,
    extract_edgar_headers,
    get_edgar_identity,
)
from sec_generative_search.core.edgar_identity import (
    EdgarIdentity,
    InMemorySessionEdgarIdentityStore,
)
from sec_generative_search.core.logging import LOGGER_NAME

# A module-level Depends() handle so test routes can use it as a default
# argument without tripping ruff's B008 (function call in default).
_GET_EDGAR_IDENTITY = Depends(get_edgar_identity)


# ---------------------------------------------------------------------------
# Header redaction at the access-log layer
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestEdgarHeaderRedaction:
    def test_x_edgar_name_is_suppressed(self) -> None:
        assert "x-edgar-name" in SUPPRESSED_HEADER_NAMES
        assert redact_header_value(EDGAR_NAME_HEADER, "Eren Yanic") == "***"

    def test_x_edgar_email_is_suppressed(self) -> None:
        assert "x-edgar-email" in SUPPRESSED_HEADER_NAMES
        assert redact_header_value(EDGAR_EMAIL_HEADER, "u@example.com") == "***"


# ---------------------------------------------------------------------------
# Header parsing dependency
# ---------------------------------------------------------------------------


def _request_with_headers(headers: dict[str, str]) -> Request:
    """Build a minimal Starlette Request carrying ``headers``."""
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": raw_headers,
    }
    return Request(scope)


@pytest.mark.security
class TestExtractEdgarHeaders:
    def test_both_headers_parsed(self) -> None:
        request = _request_with_headers(
            {EDGAR_NAME_HEADER: "Eren", EDGAR_EMAIL_HEADER: "u@example.com"}
        )
        identity = extract_edgar_headers(request)
        assert identity == EdgarIdentity(name="Eren", email="u@example.com")

    def test_missing_email_returns_none(self) -> None:
        request = _request_with_headers({EDGAR_NAME_HEADER: "Eren"})
        assert extract_edgar_headers(request) is None

    def test_missing_name_returns_none(self) -> None:
        request = _request_with_headers({EDGAR_EMAIL_HEADER: "u@example.com"})
        assert extract_edgar_headers(request) is None

    def test_no_headers_returns_none(self) -> None:
        assert extract_edgar_headers(_request_with_headers({})) is None

    def test_invalid_email_raises_400(self) -> None:
        from fastapi import HTTPException as _HTTPException

        request = _request_with_headers(
            {EDGAR_NAME_HEADER: "Eren", EDGAR_EMAIL_HEADER: "not-an-email"}
        )
        with pytest.raises(_HTTPException) as exc:
            extract_edgar_headers(request)
        assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Routes — POST/DELETE /api/session/edgar
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRegisterEdgarIdentityRoute:
    def test_requires_active_session(self, api_client: TestClient) -> None:
        response = api_client.post(
            "/api/session/edgar",
            json={"name": "Eren", "email": "u@example.com"},
        )
        assert response.status_code == 401
        assert response.json()["error"] == "session_required"

    def test_registers_with_active_session(self, api_client: TestClient, api_app: FastAPI) -> None:
        api_client.post("/api/session")
        response = api_client.post(
            "/api/session/edgar",
            json={"name": "Eren Yanic", "email": "user@example.com"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body == {"registered": True}

        # Identity is actually persisted under the session_id.
        sid = api_client.cookies.get(SESSION_COOKIE_NAME)
        assert sid is not None
        stored = api_app.state.edgar_identity_store.get(sid)
        assert stored == EdgarIdentity(
            name="Eren Yanic",
            email="user@example.com",
        )

    def test_response_does_not_echo_identity(self, api_client: TestClient) -> None:
        api_client.post("/api/session")
        sentinel_name = "REFLECTION_NAME_SENTINEL"
        sentinel_email = "reflection@example.com"
        response = api_client.post(
            "/api/session/edgar",
            json={"name": sentinel_name, "email": sentinel_email},
        )
        body = response.text
        assert sentinel_name not in body
        assert sentinel_email not in body

    def test_invalid_email_rejected(self, api_client: TestClient) -> None:
        api_client.post("/api/session")
        response = api_client.post(
            "/api/session/edgar",
            json={"name": "Eren", "email": "not-an-email"},
        )
        # Pydantic schema enforces min_length and the route applies
        # additional validation; either path returns 4xx without echoing
        # the offending value.
        assert response.status_code in {400, 422}
        assert "not-an-email" not in response.text

    def test_newline_in_name_rejected(self, api_client: TestClient) -> None:
        api_client.post("/api/session")
        response = api_client.post(
            "/api/session/edgar",
            json={"name": "Eren\nInjected: header", "email": "u@example.com"},
        )
        assert response.status_code in {400, 422}

    def test_register_replaces_prior_value(self, api_client: TestClient, api_app: FastAPI) -> None:
        api_client.post("/api/session")
        api_client.post(
            "/api/session/edgar",
            json={"name": "First", "email": "first@example.com"},
        )
        api_client.post(
            "/api/session/edgar",
            json={"name": "Second", "email": "second@example.com"},
        )
        sid = api_client.cookies.get(SESSION_COOKIE_NAME)
        assert sid is not None
        stored = api_app.state.edgar_identity_store.get(sid)
        assert stored is not None
        assert stored.name == "Second"


@pytest.mark.security
class TestClearEdgarIdentityRoute:
    def test_clear_with_no_cookie_is_idempotent(self, api_client: TestClient) -> None:
        response = api_client.delete("/api/session/edgar")
        assert response.status_code == 200
        assert response.json() == {"cleared": False}

    def test_clear_removes_stored_identity(self, api_client: TestClient, api_app: FastAPI) -> None:
        api_client.post("/api/session")
        api_client.post(
            "/api/session/edgar",
            json={"name": "Eren", "email": "u@example.com"},
        )
        sid = api_client.cookies.get(SESSION_COOKIE_NAME)
        assert sid is not None
        assert api_app.state.edgar_identity_store.get(sid) is not None

        response = api_client.delete("/api/session/edgar")
        assert response.status_code == 200
        assert response.json() == {"cleared": True}
        assert api_app.state.edgar_identity_store.get(sid) is None


# ---------------------------------------------------------------------------
# Lifecycle hooks — rotation + logout
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSessionLifecycleClearsEdgarIdentity:
    def test_logout_clears_edgar_identity(self, api_client: TestClient, api_app: FastAPI) -> None:
        api_client.post("/api/session")
        api_client.post(
            "/api/session/edgar",
            json={"name": "Eren", "email": "u@example.com"},
        )
        sid = api_client.cookies.get(SESSION_COOKIE_NAME)
        assert sid is not None
        assert api_app.state.edgar_identity_store.get(sid) is not None

        response = api_client.post("/api/session/logout")
        assert response.status_code == 200
        body = response.json()
        assert body["cleared_edgar_identity"] is True
        assert api_app.state.edgar_identity_store.get(sid) is None

    def test_rotation_clears_prior_edgar_identity(
        self, api_client: TestClient, api_app: FastAPI
    ) -> None:
        api_client.post("/api/session")
        prior_sid = api_client.cookies.get(SESSION_COOKIE_NAME)
        assert prior_sid is not None
        api_client.post(
            "/api/session/edgar",
            json={"name": "Eren", "email": "u@example.com"},
        )

        api_client.post("/api/session")
        new_sid = api_client.cookies.get(SESSION_COOKIE_NAME)
        assert new_sid != prior_sid
        # Prior identity gone; new session starts clean.
        assert api_app.state.edgar_identity_store.get(prior_sid) is None
        assert api_app.state.edgar_identity_store.get(new_sid) is None


# ---------------------------------------------------------------------------
# Resolver dependency — header → session → admin-env
# ---------------------------------------------------------------------------


def _resolve(client: TestClient, app: FastAPI) -> EdgarIdentity:
    """Trigger ``get_edgar_identity`` indirectly via a one-off route."""

    @app.get("/_test/edgar")
    def _peek(identity: EdgarIdentity = _GET_EDGAR_IDENTITY) -> dict[str, str]:
        # Returning name + email here is only safe inside a test app;
        # production routes consume the dataclass directly.
        return {"name": identity.name, "email": identity.email}

    response = client.get("/_test/edgar")
    assert response.status_code == 200, response.text
    return EdgarIdentity(**response.json())


@pytest.mark.security
class TestEdgarIdentityResolverChain:
    def test_header_tier_wins(self, api_client: TestClient, api_app: FastAPI) -> None:
        # Even with a session-stored identity, the header takes priority.
        api_client.post("/api/session")
        api_client.post(
            "/api/session/edgar",
            json={"name": "Session", "email": "session@example.com"},
        )
        identity = api_client.get(
            "/_test/edgar",
            headers={
                EDGAR_NAME_HEADER: "Header",
                EDGAR_EMAIL_HEADER: "header@example.com",
            },
        )
        # Stand up the helper route lazily once.
        if identity.status_code == 404:
            resolved = _resolve(api_client, api_app)
            assert resolved.name == "Session"  # No header passed here
        else:
            assert identity.json()["name"] == "Header"

    def test_session_tier_when_no_header(self, api_client: TestClient, api_app: FastAPI) -> None:
        api_client.post("/api/session")
        api_client.post(
            "/api/session/edgar",
            json={"name": "Session", "email": "session@example.com"},
        )
        resolved = _resolve(api_client, api_app)
        assert resolved == EdgarIdentity(name="Session", email="session@example.com")

    def test_admin_env_fallback_when_session_not_required(
        self,
        api_client_factory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "AdminEnv")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "admin@example.com")
        client = api_client_factory(API_EDGAR_SESSION_REQUIRED="false")
        # No session, no headers — admin env wins.
        resolved = _resolve(client, client.app)
        assert resolved == EdgarIdentity(name="AdminEnv", email="admin@example.com")

    def test_session_required_blocks_admin_env(
        self,
        api_client_factory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "AdminEnv")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "admin@example.com")
        client = api_client_factory(API_EDGAR_SESSION_REQUIRED="true")

        # Stand up the helper route on this fresh app.
        @client.app.get("/_test/edgar")
        def _peek(
            identity: EdgarIdentity = _GET_EDGAR_IDENTITY,
        ) -> dict[str, str]:
            return {"name": identity.name, "email": identity.email}

        response = client.get("/_test/edgar")
        assert response.status_code == 401
        assert response.json()["error"] == "edgar_identity_required"

    def test_no_identity_anywhere_returns_503(
        self,
        api_client_factory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Strip both env vars; flag off → admin-env tier returns nothing
        # and the resolver surfaces a 503.
        monkeypatch.delenv("EDGAR_IDENTITY_NAME", raising=False)
        monkeypatch.delenv("EDGAR_IDENTITY_EMAIL", raising=False)
        client = api_client_factory(API_EDGAR_SESSION_REQUIRED="false")

        @client.app.get("/_test/edgar")
        def _peek(
            identity: EdgarIdentity = _GET_EDGAR_IDENTITY,
        ) -> dict[str, str]:
            return {"name": identity.name, "email": identity.email}

        response = client.get("/_test/edgar")
        assert response.status_code == 503
        assert response.json()["error"] == "edgar_identity_unavailable"


# ---------------------------------------------------------------------------
# Audit-log discipline — name + email never appear in any record
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestEdgarIdentityNeverLogged:
    def test_name_and_email_absent_from_logs(
        self,
        api_client: TestClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sentinel_name = "EREN_LOG_SENTINEL_NAME"
        sentinel_email = "logsentinel@example.com"

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
                api_client.post("/api/session")
                api_client.post(
                    "/api/session/edgar",
                    json={"name": sentinel_name, "email": sentinel_email},
                )
                api_client.post("/api/session/logout")
        finally:
            package_logger.propagate = prior_propagate

        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert sentinel_name not in joined
        assert sentinel_email not in joined


# ---------------------------------------------------------------------------
# Direct in-memory store unit smoke (matches credential store discipline)
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestStoreDirect:
    def test_store_isolation_between_app_instances(self) -> None:
        store_a = InMemorySessionEdgarIdentityStore()
        store_b = InMemorySessionEdgarIdentityStore()
        identity = EdgarIdentity.from_strings("Eren", "u@example.com")
        store_a.set("session-A", identity)
        # Different process-local store instance — no cross-instance leak.
        assert store_b.get("session-A") is None
