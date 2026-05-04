"""Security tests for the 10A middleware stack."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.api.middleware import (
    DEFAULT_MAX_CONTENT_LENGTH,
    SECURITY_HEADERS_RAW,
)


@pytest.mark.security
class TestSecurityHeaders:
    def test_all_security_headers_present_on_health(self, api_client: TestClient) -> None:
        response = api_client.get("/api/health")
        assert response.status_code == 200
        for name, value in SECURITY_HEADERS_RAW:
            assert response.headers.get(name.decode()) == value.decode()

    def test_security_headers_on_error_response(self, api_client: TestClient) -> None:
        # 404 — must still carry the headers.
        response = api_client.get("/api/does-not-exist")
        assert response.status_code == 404
        assert response.headers.get("x-frame-options") == "DENY"
        assert response.headers.get("x-content-type-options") == "nosniff"

    def test_csp_is_restrictive(self, api_client: TestClient) -> None:
        response = api_client.get("/api/health")
        csp = response.headers.get("content-security-policy", "")
        # The policy MUST include frame-ancestors 'none' (clickjacking
        # defence) and a default-src self anchor.
        assert "frame-ancestors 'none'" in csp
        assert "default-src 'self'" in csp


@pytest.mark.security
class TestContentSizeLimit:
    def test_oversize_content_length_rejected(self, api_client: TestClient) -> None:
        oversize = DEFAULT_MAX_CONTENT_LENGTH + 1
        response = api_client.post(
            "/api/session",
            content=b"x" * oversize,
            headers={"content-length": str(oversize)},
        )
        assert response.status_code == 413
        body = response.json()
        assert body["error"] == "payload_too_large"

    def test_valid_size_passes(self, api_client: TestClient) -> None:
        response = api_client.post("/api/session")
        assert response.status_code == 201


@pytest.mark.security
class TestErrorEnvelope:
    def test_404_returns_envelope_shape(self, api_client: TestClient) -> None:
        response = api_client.get("/api/does-not-exist")
        body = response.json()
        # Strict envelope: error / message / details / hint — no leaked
        # FastAPI ``detail`` shape.
        assert set(body.keys()) >= {"error", "message"}
        assert "detail" not in body

    def test_validation_error_returns_envelope(self, api_client: TestClient) -> None:
        # POST /api/session does not require a body, so trigger a
        # validation path indirectly: GET on a POST-only route returns
        # 405 with the envelope.
        response = api_client.get("/api/session")
        assert response.status_code == 405
        body = response.json()
        assert "error" in body and "message" in body


@pytest.mark.security
class TestOpenAPIGating:
    def test_docs_available_without_api_key(self, api_client: TestClient) -> None:
        response = api_client.get("/docs")
        # In Scenario A (no key) the docs are exposed.
        assert response.status_code == 200

    def test_openapi_schema_available_without_api_key(self, api_client: TestClient) -> None:
        response = api_client.get("/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()

    def test_docs_disabled_when_api_key_set(self, api_client_factory) -> None:
        client = api_client_factory(API_KEY="rotated-test-key")  # pragma: allowlist secret
        # Docs / Redoc / openapi.json all 404 when auth is enabled.
        for path in ("/docs", "/redoc", "/openapi.json"):
            response = client.get(path)
            assert response.status_code == 404, f"{path} leaked when API_KEY set"


@pytest.mark.security
class TestRateLimitOnSessionRoute:
    def test_session_route_is_rate_limited(self, api_client_factory) -> None:
        # Make the bucket small so we can hit it deterministically.
        client = api_client_factory(API_RATE_LIMIT_GENERAL="120")
        # Mint many times — the dedicated session bucket has a lower cap
        # than ``general`` (default 20).  Exceed it deliberately.
        last = None
        for _ in range(40):
            last = client.post("/api/session")
            if last.status_code == 429:
                break
        assert last is not None
        assert last.status_code == 429
        body = last.json()
        assert body["error"] == "rate_limited"
        assert "Retry-After" in last.headers


@pytest.mark.security
class TestMiddlewareOrdering:
    def test_security_headers_added_to_rate_limited_response(self, api_client_factory) -> None:
        # Rate-limit the session bucket and assert the 429 still carries
        # the full security header set — proves SecurityHeadersMiddleware
        # sits OUTSIDE RateLimitMiddleware.
        client = api_client_factory()
        last = None
        for _ in range(40):
            last = client.post("/api/session")
            if last.status_code == 429:
                break
        assert last is not None and last.status_code == 429
        assert last.headers.get("x-frame-options") == "DENY"
        assert last.headers.get("content-security-policy") is not None
