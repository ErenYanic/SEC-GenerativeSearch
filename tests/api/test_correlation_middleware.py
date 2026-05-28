"""Tests for CorrelationIdMiddleware."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")


@pytest.mark.security
class TestCorrelationIdMiddleware:
    def test_response_carries_minted_request_id(self, api_client: TestClient) -> None:
        response = api_client.get("/api/health")
        assert response.status_code == 200
        cid = response.headers.get("x-request-id")
        assert cid is not None
        assert _HEX32.match(cid)

    def test_each_request_gets_a_distinct_id(self, api_client: TestClient) -> None:
        a = api_client.get("/api/health").headers["x-request-id"]
        b = api_client.get("/api/health").headers["x-request-id"]
        assert a != b

    def test_valid_inbound_id_is_honoured(self, api_client: TestClient) -> None:
        supplied = "client-req-0001"
        response = api_client.get("/api/health", headers={"X-Request-ID": supplied})
        assert response.headers["x-request-id"] == supplied

    def test_malformed_inbound_id_is_replaced(self, api_client: TestClient) -> None:
        # A space is outside the allowed alphabet — the middleware must
        # discard it and mint a fresh ID rather than echo the bad value.
        response = api_client.get("/api/health", headers={"X-Request-ID": "has space"})
        cid = response.headers["x-request-id"]
        assert cid != "has space"
        assert _HEX32.match(cid)

    def test_overlong_inbound_id_is_replaced(self, api_client: TestClient) -> None:
        response = api_client.get("/api/health", headers={"X-Request-ID": "a" * 200})
        cid = response.headers["x-request-id"]
        assert cid != "a" * 200
        assert _HEX32.match(cid)

    def test_request_id_present_on_error_response(self, api_client: TestClient) -> None:
        # The ID must ride along even on a 404 from the error layer.
        response = api_client.get("/api/does-not-exist")
        assert response.status_code == 404
        assert _HEX32.match(response.headers["x-request-id"])

    def test_request_id_present_on_413(self, api_client: TestClient) -> None:
        # A middleware-issued 413 (oversize body) must still carry the ID,
        # proving the correlation middleware sits outside the content-size
        # limiter in the stack.
        oversize = 2 * 1024 * 1024
        response = api_client.post(
            "/api/session",
            content=b"x" * oversize,
            headers={"content-length": str(oversize)},
        )
        assert response.status_code == 413
        assert response.headers.get("x-request-id") is not None

    def test_security_headers_coexist_with_request_id(self, api_client: TestClient) -> None:
        # The correlation middleware sits outside SecurityHeaders; both
        # header sets must survive on the same response.
        response = api_client.get("/api/health")
        assert response.headers.get("x-frame-options") == "DENY"
        assert response.headers.get("x-request-id") is not None
