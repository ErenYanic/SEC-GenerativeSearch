"""Security tests for the session minting / logout surface (10A.6 / 10.9)."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from sec_generative_search.api.dependencies import (
    SESSION_COOKIE_NAME,
    extract_session_id,
)
from sec_generative_search.api.routes import session as session_module
from sec_generative_search.core.logging import LOGGER_NAME


def _request_with_cookie(value: str) -> Request:
    """Build a minimal ASGI Request carrying our session cookie value.

    Avoids a full TestClient round-trip for direct unit tests of
    :func:`extract_session_id`.
    """
    cookie_bytes = f"{SESSION_COOKIE_NAME}={value}".encode()
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [(b"cookie", cookie_bytes)],
    }
    return Request(scope)


@pytest.mark.security
class TestSessionMintEntropy:
    def test_session_id_min_length(self) -> None:
        # ``token_urlsafe(32)`` produces a 43-char base64-url string.
        sid = session_module._mint_session_id()
        assert len(sid) >= 43

    def test_session_id_alphabet_is_url_safe(self) -> None:
        sid = session_module._mint_session_id()
        for ch in sid:
            assert ch.isalnum() or ch in {"-", "_"}

    def test_two_mints_differ(self) -> None:
        # Probability of collision at 256 bits is negligible.
        assert session_module._mint_session_id() != session_module._mint_session_id()


@pytest.mark.security
class TestSessionCookieAttributes:
    def test_cookie_set_on_mint(self, api_client: TestClient) -> None:
        response = api_client.post("/api/session")
        assert response.status_code == 201
        cookie_header = response.headers.get("set-cookie")
        assert cookie_header is not None
        assert SESSION_COOKIE_NAME in cookie_header

    def test_cookie_carries_httponly_secure_samesite(self, api_client: TestClient) -> None:
        response = api_client.post("/api/session")
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie
        assert "secure" in cookie
        assert "samesite=strict" in cookie
        # Path must be set so logout can match it.
        assert "path=/" in cookie

    def test_response_body_does_not_leak_session_id(self, api_client: TestClient) -> None:
        response = api_client.post("/api/session")
        body = response.json()
        assert "session_id" not in body
        sid = api_client.cookies.get(SESSION_COOKIE_NAME)
        assert sid is not None
        for value in body.values():
            assert value != sid

    def test_audit_log_uses_masked_tail(
        self,
        api_client: TestClient,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force a deterministic id so we can assert its raw value never appears.
        sentinel = "Z" * 43
        monkeypatch.setattr(session_module, "_mint_session_id", lambda: sentinel)

        # ``configure_logging`` sets propagate=False on the package logger;
        # caplog hooks the root logger, so re-enable propagation for the
        # duration of this test only.
        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
                api_client.post("/api/session")
        finally:
            package_logger.propagate = prior_propagate

        for record in caplog.records:
            assert sentinel not in record.getMessage()


@pytest.mark.security
class TestForgedCookieRejection:
    def test_short_cookie_is_ignored(self) -> None:
        assert extract_session_id(_request_with_cookie("short")) is None

    def test_long_cookie_is_ignored(self) -> None:
        oversized = "a" * 1024
        assert extract_session_id(_request_with_cookie(oversized)) is None

    def test_invalid_chars_are_ignored(self) -> None:
        # 40 chars but with disallowed characters.
        bad = "!" * 40
        assert extract_session_id(_request_with_cookie(bad)) is None

    def test_valid_shape_passes(self) -> None:
        good = "A" * 43
        assert extract_session_id(_request_with_cookie(good)) == good

    def test_browser_supplied_id_does_not_authenticate(self, api_client: TestClient) -> None:
        # Attacker presents a syntactically valid but unminted cookie.
        forged = "A" * 43
        api_client.cookies.set(SESSION_COOKIE_NAME, forged)
        # No store entry for the forged id, so logout reports zero clears.
        response = api_client.post("/api/session/logout")
        assert response.status_code == 200
        assert response.json()["cleared_credentials"] == 0


@pytest.mark.security
class TestSessionLogout:
    def test_logout_clears_store_entry(self, api_client: TestClient, api_app: FastAPI) -> None:
        store = api_app.state.session_store

        # Mint a session, then directly install a credential under it.
        api_client.post("/api/session")
        sid = api_client.cookies.get(SESSION_COOKIE_NAME)
        assert sid is not None
        store.set(sid, "openai", "sk-secret-test")  # pragma: allowlist secret

        response = api_client.post("/api/session/logout")
        assert response.status_code == 200
        assert response.json()["cleared_credentials"] >= 1
        assert store.get(sid, "openai") is None

    def test_logout_without_cookie_is_idempotent(self, api_client: TestClient) -> None:
        response = api_client.post("/api/session/logout")
        assert response.status_code == 200
        assert response.json()["cleared_credentials"] == 0

    def test_logout_expires_cookie(self, api_client: TestClient) -> None:
        api_client.post("/api/session")
        response = api_client.post("/api/session/logout")
        cookie = response.headers["set-cookie"].lower()
        # Either Max-Age=0 or an Expires-in-the-past — both are valid.
        assert "max-age=0" in cookie or "expires=" in cookie


@pytest.mark.security
class TestSessionRotation:
    def test_rotation_clears_prior_credentials(
        self, api_client: TestClient, api_app: FastAPI
    ) -> None:
        store = api_app.state.session_store

        api_client.post("/api/session")
        prior_sid = api_client.cookies.get(SESSION_COOKIE_NAME)
        assert prior_sid is not None
        store.set(prior_sid, "openai", "sk-prior")  # pragma: allowlist secret

        api_client.post("/api/session")
        new_sid = api_client.cookies.get(SESSION_COOKIE_NAME)
        assert new_sid != prior_sid
        assert store.get(prior_sid, "openai") is None
