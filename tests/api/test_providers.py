"""Security tests for the provider key-validation route.

Covers body validation, registry lookup failures, auth verdicts,
transient provider errors, response-body redaction, and per-IP versus
per-session rate limiting.

Strategy: the registry lookup is monkeypatched to return a stub
provider with a controllable ``validate_key`` so the tests never touch
a real upstream SDK.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.providers.registry import ProviderRegistry, ProviderSurface


class _StubProvider:
    """Minimal stand-in matching ``BaseLLMProvider.validate_key()`` shape."""

    def __init__(self, *args, behaviour: str = "ok", **kwargs) -> None:
        self._behaviour = behaviour

    def validate_key(self) -> bool:
        if self._behaviour == "ok":
            return True
        if self._behaviour == "auth_error":
            raise ProviderAuthError("rejected")
        if self._behaviour == "rate_limit":
            raise ProviderRateLimitError("upstream rate limited")
        if self._behaviour == "timeout":
            raise ProviderTimeoutError("upstream timed out")
        if self._behaviour == "generic":
            raise ProviderError("transport blew up")
        raise AssertionError(f"unknown stub behaviour: {self._behaviour}")


@pytest.fixture
def stub_provider(monkeypatch: pytest.MonkeyPatch):
    """Return a factory that points the registry at ``_StubProvider``.

    Yields a callable that pins the stub's behaviour for a single test.
    """
    state = {"behaviour": "ok"}

    class _Factory(_StubProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(behaviour=state["behaviour"], **kwargs)

    def _set(behaviour: str) -> None:
        state["behaviour"] = behaviour

    # Direct ``get_entry`` so :func:`validate_credential` finds an
    # entry that "exists" for the surface under test.
    from sec_generative_search.providers.registry import ProviderEntry

    fake_entry = ProviderEntry(
        name="openai",
        surface=ProviderSurface.LLM,
        provider_cls=_Factory,
    )

    def _get_entry(name, surface):
        if name == "openai" and surface == ProviderSurface.LLM:
            return fake_entry
        # Defer to the original for everything else so "unknown_provider"
        # tests still get the real registry behaviour.
        raise KeyError(f"No provider registered for name='{name}', surface='{surface.value}'.")

    monkeypatch.setattr(
        ProviderRegistry,
        "get_entry",
        classmethod(lambda cls, n, s: _get_entry(n, s)),
    )
    return _set


@pytest.mark.security
class TestValidateRouteSchemaGuards:
    def test_missing_body_rejected(self, api_client: TestClient) -> None:
        response = api_client.post("/api/providers/validate")
        assert response.status_code == 422
        assert response.json()["error"] == "validation_failed"

    def test_empty_api_key_rejected(self, api_client: TestClient) -> None:
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": ""},
        )
        assert response.status_code == 422

    def test_uppercase_provider_rejected(self, api_client: TestClient) -> None:
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "OPENAI", "api_key": "sk-x" * 8},  # pragma: allowlist secret
        )
        # Slug pattern is lower-case only.
        assert response.status_code == 422

    def test_oversize_api_key_rejected(self, api_client: TestClient) -> None:
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": "x" * 5000},
        )
        assert response.status_code == 422


@pytest.mark.security
class TestValidateRouteVerdicts:
    def test_valid_key(self, api_client: TestClient, stub_provider) -> None:
        stub_provider("ok")
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": "sk-good-key-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        body = response.json()
        assert body == {"valid": True, "provider": "openai", "surface": "llm"}

    def test_auth_error_collapses_to_invalid(self, api_client: TestClient, stub_provider) -> None:
        stub_provider("auth_error")
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": "sk-bad-key-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        assert response.json()["valid"] is False

    def test_rate_limit_does_not_become_verdict(
        self, api_client: TestClient, stub_provider
    ) -> None:
        stub_provider("rate_limit")
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": "sk-key-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "provider_unavailable"
        # Hint MUST steer the caller away from rotating a working key.
        assert "do not rotate" in body["hint"].lower()

    def test_generic_provider_error_is_502(self, api_client: TestClient, stub_provider) -> None:
        stub_provider("generic")
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": "sk-key-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 502
        body = response.json()
        assert body["error"] == "provider_error"


@pytest.mark.security
class TestValidateRouteNoKeyLeak:
    def test_response_body_does_not_echo_key(self, api_client: TestClient, stub_provider) -> None:
        stub_provider("ok")
        secret = "sk-NEVER-ECHO-ME-1234-abcd"  # pragma: allowlist secret
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": secret},  # pragma: allowlist secret
        )
        # Whether valid or not, the body MUST NOT carry the raw key.
        assert secret not in response.text


@pytest.mark.security
class TestValidateRouteUnknownProvider:
    def test_unknown_provider_400(self, api_client: TestClient) -> None:
        # No fixture: real registry lookup is exercised.
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "definitely-not-real", "api_key": "x" * 16},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "unknown_provider"


@pytest.mark.security
class TestValidateRouteAuthGate:
    def test_api_key_required_when_configured(self, api_client_factory, stub_provider) -> None:
        stub_provider("ok")
        client = api_client_factory(API_KEY="rotated-key")  # pragma: allowlist secret
        # Missing X-API-Key header → 401.
        response = client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": "sk-test-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"


@pytest.mark.security
class TestValidateRouteRateLimit:
    def test_per_session_window_is_separate_from_per_ip(
        self, api_client_factory, stub_provider
    ) -> None:
        stub_provider("ok")
        # Tight limits so we hit the boundary fast.
        client = api_client_factory(
            API_RATE_LIMIT_VALIDATE="100",
            API_RATE_LIMIT_VALIDATE_PER_SESSION="3",
        )
        # Mint a session so the per-session bucket has a key.
        client.post("/api/session")

        # Burn through the per-session budget on the validate route.
        statuses: list[int] = []
        for _ in range(8):
            r = client.post(
                "/api/providers/validate",
                json={"provider": "openai", "api_key": "sk-test-1234"},
            )
            statuses.append(r.status_code)
        # Must include at least one 429 — the session bucket of 3 was
        # exhausted before the IP bucket of 100.
        assert 429 in statuses
        # Must include at least some 200s before the limiter kicked in.
        assert 200 in statuses
