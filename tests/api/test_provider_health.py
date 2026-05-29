"""Tests for the admin-gated provider-health route.

Single endpoint under test: ``GET /api/providers/health``.

Coverage:

    - the snapshot body shape + ``Cache-Control: no-store``;
    - the admin gate: with both keys configured, no keys → 401,
      API-key only → 403, admin-key only → 401 (read tier rejects first),
      both → 200;
        - the route reflects no request input and the body carries only the
            content-free snapshot fields;
    - rate-limit / body-cap classification (``general`` bucket, 1 KiB).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.core.provider_health import (
    get_provider_health,
    reset_provider_health,
)

pytestmark = pytest.mark.usefixtures("_reset_provider_health_singleton")


@pytest.fixture
def _reset_provider_health_singleton() -> Iterator[None]:
    reset_provider_health()
    yield
    reset_provider_health()


# ---------------------------------------------------------------------------
# Snapshot body
# ---------------------------------------------------------------------------


class TestSnapshotBody:
    def test_empty_when_no_calls_recorded(self, api_client: TestClient) -> None:
        response = api_client.get("/api/providers/health")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        body = response.json()
        assert body == {"providers": [], "total": 0}

    def test_reflects_recorded_outcomes(self, api_client: TestClient) -> None:
        health = get_provider_health()
        health.record_success("openai", 1.25)
        health.record_failure("anthropic", "ProviderTimeoutError")

        response = api_client.get("/api/providers/health")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2

        by_name = {row["provider"]: row for row in body["providers"]}
        assert by_name["openai"]["state"] == "closed"
        assert by_name["openai"]["total_successes"] == 1
        assert by_name["openai"]["last_latency_seconds"] == 1.25
        assert by_name["anthropic"]["total_failures"] == 1
        assert by_name["anthropic"]["last_error_type"] == "ProviderTimeoutError"

    def test_open_breaker_surfaces_state(self, api_client: TestClient) -> None:
        health = get_provider_health()
        # Default threshold is 5 consecutive failures.
        for _ in range(5):
            health.record_failure("openai", "ProviderError")

        response = api_client.get("/api/providers/health")
        row = response.json()["providers"][0]
        assert row["state"] == "open"
        assert row["consecutive_failures"] == 5


# ---------------------------------------------------------------------------
# Admin gate
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestProviderHealthAuthGate:
    def test_no_keys_rejected_when_configured(self, api_client_factory) -> None:
        client = api_client_factory(
            API_KEY="shared-team-key",  # pragma: allowlist secret
            API_ADMIN_KEY="secret-admin-key",  # pragma: allowlist secret
        )
        response = client.get("/api/providers/health")
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"

    def test_api_key_only_is_forbidden(self, api_client_factory) -> None:
        # A valid read key alone must not reach the admin surface.
        client = api_client_factory(
            API_KEY="shared-team-key",  # pragma: allowlist secret
            API_ADMIN_KEY="secret-admin-key",  # pragma: allowlist secret
        )
        response = client.get(
            "/api/providers/health",
            headers={"X-API-Key": "shared-team-key"},  # pragma: allowlist secret
        )
        assert response.status_code == 403
        assert response.json()["error"] == "admin_required"

    def test_admin_key_only_surfaces_401_not_403(self, api_client_factory) -> None:
        # The read-tier dependency runs first, so an admin-key-only request
        # is rejected 401.
        client = api_client_factory(
            API_KEY="shared-team-key",  # pragma: allowlist secret
            API_ADMIN_KEY="secret-admin-key",  # pragma: allowlist secret
        )
        response = client.get(
            "/api/providers/health",
            headers={"X-Admin-Key": "secret-admin-key"},  # pragma: allowlist secret
        )
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"

    def test_both_keys_pass(self, api_client_factory) -> None:
        client = api_client_factory(
            API_KEY="shared-team-key",  # pragma: allowlist secret
            API_ADMIN_KEY="secret-admin-key",  # pragma: allowlist secret
        )
        response = client.get(
            "/api/providers/health",
            headers={
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Admin-Key": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"

    def test_open_when_no_keys_configured(self, api_client: TestClient) -> None:
        # No keys configured: the endpoint is reachable in local dev.
        response = api_client.get("/api/providers/health")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Content-free contract + classification
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestProviderHealthPrivacy:
    def test_body_reflects_no_request_input(self, api_client: TestClient) -> None:
        get_provider_health().record_success("openai", 0.5)
        response = api_client.get("/api/providers/health?ticker=AAPL&q=secret-query")
        assert response.status_code == 200
        assert "AAPL" not in response.text
        assert "secret-query" not in response.text

    def test_row_field_set_is_content_free(self, api_client: TestClient) -> None:
        get_provider_health().record_success("openai", 0.5)
        row = api_client.get("/api/providers/health").json()["providers"][0]
        assert set(row.keys()) == {
            "provider",
            "state",
            "consecutive_failures",
            "total_failures",
            "total_successes",
            "last_error_type",
            "last_failure_seconds_ago",
            "last_latency_seconds",
        }

    def test_path_classifies_as_general_with_1kib_cap(self) -> None:
        from sec_generative_search.api.policies import resolve_policy

        policy = resolve_policy("/api/providers/health", "GET")
        assert policy.rate_category == "general"
        assert policy.max_body_bytes == 1024
