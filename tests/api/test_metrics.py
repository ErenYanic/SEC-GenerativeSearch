"""Tests for the admin-gated metrics exposition route.

Single endpoint under test: ``GET /api/metrics``.

Coverage focuses on:

    - the OpenMetrics exposition body + ``text/plain`` content type +
      ``Cache-Control: no-store``;
    - the **admin gate**: with both keys configured, no keys → 401,
      API-key only → 403, both → 200 (read tier rejects first, so an
      admin-key-only probe surfaces 401 not 403);
    - the **503 fallback** when the ``[metrics]`` extra is absent;
    - the route reflects no request input (it takes none) and the body
      carries only the content-free series the facade admits;
    - rate-limit / body-cap classification (``general`` bucket, 1 KiB).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.core.metrics import get_metrics, reset_metrics

pytestmark = pytest.mark.usefixtures("_reset_metrics_singleton")


@pytest.fixture
def _reset_metrics_singleton() -> Iterator[None]:
    reset_metrics()
    yield
    reset_metrics()


# ---------------------------------------------------------------------------
# Exposition body
# ---------------------------------------------------------------------------


class TestExposition:
    def test_returns_openmetrics_body(self, api_client: TestClient) -> None:
        pytest.importorskip("prometheus_client")
        # Record a sample so the body is non-trivial.
        get_metrics().observe_ingestion(2.5)

        response = api_client.get("/api/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        # Operator-only snapshot — never cached by a shared proxy.
        assert response.headers["cache-control"] == "no-store"
        assert "sec_ingestion_duration_seconds" in response.text

    def test_503_when_extra_absent(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate a deployment without ``prometheus-client``: the facade
        # renders ``None`` and the route MUST 503 with an install hint
        # rather than an empty 200 a scraper would treat as "no samples".
        class _InertMetrics:
            def render_latest(self) -> None:
                return None

        monkeypatch.setattr(
            "sec_generative_search.api.routes.metrics.get_metrics",
            lambda: _InertMetrics(),
        )

        response = api_client.get("/api/metrics")
        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "metrics_unavailable"
        assert "metrics" in body["hint"].lower()


# ---------------------------------------------------------------------------
# Admin gate
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestMetricsAuthGate:
    def test_no_keys_rejected_when_configured(self, api_client_factory) -> None:
        client = api_client_factory(
            API_KEY="shared-team-key",  # pragma: allowlist secret
            API_ADMIN_KEY="secret-admin-key",  # pragma: allowlist secret
        )
        response = client.get("/api/metrics")
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"

    def test_api_key_only_is_forbidden(self, api_client_factory) -> None:
        # Metrics is an admin-tier surface: a valid read key alone must
        # not reach it.
        client = api_client_factory(
            API_KEY="shared-team-key",  # pragma: allowlist secret
            API_ADMIN_KEY="secret-admin-key",  # pragma: allowlist secret
        )
        response = client.get(
            "/api/metrics",
            headers={"X-API-Key": "shared-team-key"},  # pragma: allowlist secret
        )
        assert response.status_code == 403
        assert response.json()["error"] == "admin_required"

    def test_admin_key_only_surfaces_401_not_403(self, api_client_factory) -> None:
        # The read-tier dependency runs first, so an admin-key-only
        # request is rejected as 401 (missing API key), never leaking
        # that the admin tier even exists.
        client = api_client_factory(
            API_KEY="shared-team-key",  # pragma: allowlist secret
            API_ADMIN_KEY="secret-admin-key",  # pragma: allowlist secret
        )
        response = client.get(
            "/api/metrics",
            headers={"X-Admin-Key": "secret-admin-key"},  # pragma: allowlist secret
        )
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"

    def test_both_keys_pass(self, api_client_factory) -> None:
        pytest.importorskip("prometheus_client")
        client = api_client_factory(
            API_KEY="shared-team-key",  # pragma: allowlist secret
            API_ADMIN_KEY="secret-admin-key",  # pragma: allowlist secret
        )
        response = client.get(
            "/api/metrics",
            headers={
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Admin-Key": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_open_when_no_keys_configured(self, api_client: TestClient) -> None:
        # Scenario A — no keys set: the endpoint is reachable (local dev).
        pytest.importorskip("prometheus_client")
        response = api_client.get("/api/metrics")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Content-free contract + classification
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestMetricsPrivacy:
    def test_body_carries_no_request_input(self, api_client: TestClient) -> None:
        pytest.importorskip("prometheus_client")
        # The route takes no input; a query string MUST NOT appear in the
        # exposition body (the endpoint never reflects request data).
        get_metrics().observe_retrieval(0.1)
        response = api_client.get("/api/metrics?ticker=AAPL&q=secret-query")
        assert response.status_code == 200
        assert "AAPL" not in response.text
        assert "secret-query" not in response.text

    def test_path_classifies_as_general_with_1kib_cap(self) -> None:
        from sec_generative_search.api.policies import resolve_policy

        policy = resolve_policy("/api/metrics/", "GET")
        assert policy.rate_category == "general"
        assert policy.max_body_bytes == 1024
