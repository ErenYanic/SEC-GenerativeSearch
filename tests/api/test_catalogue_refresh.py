"""Tests for the admin-gated model-catalogue refresh route.

Single endpoint under test: ``POST /api/providers/catalogue/refresh``.

The refresh *engine* (bounded fetch + untrusted-input validation + overlay
write) is exercised exhaustively in ``tests/providers/test_refresh.py``; here
we own only the route's contract:

    - the admin gate (rule A): no keys → 401, API-key only → 403, admin-key
      only → 401 (read tier rejects first), both → 200;
    - the success body is a content-free, allow-list lift of
      :class:`CatalogueRefreshReport` (source + counts only — never the
      overlay filesystem path) behind ``Cache-Control: no-store``;
    - the configured source / URL / overlay path are forwarded to the engine;
    - a successful refresh resets the active catalogue (in-process reload);
    - fail-closed mapping: :class:`CatalogueRefreshError` → 502, ``OSError``
      → 500, and neither echoes a path or upstream body.

``refresh_overlay`` is patched in every test, so no test here touches the
network or writes to disk.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.core.exceptions import CatalogueRefreshError
from sec_generative_search.core.types import CatalogueRefreshReport

_ROUTE = "/api/providers/catalogue/refresh"

# A sentinel overlay path with a recognisable marker — every test asserts it
# NEVER appears in a response body (no filesystem path on the wire).
_SECRET_PATH = "/srv/data/SENTINEL-overlay-path.json"


def _report(**overrides: object) -> CatalogueRefreshReport:
    base = {
        "source": "models_dev",
        "source_url": "https://models.dev/api.json",
        "provider_count": 7,
        "model_count": 42,
        "overlay_path": _SECRET_PATH,
    }
    base.update(overrides)
    return CatalogueRefreshReport(**base)  # type: ignore[arg-type]


@pytest.fixture
def _reset_catalogue_after() -> Iterator[None]:
    """Drop the active catalogue after each test so the global stays clean."""
    from sec_generative_search.providers.catalogue import reset_catalogue

    yield
    reset_catalogue()


pytestmark = pytest.mark.usefixtures("_reset_catalogue_after")


# ---------------------------------------------------------------------------
# Success body
# ---------------------------------------------------------------------------


class TestRefreshSuccess:
    def test_returns_content_free_report(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sec_generative_search.api.routes.catalogue as route

        monkeypatch.setattr(route, "refresh_overlay", lambda **_: _report())

        response = api_client.post(_ROUTE)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        body = response.json()
        assert body == {
            "source": "models_dev",
            "source_url": "https://models.dev/api.json",
            "provider_count": 7,
            "model_count": 42,
        }

    def test_forwards_configured_source_url_and_path(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sec_generative_search.api.routes.catalogue as route

        captured: dict[str, object] = {}

        def _spy(**kwargs: object) -> CatalogueRefreshReport:
            captured.update(kwargs)
            return _report()

        monkeypatch.setattr(route, "refresh_overlay", _spy)

        api_client.post(_ROUTE)
        # Defaults from ``ProviderSettings``.
        assert captured["source"] == "models_dev"
        assert captured["url"] is None
        assert captured["overlay_path"] == "./data/model_catalogue_overlay.json"

    def test_success_resets_active_catalogue(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sec_generative_search.api.routes.catalogue as route
        from sec_generative_search.providers import catalogue as catmod

        monkeypatch.setattr(route, "refresh_overlay", lambda **_: _report())

        reset_calls = {"n": 0}
        real_reset = catmod.reset_catalogue

        def _spy_reset() -> None:
            reset_calls["n"] += 1
            real_reset()

        monkeypatch.setattr(route, "reset_catalogue", _spy_reset)

        response = api_client.post(_ROUTE)
        assert response.status_code == 200
        assert reset_calls["n"] == 1


# ---------------------------------------------------------------------------
# Fail-closed mapping
# ---------------------------------------------------------------------------


class TestRefreshFailClosed:
    def test_catalogue_refresh_error_maps_to_502(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sec_generative_search.api.routes.catalogue as route

        def _boom(**_: object) -> CatalogueRefreshReport:
            raise CatalogueRefreshError("Catalogue refresh fetch failed (ConnectTimeout).")

        monkeypatch.setattr(route, "refresh_overlay", _boom)

        response = api_client.post(_ROUTE)
        assert response.status_code == 502
        assert response.json()["error"] == "catalogue_refresh_failed"

    def test_refresh_error_does_not_reset_catalogue(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sec_generative_search.api.routes.catalogue as route

        def _boom(**_: object) -> CatalogueRefreshReport:
            raise CatalogueRefreshError("boom")

        reset_calls = {"n": 0}
        monkeypatch.setattr(route, "refresh_overlay", _boom)
        monkeypatch.setattr(route, "reset_catalogue", lambda: reset_calls.__setitem__("n", 1))

        api_client.post(_ROUTE)
        # Fail-closed: a failed refresh never touches the live catalogue.
        assert reset_calls["n"] == 0

    def test_os_error_maps_to_500_without_path(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sec_generative_search.api.routes.catalogue as route

        def _disk(**_: object) -> CatalogueRefreshReport:
            raise OSError(f"[Errno 13] Permission denied: '{_SECRET_PATH}'")

        monkeypatch.setattr(route, "refresh_overlay", _disk)

        response = api_client.post(_ROUTE)
        assert response.status_code == 500
        assert response.json()["error"] == "catalogue_write_failed"
        assert _SECRET_PATH not in response.text


# ---------------------------------------------------------------------------
# Privacy contract
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRefreshPrivacy:
    def test_overlay_path_never_on_the_wire(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sec_generative_search.api.routes.catalogue as route

        monkeypatch.setattr(route, "refresh_overlay", lambda **_: _report())

        response = api_client.post(_ROUTE)
        assert _SECRET_PATH not in response.text
        assert set(response.json().keys()) == {
            "source",
            "source_url",
            "provider_count",
            "model_count",
        }


# ---------------------------------------------------------------------------
# Admin gate (rule A)
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRefreshAuthGate:
    _API_KEY = "shared-team-key"  # pragma: allowlist secret
    _ADMIN_KEY = "secret-admin-key"  # pragma: allowlist secret

    @pytest.fixture
    def client(self, api_client_factory, monkeypatch: pytest.MonkeyPatch) -> TestClient:
        # Patch the engine on the module so even an authorised request never
        # reaches the network. The gate tests below never get that far, but a
        # both-keys request would.
        import sec_generative_search.api.routes.catalogue as route

        monkeypatch.setattr(route, "refresh_overlay", lambda **_: _report())
        return api_client_factory(API_KEY=self._API_KEY, API_ADMIN_KEY=self._ADMIN_KEY)

    def test_no_keys_rejected(self, client: TestClient) -> None:
        response = client.post(_ROUTE)
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"

    def test_api_key_only_is_forbidden(self, client: TestClient) -> None:
        response = client.post(_ROUTE, headers={"X-API-Key": self._API_KEY})
        assert response.status_code == 403
        assert response.json()["error"] == "admin_required"

    def test_admin_key_only_surfaces_401_not_403(self, client: TestClient) -> None:
        response = client.post(_ROUTE, headers={"X-Admin-Key": self._ADMIN_KEY})
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"

    def test_both_keys_pass(self, client: TestClient) -> None:
        response = client.post(
            _ROUTE,
            headers={"X-API-Key": self._API_KEY, "X-Admin-Key": self._ADMIN_KEY},
        )
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"

    def test_open_when_no_keys_configured(self, api_client: TestClient, monkeypatch) -> None:
        import sec_generative_search.api.routes.catalogue as route

        monkeypatch.setattr(route, "refresh_overlay", lambda **_: _report())
        response = api_client.post(_ROUTE)
        assert response.status_code == 200
