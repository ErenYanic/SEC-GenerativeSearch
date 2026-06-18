"""Auth-boundary and admin-operation route sweep.

The per-surface auth tests already exercise the *helpers*:

    - :mod:`tests.api.test_auth_tiers` walks the ``API_KEY`` x
      ``API_ADMIN_KEY`` matrix against ``admin_route_dependencies()`` on a
      synthetic sentinel route.
    - :mod:`tests.api.test_dependencies` walks the resolver chain.

What no existing test does is assert the invariant **across the real,
registered route surface of the live app**. This module closes that gap
by catching a future route wired the wrong way:

    1. A new business route accidentally shipped without
       ``verify_api_key`` (silent unauthenticated surface).
    2. A new destructive route wired with ``verify_admin_key`` *alone*
       — the documented footgun.  Because ``verify_admin_key`` precedes
       no API-key check, an ``X-Admin-Key``-only request would surface
       ``403`` instead of ``401``, and (worse) an unauthenticated request
       would pass the read tier it never had.
    3. A known destructive route silently down-tiered out of the admin
       gate.

The sweep introspects each route's flattened FastAPI dependency tree
rather than re-encoding the route table by hand, so it tracks the app as
it grows. A small live-behaviour probe (:class:`TestAdminKeyOnlyProbe`)
mirrors the structural assertion against the wire so the two cannot
drift.
"""

from __future__ import annotations

import pytest
from starlette.routing import Route

# Auth dependencies are matched by callable ``__name__``, never by object
# identity (identity diverges under the full-suite import graph). The names
# are unique and stable.
#
# We read the gates from TWO sources and union them, because FastAPI moved
# where router-level ``dependencies=[...]`` surface between versions:
#   - ``route.dependencies`` — the raw declared ``Depends`` list (route +
#     router level). Populated on every version we target.
#   - ``route.dependant.dependencies`` — the solved dependant tree. On the
#     local pin (fastapi 0.135.3) the router-level gates are *also* merged
#     here; on newer FastAPI they are NOT, so reading only this attribute
#     misclassified every admin route as un-gated in CI.
# Reading both makes the sweep version-robust.
_API_KEY_DEP = "verify_api_key"  # pragma: allowlist secret
_ADMIN_KEY_DEP = "verify_admin_key"

# ---------------------------------------------------------------------------
# Route-tier introspection
# ---------------------------------------------------------------------------

# The complete set of ``/api`` routes that are deliberately reachable
# **without** an API key.  Each entry is justified — a new line here is a
# security decision that must be made consciously, which is the whole
# point of pinning the set exactly.
#
#   - /api/health                 liveness probe; tiny body; rate-exempt.
#   - /api/session*               session-cookie lifecycle.  The cookie is
#                                 the bootstrap an anonymous browser needs
#                                 *before* it can carry any other tier, so
#                                 it cannot itself be API-key gated.
#   - /api/auth/*                 user-tier auth. These carry their own
#                                 ``auth_proof`` / enrolment-token /
#                                 brute-force controls and are not part
#                                 of the infrastructure ``API_KEY`` tier.
_EXPECTED_OPEN_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/api/health"),
        ("POST", "/api/session"),
        ("POST", "/api/session/logout"),
        ("POST", "/api/session/edgar"),
        ("DELETE", "/api/session/edgar"),
        ("GET", "/api/auth/login-params"),
        ("POST", "/api/auth/login"),
        ("POST", "/api/auth/enrol"),
        ("POST", "/api/auth/password"),
        ("DELETE", "/api/auth/session"),
        ("POST", "/api/auth/vault"),
    }
)

# Destructive / privileged routes that must stay behind the admin tier.
# A regression that down-tiers any of these is caught by the inventory.
_EXPECTED_ADMIN_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/api/admin/users"),
        ("DELETE", "/api/admin/users/{user_id}"),
        ("POST", "/api/admin/users/{user_id}/unlock"),
        ("GET", "/api/providers/health"),
        ("DELETE", "/api/filings/{accession}"),
        ("POST", "/api/filings/delete-by-ids"),
        ("POST", "/api/filings/bulk-delete"),
        ("DELETE", "/api/filings/"),
        ("GET", "/api/metrics/"),
    }
)


def _declared_dependency_names(route: Route) -> list[str]:
    """Gate names from the route's raw declared ``Depends`` list, in order.

    This is the canonical, version-stable home for router-level and
    route-level ``dependencies=[...]``. ``admin_route_dependencies()``
    returns ``[Depends(verify_api_key), Depends(verify_admin_key)]`` in
    that order, so ``verify_api_key`` lands before ``verify_admin_key`` —
    reversed order would make an ``X-Admin-Key``-only request surface
    ``403`` rather than ``401``.
    """
    names: list[str] = []
    for dep in getattr(route, "dependencies", None) or []:
        fn = getattr(dep, "dependency", None)
        if fn is not None:
            names.append(getattr(fn, "__name__", ""))
    return names


def _solved_dependency_names(route: Route) -> list[str]:
    """Gate names from the solved dependant tree (pre-order), as a fallback."""
    names: list[str] = []

    def _walk(deps) -> None:
        for dep in deps:
            if dep.call is not None:
                names.append(getattr(dep.call, "__name__", ""))
            _walk(dep.dependencies)

    dependant = getattr(route, "dependant", None)
    if dependant is not None:
        _walk(dependant.dependencies)
    return names


def _ordered_gate_names(route: Route) -> list[str]:
    """Ordered auth-gate names, robust to where FastAPI surfaces them.

    Prefer whichever source actually carries a gate (both list them in
    api-before-admin order); fall back to the union so classification
    still sees a gate even on a FastAPI version that populates only one
    of the two attributes.
    """
    declared = _declared_dependency_names(route)
    if _API_KEY_DEP in declared or _ADMIN_KEY_DEP in declared:
        return declared
    solved = _solved_dependency_names(route)
    if _API_KEY_DEP in solved or _ADMIN_KEY_DEP in solved:
        return solved
    return declared + solved


def _classify(route: Route) -> str:
    names = _ordered_gate_names(route)
    if _ADMIN_KEY_DEP in names:
        return "ADMIN"
    if _API_KEY_DEP in names:
        return "API"
    return "OPEN"


def _api_routes(app) -> list[Route]:
    return [
        r for r in app.routes if isinstance(r, Route) and r.path.startswith("/api") and r.methods
    ]


# ---------------------------------------------------------------------------
# 1. Auth-tier inventory
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRouteAuthTierInventory:
    """Every ``/api`` route sits in its intended auth tier.

    These assertions are introspective (no HTTP), so they stay valid
    regardless of body shape or settings and fire the instant a new route
    lands in the wrong tier.
    """

    def test_unauthenticated_route_set_is_exactly_the_allowlist(self, api_app) -> None:
        routes = _api_routes(api_app)
        open_routes = {
            (method, r.path)
            for r in routes
            if _classify(r) == "OPEN"
            for method in (r.methods or set())
            if method not in {"HEAD", "OPTIONS"}
        }
        # Exact equality both ways catches both extra and missing routes.
        assert open_routes == set(_EXPECTED_OPEN_ROUTES), (
            "Unauthenticated /api route set drifted from the reviewed "
            f"allow-list.\n  unexpected-open: {open_routes - set(_EXPECTED_OPEN_ROUTES)}"
            f"\n  no-longer-open : {set(_EXPECTED_OPEN_ROUTES) - open_routes}"
        )

    def test_known_destructive_routes_are_admin_tier(self, api_app) -> None:
        routes = _api_routes(api_app)
        admin_routes = {
            (method, r.path)
            for r in routes
            if _classify(r) == "ADMIN"
            for method in (r.methods or set())
            if method not in {"HEAD", "OPTIONS"}
        }
        missing = set(_EXPECTED_ADMIN_ROUTES) - admin_routes
        assert not missing, f"destructive routes down-tiered out of the admin gate: {missing}"

    def test_every_admin_route_runs_api_key_before_admin_key(self, api_app) -> None:
        """Admin routes must check the API key before the admin key.

        Asserts for *every* admin-tier route that ``verify_api_key`` is
        present and ordered ahead of ``verify_admin_key`` — i.e. the
        route went through ``admin_route_dependencies()`` rather than
        wiring ``verify_admin_key`` alone.
        """
        admin_routes = [r for r in _api_routes(api_app) if _classify(r) == "ADMIN"]
        # Keep the sweep non-vacuous.
        assert len(admin_routes) >= len(_EXPECTED_ADMIN_ROUTES)

        for route in admin_routes:
            names = _ordered_gate_names(route)
            assert _API_KEY_DEP in names, (
                f"{route.path} is admin-gated but missing verify_api_key — "
                "an X-Admin-Key-only request would 403 instead of 401, and "
                "an unauthenticated request would skip the read tier."
            )
            assert names.index(_API_KEY_DEP) < names.index(_ADMIN_KEY_DEP), (
                f"{route.path} runs verify_admin_key before verify_api_key; "
                "use admin_route_dependencies() so the API tier is checked first."
            )


# ---------------------------------------------------------------------------
# 2. Live behaviour probe
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestAdminKeyOnlyProbe:
    """With both keys configured, an admin-key-only request is ``401``.

    The structural sweep proves the *wiring*; this proves the *behaviour*
    so the two cannot silently diverge. Routes chosen carry no required
    body, so the auth dependency is the only gate and the status is an
    unambiguous 401/403 (never a 422 from body validation).
    """

    _API_KEY = "shared-team-key-PROBE"  # pragma: allowlist secret
    _ADMIN_KEY = "secret-admin-key-PROBE"  # pragma: allowlist secret

    # (method, path) — body-free admin routes spanning metrics,
    # provider-health, filings, and users.
    _PROBE_ROUTES = (
        ("GET", "/api/metrics/"),
        ("GET", "/api/providers/health"),
        ("DELETE", "/api/filings/0000320193-23-000077"),
        ("POST", "/api/admin/users/some-user-id/unlock"),
    )

    @pytest.fixture
    def client(self, api_client_factory):
        return api_client_factory(API_KEY=self._API_KEY, API_ADMIN_KEY=self._ADMIN_KEY)

    @pytest.mark.parametrize(("method", "path"), _PROBE_ROUTES)
    def test_no_header_returns_401_not_403(self, client, method: str, path: str) -> None:
        # Neither header supplied.  The API tier runs first and rejects
        # with 401 — the caller must never learn the admin gate even
        # exists until they clear the read tier.  This is the case whose
        # status code *flips* (401 -> 403) if the admin gate is ever
        # ordered ahead of the API-key gate, so it is the behavioural
        # mirror of the structural ordering lock above.
        response = client.request(method, path)
        assert response.status_code == 401, (
            f"{method} {path}: no-header request must surface 401 (API tier first), "
            f"got {response.status_code}"
        )
        assert response.json()["error"] == "unauthorised"

    @pytest.mark.parametrize(("method", "path"), _PROBE_ROUTES)
    def test_admin_key_only_returns_401_not_403(self, client, method: str, path: str) -> None:
        # X-Admin-Key supplied, X-API-Key omitted.
        response = client.request(method, path, headers={"X-Admin-Key": self._ADMIN_KEY})
        assert response.status_code == 401, (
            f"{method} {path}: admin-key-only must surface 401 (API tier first), "
            f"got {response.status_code}"
        )
        assert response.json()["error"] == "unauthorised"

    @pytest.mark.parametrize(("method", "path"), _PROBE_ROUTES)
    def test_api_key_only_returns_403_admin_required(self, client, method: str, path: str) -> None:
        # X-API-Key supplied, X-Admin-Key omitted.
        response = client.request(method, path, headers={"X-API-Key": self._API_KEY})
        assert response.status_code == 403, (
            f"{method} {path}: api-key-only must surface 403 admin_required, "
            f"got {response.status_code}"
        )
        assert response.json()["error"] == "admin_required"


# ---------------------------------------------------------------------------
# 3. Secret handling
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestConfiguredSecretNeverInResponseBody:
    """No configured infrastructure secret is ever echoed to the wire.

    A denial or validation envelope must never carry the server's
    configured ``API_KEY`` / ``API_ADMIN_KEY``, nor the wrong credential
    the caller supplied.
    """

    _API_KEY = "configured-api-key-SENTINEL-A"  # pragma: allowlist secret
    _ADMIN_KEY = "configured-admin-key-SENTINEL-B"  # pragma: allowlist secret
    _WRONG_KEY = "supplied-wrong-key-SENTINEL-C"  # pragma: allowlist secret
    _PROVIDER_KEY = "sk-supplied-provider-SENTINEL-D"  # pragma: allowlist secret

    @pytest.fixture
    def client(self, api_client_factory):
        return api_client_factory(API_KEY=self._API_KEY, API_ADMIN_KEY=self._ADMIN_KEY)

    def test_no_configured_or_supplied_secret_appears_in_any_body(self, client) -> None:
        bodies: list[str] = []

        # 401: missing key on a read-tier route.
        bodies.append(client.get("/api/status/").text)
        # 401: wrong key on a read-tier route.
        bodies.append(client.get("/api/status/", headers={"X-API-Key": self._WRONG_KEY}).text)
        # 403: api-key valid, admin key missing on a destructive route.
        bodies.append(
            client.delete(
                "/api/filings/0000320193-23-000077",
                headers={"X-API-Key": self._API_KEY},
            ).text
        )
        # 403: api-key valid, admin key wrong on a destructive route.
        bodies.append(
            client.delete(
                "/api/filings/0000320193-23-000077",
                headers={"X-API-Key": self._API_KEY, "X-Admin-Key": self._WRONG_KEY},
            ).text
        )
        # Validate route: a supplied provider key must never be echoed back.
        bodies.append(
            client.post(
                "/api/providers/validate",
                headers={"X-API-Key": self._API_KEY},
                json={"provider": "openai", "api_key": self._PROVIDER_KEY},
            ).text
        )

        joined = "\n".join(bodies)
        for secret in (self._API_KEY, self._ADMIN_KEY, self._WRONG_KEY, self._PROVIDER_KEY):
            assert secret not in joined, f"secret leaked into a response body: {secret!r}"
