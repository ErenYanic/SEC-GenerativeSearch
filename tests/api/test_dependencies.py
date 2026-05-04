"""Security tests for the ``Depends()`` providers and resolver factory.

10A.2 surface: ``verify_api_key``, ``verify_admin_key``, ``is_admin_request``,
``request_scoped_resolver``.  Phase-9 wiring is exercised end-to-end at
the seam where the API consumes :func:`build_llm_provider` —
``request_scoped_resolver`` builds the chain a route would pass.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request

from sec_generative_search.api.dependencies import (
    ADMIN_USER_ID,
    SESSION_COOKIE_NAME,
    is_admin_request,
    request_scoped_resolver,
)
from sec_generative_search.core.credentials import InMemorySessionCredentialStore


def _request(*, headers: dict[str, str] | None = None, cookies: dict[str, str] | None = None):
    """Build a minimal FastAPI Request with the given headers + cookies."""
    raw_headers: list[tuple[bytes, bytes]] = []
    for name, value in (headers or {}).items():
        raw_headers.append((name.lower().encode(), value.encode()))
    if cookies:
        cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
        raw_headers.append((b"cookie", cookie_str.encode()))
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
class TestIsAdminRequest:
    def test_no_admin_key_configured_returns_true(self, monkeypatch) -> None:
        monkeypatch.delenv("API_ADMIN_KEY", raising=False)
        from sec_generative_search.config.settings import reload_settings

        reload_settings()
        request = _request()
        assert is_admin_request(request) is True

    def test_admin_key_required_and_provided(self, monkeypatch) -> None:
        admin_secret = "super-admin-key-1234"  # pragma: allowlist secret
        monkeypatch.setenv("API_ADMIN_KEY", admin_secret)
        from sec_generative_search.config.settings import reload_settings

        reload_settings()
        request = _request(headers={"X-Admin-Key": admin_secret})
        assert is_admin_request(request) is True

    def test_admin_key_required_and_wrong(self, monkeypatch) -> None:
        monkeypatch.setenv("API_ADMIN_KEY", "real-admin-key")  # pragma: allowlist secret
        from sec_generative_search.config.settings import reload_settings

        reload_settings()
        request = _request(headers={"X-Admin-Key": "wrong"})
        assert is_admin_request(request) is False

    def test_admin_key_required_and_missing(self, monkeypatch) -> None:
        monkeypatch.setenv("API_ADMIN_KEY", "real-admin-key")  # pragma: allowlist secret
        from sec_generative_search.config.settings import reload_settings

        reload_settings()
        request = _request()
        assert is_admin_request(request) is False


@pytest.mark.security
class TestRequestScopedResolverChain:
    def _attach_state(
        self,
        request: Request,
        *,
        session_store=None,
        encrypted_store=None,
        admin_key: str | None = None,
    ) -> None:
        """Wire a tiny stand-in app onto the request for the resolver."""
        app = FastAPI()
        app.state.session_store = session_store
        app.state.encrypted_credential_store = encrypted_store
        # The resolver reads ``request.app.state``; Starlette's Request
        # discovers ``app`` via ``scope['app']``.
        request.scope["app"] = app
        if admin_key is not None:
            import os

            os.environ["API_ADMIN_KEY"] = admin_key
            from sec_generative_search.config.settings import reload_settings

            reload_settings()

    def test_chain_falls_through_to_admin_env(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-admin-env-1234")  # pragma: allowlist secret
        request = _request()
        self._attach_state(request)
        resolver = request_scoped_resolver(request)
        assert resolver("openai") == "sk-admin-env-1234"  # pragma: allowlist secret
        # Unknown provider with no env var → None.
        assert resolver("nonexistent-provider") is None

    def test_session_store_shadows_admin_env(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-admin-env")  # pragma: allowlist secret
        store = InMemorySessionCredentialStore(ttl_seconds=300)
        sid = "A" * 43  # valid shape per ``extract_session_id``
        store.set(sid, "openai", "sk-session-key-from-store")  # pragma: allowlist secret

        request = _request(cookies={SESSION_COOKIE_NAME: sid})
        self._attach_state(request, session_store=store)

        resolver = request_scoped_resolver(request)
        # Session tier wins over admin-env tier.
        assert resolver("openai") == "sk-session-key-from-store"  # pragma: allowlist secret

    def test_forged_cookie_falls_through(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-admin")  # pragma: allowlist secret
        store = InMemorySessionCredentialStore(ttl_seconds=300)
        # A forged but syntactically invalid cookie — must NOT key the store.
        request = _request(cookies={SESSION_COOKIE_NAME: "tooshort"})
        self._attach_state(request, session_store=store)

        resolver = request_scoped_resolver(request)
        # Falls through to admin env.
        assert resolver("openai") == "sk-admin"  # pragma: allowlist secret

    def test_encrypted_tier_only_for_admin(self, monkeypatch) -> None:
        # Stand-in store: ``CredentialStore`` Protocol is structural.
        class StubEncrypted:
            def get(self, key_id: str, provider: str) -> str | None:
                return "sk-encrypted-admin" if key_id == ADMIN_USER_ID else None

            def set(self, *a, **kw) -> None: ...
            def delete(self, *a, **kw) -> bool:
                return False

            def list_providers(self, *a, **kw) -> set:
                return set()

            def clear(self, *a, **kw) -> int:
                return 0

        admin_secret = "real-admin-key"  # pragma: allowlist secret
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Non-admin request — encrypted tier MUST be skipped.
        non_admin = _request()
        self._attach_state(
            non_admin,
            encrypted_store=StubEncrypted(),
            admin_key=admin_secret,
        )
        resolver = request_scoped_resolver(non_admin)
        assert resolver("openai") is None  # encrypted tier was skipped

        # Admin request — encrypted tier MUST be consulted.
        admin = _request(headers={"X-Admin-Key": admin_secret})
        self._attach_state(
            admin,
            encrypted_store=StubEncrypted(),
            admin_key=admin_secret,
        )
        resolver = request_scoped_resolver(admin)
        assert resolver("openai") == "sk-encrypted-admin"  # pragma: allowlist secret
