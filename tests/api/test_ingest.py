"""Tests for the ingestion routes.

Strategy
--------

The ``TaskManager`` is itself covered by :mod:`tests.api.test_task_manager`;
this file focuses on the route contract over a *stub* manager:

    - Schema validation: ticker / form-type alphabets, count bounds,
      single-vs-batch invariant on ``/add``.
    - EDGAR resolver chain integration: 401 ``edgar_identity_required``
      when ``API_EDGAR_SESSION_REQUIRED=true`` and no header / session
      identity is supplied.
    - Session-scoped ownership: foreign-session task lookups / cancels
      surface as ``404 not_found`` (not ``403``) so the route does not
      confirm existence to non-owners.
    - Audit-log discipline: every create / cancel emits a
      ``SECURITY_AUDIT:`` line; the line never carries names, emails,
      provider keys, or the raw cookie value.
    - Rate-limit bucket: the ``ingest`` category is wired in
      ``ROUTE_POLICIES`` and triggers ``429`` at the documented per-IP
      rpm cap.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.api.app import create_app
from sec_generative_search.api.tasks import (
    FilingResult,
    TaskInfo,
    TaskProgress,
    TaskQueueFullError,
    TaskState,
)
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import (
    EdgarIdentity,
    InMemorySessionEdgarIdentityStore,
)
from sec_generative_search.core.logging import LOGGER_NAME

# ---------------------------------------------------------------------------
# Stub TaskManager
# ---------------------------------------------------------------------------


@dataclass
class _StubTaskManager:
    """In-process stub for the route surface only.

    Records every ``create_task`` call so tests can assert that EDGAR
    identity is captured into the per-task resolver closure (and never
    exposed on the wire). Lookups are served from an in-memory dict so
    tests can plant tasks with arbitrary owners.
    """

    tasks: dict[str, TaskInfo] = field(default_factory=dict)
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    cancel_calls: list[str] = field(default_factory=list)
    raise_queue_full: bool = False
    cancel_succeeds: bool = True

    def create_task(self, **kwargs: Any) -> str:
        # Capture the identity by invoking the resolver immediately —
        # the route guarantees a captured closure, not a live lookup.
        resolver = kwargs.pop("edgar_identity_resolver", None)
        resolved_identity = resolver() if resolver is not None else None
        self.create_calls.append({**kwargs, "_resolved_identity": resolved_identity})

        if self.raise_queue_full:
            raise TaskQueueFullError(active=5, maximum=5)

        task_id = f"{len(self.create_calls):032x}"
        info = TaskInfo(
            task_id=task_id,
            tickers=list(kwargs.get("tickers", [])),
            form_types=list(kwargs.get("form_types", [])),
            count_mode=kwargs.get("count_mode", "latest"),
            count=kwargs.get("count"),
            year=kwargs.get("year"),
            start_date=kwargs.get("start_date"),
            end_date=kwargs.get("end_date"),
            session_id=kwargs.get("session_id"),
        )
        info.state = TaskState.PENDING
        self.tasks[task_id] = info
        return task_id

    def get_task(self, task_id: str) -> TaskInfo | None:
        return self.tasks.get(task_id)

    def list_tasks_for_session(self, session_id: str | None) -> list[TaskInfo]:
        return [t for t in self.tasks.values() if t.session_id == session_id]

    def cancel_task(self, task_id: str) -> bool:
        self.cancel_calls.append(task_id)
        return self.cancel_succeeds


def _plant_completed_task(
    manager: _StubTaskManager,
    *,
    task_id: str | None = None,
    session_id: str | None = None,
    state: TaskState = TaskState.COMPLETED,
) -> str:
    """Insert a pre-built task so list / get / cancel paths can read it."""
    tid = task_id or f"{len(manager.tasks) + 1:032x}"
    info = TaskInfo(
        task_id=tid,
        tickers=["AAPL"],
        form_types=["10-K"],
        session_id=session_id,
    )
    info.state = state
    info.progress = TaskProgress(
        filings_done=1,
        filings_total=1,
    )
    info.results = [
        FilingResult(
            ticker="AAPL",
            form_type="10-K",
            filing_date="2023-09-30",
            accession_number="0000320193-23-000077",
            segment_count=10,
            chunk_count=42,
            duration_seconds=1.5,
        )
    ]
    if state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}:
        info.started_at = datetime.now(UTC)
        info.completed_at = datetime.now(UTC)
    manager.tasks[tid] = info
    return tid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_ingest_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip API_*/EDGAR_* env vars so each test sees a clean baseline."""
    for key in list(os.environ.keys()):
        if key.startswith("API_") or key.startswith("EDGAR_"):
            monkeypatch.delenv(key, raising=False)
    # Admin-env identity is the fallback the routes consult in the
    # default (Scenario-A) configuration; set it once so the resolver
    # chain has an admin tier to land on.
    monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Test User")
    monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "test@example.com")
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def ingest_app_factory(monkeypatch: pytest.MonkeyPatch):
    """Build a fresh app with a stub TaskManager attached to ``app.state``."""

    def factory(
        *,
        env: dict[str, str] | None = None,
    ) -> tuple[Any, _StubTaskManager]:
        if env:
            for key, value in env.items():
                monkeypatch.setenv(key, value)
        reload_settings()

        app = create_app()
        manager = _StubTaskManager()
        app.state.task_manager = manager
        app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
        app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
        app.state.encrypted_credential_store = None
        return app, manager

    return factory


def _mint_session(client: TestClient) -> str:
    """Mint a server-side session cookie and return its value."""
    response = client.post("/api/session")
    assert response.status_code in (200, 201)
    cookie = client.cookies.get("sec_rag_session")
    assert cookie is not None
    return cookie


# ---------------------------------------------------------------------------
# Create-path tests
# ---------------------------------------------------------------------------


class TestIngestCreate:
    def test_add_starts_single_ticker_task(self, ingest_app_factory) -> None:
        app, manager = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/ingest/add",
            json={"tickers": ["aapl"], "form_types": ["10-K"]},
        )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "pending"
        assert re.fullmatch(r"[0-9a-f]{32}", body["task_id"])
        assert body["websocket_url"] == f"/ws/ingest/{body['task_id']}"

        assert len(manager.create_calls) == 1
        call = manager.create_calls[0]
        # Ticker is upper-cased at the schema boundary.
        assert call["tickers"] == ["AAPL"]
        # EDGAR identity was captured into the per-task resolver — the
        # closure resolves to the admin-env fallback in Scenario A.
        assert isinstance(call["_resolved_identity"], EdgarIdentity)
        assert call["_resolved_identity"].name == "Test User"

    def test_add_rejects_multi_ticker_payload(self, ingest_app_factory) -> None:
        app, manager = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/ingest/add",
            json={"tickers": ["AAPL", "MSFT"], "form_types": ["10-K"]},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "validation_error"
        assert manager.create_calls == []

    def test_batch_accepts_multi_ticker_payload(self, ingest_app_factory) -> None:
        app, manager = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/ingest/batch",
            json={"tickers": ["AAPL", "MSFT"], "form_types": ["10-K", "10-Q"]},
        )
        assert response.status_code == 202
        call = manager.create_calls[0]
        assert call["tickers"] == ["AAPL", "MSFT"]
        assert call["form_types"] == ["10-K", "10-Q"]

    def test_invalid_ticker_alphabet_rejected_at_schema(self, ingest_app_factory) -> None:
        app, _ = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/ingest/add",
            json={"tickers": ["AAPL; DROP"], "form_types": ["10-K"]},
        )
        assert response.status_code == 422

    def test_unknown_count_mode_rejected_at_schema(self, ingest_app_factory) -> None:
        app, _ = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/ingest/add",
            json={
                "tickers": ["AAPL"],
                "form_types": ["10-K"],
                "count_mode": "BOGUS",
            },
        )
        assert response.status_code == 422

    def test_count_bound_enforced(self, ingest_app_factory) -> None:
        app, _ = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/ingest/batch",
            json={
                "tickers": ["AAPL"],
                "form_types": ["10-K"],
                "count": 9999,
            },
        )
        assert response.status_code == 422

    def test_max_tickers_per_request_cap(self, ingest_app_factory) -> None:
        app, _ = ingest_app_factory(env={"API_MAX_TICKERS_PER_REQUEST": "1"})
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/ingest/batch",
            json={"tickers": ["AAPL", "MSFT"], "form_types": ["10-K"]},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "request_cap_exceeded"

    def test_max_filings_per_request_cap(self, ingest_app_factory) -> None:
        app, _ = ingest_app_factory(env={"API_MAX_FILINGS_PER_REQUEST": "2"})
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/ingest/add",
            json={
                "tickers": ["AAPL"],
                "form_types": ["10-K"],
                "count": 50,
            },
        )
        assert response.status_code == 400
        assert response.json()["error"] == "request_cap_exceeded"

    def test_queue_full_surfaces_429(self, ingest_app_factory) -> None:
        app, manager = ingest_app_factory()
        manager.raise_queue_full = True
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/ingest/add",
            json={"tickers": ["AAPL"], "form_types": ["10-K"]},
        )
        assert response.status_code == 429
        body = response.json()
        assert body["error"] == "queue_full"
        assert body["details"] == {"active": 5, "maximum": 5}


# ---------------------------------------------------------------------------
# EDGAR identity integration
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestEdgarIdentityGate:
    def test_session_required_blocks_without_identity(self, ingest_app_factory) -> None:
        app, manager = ingest_app_factory(env={"API_EDGAR_SESSION_REQUIRED": "true"})
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/ingest/add",
            json={"tickers": ["AAPL"], "form_types": ["10-K"]},
        )
        assert response.status_code == 401
        assert response.json()["error"] == "edgar_identity_required"
        assert manager.create_calls == []

    def test_session_required_allows_with_header_identity(self, ingest_app_factory) -> None:
        app, manager = ingest_app_factory(env={"API_EDGAR_SESSION_REQUIRED": "true"})
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/ingest/add",
            json={"tickers": ["AAPL"], "form_types": ["10-K"]},
            headers={
                "X-Edgar-Name": "Alice Example",
                "X-Edgar-Email": "alice@example.com",
            },
        )
        assert response.status_code == 202
        captured = manager.create_calls[0]["_resolved_identity"]
        assert isinstance(captured, EdgarIdentity)
        assert captured.name == "Alice Example"
        assert captured.email == "alice@example.com"

    def test_session_required_allows_with_registered_session(self, ingest_app_factory) -> None:
        app, manager = ingest_app_factory(env={"API_EDGAR_SESSION_REQUIRED": "true"})
        client = TestClient(app, base_url="https://testserver")

        session_id = _mint_session(client)
        # Register a per-session identity through the canonical route.
        reg = client.post(
            "/api/session/edgar",
            json={"name": "Bob Owner", "email": "bob@example.com"},
        )
        assert reg.status_code in (200, 201)

        response = client.post(
            "/api/ingest/add",
            json={"tickers": ["AAPL"], "form_types": ["10-K"]},
        )
        assert response.status_code == 202
        captured = manager.create_calls[0]["_resolved_identity"]
        assert captured.email == "bob@example.com"
        # The route captured the *current* session_id for ownership.
        assert manager.create_calls[0]["session_id"] == session_id


# ---------------------------------------------------------------------------
# Session-scoped ownership
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestOwnership:
    def test_list_returns_only_own_session_tasks(self, ingest_app_factory) -> None:
        app, manager = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")

        session_id = _mint_session(client)
        own_id = _plant_completed_task(manager, session_id=session_id)
        _plant_completed_task(manager, session_id="other-session")

        response = client.get("/api/ingest/tasks")
        assert response.status_code == 200
        body = response.json()
        ids = [t["task_id"] for t in body["tasks"]]
        assert ids == [own_id]
        assert body["total"] == 1

    def test_get_foreign_task_returns_404(self, ingest_app_factory) -> None:
        app, manager = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")

        _mint_session(client)
        foreign_id = _plant_completed_task(manager, session_id="other-session")

        response = client.get(f"/api/ingest/tasks/{foreign_id}")
        # Foreign tasks surface as 404 (not 403) so the route does not
        # confirm the existence of a task to a non-owner.
        assert response.status_code == 404
        assert response.json()["error"] == "not_found"

    def test_cancel_foreign_task_returns_404(self, ingest_app_factory) -> None:
        app, manager = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")

        _mint_session(client)
        foreign_id = _plant_completed_task(
            manager,
            session_id="other-session",
            state=TaskState.RUNNING,
        )

        response = client.delete(f"/api/ingest/tasks/{foreign_id}")
        assert response.status_code == 404
        # No cancel signal sent — the route rejected before reaching the
        # manager's cancel_task call site.
        assert manager.cancel_calls == []

    def test_cancel_already_terminal_returns_409(self, ingest_app_factory) -> None:
        app, manager = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")

        session_id = _mint_session(client)
        own_id = _plant_completed_task(manager, session_id=session_id)
        manager.cancel_succeeds = False

        response = client.delete(f"/api/ingest/tasks/{own_id}")
        assert response.status_code == 409
        assert response.json()["error"] == "conflict"

    def test_malformed_task_id_path_returns_422(self, ingest_app_factory) -> None:
        app, _ = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.get("/api/ingest/tasks/not-a-task-id")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Audit-log discipline
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestAuditLogDiscipline:
    """Audit-log lines NEVER carry credential-shaped or PII values."""

    def test_create_audit_line_omits_pii(
        self,
        ingest_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        app, _ = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")

        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            # Force log propagation so caplog sees the audit lines.
            logger = logging.getLogger(LOGGER_NAME)
            previous_propagate = logger.propagate
            logger.propagate = True
            try:
                response = client.post(
                    "/api/ingest/add",
                    json={"tickers": ["AAPL"], "form_types": ["10-K"]},
                    headers={
                        "X-Edgar-Name": "Alice Example",
                        "X-Edgar-Email": "alice@example.com",
                        "X-Provider-Key-openai": "sk-supersecret",  # pragma: allowlist secret
                    },
                )
            finally:
                logger.propagate = previous_propagate
        assert response.status_code == 202

        audit_lines = [r.message for r in caplog.records if "SECURITY_AUDIT" in r.message]
        joined = "\n".join(audit_lines)
        # PII / secrets MUST NOT appear in any audit line.
        assert "Alice Example" not in joined
        assert "alice@example.com" not in joined
        assert "sk-supersecret" not in joined
        # The ticker IS metadata-only and IS allowed to appear.
        assert any("ingest_task_created" in line for line in audit_lines)

    def test_cancel_audit_line_emits(
        self,
        ingest_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        app, manager = ingest_app_factory()
        client = TestClient(app, base_url="https://testserver")
        session_id = _mint_session(client)
        own_id = _plant_completed_task(
            manager,
            session_id=session_id,
            state=TaskState.RUNNING,
        )

        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            logger = logging.getLogger(LOGGER_NAME)
            previous_propagate = logger.propagate
            logger.propagate = True
            try:
                response = client.delete(f"/api/ingest/tasks/{own_id}")
            finally:
                logger.propagate = previous_propagate

        assert response.status_code == 200
        audit_lines = [r.message for r in caplog.records if "SECURITY_AUDIT" in r.message]
        assert any("ingest_task_cancelled" in line for line in audit_lines)
        # The raw task id is not echoed — only the masked tail (last 4).
        assert not any(own_id in line for line in audit_lines)


# ---------------------------------------------------------------------------
# Rate-limit bucket wiring
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestIngestRateLimit:
    def test_per_ip_bucket_fires(self, ingest_app_factory) -> None:
        # Drop the per-IP cap to 1 so we can prove the bucket exists.
        app, _ = ingest_app_factory(env={"API_RATE_LIMIT_INGEST": "1"})
        client = TestClient(app, base_url="https://testserver")

        body = {"tickers": ["AAPL"], "form_types": ["10-K"]}
        first = client.post("/api/ingest/add", json=body)
        assert first.status_code == 202
        second = client.post("/api/ingest/add", json=body)
        assert second.status_code == 429
        envelope = second.json()
        assert envelope["error"] == "rate_limited"
        assert envelope["details"]["category"] == "ingest"
