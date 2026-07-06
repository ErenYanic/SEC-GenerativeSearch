"""Cross-surface error-consistency locks for the RAG routes.

The three RAG surfaces — ``POST /api/rag/plan`` (HTTP), ``/query`` (HTTP),
and ``/stream`` (SSE) — classify an upstream provider / generation failure
through **one** shared ladder (``_PROVIDER_ERROR_LADDER`` in
:mod:`sec_generative_search.api.routes.rag`).  The ladder used to be copied
inline into all three handlers; centralising it removes that drift hazard.

These tests are the regression lock for that consolidation: a caller that
falls back from ``/query`` to ``/stream`` (or inspects ``/plan``) for the
same key MUST see the same ``error`` code, wording, and hint.  They assert
the surfaces *agree with each other* (drift guard) **and** match the exact
expected wording (non-vacuous), so re-inlining one surface with a reworded
copy fails here even if that surface's own per-route tests still pass.

Strategy mirrors :mod:`tests.api.test_rag_query` /
:mod:`tests.api.test_rag_stream`: hermetic in-process stubs, no real LLM /
ChromaDB / embedder.  A single app serves all three routes; the same
exception instance is injected into ``understand_query`` (plan) and the
orchestrator's ``generate`` / ``generate_stream`` (query / stream).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.api.app import create_app
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import InMemorySessionEdgarIdentityStore
from sec_generative_search.core.exceptions import (
    GenerationError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.rag.orchestrator import StreamEvent
from sec_generative_search.rag.query_understanding import QueryPlan

# ---------------------------------------------------------------------------
# Expected envelope wording (the single source of truth these tests pin).
# ``message`` carries the ``{phase}`` the shared ladder fills — "generation"
# on /query + /stream, "query understanding" on /plan.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ExpectedEnvelope:
    status_code: int
    error: str
    message: str  # already phase-substituted for the generation surfaces
    hint: str


_AUTH = _ExpectedEnvelope(
    status_code=401,
    error="provider_unauthorized",
    message="The upstream provider rejected the supplied API key.",
    hint="Verify or rotate the provider key (header / session / admin env).",
)
_TRANSIENT = _ExpectedEnvelope(
    status_code=503,
    error="provider_unavailable",
    message="The upstream provider is rate-limited or timed out.",
    hint="Retry after a short backoff; do not rotate the key.",
)
_CONNECTION = _ExpectedEnvelope(
    status_code=503,
    error="provider_unavailable",
    message="The upstream provider endpoint could not be reached.",
    hint=(
        "Verify the endpoint is running and reachable (for local_llm, "
        "that the local model server is up); retry once it recovers."
    ),
)
_PROVIDER_GENERATION = _ExpectedEnvelope(
    status_code=502,
    error="provider_error",
    message="The upstream provider returned an error during generation.",
    hint="Inspect the audit log; do not rotate the key on a non-auth error.",
)
_GENERATION = _ExpectedEnvelope(
    status_code=502,
    error="generation_error",
    message="The orchestrator could not assemble a valid answer.",
    hint="Retry the request; if the failure persists, switch model or provider.",
)


# ``(exc_factory, expected)`` for the generation surfaces (/query + /stream).
_GENERATION_CASES: list[tuple[str, Any, _ExpectedEnvelope]] = [
    ("auth", lambda: ProviderAuthError("rejected"), _AUTH),
    ("rate_limit", lambda: ProviderRateLimitError("limited"), _TRANSIENT),
    ("timeout", lambda: ProviderTimeoutError("timed out"), _TRANSIENT),
    ("connection", lambda: ProviderConnectionError("unreachable"), _CONNECTION),
    ("provider", lambda: ProviderError("boom"), _PROVIDER_GENERATION),
    ("generation", lambda: GenerationError("bad payload"), _GENERATION),
]


# ---------------------------------------------------------------------------
# In-process stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubLLM:
    provider_name: str = "openai"


@dataclass
class _StubBuild:
    """Patched ``build_llm_provider`` — always succeeds (never the error under test)."""

    def __call__(self, provider_name: str, *, api_key_resolver: Any) -> _StubLLM:
        api_key_resolver(provider_name)
        return _StubLLM(provider_name=provider_name)


@dataclass
class _StubUnderstand:
    """Patched ``understand_query`` — raises the injected exception."""

    raise_with: Exception

    def __call__(self, query: str, *, llm: Any, model: str, **_: Any) -> QueryPlan:
        raise self.raise_with


@dataclass
class _DualOrchestrator:
    """Stand-in for :class:`RAGOrchestrator` — both entry points raise.

    ``generate`` raises directly; ``generate_stream`` is a generator that
    yields nothing then raises, mimicking an LLM that fails at the opening
    of the stream (the producer thread routes it to an ``error`` event).
    """

    retrieval: Any = None
    llm: Any = None
    raise_with: Exception | None = None

    def generate(self, plan: QueryPlan, **_: Any) -> Any:
        assert self.raise_with is not None
        raise self.raise_with

    def generate_stream(self, plan: QueryPlan, **_: Any) -> Iterator[StreamEvent]:
        assert self.raise_with is not None
        yield from ()
        raise self.raise_with


_LATEST_ORCH: _DualOrchestrator | None = None


def _orchestrator_factory(*, retrieval: Any, llm: Any, **_: Any) -> _DualOrchestrator:
    assert _LATEST_ORCH is not None, "test forgot to install a stub orchestrator"
    _LATEST_ORCH.retrieval = retrieval
    _LATEST_ORCH.llm = llm
    return _LATEST_ORCH


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_rag_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in list(os.environ.keys()):
        if key.startswith(("API_", "LLM_")):
            monkeypatch.delenv(key, raising=False)
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def rag_app_factory(monkeypatch: pytest.MonkeyPatch):
    """Build one app whose plan + query + stream paths all raise ``raise_with``."""

    def factory(raise_with: Exception) -> Any:
        reload_settings()

        global _LATEST_ORCH
        _LATEST_ORCH = _DualOrchestrator(raise_with=raise_with)

        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.build_llm_provider",
            _StubBuild(),
        )
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.understand_query",
            _StubUnderstand(raise_with=raise_with),
        )
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.RAGOrchestrator",
            _orchestrator_factory,
        )

        app = create_app()
        app.state.retrieval_service = object()
        app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
        app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
        app.state.encrypted_credential_store = None
        return app

    return factory


_HEADERS = {"X-Provider-Key-openai": "sk-key-1234"}  # pragma: allowlist secret


def _plan_payload() -> dict[str, Any]:
    return {
        "raw_query": "What are AAPL's iPhone segment risks?",
        "detected_language": "en",
        "query_en": "What are AAPL's iPhone segment risks?",
        "tickers": ["AAPL"],
        "form_types": ["10-K"],
        "date_range": ["2023-01-01", "2024-01-01"],
        "intent": "Identify risks for the iPhone segment",
        "suggested_answer_mode": "analytical",
    }


def _stream_error_frame(raw_text: str) -> dict:
    """Return the ``data`` payload of the SSE ``error`` frame (asserts one exists)."""
    for block in raw_text.split("\n\n"):
        lines = block.strip("\n").split("\n")
        if any(line == "event: error" for line in lines):
            data_line = next(line for line in lines if line.startswith("data: "))
            return json.loads(data_line[len("data: ") :])
    raise AssertionError(f"no SSE error frame in body: {raw_text!r}")


# ---------------------------------------------------------------------------
# /query (HTTP) vs /stream (SSE): identical classification
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestQueryStreamErrorConsistency:
    """The HTTP raise path and the SSE payload builder share one table."""

    @pytest.mark.parametrize(
        ("name", "exc_factory", "expected"),
        _GENERATION_CASES,
        ids=[c[0] for c in _GENERATION_CASES],
    )
    def test_query_and_stream_agree(
        self,
        rag_app_factory,
        name: str,
        exc_factory: Any,
        expected: _ExpectedEnvelope,
    ) -> None:
        # --- /query: envelope arrives on the HTTP status ---
        app_q = rag_app_factory(exc_factory())
        client_q = TestClient(app_q, base_url="https://testserver")
        resp_q = client_q.post(
            "/api/rag/query",
            json={"plan": _plan_payload()},
            headers=_HEADERS,
        )
        assert resp_q.status_code == expected.status_code, name
        env_q = resp_q.json()

        # --- /stream: same envelope arrives as an in-stream error frame ---
        app_s = rag_app_factory(exc_factory())
        client_s = TestClient(app_s, base_url="https://testserver")
        resp_s = client_s.post(
            "/api/rag/stream",
            json={"plan": _plan_payload()},
            headers=_HEADERS,
        )
        # The SSE response opened (200) before the failure; the error is a
        # frame, never an HTTP status.
        assert resp_s.status_code == 200, name
        env_s = _stream_error_frame(resp_s.text)

        # Non-vacuous: each surface matches the pinned wording exactly ...
        assert (env_q["error"], env_q["message"], env_q["hint"]) == (
            expected.error,
            expected.message,
            expected.hint,
        ), name
        # ... and the two surfaces agree with each other (the drift guard).
        assert env_q["error"] == env_s["error"], name
        assert env_q["message"] == env_s["message"], name
        assert env_q["hint"] == env_s["hint"], name

    def test_provider_error_details_scoped_to_http_surface(self, rag_app_factory) -> None:
        # The generic-provider rung attaches ``{provider, kind}`` details on
        # the HTTP envelope only; the SSE ``error`` frame carries no details.
        app_q = rag_app_factory(ProviderError("boom"))
        env_q = (
            TestClient(app_q, base_url="https://testserver")
            .post("/api/rag/query", json={"plan": _plan_payload()}, headers=_HEADERS)
            .json()
        )
        assert env_q["details"] == {"provider": "openai", "kind": "ProviderError"}

        app_s = rag_app_factory(ProviderError("boom"))
        resp_s = TestClient(app_s, base_url="https://testserver").post(
            "/api/rag/stream", json={"plan": _plan_payload()}, headers=_HEADERS
        )
        assert _stream_error_frame(resp_s.text)["details"] is None


# ---------------------------------------------------------------------------
# /plan (HTTP): shares the same ladder rungs
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestPlanErrorConsistency:
    """``/plan`` consumes the same ladder; only the phase noun differs."""

    @pytest.mark.parametrize(
        ("name", "exc_factory", "expected"),
        [
            ("auth", lambda: ProviderAuthError("rejected"), _AUTH),
            ("rate_limit", lambda: ProviderRateLimitError("limited"), _TRANSIENT),
            ("timeout", lambda: ProviderTimeoutError("timed out"), _TRANSIENT),
            ("connection", lambda: ProviderConnectionError("unreachable"), _CONNECTION),
        ],
        ids=["auth", "rate_limit", "timeout", "connection"],
    )
    def test_plan_matches_phaseless_rungs(
        self,
        rag_app_factory,
        name: str,
        exc_factory: Any,
        expected: _ExpectedEnvelope,
    ) -> None:
        # The auth / transient / connection rungs carry no ``{phase}`` — so
        # /plan's envelope is byte-identical to the generation surfaces'.
        app = rag_app_factory(exc_factory())
        resp = TestClient(app, base_url="https://testserver").post(
            "/api/rag/plan", json={"query": "any"}, headers=_HEADERS
        )
        assert resp.status_code == expected.status_code, name
        env = resp.json()
        assert (env["error"], env["message"], env["hint"]) == (
            expected.error,
            expected.message,
            expected.hint,
        ), name

    def test_plan_provider_error_swaps_phase_noun_only(self, rag_app_factory) -> None:
        # The generic-provider rung names the failing stage: /plan reports
        # "query understanding" where /query + /stream report "generation".
        # Same error code + hint, phase-swapped message — the one sanctioned
        # divergence across the surfaces.
        app = rag_app_factory(ProviderError("boom"))
        env = (
            TestClient(app, base_url="https://testserver")
            .post("/api/rag/plan", json={"query": "any"}, headers=_HEADERS)
            .json()
        )
        assert env["error"] == _PROVIDER_GENERATION.error
        assert env["hint"] == _PROVIDER_GENERATION.hint
        assert (
            env["message"] == "The upstream provider returned an error during query understanding."
        )
