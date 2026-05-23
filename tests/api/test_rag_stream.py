"""Tests for the streaming RAG generation route.

Strategy
--------

Like :mod:`tests.api.test_rag_query`, the route is exercised through
hermetic in-process stubs — :func:`build_llm_provider` and the
:class:`RAGOrchestrator` factory are monkey-patched so no real LLM,
ChromaDB, or embedder is required.  The orchestrator stub yields a
controllable list of :class:`StreamEvent` instances so the test can
shape the stream's deltas + final + error transitions deterministically.

Coverage focuses on:

    - SSE framing (``event: <name>\\ndata: <json>\\n\\n``);
    - happy-path event ordering: ``delta`` * N → ``citation`` * M →
      ``final``;
    - heartbeat emission on inter-event gaps (forced via a tiny
      monkey-patched heartbeat interval);
    - response headers (``text/event-stream``, ``Cache-Control: no-cache``,
      ``X-Accel-Buffering: no``);
    - pre-stream errors map to HTTP envelopes (unknown provider,
      missing key) — they MUST NOT open the SSE response;
    - in-stream errors map to ``error`` SSE events (not HTTP errors);
    - read-tier auth gate (admin-key NOT required);
    - rate-limit classification under the ``rag`` bucket;
    - audit-log discipline: ``rag_stream`` + ``rag_stream_completed``
      lines carry metadata only — never the raw query.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.api.app import create_app
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import InMemorySessionEdgarIdentityStore
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    GenerationError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
)
from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.core.types import (
    Citation,
    FilingIdentifier,
    GenerationResult,
    TokenUsage,
)
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.orchestrator import StreamEvent
from sec_generative_search.rag.query_understanding import QueryPlan

# ---------------------------------------------------------------------------
# In-process stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubLLM:
    provider_name: str = "openai"


@dataclass
class _StubBuildRecorder:
    """Captures ``build_llm_provider`` calls + the resolved key."""

    raise_with: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        provider_name: str,
        *,
        api_key_resolver: Any,
    ) -> _StubLLM:
        resolved = api_key_resolver(provider_name)
        self.calls.append({"provider": provider_name, "resolved_key": resolved})
        if self.raise_with is not None:
            raise self.raise_with
        return _StubLLM(provider_name=provider_name)


@dataclass
class _StubOrchestrator:
    """Stand-in for :class:`RAGOrchestrator`.

    ``events`` is the canned sequence the stub yields (deltas + an
    optional terminal final).  ``raise_after`` is raised after the listed
    events are exhausted, mimicking an LLM that fails mid-stream.
    """

    retrieval: Any = None
    llm: Any = None
    events: list[StreamEvent] = field(default_factory=list)
    raise_after: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_stream(
        self,
        plan: QueryPlan,
        *,
        mode: AnswerMode | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        history: Any = None,
        prefer_structured_output: bool = False,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        self.calls.append(
            {
                "plan_raw_query": plan.raw_query,
                "plan_query_en": plan.query_en,
                "plan_tickers": list(plan.tickers),
                "mode": mode,
                "model": model,
                "max_output_tokens": max_output_tokens,
                "history": history,
                "prefer_structured_output": prefer_structured_output,
                "llm_provider_name": getattr(self.llm, "provider_name", None),
            }
        )
        yield from self.events
        if self.raise_after is not None:
            raise self.raise_after


_LATEST_STUB: _StubOrchestrator | None = None


def _orchestrator_factory(*, retrieval: Any, llm: Any, **_: Any) -> _StubOrchestrator:
    assert _LATEST_STUB is not None, "test forgot to install a stub orchestrator"
    _LATEST_STUB.retrieval = retrieval
    _LATEST_STUB.llm = llm
    return _LATEST_STUB


# ---------------------------------------------------------------------------
# Fixtures + helpers
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
def rag_stream_app_factory(monkeypatch: pytest.MonkeyPatch):
    """Build a fresh app with build_llm_provider + RAGOrchestrator stubbed."""

    def factory(
        *,
        events: list[StreamEvent] | None = None,
        raise_after: Exception | None = None,
        build_raise: Exception | None = None,
        env: dict[str, str] | None = None,
        heartbeat_seconds: float | None = None,
    ) -> tuple[Any, _StubBuildRecorder, _StubOrchestrator]:
        if env:
            for key, value in env.items():
                monkeypatch.setenv(key, value)
        reload_settings()

        global _LATEST_STUB
        _LATEST_STUB = _StubOrchestrator(
            events=list(events or []),
            raise_after=raise_after,
        )
        build_stub = _StubBuildRecorder(raise_with=build_raise)

        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.build_llm_provider",
            build_stub,
        )
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.RAGOrchestrator",
            _orchestrator_factory,
        )
        if heartbeat_seconds is not None:
            monkeypatch.setattr(
                "sec_generative_search.api.routes.rag._SSE_HEARTBEAT_SECONDS",
                heartbeat_seconds,
            )

        app = create_app()
        app.state.retrieval_service = object()
        app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
        app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
        app.state.encrypted_credential_store = None
        return app, build_stub, _LATEST_STUB

    return factory


def _sample_plan_payload() -> dict[str, Any]:
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


def _default_citation() -> Citation:
    return Citation(
        chunk_id="chunk-1",
        filing_id=FilingIdentifier(
            ticker="AAPL",
            form_type="10-K",
            filing_date=date(2023, 9, 30),
            accession_number="0000320193-23-000077",
        ),
        section_path="Part I > Item 1A > Risk Factors",
        text_span="iPhone revenue concentration risk.",
        similarity=0.84,
        display_index=1,
    )


def _default_final_result(*, citations: list[Citation] | None = None) -> GenerationResult:
    return GenerationResult(
        answer="iPhone risks include concentration [1].",
        provider="openai",
        model="gpt-5.4-mini",
        prompt_version="rag-prompt-v1",
        citations=citations or [_default_citation()],
        retrieved_chunks=[],
        token_usage=TokenUsage(input_tokens=120, output_tokens=42),
        latency_seconds=1.23,
        streamed=True,
    )


def _parse_sse_frames(raw_text: str) -> list[tuple[str, dict]]:
    """Parse the text body of an SSE response into ``[(event_name, data), ...]``.

    Tests use ``response.text`` instead of ``iter_lines`` because the
    TestClient buffers the full streaming body.  Each frame is
    ``event: <name>\\ndata: <json>\\n\\n``.
    """
    frames: list[tuple[str, dict]] = []
    for block in raw_text.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        event_name = ""
        data_payload = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_payload = line[len("data: ") :]
        if event_name:
            data = json.loads(data_payload) if data_payload else {}
            frames.append((event_name, data))
    return frames


# ---------------------------------------------------------------------------
# Happy-path framing
# ---------------------------------------------------------------------------


class TestRagStreamHappyPath:
    def test_emits_deltas_then_citation_then_final(self, rag_stream_app_factory) -> None:
        events = [
            StreamEvent(delta="Hello "),
            StreamEvent(delta="world."),
            StreamEvent(final=_default_final_result()),
        ]
        app, _build, _orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"

        frames = _parse_sse_frames(response.text)
        names = [name for name, _ in frames]
        # Two deltas → one citation → one final, in that order.
        assert names == ["delta", "delta", "citation", "final"]
        assert frames[0][1] == {"text": "Hello "}
        assert frames[1][1] == {"text": "world."}

        citation = frames[2][1]
        assert citation["chunk_id"] == "chunk-1"
        assert citation["ticker"] == "AAPL"
        assert citation["filing_date"] == "2023-09-30"
        assert citation["display_index"] == 1

        final = frames[3][1]
        assert final["answer"] == "iPhone risks include concentration [1]."
        assert final["provider"] == "openai"
        assert final["model"] == "gpt-5.4-mini"
        assert final["prompt_version"] == "rag-prompt-v1"
        assert final["streamed"] is True
        assert final["refused"] is False
        assert final["token_usage"] == {
            "input_tokens": 120,
            "output_tokens": 42,
            "total_tokens": 162,
        }

    def test_orchestrator_receives_lifted_plan(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )

        call = orch.calls[0]
        assert call["plan_raw_query"] == "What are AAPL's iPhone segment risks?"
        assert call["plan_tickers"] == ["AAPL"]
        # Mode override absent in body → orchestrator decides from plan.
        assert call["mode"] is None

    def test_body_mode_overrides_plan(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")
        client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload(), "mode": "extractive"},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        assert orch.calls[0]["mode"] == AnswerMode.EXTRACTIVE

    def test_header_resolver_forwards_user_key(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, build, _orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")
        secret = "sk-USER-PROVIDED-KEY-1234"  # pragma: allowlist secret
        client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": secret},
        )
        assert build.calls[0]["resolved_key"] == secret


# ---------------------------------------------------------------------------
# Refusal path
# ---------------------------------------------------------------------------


class TestRagStreamRefusal:
    def test_empty_retrieval_marks_refused(self, rag_stream_app_factory) -> None:
        # Refusal path: orchestrator yields the refusal text + a final
        # result with no chunks / no citations.
        refusal_result = GenerationResult(
            answer="I cannot answer this from the available filings.",
            provider="openai",
            model="gpt-5.4-mini",
            prompt_version="rag-prompt-v1",
            citations=[],
            retrieved_chunks=[],
            token_usage=TokenUsage(),
            latency_seconds=0.01,
            streamed=False,
        )
        events = [
            StreamEvent(delta=refusal_result.answer),
            StreamEvent(final=refusal_result),
        ]
        app, _build, _orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )

        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        names = [name for name, _ in frames]
        assert names == ["delta", "final"]
        assert frames[1][1]["refused"] is True


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


class TestRagStreamHeartbeat:
    def test_heartbeat_emitted_on_idle(
        self,
        rag_stream_app_factory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force the heartbeat to fire on every gap by setting it to a
        # tiny value, and slow the orchestrator's yields with a sleep
        # so the consumer has time to time out.
        import time as _time

        original_sleep = _time.sleep

        @dataclass
        class _SlowOrch:
            retrieval: Any = None
            llm: Any = None

            def generate_stream(self, plan: QueryPlan, **_: Any) -> Iterator[StreamEvent]:
                # First sleep > heartbeat interval, then yield, then end.
                original_sleep(0.05)
                yield StreamEvent(delta="x")
                original_sleep(0.05)
                yield StreamEvent(final=_default_final_result(citations=[]))

        slow_orch = _SlowOrch()
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.RAGOrchestrator",
            lambda **kwargs: slow_orch,
        )

        app, _build, _orch = rag_stream_app_factory(
            events=[],  # ignored — slow_orch overrides
            heartbeat_seconds=0.01,
        )
        # Re-patch RAGOrchestrator with our slow stub since the factory
        # wires the dataclass-based stub.
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.RAGOrchestrator",
            lambda **kwargs: slow_orch,
        )

        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        names = [name for name, _ in frames]
        # At least one heartbeat must have appeared during an idle gap.
        assert "heartbeat" in names
        # The terminal final still arrives.
        assert names[-1] == "final"


# ---------------------------------------------------------------------------
# Pre-stream errors → HTTP envelopes
# ---------------------------------------------------------------------------


class TestRagStreamPreStreamErrors:
    def test_unknown_provider_400_http(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(events=[])
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload(), "provider": "definitely-not-real"},
        )
        assert response.status_code == 400
        # Plain JSON envelope — NOT an SSE response.
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"] == "unknown_provider"

    def test_missing_key_400_http(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(
            build_raise=ConfigurationError("no key resolved"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
        )
        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body["error"] == "provider_key_required"


# ---------------------------------------------------------------------------
# In-stream errors → SSE error events
# ---------------------------------------------------------------------------


class TestRagStreamInStreamErrors:
    def test_provider_auth_error_emits_sse_error(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(delta="partial")]
        app, _build, _orch = rag_stream_app_factory(
            events=events,
            raise_after=ProviderAuthError("rejected mid-stream"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        # The stream opened with 200 → the error is an SSE event, not
        # an HTTP status.
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        names = [name for name, _ in frames]
        assert names == ["delta", "error"]
        assert frames[1][1]["error"] == "provider_unauthorized"

    def test_rate_limit_emits_sse_error(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(
            events=[],
            raise_after=ProviderRateLimitError("upstream limited"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        assert any(
            name == "error" and data["error"] == "provider_unavailable" for name, data in frames
        )

    def test_generic_provider_error_emits_sse_error(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(
            events=[],
            raise_after=ProviderError("transport blew up"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        assert any(name == "error" and data["error"] == "provider_error" for name, data in frames)

    def test_generation_error_emits_sse_error(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(
            events=[],
            raise_after=GenerationError("citation parse failed"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        assert any(name == "error" and data["error"] == "generation_error" for name, data in frames)

    def test_unknown_exception_emits_internal_error(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(
            events=[],
            raise_after=RuntimeError("unmapped"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        assert any(name == "error" and data["error"] == "internal_error" for name, data in frames)


# ---------------------------------------------------------------------------
# Schema guards
# ---------------------------------------------------------------------------


class TestRagStreamSchemaGuards:
    def test_missing_plan_rejected(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(events=[])
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/rag/stream", json={})
        assert response.status_code == 422

    def test_bad_mode_rejected(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(events=[])
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload(), "mode": "creative"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Privacy / audit-log no-leak contract
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRagStreamPrivacyContract:
    def test_audit_log_carries_metadata_not_query(
        self,
        rag_stream_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        events = [
            StreamEvent(delta="x"),
            StreamEvent(final=_default_final_result(citations=[])),
        ]
        app, _build, _orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                plan = _sample_plan_payload()
                plan["raw_query"] = "PROPRIETARY-WATCHLIST-MNEMONIC-ZZZ"
                plan["query_en"] = "PROPRIETARY-WATCHLIST-MNEMONIC-ZZZ"
                client.post(
                    "/api/rag/stream",
                    json={"plan": plan},
                    headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
                )
        finally:
            package_logger.propagate = prior_propagate

        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        # Both the open-stream and the completed-stream lines must
        # appear, and neither may carry the raw query.
        assert any("rag_stream" in line for line in audit)
        assert any("rag_stream_completed" in line for line in audit)
        assert all("PROPRIETARY-WATCHLIST-MNEMONIC-ZZZ" not in line for line in audit)

    def test_in_stream_error_event_does_not_echo_provider_key(
        self,
        rag_stream_app_factory,
    ) -> None:
        app, _build, _orch = rag_stream_app_factory(
            events=[],
            raise_after=ProviderError("kaboom"),
        )
        client = TestClient(app, base_url="https://testserver")
        secret = "sk-NEVER-ECHO-ME-1234"  # pragma: allowlist secret
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": secret},
        )
        assert response.status_code == 200
        assert secret not in response.text


# ---------------------------------------------------------------------------
# Auth tier — read-tier (no admin gate)
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRagStreamAuthGate:
    def test_api_key_required_when_configured(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, _orch = rag_stream_app_factory(
            events=events,
            env={"API_KEY": "shared-team-key"},  # pragma: allowlist secret
        )
        client = TestClient(app, base_url="https://testserver")

        unauthed = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert unauthed.status_code == 401
        assert unauthed.json()["error"] == "unauthorised"

        ok = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Provider-Key-openai": "sk-1234",  # pragma: allowlist secret
            },
        )
        assert ok.status_code == 200

    def test_admin_key_not_required(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, _orch = rag_stream_app_factory(
            events=events,
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Provider-Key-openai": "sk-1234",  # pragma: allowlist secret
            },
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Rate-limit classification
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRagStreamRateLimitClassification:
    def test_stream_path_classifies_as_rag(self) -> None:
        from sec_generative_search.api.middleware import _classify_path

        assert _classify_path("/api/rag/stream", "POST") == "rag"


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------


class TestRagStreamHistory:
    """The chat surface replays prior turns through ``body.history``.

    These tests pin the shape — history forwarded into
    :meth:`RAGOrchestrator.generate_stream` as :class:`ConversationTurn`
    instances and the load-bearing privacy invariant: retrieved chunks /
    citations from prior turns are stripped at the route boundary so a
    follow-up never re-injects the prior turn's chunk text into the
    prompt.
    """

    def test_history_forwarded_as_conversation_turns(
        self,
        rag_stream_app_factory,
    ) -> None:
        events = [
            StreamEvent(delta="ok"),
            StreamEvent(final=_default_final_result(citations=[])),
        ]
        app, _build, orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        with client.stream(
            "POST",
            "/api/rag/stream",
            json={
                "plan": _sample_plan_payload(),
                "history": [
                    {"query": "what is revenue?", "answer": "Revenue is total sales."},
                ],
            },
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        ) as response:
            # Drain so the producer completes before we inspect calls.
            for _ in response.iter_bytes():
                pass

        passed = orch.calls[0]["history"]
        assert passed is not None
        assert len(passed) == 1
        assert passed[0].query == "what is revenue?"
        assert passed[0].generation_result.answer == "Revenue is total sales."
        # Retrieval / citations stripped at the route boundary.
        assert passed[0].retrieval_results == []
        assert passed[0].generation_result.citations == []
        assert passed[0].generation_result.retrieved_chunks == []

    def test_history_omitted_becomes_none(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        with client.stream(
            "POST",
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        ) as response:
            for _ in response.iter_bytes():
                pass

        assert orch.calls[0]["history"] is None

    def test_history_oversize_rejected_pre_stream(
        self,
        rag_stream_app_factory,
    ) -> None:
        app, _build, _orch = rag_stream_app_factory()
        client = TestClient(app, base_url="https://testserver")
        body = {
            "plan": _sample_plan_payload(),
            "history": [{"query": f"q{i}", "answer": f"a{i}"} for i in range(11)],
        }
        response = client.post("/api/rag/stream", json=body)
        # Pre-stream schema rejection → 422 HTTP, NOT an SSE error event.
        assert response.status_code == 422


@pytest.mark.security
class TestRagStreamHistoryPrivacyContract:
    def test_audit_lines_carry_history_count_not_text(
        self,
        rag_stream_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, _orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        secret_q = "STREAM-PRIOR-QUESTION-SENTINEL-ABC"  # pragma: allowlist secret
        secret_a = "STREAM-PRIOR-ANSWER-SENTINEL-ABC"  # pragma: allowlist secret
        try:
            with (
                caplog.at_level(logging.WARNING, logger=LOGGER_NAME),
                client.stream(
                    "POST",
                    "/api/rag/stream",
                    json={
                        "plan": _sample_plan_payload(),
                        "history": [{"query": secret_q, "answer": secret_a}],
                    },
                    headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
                ) as response,
            ):
                for _ in response.iter_bytes():
                    pass
        finally:
            package_logger.propagate = prior_propagate

        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        # Both rag_stream (open) and rag_stream_completed (close) carry
        # the history_turns count; the text never reaches any line.
        assert any("rag_stream" in line and "history_turns=1" in line for line in audit)
        assert any("rag_stream_completed" in line and "history_turns=1" in line for line in audit)
        assert all(secret_q not in line for line in audit)
        assert all(secret_a not in line for line in audit)
