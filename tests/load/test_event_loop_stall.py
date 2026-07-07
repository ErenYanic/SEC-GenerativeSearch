"""Event-loop stall benchmark for blocking routes offloaded to the threadpool.

The probe fires a real request that blocks the handling thread for
:data:`_BLOCK_SECONDS` and samples event-loop lag while it is in flight.
If the handler blocks on the loop, one sample spikes to the full block
duration; if the handler runs in the threadpool, the loop stays responsive.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest
from httpx import ASGITransport

from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import InMemorySessionEdgarIdentityStore

pytestmark = [pytest.mark.load, pytest.mark.asyncio]


# The blocking handler holds its worker thread this long.
_BLOCK_SECONDS = 2.0

# The largest event-loop scheduling lag tolerated while a blocking request is
# in flight.
_MAX_LOOP_LAG_SECONDS = 0.5

# How often the probe samples loop lag while the request is in flight.
_TICK_SECONDS = 0.02


class _BlockingRetrieval:
    """Stand-in for :class:`RetrievalService` that blocks its calling thread."""

    def retrieve(self, query: str, **kwargs: Any) -> list:
        time.sleep(_BLOCK_SECONDS)
        return []


class _FakeLLM:
    """Minimal provider double for the generation probe."""

    provider_name = "openai"

    def close(self) -> None: ...


def _build_app(retrieval: _BlockingRetrieval):
    """Build the real ASGI app with the blocking retrieval on ``app.state``."""
    from sec_generative_search.api.app import create_app

    app = create_app()
    app.state.retrieval_service = retrieval
    app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
    app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
    app.state.encrypted_credential_store = None
    return app


async def _assert_loop_stays_responsive(app, fire_blocking) -> None:
    """Fire a blocking request and prove the event loop stays responsive."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="https://testserver") as client:
        blocking = asyncio.create_task(fire_blocking(client))
        health = asyncio.create_task(client.get("/api/health"))

        max_lag = 0.0
        deadline = time.perf_counter() + _BLOCK_SECONDS + 1.0
        # Sample until the blocking request finishes or the window closes.
        while time.perf_counter() < deadline and not blocking.done():
            tick_start = time.perf_counter()
            await asyncio.sleep(_TICK_SECONDS)
            max_lag = max(max_lag, (time.perf_counter() - tick_start) - _TICK_SECONDS)

        health_resp = await health
        blocking_resp = await blocking

        assert max_lag < _MAX_LOOP_LAG_SECONDS, (
            f"event loop lagged {max_lag:.2f}s while a blocking request was in "
            "flight — the loop is stalled (a heavy handler reverted to async def "
            "or other on-loop blocking)."
        )
        assert health_resp.status_code == 200
        assert blocking_resp.status_code < 500


async def test_blocking_search_does_not_stall_loop() -> None:
    """``POST /api/search`` blocks in the threadpool, not on the loop."""
    retrieval = _BlockingRetrieval()
    app = _build_app(retrieval)

    async def fire(client: httpx.AsyncClient) -> httpx.Response:
        return await client.post("/api/search", json={"query": "revenue growth"})

    await _assert_loop_stays_responsive(app, fire)


async def test_blocking_generation_does_not_stall_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``POST /api/rag/query`` blocks in the threadpool, not on the loop."""
    retrieval = _BlockingRetrieval()
    app = _build_app(retrieval)
    monkeypatch.setattr(
        "sec_generative_search.api.routes.rag.build_llm_provider",
        lambda *args, **kwargs: _FakeLLM(),
    )

    plan = {
        "raw_query": "revenue growth",
        "detected_language": "en",
        "query_en": "revenue growth",
        "tickers": [],
        "form_types": [],
        "date_range": None,
        "intent": "",
        "suggested_answer_mode": "concise",
    }

    async def fire(client: httpx.AsyncClient) -> httpx.Response:
        return await client.post(
            "/api/rag/query",
            json={"plan": plan, "provider": "openai"},
        )

    await _assert_loop_stays_responsive(app, fire)
