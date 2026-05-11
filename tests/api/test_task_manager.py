"""Tests for the Phase-11.1 background ``TaskManager``.

Strategy
--------

The task manager composes a :class:`FilingStore`, a :class:`MetadataRegistry`,
a :class:`FilingFetcher`, and a :class:`PipelineOrchestrator`.  Wiring
real instances would pull in ChromaDB, SQLite, edgartools, and the
sentence-transformers embedder — none of which we need to exercise the
worker's *coordination* contract.  Each collaborator is replaced by a
small in-process stub that records calls and lets the test shape the
relevant happy / sad path.

Coverage areas:

    - Lifecycle: create → terminal-state transition → history persistence.
    - Skip / duplicate / late-duplicate paths.
    - Cancellation: pre-fetch (queued), mid-pipeline (via progress
      callback), and rollback through ``FilingStore.delete_filings_batch``.
    - Filing-limit ceiling.
    - GPU semaphore serialises concurrent tasks.
    - Lazy eviction of stale terminal entries (no background thread).
    - Per-session listing.
    - **Security:** :class:`TaskInfo` carries no credential-shaped attribute
      name (mirrors :mod:`tests.core.test_types`); ``task_history`` write
      sites never receive a key argument.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, fields
from datetime import date
from typing import Any

import numpy as np
import pytest

from sec_generative_search.api.tasks import (
    FilingResult,
    TaskInfo,
    TaskManager,
    TaskProgress,
    TaskQueueFullError,
    TaskState,
)
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.exceptions import (
    DatabaseError,
    FetchError,
    FilingLimitExceededError,
)
from sec_generative_search.core.types import Chunk, ContentType, FilingIdentifier, IngestResult
from sec_generative_search.pipeline.fetch import FilingInfo
from sec_generative_search.pipeline.orchestrator import ProcessedFiling

# ---------------------------------------------------------------------------
# Stub collaborators
# ---------------------------------------------------------------------------


def _make_filing_info(
    ticker: str,
    accession_number: str,
    form_type: str = "10-K",
    filing_date: date = date(2023, 9, 30),
) -> FilingInfo:
    """Build a minimal ``FilingInfo`` (no cached ``_filing_obj``)."""
    return FilingInfo(
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date,
        accession_number=accession_number,
        company_name=f"{ticker} Inc.",
    )


def _make_processed_filing(filing_info: FilingInfo) -> ProcessedFiling:
    """Build a ProcessedFiling with one chunk + tiny embedding."""
    filing_id = filing_info.to_identifier()
    chunk = Chunk(
        content="Sample chunk text.",
        path="Part I > Item 1",
        content_type=ContentType.TEXT,
        filing_id=filing_id,
        chunk_index=0,
    )
    return ProcessedFiling(
        filing_id=filing_id,
        chunks=[chunk],
        embeddings=np.zeros((1, 4), dtype=np.float32),
        ingest_result=IngestResult(
            filing_id=filing_id,
            segment_count=1,
            chunk_count=1,
            duration_seconds=0.05,
        ),
    )


@dataclass
class _StubRegistry:
    """In-memory stand-in for :class:`MetadataRegistry`.

    Implements only the read paths and ``save_task_history`` /
    ``prune_task_history`` calls the manager actually invokes — enough
    to drive every code path without booting SQLite.
    """

    existing_accessions: set[str] = field(default_factory=set)
    filing_count: int = 0
    oldest: list[Any] = field(default_factory=list)
    history: dict[str, dict] = field(default_factory=dict)
    save_calls: list[dict] = field(default_factory=list)
    prune_calls: int = 0

    def get_existing_accessions(self, accessions: list[str]) -> set[str]:
        return {a for a in accessions if a in self.existing_accessions}

    def count(self) -> int:
        return self.filing_count

    def list_oldest_filings(self, limit: int) -> list[Any]:
        return self.oldest[:limit]

    def save_task_history(self, task_id: str, **kwargs: Any) -> None:
        # The manager never passes a key argument — assert here so a
        # future regression surfaces in *every* test, not just the
        # explicit security one.
        for forbidden in ("api_key", "secret", "credential", "auth_token"):
            assert forbidden not in kwargs, (
                f"save_task_history received forbidden kwarg {forbidden}"
            )
        self.save_calls.append({"task_id": task_id, **kwargs})
        self.history[task_id] = {"task_id": task_id, **kwargs}

    def get_task_history(self, task_id: str) -> dict | None:
        row = self.history.get(task_id)
        if row is None:
            return None
        # Shape mirrors ``MetadataRegistry.get_task_history``.
        return {
            "task_id": row["task_id"],
            "status": row["status"],
            "tickers": row.get("tickers") or [],
            "form_types": row.get("form_types") or [],
            "results": row.get("results") or [],
            "error": row.get("error"),
            "started_at": row.get("started_at"),
            "completed_at": row.get("completed_at"),
            "filings_done": row.get("filings_done", 0),
            "filings_skipped": row.get("filings_skipped", 0),
            "filings_failed": row.get("filings_failed", 0),
        }

    def prune_task_history(self) -> int:
        self.prune_calls += 1
        return 0


@dataclass
class _StubFetcher:
    """Records ``apply_identity`` calls and serves canned work lists."""

    work_lists: dict[str, list[FilingInfo]] = field(default_factory=dict)
    cross_form: dict[tuple, list[FilingInfo]] = field(default_factory=dict)
    content_map: dict[str, str] = field(default_factory=dict)
    identity_calls: list[tuple[str | None, str | None]] = field(default_factory=list)
    raise_on_fetch: dict[str, Exception] = field(default_factory=dict)

    def apply_identity(self, name: str | None = None, email: str | None = None) -> None:
        self.identity_calls.append((name, email))

    def list_available(
        self,
        ticker: str,
        form_type: str,
        *,
        count: int | None = None,
        year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[FilingInfo]:
        return list(self.work_lists.get((ticker, form_type), []))

    def list_available_across_forms(
        self,
        ticker: str,
        form_types: tuple,
        *,
        count: int | None = None,
        year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[FilingInfo]:
        return list(self.cross_form.get((ticker, form_types), []))

    def fetch_filing_content(
        self,
        filing_info: FilingInfo,
    ) -> tuple[FilingIdentifier, str]:
        if filing_info.accession_number in self.raise_on_fetch:
            raise self.raise_on_fetch[filing_info.accession_number]
        return (
            filing_info.to_identifier(),
            self.content_map.get(filing_info.accession_number, "<html></html>"),
        )


@dataclass
class _StubOrchestrator:
    """Replays a canned ``ProcessedFiling`` per filing id.

    Optionally invokes the ``progress_callback`` between each step so
    the cancel-mid-pipeline test can fire ``_CancelledError`` from a
    plausible code path.
    """

    by_accession: dict[str, ProcessedFiling] = field(default_factory=dict)
    invoke_progress: bool = False
    raise_on: dict[str, Exception] = field(default_factory=dict)

    def process_filing(
        self,
        filing_id: FilingIdentifier,
        html_content: str,
        progress_callback=None,
    ) -> ProcessedFiling:
        if filing_id.accession_number in self.raise_on:
            raise self.raise_on[filing_id.accession_number]
        if self.invoke_progress and progress_callback is not None:
            # Three steps mirroring the real orchestrator's 4-step
            # pipeline (parse / chunk / embed / complete).
            for idx, step in enumerate(("Parsing", "Chunking", "Embedding"), start=1):
                progress_callback(step, idx, 4)
        return self.by_accession[filing_id.accession_number]


@dataclass
class _StubFilingStore:
    """Records writes + deletes; the manager's single dual-store seam."""

    duplicate_accessions: set[str] = field(default_factory=set)
    stored: list[str] = field(default_factory=list)
    deleted: list[list[str]] = field(default_factory=list)
    raise_on_store: dict[str, Exception] = field(default_factory=dict)

    def store_filing(
        self,
        processed: ProcessedFiling,
        *,
        register_if_new: bool = False,
    ) -> bool:
        accession = processed.filing_id.accession_number
        if accession in self.raise_on_store:
            raise self.raise_on_store[accession]
        if register_if_new and accession in self.duplicate_accessions:
            return False
        self.stored.append(accession)
        return True

    def delete_filings_batch(self, accessions: list[str]) -> int:
        # Manager passes a list copy; capture it verbatim.
        self.deleted.append(list(accessions))
        return len(accessions)


def _build_manager(
    *,
    registry: _StubRegistry | None = None,
    fetcher: _StubFetcher | None = None,
    orchestrator: _StubOrchestrator | None = None,
    filing_store: _StubFilingStore | None = None,
) -> tuple[TaskManager, _StubFilingStore, _StubRegistry, _StubFetcher, _StubOrchestrator]:
    registry = registry or _StubRegistry()
    fetcher = fetcher or _StubFetcher()
    orchestrator = orchestrator or _StubOrchestrator()
    filing_store = filing_store or _StubFilingStore()
    manager = TaskManager(
        filing_store=filing_store,  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
        fetcher=fetcher,  # type: ignore[arg-type]
        orchestrator=orchestrator,  # type: ignore[arg-type]
    )
    return manager, filing_store, registry, fetcher, orchestrator


def _wait_for_state(
    manager: TaskManager,
    task_id: str,
    *,
    target: TaskState | set[TaskState],
    timeout: float = 3.0,
) -> TaskInfo:
    """Poll until the task reaches the target state or the deadline."""
    targets = target if isinstance(target, set) else {target}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = manager.get_task(task_id)
        if info is not None and info.state in targets:
            return info
        time.sleep(0.01)
    raise AssertionError(
        f"task {task_id[:8]} did not reach {targets} within {timeout}s "
        f"(last state: {info.state if info else 'unknown'})"
    )


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch: pytest.MonkeyPatch):
    """Clear API_ env vars between tests so settings stay deterministic."""
    import os

    for key in list(os.environ.keys()):
        if key.startswith("API_"):
            monkeypatch.delenv(key, raising=False)
    reload_settings()
    yield
    reload_settings()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestTaskLifecycle:
    """A single-filing task progresses to COMPLETED and persists history."""

    def test_completed_task_persists_history(self) -> None:
        filing_info = _make_filing_info("AAPL", "0000320193-23-000077")
        processed = _make_processed_filing(filing_info)

        fetcher = _StubFetcher(
            work_lists={("AAPL", "10-K"): [filing_info]},
        )
        orchestrator = _StubOrchestrator(by_accession={filing_info.accession_number: processed})

        manager, store, registry, fetcher, _ = _build_manager(
            fetcher=fetcher,
            orchestrator=orchestrator,
        )

        task_id = manager.create_task(
            tickers=["AAPL"],
            form_types=["10-K"],
            session_id="sess-A",
        )
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        # Storage went through FilingStore.store_filing.
        assert store.stored == [filing_info.accession_number]
        # Single FilingResult recorded.
        assert len(info.results) == 1
        assert info.results[0].accession_number == filing_info.accession_number
        # History row written at terminal transition.
        assert task_id in registry.history
        assert registry.history[task_id]["status"] == "completed"

    def test_get_task_falls_back_to_history(self) -> None:
        """Tasks evicted from memory still resolve via ``task_history``."""
        manager, _, registry, _, _ = _build_manager()
        # Seed the history table directly so we can simulate an evicted
        # task without driving a full worker run.
        registry.history["evicted-id"] = {
            "task_id": "evicted-id",
            "status": "completed",
            "tickers": ["AAPL"],
            "form_types": ["10-K"],
            "results": [],
            "error": None,
            "started_at": None,
            "completed_at": None,
            "filings_done": 0,
            "filings_skipped": 0,
            "filings_failed": 0,
        }
        info = manager.get_task("evicted-id")
        assert info is not None
        assert info.state == TaskState.COMPLETED


# ---------------------------------------------------------------------------
# Duplicate / skip handling
# ---------------------------------------------------------------------------


class TestDuplicateHandling:
    """The two skip paths: known-up-front and late (race-window)."""

    def test_batch_dup_check_skips_known_accession(self) -> None:
        filing_info = _make_filing_info("AAPL", "0000320193-23-000077")
        registry = _StubRegistry(existing_accessions={filing_info.accession_number})
        manager, store, _, _, _ = _build_manager(
            registry=registry,
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [filing_info]}),
        )
        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)
        # No write went through the store — the dup short-circuit fired.
        assert store.stored == []
        assert info.progress.filings_skipped == 1

    def test_late_duplicate_via_register_if_new_false(self) -> None:
        """``FilingStore.store_filing(register_if_new=True)`` may return False."""
        filing_info = _make_filing_info("AAPL", "0000320193-23-000077")
        processed = _make_processed_filing(filing_info)
        store = _StubFilingStore(duplicate_accessions={filing_info.accession_number})
        manager, store, _, _, _ = _build_manager(
            filing_store=store,
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [filing_info]}),
            orchestrator=_StubOrchestrator(
                by_accession={filing_info.accession_number: processed},
            ),
        )
        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)
        assert info.progress.filings_skipped == 1
        assert info.progress.filings_done == 1
        # No stored accession — the atomic claim path lost the race.
        assert store.stored == []


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_before_acquire_marks_cancelled(self, monkeypatch) -> None:
        """A cancel before the GPU acquire still terminates cleanly."""
        # Hold the GPU slot so the worker blocks on ``acquire``.
        manager, store, _, _, _ = _build_manager()
        manager._gpu_semaphore.acquire()
        try:
            task_id = manager.create_task(
                tickers=["AAPL"],
                form_types=["10-K"],
            )
            # Cancel while the worker is queued behind the semaphore.
            assert manager.cancel_task(task_id) is True
        finally:
            # Let the worker proceed.
            manager._gpu_semaphore.release()

        _wait_for_state(manager, task_id, target=TaskState.CANCELLED)
        # No filings were even attempted, so no rollback delete fired.
        assert store.deleted == []
        # Cancel signal returns False on a terminal task.
        assert manager.cancel_task(task_id) is False

    def test_cancel_during_pipeline_rolls_back_stored_filings(self) -> None:
        """The progress callback's cancel check triggers rollback.

        Two-filing race-free design: the orchestrator stub for f2 looks
        up the running task via ``manager.list_tasks()`` rather than
        capturing the task info ahead of time. The manager reference is
        bound before the worker thread starts, so there is no window
        where the stub could execute before the holder is populated.
        """
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("AAPL", "0000320193-23-000002")
        p1 = _make_processed_filing(f1)
        p2 = _make_processed_filing(f2)

        manager_holder: dict[str, TaskManager] = {}

        class _CancelOnSecond(_StubOrchestrator):
            def process_filing(self, filing_id, html_content, progress_callback=None):
                if filing_id.accession_number == f2.accession_number:
                    mgr = manager_holder["mgr"]
                    for task in mgr.list_tasks():
                        if task.state == TaskState.RUNNING:
                            task.cancel_event.set()
                    if progress_callback is not None:
                        progress_callback("Embedding", 3, 4)
                return self.by_accession[filing_id.accession_number]

        manager, store, _, _, _ = _build_manager(
            fetcher=_StubFetcher(
                work_lists={("AAPL", "10-K"): [f1, f2]},
            ),
            orchestrator=_CancelOnSecond(
                by_accession={f1.accession_number: p1, f2.accession_number: p2},
            ),
        )
        # Bind the manager *before* spawning the worker so the stub
        # cannot race ahead of holder population.
        manager_holder["mgr"] = manager

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.CANCELLED)
        # f1 was stored, then rolled back via FilingStore.
        assert store.stored == [f1.accession_number]
        assert store.deleted == [[f1.accession_number]]
        # The rollback path cleared ``_stored_accessions``.
        assert info._stored_accessions == []

    def test_rollback_swallows_database_error(self, caplog) -> None:
        """A failing rollback logs but does not raise.

        Two-filing scenario: f1 stores successfully, then the
        progress-callback cancel fires inside f2's orchestrator call,
        forcing the rollback path through ``delete_filings_batch``
        which is configured to raise. The worker must still reach the
        ``CANCELLED`` terminal state.
        """
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("AAPL", "0000320193-23-000002")
        p1 = _make_processed_filing(f1)
        p2 = _make_processed_filing(f2)

        class _FailingStore(_StubFilingStore):
            def delete_filings_batch(self, accessions: list[str]) -> int:
                raise DatabaseError("chroma down")

        manager_holder: dict[str, TaskManager] = {}

        class _CancelOnSecond(_StubOrchestrator):
            def process_filing(self, filing_id, html_content, progress_callback=None):
                if filing_id.accession_number == f2.accession_number:
                    mgr = manager_holder["mgr"]
                    for task in mgr.list_tasks():
                        if task.state == TaskState.RUNNING:
                            task.cancel_event.set()
                    if progress_callback is not None:
                        progress_callback("Embedding", 3, 4)
                return self.by_accession[filing_id.accession_number]

        manager, _, _, _, _ = _build_manager(
            filing_store=_FailingStore(),
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1, f2]}),
            orchestrator=_CancelOnSecond(
                by_accession={f1.accession_number: p1, f2.accession_number: p2},
            ),
        )
        manager_holder["mgr"] = manager

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.CANCELLED)
        # Worker terminated cleanly even though rollback raised
        # ``DatabaseError`` inside ``delete_filings_batch``.
        assert info.state == TaskState.CANCELLED


# ---------------------------------------------------------------------------
# Filing-limit ceiling
# ---------------------------------------------------------------------------


class TestFilingLimit:
    def test_exceeded_limit_fails_task(self, monkeypatch) -> None:
        """When the registry is at capacity the worker surfaces a failure."""
        monkeypatch.setenv("DB_MAX_FILINGS", "1")
        reload_settings()

        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        registry = _StubRegistry(filing_count=1)
        manager, _, _, _, _ = _build_manager(
            registry=registry,
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(
                by_accession={f1.accession_number: _make_processed_filing(f1)},
            ),
        )
        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.FAILED)
        assert info.error is not None
        assert info.error == FilingLimitExceededError(1, 1).message


# ---------------------------------------------------------------------------
# Queue cap
# ---------------------------------------------------------------------------


class TestQueueCap:
    def test_queue_cap_rejects_new_task(self, monkeypatch) -> None:
        monkeypatch.setenv("API_MAX_TASK_QUEUE_SIZE", "1")
        reload_settings()

        manager, _, _, _, _ = _build_manager()
        # Hold the GPU slot so the first task stays PENDING and counts
        # against the active cap.
        manager._gpu_semaphore.acquire()
        try:
            first = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
            with pytest.raises(TaskQueueFullError) as excinfo:
                manager.create_task(tickers=["MSFT"], form_types=["10-K"])
            assert excinfo.value.details == {"active": 1, "maximum": 1}
        finally:
            manager._gpu_semaphore.release()
        # Drain the first task so the fixture's reload_settings doesn't
        # race with a daemon thread still iterating.
        info = manager.get_task(first)
        assert info is not None
        info.cancel_event.set()
        _wait_for_state(manager, first, target={TaskState.CANCELLED, TaskState.COMPLETED})

    def test_shutdown_blocks_new_tasks(self) -> None:
        manager, _, _, _, _ = _build_manager()
        manager.shutdown()
        with pytest.raises(TaskQueueFullError):
            manager.create_task(tickers=["AAPL"], form_types=["10-K"])


# ---------------------------------------------------------------------------
# GPU semaphore serialisation
# ---------------------------------------------------------------------------


class TestGpuSerialisation:
    def test_two_tasks_run_sequentially(self) -> None:
        """The ``Semaphore(1)`` gate forbids overlapping pipeline runs."""
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("MSFT", "0000789019-23-000007")
        p1 = _make_processed_filing(f1)
        p2 = _make_processed_filing(f2)

        # Track concurrent entries into the orchestrator. A correct
        # implementation gates with the GPU semaphore, so this counter
        # must never exceed 1 across the run.
        active_lock = threading.Lock()
        active = {"count": 0, "max": 0}

        class _CountingOrch(_StubOrchestrator):
            def process_filing(self, filing_id, html_content, progress_callback=None):
                with active_lock:
                    active["count"] += 1
                    active["max"] = max(active["max"], active["count"])
                try:
                    # Sleep so concurrent tasks would actually overlap
                    # if the semaphore were broken.
                    time.sleep(0.05)
                    return self.by_accession[filing_id.accession_number]
                finally:
                    with active_lock:
                        active["count"] -= 1

        manager, store, _, _, _ = _build_manager(
            fetcher=_StubFetcher(
                work_lists={
                    ("AAPL", "10-K"): [f1],
                    ("MSFT", "10-K"): [f2],
                }
            ),
            orchestrator=_CountingOrch(
                by_accession={f1.accession_number: p1, f2.accession_number: p2},
            ),
        )

        task_a = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        task_b = manager.create_task(tickers=["MSFT"], form_types=["10-K"])
        _wait_for_state(manager, task_a, target=TaskState.COMPLETED)
        _wait_for_state(manager, task_b, target=TaskState.COMPLETED)

        # Both tasks stored their filing; neither overlapped in
        # ``process_filing`` (the GPU semaphore is the gate).
        assert set(store.stored) == {f1.accession_number, f2.accession_number}
        assert active["max"] == 1


# ---------------------------------------------------------------------------
# Session ownership
# ---------------------------------------------------------------------------


class TestSessionOwnership:
    def test_list_tasks_for_session_filters(self) -> None:
        manager, _, _, _, _ = _build_manager()
        # Suspend the GPU so the tasks stay in flight and we can list
        # them without racing the COMPLETED transition.
        manager._gpu_semaphore.acquire()
        try:
            a = manager.create_task(
                tickers=["AAPL"],
                form_types=["10-K"],
                session_id="sess-A",
            )
            b = manager.create_task(
                tickers=["MSFT"],
                form_types=["10-K"],
                session_id="sess-B",
            )
            c = manager.create_task(
                tickers=["NVDA"],
                form_types=["10-K"],
                session_id=None,
            )

            for_a = {t.task_id for t in manager.list_tasks_for_session("sess-A")}
            for_b = {t.task_id for t in manager.list_tasks_for_session("sess-B")}
            for_none = {t.task_id for t in manager.list_tasks_for_session(None)}

            assert for_a == {a}
            assert for_b == {b}
            assert for_none == {c}
        finally:
            manager._gpu_semaphore.release()
            for task_id in (a, b, c):
                info = manager.get_task(task_id)
                if info is not None:
                    info.cancel_event.set()


# ---------------------------------------------------------------------------
# Lazy eviction
# ---------------------------------------------------------------------------


class TestLazyEviction:
    def test_stale_terminal_task_evicted_on_read(self, monkeypatch) -> None:
        """Terminal tasks older than the TTL are dropped from memory on the
        next ``list_tasks`` / ``get_task`` / ``create_task`` call."""
        from datetime import UTC, datetime, timedelta

        manager, _, registry, _, _ = _build_manager()
        # Insert a synthetic terminal task with an old ``completed_at``.
        info = TaskInfo(
            task_id="ancient",
            tickers=["AAPL"],
            form_types=["10-K"],
        )
        info.state = TaskState.COMPLETED
        info.completed_at = datetime.now(UTC) - timedelta(days=2)
        manager._tasks["ancient"] = info

        # Force a low TTL so the eviction sweep would fire even on a
        # younger task — keeps the test robust against clock skew.
        monkeypatch.setattr(
            "sec_generative_search.api.tasks._TASK_TTL_SECONDS",
            1,
        )

        # First read sweeps. Even though the task is gone from memory,
        # ``get_task`` does NOT need a history fallback for this case
        # (we never wrote one) so ``None`` is the expected result.
        result = manager.get_task("ancient")
        assert result is None
        assert "ancient" not in manager._tasks
        # Eviction also called ``prune_task_history`` so the on-disk
        # row count stays bounded by the operator-configured retention.
        assert registry.prune_calls >= 1


# ---------------------------------------------------------------------------
# EDGAR identity resolver
# ---------------------------------------------------------------------------


class TestEdgarIdentityResolver:
    def test_resolver_value_never_lands_on_task_info(self) -> None:
        """The resolver lives on the manager dict, not on TaskInfo."""
        from sec_generative_search.core.edgar_identity import EdgarIdentity

        captured: list[tuple] = []

        def _resolver() -> EdgarIdentity:
            return EdgarIdentity(
                name="Test User",
                email="test@example.invalid",
            )

        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        p1 = _make_processed_filing(f1)
        fetcher = _StubFetcher(
            work_lists={("AAPL", "10-K"): [f1]},
        )
        # Capture identity arguments rather than asserting in the stub
        # so we can inspect the call ordering.
        original_apply = fetcher.apply_identity

        def _apply(name=None, email=None):
            captured.append((name, email))
            return original_apply(name, email)

        fetcher.apply_identity = _apply  # type: ignore[method-assign]

        manager, _, _, _, _ = _build_manager(
            fetcher=fetcher,
            orchestrator=_StubOrchestrator(by_accession={f1.accession_number: p1}),
        )

        task_id = manager.create_task(
            tickers=["AAPL"],
            form_types=["10-K"],
            session_id="sess",
            edgar_identity_resolver=_resolver,
        )
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        # apply_identity received the resolver's identity at least once.
        assert ("Test User", "test@example.invalid") in captured

        # TaskInfo has no field carrying the identity value.
        for f in fields(info):
            if isinstance(getattr(info, f.name), str):
                assert getattr(info, f.name) != "test@example.invalid"
                assert getattr(info, f.name) != "Test User"

        # Resolver is cleared from the parallel dict in the worker's
        # ``finally``.
        assert task_id not in manager._task_resolvers


# ---------------------------------------------------------------------------
# Fetcher errors
# ---------------------------------------------------------------------------


class TestFetcherErrors:
    def test_fetch_error_marks_filing_failed_continues(self) -> None:
        """A single fetch failure should not nuke the whole batch."""
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("AAPL", "0000320193-23-000002")
        p2 = _make_processed_filing(f2)

        fetcher = _StubFetcher(
            work_lists={("AAPL", "10-K"): [f1, f2]},
            raise_on_fetch={f1.accession_number: FetchError("EDGAR 502")},
        )
        manager, store, _, _, _ = _build_manager(
            fetcher=fetcher,
            orchestrator=_StubOrchestrator(by_accession={f2.accession_number: p2}),
        )
        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)
        # f1 marked failed, f2 succeeded.
        assert info.progress.filings_failed == 1
        assert store.stored == [f2.accession_number]


# ---------------------------------------------------------------------------
# Security — zero-key contract
# ---------------------------------------------------------------------------


_SECRET_FIELD_HINTS = (
    "api_key",
    "api-key",
    "apikey",
    "secret",
    "password",
    "credential",
    "private_key",
    "auth_token",
    "bearer",
)


@pytest.mark.security
class TestTaskInfoZeroKeyContract:
    """``TaskInfo`` carries no credential-shaped attribute name.

    Mirrors the domain-type security check pattern used elsewhere in
    the codebase. ``TaskInfo`` lives in :mod:`api.tasks` rather than
    :mod:`core.types`, so it needs an explicit test here.
    """

    @pytest.mark.parametrize("cls", [TaskInfo, TaskProgress, FilingResult])
    def test_no_secret_looking_fields(self, cls: type) -> None:
        for f in fields(cls):
            lowered = f.name.lower()
            for hint in _SECRET_FIELD_HINTS:
                assert hint not in lowered, (
                    f"{cls.__name__}.{f.name} looks credential-bearing; "
                    "TaskManager dataclasses must not carry secrets."
                )

    def test_task_info_does_not_expose_edgar_identity(self) -> None:
        """Legacy ``TaskInfo`` carried ``edgar_name`` / ``edgar_email``.

        Identity now lives on the manager-level resolver dict. Verify
        the legacy field names are gone so a careless re-introduction
        trips the test.
        """
        field_names = {f.name for f in fields(TaskInfo)}
        assert "edgar_name" not in field_names
        assert "edgar_email" not in field_names


@pytest.mark.security
class TestTaskHistoryNoKeys:
    """``save_task_history`` never receives a key argument from the worker.

    The ``_StubRegistry.save_task_history`` asserts this inline for
    every test that triggers a terminal transition. The explicit case
    here proves the inline check is actually enforcing something.
    """

    def test_completed_task_save_call_has_no_secret_kwargs(self) -> None:
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        p1 = _make_processed_filing(f1)
        manager, _, registry, _, _ = _build_manager(
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(by_accession={f1.accession_number: p1}),
        )
        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        _wait_for_state(manager, task_id, target=TaskState.COMPLETED)
        assert registry.save_calls, "expected a save_task_history call on terminal transition"
        call = registry.save_calls[0]
        for forbidden in _SECRET_FIELD_HINTS:
            assert all(forbidden not in k.lower() for k in call)
