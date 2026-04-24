"""Dual-store coordinator for the ChromaDB + SQLite storage layer.

:class:`FilingStore` is the single seam that composes
:class:`ChromaDBClient` and :class:`MetadataRegistry` and enforces the
project's dual-store ordering rule.  The split is load-bearing:

- ChromaDB owns the chunks and their embeddings (big, slow, most
  likely to fail).
- SQLite owns the filing-level record (small, fast, source of truth
  for "does this filing exist?").

Two store paths are supported — the choice is the caller's:

- **Default (``register_if_new=False``)** — carry-over pattern from
  ``cli/ingest.py`` / ``api/tasks.py``: **ChromaDB first, then SQLite**.
  A SQLite failure rolls back the ChromaDB write so a retry does not
  produce duplicate vectors.  The caller is responsible for
  pre-checking duplicates (usually
  :meth:`MetadataRegistry.get_existing_accessions` before embedding)
  — the default path does not protect against two concurrent writers
  targeting the same accession.  Pre-check makes that case
  unreachable, which is cheaper than the extra round-trip a
  store-layer check would cost.

- **Atomic (``register_if_new=True``)** — SQLite claims the accession
  first (via :meth:`MetadataRegistry.register_filing_if_new`, which
  holds its internal lock across check-and-insert), then ChromaDB
  writes.  The order inversion is deliberate: it is the only way to
  keep concurrent writers from corrupting each other via
  rollback-by-accession on ChromaDB, since ChromaDB's ``add()``
  silently no-ops duplicate IDs rather than raising, so a losing
  writer's rollback in the default path would delete a winning
  writer's chunks.  On a ChromaDB failure after SQLite claim, the
  SQLite row is rolled back.

Deletes follow the ChromaDB-first order unconditionally — it mirrors
the "drop the expensive side first" intent of the write path and
keeps the two paths symmetric.  If ChromaDB delete fails, SQLite is
left untouched so the caller can retry the whole operation.  If SQLite
delete fails after ChromaDB succeeded, the error propagates; SQLite is
the source of truth, so an orphan SQLite row is what a retry will
clean up (ChromaDB delete is idempotent on empty state).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.core.logging import get_logger

if TYPE_CHECKING:
    from sec_generative_search.database.client import ChromaDBClient
    from sec_generative_search.database.metadata import MetadataRegistry
    from sec_generative_search.pipeline.orchestrator import ProcessedFiling

logger = get_logger(__name__)


class FilingStore:
    """Coordinator that writes and deletes across both backing stores.

    The store owns no state of its own beyond references to its two
    collaborators.  Every wiring site (pipeline, CLI, API lifespan,
    tests) constructs :class:`FilingStore` directly over a
    pre-configured :class:`ChromaDBClient` and :class:`MetadataRegistry`
    — there is no factory and no lazy collaborator construction.

    Example:
        >>> from sec_generative_search.database import (
        ...     ChromaDBClient, FilingStore, MetadataRegistry,
        ... )
        >>> chroma = ChromaDBClient(stamp)
        >>> registry = MetadataRegistry()
        >>> store = FilingStore(chroma, registry)
        >>> stored = store.store_filing(processed_filing, register_if_new=True)
        >>> store.delete_filing("0000320193-23-000077")
    """

    def __init__(
        self,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
    ) -> None:
        """Wire the coordinator over its two collaborators.

        Both collaborators must already be initialised.  The store does
        not open, seal, or close either of them — its lifetime is a
        subset of theirs.

        Args:
            chroma: A stamped :class:`ChromaDBClient`.
            registry: A :class:`MetadataRegistry` backed by SQLite or
                SQLCipher, per the deployment scenario.
        """
        self._chroma = chroma
        self._registry = registry

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def store_filing(
        self,
        processed_filing: ProcessedFiling,
        *,
        register_if_new: bool = False,
    ) -> bool:
        """Persist a processed filing across both stores.

        See module docstring for the distinction between the two
        paths.  Both paths roll back the earlier step on failure so
        the dual-store state stays consistent.

        Args:
            processed_filing: Output from :class:`PipelineOrchestrator`.
                Must carry non-``None`` embeddings; ChromaDB's own
                ``store_filing`` enforces that, and this method
                preserves the rejection.
            register_if_new: When ``True``, take the atomic SQLite-first
                path and treat duplicate accessions as a no-op
                (returns ``False``).  When ``False`` (the default),
                take the carry-over ChromaDB-first path and let any
                SQLite duplicate rise as :class:`DatabaseError` — the
                caller is responsible for pre-checking.

        Returns:
            ``True`` when the filing was newly stored.  ``False`` only
            when ``register_if_new=True`` and the accession was
            already registered.

        Raises:
            DatabaseError: ChromaDB or SQLite failed.  Whichever side
                succeeded first has been rolled back on a best-effort
                basis; rollback failures are logged at ``error`` but
                never shadow the original exception.
        """
        if register_if_new:
            return self._store_filing_atomic(processed_filing)
        self._store_filing_carry_over(processed_filing)
        return True

    def _store_filing_carry_over(self, processed_filing: ProcessedFiling) -> None:
        """ChromaDB first, then SQLite (default path).

        Preserves the carry-over pattern from the legacy CLI / API
        ingest code.  A SQLite failure triggers a ChromaDB rollback by
        accession so a retry is not blocked by orphan chunks.  Does
        **not** protect against concurrent writers targeting the same
        accession — the rollback would delete a winner's chunks
        because ChromaDB silently no-ops duplicate IDs on ``add()``.
        Callers that need concurrency safety must pass
        ``register_if_new=True`` instead.
        """
        accession = processed_filing.filing_id.accession_number

        self._chroma.store_filing(processed_filing)

        try:
            self._registry.register_filing(
                processed_filing.filing_id,
                processed_filing.ingest_result.chunk_count,
            )
        except DatabaseError:
            self._rollback_chroma(
                accession,
                reason="SQLite register failed",
            )
            raise

    def _store_filing_atomic(self, processed_filing: ProcessedFiling) -> bool:
        """SQLite first (atomic claim), then ChromaDB.

        ``register_filing_if_new`` holds the registry's internal lock
        across its check-and-insert, so concurrent callers converge on
        exactly one winner.  A losing caller sees ``False`` and makes
        no ChromaDB write — the order inversion is what prevents the
        duplicate-ID no-op trap that the default path falls into.

        A ChromaDB failure after the SQLite claim rolls back the
        SQLite row so the caller can retry without leaving an orphan
        record.
        """
        accession = processed_filing.filing_id.accession_number

        registered = self._registry.register_filing_if_new(
            processed_filing.filing_id,
            processed_filing.ingest_result.chunk_count,
        )
        if not registered:
            return False

        try:
            self._chroma.store_filing(processed_filing)
        except DatabaseError:
            self._rollback_sqlite(
                accession,
                reason="ChromaDB store failed after SQLite claim",
            )
            raise

        return True

    # ------------------------------------------------------------------
    # Delete path
    # ------------------------------------------------------------------

    def delete_filing(self, accession_number: str) -> bool:
        """Remove a filing from both stores.

        Ordering mirrors the default write path — ChromaDB first,
        SQLite second.  A ChromaDB failure leaves SQLite untouched so
        the caller can retry.  A SQLite failure after a successful
        ChromaDB delete propagates; SQLite is the source of truth, so
        a retry targets a still-existing SQLite row with an
        already-empty ChromaDB side (delete is idempotent on an empty
        collection slice).

        Args:
            accession_number: SEC accession number identifying the
                filing to remove.

        Returns:
            ``True`` when a SQLite row was removed, ``False`` when the
            accession was not in the registry to begin with.  ChromaDB
            deletes are treated as idempotent and do not contribute to
            the return value.

        Raises:
            DatabaseError: Either store failed.
        """
        self._chroma.delete_filing(accession_number)
        return self._registry.remove_filing(accession_number)

    def delete_filings_batch(self, accession_numbers: list[str]) -> int:
        """Batch-delete filings across both stores.

        Same ordering and failure semantics as :meth:`delete_filing`.
        Uses the batch methods on both collaborators to collapse N
        round-trips to one (ChromaDB ``$in`` and SQLite ``IN (…)``).

        Args:
            accession_numbers: Accession numbers to delete.  Empty
                lists short-circuit to ``0`` without touching either
                store.

        Returns:
            Number of SQLite rows removed.

        Raises:
            DatabaseError: Either store failed.
        """
        if not accession_numbers:
            return 0

        self._chroma.delete_filings_batch(accession_numbers)
        return self._registry.remove_filings_batch(accession_numbers)

    def clear_all(self) -> tuple[int, int]:
        """Remove everything from both stores.

        ChromaDB first (drops and re-seals the collection to keep the
        stamp invariant intact), SQLite second.  See
        :meth:`ChromaDBClient.clear_collection` for the re-seal
        rationale.

        Returns:
            ``(chunks_removed, filings_removed)`` — chunk count from
            ChromaDB and row count from SQLite.

        Raises:
            DatabaseError: Either store failed.
        """
        chunks_removed = self._chroma.clear_collection()
        filings_removed = self._registry.clear_all()
        return chunks_removed, filings_removed

    # ------------------------------------------------------------------
    # Internal rollback helpers
    # ------------------------------------------------------------------

    def _rollback_chroma(self, accession_number: str, *, reason: str) -> None:
        """Best-effort ChromaDB rollback for a failed SQLite step.

        Any :class:`DatabaseError` raised by the rollback is swallowed
        and logged at ``error``.  The caller re-raises the original
        failure so the root cause wins — a rollback failure is a
        secondary symptom, never the root cause.
        """
        try:
            self._chroma.delete_filing(accession_number)
            logger.info(
                "Rolled back ChromaDB chunks for %s (%s)",
                accession_number,
                reason,
            )
        except DatabaseError as exc:
            logger.error(
                "Rollback failed for %s after %s — ChromaDB may hold "
                "orphan chunks that the next delete will clean up: %s",
                accession_number,
                reason,
                exc.message,
            )

    def _rollback_sqlite(self, accession_number: str, *, reason: str) -> None:
        """Best-effort SQLite rollback for a failed ChromaDB step."""
        try:
            self._registry.remove_filing(accession_number)
            logger.info(
                "Rolled back SQLite registration for %s (%s)",
                accession_number,
                reason,
            )
        except DatabaseError as exc:
            logger.error(
                "Rollback failed for %s after %s — SQLite may hold an "
                "orphan row that the next delete will clean up: %s",
                accession_number,
                reason,
                exc.message,
            )
