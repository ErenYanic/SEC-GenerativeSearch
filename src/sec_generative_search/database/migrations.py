"""Forward-only SQLite schema versioning for :class:`MetadataRegistry`.

The metadata registry tracks its schema generation in a dedicated
``schema_version`` table — one row per applied version — so a database
opened by a newer build of the application can pick up unapplied
migrations on its own.  Adding a migration is a one-line append to
:data:`MIGRATIONS`; there is no decorator magic and no reflection.

The bootstrap path runs once per database open and recognises three
states:

1. **Brand-new** (no ``schema_version`` table, no ``filings`` table) —
   stamp v1 *without* running a migration body.  The idempotent
   ``CREATE TABLE IF NOT EXISTS`` path in
   :meth:`MetadataRegistry._create_table` produces the v1 shape, so
   there is nothing for a "migration to v1" to do.
2. **v1-shaped unstamped** (``filings`` and ``task_history`` already
   exist, no ``schema_version``) — also stamp v1 without a body run.
   This is the lossless upgrade path for production / dev databases
   that pre-date this phase: row counts and column data are
   preserved exactly.
3. **Malformed unstamped** (some v1 tables missing, others present) —
   refuse with :class:`DatabaseError` naming the missing table(s).
   The operator recovery is filesystem backup + recreate, mirroring
   the rollback-out-of-scope-by-design contract that
   :class:`ReindexService` adopts for ChromaDB.

The design is **forward-only**: there are no inverse migrations.  If a
migration misbehaves, the recovery is to restore from a filesystem
backup of the SQLite file.

ChromaDB is intentionally **out of scope** for this module — the
:class:`EmbedderStamp` plus the ``_MIGRATION_FLAG`` collection metadata
already version the vector store, and the refuse-with-``sec-rag manage
reindex``-hint contract is the recovery story.  Layering a second
mechanism there would duplicate the seal.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sec_generative_search.core.exceptions import DatabaseError

__all__ = [
    "MIGRATIONS",
    "MigrationFn",
    "apply_pending_migrations",
    "bootstrap_schema_version",
]


MigrationFn = Callable[[Any], None]
"""A migration body.  Receives the raw DB-API connection.

Every migration body must use ``?``-bound parameter binding rather
than f-string interpolation, regardless of whether the values flow
from settings, the registry, or a hard-coded constant — the only
exception is the ``CREATE TABLE`` DDL itself, which has no value
substitution.  The same convention is honoured by every other SQL
helper in the database package (see the ``# noqa: S608`` placeholder
trail in :mod:`sec_generative_search.database.metadata`).
"""


# The append-only migration registry.  Tuples are ``(version, body)``
# pairs in **strictly ascending** order; ``apply_pending_migrations``
# raises if that contract is violated.
#
# v1 is the *baseline* shape produced by
# :meth:`MetadataRegistry._create_table`.  It has no migration body of
# its own — the idempotent ``CREATE TABLE IF NOT EXISTS`` path covers
# both brand-new and v1-shaped-unstamped databases.  v2+ are the first
# entries that actually run SQL through this surface.


def _migrate_v2_provider_credentials(conn: Any) -> None:
    """Create the encrypted ``provider_credentials`` table.

    Holds user-supplied provider API keys for callers that opt in via
    ``DB_PERSIST_PROVIDER_CREDENTIALS=true``.  Encryption-at-rest is
    SQLCipher's whole-database encryption — there is no per-row crypto
    layer.  The table is created unconditionally by the migration so a
    deployment that later flips the toggle on does not need a separate
    "first-time create" branch; the
    :class:`~sec_generative_search.database.credentials.EncryptedCredentialStore`
    constructor still refuses to operate when the toggle is off, which
    is the load-bearing access gate.

    Schema rationale:

        - ``user_id`` is opaque to this layer.  The caller decides what it
            means (likely ``sha256(API_KEY)`` for the simple shared-key
            deployment, a real user record once auth lands).
        - ``api_key`` is stored as ``TEXT`` (the SQLCipher driver encrypts
            every page on disk).  No second crypto layer; SQLCipher is the
            single sealing seam.
    - Composite primary key ``(user_id, provider)`` enforces "at most
      one credential per (user, provider)" without a separate
      ``UNIQUE`` index.  ``REPLACE`` semantics on conflict are the
      standard upsert pattern used in the rest of the package.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS provider_credentials (
            user_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            api_key TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, provider)
        )
    """)


MIGRATIONS: tuple[tuple[int, MigrationFn], ...] = ((2, _migrate_v2_provider_credentials),)


_SCHEMA_VERSION_DDL = """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    )
"""


# Tables that define the v1 schema, used for the v1-shape detection
# branch of the bootstrap.  Kept in a tuple rather than inlined so the
# malformed-DB error message stays in lockstep with the manifest.
_V1_TABLES: tuple[str, ...] = ("filings", "task_history")


def bootstrap_schema_version(conn: Any) -> None:
    """Stamp an unstamped database at version 1, or refuse a malformed one.

    Idempotent — an already-stamped database is a no-op.  The caller
    is responsible for holding the registry's threading lock and
    wrapping the call in a transaction (``with conn:`` in DB-API
    terms); this function does neither itself, so the
    :meth:`MetadataRegistry._initialise_schema` orchestrator can run
    bootstrap, migrations, and table creation as a single atomic
    block.

    See the module docstring for the three states this function
    recognises.

    Args:
        conn: An open DB-API connection (``sqlite3`` or
            ``pysqlcipher3``).

    Raises:
        DatabaseError: When the database has some v1 tables but not
            others.  The error names the missing table(s) and points
            the operator at the backup-then-clear recovery path.
    """
    conn.execute(_SCHEMA_VERSION_DDL)

    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    if row is not None and row[0] is not None:
        return  # Already stamped — nothing to bootstrap.

    existing = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    present = [t for t in _V1_TABLES if t in existing]
    missing = [t for t in _V1_TABLES if t not in existing]

    # Brand-new (none of the v1 tables present) and v1-shaped (all
    # present) both stamp v1.  ``MetadataRegistry._create_table`` runs
    # immediately afterwards: it is a no-op on the v1-shaped path and
    # a single CREATE TABLE on the brand-new path.
    if not present:
        _stamp_version(conn, 1)
        return
    if not missing:
        _stamp_version(conn, 1)
        return

    raise DatabaseError(
        "Refusing to bootstrap a partially-populated database without a "
        "schema_version stamp. Missing table(s): "
        + ", ".join(missing)
        + ". Operator recovery: take a filesystem backup of the database "
        "file and recreate it, or restore from a stamped backup.",
    )


def apply_pending_migrations(
    conn: Any,
    *,
    migrations: tuple[tuple[int, MigrationFn], ...] | None = None,
) -> None:
    """Run every migration whose version is greater than the current.

    Each migration body is run inside the caller's transaction; the
    version stamp is inserted into ``schema_version`` *after* the body
    returns, so a crash mid-loop leaves the database at the last fully
    applied version (the next open will retry from there).

    The *migrations* parameter is exposed for tests that inject a fake
    ordered list to exercise the apply path; production callers omit
    it and pick up the module-level :data:`MIGRATIONS` default.  The
    tuple must be **strictly ascending** by version — a duplicate or
    backwards step raises :class:`DatabaseError` before any body runs.

    Args:
        conn: An open DB-API connection.  ``schema_version`` must
            already exist; call :func:`bootstrap_schema_version`
            first.
        migrations: Override for the module-level :data:`MIGRATIONS`
            tuple.  Test seam.

    Raises:
        DatabaseError: When *migrations* is not strictly ascending.
    """
    pending = MIGRATIONS if migrations is None else migrations

    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    current = row[0] if row is not None and row[0] is not None else 0

    prev_version = 0
    for version, body in pending:
        if version <= prev_version:
            raise DatabaseError(
                "Migration registry is not strictly ascending: "
                f"version {version} is not greater than {prev_version}.",
            )
        prev_version = version
        if version <= current:
            continue
        body(conn)
        _stamp_version(conn, version)


def _stamp_version(conn: Any, version: int) -> None:
    """Record *version* as applied in ``schema_version``."""
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (version, datetime.now(UTC).isoformat()),
    )
