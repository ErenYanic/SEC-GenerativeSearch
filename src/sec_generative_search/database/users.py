"""Per-user accounts + encrypted vault store.

Layers a small CRUD surface over the ``users`` table that migration v3
creates inside the SQLCipher-encrypted metadata database. The store is
the server-side seam for:

- enrolment (admin mints a one-time token; user POSTs ``salt_M`` +
  ``auth_hash`` + initial empty ``ciphertext_vault``),
- login (server validates ``auth_proof`` against ``auth_hash``; client
  decrypts the ciphertext blob locally),
- vault mutation (re-encrypted blob + fresh IV on every write),
- account lockout (consecutive-failure → soft-lock state machine).

Design notes:

- **Server zero-knowledge of password and KEK.** The store never
  receives the user password or the AES-GCM key. It only stores
  ``salt_M``, ``HMAC-SHA256(server_pepper, auth_proof)`` (the
  ``auth_hash`` column), and the opaque ciphertext blob.

- **Pepper is mandatory at runtime.** The store does NOT validate the
  pepper itself (that lives on
  :mod:`sec_generative_search.core.user_auth`); but the *construction*
  refuses to operate a non-empty ``users`` table when the pepper is
  missing — so a forgotten ``API_AUTH_PEPPER`` after a deployment
  rotation surfaces at lifespan startup, not at the first login.

- **Reuses the registry's connection.** Same intentional intra-package
  coupling as :class:`EncryptedCredentialStore`. The migration ensures
  the table exists before the store is constructed.

- **No second crypto layer over the vault blob.** AES-GCM under the
  browser-derived KEK is the load-bearing seal; SQLCipher's
  whole-database encryption is defence-in-depth around that. Layering
  Fernet would conflict with the "server holds no KEK" property.

- **Lockout window is sliding, not absolute.** A 15-minute window over
  consecutive failures means a slow drip of guesses (one every 16
  minutes) never trips the lock. Combined with the per-IP + per-
  username rate limits on ``POST /api/auth/login`` (5 rpm + 3 rpm), the
  attacker's effective rate is bounded well below the lockout floor.

- **No permanent lockout.** A permanent lock is a denial-of-service
  vector — an attacker who knows a username can simply spam the login
  endpoint with garbage proofs to keep the account perpetually locked.
  The soft 15-minute lock + the admin-tier early-clear seam
  (``POST /api/admin/users/{id}/unlock``) keep recovery cheap.

- **Login-side errors are opaque to callers.** Methods raise
  :class:`AuthError` (no subclass detail) when the password proof
  fails, the account is locked, or the user doesn't exist. Username
  enumeration via timing / wire shape is forbidden by the wire
  contract that the route layer enforces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sec_generative_search.config.settings import (
    ApiSettings,
    DatabaseSettings,
    get_settings,
)
from sec_generative_search.core.exceptions import (
    AuthError,
    ConfigurationError,
    DatabaseError,
)
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.security import mask_secret
from sec_generative_search.core.user_auth import (
    AUTH_HASH_BYTES,
    SALT_BYTES,
    derive_auth_hash,
    verify_auth_hash,
)
from sec_generative_search.database.metadata import MetadataRegistry

__all__ = [
    "DEFAULT_LOCKOUT_THRESHOLD",
    "DEFAULT_LOCKOUT_WINDOW_MINUTES",
    "UserRecord",
    "UserStore",
]

logger = get_logger(__name__)


#: Consecutive-failure threshold inside the sliding window — once the
#: counter reaches this, the next failure flips the row into the locked
#: state. 10 is the documented OWASP "moderate" bound; pairs naturally
#: with the per-IP + per-username rate-limit floors.
DEFAULT_LOCKOUT_THRESHOLD: int = 10

#: Width of the lockout window in minutes. Same value also doubles as
#: the lock duration: a hit-and-lock attacker pays 15 minutes per
#: 10-guess burst.
DEFAULT_LOCKOUT_WINDOW_MINUTES: int = 15


@dataclass(frozen=True)
class UserRecord:
    """A row from the ``users`` table without the auth proof or vault.

    Carries everything the login flow needs to render the response
    envelope EXCEPT the secrets — :meth:`UserStore.fetch_login_payload`
    is the sanctioned read site for the ciphertext blob, and the
    ``auth_hash`` never leaves the store at all.
    """

    id: int
    username: str
    salt_m: bytes
    kdf_algo: str
    pbkdf2_iterations: int
    created_at: str
    last_login: str | None
    failed_login_count: int
    locked_until: str | None
    must_enrol: bool


@dataclass(frozen=True)
class LoginPayload:
    """Server-side response envelope for a successful login.

    The route serialises this to the wire; the ciphertext blob and IV
    are opaque to the route layer.
    """

    user_id: int
    ciphertext_vault: bytes
    vault_iv: bytes


class UserStore:
    """SQLCipher-backed CRUD over the ``users`` table.

    Conforms to no Protocol — the surface is too purpose-specific to
    fit a generic store ABC. See module docstring for design rationale.
    """

    _TABLE = "users"

    def __init__(
        self,
        registry: MetadataRegistry,
        *,
        api_settings: ApiSettings | None = None,
        db_settings: DatabaseSettings | None = None,
        lockout_threshold: int = DEFAULT_LOCKOUT_THRESHOLD,
        lockout_window_minutes: int = DEFAULT_LOCKOUT_WINDOW_MINUTES,
    ) -> None:
        settings = get_settings()
        self._api_settings = api_settings or settings.api
        self._db_settings = db_settings or settings.database

        # Refuse construction when the migration prerequisites are not
        # met.  The ``users`` table holds user-tier secrets — its
        # confidentiality story rests on SQLCipher's whole-database
        # encryption + the pepper.  Either missing is a release-block.
        if not registry.encrypted:
            raise ConfigurationError(
                "UserStore refuses to operate without SQLCipher.  The "
                "users table holds auth_hash, ciphertext_vault, "
                "and salts — running it on a plaintext SQLite file would "
                "defeat the at-rest encryption story.  Configure "
                "DB_ENCRYPTION_KEY (or DB_ENCRYPTION_KEY_FILE)."
            )

        # Intentional intra-package coupling — see module docstring.
        self._registry = registry
        self._conn = registry._conn
        self._lock = registry._lock
        self._db_error = registry._db_error

        self._lockout_threshold = lockout_threshold
        self._lockout_window = timedelta(minutes=lockout_window_minutes)

        # The pepper-required-when-non-empty contract.  We refuse at
        # construction time so the lifespan startup surfaces the
        # misconfiguration; deferring to first-login would let an
        # operator deploy with a missing pepper and only notice at
        # 3am when the first user signs in.
        if self.count() > 0 and not self._api_settings.auth_pepper:
            raise ConfigurationError(
                "UserStore: the users table is non-empty but "
                "API_AUTH_PEPPER (or API_AUTH_PEPPER_FILE) is unset.  "
                "The auth surface cannot validate logins "
                "or mint enrolment tokens without the pepper.  Configure "
                "the pepper before restarting the API."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _row_to_record(row: object) -> UserRecord:
        # ``row`` is a ``sqlite3.Row`` — column-name indexing keeps the
        # call site readable.
        return UserRecord(
            id=row["id"],  # type: ignore[index]
            username=row["username"],  # type: ignore[index]
            salt_m=bytes(row["salt_m"]),  # type: ignore[index]
            kdf_algo=row["kdf_algo"],  # type: ignore[index]
            pbkdf2_iterations=row["pbkdf2_iterations"],  # type: ignore[index]
            created_at=row["created_at"],  # type: ignore[index]
            last_login=row["last_login"],  # type: ignore[index]
            failed_login_count=row["failed_login_count"],  # type: ignore[index]
            locked_until=row["locked_until"],  # type: ignore[index]
            must_enrol=bool(row["must_enrol"]),  # type: ignore[index]
        )

    def _pepper(self) -> str | None:
        """Cached settings-side pepper (refreshed on every call so
        operator rotation through a settings reload takes effect)."""
        return self._api_settings.auth_pepper

    def _check_locked(self, record: UserRecord, *, now: datetime | None = None) -> None:
        """Raise :class:`AuthError` if the account is currently locked.

        The lockout column carries an absolute timestamp — comparing
        against ``now`` avoids any drift from monotonic clocks. An
        already-expired ``locked_until`` is a soft state, not an error;
        :meth:`record_login_success` and :meth:`record_login_failure`
        are the seams that actually clear / refresh the column.
        """
        if record.locked_until is None:
            return
        try:
            until = datetime.fromisoformat(record.locked_until)
        except ValueError:
            # A corrupted column is treated as "not locked" for the
            # check but the row will be normalised on the next write.
            return
        current = now or datetime.now(UTC)
        if current < until:
            raise AuthError("Account is locked.")

    # ------------------------------------------------------------------
    # CRUD: enrolment / lookup / deletion
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of rows in the users table (for the pepper guard)."""
        try:
            with self._lock:
                row = self._conn.execute(
                    f"SELECT COUNT(*) FROM {self._TABLE}"  # noqa: S608 — constant table
                ).fetchone()
            return row[0]
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to count users",
                details=str(exc),
            ) from exc

    def create_user(
        self,
        *,
        username: str,
        salt_m: bytes,
        auth_proof: bytes,
        ciphertext_vault: bytes,
        vault_iv: bytes,
        kdf_algo: str,
        pbkdf2_iterations: int,
        enrolment_nonce: str | None = None,
    ) -> int:
        """Insert a new user row from a completed enrolment exchange.

        Stores ``auth_hash = HMAC(pepper, auth_proof)`` rather than the
        raw proof — the route handler hands us ``auth_proof`` from the
        wire and we apply the pepper here so a single seam owns the
        HMAC discipline.

        ``enrolment_nonce`` carries the (already-verified) token's
        nonce so a future replay attempt can be detected even after the
        row is created — the consume seam clears it as part of the
        first successful login.
        """
        if len(salt_m) != SALT_BYTES:
            raise ValueError(f"salt_m must be {SALT_BYTES} bytes; got {len(salt_m)}.")

        auth_hash = derive_auth_hash(auth_proof, self._pepper())
        now = self._now_iso()
        sql = (
            f"INSERT INTO {self._TABLE} "  # noqa: S608 — constant table
            "(username, salt_m, auth_hash, ciphertext_vault, vault_iv, "
            "kdf_algo, pbkdf2_iterations, created_at, must_enrol, enrolment_nonce) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)"
        )
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(
                    sql,
                    (
                        username,
                        salt_m,
                        auth_hash,
                        ciphertext_vault,
                        vault_iv,
                        kdf_algo,
                        pbkdf2_iterations,
                        now,
                        enrolment_nonce,
                    ),
                )
                user_id = cursor.lastrowid
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to create user",
                details=str(exc),
            ) from exc

        audit_log(
            "user_create",
            detail=f"user_id={user_id} username_tail={mask_secret(username)}",
        )
        return int(user_id)

    def get_by_username(self, username: str) -> UserRecord | None:
        sql = (
            f"SELECT id, username, salt_m, kdf_algo, pbkdf2_iterations, "  # noqa: S608
            f"created_at, last_login, failed_login_count, locked_until, "
            f"must_enrol FROM {self._TABLE} WHERE username = ? LIMIT 1"
        )
        try:
            with self._lock:
                row = self._conn.execute(sql, (username,)).fetchone()
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to read user",
                details=str(exc),
            ) from exc
        if row is None:
            return None
        return self._row_to_record(row)

    def get_by_id(self, user_id: int) -> UserRecord | None:
        sql = (
            f"SELECT id, username, salt_m, kdf_algo, pbkdf2_iterations, "  # noqa: S608
            f"created_at, last_login, failed_login_count, locked_until, "
            f"must_enrol FROM {self._TABLE} WHERE id = ? LIMIT 1"
        )
        try:
            with self._lock:
                row = self._conn.execute(sql, (user_id,)).fetchone()
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to read user by id",
                details=str(exc),
            ) from exc
        if row is None:
            return None
        return self._row_to_record(row)

    def delete_user(self, user_id: int) -> bool:
        """Hard delete a user row. Wipes vault + auth_hash + salt in one go.

        Operator-driven via ``DELETE /api/admin/users/{id}`` — the wipe
        is unambiguous and forces a fresh enrolment for that username.
        Returns ``True`` when a row was removed.
        """
        sql = f"DELETE FROM {self._TABLE} WHERE id = ?"  # noqa: S608
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(sql, (user_id,))
                removed = cursor.rowcount > 0
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to delete user",
                details=str(exc),
            ) from exc
        if removed:
            audit_log("user_delete", detail=f"user_id={user_id}")
        return removed

    # ------------------------------------------------------------------
    # Auth check + lockout state machine
    # ------------------------------------------------------------------

    def verify_login(
        self,
        username: str,
        auth_proof: bytes,
        *,
        now: datetime | None = None,
    ) -> LoginPayload:
        """Validate a login attempt and return the ciphertext payload.

        Single-source-of-truth for the per-attempt state machine:

        1. Resolve the row by username. Missing user → ``AuthError``.
        2. Refuse if the row is currently locked.
        3. Read the stored ``auth_hash`` and compare against
           ``derive_auth_hash(auth_proof, pepper)`` in constant time.
        4. On success: clear ``failed_login_count``, refresh
           ``last_login``, return the payload.
        5. On failure: increment ``failed_login_count``; flip
           ``locked_until`` if the threshold is reached.

        Wire-shape note: every failure mode raises the same opaque
        :class:`AuthError`. The route handler maps it to a single
        ``401`` envelope so a wire observer cannot distinguish "wrong
        password" from "no such user" from "locked".
        """
        current = now or datetime.now(UTC)
        record = self.get_by_username(username)
        if record is None:
            raise AuthError("Login refused.")

        self._check_locked(record, now=current)

        # Fetch the auth_hash via a separate dedicated SELECT — the
        # column is intentionally absent from :class:`UserRecord` so
        # the secret only crosses the seam at the comparison site.
        try:
            with self._lock:
                row = self._conn.execute(
                    f"SELECT auth_hash, ciphertext_vault, vault_iv "  # noqa: S608
                    f"FROM {self._TABLE} WHERE id = ? LIMIT 1",
                    (record.id,),
                ).fetchone()
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to read user auth state",
                details=str(exc),
            ) from exc
        if row is None:  # Race: row deleted between lookups.
            raise AuthError("Login refused.")

        stored_hash = bytes(row["auth_hash"])
        if len(stored_hash) != AUTH_HASH_BYTES:
            # Defensive: a corrupted column would otherwise short-circuit
            # the comparator. Treat it as a hard failure.
            self._record_failure(record.id, current)
            raise AuthError("Login refused.")

        if not verify_auth_hash(stored_hash, auth_proof, self._pepper()):
            self._record_failure(record.id, current)
            raise AuthError("Login refused.")

        self._record_success(record.id, current)
        return LoginPayload(
            user_id=record.id,
            ciphertext_vault=bytes(row["ciphertext_vault"]),
            vault_iv=bytes(row["vault_iv"]),
        )

    def _record_failure(self, user_id: int, now: datetime) -> None:
        """Increment the failure count, flip the lock at threshold."""
        sql_read = (
            f"SELECT failed_login_count FROM {self._TABLE} "  # noqa: S608
            f"WHERE id = ? LIMIT 1"
        )
        sql_update = (
            f"UPDATE {self._TABLE} SET failed_login_count = ?, "  # noqa: S608
            f"locked_until = ? WHERE id = ?"
        )
        try:
            with self._lock, self._conn:
                row = self._conn.execute(sql_read, (user_id,)).fetchone()
                if row is None:
                    return
                new_count = int(row["failed_login_count"]) + 1
                locked_until_iso: str | None = None
                if new_count >= self._lockout_threshold:
                    locked_until_iso = (now + self._lockout_window).isoformat()
                self._conn.execute(sql_update, (new_count, locked_until_iso, user_id))
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to record login failure",
                details=str(exc),
            ) from exc
        audit_log(
            "login_failure",
            detail=f"user_id={user_id} count={new_count} locked={locked_until_iso is not None}",
        )

    def _record_success(self, user_id: int, now: datetime) -> None:
        """Clear the failure count + lock; refresh ``last_login``.

        Also clears ``enrolment_nonce`` and ``must_enrol`` on first
        successful login — that closes the enrolment-token replay door
        even if the bearer somehow re-presents the token after the
        first login.
        """
        sql = (
            f"UPDATE {self._TABLE} SET failed_login_count = 0, "  # noqa: S608
            f"locked_until = NULL, last_login = ?, must_enrol = 0, "
            f"enrolment_nonce = NULL WHERE id = ?"
        )
        try:
            with self._lock, self._conn:
                self._conn.execute(sql, (now.isoformat(), user_id))
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to record login success",
                details=str(exc),
            ) from exc
        audit_log("login_success", detail=f"user_id={user_id}")

    def unlock(self, user_id: int) -> bool:
        """Admin-tier early-clear seam (``POST /api/admin/users/{id}/unlock``).

        Clears both the failed-login counter and the ``locked_until``
        column unconditionally. Returns ``True`` when the row existed.
        """
        sql = (
            f"UPDATE {self._TABLE} SET failed_login_count = 0, "  # noqa: S608
            f"locked_until = NULL WHERE id = ?"
        )
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(sql, (user_id,))
                cleared = cursor.rowcount > 0
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to unlock user",
                details=str(exc),
            ) from exc
        if cleared:
            audit_log("user_unlock", detail=f"user_id={user_id}")
        return cleared

    # ------------------------------------------------------------------
    # Vault + password mutation
    # ------------------------------------------------------------------

    def update_vault(
        self,
        user_id: int,
        *,
        ciphertext_vault: bytes,
        vault_iv: bytes,
    ) -> bool:
        """Replace the vault blob + IV under an existing auth_hash.

        Used both by the provider-key write path and by the EDGAR
        identity routes. The IV must always rotate (the route enforces
        a fresh 12-byte random IV on every call); this layer does not
        compare against the prior IV.
        """
        sql = (
            f"UPDATE {self._TABLE} SET ciphertext_vault = ?, vault_iv = ? "  # noqa: S608
            f"WHERE id = ?"
        )
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(sql, (ciphertext_vault, vault_iv, user_id))
                updated = cursor.rowcount > 0
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to update vault",
                details=str(exc),
            ) from exc
        if updated:
            audit_log("vault_update", detail=f"user_id={user_id}")
        return updated

    def update_password(
        self,
        user_id: int,
        *,
        salt_m: bytes,
        auth_proof: bytes,
        ciphertext_vault: bytes,
        vault_iv: bytes,
        kdf_algo: str,
        pbkdf2_iterations: int,
    ) -> bool:
        """Atomically replace ``salt_m`` / ``auth_hash`` / vault.

        Password change re-derives ``salt_M``, the KEK, and the
        ciphertext blob client-side; the server-side write must land
        all four columns in a single transaction so a partial update
        cannot lock the user out.

        ``kdf_algo`` is forward-only — the route layer enforces that
        the new algorithm sorts ``>=`` the current one (or is the same).
        """
        if len(salt_m) != SALT_BYTES:
            raise ValueError(f"salt_m must be {SALT_BYTES} bytes; got {len(salt_m)}.")

        auth_hash = derive_auth_hash(auth_proof, self._pepper())
        sql = (
            f"UPDATE {self._TABLE} SET salt_m = ?, auth_hash = ?, "  # noqa: S608
            f"ciphertext_vault = ?, vault_iv = ?, kdf_algo = ?, "
            f"pbkdf2_iterations = ? WHERE id = ?"
        )
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(
                    sql,
                    (
                        salt_m,
                        auth_hash,
                        ciphertext_vault,
                        vault_iv,
                        kdf_algo,
                        pbkdf2_iterations,
                        user_id,
                    ),
                )
                updated = cursor.rowcount > 0
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to update password",
                details=str(exc),
            ) from exc
        if updated:
            audit_log("password_change", detail=f"user_id={user_id}")
        return updated

    # ------------------------------------------------------------------
    # Vault read (for in-band EDGAR + provider-key seams)
    # ------------------------------------------------------------------

    def fetch_vault(self, user_id: int) -> LoginPayload | None:
        """Return the current ciphertext + IV for an authenticated user.

        Distinct from :meth:`verify_login` — this is the seam used by
        re-encryption flows. The route fetches the blob server-side,
        but the server cannot decrypt without the KEK. Returns ``None``
        for an unknown id.
        """
        sql = (
            f"SELECT ciphertext_vault, vault_iv FROM {self._TABLE} "  # noqa: S608
            f"WHERE id = ? LIMIT 1"
        )
        try:
            with self._lock:
                row = self._conn.execute(sql, (user_id,)).fetchone()
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to read vault",
                details=str(exc),
            ) from exc
        if row is None:
            return None
        return LoginPayload(
            user_id=user_id,
            ciphertext_vault=bytes(row["ciphertext_vault"]),
            vault_iv=bytes(row["vault_iv"]),
        )

    # ------------------------------------------------------------------
    # Enrolment token consume (single-use enforcement)
    # ------------------------------------------------------------------

    def consume_enrolment_nonce(self, user_id: int, nonce: str) -> bool:
        """Atomically clear ``enrolment_nonce`` if it matches ``nonce``.

        Returns ``True`` when the clear succeeded — used by the
        enrolment route to detect a replay. A second attempt (the same
        token presented twice) finds ``enrolment_nonce IS NULL`` and
        returns ``False``, which the route maps to
        ``409 enrolment_already_completed``.
        """
        sql = (
            f"UPDATE {self._TABLE} SET enrolment_nonce = NULL, "  # noqa: S608
            f"must_enrol = 0 WHERE id = ? AND enrolment_nonce = ?"
        )
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(sql, (user_id, nonce))
                consumed = cursor.rowcount > 0
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to consume enrolment nonce",
                details=str(exc),
            ) from exc
        return consumed

    def __repr__(self) -> str:
        return f"UserStore(encrypted={self._registry.encrypted!r})"
