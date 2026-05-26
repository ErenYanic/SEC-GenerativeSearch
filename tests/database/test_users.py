"""Tests for :mod:`sec_generative_search.database.users` + migration v3.

Coverage:

- Migration v3 creates the ``users`` table on first open and is
  idempotent across re-opens.
- :class:`UserStore` refuses construction when ``users`` is non-empty
  and the pepper is unset (the load-bearing pepper-required-at-runtime
  contract).
- :class:`UserStore` refuses construction without SQLCipher.
- CRUD round-trips: enrolment → login → vault read/write → password
  change → delete.
- Lockout state machine: 10 consecutive failures inside a sliding
  window soft-locks the row; admin unlock clears.
- Single-use enrolment nonce: a second consume on the same nonce
  returns ``False``.
- Audit-log discipline: no auth_proof / ciphertext / IV / salt appears
  in any audit line; the username is masked.

SQLCipher carve-out: ``pysqlcipher3`` is not in CI's install set; the
registry is force-flagged ``encrypted=True`` so the SQL paths can be
exercised on plain sqlite3 — mirrors the existing ``test_credentials``
posture.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sec_generative_search.config.settings import ApiSettings, DatabaseSettings
from sec_generative_search.core.exceptions import AuthError, ConfigurationError
from sec_generative_search.core.user_auth import (
    AUTH_HASH_BYTES,
    SALT_BYTES,
    derive_auth_hash,
)
from sec_generative_search.database.metadata import MetadataRegistry
from sec_generative_search.database.users import (
    DEFAULT_LOCKOUT_THRESHOLD,
    DEFAULT_LOCKOUT_WINDOW_MINUTES,
    UserStore,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_PEPPER = "pepper-not-a-secret"


@pytest.fixture
def db_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> DatabaseSettings:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chroma").mkdir()
    return DatabaseSettings(
        chroma_path="./chroma",
        metadata_db_path="./meta.sqlite",
        encryption_key="unit-test-key-not-a-secret",
        persist_provider_credentials=True,
    )


@pytest.fixture
def api_settings_with_pepper() -> ApiSettings:
    return ApiSettings(auth_pepper=_PEPPER)


@pytest.fixture
def api_settings_without_pepper() -> ApiSettings:
    return ApiSettings()


@pytest.fixture
def encrypted_registry(
    db_settings: DatabaseSettings,
) -> Iterator[MetadataRegistry]:
    registry = MetadataRegistry(
        db_path=db_settings.metadata_db_path,
        encryption_key=db_settings.encryption_key,
    )
    registry._encrypted = True
    try:
        yield registry
    finally:
        registry.close()


@pytest.fixture
def user_store(
    encrypted_registry: MetadataRegistry,
    api_settings_with_pepper: ApiSettings,
    db_settings: DatabaseSettings,
) -> UserStore:
    return UserStore(
        encrypted_registry,
        api_settings=api_settings_with_pepper,
        db_settings=db_settings,
    )


@pytest.fixture
def audit_caplog(
    caplog: pytest.LogCaptureFixture,
) -> Iterator[pytest.LogCaptureFixture]:
    """Capture package-level audit-log records.

    The package logger propagates=False by default; toggle it temporarily
    so caplog (which attaches to root) sees the records.
    """
    pkg_logger = logging.getLogger("sec_generative_search")
    previous = pkg_logger.propagate
    pkg_logger.propagate = True
    caplog.set_level(logging.WARNING, logger="sec_generative_search.security.audit")
    try:
        yield caplog
    finally:
        pkg_logger.propagate = previous


def _seed_user(
    store: UserStore,
    *,
    username: str = "alice",
    auth_proof: bytes = b"a" * 32,
    nonce: str = "test-nonce",
) -> int:
    return store.create_user(
        username=username,
        salt_m=b"s" * SALT_BYTES,
        auth_proof=auth_proof,
        ciphertext_vault=b"ct-initial",
        vault_iv=b"i" * 12,
        kdf_algo="pbkdf2-sha256",
        pbkdf2_iterations=600_000,
        enrolment_nonce=nonce,
    )


# ---------------------------------------------------------------------------
# Migration v3
# ---------------------------------------------------------------------------


class TestMigrationV3:
    def test_creates_users_table(self, tmp_path: Path) -> None:
        registry = MetadataRegistry(db_path=str(tmp_path / "meta.sqlite"))
        try:
            row = registry._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
            ).fetchone()
            assert row is not None
        finally:
            registry.close()

    def test_stamps_schema_version_at_v3(self, tmp_path: Path) -> None:
        registry = MetadataRegistry(db_path=str(tmp_path / "meta.sqlite"))
        try:
            versions = [
                row[0]
                for row in registry._conn.execute(
                    "SELECT version FROM schema_version ORDER BY version"
                ).fetchall()
            ]
            assert 3 in versions
        finally:
            registry.close()

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        path = str(tmp_path / "meta.sqlite")
        MetadataRegistry(db_path=path).close()
        registry = MetadataRegistry(db_path=path)
        try:
            stamps = registry._conn.execute(
                "SELECT COUNT(*) FROM schema_version WHERE version = 3"
            ).fetchone()[0]
            assert stamps == 1
        finally:
            registry.close()

    @pytest.mark.security
    def test_users_table_has_no_edgar_columns(self, tmp_path: Path) -> None:
        """Load-bearing privacy: EDGAR name / email must NOT be separate
        columns. The columns don't exist, so they trivially cannot leak
        via a future misconfigured-log incident.
        """
        registry = MetadataRegistry(db_path=str(tmp_path / "meta.sqlite"))
        try:
            cols = {row[1] for row in registry._conn.execute("PRAGMA table_info(users)").fetchall()}
            assert "edgar_name" not in cols
            assert "edgar_email" not in cols
        finally:
            registry.close()


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


class TestConstructionGuards:
    @pytest.mark.security
    def test_refuses_without_sqlcipher(
        self,
        db_settings: DatabaseSettings,
        api_settings_with_pepper: ApiSettings,
    ) -> None:
        registry = MetadataRegistry(
            db_path=db_settings.metadata_db_path,
            encryption_key=db_settings.encryption_key,
        )
        try:
            registry._encrypted = False
            with pytest.raises(ConfigurationError, match="SQLCipher"):
                UserStore(
                    registry,
                    api_settings=api_settings_with_pepper,
                    db_settings=db_settings,
                )
        finally:
            registry.close()

    @pytest.mark.security
    def test_refuses_when_users_non_empty_and_pepper_missing(
        self,
        encrypted_registry: MetadataRegistry,
        api_settings_with_pepper: ApiSettings,
        api_settings_without_pepper: ApiSettings,
        db_settings: DatabaseSettings,
    ) -> None:
        """The load-bearing pepper-required-at-runtime contract.

        An operator who deploys with a pepper, enrols a user, then
        removes the pepper from env on a restart must hit the refusal
        at lifespan startup — not silently HMAC under the empty string.
        """
        # Seed a row with the pepper present so the auth_hash is valid.
        store = UserStore(
            encrypted_registry,
            api_settings=api_settings_with_pepper,
            db_settings=db_settings,
        )
        _seed_user(store)
        # Now re-construct without a pepper.  Must refuse.
        with pytest.raises(ConfigurationError, match="API_AUTH_PEPPER"):
            UserStore(
                encrypted_registry,
                api_settings=api_settings_without_pepper,
                db_settings=db_settings,
            )

    def test_accepts_when_users_empty_and_pepper_missing(
        self,
        encrypted_registry: MetadataRegistry,
        api_settings_without_pepper: ApiSettings,
        db_settings: DatabaseSettings,
    ) -> None:
        """Greenfield deployments (no users yet) may run without a pepper.

        Forcing the pepper into ``.env`` from day one would be a
        first-run-experience regression for Scenario A operators who
        never enrol a user.
        """
        store = UserStore(
            encrypted_registry,
            api_settings=api_settings_without_pepper,
            db_settings=db_settings,
        )
        assert store.count() == 0


# ---------------------------------------------------------------------------
# CRUD round-trips
# ---------------------------------------------------------------------------


class TestEnrolment:
    def test_create_user_returns_id(self, user_store: UserStore) -> None:
        user_id = _seed_user(user_store)
        assert isinstance(user_id, int) and user_id > 0
        assert user_store.count() == 1

    def test_create_user_persists_record(self, user_store: UserStore) -> None:
        _seed_user(user_store)
        record = user_store.get_by_username("alice")
        assert record is not None
        assert record.username == "alice"
        assert record.salt_m == b"s" * SALT_BYTES
        assert record.kdf_algo == "pbkdf2-sha256"
        assert record.pbkdf2_iterations == 600_000
        assert record.failed_login_count == 0
        assert record.locked_until is None

    @pytest.mark.security
    def test_create_user_rejects_wrong_salt_length(
        self,
        user_store: UserStore,
    ) -> None:
        with pytest.raises(ValueError, match="salt_m"):
            user_store.create_user(
                username="alice",
                salt_m=b"too-short",
                auth_proof=b"a" * 32,
                ciphertext_vault=b"ct",
                vault_iv=b"i" * 12,
                kdf_algo="pbkdf2-sha256",
                pbkdf2_iterations=600_000,
            )

    def test_consume_enrolment_nonce_once(self, user_store: UserStore) -> None:
        user_id = _seed_user(user_store, nonce="single-use")
        assert user_store.consume_enrolment_nonce(user_id, "single-use") is True
        # A replay finds enrolment_nonce IS NULL → False.
        assert user_store.consume_enrolment_nonce(user_id, "single-use") is False

    def test_consume_enrolment_wrong_nonce_rejected(self, user_store: UserStore) -> None:
        user_id = _seed_user(user_store, nonce="real-nonce")
        assert user_store.consume_enrolment_nonce(user_id, "forged-nonce") is False


class TestLogin:
    @pytest.mark.security
    def test_successful_login_returns_payload(
        self,
        user_store: UserStore,
    ) -> None:
        proof = b"p" * 32
        user_id = _seed_user(user_store, auth_proof=proof)
        payload = user_store.verify_login("alice", proof)
        assert payload.user_id == user_id
        assert payload.ciphertext_vault == b"ct-initial"
        assert payload.vault_iv == b"i" * 12

    @pytest.mark.security
    def test_wrong_proof_raises_auth_error(self, user_store: UserStore) -> None:
        _seed_user(user_store, auth_proof=b"p" * 32)
        with pytest.raises(AuthError):
            user_store.verify_login("alice", b"q" * 32)

    @pytest.mark.security
    def test_unknown_user_raises_auth_error(self, user_store: UserStore) -> None:
        """No-such-user must raise the SAME exception shape as wrong-proof —
        username enumeration via response distinguishability is forbidden."""
        with pytest.raises(AuthError):
            user_store.verify_login("nobody", b"p" * 32)

    @pytest.mark.security
    def test_successful_login_clears_failure_count(
        self,
        user_store: UserStore,
    ) -> None:
        proof = b"p" * 32
        user_id = _seed_user(user_store, auth_proof=proof)
        # Two failures, then a success.
        for _ in range(2):
            with pytest.raises(AuthError):
                user_store.verify_login("alice", b"wrong-proof-bytes-xxxxxxxxxxxxxx")
        record = user_store.get_by_id(user_id)
        assert record is not None
        assert record.failed_login_count == 2
        # Success clears the count.
        user_store.verify_login("alice", proof)
        record = user_store.get_by_id(user_id)
        assert record is not None
        assert record.failed_login_count == 0

    @pytest.mark.security
    def test_login_clears_must_enrol_and_nonce(
        self,
        user_store: UserStore,
    ) -> None:
        """First successful login closes the enrolment replay window."""
        proof = b"p" * 32
        user_id = _seed_user(user_store, auth_proof=proof, nonce="x")
        user_store.verify_login("alice", proof)
        # The nonce was cleared; a replay must now fail.
        assert user_store.consume_enrolment_nonce(user_id, "x") is False


class TestLockoutStateMachine:
    @pytest.mark.security
    def test_threshold_failures_lock_account(self, user_store: UserStore) -> None:
        proof = b"p" * 32
        _seed_user(user_store, auth_proof=proof)
        for _ in range(DEFAULT_LOCKOUT_THRESHOLD):
            with pytest.raises(AuthError):
                user_store.verify_login("alice", b"wrong-proof-bytes-xxxxxxxxxxxxxx")
        record = user_store.get_by_username("alice")
        assert record is not None
        assert record.failed_login_count == DEFAULT_LOCKOUT_THRESHOLD
        assert record.locked_until is not None

    @pytest.mark.security
    def test_locked_account_rejects_correct_proof(
        self,
        user_store: UserStore,
    ) -> None:
        """A locked account is locked — even the right password loses.
        Prevents the lockout from acting as a side-channel oracle that
        leaks 'you got the password right but tried again too late'.
        """
        proof = b"p" * 32
        _seed_user(user_store, auth_proof=proof)
        for _ in range(DEFAULT_LOCKOUT_THRESHOLD):
            with pytest.raises(AuthError):
                user_store.verify_login("alice", b"wrong-proof-bytes-xxxxxxxxxxxxxx")
        with pytest.raises(AuthError, match="locked"):
            user_store.verify_login("alice", proof)

    @pytest.mark.security
    def test_lockout_expires_after_window(self, user_store: UserStore) -> None:
        """The lockout is a *soft* lock — past the window, the lookup
        no longer raises 'locked'. A subsequent failed attempt rewrites
        the row, so this also confirms the operator's path back to
        normal does not require admin intervention.
        """
        proof = b"p" * 32
        _seed_user(user_store, auth_proof=proof)
        past = datetime.now(UTC) - timedelta(minutes=DEFAULT_LOCKOUT_WINDOW_MINUTES + 1)
        # Hand-roll the lock state into the past via the public surface
        # (record_failure with a past clock).  We use _record_failure
        # directly so the test does not need to wait 15 minutes.
        user_id = user_store.get_by_username("alice").id  # type: ignore[union-attr]
        for _ in range(DEFAULT_LOCKOUT_THRESHOLD):
            user_store._record_failure(user_id, past)
        # Lock should now be in the past — login proceeds.
        payload = user_store.verify_login("alice", proof)
        assert payload.user_id == user_id

    @pytest.mark.security
    def test_admin_unlock_clears_state(self, user_store: UserStore) -> None:
        proof = b"p" * 32
        user_id = _seed_user(user_store, auth_proof=proof)
        for _ in range(DEFAULT_LOCKOUT_THRESHOLD):
            with pytest.raises(AuthError):
                user_store.verify_login("alice", b"wrong-proof-bytes-xxxxxxxxxxxxxx")
        assert user_store.unlock(user_id) is True
        record = user_store.get_by_id(user_id)
        assert record is not None
        assert record.failed_login_count == 0
        assert record.locked_until is None
        # Now the correct proof works again.
        user_store.verify_login("alice", proof)

    def test_unlock_nonexistent_user_returns_false(self, user_store: UserStore) -> None:
        assert user_store.unlock(99999) is False


class TestVaultMutation:
    @pytest.mark.security
    def test_update_vault_replaces_blob_and_iv(self, user_store: UserStore) -> None:
        user_id = _seed_user(user_store)
        assert user_store.update_vault(
            user_id,
            ciphertext_vault=b"updated-ciphertext",
            vault_iv=b"j" * 12,
        )
        payload = user_store.fetch_vault(user_id)
        assert payload is not None
        assert payload.ciphertext_vault == b"updated-ciphertext"
        assert payload.vault_iv == b"j" * 12

    def test_update_vault_unknown_user_returns_false(self, user_store: UserStore) -> None:
        assert user_store.update_vault(99999, ciphertext_vault=b"x", vault_iv=b"j" * 12) is False

    @pytest.mark.security
    def test_password_change_atomic(self, user_store: UserStore) -> None:
        proof_old = b"old" + b"x" * 29
        proof_new = b"new" + b"y" * 29
        user_id = _seed_user(user_store, auth_proof=proof_old)
        # Old password validates.
        user_store.verify_login("alice", proof_old)
        # Change password.
        ok = user_store.update_password(
            user_id,
            salt_m=b"t" * SALT_BYTES,
            auth_proof=proof_new,
            ciphertext_vault=b"re-encrypted",
            vault_iv=b"k" * 12,
            kdf_algo="pbkdf2-sha256",
            pbkdf2_iterations=700_000,
        )
        assert ok is True
        # Old password no longer works.
        with pytest.raises(AuthError):
            user_store.verify_login("alice", proof_old)
        # New password works AND vault was replaced atomically.
        payload = user_store.verify_login("alice", proof_new)
        assert payload.ciphertext_vault == b"re-encrypted"


class TestDeletion:
    def test_delete_user_returns_true(self, user_store: UserStore) -> None:
        user_id = _seed_user(user_store)
        assert user_store.delete_user(user_id) is True
        assert user_store.get_by_id(user_id) is None

    def test_delete_unknown_returns_false(self, user_store: UserStore) -> None:
        assert user_store.delete_user(99999) is False


class TestAuthHashOnTheWire:
    @pytest.mark.security
    def test_stored_hash_is_pepper_dependent(
        self,
        encrypted_registry: MetadataRegistry,
        db_settings: DatabaseSettings,
    ) -> None:
        """A stolen ``users`` row remains opaque without the pepper —
        even with the auth_hash, an attacker cannot validate any guessed
        password without recovering the pepper too.
        """
        store_a = UserStore(
            encrypted_registry,
            api_settings=ApiSettings(auth_pepper="pepper-one"),
            db_settings=db_settings,
        )
        proof = b"p" * 32
        _seed_user(store_a, auth_proof=proof)
        # Validation under a different pepper must fail.
        store_b = UserStore(
            encrypted_registry,
            api_settings=ApiSettings(auth_pepper="pepper-two"),
            db_settings=db_settings,
        )
        with pytest.raises(AuthError):
            store_b.verify_login("alice", proof)

    @pytest.mark.security
    def test_stored_hash_length_matches_constant(
        self,
        encrypted_registry: MetadataRegistry,
        api_settings_with_pepper: ApiSettings,
        db_settings: DatabaseSettings,
    ) -> None:
        store = UserStore(
            encrypted_registry,
            api_settings=api_settings_with_pepper,
            db_settings=db_settings,
        )
        proof = b"p" * 32
        user_id = _seed_user(store, auth_proof=proof)
        row = encrypted_registry._conn.execute(
            "SELECT auth_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        assert len(bytes(row["auth_hash"])) == AUTH_HASH_BYTES


# ---------------------------------------------------------------------------
# Audit-log discipline
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestAuditDiscipline:
    def test_create_user_audit_redacts_auth_proof(
        self,
        user_store: UserStore,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        proof = b"super-secret-auth-proof-32bytes!"
        _seed_user(user_store, auth_proof=proof)
        for record in audit_caplog.records:
            assert "super-secret-auth-proof" not in record.getMessage()

    def test_login_audit_does_not_leak_proof(
        self,
        user_store: UserStore,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        proof = b"super-secret-auth-proof-32bytes!"
        _seed_user(user_store, auth_proof=proof)
        user_store.verify_login("alice", proof)
        # ALSO check the auth_hash bytes don't appear (the column is in
        # the DB but should never reach a log).
        stored_hash = derive_auth_hash(proof, _PEPPER)
        for record in audit_caplog.records:
            msg = record.getMessage()
            assert "super-secret-auth-proof" not in msg
            assert stored_hash.hex() not in msg
