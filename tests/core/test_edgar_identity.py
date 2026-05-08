"""Tests for :mod:`sec_generative_search.core.edgar_identity`.

Coverage:

- :class:`EdgarIdentity` — frozen, dataclass, no slot for credentials.
- :func:`validate_edgar_name` / :func:`validate_edgar_email` —
  empty / overlong / control-char rejection; valid input passes through
  with whitespace stripped; error messages NEVER echo the rejected value.
- :class:`InMemorySessionEdgarIdentityStore` — set/get/delete; TTL-based
  lazy eviction; ``ttl_seconds <= 0`` rejection; concurrent-set semantics
  (replaces); audit-log discipline (only masked session-id tails appear,
  identity values NEVER appear).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from sec_generative_search.core.edgar_identity import (
    EDGAR_EMAIL_MAX_LEN,
    EDGAR_NAME_MAX_LEN,
    EdgarIdentity,
    InMemorySessionEdgarIdentityStore,
    validate_edgar_email,
    validate_edgar_name,
)

# ---------------------------------------------------------------------------
# Audit-log capture (matches the pattern used by test_credentials).
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_caplog(
    caplog: pytest.LogCaptureFixture,
) -> Iterator[pytest.LogCaptureFixture]:
    pkg_logger = logging.getLogger("sec_generative_search")
    previous_propagate = pkg_logger.propagate
    pkg_logger.propagate = True
    caplog.set_level(logging.WARNING, logger="sec_generative_search.security.audit")
    try:
        yield caplog
    finally:
        pkg_logger.propagate = previous_propagate


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestValidateEdgarName:
    def test_valid_name_passes(self) -> None:
        assert validate_edgar_name("Eren Yanic") == "Eren Yanic"

    def test_strips_surrounding_whitespace(self) -> None:
        assert validate_edgar_name("  Eren  ") == "Eren"

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_edgar_name("")

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_edgar_name("   ")

    def test_overlong_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_edgar_name("X" * (EDGAR_NAME_MAX_LEN + 1))

    def test_newline_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_edgar_name("Eren\nInjected: header")

    def test_carriage_return_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_edgar_name("Eren\rInjected: header")

    def test_null_byte_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_edgar_name("Eren\x00")

    def test_error_does_not_echo_value(self) -> None:
        # The supplied value might be PII or attacker-controlled —
        # neither should round-trip through ValueError messages.
        sentinel = "\nINJECTED_HEADER_VALUE"
        try:
            validate_edgar_name(f"Eren{sentinel}")
        except ValueError as exc:
            assert sentinel not in str(exc)
        else:
            pytest.fail("Expected ValueError")


@pytest.mark.security
class TestValidateEdgarEmail:
    def test_valid_email_passes(self) -> None:
        assert validate_edgar_email("user@example.com") == "user@example.com"

    def test_strips_whitespace(self) -> None:
        assert validate_edgar_email("  user@example.com  ") == "user@example.com"

    def test_missing_at_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_edgar_email("not-an-email")

    def test_missing_domain_dot_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_edgar_email("user@example")

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_edgar_email("")

    def test_overlong_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_edgar_email("a" * (EDGAR_EMAIL_MAX_LEN + 1) + "@example.com")

    def test_newline_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_edgar_email("user@example.com\nBcc: attacker@example.com")

    def test_internal_whitespace_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_edgar_email("us er@example.com")

    def test_error_does_not_echo_value(self) -> None:
        sentinel = "ATTACKER_INJECTED_EMAIL"
        try:
            validate_edgar_email(sentinel)
        except ValueError as exc:
            assert sentinel not in str(exc)
        else:
            pytest.fail("Expected ValueError")


# ---------------------------------------------------------------------------
# EdgarIdentity dataclass
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestEdgarIdentity:
    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        identity = EdgarIdentity.from_strings("Eren Yanic", "user@example.com")
        with pytest.raises(FrozenInstanceError):
            identity.name = "Other"  # type: ignore[misc]

    def test_from_strings_validates(self) -> None:
        # Embedded control char survives ``.strip()`` and is rejected.
        with pytest.raises(ValueError):
            EdgarIdentity.from_strings("Eren\nInjected: header", "user@example.com")
        with pytest.raises(ValueError):
            EdgarIdentity.from_strings("Eren", "not-an-email")

    def test_from_strings_strips_whitespace(self) -> None:
        identity = EdgarIdentity.from_strings("  Eren  ", "  user@example.com  ")
        assert identity.name == "Eren"
        assert identity.email == "user@example.com"

    def test_no_credential_shaped_field_names(self) -> None:
        # Belt + braces: `tests/core/test_types.py` enforces this on the
        # full domain model, but EdgarIdentity also lives in the "must
        # never leak" category.
        forbidden = {"api_key", "authorization", "bearer", "secret", "token"}
        for field_name in EdgarIdentity.__dataclass_fields__:
            assert field_name not in forbidden


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestEdgarIdentityStore:
    def test_set_and_get_round_trip(self) -> None:
        store = InMemorySessionEdgarIdentityStore()
        identity = EdgarIdentity.from_strings("Eren", "user@example.com")
        store.set("session-A", identity)
        assert store.get("session-A") == identity

    def test_get_unknown_returns_none(self) -> None:
        store = InMemorySessionEdgarIdentityStore()
        assert store.get("absent") is None

    def test_set_replaces_existing(self) -> None:
        store = InMemorySessionEdgarIdentityStore()
        first = EdgarIdentity.from_strings("First", "first@example.com")
        second = EdgarIdentity.from_strings("Second", "second@example.com")
        store.set("session-A", first)
        store.set("session-A", second)
        assert store.get("session-A") == second

    def test_delete_returns_true_when_present(self) -> None:
        store = InMemorySessionEdgarIdentityStore()
        store.set("session-A", EdgarIdentity.from_strings("X", "x@example.com"))
        assert store.delete("session-A") is True
        assert store.get("session-A") is None

    def test_delete_returns_false_when_absent(self) -> None:
        store = InMemorySessionEdgarIdentityStore()
        assert store.delete("absent") is False

    def test_per_session_isolation(self) -> None:
        store = InMemorySessionEdgarIdentityStore()
        a = EdgarIdentity.from_strings("Alice", "alice@example.com")
        b = EdgarIdentity.from_strings("Bob", "bob@example.com")
        store.set("session-A", a)
        store.set("session-B", b)
        assert store.get("session-A") == a
        assert store.get("session-B") == b
        store.delete("session-A")
        assert store.get("session-B") == b

    def test_ttl_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            InMemorySessionEdgarIdentityStore(ttl_seconds=0)

    def test_ttl_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            InMemorySessionEdgarIdentityStore(ttl_seconds=-1)

    def test_idle_eviction(self) -> None:
        clock = _FakeClock()
        store = InMemorySessionEdgarIdentityStore(ttl_seconds=60, clock=clock)
        store.set("session-A", EdgarIdentity.from_strings("Eren", "user@example.com"))

        # Just inside the window — still readable, refreshes touch time.
        clock.now += 30
        assert store.get("session-A") is not None

        # Cross the TTL after the refresh.
        clock.now += 61
        assert store.get("session-A") is None

    def test_get_refreshes_last_touched(self) -> None:
        clock = _FakeClock()
        store = InMemorySessionEdgarIdentityStore(ttl_seconds=60, clock=clock)
        store.set("session-A", EdgarIdentity.from_strings("Eren", "user@example.com"))

        # Repeated reads keep the entry alive past one TTL window.
        for _ in range(5):
            clock.now += 30
            assert store.get("session-A") is not None


# ---------------------------------------------------------------------------
# Audit-log discipline
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestEdgarIdentityAuditLog:
    def test_set_emits_audit_log(self, audit_caplog: pytest.LogCaptureFixture) -> None:
        store = InMemorySessionEdgarIdentityStore()
        store.set(
            "Z" * 43,
            EdgarIdentity.from_strings("Eren Yanic", "user@example.com"),
        )
        messages = [r.getMessage() for r in audit_caplog.records]
        assert any("edgar_identity_set" in m for m in messages)

    def test_set_does_not_log_name_or_email(
        self,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = InMemorySessionEdgarIdentityStore()
        sentinel_name = "EDGAR_NAME_SENTINEL"
        sentinel_email = "edgarsentinel@example.com"
        store.set(
            "Z" * 43,
            EdgarIdentity.from_strings(sentinel_name, sentinel_email),
        )
        messages = "\n".join(r.getMessage() for r in audit_caplog.records)
        assert sentinel_name not in messages
        assert sentinel_email not in messages

    def test_delete_emits_audit_log_when_present(
        self,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = InMemorySessionEdgarIdentityStore()
        store.set(
            "Z" * 43,
            EdgarIdentity.from_strings("Eren", "user@example.com"),
        )
        audit_caplog.clear()
        store.delete("Z" * 43)
        messages = [r.getMessage() for r in audit_caplog.records]
        assert any("edgar_identity_delete" in m for m in messages)

    def test_delete_silent_when_absent(
        self,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = InMemorySessionEdgarIdentityStore()
        store.delete("never-set")
        messages = [r.getMessage() for r in audit_caplog.records]
        # No deletion line — only the absence is observable to operators.
        assert not any("edgar_identity_delete" in m for m in messages)

    def test_set_emits_masked_session_id_tail(
        self,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = InMemorySessionEdgarIdentityStore()
        sid = "S" * 43  # well-shaped — guaranteed to mask, not show in full
        store.set(sid, EdgarIdentity.from_strings("Eren", "user@example.com"))
        messages = "\n".join(r.getMessage() for r in audit_caplog.records)
        assert sid not in messages
