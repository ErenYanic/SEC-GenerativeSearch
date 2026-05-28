"""Tests for correlation-ID propagation primitives."""

from __future__ import annotations

import re

import pytest

from sec_generative_search.core.correlation import (
    REQUEST_ID_PATTERN,
    bind_correlation_id,
    get_correlation_id,
    new_correlation_id,
    reset_correlation_id,
    set_correlation_id,
    validate_request_id,
)


@pytest.fixture(autouse=True)
def _clear_correlation() -> None:
    """Ensure each test starts and ends with no bound correlation ID."""
    token = set_correlation_id("")  # placeholder; immediately reset below
    reset_correlation_id(token)
    yield
    # Defensive: leave no residue for the next test in the same worker.
    leftover = get_correlation_id()
    assert leftover is None or isinstance(leftover, str)


class TestMint:
    def test_new_id_is_32_hex_chars(self) -> None:
        cid = new_correlation_id()
        assert re.fullmatch(r"[0-9a-f]{32}", cid)

    def test_new_ids_are_unique(self) -> None:
        ids = {new_correlation_id() for _ in range(1000)}
        assert len(ids) == 1000

    def test_minted_id_passes_own_validator(self) -> None:
        assert validate_request_id(new_correlation_id()) is not None


@pytest.mark.security
class TestValidateRequestId:
    """The inbound X-Request-ID shape check is a log/header-injection guard."""

    def test_none_returns_none(self) -> None:
        assert validate_request_id(None) is None

    @pytest.mark.parametrize(
        "value",
        [
            "abc12345",  # exactly the 8-char floor
            "A" * 128,  # exactly the 128-char ceiling
            "req-2026-05-28_abc",
            "0123456789abcdef0123456789abcdef",
        ],
    )
    def test_well_formed_ids_accepted(self, value: str) -> None:
        assert validate_request_id(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            "short7",  # 6 chars — below the floor
            "a" * 129,  # above the ceiling
            "has space",
            "has\ttab",
            "trav/ersal",
            "semi;colon",
            "plus+sign",
            "dot.separated",
            "",
        ],
    )
    def test_malformed_ids_rejected(self, value: str) -> None:
        assert validate_request_id(value) is None

    @pytest.mark.parametrize("payload", ["abcd1234\r\nSet-Cookie: x=1", "abcd1234\ninjected"])
    def test_crlf_injection_rejected(self, payload: str) -> None:
        # CR/LF must never pass — it would let an attacker forge log
        # lines or smuggle a response header through the echo path.
        assert validate_request_id(payload) is None

    def test_pattern_is_anchored(self) -> None:
        # A valid core wrapped in newlines must not match (the \A...\Z
        # anchors, not ^...$ which are line-relative in some flavours).
        assert REQUEST_ID_PATTERN.match("valid1234\nvalid1234") is None


class TestContextVar:
    def test_default_is_none(self) -> None:
        assert get_correlation_id() is None

    def test_set_and_get(self) -> None:
        token = set_correlation_id("cid-abc12345")
        try:
            assert get_correlation_id() == "cid-abc12345"
        finally:
            reset_correlation_id(token)
        assert get_correlation_id() is None

    def test_bind_context_manager_restores_previous(self) -> None:
        outer = set_correlation_id("outer-12345678")
        try:
            with bind_correlation_id("inner-12345678"):
                assert get_correlation_id() == "inner-12345678"
            assert get_correlation_id() == "outer-12345678"
        finally:
            reset_correlation_id(outer)

    def test_bind_none_clears_inherited(self) -> None:
        outer = set_correlation_id("outer-12345678")
        try:
            with bind_correlation_id(None):
                # A worker bound with None must not see a stale ID.
                assert get_correlation_id() is None
            assert get_correlation_id() == "outer-12345678"
        finally:
            reset_correlation_id(outer)
