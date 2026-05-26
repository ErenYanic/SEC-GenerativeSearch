"""Security tests for :mod:`sec_generative_search.core.user_auth`.

The module is the dependency-free seam behind the Phase-13.11 user-tier
auth surface. Every test in this file is tagged ``@pytest.mark.security``
because every primitive it exercises is load-bearing on the
authentication contract.
"""

from __future__ import annotations

import time

import pytest

from sec_generative_search.core.exceptions import (
    ConfigurationError,
    EnrolmentTokenError,
)
from sec_generative_search.core.user_auth import (
    AUTH_HASH_BYTES,
    SALT_BYTES,
    decoy_salt,
    derive_auth_hash,
    mint_enrolment_token,
    verify_auth_hash,
    verify_enrolment_token,
)

# ---------------------------------------------------------------------------
# Auth hash
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestDeriveAuthHash:
    def test_output_is_32_bytes(self) -> None:
        h = derive_auth_hash(b"a" * 32, "pepper-not-a-secret")
        assert len(h) == AUTH_HASH_BYTES == 32

    def test_deterministic_for_same_inputs(self) -> None:
        a = derive_auth_hash(b"proof", "pepper-not-a-secret")
        b = derive_auth_hash(b"proof", "pepper-not-a-secret")
        assert a == b

    def test_different_proof_yields_different_hash(self) -> None:
        a = derive_auth_hash(b"proof-a", "pepper-not-a-secret")
        b = derive_auth_hash(b"proof-b", "pepper-not-a-secret")
        assert a != b

    def test_different_pepper_yields_different_hash(self) -> None:
        """Pepper is HMAC key — rotating it must invalidate every stored hash."""
        a = derive_auth_hash(b"proof", "pepper-one")
        b = derive_auth_hash(b"proof", "pepper-two")
        assert a != b

    def test_missing_pepper_is_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="API_AUTH_PEPPER"):
            derive_auth_hash(b"proof", None)

    def test_empty_pepper_is_configuration_error(self) -> None:
        """Empty string from a stripped pepper file must fail loud, not HMAC under ``b''``."""
        with pytest.raises(ConfigurationError, match="API_AUTH_PEPPER"):
            derive_auth_hash(b"proof", "")


@pytest.mark.security
class TestVerifyAuthHash:
    def test_verify_matches_derive(self) -> None:
        stored = derive_auth_hash(b"proof", "pepper-not-a-secret")
        assert verify_auth_hash(stored, b"proof", "pepper-not-a-secret") is True

    def test_verify_rejects_wrong_proof(self) -> None:
        stored = derive_auth_hash(b"proof", "pepper-not-a-secret")
        assert verify_auth_hash(stored, b"different-proof", "pepper-not-a-secret") is False

    def test_verify_rejects_wrong_pepper(self) -> None:
        stored = derive_auth_hash(b"proof", "pepper-one")
        assert verify_auth_hash(stored, b"proof", "pepper-two") is False

    def test_verify_short_hash_returns_false(self) -> None:
        """Truncated DB read must not silently authenticate.

        ``secure_compare`` returns ``False`` on mismatched lengths; this
        test pins that property end-to-end so a future refactor cannot
        relax it.
        """
        stored = b"\x00" * 8  # wrong length
        assert verify_auth_hash(stored, b"proof", "pepper-not-a-secret") is False


# ---------------------------------------------------------------------------
# Decoy salt
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestDecoySalt:
    def test_output_is_16_bytes(self) -> None:
        s = decoy_salt("alice", "pepper-not-a-secret")
        assert len(s) == SALT_BYTES == 16

    def test_deterministic_for_same_username(self) -> None:
        """Same input must produce identical bytes — load-bearing for
        enumeration defence. If decoy salts varied across calls, an
        attacker could simply hit ``login-params`` twice and compare.
        """
        a = decoy_salt("alice", "pepper-not-a-secret")
        b = decoy_salt("alice", "pepper-not-a-secret")
        assert a == b

    def test_different_username_yields_different_salt(self) -> None:
        a = decoy_salt("alice", "pepper-not-a-secret")
        b = decoy_salt("bob", "pepper-not-a-secret")
        assert a != b

    def test_different_pepper_yields_different_salt(self) -> None:
        a = decoy_salt("alice", "pepper-one")
        b = decoy_salt("alice", "pepper-two")
        assert a != b

    def test_missing_pepper_is_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="API_AUTH_PEPPER"):
            decoy_salt("alice", None)

    def test_decoy_is_distinct_from_auth_hash(self) -> None:
        """Domain separation: a salt computed for the decoy purpose
        must NOT collide with an auth_hash computed over the same input,
        even with the same pepper as the key."""
        decoy = decoy_salt("alice", "pepper-not-a-secret")
        auth = derive_auth_hash(b"alice", "pepper-not-a-secret")
        # Decoy is 16 B; auth_hash is 32 B; even comparing the shared
        # prefix length these must not collide.
        assert decoy != auth[:SALT_BYTES]


# ---------------------------------------------------------------------------
# Enrolment tokens
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestEnrolmentToken:
    def test_round_trip(self) -> None:
        token = mint_enrolment_token("alice", "pepper-not-a-secret")
        payload = verify_enrolment_token(token, "pepper-not-a-secret")
        assert payload.username == "alice"
        assert payload.expires_at > int(time.time())
        assert payload.nonce  # non-empty

    def test_each_mint_has_a_unique_nonce(self) -> None:
        """Two tokens issued back-to-back must differ — the nonce is the
        side-table key the consume seam uses for single-use enforcement."""
        a = mint_enrolment_token("alice", "pepper-not-a-secret")
        b = mint_enrolment_token("alice", "pepper-not-a-secret")
        assert a != b
        pa = verify_enrolment_token(a, "pepper-not-a-secret")
        pb = verify_enrolment_token(b, "pepper-not-a-secret")
        assert pa.nonce != pb.nonce

    def test_signature_mismatch_rejected(self) -> None:
        """A token signed with one pepper must not verify under another."""
        token = mint_enrolment_token("alice", "pepper-one")
        with pytest.raises(EnrolmentTokenError):
            verify_enrolment_token(token, "pepper-two")

    def test_tampered_payload_rejected(self) -> None:
        token = mint_enrolment_token("alice", "pepper-not-a-secret")
        version, _payload_b64, sig_b64 = token.split(".")
        # Flip one bit of the payload by re-encoding a slightly different
        # username under the same envelope.  The sig was computed over
        # the original payload, so verification must fail.
        from base64 import urlsafe_b64encode

        forged = urlsafe_b64encode(b"mallory|9999999999|nonce").rstrip(b"=").decode("ascii")
        tampered = f"{version}.{forged}.{sig_b64}"
        with pytest.raises(EnrolmentTokenError):
            verify_enrolment_token(tampered, "pepper-not-a-secret")

    def test_malformed_envelope_rejected(self) -> None:
        with pytest.raises(EnrolmentTokenError):
            verify_enrolment_token("not-a-real-token", "pepper-not-a-secret")

    def test_unknown_version_rejected(self) -> None:
        token = mint_enrolment_token("alice", "pepper-not-a-secret")
        _, payload_b64, sig_b64 = token.split(".")
        future = f"v99.{payload_b64}.{sig_b64}"
        with pytest.raises(EnrolmentTokenError):
            verify_enrolment_token(future, "pepper-not-a-secret")

    def test_malformed_base64_rejected(self) -> None:
        token = mint_enrolment_token("alice", "pepper-not-a-secret")
        version, _, sig_b64 = token.split(".")
        bad = f"{version}.@@@bad-base64@@@.{sig_b64}"
        with pytest.raises(EnrolmentTokenError):
            verify_enrolment_token(bad, "pepper-not-a-secret")

    def test_expired_token_rejected(self) -> None:
        """Verify ``now`` injection: a token minted at t=0 must be
        rejected when ``now`` is past its expiry."""
        token = mint_enrolment_token("alice", "pepper-not-a-secret", ttl_seconds=60, now=1_000_000)
        with pytest.raises(EnrolmentTokenError, match="expired"):
            verify_enrolment_token(token, "pepper-not-a-secret", now=1_000_000 + 61)

    def test_token_at_expiry_boundary_rejected(self) -> None:
        """``now == expires_at`` must be treated as expired, not still-valid."""
        token = mint_enrolment_token("alice", "pepper-not-a-secret", ttl_seconds=60, now=1_000_000)
        with pytest.raises(EnrolmentTokenError, match="expired"):
            verify_enrolment_token(token, "pepper-not-a-secret", now=1_000_060)

    def test_ttl_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds"):
            mint_enrolment_token("alice", "pepper-not-a-secret", ttl_seconds=0)
        with pytest.raises(ValueError, match="ttl_seconds"):
            mint_enrolment_token("alice", "pepper-not-a-secret", ttl_seconds=-1)

    def test_missing_pepper_is_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="API_AUTH_PEPPER"):
            mint_enrolment_token("alice", None)
        token = mint_enrolment_token("alice", "pepper-not-a-secret")
        with pytest.raises(ConfigurationError, match="API_AUTH_PEPPER"):
            verify_enrolment_token(token, None)
