"""User-tier authentication primitives.

Stdlib-only HMAC / token helpers behind the user-tier login + enrolment
surface. The module lives next to :mod:`core.security` and follows the
same dependency-free posture so it can be imported from anywhere
without dragging in :mod:`pydantic_settings` or the database layer.

Contents:

- :func:`derive_auth_hash` — server-side HMAC-SHA256 over the
  client-derived ``auth_proof`` with the deployment pepper.
- :func:`verify_auth_hash` — constant-time comparator over
  :func:`derive_auth_hash`.
- :func:`decoy_salt` — deterministic 16-byte salt for unknown
  usernames. Identical bytes across calls for the same input, so
  ``GET /api/auth/login-params`` returns the same shape and timing for
  real and unknown users alike.
- :func:`mint_enrolment_token` /
  :func:`verify_enrolment_token` — short-TTL signed tokens that an
  operator hands to a fresh user out of band. Single-use semantics are
  enforced by the route (consuming the token flips ``users.must_enrol``);
  this module is the pure crypto + envelope.

Design notes:

- **Server zero-knowledge of password and KEK.** The browser derives
  ``auth_proof`` from the password via PBKDF2 → HKDF and posts only
  ``auth_proof`` to the server. The server HMACs it with the pepper to
  obtain ``auth_hash``. Reversing ``auth_hash`` to the user's password
  requires recovering the PBKDF2 input — paid client-side at 600 000
  iterations. The pepper turns a stolen ``auth_hash`` column into a
  useless artefact for offline attack until the pepper also leaks.
- **No bcrypt / argon2 dependency.** The client-side PBKDF2 cost IS the
  work-factor; the server's job is only to store + compare. A second
  KDF layer here would re-introduce a dependency we deliberately do
  not need.
- **Decoy salts must be deterministic.** A random decoy per call would
  let an attacker distinguish real from unknown usernames by simply
  hitting ``login-params`` twice and comparing. The same input must
  yield the same output every time — :func:`decoy_salt` is HMAC over
  the username with the pepper, which gives that property for free.
- **Pepper is mandatory.** Every function refuses an empty / ``None``
  pepper at call time with :class:`ConfigurationError`. The default
  argument is intentionally absent so a forgotten ``settings.api.auth_pepper``
  resolution surfaces at the first auth call instead of silently
  HMACing under the empty string.
- **Single-use enrolment tokens.** :func:`verify_enrolment_token`
  returns the parsed payload on success; the caller is responsible for
  the consume step (flipping ``users.must_enrol`` or marking the token
  consumed via a side-table). Stateless tokens cannot enforce
  single-use on their own.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

from sec_generative_search.core.exceptions import (
    ConfigurationError,
    EnrolmentTokenError,
)
from sec_generative_search.core.security import secure_compare

__all__ = [
    "AUTH_HASH_BYTES",
    "DEFAULT_ENROLMENT_TTL_SECONDS",
    "SALT_BYTES",
    "EnrolmentTokenPayload",
    "decoy_salt",
    "derive_auth_hash",
    "mint_enrolment_token",
    "verify_auth_hash",
    "verify_enrolment_token",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Length in bytes of an ``auth_hash`` column value — fixed by
#: HMAC-SHA256's output width.  Stored as a fixed-width ``BLOB`` in the
#: ``users`` table so a row-level length mismatch surfaces at the
#: database layer rather than as a silent comparison failure.
AUTH_HASH_BYTES: int = hashlib.sha256().digest_size

#: Length in bytes of every salt this module produces — both the random
#: ``salt_M`` minted at enrolment (16 bytes from :func:`secrets.token_bytes`)
#: and the deterministic decoy salt for unknown usernames.  Picked to
#: match the OWASP PBKDF2 baseline that the browser side runs against
#: this salt.
SALT_BYTES: int = 16

#: Default TTL for enrolment tokens — 30 minutes.  Long enough for an
#: admin to share the link with a user out of band; short enough that a
#: stolen token has a small useful window.  Callers may override via
#: :func:`mint_enrolment_token`'s ``ttl_seconds`` argument.
DEFAULT_ENROLMENT_TTL_SECONDS: int = 30 * 60


# Token envelope version.  Bumped if the payload structure or signing
# domain ever changes — the verifier refuses unknown versions hard
# instead of silently accepting a future format.
_TOKEN_ENVELOPE_VERSION: str = "v1"

# Domain separators for the HMAC contexts inside this module.  Every
# HMAC the server computes is namespaced so a value computed for one
# purpose cannot be replayed against another — even with the same
# pepper as the key.
_DECOY_SALT_DOMAIN: bytes = b"sec-gs/auth-decoy-salt/v1"
_ENROLMENT_TOKEN_DOMAIN: bytes = b"sec-gs/enrolment-token/v1"
_AUTH_HASH_DOMAIN: bytes = b"sec-gs/auth-hash/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_pepper(pepper: str | None, *, caller: str) -> bytes:
    """Refuse a missing pepper at call time.

    Settings load coerces empty-string to ``None`` so both states reduce
    to the same refusal here.  The caller name is interpolated into the
    error so operators get an actionable trace.
    """
    if not pepper:
        raise ConfigurationError(
            f"{caller} requires API_AUTH_PEPPER (or API_AUTH_PEPPER_FILE) to "
            "be configured. The Phase-13.11 user-tier auth surface cannot "
            "compute auth_hash / decoy salts / enrolment tokens without a "
            "deployment-wide pepper."
        )
    return pepper.encode("utf-8")


def _b64url_encode(value: bytes) -> str:
    """URL-safe base64 without padding — same encoding as ``session_id``."""
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    """URL-safe base64 decode that tolerates the missing-padding form.

    Refuses any character outside the URL-safe alphabet by letting
    :func:`base64.urlsafe_b64decode` raise; the caller translates the
    raised :class:`binascii.Error` into an :class:`EnrolmentTokenError`.
    """
    padding = (-len(value)) % 4
    return base64.urlsafe_b64decode(value + ("=" * padding))


# ---------------------------------------------------------------------------
# Auth hash (login proof verification)
# ---------------------------------------------------------------------------


def derive_auth_hash(auth_proof: bytes, pepper: str | None) -> bytes:
    """Compute ``HMAC-SHA256(pepper, domain || auth_proof)``.

    ``auth_proof`` is the 32-byte HKDF output the browser derived from
    the user's password (the ``"sec-gs/auth/v1"`` context). This server
    seam wraps it under a domain-separated HMAC so the stored value
    cannot be replayed against any other HMAC site that happens to use
    the same pepper as the key.

    Raises:
        ConfigurationError: When ``pepper`` is ``None`` or empty.
    """
    key = _require_pepper(pepper, caller="derive_auth_hash")
    mac = hmac.new(key, _AUTH_HASH_DOMAIN + auth_proof, hashlib.sha256)
    return mac.digest()


def verify_auth_hash(
    stored_hash: bytes,
    auth_proof: bytes,
    pepper: str | None,
) -> bool:
    """Constant-time check that ``derive_auth_hash(auth_proof, pepper)`` matches ``stored_hash``.

    Wraps :func:`secure_compare` so callers never have to remember which
    comparator to reach for. ``stored_hash`` length validation is the
    storage layer's responsibility — but a mismatched length still
    returns ``False`` here via the underlying comparator.
    """
    candidate = derive_auth_hash(auth_proof, pepper)
    return secure_compare(stored_hash, candidate)


# ---------------------------------------------------------------------------
# Decoy salt (username enumeration defence)
# ---------------------------------------------------------------------------


def decoy_salt(username: str, pepper: str | None) -> bytes:
    """Return a deterministic 16-byte salt for an unknown username.

    ``GET /api/auth/login-params?username=alice`` returns the real
    ``salt_M`` for an enrolled Alice, but the same shape (and the same
    bytes on every call) for an unknown Alice — so an attacker cannot
    distinguish "no such user" from "yes, here is the salt".

    The output is HMAC-SHA256 of the username under the pepper, scoped
    to the decoy domain so it cannot be replayed as an ``auth_hash`` or
    an enrolment-token signature. Only the first :data:`SALT_BYTES`
    bytes of the HMAC are returned — sufficient entropy for the
    enumeration defence; the rest of the digest is discarded.

    Raises:
        ConfigurationError: When ``pepper`` is ``None`` or empty.
    """
    key = _require_pepper(pepper, caller="decoy_salt")
    mac = hmac.new(key, _DECOY_SALT_DOMAIN + username.encode("utf-8"), hashlib.sha256)
    return mac.digest()[:SALT_BYTES]


# ---------------------------------------------------------------------------
# Enrolment tokens (admin-driven user creation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnrolmentTokenPayload:
    """Decoded enrolment-token contents on a successful verify.

    The ``nonce`` field is opaque random bytes minted at issue time —
    it gives the consume-side a stable identifier for replay defence
    (the route may persist the nonce alongside ``must_enrol=false`` so
    a replayed token after consumption surfaces as 409 even before the
    user row is looked up).
    """

    username: str
    expires_at: int  # epoch seconds (UTC)
    nonce: str  # base64url, 16 bytes of entropy


def _enrolment_signing_input(version: str, payload: bytes) -> bytes:
    """The exact bytes signed by the HMAC inside the token envelope.

    The envelope version is included in the signing input so a future
    ``v2`` payload signed with the same pepper cannot be downgraded to
    ``v1`` semantics by an attacker stripping a byte at the boundary.
    """
    return _ENROLMENT_TOKEN_DOMAIN + version.encode("ascii") + b"|" + payload


def mint_enrolment_token(
    username: str,
    pepper: str | None,
    *,
    ttl_seconds: int = DEFAULT_ENROLMENT_TTL_SECONDS,
    now: float | None = None,
) -> str:
    """Mint a short-TTL signed enrolment token for ``username``.

    The returned token has the shape ``v1.<b64url_payload>.<b64url_sig>``;
    a future ``v2`` envelope can land alongside without touching this
    function. Tokens are stateless — single-use enforcement is the
    caller's responsibility (consume the nonce, flip
    ``users.must_enrol``).

    Args:
        username: The username to bind to the token. Stored verbatim;
            the route is responsible for prior shape validation.
        pepper: HMAC key (the deployment-wide auth pepper).
        ttl_seconds: How long the token is valid for. Defaults to
            :data:`DEFAULT_ENROLMENT_TTL_SECONDS` (30 min). The route
            layer may shorten this for high-security deployments.
        now: Injectable clock for tests. Production callers omit it.

    Raises:
        ConfigurationError: When ``pepper`` is ``None`` or empty.
        ValueError: When ``ttl_seconds`` is non-positive.
    """
    if ttl_seconds <= 0:
        raise ValueError(
            f"ttl_seconds must be > 0; got {ttl_seconds}. An enrolment "
            "token with a zero or negative TTL is expired at issue time."
        )
    key = _require_pepper(pepper, caller="mint_enrolment_token")

    issued_at = int(now if now is not None else time.time())
    expires_at = issued_at + ttl_seconds
    nonce = _b64url_encode(secrets.token_bytes(16))

    # Pipe separator is unambiguous because username shape (validated at
    # the route) excludes "|"; expiry is an integer.  Keeping the
    # payload textual makes log redaction trivial (it never holds raw
    # secrets — username + expiry + nonce are all already operator-
    # known).
    payload_text = f"{username}|{expires_at}|{nonce}"
    payload = payload_text.encode("utf-8")

    mac = hmac.new(
        key,
        _enrolment_signing_input(_TOKEN_ENVELOPE_VERSION, payload),
        hashlib.sha256,
    ).digest()

    return f"{_TOKEN_ENVELOPE_VERSION}.{_b64url_encode(payload)}.{_b64url_encode(mac)}"


def verify_enrolment_token(
    token: str,
    pepper: str | None,
    *,
    now: float | None = None,
) -> EnrolmentTokenPayload:
    """Verify ``token`` and return its decoded payload.

    Refuses unknown envelope versions, malformed base64, tampered
    signatures, and expired tokens — every failure raises
    :class:`EnrolmentTokenError` with no hint about which check failed.
    The route layer collapses every error to a single
    ``401 enrolment_token_invalid`` envelope.

    Args:
        token: The token as produced by :func:`mint_enrolment_token`.
        pepper: HMAC key (the deployment-wide auth pepper).
        now: Injectable clock for tests.

    Raises:
        ConfigurationError: When ``pepper`` is ``None`` or empty.
        EnrolmentTokenError: When the token is malformed, tampered, or
            expired.
    """
    key = _require_pepper(pepper, caller="verify_enrolment_token")

    parts = token.split(".")
    if len(parts) != 3:
        raise EnrolmentTokenError("Enrolment token is malformed.")
    version, payload_b64, sig_b64 = parts
    if version != _TOKEN_ENVELOPE_VERSION:
        raise EnrolmentTokenError("Enrolment token envelope version is unknown.")

    try:
        payload = _b64url_decode(payload_b64)
        signature = _b64url_decode(sig_b64)
    except (ValueError, TypeError) as exc:
        # ``binascii.Error`` is a ``ValueError`` subclass.
        raise EnrolmentTokenError("Enrolment token contains malformed base64.") from exc

    expected_sig = hmac.new(
        key,
        _enrolment_signing_input(version, payload),
        hashlib.sha256,
    ).digest()
    if not secure_compare(signature, expected_sig):
        raise EnrolmentTokenError("Enrolment token signature mismatch.")

    try:
        username, expires_at_text, nonce = payload.decode("utf-8").split("|")
        expires_at = int(expires_at_text)
    except (UnicodeDecodeError, ValueError) as exc:
        raise EnrolmentTokenError("Enrolment token payload is structurally invalid.") from exc

    current_epoch = int(now if now is not None else time.time())
    if current_epoch >= expires_at:
        raise EnrolmentTokenError("Enrolment token has expired.")

    return EnrolmentTokenPayload(
        username=username,
        expires_at=expires_at,
        nonce=nonce,
    )
