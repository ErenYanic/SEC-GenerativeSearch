"""Per-session EDGAR identity store.

The SEC EDGAR API requires every request to carry an identifying
``User-Agent`` header (a name + an email address) so the SEC can throttle
abusive callers and reach a human if a deployment misbehaves.  In a
single-tenant local install (Scenario A) the identity comes from
:envvar:`EDGAR_IDENTITY_NAME` and :envvar:`EDGAR_IDENTITY_EMAIL`.  In
multi-tenant deployments (Scenarios B/C) every request must instead
carry the identity of the *acting* user, not a shared service account —
otherwise EDGAR's rate-limit token bucket and incident-response trail
collapse onto a single shared identity that obscures who actually drove
the abuse.

Two concerns live here:

1. **The :class:`EdgarIdentity` dataclass** — a frozen ``(name, email)``
   pair.  Frozen so the record cannot be mutated after handing it to a
   downstream caller, and dataclass-first per the project's domain
   model rule (Pydantic stays in settings + API schemas).

2. **The in-memory per-session store** — sliding-TTL, lazy eviction,
   threading-locked, opaque ``session_id`` key.  Mirrors
   :class:`InMemorySessionCredentialStore` so the operational surface
   (lifecycle, eviction, audit-log shape) stays familiar.  No background
   thread by deliberate analogy to the credential store: a leaked timer
   thread holding live identities is strictly worse than lazy eviction.

Audit-log discipline:

    - Every store touch emits a structured ``SECURITY_AUDIT:`` line
      carrying only the ``session_id`` tail.  The name and email are
      never logged, hashed, or echoed; the redaction is *suppression*,
      not masking — see :data:`api.access_log.SUPPRESSED_HEADER_NAMES`
      for the corresponding HTTP-layer rule.
    - The store does NOT log on every :meth:`get` (that would flood at
      request rate).  The route layer is responsible for emitting a
      single ``edgar_identity_resolved`` line per request when the
      resolver hits — the higher tier owns the audit cadence so the
      store can stay agnostic to caller cardinality.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.security import mask_secret

__all__ = [
    "EDGAR_EMAIL_MAX_LEN",
    "EDGAR_NAME_MAX_LEN",
    "EdgarIdentity",
    "InMemorySessionEdgarIdentityStore",
    "validate_edgar_email",
    "validate_edgar_name",
]


logger = get_logger(__name__)


# Length bounds keep a single identity from monopolising memory or
# overflowing log emit paths.  ``EDGAR_EMAIL_MAX_LEN`` matches RFC 5321's
# 254-character path-length cap; ``EDGAR_NAME_MAX_LEN`` is generous for
# real names without inviting abuse.
EDGAR_NAME_MAX_LEN = 128
EDGAR_EMAIL_MAX_LEN = 254


# A deliberately narrow email shape check.  Edge-case full RFC 5322
# parsing is out of scope — we only need to reject obvious junk
# (newlines, control characters, missing ``@``) before the value reaches
# the SEC's User-Agent header.  edgartools applies its own validation
# downstream; this is the boundary check.
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


# Reject control characters in the name.  CR / LF in particular would
# allow a header-injection style smuggle through any code path that
# echoed the name back as a header value.
_CONTROL_CHARS = frozenset(chr(c) for c in range(0x20)) | {chr(0x7F)}


# Default TTL for an idle session entry.  Mirrors the credential
# store's default so the cookie's ``Max-Age`` cannot outlive the
# identity that it points at.
_DEFAULT_SESSION_TTL_SECONDS = 60 * 60  # one hour


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_edgar_name(name: str) -> str:
    """Return a stripped, validated name or raise :class:`ValueError`.

    Rules: non-empty after strip, no control characters (CR/LF in
    particular), at most :data:`EDGAR_NAME_MAX_LEN` characters.  The
    error message NEVER echoes the supplied value — a malformed name
    might be PII or attacker-controlled and we treat both identically.
    """
    stripped = name.strip()
    if not stripped:
        raise ValueError("EDGAR name must be a non-empty string.")
    if len(stripped) > EDGAR_NAME_MAX_LEN:
        raise ValueError(f"EDGAR name exceeds maximum length of {EDGAR_NAME_MAX_LEN} characters.")
    if any(ch in _CONTROL_CHARS for ch in stripped):
        raise ValueError("EDGAR name contains control characters.")
    return stripped


def validate_edgar_email(email: str) -> str:
    """Return a stripped, validated email or raise :class:`ValueError`.

    Same suppression rule as :func:`validate_edgar_name` — errors do not
    echo the offending value.
    """
    stripped = email.strip()
    if not stripped:
        raise ValueError("EDGAR email must be a non-empty string.")
    if len(stripped) > EDGAR_EMAIL_MAX_LEN:
        raise ValueError(f"EDGAR email exceeds maximum length of {EDGAR_EMAIL_MAX_LEN} characters.")
    if any(ch in _CONTROL_CHARS for ch in stripped):
        raise ValueError("EDGAR email contains control characters.")
    if not _EMAIL_RE.match(stripped):
        raise ValueError("EDGAR email is not a syntactically valid address.")
    return stripped


# ---------------------------------------------------------------------------
# Identity dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EdgarIdentity:
    """A validated EDGAR ``(name, email)`` pair.

    Frozen so once a route resolves an identity it cannot be mutated by
    a downstream caller.  ``__repr__`` is ``@final`` only by virtue of
    the fields being public — but the values must NEVER reach a log
    record at any level. Treat ``str(identity)`` as
    PII-bearing and route every emission through the explicit log-site
    suppression rules.
    """

    name: str
    email: str

    @classmethod
    def from_strings(cls, name: str, email: str) -> EdgarIdentity:
        """Build a validated identity from raw header / form input."""
        return cls(name=validate_edgar_name(name), email=validate_edgar_email(email))


# ---------------------------------------------------------------------------
# In-memory session-scoped store
# ---------------------------------------------------------------------------


@dataclass
class _IdentityEntry:
    """One session's identity.  Mutable; held only inside the store's lock."""

    identity: EdgarIdentity
    last_touched: float


class InMemorySessionEdgarIdentityStore:
    """Process-local EDGAR identity store, keyed by an opaque ``session_id``.

    Mirrors :class:`InMemorySessionCredentialStore` deliberately:

        - Sliding TTL, lazy eviction (no background thread): per-key on
          access plus an amortised global sweep (:meth:`_maybe_sweep_locked`)
          so abandoned sessions' PII identities do not linger past TTL.
        - Single :class:`threading.Lock` serialises every operation.
        - Opaque ``session_id`` strings — caller owns minting.
        - Audit log emits only the masked session-id tail; the identity
          itself is never logged.

    The store deliberately holds *one* identity per session: an EDGAR
    request requires exactly one active identity at a time, and offering
    a per-provider dimension would invite operators to "rotate" within a
    session — which masks who actually drove the request.  For session
    rotation the route layer should call :meth:`delete` on the prior
    session and :meth:`set` on the fresh one.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = _DEFAULT_SESSION_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(
                f"ttl_seconds must be > 0; got {ttl_seconds}. "
                "An always-expired store is never useful."
            )
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, _IdentityEntry] = {}
        # Monotonic timestamp of the last full sweep — seeded at
        # construction so the first sweep cannot fire before one TTL
        # window elapses.  See :meth:`_maybe_sweep_locked`.
        self._last_sweep = self._clock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_sweep_locked(self) -> None:
        """Amortised full sweep of expired identities.  Caller MUST hold the lock.

        Mirrors :meth:`InMemorySessionCredentialStore._maybe_sweep_locked`.
        Per-key eviction (:meth:`_evict_if_expired`) only collects a
        session that is *accessed again*; an abandoned session would
        otherwise keep its (PII-bearing) EDGAR name + email resident in
        RAM until the process restarts, and leave the entry count
        unbounded.  This runs at most one ``O(sessions)`` pass per TTL
        window, piggybacked on the lock every public operation already
        takes — no background thread (the deliberate constraint in the
        class docstring stands).  An abandoned entry is therefore evicted
        within ``2 * ttl_seconds`` of its last use.
        """
        now = self._clock()
        if (now - self._last_sweep) < self._ttl_seconds:
            return
        self._last_sweep = now
        expired = [
            key_id
            for key_id, entry in self._entries.items()
            if (now - entry.last_touched) > self._ttl_seconds
        ]
        for key_id in expired:
            del self._entries[key_id]

    def _evict_if_expired(self, key_id: str) -> None:
        """Drop the entry if its TTL has elapsed.  Caller MUST hold the lock."""
        entry = self._entries.get(key_id)
        if entry is None:
            return
        if (self._clock() - entry.last_touched) > self._ttl_seconds:
            del self._entries[key_id]

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def get(self, key_id: str) -> EdgarIdentity | None:
        """Return the stored identity, or ``None`` if absent / evicted.

        A successful read refreshes ``last_touched`` so an active
        session does not roll over to TTL eviction.
        """
        with self._lock:
            self._maybe_sweep_locked()
            self._evict_if_expired(key_id)
            entry = self._entries.get(key_id)
            if entry is None:
                return None
            entry.last_touched = self._clock()
            return entry.identity

    def set(self, key_id: str, identity: EdgarIdentity) -> None:
        """Store ``identity`` for ``key_id``, replacing any prior value.

        Audit-log carries only the masked session-id tail.  The name and
        email are NEVER logged.
        """
        with self._lock:
            self._maybe_sweep_locked()
            self._entries[key_id] = _IdentityEntry(
                identity=identity,
                last_touched=self._clock(),
            )
        audit_log(
            "edgar_identity_set",
            detail=(f"store=in_memory_session session_id_tail={mask_secret(key_id)}"),
        )

    def delete(self, key_id: str) -> bool:
        """Remove the stored identity.  Returns ``True`` iff one was removed."""
        with self._lock:
            self._maybe_sweep_locked()
            entry = self._entries.pop(key_id, None)
        removed = entry is not None
        if removed:
            audit_log(
                "edgar_identity_delete",
                detail=(f"store=in_memory_session session_id_tail={mask_secret(key_id)}"),
            )
        return removed
