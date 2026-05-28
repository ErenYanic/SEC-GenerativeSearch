"""Correlation-ID propagation primitives.

A *correlation ID* is a content-free, low-cardinality opaque token
minted once per inbound API request and threaded through the
middleware stack → background workers → retrieval → generation so that
every log record emitted while serving a single request can be stitched
back together in an aggregator.

Security contract:

    - The ID MUST stay content-free. It is either an opaque random
      token minted here, or an operator-supplied ``X-Request-ID`` that
      has passed :func:`validate_request_id`. It MUST NEVER be derived
      from — or made to carry — a ticker, query, accession number,
      ``user_id``, or ``session_id``.
    - Inbound ``X-Request-ID`` is honoured **only** after a strict shape
      check. The bounded alphabet rejects CR/LF (a log/header-injection
      vector) and control characters, and the length bound stops an
      attacker bloating every downstream log line.

The module is intentionally dependency-free (standard library only): it
sits below middleware in the import graph and is trivially unit-testable
without a FastAPI runtime. Do not add imports.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

__all__ = [
    "REQUEST_ID_PATTERN",
    "bind_correlation_id",
    "get_correlation_id",
    "new_correlation_id",
    "reset_correlation_id",
    "set_correlation_id",
    "validate_request_id",
]


# The active correlation ID for the current execution context. ``None``
# means "no request scope" (e.g. CLI, startup, an unbound worker) and
# renders as ``-`` in log output.
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


# Inbound ``X-Request-ID`` shape: 8..128 chars drawn from an alphabet of
# ASCII letters, digits, dash, and underscore. The alphabet rejects
# CR/LF and every other control character (header/log-injection
# vectors); the upper bound caps how much an attacker-supplied value can
# bloat every downstream log line.
REQUEST_ID_PATTERN = re.compile(r"\A[A-Za-z0-9_-]{8,128}\Z")


def new_correlation_id() -> str:
    """Mint a fresh opaque correlation ID (32 lowercase hex characters)."""
    return secrets.token_hex(16)


def validate_request_id(raw: str | None) -> str | None:
    """Return ``raw`` when it matches :data:`REQUEST_ID_PATTERN`, else ``None``.

    Used by the API middleware to decide whether an inbound
    ``X-Request-ID`` may be adopted as the request's correlation ID. A
    rejected value falls back to a freshly minted ID — the request is
    never refused on a malformed header.
    """
    if raw is None:
        return None
    return raw if REQUEST_ID_PATTERN.match(raw) else None


def get_correlation_id() -> str | None:
    """Return the correlation ID bound to the current context, or ``None``."""
    return _correlation_id.get()


def set_correlation_id(value: str) -> Token[str | None]:
    """Bind ``value`` as the current correlation ID; return the reset token."""
    return _correlation_id.set(value)


def reset_correlation_id(token: Token[str | None]) -> None:
    """Restore the correlation ID to the value captured by ``token``."""
    _correlation_id.reset(token)


@contextmanager
def bind_correlation_id(value: str | None) -> Iterator[None]:
    """Bind ``value`` for the duration of the ``with`` block.

    Background worker threads do not inherit the request's
    :class:`~contextvars.ContextVar` automatically, so the
    :class:`~sec_generative_search.api.tasks.TaskManager` worker captures
    the originating request's ID at enqueue time and re-binds it here.

    A ``None`` value still binds (clearing any inherited ID) so a worker
    spawned outside a request scope logs ``-`` rather than leaking a
    stale ID from whichever context happened to spawn the thread.
    """
    token = _correlation_id.set(value)
    try:
        yield
    finally:
        _correlation_id.reset(token)
