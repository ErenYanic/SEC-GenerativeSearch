"""Security and privacy primitives for SEC-GenerativeSearch.

This module centralises small, dependency-free helpers that every other
module can import without pulling in the full settings hierarchy. It is
intentionally light on state so that it can be imported from logging,
config, and future API modules alike.

Contents:
        - :class:`DataTier` — formal data classification model.
        - :func:`mask_secret` — partial masking for log/error output.
        - :func:`secure_compare` — constant-time secret comparison.
    - :func:`sanitize_retrieved_context` — neutralise prompt-injection
            control tokens in retrieved filing text.

Design notes:
    - No runtime dependency on :mod:`pydantic_settings` — avoids a
      circular import path between settings and security helpers, and
      lets :mod:`core.logging` call :func:`mask_secret` without bootstrap
      ordering pain.
    - None of these primitives persist or emit the secret they touch.
      In particular, :func:`secure_compare` intentionally returns a
      bool — not an exception with a message — so unit tests and callers
      never handle a value that echoes the rejected input.
    - :func:`sanitize_retrieved_context` is **defence-in-depth**, not a
      silver bullet: prompt templates must still wrap the sanitised
      string in untrusted-data delimiters so the model sees an explicit
      trust boundary.  Cat-and-mouse pattern lists fail eventually; the
      template is the load-bearing control.
"""

from __future__ import annotations

import hmac
import re
from enum import Enum

__all__ = [
    "DataTier",
    "mask_secret",
    "sanitize_retrieved_context",
    "secure_compare",
]


class DataTier(Enum):
    """Formal data classification for every record handled by the system.

    Captured in code so downstream modules can reason about tiers by
    identity (``DataTier.USER_GENERATED``) instead
    of stringly-typed comparisons.  The tier that a value carries
    determines which controls apply — redaction at log sites, encryption
    at rest, allowed persistence sinks, and so on.

    Members:
        PUBLIC: Tier 1. Filing chunks, filing metadata, embedding
            vectors.  Derived from SEC EDGAR — publicly available and
            safe to persist, log, and return to any caller.
        APP_GENERATED: Tier 2. Application-generated records such as
            the filings registry, task history, and audit logs.  Not
            secrets, but disclosure to other tenants is undesirable in
            team/cloud deployments.  Encrypted at rest when SQLCipher
            is enabled.
        USER_GENERATED: Tier 3. Anything supplied by the user —
            search/RAG queries, prompts, provider API keys, and the
            EDGAR identity.  Must be redacted at log sites, never
            returned to other users, never embedded in frontend
            bundles, and never persisted unless explicitly opted in
            (see ``RAG_CHAT_HISTORY_ENABLED``).
    """

    PUBLIC = "public"
    APP_GENERATED = "app_generated"
    USER_GENERATED = "user_generated"


# ---------------------------------------------------------------------------
# Secret masking
# ---------------------------------------------------------------------------

# Number of trailing characters kept visible when a secret is masked.
# Matches the convention used by every provider console (OpenAI, Anthropic,
# Google, ...), which shows last-4 for key fingerprinting without
# disclosing enough bytes to be useful to an attacker.
_MASK_TAIL_LENGTH = 4

# Minimum secret length before any tail is shown.  Below this the value
# is masked in full — preserves safety when a user supplies a near-empty
# or truncated key by accident.
_MASK_MIN_LENGTH_FOR_TAIL = 8


def mask_secret(value: str | None, *, placeholder: str = "***") -> str:
    """Return a safe-to-log representation of *value*.

    Shows only the last :data:`_MASK_TAIL_LENGTH` characters of a
    reasonably-long secret, preceded by ``placeholder``.  Short or empty
    values are masked in full so that a truncated credential does not
    leak to log output.

    Args:
        value: The secret string.  ``None`` is treated as "unset".
        placeholder: The opaque prefix shown before the visible tail
            (default ``"***"``).

    Returns:
        A masked string suitable for logs, CLI/API error hints, and
        debug prints.  Never returns *value* unchanged for a non-empty
        input.

    Examples:
        >>> mask_secret("sk-proj-ABCDEFGHIJKLMNOP")
        '***MNOP'
        >>> mask_secret("tiny")
        '***'
        >>> mask_secret(None)
        '<unset>'
    """
    if value is None:
        return "<unset>"
    if len(value) < _MASK_MIN_LENGTH_FOR_TAIL:
        # Fully mask to avoid leaking a large fraction of a short value.
        return placeholder
    return f"{placeholder}{value[-_MASK_TAIL_LENGTH:]}"


# ---------------------------------------------------------------------------
# Constant-time secret comparison
# ---------------------------------------------------------------------------


def secure_compare(a: str | bytes | None, b: str | bytes | None) -> bool:
    """Compare two secrets in constant time.

    Thin wrapper over :func:`hmac.compare_digest` with the error-prone
    edges removed:

    - ``None`` on either side is a type mismatch and returns ``False``.
    - Mixed ``str`` / ``bytes`` operands return ``False`` rather than
      raising — callers must not have to wrap comparisons in
      ``try`` blocks.
    - ``str`` inputs are encoded as UTF-8 bytes before comparison so
      the underlying C routine always operates on a fixed type.

    Returns ``True`` iff the inputs are equal.  The runtime does not
    short-circuit on the first mismatching byte, which prevents the
    timing-oracle attacks that a naive ``a == b`` would enable for
    API keys and admin tokens.
    """
    if a is None or b is None:
        return False
    if type(a) is not type(b):
        return False
    if isinstance(a, str):
        # ``b`` has the same type at this point.
        return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    return hmac.compare_digest(a, b)


# ---------------------------------------------------------------------------
# Prompt-injection neutralisation for retrieved context
# ---------------------------------------------------------------------------

# Chat-template and instruction-control tokens that commonly appear in
# prompt-injection payloads.  Sourced from the public templates of the
# providers we target (OpenAI, Anthropic, Google) and from
# common open-source chat formats (Llama, Mistral, ChatML).  We
# neutralise — not strip — so the rewritten text remains readable in
# the UI's source panel while no longer being executable as a control
# directive.
_CONTROL_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # ChatML-style markers, e.g. <|system|>, <|user|>, <|im_start|>.
    (re.compile(r"<\|[^|>\r\n]{1,64}\|>"), "[sanitised-chatml]"),
    # Llama-2 system block markers.
    (re.compile(r"<<\s*SYS\s*>>", re.IGNORECASE), "[sanitised-sys-open]"),
    (re.compile(r"<<\s*/\s*SYS\s*>>", re.IGNORECASE), "[sanitised-sys-close]"),
    # Llama / Mistral instruction markers.
    (re.compile(r"\[\s*INST\s*\]", re.IGNORECASE), "[sanitised-inst-open]"),
    (re.compile(r"\[\s*/\s*INST\s*\]", re.IGNORECASE), "[sanitised-inst-close]"),
    # Anthropic conversation markers occasionally echoed back in docs.
    (re.compile(r"\b(?:Human|Assistant):", re.IGNORECASE), "[sanitised-role]"),
)


# Length cap for the sanitised text.  Filings can contain megabyte-scale
# exhibits; imposing a per-chunk cap here keeps an adversarial chunk
# from exploding downstream token budgets before the RAG orchestrator
# has a chance to measure it.  The cap is generous — a single retrieved
# chunk is expected to be well under this — and the truncation is
# visible to the caller so bugs are observable, not silent.
_MAX_SANITISED_CHARACTERS = 50_000
_TRUNCATION_SUFFIX = "\n[sanitised-truncated]"


def sanitize_retrieved_context(text: str) -> str:
    """Neutralise prompt-injection control tokens in a retrieved chunk.

    Designed for use by the RAG orchestrator on every retrieved
    chunk **before** it is interpolated into a prompt template.  The
    transform is purely textual and reversible by inspection — nothing
    is silently dropped.

    What it does:

    - Replaces known chat-template control tokens (ChatML ``<|...|>``,
      Llama ``<<SYS>>``/``[INST]``, echoed ``Human:``/``Assistant:``
      markers) with visible ``[sanitised-...]`` placeholders.
    - Caps the returned string at :data:`_MAX_SANITISED_CHARACTERS`
      characters and appends a visible truncation marker when the
      input exceeded the cap.

    What it **deliberately does not** do:

    - It does not attempt natural-language detection of instructions
      like "ignore previous instructions".  That is a cat-and-mouse
      pattern list that fails by construction.  The load-bearing control
      is the prompt template's trust boundary (see module docstring).
    - It does not strip or rewrite HTML, URLs, or filing-specific
      structure.  Filings routinely contain legal phrasing that would
      trip an aggressive filter.

    Args:
        text: Raw chunk text straight off the retrieval store.

    Returns:
        A sanitised string safe to interpolate into a user-visible
        prompt context block.  The output is deterministic for a given
        input, which keeps diffs stable across reruns.
    """
    if not text:
        return ""

    sanitised = text
    for pattern, replacement in _CONTROL_TOKEN_PATTERNS:
        sanitised = pattern.sub(replacement, sanitised)

    if len(sanitised) > _MAX_SANITISED_CHARACTERS:
        sanitised = sanitised[:_MAX_SANITISED_CHARACTERS] + _TRUNCATION_SUFFIX

    return sanitised
