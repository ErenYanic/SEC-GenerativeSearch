"""Shared JSON-output plumbing for the operator CLI.

The flag flips each command between its existing Rich-rendered table /
panel output and a stable JSON shape that mirrors the corresponding API
response (allow-list lift, never a ``**asdict()`` splat).

Trust model and load-bearing rules:

1. JSON output goes to ``stdout`` via :func:`print` — never through
   :class:`rich.console.Console`.  Rich would inject ANSI escapes /
   wrapping that break machine parsing.
2. Error envelopes match the API shape ``{error, message, hint, details}``
   so a future consolidation can lift them verbatim.  ``hint`` and
   ``details`` are omitted when ``None`` so the envelope stays
   information-dense.
3. ``print_json`` writes a *single* JSON document followed by a single
   newline.  No pretty-printing — operators piping into ``jq`` /
   ``python -m json.tool`` already get their preferred formatting.
4. NEVER include credential-shaped fields in any JSON output.  The
   ``provider list`` JSON exposes a boolean ``key_resolves`` flag and
   nothing else — not even the masked tail (the masked tail is for
   *visual* operator feedback at the Rich table, where it cannot be
   accidentally piped into a log shipper).  A parametrised
   ``@pytest.mark.security`` test enforces this across every command.
5. The flag's surface is intentionally tight: ``text`` (default) and
    ``json``.  No ``--format yaml``, no ``--pretty``.  Adding more
    formats is a future decision, not a default.
"""

from __future__ import annotations

import json
import sys
from enum import StrEnum
from typing import Any

import typer

__all__ = [
    "OutputFormat",
    "coerce_output_format",
    "error_envelope",
    "is_json",
    "print_json",
]


class OutputFormat(StrEnum):
    """Operator-selected output format.

    ``text`` keeps the existing Rich-rendered output unchanged.
    ``json`` writes a single JSON document to stdout and suppresses
    Rich rendering.  No third value — adding one is a future-phase
    decision.
    """

    TEXT = "text"
    JSON = "json"


def coerce_output_format(value: str) -> OutputFormat:
    """Lift the ``--output`` flag onto :class:`OutputFormat` strictly.

    Fails closed on unknown values via :class:`typer.BadParameter`.
    Same discipline as ``cli/rag.py::_coerce_mode`` —
    silently coercing a typo to ``text`` would mask the error and the
    operator would never see it.
    """
    normalised = value.strip().lower()
    try:
        return OutputFormat(normalised)
    except ValueError as exc:
        valid = ", ".join(f.value for f in OutputFormat)
        raise typer.BadParameter(f"Invalid --output: {value!r}. Expected one of: {valid}.") from exc


def is_json(output: OutputFormat) -> bool:
    """Convenience predicate so callers do not import the enum directly.

    Equivalent to ``output is OutputFormat.JSON`` but reads naturally
    in conditionals (``if is_json(output): ...``).
    """
    return output is OutputFormat.JSON


def print_json(payload: dict[str, Any] | list[Any]) -> None:
    """Write a single JSON document to ``stdout`` and flush.

    Bypasses :class:`rich.console.Console` so the output is parseable
    by ``jq`` / ``python -m json.tool`` without ANSI escape stripping.
    ``ensure_ascii=False`` keeps non-ASCII text (Unicode tickers,
    non-English query plans) intact — operators piping the output into
    a UTF-8 log shipper want the raw bytes, not ``\\uXXXX`` escapes.

    The :func:`print` call writes to ``sys.stdout`` because Typer's
    ``CliRunner`` re-binds ``sys.stdout`` per invocation; the default
    ``file=sys.stdout`` resolves to the rebound stream so tests can
    capture the JSON.
    """
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def error_envelope(
    error: str,
    message: str,
    *,
    hint: str | None = None,
    details: str | None = None,
) -> dict[str, Any]:
    """Build a JSON error envelope matching the API shape.

    Shape mirrors :class:`api.errors.ErrorEnvelope`:
    ``{error, message, hint?, details?}``.  ``hint`` and ``details``
    are omitted entirely when ``None`` so the document does not carry
    ``null``-valued keys an operator's parser must handle separately.

    The envelope is *only* the data structure — the caller still owns
    the :func:`typer.Exit` raise (or the silent return).  Decoupling
    the shape from the exit lets a single failure path render text or
    JSON without duplicating the error message.
    """
    envelope: dict[str, Any] = {"error": error, "message": message}
    if hint is not None:
        envelope["hint"] = hint
    if details is not None:
        envelope["details"] = details
    return envelope
