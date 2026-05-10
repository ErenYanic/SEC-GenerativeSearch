"""ASGI middleware stack for the FastAPI surface.

Stack ordering matters and is documented at the call site
(:func:`sec_generative_search.api.app.create_app`).  Read top-to-bottom
of the ``add_middleware`` block as outermost-to-innermost — the
*last-added* middleware is the *first* to see an inbound request.

Components in this module:

    - :class:`SecurityHeadersMiddleware` — pure ASGI; appends fixed
      response headers (CSP, X-Frame-Options, ...).
    - :class:`ContentSizeLimitMiddleware` — pure ASGI; rejects oversize
      bodies before they reach a handler, defends against chunked
      transfer encoding mid-stream.
    - :class:`InsecureTransportWarningMiddleware` — logs a one-shot
      warning when authenticated traffic arrives over plain HTTP.
        - :class:`RateLimitMiddleware` — per-IP sliding-window limiter with
            a separate per-``session_id`` window enforced on top of the per-IP
            window for the provider-validation route. Both windows must allow
            the request; exhausting either returns 429.

CORS is supplied by Starlette's :class:`CORSMiddleware` directly; we do
not wrap it.

Security notes:

    - All four bespoke middlewares are dependency-free outside the
      Python standard library, FastAPI, and ``core.logging``.  Do not
      add provider, database, or chromadb imports here.
    - The error responses sent from middleware MUST carry the same
      :data:`~sec_generative_search.api.errors.ErrorEnvelope` shape as
      handler-raised errors so downstream consumers do not need to
      branch on response source.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sec_generative_search.api.errors import envelope
from sec_generative_search.api.policies import resolve_policy
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.logging import get_logger

# Imported lazily inside the dispatch path so the middleware module
# stays cheap at import time. The dependency is one-way.
_SESSION_COOKIE_LAZY: str | None = None


def _session_cookie_name() -> str:
    """Return the session-cookie name without forcing an early import.

    The cookie name lives in ``api.dependencies``, which itself does not
    import middleware. Read it once and cache so the per-request rate
    limit path stays O(1).
    """
    global _SESSION_COOKIE_LAZY
    if _SESSION_COOKIE_LAZY is None:
        from sec_generative_search.api.dependencies import SESSION_COOKIE_NAME

        _SESSION_COOKIE_LAZY = SESSION_COOKIE_NAME
    return _SESSION_COOKIE_LAZY


__all__ = [
    "DEFAULT_MAX_CONTENT_LENGTH",
    "ContentSizeLimitMiddleware",
    "InsecureTransportWarningMiddleware",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware",
]


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


_CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "base-uri 'self'",
        "frame-ancestors 'none'",
        "img-src 'self' data: blob: https:",
        "font-src 'self' data: https:",
        "style-src 'self' 'unsafe-inline'",
        "script-src 'self'",
        "connect-src 'self' ws: wss:",
    ]
)

_PERMISSIONS_POLICY = (
    "camera=(), microphone=(), geolocation=(), payment=(), usb=(), interest-cohort=()"
)

_SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"x-xss-protection", b"1; mode=block"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"content-security-policy", _CONTENT_SECURITY_POLICY.encode()),
    (b"permissions-policy", _PERMISSIONS_POLICY.encode()),
)


class SecurityHeadersMiddleware:
    """Append a fixed set of security headers to every HTTP response.

    Pure ASGI — intercepts the ``http.response.start`` message and
    extends the headers list directly.  Avoids Starlette's
    ``BaseHTTPMiddleware`` threadpool dispatch, which adds latency for
    a no-op pass-through.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # Filter out any header we are about to set so the final
                # response carries exactly one of each (callers that try
                # to override are silently corrected, not duplicated).
                forced_names = {name for name, _ in _SECURITY_HEADERS}
                headers = [(n, v) for n, v in headers if n not in forced_names]
                headers.extend(_SECURITY_HEADERS)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


# ---------------------------------------------------------------------------
# Content size limit
# ---------------------------------------------------------------------------


# 1 MiB.  Matches the nginx ``client_max_body_size`` we expect at the
# reverse proxy; the in-process check is defence in depth for direct
# uvicorn access (local dev without a proxy).
DEFAULT_MAX_CONTENT_LENGTH = 1 * 1024 * 1024


class ContentSizeLimitMiddleware:
    """Reject oversize request bodies, per-route.

    The effective bound is the **smaller** of:

    1. The constructor's ``max_bytes`` (global ceiling — kept for the
       scenarios in which an operator wants a tighter cap than any
       route's table entry, e.g. an ingress proxy is already enforcing
       a smaller limit and we want consistency below it).
    2. The route policy's ``max_body_bytes`` resolved from
       :func:`~sec_generative_search.api.policies.resolve_policy`.

    Two layers of defence inside the chosen bound:

    1. *Content-Length check* — short-circuits before reading the body
       when the declared size already exceeds the limit.
    2. *Stream-counting wrapper* — intercepts ``http.request`` messages
       and tallies the bytes received so far.  Rejects mid-stream when
       the tally exceeds the limit, which catches chunked transfer
       encoding (no ``Content-Length`` header).
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int = DEFAULT_MAX_CONTENT_LENGTH) -> None:
        self.app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Resolve the per-route cap and clamp by the global ceiling
        # so a tighter constructor-supplied bound always wins.
        path = scope.get("path", "")
        method = scope.get("method", "")
        policy_max = resolve_policy(path, method).max_body_bytes
        effective_max = min(policy_max, self._max_bytes)

        headers = dict(scope.get("headers", []))
        content_length_raw = headers.get(b"content-length")

        if content_length_raw is not None:
            try:
                length = int(content_length_raw)
            except (ValueError, UnicodeDecodeError):
                await self._send_payload_error(
                    send,
                    status=400,
                    error="invalid_content_length",
                    message="Invalid Content-Length header.",
                    hint="Send Content-Length as an unsigned decimal integer.",
                )
                return

            if length > effective_max:
                await self._send_payload_error(
                    send,
                    status=413,
                    error="payload_too_large",
                    message=(
                        f"Request body too large ({length:,} bytes). "
                        f"Maximum allowed: {effective_max:,} bytes."
                    ),
                    hint="Reduce the request payload size.",
                )
                return

        bytes_received = 0
        rejected = False

        async def receive_with_limit() -> Message:
            nonlocal bytes_received, rejected
            message = await receive()
            if message["type"] == "http.request":
                bytes_received += len(message.get("body", b""))
                if bytes_received > effective_max:
                    rejected = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def send_with_guard(message: Message) -> None:
            # Suppress whatever the app tried to send — we will issue
            # our own 413 below.
            if rejected and message["type"] in ("http.response.start", "http.response.body"):
                return
            await send(message)

        await self.app(scope, receive_with_limit, send_with_guard)

        if rejected:
            await self._send_payload_error(
                send,
                status=413,
                error="payload_too_large",
                message=(
                    f"Request body too large (>{effective_max:,} bytes). "
                    f"Maximum allowed: {effective_max:,} bytes."
                ),
                hint="Reduce the request payload size.",
            )

    @staticmethod
    async def _send_payload_error(
        send: Send,
        *,
        status: int,
        error: str,
        message: str,
        hint: str,
    ) -> None:
        body = json.dumps(envelope(error=error, message=message, hint=hint)).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# ---------------------------------------------------------------------------
# Insecure transport warning
# ---------------------------------------------------------------------------


class InsecureTransportWarningMiddleware(BaseHTTPMiddleware):
    """Log once when authenticated traffic arrives over plain HTTP.

    Triggers only when authentication or per-session EDGAR credentials
    are configured (i.e. a non-local deployment) and the
    ``X-Forwarded-Proto`` header indicates ``http``.  The warning is
    suppressed in local dev because there is nothing to defend against
    when no secret leaves the machine.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._warned = False
        self._lock = threading.Lock()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        self._maybe_warn(request)
        return await call_next(request)

    def _maybe_warn(self, request: Request) -> None:
        # Cheap fast-path before we take the lock or read settings.
        if self._warned:
            return
        settings = get_settings()
        api = settings.api
        if not (api.key or api.admin_key or api.edgar_session_required):
            return

        forwarded_proto = request.headers.get("x-forwarded-proto")
        if forwarded_proto is None:
            return
        proto = forwarded_proto.split(",", 1)[0].strip().lower()
        if proto != "http":
            return

        with self._lock:
            if self._warned:
                return
            self._warned = True
            logger.warning(
                "Insecure transport detected (X-Forwarded-Proto=http) while "
                "authentication or per-session EDGAR credentials are enabled. "
                "Scenarios B/C require TLS — terminate HTTPS at the reverse "
                "proxy or launch uvicorn with --ssl-certfile/--ssl-keyfile."
            )


# ---------------------------------------------------------------------------
# Rate limiting (per-IP sliding window)
# ---------------------------------------------------------------------------


# How often to prune stale buckets (seconds).
_CLEANUP_INTERVAL = 300.0


class _SlidingWindow:
    """Per-key sliding-window counter with periodic cleanup."""

    __slots__ = ("_last_cleanup", "_limit", "_lock", "_requests", "_window")

    def __init__(self, requests_per_minute: int) -> None:
        self._limit = requests_per_minute
        self._window = 60.0
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    @property
    def limit(self) -> int:
        return self._limit

    def is_allowed(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            if now - self._last_cleanup > _CLEANUP_INTERVAL:
                self._prune(now)
                self._last_cleanup = now

            cutoff = now - self._window
            timestamps = self._requests[key]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self._limit:
                retry_after = int(self._window - (now - timestamps[0])) + 1
                return False, max(retry_after, 1)

            timestamps.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        stale = [k for k, ts in self._requests.items() if not ts or ts[-1] <= cutoff]
        for k in stale:
            del self._requests[k]


def _classify_path(path: str, method: str) -> str | None:
    """Map a request path/method to a rate-limit category.

    Thin shim over :func:`~sec_generative_search.api.policies.resolve_policy`
    preserved for the existing tests that import it directly.  The
    return semantics mirror the original classifier:

    - ``None`` — the route is exempt from the rate limiter (e.g.
      ``/api/health``) **or** the path falls outside ``/api/`` and so
      should not be rate-limited at all (Swagger / Redoc / static).
    - any non-``None`` string — the rate-limit bucket name.
    """
    # Non-``/api/`` paths (Swagger, Redoc, openapi.json, static) skip
    # the rate limiter entirely — they are unauthenticated read
    # surfaces gated by ``API_KEY`` presence in the FastAPI factory.
    if not path.startswith("/api/") and path != "/api":
        return None
    return resolve_policy(path, method).rate_category


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP sliding-window limiter with optional per-session keying.

    Buckets are keyed by ``request.client.host``.  A ``0`` limit
    disables that category.

    The validate category additionally enforces a per-``session_id``
    sliding window on top of the per-IP bucket: both must allow the
    request. A shared NAT cannot consume the per-IP bucket on behalf of
    one tenant, and a single session cannot brute-force key validation
    across many origins. Requests without a shape-valid ``session_id``
    cookie skip the per-session check and rely on the per-IP bucket
    alone.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        search_rpm: int = 60,
        ingest_rpm: int = 10,
        delete_rpm: int = 30,
        general_rpm: int = 120,
        rag_rpm: int = 60,
        validate_rpm: int = 10,
        validate_per_session_rpm: int = 5,
        session_rpm: int = 20,
    ) -> None:
        super().__init__(app)
        self._buckets: dict[str, _SlidingWindow] = {}
        for category, rpm in (
            ("search", search_rpm),
            ("ingest", ingest_rpm),
            ("delete", delete_rpm),
            ("general", general_rpm),
            ("rag", rag_rpm),
            ("validate", validate_rpm),
            ("session", session_rpm),
        ):
            if rpm > 0:
                self._buckets[category] = _SlidingWindow(rpm)
        # Separate bucket so the per-IP and per-session windows do not
        # share a counter.
        self._validate_session_bucket: _SlidingWindow | None = (
            _SlidingWindow(validate_per_session_rpm) if validate_per_session_rpm > 0 else None
        )

    def reset(self) -> None:
        for bucket in self._buckets.values():
            bucket.reset()
        if self._validate_session_bucket is not None:
            self._validate_session_bucket.reset()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        category = _classify_path(request.url.path, request.method)
        if category is None or category not in self._buckets:
            return await call_next(request)

        bucket = self._buckets[category]
        client_ip = request.client.host if request.client else "unknown"

        allowed, retry_after = bucket.is_allowed(client_ip)
        if not allowed:
            return self._reject(category, request, client_ip, bucket.limit, retry_after)

        # Per-session check piggy-backs on the per-IP gate above for the
        # validate route only. We do not log on success; rejection uses
        # the same structured 429 envelope as every other rate-limit response.
        if category == "validate" and self._validate_session_bucket is not None:
            sid = request.cookies.get(_session_cookie_name())
            if sid is not None and sid:
                allowed_s, retry_s = self._validate_session_bucket.is_allowed(sid)
                if not allowed_s:
                    return self._reject(
                        "validate",
                        request,
                        f"session={sid[:6]}...",
                        self._validate_session_bucket.limit,
                        retry_s,
                    )

        return await call_next(request)

    def _reject(
        self,
        category: str,
        request: Request,
        key_label: str,
        limit: int,
        retry_after: int,
    ) -> JSONResponse:
        logger.warning(
            "Rate limit exceeded: %s from %s on %s %s (limit=%d/min)",
            category,
            key_label,
            request.method,
            request.url.path,
            limit,
        )
        return JSONResponse(
            status_code=429,
            content=envelope(
                error="rate_limited",
                message=(f"Rate limit exceeded for {category}. Retry in {retry_after}s."),
                details={"category": category, "limit_per_minute": limit},
                hint=f"Maximum {limit} {category} requests per minute.",
            ),
            headers={"Retry-After": str(retry_after)},
        )


# ---------------------------------------------------------------------------
# Internal: typing helper for tests
# ---------------------------------------------------------------------------


def _security_headers_for_test() -> dict[str, str]:
    """Expose the static security-headers map for unit assertions."""
    return {name.decode(): value.decode() for name, value in _SECURITY_HEADERS}


_security_headers_for_test.__test__ = False  # type: ignore[attr-defined]


# Re-exported for tests that want the raw bytes pairs.
SECURITY_HEADERS_RAW: tuple[tuple[bytes, bytes], ...] = _SECURITY_HEADERS


# Re-exported for tests that want to inspect the typing of dispatched messages.
_MessageT = dict[str, Any]
