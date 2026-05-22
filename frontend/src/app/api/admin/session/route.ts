// POST /api/admin/session — validate operator keys, mint admin-session cookie.
// DELETE /api/admin/session — revoke the cookie + clear server-side keys.
//
// Submitted keys are forwarded to the backend exactly once to verify the
// API key (via `GET /api/status/`). The admin key is taken on trust at
// submission time — the backend will reject it on the first admin
// operation if it is wrong; the alternative would require hitting a
// destructive endpoint just to validate, which is not worth the risk.

import { NextResponse, type NextRequest } from "next/server";

import {
  ADMIN_SESSION_COOKIE,
  adminSessionCookieAttributes,
  createSession,
  lookupSession,
  revokeSession,
} from "@/lib/admin-session";
import { buildBackendUrl } from "@/lib/backend";

// Force Node runtime — `node:crypto` and the in-memory session map are not
// portable across edge instances and would in any case be wiped per-region.
export const runtime = "nodejs";

interface LoginBody {
  api_key?: unknown;
  admin_key?: unknown;
}

function envelope(
  error: string,
  message: string,
  hint?: string,
): { error: string; message: string; hint?: string } {
  return hint ? { error, message, hint } : { error, message };
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  let body: LoginBody;
  try {
    body = (await request.json()) as LoginBody;
  } catch {
    return NextResponse.json(
      envelope("invalid_body", "request body must be JSON"),
      { status: 400 },
    );
  }

  const apiKey = typeof body.api_key === "string" ? body.api_key.trim() : "";   // pragma: allowlist secret
  const adminKey = typeof body.admin_key === "string" ? body.admin_key.trim() : "";   // pragma: allowlist secret

  if (apiKey === "" || adminKey === "") {
    return NextResponse.json(
      envelope(
        "missing_credentials",
        "api_key and admin_key are required",
        "Both keys are needed for admin operations.",
      ),
      { status: 400 },
    );
  }

  let upstream: Response;
  try {
    upstream = await fetch(buildBackendUrl("/api/status/"), {
      method: "GET",
      headers: {
        "X-API-Key": apiKey,
        Accept: "application/json",
      },
      cache: "no-store",
    });
  } catch {
    return NextResponse.json(
      envelope(
        "backend_unreachable",
        "backend did not respond",
        "Check that the SEC-GenerativeSearch API is running.",
      ),
      { status: 502 },
    );
  }

  if (upstream.status === 401 || upstream.status === 403) {
    return NextResponse.json(
      envelope(
        "invalid_api_key",
        "API key was rejected by the backend",
        "Verify API_KEY matches the running backend.",
      ),
      { status: 401 },
    );
  }
  if (!upstream.ok) {
    return NextResponse.json(
      envelope(
        "backend_error",
        `backend returned status ${upstream.status}`,
      ),
      { status: 502 },
    );
  }

  // Mint a backend `session_id` cookie alongside the admin session.
  //
  // The backend session is the only scope under which per-session
  // EDGAR identity (POST /api/session/edgar) can be registered. We mint
  // it here so the browser ends up with both cookies in a single
  // round-trip; failure to mint is non-fatal — operators without EDGAR
  // identity needs (Scenario A) still get a usable admin session.
  const backendSessionCookies: string[] = [];
  try {
    const mintResp = await fetch(buildBackendUrl("/api/session"), {
      method: "POST",
      headers: {
        "X-API-Key": apiKey,
        Accept: "application/json",
      },
      cache: "no-store",
    });
    if (mintResp.ok) {
      for (const cookie of mintResp.headers.getSetCookie()) {
        backendSessionCookies.push(cookie);
      }
    }
  } catch {
    // Backend session minting failed; admin session still proceeds. The
    // EDGAR registration UI will surface this as a missing-session
    // error on first attempt and the operator can sign out / back in.
  }

  const sessionId = createSession(apiKey, adminKey);

  const response = NextResponse.json({ ok: true }, { status: 200 });
  response.headers.append(
    "Set-Cookie",
    `${ADMIN_SESSION_COOKIE}=${sessionId}; ${adminSessionCookieAttributes()}`,
  );
  for (const cookie of backendSessionCookies) {
    response.headers.append("Set-Cookie", cookie);
  }
  // Defence-in-depth — these endpoints must never be cached.
  response.headers.set("Cache-Control", "no-store");
  return response;
}

export async function DELETE(request: NextRequest): Promise<NextResponse> {
  const sessionId = request.cookies.get(ADMIN_SESSION_COOKIE)?.value;
  const keys = lookupSession(sessionId);
  revokeSession(sessionId);

  // Tell the backend to drop the session in lockstep with the admin
  // session. Best-effort: if the backend is unreachable, the browser's
  // session_id cookie still expires below and the backend store's
  // sliding TTL will evict the entry eventually.
  const backendSessionCookie = request.cookies.get("session_id");
  if (keys !== null && backendSessionCookie !== undefined) {
    try {
      await fetch(buildBackendUrl("/api/session/logout"), {
        method: "POST",
        headers: {
          "X-API-Key": keys.apiKey,
          Cookie: `session_id=${backendSessionCookie.value}`,
          Accept: "application/json",
        },
        cache: "no-store",
      });
    } catch {
      // Best-effort; cookie still expires below.
    }
  }

  const response = NextResponse.json({ ok: true }, { status: 200 });
  // Empty value + Max-Age=0 expires both cookies immediately.
  response.headers.append(
    "Set-Cookie",
    `${ADMIN_SESSION_COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=0`,
  );
  response.headers.append(
    "Set-Cookie",
    "session_id=; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=0",
  );
  response.headers.set("Cache-Control", "no-store");
  return response;
}

export function GET(request: NextRequest): NextResponse {
  // Lightweight authenticated/unauthenticated probe used by WelcomeGate to
  // decide whether to render the login form or pass through to the app.
  // Returns only a boolean — never echoes keys.
  const sessionId = request.cookies.get(ADMIN_SESSION_COOKIE)?.value;
  const authenticated = lookupSession(sessionId) !== null;
  const response = NextResponse.json({ authenticated }, { status: 200 });
  response.headers.set("Cache-Control", "no-store");
  return response;
}
