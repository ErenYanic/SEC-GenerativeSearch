// POST/DELETE/GET /api/admin/session route handler.
//
// Asserts:
//  - POST with valid keys mints an HttpOnly + Secure + SameSite=Strict cookie
//  - POST never reaches the backend with a missing/blank body
//  - POST forwards X-API-Key only (admin key never leaves the server)
//  - POST surfaces backend 401 as `invalid_api_key`
//  - DELETE expires the cookie with Max-Age=0
//  - GET returns authenticated=false for forged cookies

import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ADMIN_SESSION_COOKIE,
  _resetForTests,
} from "@/lib/admin-session";

import * as sessionRoute from "@/app/api/admin/session/route";

const originalFetch = globalThis.fetch;

function jsonRequest(body: unknown): NextRequest {
  return new NextRequest("https://app.test/api/admin/session", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

beforeEach(() => {
  _resetForTests();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
  _resetForTests();
});

describe("POST /api/admin/session", () => {
  it("returns 400 when api_key or admin_key is missing", async () => {
    const res = await sessionRoute.POST(jsonRequest({}));
    expect(res.status).toBe(400);
    const body = (await res.json()) as { error: string };
    expect(body.error).toBe("missing_credentials");
  });

  it("returns 401 when the backend rejects the API key", async () => {
    globalThis.fetch = vi.fn(async () => new Response("", { status: 401 })) as unknown as typeof fetch;
    const res = await sessionRoute.POST(
      jsonRequest({ api_key: "wrong", admin_key: "doesnt-matter" }), // pragma: allowlist secret
    );
    expect(res.status).toBe(401);
    const body = (await res.json()) as { error: string };
    expect(body.error).toBe("invalid_api_key");
    // No Set-Cookie on rejection.
    expect(res.headers.get("set-cookie")).toBeNull();
  });

  it("mints a session cookie with full security attributes on success", async () => {
    globalThis.fetch = vi.fn(async () => new Response(JSON.stringify({}), { status: 200 })) as unknown as typeof fetch;
    const res = await sessionRoute.POST(
      jsonRequest({ api_key: "good-api", admin_key: "good-admin" }), // pragma: allowlist secret
    );
    expect(res.status).toBe(200);
    const setCookie = res.headers.get("set-cookie");
    expect(setCookie).not.toBeNull();
    expect(setCookie).toContain(`${ADMIN_SESSION_COOKIE}=`);
    expect(setCookie).toContain("HttpOnly");
    expect(setCookie).toContain("Secure");
    expect(setCookie).toContain("SameSite=Strict");
    expect(setCookie).toContain("Path=/");
  });

  it("never sends the admin key to the backend during validation", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({}), { status: 200 }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    await sessionRoute.POST(
      jsonRequest({ api_key: "the-api-key", admin_key: "the-admin-key" }), // pragma: allowlist secret
    );
    const firstCall = fetchMock.mock.calls[0];
    expect(firstCall).toBeDefined();
    const [, init] = firstCall as unknown as [string, RequestInit];
    const headers = new Headers(init.headers as HeadersInit);
    expect(headers.get("x-api-key")).toBe("the-api-key");
    // The admin key MUST NOT be forwarded — validation is API-key-only.
    expect(headers.get("x-admin-key")).toBeNull();
  });

  it("does not echo the keys back in the response body", async () => {
    globalThis.fetch = vi.fn(async () => new Response("", { status: 200 })) as unknown as typeof fetch;
    const res = await sessionRoute.POST(
      jsonRequest({
        api_key: "secret-api-12345",  // pragma: allowlist secret
        admin_key: "secret-admin-67890",  // pragma: allowlist secret
      }),
    );
    const text = await res.text();
    expect(text).not.toContain("secret-api-12345");
    expect(text).not.toContain("secret-admin-67890");
  });

  it("forwards backend session_id Set-Cookie alongside admin_session", async () => {
    // The backend mint response carries Set-Cookie for session_id; the
    // login handler must forward it so the EDGAR registration UI has a
    // session cookie to authenticate against.
    let call = 0;
    globalThis.fetch = vi.fn(async () => {
      call += 1;
      if (call === 1) {
        // First call: GET /api/status/ (API key validation)
        return new Response(JSON.stringify({}), { status: 200 });
      }
      // Second call: POST /api/session (backend session mint)
      const resp = new Response(JSON.stringify({ issued: true }), {
        status: 201,
      });
      resp.headers.append(
        "Set-Cookie",
        "session_id=backend-mint-xyz; HttpOnly; Secure; SameSite=Strict; Path=/",
      );
      return resp;
    }) as unknown as typeof fetch;

    const res = await sessionRoute.POST(
      jsonRequest({ api_key: "k", admin_key: "a" }), // pragma: allowlist secret
    );
    expect(res.status).toBe(200);
    const cookies = res.headers.getSetCookie();
    // Admin session cookie + forwarded backend session cookie.
    expect(cookies).toHaveLength(2);
    expect(cookies.some((c) => c.startsWith(`${ADMIN_SESSION_COOKIE}=`))).toBe(
      true,
    );
    const backendCookie = cookies.find((c) =>
      c.startsWith("session_id=backend-mint-xyz"),
    );
    expect(backendCookie).toBeDefined();
    expect(backendCookie).toContain("HttpOnly");
  });

  it("still mints the admin session if backend session mint fails", async () => {
    // Defensive: a failing backend mint must not block admin login.
    // Scenario A (no session-tier features) still wants a usable admin
    // session.
    let call = 0;
    globalThis.fetch = vi.fn(async () => {
      call += 1;
      if (call === 1) {
        return new Response(JSON.stringify({}), { status: 200 });
      }
      return new Response("", { status: 500 });
    }) as unknown as typeof fetch;

    const res = await sessionRoute.POST(
      jsonRequest({ api_key: "k", admin_key: "a" }), // pragma: allowlist secret
    );
    expect(res.status).toBe(200);
    const cookies = res.headers.getSetCookie();
    // Only the admin cookie; no backend session cookie.
    expect(cookies).toHaveLength(1);
    expect(cookies[0]).toMatch(new RegExp(`^${ADMIN_SESSION_COOKIE}=`));
  });

  it("returns 502 when the backend is unreachable", async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new Error("ECONNREFUSED");
    }) as unknown as typeof fetch;
    const res = await sessionRoute.POST(
      jsonRequest({ api_key: "a", admin_key: "b" }), // pragma: allowlist secret
    );
    expect(res.status).toBe(502);
  });
});

describe("DELETE /api/admin/session", () => {
  it("expires both the admin and backend session cookies with Max-Age=0", async () => {
    // No backend fetch should happen when there's no active session.
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const req = new NextRequest("https://app.test/api/admin/session", {
      method: "DELETE",
    });
    const res = await sessionRoute.DELETE(req);
    expect(res.status).toBe(200);
    const cookies = res.headers.getSetCookie();
    const adminCookie = cookies.find((c) =>
      c.startsWith(`${ADMIN_SESSION_COOKIE}=`),
    );
    const backendCookie = cookies.find((c) => c.startsWith("session_id="));
    expect(adminCookie).toContain("Max-Age=0");
    expect(adminCookie).toContain("HttpOnly");
    expect(adminCookie).toContain("Secure");
    expect(backendCookie).toContain("Max-Age=0");
    expect(backendCookie).toContain("HttpOnly");
    expect(backendCookie).toContain("Secure");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("calls backend /api/session/logout in lockstep when both cookies are present", async () => {
    const fetchMock = vi.fn(async () => new Response("{}", { status: 200 }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    // Need a real admin session entry so lookupSession returns keys.
    const adminSessionId = (
      await import("@/lib/admin-session")
    ).createSession("api-k", "admin-k"); // pragma: allowlist secret

    const req = new NextRequest("https://app.test/api/admin/session", {
      method: "DELETE",
    });
    req.cookies.set(ADMIN_SESSION_COOKIE, adminSessionId);
    req.cookies.set("session_id", "backend-cookie-xyz");

    const res = await sessionRoute.DELETE(req);
    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toContain("/api/session/logout");
    const headers = new Headers(init.headers as HeadersInit);
    expect(headers.get("x-api-key")).toBe("api-k");
    expect(headers.get("cookie")).toBe("session_id=backend-cookie-xyz");
  });
});

describe("GET /api/admin/session", () => {
  it("reports authenticated=false for missing cookie", async () => {
    const req = new NextRequest("https://app.test/api/admin/session", {
      method: "GET",
    });
    const res = sessionRoute.GET(req);
    const body = (await res.json()) as { authenticated: boolean };
    expect(body.authenticated).toBe(false);
  });

  it("reports authenticated=false for a forged cookie", async () => {
    const req = new NextRequest("https://app.test/api/admin/session", {
      method: "GET",
    });
    req.cookies.set(ADMIN_SESSION_COOKIE, "a".repeat(43));
    const res = sessionRoute.GET(req);
    const body = (await res.json()) as { authenticated: boolean };
    expect(body.authenticated).toBe(false);
  });
});
