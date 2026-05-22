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
  it("expires the cookie with Max-Age=0", () => {
    const req = new NextRequest("https://app.test/api/admin/session", {
      method: "DELETE",
    });
    const res = sessionRoute.DELETE(req);
    expect(res.status).toBe(200);
    const setCookie = res.headers.get("set-cookie");
    expect(setCookie).toContain(`${ADMIN_SESSION_COOKIE}=`);
    expect(setCookie).toContain("Max-Age=0");
    expect(setCookie).toContain("HttpOnly");
    expect(setCookie).toContain("Secure");
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
