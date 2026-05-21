// Assert the middleware writes the full security-header set and a
// nonce-bearing CSP to every response.
//
// The middleware lives in /middleware.ts. We import it directly and invoke
// it with a fake `NextRequest` shape — no real HTTP server needed.

import { describe, expect, it, vi } from "vitest";

// `next/server` ships an ESM-only NextRequest. We exercise the middleware via
// its public function signature with a Headers-shaped request stub.
import { NextRequest } from "next/server";

import { middleware } from "../../middleware";

function makeRequest(): NextRequest {
  return new NextRequest("https://example.test/");
}

describe("middleware response headers", () => {
  it("writes a Content-Security-Policy header with a nonce", () => {
    const response = middleware(makeRequest());
    const csp = response.headers.get("Content-Security-Policy");
    expect(csp).toBeTruthy();
    expect(csp).toMatch(/script-src 'self' 'nonce-[A-Za-z0-9+/]+'/);
  });

  it("writes the full static security-header set", () => {
    const response = middleware(makeRequest());
    expect(response.headers.get("Strict-Transport-Security")).toContain(
      "preload",
    );
    expect(response.headers.get("X-Content-Type-Options")).toBe("nosniff");
    expect(response.headers.get("Referrer-Policy")).toBe("no-referrer");
    expect(response.headers.get("Cross-Origin-Opener-Policy")).toBe(
      "same-origin",
    );
    expect(response.headers.get("Cross-Origin-Embedder-Policy")).toBe(
      "require-corp",
    );
    expect(response.headers.get("Cross-Origin-Resource-Policy")).toBe(
      "same-origin",
    );
    expect(response.headers.get("Permissions-Policy")).toContain("camera=()");
  });

  it("issues a unique nonce per request", () => {
    const a = middleware(makeRequest()).headers.get("Content-Security-Policy");
    const b = middleware(makeRequest()).headers.get("Content-Security-Policy");
    expect(a).not.toBe(b);
  });
});

describe("middleware refuses to relax CSP in production", () => {
  it("never emits 'unsafe-eval' when NODE_ENV !== development", () => {
    // The IS_DEVELOPMENT constant is evaluated at module load. We assert the
    // current process's NODE_ENV is not "development" in the test runner;
    // if it is, the test environment is misconfigured.
    if (process.env.NODE_ENV === "development") {
      return; // dev environment — covered by csp.test.ts carve-out cases
    }
    const csp = middleware(makeRequest()).headers.get(
      "Content-Security-Policy",
    );
    expect(csp).not.toContain("'unsafe-eval'");
    expect(csp).not.toContain("ws://localhost");
  });
});
