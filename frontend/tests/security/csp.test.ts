// Asserts:
//  - production CSP contains NEITHER `'unsafe-eval'` NOR `'unsafe-inline'`
//    in any directive, in any environment
//  - dev-only carve-outs (`'unsafe-eval'` for HMR, `ws://localhost:*` in
//    `connect-src`) appear ONLY when NODE_ENV === "development"
//  - per-request nonces actually vary
//  - the directive set matches the shared security-header policy

import { describe, expect, it } from "vitest";

import {
  buildContentSecurityPolicy,
  generateNonce,
  staticSecurityHeaders,
} from "@/lib/security-headers";

const PROD_NONCE = "ZGV0ZXJtaW5pc3RpYy1ub25jZS1mb3ItdGVzdA";

describe("Content-Security-Policy — production", () => {
  const csp = buildContentSecurityPolicy(PROD_NONCE, false);

  it("declares default-src 'self'", () => {
    expect(csp).toContain("default-src 'self'");
  });

  it("carries the per-request nonce in script-src", () => {
    expect(csp).toContain(`script-src 'self' 'nonce-${PROD_NONCE}'`);
  });

  it("carries the per-request nonce in style-src", () => {
    expect(csp).toContain(`style-src 'self' 'nonce-${PROD_NONCE}'`);
  });

  it("denies framing", () => {
    expect(csp).toContain("frame-ancestors 'none'");
  });

  it("denies plugin objects", () => {
    expect(csp).toContain("object-src 'none'");
  });

  it("locks base-uri to self", () => {
    expect(csp).toContain("base-uri 'self'");
  });

  it("locks form-action to self", () => {
    expect(csp).toContain("form-action 'self'");
  });

  it("requires Trusted Types for script sinks", () => {
    expect(csp).toContain("require-trusted-types-for 'script'");
  });

  it("does NOT allow 'unsafe-eval' anywhere", () => {
    expect(csp).not.toContain("'unsafe-eval'");
  });

  it("does NOT allow 'unsafe-inline' anywhere", () => {
    expect(csp).not.toContain("'unsafe-inline'");
  });

  it("does NOT permit ws://localhost in connect-src", () => {
    expect(csp).not.toContain("ws://localhost");
  });

  it("does NOT name third-party origins", () => {
    // CDN poisoning defence — every directive lists 'self' or 'none'.
    const allowedTokens = [
      "'self'",
      "'none'",
      "data:",
      `'nonce-${PROD_NONCE}'`,
      "'script'",
    ];
    const offending = csp
      .split(";")
      .map((d) => d.trim())
      .filter((d) => {
        const tokens = d.split(/\s+/).slice(1);
        return tokens.some(
          (t) =>
            t &&
            !allowedTokens.includes(t) &&
            t !== "upgrade-insecure-requests",
        );
      });
    expect(offending).toEqual([]);
  });
});

describe("Content-Security-Policy — development carve-outs", () => {
  const csp = buildContentSecurityPolicy(PROD_NONCE, true);

  it("permits 'unsafe-eval' in script-src for HMR", () => {
    expect(csp).toMatch(/script-src [^;]*'unsafe-eval'/);
  });

  it("permits ws://localhost in connect-src for HMR", () => {
    expect(csp).toMatch(/connect-src [^;]*ws:\/\/localhost:\*/);
  });

  it("still refuses 'unsafe-inline' even in development", () => {
    expect(csp).not.toContain("'unsafe-inline'");
  });

  it("still requires Trusted Types in development", () => {
    expect(csp).toContain("require-trusted-types-for 'script'");
  });

  it("still denies framing in development", () => {
    expect(csp).toContain("frame-ancestors 'none'");
  });
});

describe("Nonce generation", () => {
  it("produces non-empty values", () => {
    expect(generateNonce()).not.toBe("");
  });

  it("varies between requests", () => {
    const samples = new Set<string>();
    for (let i = 0; i < 32; i++) {
      samples.add(generateNonce());
    }
    expect(samples.size).toBe(32);
  });

  it("uses base64 alphabet only", () => {
    expect(generateNonce()).toMatch(/^[A-Za-z0-9+/]+$/);
  });
});

describe("Static security headers", () => {
  const headers = staticSecurityHeaders();

  it("ships HSTS with preload", () => {
    expect(headers["Strict-Transport-Security"]).toContain("preload");
    expect(headers["Strict-Transport-Security"]).toContain("includeSubDomains");
  });

  it("ships nosniff", () => {
    expect(headers["X-Content-Type-Options"]).toBe("nosniff");
  });

  it("ships no-referrer", () => {
    expect(headers["Referrer-Policy"]).toBe("no-referrer");
  });

  it("ships COOP/COEP/CORP", () => {
    expect(headers["Cross-Origin-Opener-Policy"]).toBe("same-origin");
    expect(headers["Cross-Origin-Embedder-Policy"]).toBe("require-corp");
    expect(headers["Cross-Origin-Resource-Policy"]).toBe("same-origin");
  });

  it("locks Permissions-Policy down across sensitive features", () => {
    const policy = headers["Permissions-Policy"];
    for (const feature of [
      "camera",
      "microphone",
      "geolocation",
      "payment",
      "usb",
    ]) {
      expect(policy).toContain(`${feature}=()`);
    }
  });
});
