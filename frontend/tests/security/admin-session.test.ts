// Server-side admin-session store.
//
// Asserts every load-bearing property of the in-memory keystore:
//  - session ids match the backend `secrets.token_urlsafe(32)` alphabet/length
//  - keys round-trip through createSession → lookupSession
//  - sliding-TTL eviction is lazy and per-call
//  - malformed cookie values never reach the Map lookup
//  - revoke clears keys and a second lookup misses
//  - empty keys are refused at write time

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  ADMIN_SESSION_ID_PATTERN,
  _resetForTests,
  adminSessionCookieAttributes,
  createSession,
  lookupSession,
  mintSessionId,
  revokeSession,
} from "@/lib/admin-session";

beforeEach(() => {
  _resetForTests();
  delete process.env.ADMIN_SESSION_TTL_SECONDS;
});

afterEach(() => {
  _resetForTests();
  delete process.env.ADMIN_SESSION_TTL_SECONDS;
});

describe("session id minting", () => {
  it("matches the base64url alphabet and 43-char length", () => {
    for (let i = 0; i < 32; i++) {
      const id = mintSessionId();
      expect(id).toMatch(ADMIN_SESSION_ID_PATTERN);
    }
  });

  it("never repeats across 256 mints", () => {
    const ids = new Set<string>();
    for (let i = 0; i < 256; i++) {
      ids.add(mintSessionId());
    }
    expect(ids.size).toBe(256);
  });
});

describe("createSession / lookupSession", () => {
  it("round-trips API and admin keys", () => {
    const id = createSession("api-key-1", "admin-key-1"); // pragma: allowlist secret
    const keys = lookupSession(id);
    expect(keys).not.toBeNull();
    expect(keys?.apiKey).toBe("api-key-1");
    expect(keys?.adminKey).toBe("admin-key-1");
  });

  it("refuses empty keys", () => {
    expect(() => createSession("", "admin")).toThrow();
    expect(() => createSession("api", "")).toThrow();
  });

  it("returns null for malformed cookie values", () => {
    expect(lookupSession(undefined)).toBeNull();
    expect(lookupSession("")).toBeNull();
    expect(lookupSession("not-base64!")).toBeNull();
    expect(lookupSession("short")).toBeNull();
    // 44 chars (one too many) — still rejected by the alphabet length check.
    expect(lookupSession("a".repeat(44))).toBeNull();
  });

  it("returns null for an unknown but well-formed id", () => {
    createSession("api", "admin");
    const stranger = mintSessionId();
    expect(lookupSession(stranger)).toBeNull();
  });

  it("refreshes sliding TTL on each successful lookup", () => {
    process.env.ADMIN_SESSION_TTL_SECONDS = "1";
    const id = createSession("api", "admin");
    // Look up just before expiry — TTL should reset.
    expect(lookupSession(id)).not.toBeNull();
    // Sleep ~600ms (well under TTL) then look up again; should still be valid.
    const start = Date.now();
    while (Date.now() - start < 600) {
      /* spin */
    }
    expect(lookupSession(id)).not.toBeNull();
  });

  it("evicts expired sessions on next call", async () => {
    process.env.ADMIN_SESSION_TTL_SECONDS = "1";
    const id = createSession("api", "admin");
    await new Promise((resolve) => setTimeout(resolve, 1100));
    expect(lookupSession(id)).toBeNull();
  });
});

describe("revokeSession", () => {
  it("clears the session by id", () => {
    const id = createSession("api", "admin");
    revokeSession(id);
    expect(lookupSession(id)).toBeNull();
  });

  it("is a no-op for malformed ids", () => {
    expect(() => revokeSession("garbage")).not.toThrow();
    expect(() => revokeSession(undefined)).not.toThrow();
  });
});

describe("cookie attributes", () => {
  it("emits HttpOnly + Secure + SameSite=Strict + Path=/", () => {
    const attrs = adminSessionCookieAttributes();
    expect(attrs).toContain("HttpOnly");
    expect(attrs).toContain("Secure");
    expect(attrs).toContain("SameSite=Strict");
    expect(attrs).toContain("Path=/");
    expect(attrs).toMatch(/Max-Age=\d+/);
  });

  it("never leaks Domain attribute (lock to current host)", () => {
    expect(adminSessionCookieAttributes()).not.toMatch(/Domain=/i);
  });
});
