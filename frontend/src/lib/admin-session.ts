// Server-side admin-session store.
//
// Holds the operator's API + admin keys for the lifetime of a browser
// session. Keys are minted server-side and addressed via an HttpOnly cookie
// whose value never appears in client JS, localStorage, or URLs.
//
// Mirrors the backend's `InMemorySessionCredentialStore` pattern: sliding
// TTL, lazy eviction, no background thread. There is no second source of
// truth — restarting the Next.js process invalidates every admin session.

import { timingSafeEqual } from "node:crypto";

const SESSION_ID_BYTES = 32;
const DEFAULT_TTL_SECONDS = 3600;

export const ADMIN_SESSION_COOKIE = "admin_session";
export const ADMIN_SESSION_ID_PATTERN = /^[A-Za-z0-9_-]{43}$/;

interface AdminSessionRecord {
  readonly apiKey: string;
  readonly adminKey: string;
  expiresAt: number;
}

const sessions = new Map<string, AdminSessionRecord>();

function ttlSeconds(): number {
  const raw = process.env.ADMIN_SESSION_TTL_SECONDS;
  if (raw === undefined || raw === "") {
    return DEFAULT_TTL_SECONDS;
  }
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_TTL_SECONDS;
  }
  return parsed;
}

function now(): number {
  return Date.now();
}

function evictExpired(): void {
  const cutoff = now();
  for (const [sessionId, record] of sessions) {
    if (record.expiresAt <= cutoff) {
      sessions.delete(sessionId);
    }
  }
}

export function mintSessionId(): string {
  const bytes = new Uint8Array(SESSION_ID_BYTES);
  crypto.getRandomValues(bytes);
  // base64url, unpadded — 43 chars for 32 bytes. Matches backend
  // `secrets.token_urlsafe(32)` alphabet ([A-Za-z0-9_-]).
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function createSession(apiKey: string, adminKey: string): string {
  if (apiKey === "" || adminKey === "") {
    throw new Error("api_key and admin_key must be non-empty");
  }
  evictExpired();
  const sessionId = mintSessionId();
  sessions.set(sessionId, {
    apiKey,
    adminKey,
    expiresAt: now() + ttlSeconds() * 1000,
  });
  return sessionId;
}

/** Returns the stored keys for a session id, refreshing the sliding TTL.
 *
 * Returns `null` if the id is malformed, unknown, or expired. The shape
 * check rejects every forged cookie value before a Map lookup runs.
 */
export function lookupSession(
  rawSessionId: string | undefined,
): { apiKey: string; adminKey: string } | null {
  if (!rawSessionId || !ADMIN_SESSION_ID_PATTERN.test(rawSessionId)) {
    return null;
  }
  evictExpired();

  // Constant-time match against the in-memory key set. Standard Map.get is
  // O(1) average and itself avoids per-byte comparison, but iterating with
  // timingSafeEqual neutralises any timing oracle a future Node change in
  // hash-collision behaviour could introduce.
  const provided = Buffer.from(rawSessionId, "utf8");
  for (const [storedId, record] of sessions) {
    const stored = Buffer.from(storedId, "utf8");
    if (stored.length !== provided.length) {
      continue;
    }
    if (timingSafeEqual(stored, provided)) {
      record.expiresAt = now() + ttlSeconds() * 1000;
      return { apiKey: record.apiKey, adminKey: record.adminKey };
    }
  }
  return null;
}

export function revokeSession(rawSessionId: string | undefined): void {
  if (!rawSessionId || !ADMIN_SESSION_ID_PATTERN.test(rawSessionId)) {
    return;
  }
  sessions.delete(rawSessionId);
}

/** Test seam. Wipes every session; never call from production paths. */
export function _resetForTests(): void {
  sessions.clear();
}

/** Returns the cookie attribute string for the admin-session cookie.
 *
 * Attributes are unconditional and mirror the backend session cookie:
 * HttpOnly, Secure, SameSite=Strict, Path=/. The browser treats
 * `localhost` as a secure context, so `Secure` does not break dev.
 */
export function adminSessionCookieAttributes(): string {
  const maxAge = ttlSeconds();
  return [
    "Path=/",
    "HttpOnly",
    "Secure",
    "SameSite=Strict",
    `Max-Age=${maxAge}`,
  ].join("; ");
}
