// Holds per-provider API keys for the lifetime of the browser tab.
// Backed by `sessionStorage` (closes when the tab closes) — NEVER
// `localStorage` (which persists across tabs and sessions and survives a
// browser restart). The `localStorage` ban is enforced by an ESLint
// rule on every other module; this file is the sole sanctioned reader
// of `sessionStorage` for provider keys.
//
// Trust boundary
// --------------
// The browser tab is the only place these keys live in cleartext. They
// are NOT persisted to disk, NOT serialised into any cookie, and NOT
// shipped to the Next.js server (the Next.js admin proxy holds only the
// operator's `API_KEY` / `API_ADMIN_KEY` — never the per-provider keys).
//
// When the operator visits a page that performs a downstream call to a
// rate-limited / metered provider (RAG, search-with-embedder), the
// browser attaches `X-Provider-Key-{provider}` headers built from this
// store. The Next.js proxy forwards those headers verbatim; the backend
// `parse_provider_key_headers` shape-checks the suffix and feeds the key
// into the request-scoped resolver chain — first-hit-wins ahead of the
// session and encrypted-user tiers.
//
// Layout in sessionStorage
// ------------------------
// One key per provider, prefixed with `STORAGE_PREFIX` to namespace
// against any other future session-tier data. The value is the raw key
// string; we deliberately avoid JSON-wrapping because a JSON parse seam
// is a needless attack surface (an injected script that can reach this
// API can already read the cleartext value directly).

const STORAGE_PREFIX = "sec.providerKey.";

// Provider slug shape MUST match the backend `_PROVIDER_NAME_RE` in
// `api/dependencies.py` (`^[a-z0-9][a-z0-9_-]{0,31}$`). A name that does
// not satisfy this regex is silently rejected by the parser, so we
// shape-check at write time to fail loud rather than ship a header that
// the backend will discard.
const PROVIDER_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,31}$/;

// Provider keys themselves are bounded at the backend schema layer
// (`ProviderValidateRequest.api_key` is 1..4096 chars). We mirror the
// upper bound to refuse pathological input before the request layer.
const MAX_KEY_LENGTH = 4096;

export type ProviderKeyMap = Readonly<Record<string, string>>;
export type Listener = (snapshot: ProviderKeyMap) => void;

const listeners = new Set<Listener>();

// `useSyncExternalStore` requires `getSnapshot()` to return the same
// reference until a mutation actually happens — otherwise React loops.
// We rebuild the snapshot on read but cache it; `invalidateSnapshot`
// drops the cache on every mutation seam (set / remove / clear).
let cachedSnapshot: ProviderKeyMap | null = null;

function isStorageAvailable(): boolean {
  // `sessionStorage` is undefined during SSR and inside any non-browser
  // execution context (Node, Vitest's happy-dom exposes it). We check
  // for `window` instead of `typeof sessionStorage` because Next.js
  // sometimes injects a stub that throws on access.
  return typeof window !== "undefined" && typeof window.sessionStorage !== "undefined";
}

function validateProviderName(provider: string): void {
  if (!PROVIDER_NAME_RE.test(provider)) {
    throw new Error(
      `Invalid provider slug: must match ${PROVIDER_NAME_RE.source}`,
    );
  }
}

function validateApiKey(apiKey: string): void {
  if (apiKey.length === 0) {
    throw new Error("Provider API key must not be empty.");
  }
  if (apiKey.length > MAX_KEY_LENGTH) {
    throw new Error(
      `Provider API key exceeds the ${MAX_KEY_LENGTH}-character upper bound.`,
    );
  }
}

function snapshot(): ProviderKeyMap {
  if (cachedSnapshot !== null) {
    return cachedSnapshot;
  }
  if (!isStorageAvailable()) {
    cachedSnapshot = Object.freeze({});
    return cachedSnapshot;
  }
  const store = window.sessionStorage;
  const out: Record<string, string> = {};
  for (let i = 0; i < store.length; i += 1) {
    const key = store.key(i);
    if (key === null || !key.startsWith(STORAGE_PREFIX)) {
      continue;
    }
    const provider = key.slice(STORAGE_PREFIX.length);
    if (!PROVIDER_NAME_RE.test(provider)) {
      continue;
    }
    const value = store.getItem(key);
    if (value === null || value === "") {
      continue;
    }
    out[provider] = value;
  }
  cachedSnapshot = Object.freeze(out);
  return cachedSnapshot;
}

function notify(): void {
  // Drop the cache so the next `getSnapshot()` reads from
  // `sessionStorage` again and returns a new reference; React then
  // re-renders subscribed consumers.
  cachedSnapshot = null;
  const snap = snapshot();
  for (const listener of listeners) {
    try {
      listener(snap);
    } catch {
      // A subscriber that throws must not poison the rest of the chain.
    }
  }
}

/** Return a read-only snapshot of every stored provider key. */
export function loadProviderKeys(): ProviderKeyMap {
  return snapshot();
}

/** Persist a provider key in `sessionStorage` and notify subscribers. */
export function setProviderKey(provider: string, apiKey: string): void {
  validateProviderName(provider);
  validateApiKey(apiKey);
  if (!isStorageAvailable()) {
    return;
  }
  window.sessionStorage.setItem(STORAGE_PREFIX + provider, apiKey);
  notify();
}

/** Remove a provider key. No-op if it was not set. */
export function removeProviderKey(provider: string): void {
  validateProviderName(provider);
  if (!isStorageAvailable()) {
    return;
  }
  window.sessionStorage.removeItem(STORAGE_PREFIX + provider);
  notify();
}

/** Drop every key in the store. Used by the sign-out flow. */
export function clearProviderKeys(): void {
  if (!isStorageAvailable()) {
    return;
  }
  const store = window.sessionStorage;
  const toRemove: string[] = [];
  for (let i = 0; i < store.length; i += 1) {
    const key = store.key(i);
    if (key !== null && key.startsWith(STORAGE_PREFIX)) {
      toRemove.push(key);
    }
  }
  for (const key of toRemove) {
    store.removeItem(key);
  }
  notify();
}

/**
 * Subscribe to changes. Returns an unsubscribe function. Designed for
 * `useSyncExternalStore` consumers.
 */
export function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/**
 * Build the `X-Provider-Key-{provider}` header map for an outbound
 * request. The map keys carry the canonical capitalisation; HTTP
 * headers are case-insensitive but case-preserving via `Headers`.
 */
export function providerKeyHeaders(): Record<string, string> {
  const snap = snapshot();
  const headers: Record<string, string> = {};
  for (const [provider, value] of Object.entries(snap)) {
    headers[`X-Provider-Key-${provider}`] = value;
  }
  return headers;
}
