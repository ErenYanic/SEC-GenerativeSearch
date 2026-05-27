// Per-provider API key store.
//
// This module is now a THIN CACHE over the per-user encrypted vault
// (`user-vault.ts`). It is no longer the `sessionStorage` seam — the
// keys live, in cleartext, only inside the in-memory vault snapshot
// while the user is logged in. The ciphertext is persisted server-side
// in `users.ciphertext_vault`; the server cannot read it (Pattern D).
//
// Storage discipline
// ------------------
// Pre-13.11 this file was the SOLE sanctioned `sessionStorage` reader.
// Post-13.11 it touches neither `sessionStorage` nor `localStorage`:
// the cleartext map is held by `user-vault.ts` in module memory and
// dies with the tab (or on sign-out). The storage-discipline
// regression net is relaxed in lockstep — `localStorage` stays banned
// project-wide; `sessionStorage` is no longer used anywhere.
//
// Trust boundary
// --------------
// The browser tab is the only place these keys live in cleartext. They
// are encrypted client-side under the user's KEK before they ever leave
// the tab. When a page performs a downstream call to a metered provider
// (RAG / search / validate), the browser attaches
// `X-Provider-Key-{provider}` headers built from this cache; the Next.js
// proxy forwards them verbatim and the backend uses each key for the
// single request, then discards it.
//
// Public surface (unchanged for consumers)
// -----------------------------------------
//   loadProviderKeys()        → read-only `{name: value}` snapshot
//   setProviderKey(name, key) → re-encrypt vault + upload
//   removeProviderKey(name)   → re-encrypt vault + upload
//   clearProviderKeys()       → wipe every provider entry + upload
//   subscribe(listener)       → useSyncExternalStore seam
//   providerKeyHeaders()      → `X-Provider-Key-*` header map

import {
  mutateVault,
  readProviders,
  subscribe as subscribeVault,
  type VaultCleartext,
} from "@/lib/user-vault";

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

const EMPTY_MAP: ProviderKeyMap = Object.freeze({});

// `useSyncExternalStore` requires `getSnapshot()` to return the same
// reference until a mutation actually happens. We cache a flattened
// `{name: value}` view of the vault's providers section and invalidate
// it whenever the underlying vault notifies a change.
let cachedSnapshot: ProviderKeyMap | null = null;

// Permanent self-subscription to the vault. Any vault mutation (login,
// key change, sign-out) drops the cache so the next `snapshot()` read
// rebuilds from the live vault — correct even when no React consumer
// has subscribed (e.g. `providerKeyHeaders()` called from `api.ts`).
subscribeVault(() => {
  cachedSnapshot = null;
});

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

function buildSnapshot(): ProviderKeyMap {
  const providers = readProviders();
  const names = Object.keys(providers);
  if (names.length === 0) {
    return EMPTY_MAP;
  }
  const out: Record<string, string> = {};
  for (const name of names) {
    // Defensive: only surface slugs the backend would accept.
    if (!PROVIDER_NAME_RE.test(name)) {
      continue;
    }
    const entry = providers[name];
    if (entry !== undefined && entry.value !== "") {
      out[name] = entry.value;
    }
  }
  return Object.freeze(out);
}

function snapshot(): ProviderKeyMap {
  if (cachedSnapshot === null) {
    cachedSnapshot = buildSnapshot();
  }
  return cachedSnapshot;
}

/** Return a read-only snapshot of every stored provider key. */
export function loadProviderKeys(): ProviderKeyMap {
  return snapshot();
}

/**
 * Persist a provider key into the user vault. Re-encrypts the whole
 * vault under the in-memory KEK with a fresh IV and uploads it to
 * `POST /api/auth/vault`. Validation errors (invalid slug, empty / over-
 * length key) throw **synchronously**; the AES-GCM seal + upload are
 * deferred onto the returned promise.
 *
 * Throws `VaultLockedError` (from `user-vault.ts`) when the vault is
 * locked (the user is signed out). Callers MUST be logged in.
 */
export function setProviderKey(
  provider: string,
  apiKey: string,
): Promise<void> {
  validateProviderName(provider);
  validateApiKey(apiKey);
  return mutateVault((current: VaultCleartext) => ({
    ...current,
    providers: {
      ...current.providers,
      [provider]: {
        value: apiKey,
        updated_at: new Date().toISOString(),
      },
    },
  }));
}

/**
 * Remove a provider key from the vault. No-op (still re-uploads) if it
 * was not set. Same sync-throw / async-resolve contract as
 * `setProviderKey()`.
 */
export function removeProviderKey(provider: string): Promise<void> {
  validateProviderName(provider);
  return mutateVault((current: VaultCleartext) => {
    if (!Object.prototype.hasOwnProperty.call(current.providers, provider)) {
      return current;
    }
    const next: Record<string, VaultCleartext["providers"][string]> = {
      ...current.providers,
    };
    delete next[provider];
    return { ...current, providers: next };
  });
}

/**
 * Drop EVERY provider key from the vault (re-encrypt + upload an empty
 * providers section). This is the "Clear all keys" action on the
 * Provider Settings page — a deliberate, persistent wipe.
 *
 * NOTE: this is NOT the sign-out path. Sign-out drops the in-memory KEK
 * via `user-vault.ts::signOutUser` and leaves the ciphertext intact for
 * the next login. Use this only when the user explicitly wants their
 * stored keys gone server-side.
 */
export function clearProviderKeys(): Promise<void> {
  return mutateVault((current: VaultCleartext) => ({
    ...current,
    providers: {},
  }));
}

/**
 * Subscribe to changes. Returns an unsubscribe function. Designed for
 * `useSyncExternalStore` consumers. Delegates to the vault subscription
 * and invalidates the cached flattened snapshot on every notify.
 */
export function subscribe(listener: Listener): () => void {
  return subscribeVault(() => {
    cachedSnapshot = null;
    listener(snapshot());
  });
}

/**
 * Build the `X-Provider-Key-{provider}` header map for an outbound
 * request. Returns an empty map when the vault is locked (the snapshot
 * is empty), so outbound requests fall through to the server-side
 * resolver chain (admin-default / admin-env) rather than ship a
 * partial header set.
 */
export function providerKeyHeaders(): Record<string, string> {
  const snap = snapshot();
  const headers: Record<string, string> = {};
  for (const [provider, value] of Object.entries(snap)) {
    headers[`X-Provider-Key-${provider}`] = value;
  }
  return headers;
}
