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
// Storage modes
// -------------
// Two opt-in modes share this seam:
//
//   - **Plain** (default): per-provider entries live as
//     `sec.providerKey.{provider} = "<cleartext>"`. Backward-compatible
//     with existing plain-mode storage.
//   - **Encrypted**: a single vault entry
//     `sec.providerKeyVault` carries a base64-wrapped AES-GCM blob over
//     the JSON-encoded `{provider: key}` map. A sentinel
//     `sec.providerKeyMode = "encrypted"` flags the mode so the seam
//     refuses to mix cleartext + vault writes. The vault key is derived
//     from a per-tab passphrase via PBKDF2-SHA256 (250 000 iterations)
//     and held only in module memory as a non-extractable
//     `CryptoKey` — it dies with the tab.
//
// Encrypted mode adds three states observable through `isEncrypted()`
// + `isUnlocked()`:
//
//   - `(false, _)`         → plain mode
//   - `(true,  false)`     → encrypted + locked  (vault sealed; reads
//                                                  return empty; writes
//                                                  throw `VaultLockedError`)
//   - `(true,  true)`      → encrypted + unlocked (in-memory cleartext
//                                                  map; writes re-encrypt
//                                                  the vault with a
//                                                  fresh IV)

import {
  decryptVault,
  deriveKey,
  encryptMap,
  generateSalt,
  saltFromVault,
  type CleartextMap,
  type VaultBlob,
} from "@/lib/provider-keys-crypto";

const STORAGE_PREFIX = "sec.providerKey.";
const VAULT_KEY = "sec.providerKeyVault";
const MODE_KEY = "sec.providerKeyMode";
const ENCRYPTED_MODE_VALUE = "encrypted";

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

export class VaultLockedError extends Error {
  constructor() {
    super("Provider-key vault is locked. Unlock it before mutating keys.");
    this.name = "VaultLockedError";
  }
}

export class EncryptionAlreadyEnabledError extends Error {
  constructor() {
    super("Provider-key vault is already encrypted.");
    this.name = "EncryptionAlreadyEnabledError";
  }
}

export class EncryptionDisabledError extends Error {
  constructor() {
    super("Provider-key vault is not encrypted.");
    this.name = "EncryptionDisabledError";
  }
}

const listeners = new Set<Listener>();

// `useSyncExternalStore` requires `getSnapshot()` to return the same
// reference until a mutation actually happens — otherwise React loops.
// We rebuild the snapshot on read but cache it; `invalidateSnapshot`
// drops the cache on every mutation seam (set / remove / clear / mode
// change).
let cachedSnapshot: ProviderKeyMap | null = null;

// Encrypted-mode in-memory state. `liveKey` is the non-extractable
// CryptoKey; `liveSalt` is the salt the vault was sealed with (kept so
// re-encrypt on every write can reuse it); `liveMap` is the cleartext
// authoritative source between mutations while unlocked.
let liveKey: CryptoKey | null = null;
let liveSalt: Uint8Array | null = null;
let liveMap: Record<string, string> | null = null;

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

function readVault(): VaultBlob | null {
  if (!isStorageAvailable()) {
    return null;
  }
  const raw = window.sessionStorage.getItem(VAULT_KEY);
  if (raw === null) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (
    typeof parsed !== "object"
    || parsed === null
    || typeof (parsed as VaultBlob).v !== "number"
    || typeof (parsed as VaultBlob).salt !== "string"
    || typeof (parsed as VaultBlob).iv !== "string"
    || typeof (parsed as VaultBlob).ciphertext !== "string"
  ) {
    return null;
  }
  return parsed as VaultBlob;
}

function writeVault(vault: VaultBlob): void {
  if (!isStorageAvailable()) {
    return;
  }
  window.sessionStorage.setItem(VAULT_KEY, JSON.stringify(vault));
}

function modeSentinelPresent(): boolean {
  if (!isStorageAvailable()) {
    return false;
  }
  return window.sessionStorage.getItem(MODE_KEY) === ENCRYPTED_MODE_VALUE;
}

function readPlainEntries(): Record<string, string> {
  if (!isStorageAvailable()) {
    return {};
  }
  const store = window.sessionStorage;
  const out: Record<string, string> = {};
  for (let i = 0; i < store.length; i += 1) {
    const k = store.key(i);
    if (k === null || !k.startsWith(STORAGE_PREFIX)) {
      continue;
    }
    const provider = k.slice(STORAGE_PREFIX.length);
    if (!PROVIDER_NAME_RE.test(provider)) {
      continue;
    }
    const value = store.getItem(k);
    if (value === null || value === "") {
      continue;
    }
    out[provider] = value;
  }
  return out;
}

function removePlainEntries(): void {
  if (!isStorageAvailable()) {
    return;
  }
  const store = window.sessionStorage;
  const toRemove: string[] = [];
  for (let i = 0; i < store.length; i += 1) {
    const k = store.key(i);
    if (k !== null && k.startsWith(STORAGE_PREFIX)) {
      toRemove.push(k);
    }
  }
  for (const k of toRemove) {
    store.removeItem(k);
  }
}

function snapshot(): ProviderKeyMap {
  if (cachedSnapshot !== null) {
    return cachedSnapshot;
  }
  if (!isStorageAvailable()) {
    cachedSnapshot = EMPTY_MAP;
    return cachedSnapshot;
  }
  if (modeSentinelPresent()) {
    // Encrypted mode. When unlocked, the in-memory map is authoritative
    // (the vault is a sealed projection of it). When locked, callers
    // see an empty map by design — they MUST unlock to recover access.
    if (liveMap !== null) {
      cachedSnapshot = Object.freeze({ ...liveMap });
    } else {
      cachedSnapshot = EMPTY_MAP;
    }
    return cachedSnapshot;
  }
  cachedSnapshot = Object.freeze(readPlainEntries());
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

async function persistVault(): Promise<void> {
  if (liveKey === null || liveSalt === null || liveMap === null) {
    throw new VaultLockedError();
  }
  const blob = await encryptMap(liveKey, liveMap, liveSalt);
  writeVault(blob);
}

/** Return a read-only snapshot of every stored provider key. */
export function loadProviderKeys(): ProviderKeyMap {
  return snapshot();
}

/**
 * Persist a provider key. In plain mode this writes a cleartext entry
 * to `sessionStorage`. In encrypted mode the vault is re-encrypted
 * with a fresh IV and the in-memory map updated. Throws
 * `VaultLockedError` when called against an encrypted-but-locked seam.
 *
 * Returns a `Promise<void>` so the encrypted-mode write completes
 * before the caller surfaces success. Validation errors
 * (`Invalid provider`, length bounds, `VaultLockedError`) are thrown
 * **synchronously**; only the AES-GCM seal in encrypted mode is
 * deferred onto the promise — this preserves the
 * `expect(() => setProviderKey(bad)).toThrow()` contract callers
 * relied on before encrypted mode was added.
 */
export function setProviderKey(
  provider: string,
  apiKey: string,
): Promise<void> {
  validateProviderName(provider);
  validateApiKey(apiKey);
  if (!isStorageAvailable()) {
    return Promise.resolve();
  }
  if (modeSentinelPresent()) {
    if (liveKey === null || liveMap === null) {
      throw new VaultLockedError();
    }
    liveMap[provider] = apiKey;
    return persistVault().then(() => {
      notify();
    });
  }
  window.sessionStorage.setItem(STORAGE_PREFIX + provider, apiKey);
  notify();
  return Promise.resolve();
}

/**
 * Remove a provider key. No-op if it was not set. Same sync-throw /
 * async-resolve contract as `setProviderKey()` — see that docstring.
 */
export function removeProviderKey(provider: string): Promise<void> {
  validateProviderName(provider);
  if (!isStorageAvailable()) {
    return Promise.resolve();
  }
  if (modeSentinelPresent()) {
    if (liveKey === null || liveMap === null) {
      throw new VaultLockedError();
    }
    if (Object.prototype.hasOwnProperty.call(liveMap, provider)) {
      delete liveMap[provider];
      return persistVault().then(() => {
        notify();
      });
    }
    return Promise.resolve();
  }
  window.sessionStorage.removeItem(STORAGE_PREFIX + provider);
  notify();
  return Promise.resolve();
}

/**
 * Drop every key in the store. Used by the sign-out flow. In encrypted
 * mode this also wipes the vault, the mode sentinel, and the in-memory
 * CryptoKey — sign-out fully resets the seam back to plain mode.
 */
export function clearProviderKeys(): void {
  if (!isStorageAvailable()) {
    return;
  }
  const store = window.sessionStorage;
  if (modeSentinelPresent()) {
    store.removeItem(VAULT_KEY);
    store.removeItem(MODE_KEY);
    liveKey = null;
    liveSalt = null;
    liveMap = null;
    notify();
    return;
  }
  removePlainEntries();
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
 *
 * Returns an empty map while the vault is locked — outbound requests
 * MUST then fall through to the server-side resolver chain (session /
 * encrypted-user / admin-env) rather than ship the partial header set.
 */
export function providerKeyHeaders(): Record<string, string> {
  const snap = snapshot();
  const headers: Record<string, string> = {};
  for (const [provider, value] of Object.entries(snap)) {
    headers[`X-Provider-Key-${provider}`] = value;
  }
  return headers;
}

// ---------------------------------------------------------------------
// Encrypted vault mode
// ---------------------------------------------------------------------

/** Whether the current tab is in encrypted mode. */
export function isEncrypted(): boolean {
  return modeSentinelPresent();
}

/**
 * Whether the in-memory CryptoKey is currently held (i.e. the operator
 * has unlocked the vault this session). Always `false` in plain mode.
 */
export function isUnlocked(): boolean {
  return modeSentinelPresent() && liveKey !== null && liveMap !== null;
}

/**
 * Turn on encrypted mode for this tab. Migrates any existing cleartext
 * entries into the freshly-sealed vault, then wipes the cleartext
 * entries. Throws if the seam is already encrypted.
 *
 * The derived CryptoKey is held in memory until `lock()` /
 * `clearProviderKeys()` / tab close. The passphrase itself is never
 * stored.
 */
export async function enableEncryption(passphrase: string): Promise<void> {
  if (!isStorageAvailable()) {
    return;
  }
  if (modeSentinelPresent()) {
    throw new EncryptionAlreadyEnabledError();
  }
  const salt = generateSalt();
  const key = await deriveKey(passphrase, salt);
  const migrated: Record<string, string> = { ...readPlainEntries() };
  const blob = await encryptMap(key, migrated, salt);
  // Order matters: write the vault FIRST so a crash between the two
  // sessionStorage writes does not leave a half-encrypted seam
  // (cleartext entries still readable in plain mode).
  writeVault(blob);
  window.sessionStorage.setItem(MODE_KEY, ENCRYPTED_MODE_VALUE);
  removePlainEntries();
  liveKey = key;
  liveSalt = salt;
  liveMap = migrated;
  notify();
}

/**
 * Turn off encrypted mode. Requires the vault to be unlocked so the
 * cleartext map can be written back as plain entries. Throws if the
 * seam is not currently encrypted or is locked.
 */
export async function disableEncryption(): Promise<void> {
  if (!isStorageAvailable()) {
    return;
  }
  if (!modeSentinelPresent()) {
    throw new EncryptionDisabledError();
  }
  if (liveKey === null || liveMap === null) {
    throw new VaultLockedError();
  }
  const snapshotMap: CleartextMap = Object.freeze({ ...liveMap });
  // Drop the encrypted blob + sentinel before writing cleartext back
  // so a crash mid-flow leaves the vault wiped (worst case is the
  // operator re-enters keys), never both cleartext and ciphertext live.
  window.sessionStorage.removeItem(VAULT_KEY);
  window.sessionStorage.removeItem(MODE_KEY);
  liveKey = null;
  liveSalt = null;
  liveMap = null;
  for (const [provider, value] of Object.entries(snapshotMap)) {
    window.sessionStorage.setItem(STORAGE_PREFIX + provider, value);
  }
  notify();
}

/**
 * Decrypt the on-disk vault using the operator-supplied passphrase.
 * On a wrong passphrase the underlying AES-GCM auth tag rejects, and we
 * re-raise as `InvalidPassphraseError` so the caller can surface a
 * passphrase-mismatch hint without leaking the underlying error type.
 */
export async function unlock(passphrase: string): Promise<void> {
  if (!isStorageAvailable()) {
    return;
  }
  if (!modeSentinelPresent()) {
    throw new EncryptionDisabledError();
  }
  const vault = readVault();
  if (vault === null) {
    throw new EncryptionDisabledError();
  }
  const salt = saltFromVault(vault);
  const key = await deriveKey(passphrase, salt);
  const map = await decryptVault(key, vault);
  liveKey = key;
  liveSalt = salt;
  liveMap = { ...map };
  notify();
}

/**
 * Drop the in-memory CryptoKey + cleartext map. The vault stays sealed
 * in `sessionStorage`; calling `unlock(passphrase)` later restores
 * access. Reads return an empty map until then.
 */
export function lock(): void {
  if (!modeSentinelPresent()) {
    return;
  }
  liveKey = null;
  liveSalt = null;
  liveMap = null;
  notify();
}

// Re-export the crypto-error type so consumers can `instanceof`-narrow
// without importing the lower-level module directly.
export { InvalidPassphraseError } from "@/lib/provider-keys-crypto";
