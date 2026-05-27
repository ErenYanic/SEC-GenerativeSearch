// Client-side user vault.
//
// The browser tab is the only place per-user provider keys and EDGAR
// identity live in cleartext. The server holds only the ciphertext and
// the auth-hash; it cannot decrypt the vault without the user's
// password, which never touches the wire.
//
// Pattern D:
//   - The user submits a password to the browser. The password NEVER
//     leaves the tab.
//   - `derived_key = PBKDF2-SHA256(password, salt_m, iterations)` —
//     iterations come from the server's `login-params` response so the
//     per-row stored work-factor is honoured (forward-only rotation).
//   - HKDF-SHA256 splits `derived_key` into two completely independent
//     outputs via DISTINCT `info` strings:
//       * `auth_proof = HKDF(derived_key, info="sec-gs/auth/v1")` — the
//         32-byte proof posted to `/api/auth/login`. The server HMACs
//         this with the deployment pepper to recover `auth_hash`.
//       * `kek = HKDF(derived_key, info="sec-gs/kek/v1")` — the 32-byte
//         AES-GCM-256 key that unwraps the vault ciphertext. Imported
//         as a non-extractable `CryptoKey` so a future XSS that lands
//         on the page cannot `exportKey()` the raw bytes.
//     Collapsing the two `info` strings would turn `auth_hash` into a
//     decryption-capable artefact — load-bearing for the model.
//   - Vault plaintext is JSON `{ providers: {name: {value, updated_at}},
//     edgar: {name, email} | null }`. Fresh 12-byte IV per write (AES-GCM
//     IV reuse against the same key is catastrophic — generate, never
//     reuse).
//
// Storage discipline: this module deliberately holds zero references to
// `sessionStorage` / `localStorage`. On page refresh the KEK + cleartext
// map die; the user must log in again to decrypt the vault.

import {
  changePasswordRequest,
  enrolUserRequest,
  loginParamsRequest,
  loginRequest,
  signOutRequest,
  updateVaultRequest,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Constants — mirror the backend `core/user_auth.py` discipline
// ---------------------------------------------------------------------------

/** PBKDF2 iteration count enforced on every fresh enrolment. */
export const DEFAULT_PBKDF2_ITERATIONS = 600_000;

/** Algorithm slug for fresh enrolments. Forward-only rotation seam. */
export const DEFAULT_KDF_ALGO = "pbkdf2-sha256" as const;

/** Salt size in bytes (`SALT_BYTES` on the backend). */
export const SALT_BYTES = 16;

/** AES-GCM IV size in bytes (`_VAULT_IV_BYTES` on the backend). */
export const IV_BYTES = 12;

/** HKDF `info` for the auth proof — distinct from the KEK domain. */
const HKDF_AUTH_INFO = "sec-gs/auth/v1";

/** HKDF `info` for the KEK — distinct from the auth-proof domain. */
const HKDF_KEK_INFO = "sec-gs/kek/v1";

/** Output width of both HKDF derivations (one PBKDF2 + one AES-GCM key). */
const HKDF_OUTPUT_BITS = 256;

// ---------------------------------------------------------------------------
// Error types — narrow on `instanceof` rather than parsing messages
// ---------------------------------------------------------------------------

export class CryptoUnavailableError extends Error {
  constructor() {
    super("WebCrypto subtle is unavailable in this runtime.");
    this.name = "CryptoUnavailableError";
  }
}

export class VaultLockedError extends Error {
  constructor() {
    super("User vault is locked. Sign in to unlock.");
    this.name = "VaultLockedError";
  }
}

export class InvalidPasswordError extends Error {
  constructor() {
    super("Password is incorrect or the vault is corrupted.");
    this.name = "InvalidPasswordError";
  }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface VaultProviderEntry {
  /** Cleartext provider API key. Never persisted, never on the wire. */
  readonly value: string;
  /** ISO timestamp of the last mutation. Carried for operator triage. */
  readonly updated_at: string;
}

export interface VaultEdgarIdentity {
  readonly name: string;
  readonly email: string;
}

export interface VaultCleartext {
  readonly providers: Readonly<Record<string, VaultProviderEntry>>;
  readonly edgar: VaultEdgarIdentity | null;
}

export interface VaultSessionInfo {
  readonly userId: number;
  readonly username: string;
}

export type VaultListener = (snapshot: VaultCleartext | null) => void;

const EMPTY_VAULT: VaultCleartext = Object.freeze({
  providers: Object.freeze({}),
  edgar: null,
});

// ---------------------------------------------------------------------------
// Module-level state — singleton bound to this tab
// ---------------------------------------------------------------------------

let liveKek: CryptoKey | null = null;
let liveMap: VaultCleartext | null = null;
let liveSession: VaultSessionInfo | null = null;

const listeners = new Set<VaultListener>();

function notify(): void {
  for (const listener of listeners) {
    try {
      listener(liveMap);
    } catch {
      // A subscriber that throws must not poison the rest of the chain.
    }
  }
}

// ---------------------------------------------------------------------------
// Low-level WebCrypto helpers
// ---------------------------------------------------------------------------

function subtle(): SubtleCrypto {
  const sub = globalThis.crypto?.subtle;
  if (sub === undefined) {
    throw new CryptoUnavailableError();
  }
  return sub;
}

function randomBytes(length: number): Uint8Array {
  const buf = new Uint8Array(length);
  globalThis.crypto.getRandomValues(buf);
  return buf;
}

function utf8(value: string): Uint8Array {
  return new TextEncoder().encode(value);
}

// Base64url helpers — wire encoding for `auth_proof`, `salt_m`,
// `vault_iv`, and `ciphertext_vault`. The backend uses the same
// alphabet; padding is stripped in both directions.

function bytesToBase64Url(bytes: Uint8Array): string {
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  const b64 = btoa(binary);
  return b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64UrlToBytes(value: string): Uint8Array {
  // Restore the standard alphabet and pad.
  let b64 = value.replace(/-/g, "+").replace(/_/g, "/");
  const padding = (4 - (b64.length % 4)) % 4;
  b64 += "=".repeat(padding);
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    out[i] = binary.charCodeAt(i);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Pattern D crypto primitives (exposed for tests + the enrolment flow)
// ---------------------------------------------------------------------------

interface DerivedMaterial {
  /** 32-byte HKDF output for `/api/auth/login` and `/api/auth/enrol`. */
  readonly authProof: Uint8Array;
  /** Non-extractable AES-GCM-256 key. Used for vault decrypt + encrypt. */
  readonly kek: CryptoKey;
}

/**
 * Derive `auth_proof` + `kek` from a user-supplied password and salt.
 *
 * Iterations are taken from the parameter — production code passes the
 * value the server returned from `/api/auth/login-params`, tests pass a
 * low value so the suite stays fast.
 *
 * Both outputs are produced under DISTINCT HKDF `info` strings. The
 * `auth_proof` is harmless to ship over the wire — it cannot be reversed
 * to the password (PBKDF2 cost) and it cannot decrypt the vault (the
 * KEK uses a different HKDF context). Collapsing the two contexts would
 * break the model — the test net pins this invariant.
 */
export async function derivePasswordMaterial(
  password: string,
  saltM: Uint8Array,
  iterations: number,
): Promise<DerivedMaterial> {
  if (password.length === 0) {
    throw new Error("Password must not be empty.");
  }
  if (iterations < 1) {
    throw new Error("PBKDF2 iterations must be positive.");
  }
  const sub = subtle();
  // 1) Import the password as a PBKDF2 base key. usages: deriveBits so
  //    we can feed the PBKDF2 output into a separate HKDF stage.
  const pbkdf2Base = await sub.importKey(
    "raw",
    utf8(password) as BufferSource,
    { name: "PBKDF2" },
    false,
    ["deriveBits"],
  );
  // 2) Derive 256 bits via PBKDF2 over (password, salt_m, iterations).
  const stretched = new Uint8Array(
    await sub.deriveBits(
      {
        name: "PBKDF2",
        salt: saltM as BufferSource,
        iterations,
        hash: "SHA-256",
      },
      pbkdf2Base,
      HKDF_OUTPUT_BITS,
    ),
  );
  // 3) Re-import the PBKDF2 output as an HKDF base key so we can run
  //    two independent HKDF derivations against it.
  const hkdfBase = await sub.importKey(
    "raw",
    stretched as BufferSource,
    { name: "HKDF" },
    false,
    ["deriveBits", "deriveKey"],
  );
  // 4) Derive the 32-byte `auth_proof` under the auth domain. We use
  //    `deriveBits` so we can ship the raw bytes on the wire.
  const authProofBuf = await sub.deriveBits(
    {
      name: "HKDF",
      hash: "SHA-256",
      // HKDF-Extract takes a salt; RFC 5869 §2.2 permits a zero-length
      // salt (it then collapses to a string of zero octets). The
      // `info` parameter is the contextual differentiator.
      salt: new Uint8Array(0) as BufferSource,
      info: utf8(HKDF_AUTH_INFO) as BufferSource,
    },
    hkdfBase,
    HKDF_OUTPUT_BITS,
  );
  const authProof = new Uint8Array(authProofBuf);
  // 5) Derive the KEK under the KEK domain. `deriveKey` returns a
  //    non-extractable AES-GCM CryptoKey — `exportKey` against it
  //    throws, so an XSS in the page cannot dump the raw bytes.
  const kek = await sub.deriveKey(
    {
      name: "HKDF",
      hash: "SHA-256",
      salt: new Uint8Array(0) as BufferSource,
      info: utf8(HKDF_KEK_INFO) as BufferSource,
    },
    hkdfBase,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
  return { authProof, kek };
}

interface EncryptedVault {
  readonly ciphertext: Uint8Array;
  readonly iv: Uint8Array;
}

/**
 * Encrypt the vault plaintext under the supplied KEK. Always uses a
 * fresh 12-byte IV — the caller is responsible for shipping it
 * alongside the ciphertext. IV reuse against the same AES-GCM key is
 * catastrophic; never recycle.
 */
export async function encryptVault(
  kek: CryptoKey,
  plain: VaultCleartext,
): Promise<EncryptedVault> {
  const iv = randomBytes(IV_BYTES);
  const plaintext = utf8(JSON.stringify(plain));
  const buf = await subtle().encrypt(
    { name: "AES-GCM", iv: iv as BufferSource },
    kek,
    plaintext as BufferSource,
  );
  return { ciphertext: new Uint8Array(buf), iv };
}

/**
 * Decrypt a vault blob under the supplied KEK. Every failure path
 * (auth-tag mismatch, malformed JSON, schema mismatch) maps to
 * `InvalidPasswordError` so a wire observer cannot distinguish
 * "wrong password" from "corrupted vault".
 */
export async function decryptVault(
  kek: CryptoKey,
  ciphertext: Uint8Array,
  iv: Uint8Array,
): Promise<VaultCleartext> {
  let plaintextBuf: ArrayBuffer;
  try {
    plaintextBuf = await subtle().decrypt(
      { name: "AES-GCM", iv: iv as BufferSource },
      kek,
      ciphertext as BufferSource,
    );
  } catch {
    throw new InvalidPasswordError();
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(plaintextBuf));
  } catch {
    throw new InvalidPasswordError();
  }
  return normaliseVault(parsed);
}

function normaliseVault(value: unknown): VaultCleartext {
  if (typeof value !== "object" || value === null) {
    throw new InvalidPasswordError();
  }
  const record = value as Record<string, unknown>;
  // Providers — accept either the legacy `{name: value}` shape (in case
  // a server-side migration ever materialises one) or the documented
  // `{name: {value, updated_at}}` shape. We normalise to the latter.
  const providersIn =
    typeof record.providers === "object" && record.providers !== null
      ? (record.providers as Record<string, unknown>)
      : {};
  const providers: Record<string, VaultProviderEntry> = {};
  for (const [name, entry] of Object.entries(providersIn)) {
    if (typeof entry === "string") {
      providers[name] = Object.freeze({
        value: entry,
        updated_at: "1970-01-01T00:00:00Z",
      });
      continue;
    }
    if (typeof entry !== "object" || entry === null) {
      continue;
    }
    const obj = entry as Record<string, unknown>;
    if (typeof obj.value !== "string") {
      continue;
    }
    const updatedAt =
      typeof obj.updated_at === "string"
        ? obj.updated_at
        : "1970-01-01T00:00:00Z";
    providers[name] = Object.freeze({ value: obj.value, updated_at: updatedAt });
  }
  // EDGAR identity — explicit null means "never set". Reject anything else
  // shapeless rather than half-populating.
  let edgar: VaultEdgarIdentity | null = null;
  if (record.edgar !== undefined && record.edgar !== null) {
    if (typeof record.edgar !== "object") {
      throw new InvalidPasswordError();
    }
    const e = record.edgar as Record<string, unknown>;
    if (typeof e.name === "string" && typeof e.email === "string") {
      edgar = Object.freeze({ name: e.name, email: e.email });
    }
  }
  return Object.freeze({
    providers: Object.freeze(providers),
    edgar,
  });
}

// ---------------------------------------------------------------------------
// State queries
// ---------------------------------------------------------------------------

export function isUnlocked(): boolean {
  return liveKek !== null && liveMap !== null;
}

export function snapshot(): VaultCleartext | null {
  return liveMap;
}

export function currentSession(): VaultSessionInfo | null {
  return liveSession;
}

export function subscribe(listener: VaultListener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

// ---------------------------------------------------------------------------
// Login flow — server params → derive → POST login → decrypt
// ---------------------------------------------------------------------------

/**
 * Resolve `salt_M` + KDF params for the supplied username. The server
 * returns a deterministic decoy for unknown usernames so the wire never
 * distinguishes them from real ones; the caller therefore cannot tell
 * (without attempting login) whether the username is enrolled.
 */
export async function fetchLoginParams(username: string): Promise<{
  saltM: Uint8Array;
  iterations: number;
  kdfAlgo: string;
}> {
  const params = await loginParamsRequest(username);
  return {
    saltM: base64UrlToBytes(params.salt_m),
    iterations: params.pbkdf2_iterations,
    kdfAlgo: params.kdf_algo,
  };
}

/**
 * Log the user in. Derives `auth_proof` + KEK from the password, posts
 * to `/api/auth/login`, decrypts the returned vault ciphertext, and
 * hydrates the in-memory cache.
 *
 * The password is NOT retained — the caller MUST clear its React state
 * the moment this resolves (the LoginGate test net pins that contract).
 *
 * Throws:
 *   - `ApiError(401)` — login refused (unknown user / wrong password /
 *     locked account; the wire is intentionally opaque about which).
 *     The caller MUST surface a generic "login refused" notice.
 *   - `ApiError(429)` — rate-limited (per-IP or per-username).
 *   - `InvalidPasswordError` — server accepted the proof but AES-GCM
 *     rejected the ciphertext. In practice this is unreachable (proof
 *     and KEK derive from the same password); it surfaces as a hard
 *     reset on the SPA so a corrupted vault cannot pretend to log in.
 */
export async function loginUser(
  username: string,
  password: string,
): Promise<VaultSessionInfo> {
  const params = await fetchLoginParams(username);
  const { authProof, kek } = await derivePasswordMaterial(
    password,
    params.saltM,
    params.iterations,
  );
  const response = await loginRequest({
    username,
    auth_proof: bytesToBase64Url(authProof),
  });
  const ciphertext = base64UrlToBytes(response.ciphertext_vault);
  const iv = base64UrlToBytes(response.vault_iv);
  const cleartext = await decryptVault(kek, ciphertext, iv);
  liveKek = kek;
  liveMap = cleartext;
  liveSession = Object.freeze({
    userId: response.user_id,
    username: response.username,
  });
  notify();
  return liveSession;
}

// ---------------------------------------------------------------------------
// Enrolment flow — token → fresh salt → derive → encrypt empty → POST
// ---------------------------------------------------------------------------

/**
 * Complete enrolment with the operator-issued single-use token. Derives
 * everything client-side, encrypts an empty vault, and posts. On
 * success the user is NOT auto-logged-in — they redirect to the login
 * surface and complete a normal login round-trip. That keeps the
 * enrolment endpoint's response shape minimal and lets the login flow
 * remain the single seam that hydrates the in-memory cache.
 */
export async function enrolUser(
  token: string,
  password: string,
): Promise<{ user_id: number; username: string }> {
  const saltM = randomBytes(SALT_BYTES);
  const iterations = DEFAULT_PBKDF2_ITERATIONS;
  const { authProof, kek } = await derivePasswordMaterial(
    password,
    saltM,
    iterations,
  );
  const encrypted = await encryptVault(kek, EMPTY_VAULT);
  const response = await enrolUserRequest({
    token,
    salt_m: bytesToBase64Url(saltM),
    auth_proof: bytesToBase64Url(authProof),
    ciphertext_vault: bytesToBase64Url(encrypted.ciphertext),
    vault_iv: bytesToBase64Url(encrypted.iv),
    kdf_algo: DEFAULT_KDF_ALGO,
    pbkdf2_iterations: iterations,
  });
  return { user_id: response.user_id, username: response.username };
}

// ---------------------------------------------------------------------------
// Mutation seam — read live → mutate → re-encrypt → POST /api/auth/vault
// ---------------------------------------------------------------------------

/**
 * Apply `updater` to the live cleartext map, re-encrypt under the
 * in-memory KEK with a fresh IV, and ship the new ciphertext to
 * `/api/auth/vault`. On a successful POST the in-memory map is
 * replaced; on failure the prior state is preserved (atomic from the
 * SPA's perspective).
 *
 * Throws `VaultLockedError` when the vault is locked (logged out).
 */
export async function mutateVault(
  updater: (current: VaultCleartext) => VaultCleartext,
): Promise<void> {
  if (liveKek === null || liveMap === null) {
    throw new VaultLockedError();
  }
  const next = updater(liveMap);
  const encrypted = await encryptVault(liveKek, next);
  await updateVaultRequest({
    ciphertext_vault: bytesToBase64Url(encrypted.ciphertext),
    vault_iv: bytesToBase64Url(encrypted.iv),
  });
  liveMap = next;
  notify();
}

// ---------------------------------------------------------------------------
// Password change — derive old + new client-side, atomic server rotate
// ---------------------------------------------------------------------------

/**
 * Atomic password change. Derives the old `auth_proof` to prove
 * possession of the current password, derives a new `salt_M` + KEK +
 * `auth_proof_new` from the new password, re-encrypts the vault under
 * the new KEK, and ships the whole payload in a single POST.
 *
 * On success the in-memory KEK is replaced — the caller stays signed
 * in. On failure the prior KEK + map remain authoritative.
 */
export async function changePassword(
  oldPassword: string,
  newPassword: string,
): Promise<void> {
  if (liveKek === null || liveMap === null || liveSession === null) {
    throw new VaultLockedError();
  }
  // Derive the OLD proof so the server can validate it the same way
  // the login surface would.
  const oldParams = await fetchLoginParams(liveSession.username);
  const oldDerived = await derivePasswordMaterial(
    oldPassword,
    oldParams.saltM,
    oldParams.iterations,
  );
  // Mint a fresh salt for the new password — never reuse the old one.
  const newSalt = randomBytes(SALT_BYTES);
  const newIterations = DEFAULT_PBKDF2_ITERATIONS;
  const newDerived = await derivePasswordMaterial(
    newPassword,
    newSalt,
    newIterations,
  );
  // Re-encrypt the live cleartext map under the new KEK. The IV is
  // fresh; the salt rotation makes a stolen `auth_hash` from before the
  // change unusable against the new vault.
  const encrypted = await encryptVault(newDerived.kek, liveMap);
  await changePasswordRequest({
    auth_proof_old: bytesToBase64Url(oldDerived.authProof),
    auth_proof_new: bytesToBase64Url(newDerived.authProof),
    salt_m: bytesToBase64Url(newSalt),
    ciphertext_vault: bytesToBase64Url(encrypted.ciphertext),
    vault_iv: bytesToBase64Url(encrypted.iv),
    kdf_algo: DEFAULT_KDF_ALGO,
    pbkdf2_iterations: newIterations,
  });
  // Replace the in-memory KEK only AFTER the server commits — a
  // failure leaves us holding the old KEK + map (still valid).
  liveKek = newDerived.kek;
  notify();
}

// ---------------------------------------------------------------------------
// Sign-out — DELETE /api/auth/session + wipe local state
// ---------------------------------------------------------------------------

/**
 * Drop the in-memory KEK + cleartext map and revoke the server session.
 * Best-effort on the network call — the cookie expires regardless, and
 * the in-memory state is wiped before the request fires so a slow / 502
 * response cannot leave secrets reachable from a stale render.
 */
export async function signOutUser(): Promise<void> {
  // Wipe first so a failure on the wire cannot leave a half-signed-out
  // state where the cookie is gone but the cleartext map is still live.
  liveKek = null;
  liveMap = null;
  liveSession = null;
  notify();
  try {
    await signOutRequest();
  } catch {
    // Best-effort — cookie expires server-side anyway.
  }
}

/**
 * Hard reset — drop in-memory state without making a network call.
 * Used by the test net + the outer admin-tier sign-out path (admin
 * sign-out wipes the user-tier session as a side effect).
 */
export function resetLocalState(): void {
  liveKek = null;
  liveMap = null;
  liveSession = null;
  notify();
}

// ---------------------------------------------------------------------------
// Convenience helpers exported for the provider-keys / EDGAR card cache
// ---------------------------------------------------------------------------

/**
 * Read a copy of the providers section of the live vault. Returns
 * `{}` when the vault is locked.
 */
export function readProviders(): Readonly<Record<string, VaultProviderEntry>> {
  if (liveMap === null) {
    return EMPTY_VAULT.providers;
  }
  return liveMap.providers;
}

/** Read the EDGAR identity slot or `null` if not set / vault locked. */
export function readEdgarIdentity(): VaultEdgarIdentity | null {
  if (liveMap === null) {
    return null;
  }
  return liveMap.edgar;
}

// Re-export the base64url helpers — tests need them to round-trip
// fixtures without re-implementing the alphabet.
export const _internals = {
  bytesToBase64Url,
  base64UrlToBytes,
  HKDF_AUTH_INFO,
  HKDF_KEK_INFO,
} as const;
