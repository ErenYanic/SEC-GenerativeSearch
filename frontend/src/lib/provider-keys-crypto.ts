// Pure cryptography — this module deliberately never touches
// `sessionStorage` or `localStorage`. The storage-discipline regression
// net pins `sessionStorage` to `provider-keys.ts`; this module stays on
// the cleartext-map ↔ vault-blob seam so that net keeps its bite.
//
// Threat closed
// -------------
// Browser extensions, OS-level storage stealers, and malware that reads
// the per-tab `sessionStorage` snapshot directly cannot recover the
// per-provider API keys without the operator's passphrase. The
// passphrase is prompted per tab, derived into an AES-GCM CryptoKey,
// held only in module memory (and made non-extractable so a malicious
// page script cannot dump the raw key bytes), and dies with the tab.
//
// Algorithm
// ---------
//   - PBKDF2-SHA256, 250 000 iterations (NIST SP 800-132 floor + 2×
//     the OWASP 2023 baseline) over a fresh 16-byte salt per
//     `enableEncryption()`. Derives a 256-bit key.
//   - AES-GCM-256 with a fresh 12-byte IV per write. The auth tag is
//     the load-bearing detector for a wrong passphrase: on a passphrase
//     mismatch, `crypto.subtle.decrypt` throws and we surface
//     `InvalidPassphraseError` to the caller.
//   - Plaintext is the JSON-encoded `{provider: key}` map, UTF-8.
//
// Vault wire shape: `{ v: 1, salt: base64, iv: base64, ciphertext: base64 }`.
// `v` carries a schema discriminator so a future algorithm rotation can
// refuse old blobs deterministically rather than silently misdecrypt.

export const VAULT_SCHEMA_VERSION = 1;
const PBKDF2_ITERATIONS = 250_000;
const SALT_BYTES = 16;
const IV_BYTES = 12;
const AES_KEY_BITS = 256;

export class InvalidPassphraseError extends Error {
  constructor() {
    super("Passphrase is incorrect or the vault is corrupted.");
    this.name = "InvalidPassphraseError";
  }
}

export class CryptoUnavailableError extends Error {
  constructor() {
    super("WebCrypto subtle is unavailable in this runtime.");
    this.name = "CryptoUnavailableError";
  }
}

export interface VaultBlob {
  readonly v: number;
  readonly salt: string;
  readonly iv: string;
  readonly ciphertext: string;
}

export type CleartextMap = Readonly<Record<string, string>>;

function subtle(): SubtleCrypto {
  // `globalThis.crypto.subtle` is the surface we want; happy-dom does
  // not expose it by default, so the Vitest setup file polyfills via
  // `node:crypto`. A missing seam at runtime would only happen on a
  // truly hostile environment — fail loud rather than silently corrupt
  // the vault.
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

function bytesToBase64(bytes: Uint8Array): string {
  // `btoa` expects a binary string. We feed it the byte values one at a
  // time. Chunking guards against the argument-list cap on very large
  // ciphertexts, even though the vault payload is small in practice.
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    const slice = bytes.subarray(i, i + CHUNK);
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}

function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    out[i] = binary.charCodeAt(i);
  }
  return out;
}

/**
 * Derive a non-extractable AES-GCM CryptoKey from the operator's
 * passphrase and the vault salt. The returned key is usable for
 * `encrypt` / `decrypt` only; the raw bytes never leave the WebCrypto
 * boundary.
 */
export async function deriveKey(
  passphrase: string,
  salt: Uint8Array,
): Promise<CryptoKey> {
  if (passphrase.length === 0) {
    throw new Error("Passphrase must not be empty.");
  }
  const enc = new TextEncoder().encode(passphrase);
  const sub = subtle();
  // TS 6 lib.dom narrows BufferSource to `ArrayBufferView<ArrayBuffer>`
  // while `TextEncoder.encode` returns `Uint8Array<ArrayBufferLike>`.
  // The runtime payload is byte-identical; the cast is a typing
  // workaround, not a safety relaxation.
  const baseKey = await sub.importKey(
    "raw",
    enc as BufferSource,
    { name: "PBKDF2" },
    false,
    ["deriveKey"],
  );
  return sub.deriveKey(
    {
      name: "PBKDF2",
      salt: salt as BufferSource,
      iterations: PBKDF2_ITERATIONS,
      hash: "SHA-256",
    },
    baseKey,
    { name: "AES-GCM", length: AES_KEY_BITS },
    // `extractable=false` — a future XSS that lands in the page cannot
    // call `exportKey` to dump the AES key. The map mutators still work
    // because they call `encrypt` / `decrypt` against the handle.
    false,
    ["encrypt", "decrypt"],
  );
}

/** Generate a fresh 16-byte salt for `enableEncryption()`. */
export function generateSalt(): Uint8Array {
  return randomBytes(SALT_BYTES);
}

/** Generate a fresh 12-byte IV for a single AES-GCM encryption. */
export function generateIv(): Uint8Array {
  return randomBytes(IV_BYTES);
}

/**
 * Encrypt the cleartext provider-key map under `key`. Always uses a
 * fresh IV; the caller is responsible for storing it alongside the
 * ciphertext.
 */
export async function encryptMap(
  key: CryptoKey,
  map: CleartextMap,
  salt: Uint8Array,
): Promise<VaultBlob> {
  const iv = generateIv();
  const plaintext = new TextEncoder().encode(JSON.stringify(map));
  const ciphertext = await subtle().encrypt(
    { name: "AES-GCM", iv: iv as BufferSource },
    key,
    plaintext as BufferSource,
  );
  return {
    v: VAULT_SCHEMA_VERSION,
    salt: bytesToBase64(salt),
    iv: bytesToBase64(iv),
    ciphertext: bytesToBase64(new Uint8Array(ciphertext)),
  };
}

/**
 * Decrypt a vault blob and return the cleartext map. Raises
 * `InvalidPassphraseError` when AES-GCM rejects the auth tag — that is
 * the wrong-passphrase signal callers should surface to the operator.
 */
export async function decryptVault(
  key: CryptoKey,
  vault: VaultBlob,
): Promise<CleartextMap> {
  if (vault.v !== VAULT_SCHEMA_VERSION) {
    throw new InvalidPassphraseError();
  }
  const iv = base64ToBytes(vault.iv);
  const ciphertext = base64ToBytes(vault.ciphertext);
  let plaintext: ArrayBuffer;
  try {
    plaintext = await subtle().decrypt(
      { name: "AES-GCM", iv: iv as BufferSource },
      key,
      ciphertext as BufferSource,
    );
  } catch {
    // AES-GCM throws an opaque OperationError on auth-tag mismatch.
    // Map every decrypt failure to `InvalidPassphraseError` — the
    // operator-facing message is identical regardless of the underlying
    // cause (wrong passphrase, corrupted blob, truncated ciphertext).
    throw new InvalidPassphraseError();
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(plaintext));
  } catch {
    throw new InvalidPassphraseError();
  }
  if (!isStringMap(parsed)) {
    throw new InvalidPassphraseError();
  }
  return Object.freeze({ ...parsed });
}

/** Extract the salt from a stored vault blob. */
export function saltFromVault(vault: VaultBlob): Uint8Array {
  return base64ToBytes(vault.salt);
}

/** Strict runtime check before we hand a vault back to the caller. */
function isStringMap(value: unknown): value is Record<string, string> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  for (const v of Object.values(value)) {
    if (typeof v !== "string") {
      return false;
    }
  }
  return true;
}
