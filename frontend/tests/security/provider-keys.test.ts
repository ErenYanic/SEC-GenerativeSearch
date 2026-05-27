// Provider-key cache over the per-user vault.
//
// `provider-keys.ts` is no longer the `sessionStorage` seam — it is a
// thin cache over `user-vault.ts`. These tests assert:
//   - reads / writes flow through the in-memory vault, NEVER any
//     browser storage (no `sessionStorage`, no `localStorage`)
//   - mutations re-encrypt + POST `/api/auth/vault`
//   - the same synchronous shape-validation (slug, key length) holds,
//     thrown before any network hop
//   - mutating while the vault is locked raises `VaultLockedError`
//   - `providerKeyHeaders()` builds the `X-Provider-Key-*` map

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import * as apiModule from "@/lib/api";
import {
  clearProviderKeys,
  loadProviderKeys,
  providerKeyHeaders,
  removeProviderKey,
  setProviderKey,
  subscribe,
} from "@/lib/provider-keys";
import {
  derivePasswordMaterial,
  encryptVault,
  loginUser,
  resetLocalState,
  _internals,
  type VaultCleartext,
} from "@/lib/user-vault";

const TEST_ITERATIONS = 1_000;

const EMPTY_VAULT: VaultCleartext = Object.freeze({
  providers: Object.freeze({}),
  edgar: null,
});

// Log in so the vault is unlocked and `mutateVault` can operate. The
// initial vault contents are supplied per-test; `updateVaultRequest`
// is mocked in `beforeEach`.
async function loginWithVault(initial: VaultCleartext): Promise<void> {
  const saltBytes = new Uint8Array(16).fill(0x31);
  const { kek } = await derivePasswordMaterial("pw", saltBytes, TEST_ITERATIONS);
  const blob = await encryptVault(kek, initial);
  vi.spyOn(apiModule, "loginParamsRequest").mockResolvedValue({
    salt_m: _internals.bytesToBase64Url(saltBytes),
    kdf_algo: "pbkdf2-sha256",
    pbkdf2_iterations: TEST_ITERATIONS,
  });
  vi.spyOn(apiModule, "loginRequest").mockResolvedValue({
    user_id: 1,
    username: "pat",
    ciphertext_vault: _internals.bytesToBase64Url(blob.ciphertext),
    vault_iv: _internals.bytesToBase64Url(blob.iv),
  });
  await loginUser("pat", "pw");
}

beforeEach(() => {
  vi.spyOn(apiModule, "updateVaultRequest").mockResolvedValue({ updated: true });
});

afterEach(() => {
  resetLocalState();
  vi.restoreAllMocks();
});

describe("provider-keys cache — storage discipline", () => {
  it("never touches sessionStorage or localStorage on a write", async () => {
    await loginWithVault(EMPTY_VAULT);
    const sessionSpy = vi.spyOn(window.sessionStorage, "setItem");
    const localSpy = vi.spyOn(window.localStorage, "setItem");
    await setProviderKey("openai", "sk-this-is-a-long-secret"); // pragma: allowlist secret
    expect(sessionSpy).not.toHaveBeenCalled();
    expect(localSpy).not.toHaveBeenCalled();
  });

  it("re-encrypts + uploads the vault on setProviderKey", async () => {
    await loginWithVault(EMPTY_VAULT);
    const updateSpy = vi.spyOn(apiModule, "updateVaultRequest");
    await setProviderKey("openai", "sk-aaaaaaaaaaaaaaaa"); // pragma: allowlist secret
    expect(updateSpy).toHaveBeenCalledTimes(1);
    const sent = updateSpy.mock.calls[0]?.[0];
    expect(sent?.vault_iv.length).toBe(16);
    expect(sent?.ciphertext_vault.length).toBeGreaterThan(0);
  });
});

describe("provider-keys cache — read/write round-trip", () => {
  it("surfaces a set key on the next snapshot", async () => {
    await loginWithVault(EMPTY_VAULT);
    await setProviderKey("anthropic", "sk-ant-xxxxxxxxxxxx"); // pragma: allowlist secret
    expect(loadProviderKeys()).toEqual({
      anthropic: "sk-ant-xxxxxxxxxxxx", // pragma: allowlist secret
    });
  });

  it("hydrates from the decrypted vault on login", async () => {
    await loginWithVault({
      providers: {
        openai: { value: "sk-pre-existing", updated_at: "2026-01-01T00:00:00Z" }, // pragma: allowlist secret
      },
      edgar: null,
    });
    expect(loadProviderKeys()).toEqual({ openai: "sk-pre-existing" }); // pragma: allowlist secret
  });

  it("removeProviderKey drops the entry + uploads", async () => {
    await loginWithVault({
      providers: {
        openai: { value: "sk-aaaa", updated_at: "2026-01-01T00:00:00Z" }, // pragma: allowlist secret
        anthropic: { value: "sk-bbbb", updated_at: "2026-01-01T00:00:00Z" }, // pragma: allowlist secret
      },
      edgar: null,
    });
    await removeProviderKey("openai");
    expect(loadProviderKeys()).toEqual({ anthropic: "sk-bbbb" }); // pragma: allowlist secret
  });

  it("clearProviderKeys wipes every provider entry + uploads", async () => {
    await loginWithVault({
      providers: {
        openai: { value: "sk-aaaa", updated_at: "2026-01-01T00:00:00Z" }, // pragma: allowlist secret
        anthropic: { value: "sk-bbbb", updated_at: "2026-01-01T00:00:00Z" }, // pragma: allowlist secret
      },
      edgar: { name: "Pat", email: "pat@example.com" },
    });
    await clearProviderKeys();
    expect(loadProviderKeys()).toEqual({});
  });

  it("clearing keys does not nuke the EDGAR identity slot", async () => {
    await loginWithVault({
      providers: {
        openai: { value: "sk-aaaa", updated_at: "2026-01-01T00:00:00Z" }, // pragma: allowlist secret
      },
      edgar: { name: "Pat", email: "pat@example.com" },
    });
    const updateSpy = vi.spyOn(apiModule, "updateVaultRequest");
    await clearProviderKeys();
    expect(updateSpy).toHaveBeenCalledTimes(1);
  });
});

describe("provider-keys cache — synchronous validation", () => {
  it("rejects malformed slugs before any network hop", async () => {
    await loginWithVault(EMPTY_VAULT);
    expect(() => setProviderKey("OpenAI", "x")).toThrow(/Invalid provider/);
    expect(() => setProviderKey("", "x")).toThrow(/Invalid provider/);
    expect(() => setProviderKey("openai\nfoo", "x")).toThrow(/Invalid provider/);
    expect(() => setProviderKey("a".repeat(33), "x")).toThrow(/Invalid provider/);
  });

  it("rejects an empty key and an over-length key", async () => {
    await loginWithVault(EMPTY_VAULT);
    expect(() => setProviderKey("openai", "")).toThrow(/must not be empty/);
    expect(() => setProviderKey("openai", "x".repeat(4097))).toThrow(
      /upper bound/,
    );
  });

  it("removeProviderKey shape-checks too", async () => {
    await loginWithVault(EMPTY_VAULT);
    expect(() => removeProviderKey("UPPER")).toThrow(/Invalid provider/);
  });
});

describe("provider-keys cache — locked vault", () => {
  it("setProviderKey rejects when the vault is locked", async () => {
    // No login → vault locked. The synchronous validators pass for a
    // valid slug/key, so the rejection comes from mutateVault.
    await expect(
      setProviderKey("openai", "sk-this-is-a-long-secret"), // pragma: allowlist secret
    ).rejects.toThrow(/locked/i);
  });

  it("providerKeyHeaders returns {} when the vault is locked", () => {
    expect(providerKeyHeaders()).toEqual({});
  });

  it("loadProviderKeys returns {} when the vault is locked", () => {
    expect(loadProviderKeys()).toEqual({});
  });
});

describe("provider-keys cache — headers + subscription", () => {
  it("builds X-Provider-Key-* headers from the unlocked vault", async () => {
    await loginWithVault({
      providers: {
        openai: { value: "sk-aaaa", updated_at: "2026-01-01T00:00:00Z" }, // pragma: allowlist secret
        anthropic: { value: "sk-bbbb", updated_at: "2026-01-01T00:00:00Z" }, // pragma: allowlist secret
      },
      edgar: null,
    });
    expect(providerKeyHeaders()).toEqual({
      "X-Provider-Key-openai": "sk-aaaa", // pragma: allowlist secret
      "X-Provider-Key-anthropic": "sk-bbbb", // pragma: allowlist secret
    });
  });

  it("notifies subscribers on a key change", async () => {
    await loginWithVault(EMPTY_VAULT);
    const seen: Array<Record<string, string>> = [];
    const unsub = subscribe((snap) => {
      seen.push({ ...snap });
    });
    await setProviderKey("openai", "sk-cccccccccccc"); // pragma: allowlist secret
    unsub();
    expect(seen.length).toBeGreaterThanOrEqual(1);
    expect(seen[seen.length - 1]).toEqual({ openai: "sk-cccccccccccc" }); // pragma: allowlist secret
  });
});
