// Pattern D crypto + state machine regression net.
//
// What this test file pins:
//
//   1. PBKDF2 + HKDF split — the same password + salt MUST produce the
//      same auth_proof + a KEK that round-trips a vault. Changing the
//      derivation breaks every enrolled user, so the test fixtures use
//      a deterministic vector.
//   2. `info` strings are LOAD-BEARING and DISTINCT — using either
//      string twice MUST yield different output. Collapsing them turns
//      `auth_hash` into a decryption-capable artefact.
//   3. KEK is non-extractable — `crypto.subtle.exportKey` MUST refuse.
//      A future XSS landing on the page cannot dump the raw key bytes.
//   4. Fresh IV per write — two `encryptVault` calls against the same
//      cleartext MUST yield two different IVs (and two different
//      ciphertexts). AES-GCM IV reuse against the same key is
//      catastrophic.
//   5. base64url helpers round-trip arbitrary bytes (no alphabet drift).
//   6. login / enrol / sign-out wire shape — the network seams are
//      mocked; we assert what crosses the wire, never the password.
//   7. Sign-out wipes module-local state BEFORE the network call fires.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import * as apiModule from "@/lib/api";
import {
  changePassword,
  decryptVault,
  derivePasswordMaterial,
  encryptVault,
  enrolUser,
  isUnlocked,
  loginUser,
  mutateVault,
  readEdgarIdentity,
  readProviders,
  resetLocalState,
  signOutUser,
  snapshot,
  subscribe,
  _internals,
  type VaultCleartext,
} from "@/lib/user-vault";

// PBKDF2 at 600 000 iterations is too slow for tests. The
// production code reads iterations from the server's login-params
// response, so each test mocks the API and supplies a low iteration
// count. 1 000 is well above the SubtleCrypto floor and finishes in a
// few milliseconds on every CI runner we use.
const TEST_ITERATIONS = 1_000;

const EMPTY_VAULT: VaultCleartext = Object.freeze({
  providers: Object.freeze({}),
  edgar: null,
});

afterEach(() => {
  resetLocalState();
  vi.restoreAllMocks();
});

describe("derivePasswordMaterial", () => {
  it("is deterministic over (password, salt, iterations)", async () => {
    const salt = new Uint8Array(16).fill(0x41);
    const first = await derivePasswordMaterial("correct horse", salt, TEST_ITERATIONS);
    const second = await derivePasswordMaterial("correct horse", salt, TEST_ITERATIONS);
    expect(first.authProof).toEqual(second.authProof);
  });

  it("changes when the password changes", async () => {
    const salt = new Uint8Array(16).fill(0x42);
    const a = await derivePasswordMaterial("alpha", salt, TEST_ITERATIONS);
    const b = await derivePasswordMaterial("beta", salt, TEST_ITERATIONS);
    expect(a.authProof).not.toEqual(b.authProof);
  });

  it("changes when the salt changes", async () => {
    const a = await derivePasswordMaterial(
      "p", new Uint8Array(16).fill(0x01), TEST_ITERATIONS,
    );
    const b = await derivePasswordMaterial(
      "p", new Uint8Array(16).fill(0x02), TEST_ITERATIONS,
    );
    expect(a.authProof).not.toEqual(b.authProof);
  });

  it("produces a 32-byte auth_proof", async () => {
    const { authProof } = await derivePasswordMaterial(
      "x", new Uint8Array(16), TEST_ITERATIONS,
    );
    expect(authProof.byteLength).toBe(32);
  });

  it("refuses an empty password", async () => {
    await expect(
      derivePasswordMaterial("", new Uint8Array(16), TEST_ITERATIONS),
    ).rejects.toThrow(/password/i);
  });

  it("refuses zero iterations", async () => {
    await expect(
      derivePasswordMaterial("p", new Uint8Array(16), 0),
    ).rejects.toThrow();
  });
});

describe("HKDF info domain separation (load-bearing)", () => {
  it("authProof and KEK output do not collide for the same input", async () => {
    // The KEK is non-extractable, so we can't compare bytes. Instead
    // we round-trip a ciphertext encrypted under the KEK and then try
    // to "decrypt" the same blob under a fake KEK derived as if the
    // info string were the auth one. The fake decrypt MUST fail.
    const salt = new Uint8Array(16).fill(0x55);
    const { kek } = await derivePasswordMaterial("p", salt, TEST_ITERATIONS);
    const blob = await encryptVault(kek, EMPTY_VAULT);
    // Reconstruct a key as if HKDF was run with the AUTH info string
    // — by importing the 32-byte authProof as an AES-GCM key and
    // attempting to decrypt. The auth proof is the wrong domain; the
    // decrypt MUST throw (auth-tag mismatch).
    const { authProof } = await derivePasswordMaterial("p", salt, TEST_ITERATIONS);
    const fakeKek = await globalThis.crypto.subtle.importKey(
      "raw",
      authProof as BufferSource,
      { name: "AES-GCM" },
      false,
      ["decrypt"],
    );
    await expect(
      decryptVault(fakeKek, blob.ciphertext, blob.iv),
    ).rejects.toThrow();
  });

  it("info constants are DISTINCT strings — collapsing them breaks the model", () => {
    expect(_internals.HKDF_AUTH_INFO).not.toBe(_internals.HKDF_KEK_INFO);
    expect(_internals.HKDF_AUTH_INFO).toMatch(/sec-gs\/auth/);
    expect(_internals.HKDF_KEK_INFO).toMatch(/sec-gs\/kek/);
  });
});

describe("KEK extractability", () => {
  it("refuses exportKey on the derived KEK", async () => {
    const salt = new Uint8Array(16).fill(0xaa);
    const { kek } = await derivePasswordMaterial("p", salt, TEST_ITERATIONS);
    await expect(
      globalThis.crypto.subtle.exportKey("raw", kek),
    ).rejects.toThrow();
  });
});

describe("encryptVault / decryptVault", () => {
  it("round-trips a vault under the same KEK", async () => {
    const salt = new Uint8Array(16).fill(0x11);
    const { kek } = await derivePasswordMaterial("p", salt, TEST_ITERATIONS);
    const plain: VaultCleartext = {
      providers: {
        openai: { value: "sk-test-key", updated_at: "2026-05-26T00:00:00Z" }, // pragma: allowlist secret
      },
      edgar: { name: "Pat", email: "pat@example.com" },
    };
    const blob = await encryptVault(kek, plain);
    const decoded = await decryptVault(kek, blob.ciphertext, blob.iv);
    expect(decoded.providers.openai?.value).toBe("sk-test-key"); // pragma: allowlist secret
    expect(decoded.edgar).toEqual({ name: "Pat", email: "pat@example.com" });
  });

  it("uses a fresh IV per write (AES-GCM IV reuse is catastrophic)", async () => {
    const salt = new Uint8Array(16);
    const { kek } = await derivePasswordMaterial("p", salt, TEST_ITERATIONS);
    const blobA = await encryptVault(kek, EMPTY_VAULT);
    const blobB = await encryptVault(kek, EMPTY_VAULT);
    expect(blobA.iv).not.toEqual(blobB.iv);
    expect(blobA.ciphertext).not.toEqual(blobB.ciphertext);
  });

  it("rejects a wrong KEK with InvalidPasswordError", async () => {
    const salt = new Uint8Array(16);
    const { kek: realKek } = await derivePasswordMaterial("right", salt, TEST_ITERATIONS);
    const { kek: wrongKek } = await derivePasswordMaterial("wrong", salt, TEST_ITERATIONS);
    const blob = await encryptVault(realKek, EMPTY_VAULT);
    await expect(
      decryptVault(wrongKek, blob.ciphertext, blob.iv),
    ).rejects.toThrow(/password/i);
  });
});

describe("base64url helpers (wire encoding)", () => {
  it("round-trips arbitrary bytes including padding edges", () => {
    const cases = [
      new Uint8Array([]),
      new Uint8Array([0x00]),
      new Uint8Array([0xff, 0xfe, 0xfd]),
      new Uint8Array(32).map((_, i) => i & 0xff),
    ];
    for (const sample of cases) {
      const encoded = _internals.bytesToBase64Url(sample);
      const decoded = _internals.base64UrlToBytes(encoded);
      expect(decoded).toEqual(sample);
    }
  });

  it("uses the URL-safe alphabet (no + or / or =)", () => {
    const encoded = _internals.bytesToBase64Url(
      new Uint8Array([0xfb, 0xff, 0xfe]),
    );
    expect(encoded).not.toMatch(/[+/=]/);
  });
});

// ---------------------------------------------------------------------------
// Network seams — login / enrol / sign-out / mutate
// ---------------------------------------------------------------------------

function freshSalt(byte: number): { saltBytes: Uint8Array; b64: string } {
  const saltBytes = new Uint8Array(16).fill(byte);
  const b64 = _internals.bytesToBase64Url(saltBytes);
  return { saltBytes, b64 };
}

async function freshVaultBlob(
  password: string,
  saltBytes: Uint8Array,
  cleartext: VaultCleartext,
): Promise<{ ciphertextB64: string; ivB64: string }> {
  const { kek } = await derivePasswordMaterial(password, saltBytes, TEST_ITERATIONS);
  const blob = await encryptVault(kek, cleartext);
  return {
    ciphertextB64: _internals.bytesToBase64Url(blob.ciphertext),
    ivB64: _internals.bytesToBase64Url(blob.iv),
  };
}

describe("loginUser", () => {
  it("derives auth_proof + KEK, posts, decrypts, hydrates the cache", async () => {
    const { saltBytes, b64: saltB64 } = freshSalt(0xab);
    const cleartext: VaultCleartext = {
      providers: {
        anthropic: { value: "sk-ant-vault", updated_at: "2026-01-01T00:00:00Z" }, // pragma: allowlist secret
      },
      edgar: null,
    };
    const { ciphertextB64, ivB64 } = await freshVaultBlob(
      "correct horse",
      saltBytes,
      cleartext,
    );

    const loginParamsSpy = vi
      .spyOn(apiModule, "loginParamsRequest")
      .mockResolvedValue({
        salt_m: saltB64,
        kdf_algo: "pbkdf2-sha256",
        pbkdf2_iterations: TEST_ITERATIONS,
      });
    const loginSpy = vi.spyOn(apiModule, "loginRequest").mockResolvedValue({
      user_id: 42,
      username: "pat",
      ciphertext_vault: ciphertextB64,
      vault_iv: ivB64,
    });

    const session = await loginUser("pat", "correct horse");

    expect(loginParamsSpy).toHaveBeenCalledWith("pat");
    expect(loginSpy).toHaveBeenCalledTimes(1);
    const sent = loginSpy.mock.calls[0]?.[0];
    expect(sent?.username).toBe("pat");
    // auth_proof is 32 raw bytes → 43 base64url chars (no padding).
    expect(sent?.auth_proof).toMatch(/^[A-Za-z0-9_-]+$/);
    expect(sent?.auth_proof.length).toBe(43);
    // The password MUST NOT appear anywhere on the wire payload.
    expect(JSON.stringify(sent)).not.toContain("correct horse");

    expect(session.userId).toBe(42);
    expect(session.username).toBe("pat");
    expect(isUnlocked()).toBe(true);
    expect(snapshot()?.providers.anthropic?.value).toBe("sk-ant-vault"); // pragma: allowlist secret
  });

  it("does not hydrate the cache on a refused login", async () => {
    const { b64: saltB64 } = freshSalt(0x11);
    vi.spyOn(apiModule, "loginParamsRequest").mockResolvedValue({
      salt_m: saltB64,
      kdf_algo: "pbkdf2-sha256",
      pbkdf2_iterations: TEST_ITERATIONS,
    });
    vi.spyOn(apiModule, "loginRequest").mockRejectedValue(
      new apiModule.ApiError(401, "login_refused", "Login refused."),
    );
    await expect(loginUser("pat", "wrong")).rejects.toBeInstanceOf(
      apiModule.ApiError,
    );
    expect(isUnlocked()).toBe(false);
    expect(snapshot()).toBeNull();
  });
});

describe("enrolUser", () => {
  it("ships fresh salt + ciphertext, never the password", async () => {
    const enrolSpy = vi.spyOn(apiModule, "enrolUserRequest").mockResolvedValue({
      enrolled: true,
      user_id: 7,
      username: "newcomer",
    });
    const result = await enrolUser("token-abc", "hunter2 hunter2");
    expect(result).toEqual({ user_id: 7, username: "newcomer" });
    const sent = enrolSpy.mock.calls[0]?.[0];
    expect(sent?.token).toBe("token-abc");
    expect(sent?.kdf_algo).toBe("pbkdf2-sha256");
    expect(sent?.pbkdf2_iterations).toBeGreaterThanOrEqual(100_000);
    // 16-byte salt → 22 base64url chars (no padding).
    expect(sent?.salt_m.length).toBe(22);
    expect(sent?.auth_proof.length).toBe(43);
    // 12-byte IV → 16 base64url chars (no padding).
    expect(sent?.vault_iv.length).toBe(16);
    expect(JSON.stringify(sent)).not.toContain("hunter2 hunter2");
  });

  it("does not hydrate the in-memory cache (login is the only seam)", async () => {
    vi.spyOn(apiModule, "enrolUserRequest").mockResolvedValue({
      enrolled: true,
      user_id: 1,
      username: "pat",
    });
    await enrolUser("token", "password");
    expect(isUnlocked()).toBe(false);
  });
});

describe("mutateVault", () => {
  it("refuses when the vault is locked", async () => {
    await expect(
      mutateVault((v) => v),
    ).rejects.toThrow(/locked/i);
  });

  it("re-encrypts + posts; mutates the local cache on success", async () => {
    // First log in so the vault has live state.
    const { saltBytes, b64: saltB64 } = freshSalt(0x33);
    const { ciphertextB64, ivB64 } = await freshVaultBlob(
      "pw",
      saltBytes,
      EMPTY_VAULT,
    );
    vi.spyOn(apiModule, "loginParamsRequest").mockResolvedValue({
      salt_m: saltB64,
      kdf_algo: "pbkdf2-sha256",
      pbkdf2_iterations: TEST_ITERATIONS,
    });
    vi.spyOn(apiModule, "loginRequest").mockResolvedValue({
      user_id: 1,
      username: "pat",
      ciphertext_vault: ciphertextB64,
      vault_iv: ivB64,
    });
    await loginUser("pat", "pw");

    const updateSpy = vi
      .spyOn(apiModule, "updateVaultRequest")
      .mockResolvedValue({ updated: true });

    await mutateVault((current) => ({
      ...current,
      providers: {
        ...current.providers,
        openai: { value: "sk-fresh", updated_at: "2026-05-26T00:00:00Z" }, // pragma: allowlist secret
      },
    }));

    expect(updateSpy).toHaveBeenCalledTimes(1);
    const sent = updateSpy.mock.calls[0]?.[0];
    expect(sent?.vault_iv.length).toBe(16);
    expect(sent?.ciphertext_vault.length).toBeGreaterThan(0);
    expect(readProviders().openai?.value).toBe("sk-fresh"); // pragma: allowlist secret
  });

  it("rolls back the local cache when the upload fails", async () => {
    const { saltBytes, b64: saltB64 } = freshSalt(0x44);
    const { ciphertextB64, ivB64 } = await freshVaultBlob(
      "pw",
      saltBytes,
      {
        providers: { keep: { value: "x", updated_at: "2026-01-01T00:00:00Z" } },
        edgar: null,
      },
    );
    vi.spyOn(apiModule, "loginParamsRequest").mockResolvedValue({
      salt_m: saltB64,
      kdf_algo: "pbkdf2-sha256",
      pbkdf2_iterations: TEST_ITERATIONS,
    });
    vi.spyOn(apiModule, "loginRequest").mockResolvedValue({
      user_id: 1,
      username: "pat",
      ciphertext_vault: ciphertextB64,
      vault_iv: ivB64,
    });
    await loginUser("pat", "pw");

    vi.spyOn(apiModule, "updateVaultRequest").mockRejectedValue(
      new apiModule.ApiError(500, "database_error", "fail"),
    );

    await expect(
      mutateVault((v) => ({ ...v, providers: {} })),
    ).rejects.toBeInstanceOf(apiModule.ApiError);
    // Original entry survives the failed update.
    expect(readProviders().keep?.value).toBe("x");
  });
});

describe("EDGAR identity slot", () => {
  it("is `null` while locked", () => {
    expect(readEdgarIdentity()).toBeNull();
  });

  it("flows through mutateVault unchanged", async () => {
    const { saltBytes, b64: saltB64 } = freshSalt(0x55);
    const { ciphertextB64, ivB64 } = await freshVaultBlob(
      "pw",
      saltBytes,
      EMPTY_VAULT,
    );
    vi.spyOn(apiModule, "loginParamsRequest").mockResolvedValue({
      salt_m: saltB64,
      kdf_algo: "pbkdf2-sha256",
      pbkdf2_iterations: TEST_ITERATIONS,
    });
    vi.spyOn(apiModule, "loginRequest").mockResolvedValue({
      user_id: 1,
      username: "pat",
      ciphertext_vault: ciphertextB64,
      vault_iv: ivB64,
    });
    await loginUser("pat", "pw");

    vi.spyOn(apiModule, "updateVaultRequest").mockResolvedValue({ updated: true });
    await mutateVault((current) => ({
      ...current,
      edgar: { name: "Pat", email: "pat@example.com" },
    }));

    expect(readEdgarIdentity()).toEqual({ name: "Pat", email: "pat@example.com" });
  });
});

describe("signOutUser", () => {
  it("wipes local state BEFORE the network call (network failure cannot leak state)", async () => {
    const { saltBytes, b64: saltB64 } = freshSalt(0x66);
    const { ciphertextB64, ivB64 } = await freshVaultBlob(
      "pw",
      saltBytes,
      EMPTY_VAULT,
    );
    vi.spyOn(apiModule, "loginParamsRequest").mockResolvedValue({
      salt_m: saltB64,
      kdf_algo: "pbkdf2-sha256",
      pbkdf2_iterations: TEST_ITERATIONS,
    });
    vi.spyOn(apiModule, "loginRequest").mockResolvedValue({
      user_id: 1,
      username: "pat",
      ciphertext_vault: ciphertextB64,
      vault_iv: ivB64,
    });
    await loginUser("pat", "pw");
    expect(isUnlocked()).toBe(true);

    let wasUnlockedDuringNetworkCall = true;
    vi.spyOn(apiModule, "signOutRequest").mockImplementation(async () => {
      wasUnlockedDuringNetworkCall = isUnlocked();
      return { cleared: true };
    });
    await signOutUser();
    expect(wasUnlockedDuringNetworkCall).toBe(false);
    expect(isUnlocked()).toBe(false);
    expect(snapshot()).toBeNull();
  });

  it("survives a 502 from the sign-out endpoint", async () => {
    const { saltBytes, b64: saltB64 } = freshSalt(0x77);
    const { ciphertextB64, ivB64 } = await freshVaultBlob(
      "pw",
      saltBytes,
      EMPTY_VAULT,
    );
    vi.spyOn(apiModule, "loginParamsRequest").mockResolvedValue({
      salt_m: saltB64,
      kdf_algo: "pbkdf2-sha256",
      pbkdf2_iterations: TEST_ITERATIONS,
    });
    vi.spyOn(apiModule, "loginRequest").mockResolvedValue({
      user_id: 1,
      username: "pat",
      ciphertext_vault: ciphertextB64,
      vault_iv: ivB64,
    });
    await loginUser("pat", "pw");

    vi.spyOn(apiModule, "signOutRequest").mockRejectedValue(
      new apiModule.ApiError(502, "backend_unreachable", "x"),
    );
    await expect(signOutUser()).resolves.toBeUndefined();
    expect(isUnlocked()).toBe(false);
  });
});

describe("subscribe", () => {
  it("fires listeners on every mutation including sign-out", async () => {
    const seen: Array<VaultCleartext | null> = [];
    const unsub = subscribe((snap) => {
      seen.push(snap);
    });
    const { saltBytes, b64: saltB64 } = freshSalt(0x88);
    const { ciphertextB64, ivB64 } = await freshVaultBlob(
      "pw",
      saltBytes,
      EMPTY_VAULT,
    );
    vi.spyOn(apiModule, "loginParamsRequest").mockResolvedValue({
      salt_m: saltB64,
      kdf_algo: "pbkdf2-sha256",
      pbkdf2_iterations: TEST_ITERATIONS,
    });
    vi.spyOn(apiModule, "loginRequest").mockResolvedValue({
      user_id: 1,
      username: "pat",
      ciphertext_vault: ciphertextB64,
      vault_iv: ivB64,
    });
    vi.spyOn(apiModule, "signOutRequest").mockResolvedValue({ cleared: true });
    await loginUser("pat", "pw");
    await signOutUser();
    unsub();
    // At minimum: one notify after login (snapshot is the new vault),
    // one after sign-out (snapshot is null).
    expect(seen.length).toBeGreaterThanOrEqual(2);
    expect(seen[seen.length - 1]).toBeNull();
  });
});

describe("changePassword", () => {
  it("rotates salt + KEK and ships the new ciphertext atomically", async () => {
    // Initial login.
    const { saltBytes: oldSalt, b64: oldSaltB64 } = freshSalt(0x99);
    const initialVault: VaultCleartext = {
      providers: { openai: { value: "keep-me", updated_at: "2026-01-01T00:00:00Z" } },
      edgar: null,
    };
    const { ciphertextB64, ivB64 } = await freshVaultBlob(
      "old-password", oldSalt, initialVault,
    );

    // The login flow calls loginParamsRequest, then loginRequest.
    // changePassword internally calls loginParamsRequest AGAIN to get
    // the old salt. We program the spy to return the SAME salt on
    // every call (fine because the old-password salt is the only
    // valid one until rotation succeeds).
    vi.spyOn(apiModule, "loginParamsRequest").mockResolvedValue({
      salt_m: oldSaltB64,
      kdf_algo: "pbkdf2-sha256",
      pbkdf2_iterations: TEST_ITERATIONS,
    });
    vi.spyOn(apiModule, "loginRequest").mockResolvedValue({
      user_id: 5,
      username: "pat",
      ciphertext_vault: ciphertextB64,
      vault_iv: ivB64,
    });
    await loginUser("pat", "old-password");

    const changeSpy = vi
      .spyOn(apiModule, "changePasswordRequest")
      .mockResolvedValue({ rotated: true });
    await changePassword("old-password", "new-password");

    const sent = changeSpy.mock.calls[0]?.[0];
    expect(sent?.auth_proof_old).toMatch(/^[A-Za-z0-9_-]+$/);
    expect(sent?.auth_proof_new).toMatch(/^[A-Za-z0-9_-]+$/);
    expect(sent?.auth_proof_old).not.toBe(sent?.auth_proof_new);
    // New salt is freshly minted — different from the old one.
    expect(sent?.salt_m).not.toBe(oldSaltB64);
    expect(sent?.kdf_algo).toBe("pbkdf2-sha256");
    expect(sent?.vault_iv.length).toBe(16);
    // The KEK rotation must keep the SPA signed in (live cache
    // preserved). We test the cleartext map survives.
    expect(snapshot()?.providers.openai?.value).toBe("keep-me");

    expect(JSON.stringify(sent)).not.toContain("old-password");
    expect(JSON.stringify(sent)).not.toContain("new-password");
  });
});
