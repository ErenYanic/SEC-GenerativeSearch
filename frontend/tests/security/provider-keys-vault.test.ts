// The legacy `provider-keys.test.ts` pins the plain-mode invariants;
// this file pins the encrypted-mode trust boundary:
//
//   - `enableEncryption()` migrates existing cleartext entries into the
//     sealed vault AND wipes every plain `sec.providerKey.*` entry —
//     `sessionStorage` must not hold the cleartext after this point.
//   - Each write re-encrypts the whole map with a fresh IV (ciphertext
//     never repeats — a regression would leak a known-plaintext oracle).
//   - `lock()` drops the in-memory key but keeps the vault sealed in
//     storage; reads return an empty map until `unlock()`.
//   - `unlock()` with the wrong passphrase raises `InvalidPassphraseError`
//     (AES-GCM auth-tag rejection); the vault stays sealed.
//   - Writes / removes in encrypted+locked state raise `VaultLockedError`
//     — a regression that silently no-op'd would lose the operator's edit.
//   - `clearProviderKeys()` in encrypted mode wipes vault + sentinel +
//     in-memory state so sign-out reverts to plain mode.
//   - `disableEncryption()` requires unlocked state and restores
//     cleartext entries; throws otherwise.
//   - `providerKeyHeaders()` returns an empty map while locked (so a
//     downstream RAG request falls through to the server-side resolver
//     chain instead of shipping partial headers).

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  clearProviderKeys,
  disableEncryption,
  EncryptionAlreadyEnabledError,
  EncryptionDisabledError,
  enableEncryption,
  InvalidPassphraseError,
  isEncrypted,
  isUnlocked,
  lock,
  loadProviderKeys,
  providerKeyHeaders,
  removeProviderKey,
  setProviderKey,
  subscribe,
  unlock,
  VaultLockedError,
} from "@/lib/provider-keys";

const VAULT_KEY = "sec.providerKeyVault";
const MODE_KEY = "sec.providerKeyMode";

beforeEach(() => {
  // Storage + module state are sticky across tests; reset both. The
  // `clearProviderKeys()` call here also walks the encrypted-mode
  // teardown path so any leftover liveKey / liveMap from a previous
  // test is dropped before the next case sets up.
  window.sessionStorage.clear();
  clearProviderKeys();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("enableEncryption", () => {
  it("migrates existing cleartext entries into the sealed vault and wipes the plain entries", async () => {
    await setProviderKey("openai", "sk-aaaaaaaaaaaa"); // pragma: allowlist secret
    await setProviderKey("anthropic", "sk-ant-bbbbbbbbbbbb"); // pragma: allowlist secret

    await enableEncryption("correct-horse-battery-staple");

    expect(isEncrypted()).toBe(true);
    expect(isUnlocked()).toBe(true);

    // No plain entries remain in sessionStorage — only the vault + sentinel.
    expect(window.sessionStorage.getItem("sec.providerKey.openai")).toBeNull();
    expect(
      window.sessionStorage.getItem("sec.providerKey.anthropic"),
    ).toBeNull();
    expect(window.sessionStorage.getItem(MODE_KEY)).toBe("encrypted");

    const vaultRaw = window.sessionStorage.getItem(VAULT_KEY);
    expect(vaultRaw).not.toBeNull();
    const vault = JSON.parse(vaultRaw!);
    expect(vault.v).toBe(1);
    expect(typeof vault.salt).toBe("string");
    expect(typeof vault.iv).toBe("string");
    expect(typeof vault.ciphertext).toBe("string");

    // The stored ciphertext does NOT contain the cleartext key bytes.
    expect(vault.ciphertext).not.toContain("sk-aaaaaaaaaaaa");
    expect(vault.ciphertext).not.toContain("sk-ant-bbbbbbbbbbbb");

    // In-memory snapshot still resolves to the cleartext map (unlocked).
    expect(loadProviderKeys()).toEqual({
      openai: "sk-aaaaaaaaaaaa",
      anthropic: "sk-ant-bbbbbbbbbbbb",
    });
  });

  it("refuses to enable encryption a second time", async () => {
    await enableEncryption("passphrase-one");
    await expect(enableEncryption("passphrase-two")).rejects.toBeInstanceOf(
      EncryptionAlreadyEnabledError,
    );
  });
});

describe("writes in encrypted+unlocked mode re-seal the vault with a fresh IV", () => {
  it("each setProviderKey re-encrypts the whole map and yields a new ciphertext", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-cccccccccccc"); // pragma: allowlist secret
    const firstVault = JSON.parse(window.sessionStorage.getItem(VAULT_KEY)!);

    await setProviderKey("openai", "sk-dddddddddddd"); // pragma: allowlist secret
    const secondVault = JSON.parse(window.sessionStorage.getItem(VAULT_KEY)!);

    // Different IV per write (12-byte AES-GCM nonce must never repeat
    // under the same key — a regression that froze the IV would break
    // GCM confidentiality outright).
    expect(secondVault.iv).not.toBe(firstVault.iv);
    expect(secondVault.ciphertext).not.toBe(firstVault.ciphertext);
    // Salt stays constant for the life of the vault — that is the
    // PBKDF2 input that pairs with the operator passphrase. Changing
    // it per write would render the vault unrecoverable after a lock.
    expect(secondVault.salt).toBe(firstVault.salt);

    // The cleartext never appears in the ciphertext payload.
    expect(secondVault.ciphertext).not.toContain("sk-dddddddddddd");
  });

  it("removeProviderKey re-encrypts the remainder", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-eeeeeeeeeeee"); // pragma: allowlist secret
    await setProviderKey("anthropic", "sk-ant-ffffffffffff"); // pragma: allowlist secret
    await removeProviderKey("openai");

    expect(loadProviderKeys()).toEqual({
      anthropic: "sk-ant-ffffffffffff",
    });
    // The removed key MUST disappear from the on-disk ciphertext too.
    // We round-trip via lock → unlock to prove the persisted vault no
    // longer carries the entry.
    lock();
    await unlock("correct-horse-battery-staple");
    expect(loadProviderKeys()).toEqual({
      anthropic: "sk-ant-ffffffffffff",
    });
  });
});

describe("lock + unlock", () => {
  it("lock() drops the in-memory map but keeps the vault sealed in storage", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-gggggggggggg"); // pragma: allowlist secret

    lock();

    expect(isEncrypted()).toBe(true);
    expect(isUnlocked()).toBe(false);
    expect(loadProviderKeys()).toEqual({});
    expect(providerKeyHeaders()).toEqual({});
    // Vault + sentinel stay so a later unlock can restore the keys.
    expect(window.sessionStorage.getItem(VAULT_KEY)).not.toBeNull();
    expect(window.sessionStorage.getItem(MODE_KEY)).toBe("encrypted");
  });

  it("unlock() restores the cleartext map", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-hhhhhhhhhhhh"); // pragma: allowlist secret
    await setProviderKey("anthropic", "sk-ant-iiiiiiiiiiii"); // pragma: allowlist secret
    lock();

    await unlock("correct-horse-battery-staple");

    expect(isUnlocked()).toBe(true);
    expect(loadProviderKeys()).toEqual({
      openai: "sk-hhhhhhhhhhhh",
      anthropic: "sk-ant-iiiiiiiiiiii",
    });
  });

  it("unlock() with the wrong passphrase raises InvalidPassphraseError and leaves the vault sealed", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-jjjjjjjjjjjj"); // pragma: allowlist secret
    lock();

    await expect(unlock("not-the-passphrase")).rejects.toBeInstanceOf(
      InvalidPassphraseError,
    );

    // Still locked — a regression that half-unlocked the seam would
    // be silently catastrophic.
    expect(isEncrypted()).toBe(true);
    expect(isUnlocked()).toBe(false);
    expect(loadProviderKeys()).toEqual({});
  });

  it("unlock() outside encrypted mode raises EncryptionDisabledError", async () => {
    await expect(unlock("anything")).rejects.toBeInstanceOf(
      EncryptionDisabledError,
    );
  });
});

describe("set / remove while locked", () => {
  it("setProviderKey raises VaultLockedError without touching storage", async () => {
    await enableEncryption("correct-horse-battery-staple");
    const beforeVault = window.sessionStorage.getItem(VAULT_KEY);
    lock();

    expect(() =>
      setProviderKey("openai", "sk-kkkkkkkkkkkk"),
    ).toThrow(VaultLockedError);

    // Vault must be byte-identical — a silent re-seal under the
    // attacker's passphrase would be an exfiltration trap.
    expect(window.sessionStorage.getItem(VAULT_KEY)).toBe(beforeVault);
  });

  it("removeProviderKey raises VaultLockedError", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-llllllllllll"); // pragma: allowlist secret
    lock();

    expect(() => removeProviderKey("openai")).toThrow(VaultLockedError);
  });
});

describe("validation throws stay synchronous even in encrypted mode", () => {
  it("setProviderKey shape-check throws sync (not a rejected Promise)", async () => {
    await enableEncryption("correct-horse-battery-staple");
    expect(() => setProviderKey("UPPER", "sk-x")).toThrow(/Invalid provider/);
  });

  it("setProviderKey empty-key throws sync", async () => {
    await enableEncryption("correct-horse-battery-staple");
    expect(() => setProviderKey("openai", "")).toThrow(/must not be empty/);
  });
});

describe("clearProviderKeys in encrypted mode", () => {
  it("wipes the vault, the sentinel, and the in-memory CryptoKey", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-mmmmmmmmmmmm"); // pragma: allowlist secret

    clearProviderKeys();

    expect(isEncrypted()).toBe(false);
    expect(isUnlocked()).toBe(false);
    expect(window.sessionStorage.getItem(VAULT_KEY)).toBeNull();
    expect(window.sessionStorage.getItem(MODE_KEY)).toBeNull();
    expect(loadProviderKeys()).toEqual({});
  });

  it("after clearProviderKeys, plain-mode writes resume", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-nnnnnnnnnnnn"); // pragma: allowlist secret
    clearProviderKeys();

    await setProviderKey("openai", "sk-oooooooooooo"); // pragma: allowlist secret
    expect(window.sessionStorage.getItem("sec.providerKey.openai")).toBe(
      "sk-oooooooooooo",
    );
  });
});

describe("disableEncryption", () => {
  it("restores cleartext entries when unlocked", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-pppppppppppp"); // pragma: allowlist secret
    await setProviderKey("anthropic", "sk-ant-qqqqqqqqqqqq"); // pragma: allowlist secret

    await disableEncryption();

    expect(isEncrypted()).toBe(false);
    expect(window.sessionStorage.getItem(VAULT_KEY)).toBeNull();
    expect(window.sessionStorage.getItem(MODE_KEY)).toBeNull();
    expect(window.sessionStorage.getItem("sec.providerKey.openai")).toBe(
      "sk-pppppppppppp",
    );
    expect(window.sessionStorage.getItem("sec.providerKey.anthropic")).toBe(
      "sk-ant-qqqqqqqqqqqq",
    );
  });

  it("raises VaultLockedError when called while locked", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-rrrrrrrrrrrr"); // pragma: allowlist secret
    lock();

    await expect(disableEncryption()).rejects.toBeInstanceOf(VaultLockedError);
    // Vault remains sealed; the cleartext does NOT escape on a locked
    // disable attempt.
    expect(isEncrypted()).toBe(true);
    expect(window.sessionStorage.getItem("sec.providerKey.openai")).toBeNull();
  });

  it("raises EncryptionDisabledError when called in plain mode", async () => {
    await expect(disableEncryption()).rejects.toBeInstanceOf(
      EncryptionDisabledError,
    );
  });
});

describe("snapshot reference + subscribe contract", () => {
  it("subscribers fire on every state transition", async () => {
    const listener = vi.fn();
    const unsubscribe = subscribe(listener);

    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-ssssssssssss"); // pragma: allowlist secret
    lock();
    await unlock("correct-horse-battery-staple");
    await disableEncryption();

    // enable + set + lock + unlock + disable = 5 transitions minimum.
    expect(listener.mock.calls.length).toBeGreaterThanOrEqual(5);

    unsubscribe();
  });

  it("locked snapshot is a frozen empty map (stable reference for useSyncExternalStore)", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-tttttttttttt"); // pragma: allowlist secret
    lock();

    const snapA = loadProviderKeys();
    const snapB = loadProviderKeys();
    expect(snapA).toBe(snapB);
    expect(Object.isFrozen(snapA)).toBe(true);
    expect(snapA).toEqual({});
  });
});

describe("providerKeyHeaders while locked", () => {
  it("returns an empty map so outbound requests fall through to the server-side resolver", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-uuuuuuuuuuuu"); // pragma: allowlist secret
    lock();

    expect(providerKeyHeaders()).toEqual({});
  });

  it("attaches headers once unlocked", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-vvvvvvvvvvvv"); // pragma: allowlist secret
    lock();
    await unlock("correct-horse-battery-staple");

    expect(providerKeyHeaders()).toEqual({
      "X-Provider-Key-openai": "sk-vvvvvvvvvvvv",
    });
  });
});

describe("trust boundary: ciphertext never carries the cleartext payload", () => {
  it("the on-disk vault does not contain any cleartext key substring", async () => {
    await enableEncryption("correct-horse-battery-staple");
    await setProviderKey("openai", "sk-uniqueneedlex"); // pragma: allowlist secret
    await setProviderKey("anthropic", "sk-ant-uniqueneedley"); // pragma: allowlist secret

    const vaultRaw = window.sessionStorage.getItem(VAULT_KEY);
    expect(vaultRaw).not.toBeNull();
    expect(vaultRaw).not.toContain("sk-uniqueneedlex");
    expect(vaultRaw).not.toContain("sk-ant-uniqueneedley");
    // Provider slugs DO appear inside the encrypted JSON payload but
    // only as part of the ciphertext — never as a recoverable string
    // in the wrapper JSON.
    const wrapper = JSON.parse(vaultRaw!);
    expect(Object.keys(wrapper).sort()).toEqual([
      "ciphertext",
      "iv",
      "salt",
      "v",
    ]);
  });
});
