// Browser-side provider-key store (`src/lib/provider-keys.ts`).
//
// Asserts the trust-boundary invariants the store depends on:
//   - keys live in `sessionStorage` only — `localStorage` is never
//     touched at runtime
//   - invalid provider slugs are rejected at write time (so the backend
//     parser never gets a slug it would discard)
//   - empty / oversized keys are rejected
//   - subscribers are notified on every mutation
//   - `providerKeyHeaders()` builds `X-Provider-Key-{provider}` headers
//   - the snapshot is read-only (callers cannot mutate the store
//     through the returned reference)
//   - `clearProviderKeys()` only touches the namespaced keys

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  clearProviderKeys,
  loadProviderKeys,
  providerKeyHeaders,
  removeProviderKey,
  setProviderKey,
  subscribe,
} from "@/lib/provider-keys";

beforeEach(() => {
  // happy-dom retains storage between tests; wipe both stores so each
  // case starts from a clean slate. `clearProviderKeys()` also drops
  // the module's in-memory snapshot cache (which would otherwise
  // outlive a raw `sessionStorage.clear()`).
  window.sessionStorage.clear();
  window.localStorage.clear();
  clearProviderKeys();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("provider-keys store: trust boundary", () => {
  it("persists keys to sessionStorage and NEVER to localStorage", () => {
    setProviderKey("openai", "sk-this-is-a-long-secret"); // pragma: allowlist secret

    expect(window.sessionStorage.getItem("sec.providerKey.openai")).toBe(
      "sk-this-is-a-long-secret",
    );
    // Nothing in this module touches `localStorage`; a future drive-by
    // edit that copies the value over is caught by this assertion.
    expect(window.localStorage.length).toBe(0);
  });

  it("snapshot returns only namespaced entries — ignores unrelated localStorage / sessionStorage keys", () => {
    window.sessionStorage.setItem("unrelated", "should-not-appear");
    window.localStorage.setItem(
      "sec.providerKey.poisoned",
      "should-not-appear", // pragma: allowlist secret
    );
    setProviderKey("anthropic", "sk-ant-xxxxxxxxxxxx"); // pragma: allowlist secret

    const snap = loadProviderKeys();
    expect(Object.keys(snap)).toEqual(["anthropic"]);
    expect(snap.anthropic).toBe("sk-ant-xxxxxxxxxxxx");
  });

  it("returned snapshot is frozen — callers cannot mutate the store through it", () => {
    setProviderKey("openai", "sk-aaaaaaaaaaaa"); // pragma: allowlist secret
    const snap = loadProviderKeys();
    expect(Object.isFrozen(snap)).toBe(true);
    // Strict mode throws on writes to a frozen object.
    expect(() => {
      (snap as Record<string, string>).openai = "tampered";
    }).toThrow();
  });

  it("clearProviderKeys only wipes the namespaced keys", () => {
    window.sessionStorage.setItem("unrelated", "keep");
    setProviderKey("openai", "sk-yyyyyyyyyyyy"); // pragma: allowlist secret
    setProviderKey("anthropic", "sk-ant-zzzzzzzzzzzz"); // pragma: allowlist secret

    clearProviderKeys();

    expect(loadProviderKeys()).toEqual({});
    expect(window.sessionStorage.getItem("unrelated")).toBe("keep");
  });
});

describe("provider-keys store: validation", () => {
  it("rejects provider slugs that fail the backend shape check", () => {
    // Uppercase — would be discarded silently by parse_provider_key_headers.
    expect(() => setProviderKey("OpenAI", "x")).toThrow(/Invalid provider/);
    // Empty.
    expect(() => setProviderKey("", "x")).toThrow(/Invalid provider/);
    // CR/LF — header-injection vector.
    expect(() => setProviderKey("openai\nfoo", "x")).toThrow(/Invalid provider/);
    // Oversized.
    expect(() => setProviderKey("a".repeat(33), "x")).toThrow(/Invalid provider/);
  });

  it("rejects empty API keys", () => {
    expect(() => setProviderKey("openai", "")).toThrow(/must not be empty/);
  });

  it("rejects keys above the 4096-character backend bound", () => {
    expect(() => setProviderKey("openai", "x".repeat(4097))).toThrow(
      /exceeds the 4096/,
    );
  });

  it("removeProviderKey ignores never-set entries (no throw)", () => {
    expect(() => removeProviderKey("openai")).not.toThrow();
  });

  it("removeProviderKey shape-checks too — refuses a malformed slug", () => {
    expect(() => removeProviderKey("UPPER")).toThrow(/Invalid provider/);
  });
});

describe("provider-keys store: subscribe / notify", () => {
  it("notifies subscribers on set, remove, and clear", () => {
    const listener = vi.fn();
    const unsubscribe = subscribe(listener);

    setProviderKey("openai", "sk-aaaaaaaaaaaa"); // pragma: allowlist secret
    setProviderKey("anthropic", "sk-ant-bbbbbbbbbbb"); // pragma: allowlist secret
    removeProviderKey("openai");
    clearProviderKeys();

    expect(listener).toHaveBeenCalledTimes(4);
    // Last snapshot is empty.
    const lastCall = listener.mock.calls[3];
    expect(lastCall).toBeDefined();
    expect(lastCall![0]).toEqual({});

    unsubscribe();
    setProviderKey("openai", "sk-cccccccccccc"); // pragma: allowlist secret
    expect(listener).toHaveBeenCalledTimes(4);
  });

  it("a throwing subscriber does not poison the rest of the chain", () => {
    const ok = vi.fn();
    subscribe(() => {
      throw new Error("listener exploded");
    });
    subscribe(ok);

    setProviderKey("openai", "sk-dddddddddddd"); // pragma: allowlist secret
    expect(ok).toHaveBeenCalledTimes(1);
  });
});

describe("providerKeyHeaders()", () => {
  it("builds a header map of X-Provider-Key-{provider} entries", () => {
    setProviderKey("openai", "sk-eeeeeeeeeeee"); // pragma: allowlist secret
    setProviderKey("anthropic", "sk-ant-ffffffffffff"); // pragma: allowlist secret

    const headers = providerKeyHeaders();
    expect(headers["X-Provider-Key-openai"]).toBe("sk-eeeeeeeeeeee");
    expect(headers["X-Provider-Key-anthropic"]).toBe("sk-ant-ffffffffffff");
    // No bare-name leakage.
    expect(headers["openai"]).toBeUndefined();
  });

  it("returns an empty map when the store is empty", () => {
    expect(providerKeyHeaders()).toEqual({});
  });
});
