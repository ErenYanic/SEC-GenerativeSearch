// EDGAR identity card UI.
//
// Asserts:
//  - submitting pushes to the session store (POST /api/admin/session/edgar)
//    AND persists into the encrypted vault (POST /api/admin/auth/vault)
//  - on success, the card swaps to the vault-saved state and clears
//    local form state
//  - on 400/422 the error message NEVER echoes the offending input value
//  - the registered view is driven by the vault's EDGAR slot
//  - clear hits DELETE /api/admin/session/edgar and wipes the vault slot

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { EdgarIdentityCard } from "@/components/edgar-identity-card";
import * as apiModule from "@/lib/api";
import {
  derivePasswordMaterial,
  encryptVault,
  loginUser,
  resetLocalState,
  _internals,
} from "@/lib/user-vault";

const originalFetch = globalThis.fetch;
const TEST_ITERATIONS = 1_000;

// Unlock an empty vault so the card's `mutateVault` can persist. The
// login + vault-upload seams are spied directly; `registerEdgarIdentity`
// / `clearEdgarIdentity` flow through the mocked global fetch so the
// tests can assert on the session-edgar URL + method.
async function unlockVault(): Promise<void> {
  const saltBytes = new Uint8Array(16).fill(0x52);
  const { kek } = await derivePasswordMaterial("pw", saltBytes, TEST_ITERATIONS);
  const blob = await encryptVault(kek, { providers: {}, edgar: null });
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
  vi.spyOn(apiModule, "updateVaultRequest").mockResolvedValue({ updated: true });
  await loginUser("pat", "pw");
}

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  resetLocalState();
  vi.restoreAllMocks();
});

describe("EdgarIdentityCard", () => {
  it("renders the form when no identity is in the vault yet", async () => {
    await unlockVault();
    render(<EdgarIdentityCard />);
    expect(
      screen.getByRole("form", { name: /edgar identity registration/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/full name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
  });

  it("pushes to the session store AND persists into the vault on submit", async () => {
    await unlockVault();
    const fetchMock = vi.fn(async () => {
      return new Response(JSON.stringify({ registered: true }), { status: 201 });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const updateSpy = vi.spyOn(apiModule, "updateVaultRequest");

    const onRegistered = vi.fn();
    render(<EdgarIdentityCard onRegistered={onRegistered} />);

    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/full name/i), "Alice Example");
    await user.type(screen.getByLabelText(/email/i), "alice@example.com");
    await user.click(screen.getByRole("button", { name: /save to vault/i }));

    await waitFor(() => {
      expect(screen.getByText(/saved to your vault/i)).toBeInTheDocument();
    });

    // Session-store push.
    const sessionCall = fetchMock.mock.calls.find((c) =>
      String((c as unknown[])[0]).endsWith("/session/edgar"),
    ) as unknown as [string, RequestInit] | undefined;
    expect(sessionCall).toBeDefined();
    expect(sessionCall![1].method).toBe("POST");
    const body = JSON.parse(sessionCall![1].body as string) as {
      name: string;
      email: string;
    };
    expect(body).toEqual({ name: "Alice Example", email: "alice@example.com" });

    // Vault persistence.
    expect(updateSpy).toHaveBeenCalledTimes(1);
    expect(onRegistered).toHaveBeenCalledTimes(1);
  });

  it("clears local form state after a successful submit", async () => {
    await unlockVault();
    globalThis.fetch = vi.fn(async () => {
      return new Response(JSON.stringify({ registered: true }), { status: 201 });
    }) as unknown as typeof fetch;

    render(<EdgarIdentityCard />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/full name/i), "Bob Tester");
    await user.type(screen.getByLabelText(/email/i), "bob@example.com");
    await user.click(screen.getByRole("button", { name: /save to vault/i }));

    await waitFor(() => {
      expect(screen.getByText(/saved to your vault/i)).toBeInTheDocument();
    });

    // After success the form is gone; values must not linger in the DOM.
    expect(document.body.innerHTML).not.toContain("Bob Tester");
    expect(document.body.innerHTML).not.toContain("bob@example.com");
  });

  it("never echoes the offending value when the backend rejects the input", async () => {
    await unlockVault();
    globalThis.fetch = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          error: "invalid_edgar_identity",
          message: "EDGAR identity failed validation.",
          hint: "name must not contain control characters",
        }),
        { status: 400 },
      );
    }) as unknown as typeof fetch;

    render(<EdgarIdentityCard />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/full name/i), "BadValueXYZ");
    await user.type(screen.getByLabelText(/email/i), "bad@example.com");
    await user.click(screen.getByRole("button", { name: /save to vault/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(/EDGAR identity failed validation/);
    expect(alert).toHaveTextContent(/control characters/);
    expect(alert.textContent ?? "").not.toContain("BadValueXYZ");
  });

  it("does not persist into the vault when the session-store push fails", async () => {
    await unlockVault();
    globalThis.fetch = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          error: "invalid_edgar_identity",
          message: "EDGAR identity failed validation.",
        }),
        { status: 400 },
      );
    }) as unknown as typeof fetch;
    const updateSpy = vi.spyOn(apiModule, "updateVaultRequest");

    render(<EdgarIdentityCard />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/full name/i), "Carol");
    await user.type(screen.getByLabelText(/email/i), "carol@example.com");
    await user.click(screen.getByRole("button", { name: /save to vault/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
    // The session push threw, so the vault write must not have happened.
    expect(updateSpy).not.toHaveBeenCalled();
  });

  it("renders the saved state when the vault already carries an identity", async () => {
    // Unlock a vault that already has an EDGAR identity.
    const saltBytes = new Uint8Array(16).fill(0x53);
    const { kek } = await derivePasswordMaterial("pw", saltBytes, TEST_ITERATIONS);
    const blob = await encryptVault(kek, {
      providers: {},
      edgar: { name: "Dave", email: "dave@example.com" },
    });
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
    vi.spyOn(apiModule, "updateVaultRequest").mockResolvedValue({ updated: true });
    // Re-push to the session store is best-effort via the global fetch.
    globalThis.fetch = vi.fn(async () => {
      return new Response(JSON.stringify({ registered: true }), { status: 201 });
    }) as unknown as typeof fetch;
    await loginUser("pat", "pw");

    render(<EdgarIdentityCard />);
    await waitFor(() => {
      expect(screen.getByText(/saved to your vault/i)).toBeInTheDocument();
    });
    // The name/email must NOT render into the DOM — only the saved badge.
    expect(document.body.innerHTML).not.toContain("dave@example.com");
  });

  it("clear hits DELETE /api/admin/session/edgar and wipes the vault slot", async () => {
    // Vault already carries an identity.
    const saltBytes = new Uint8Array(16).fill(0x54);
    const { kek } = await derivePasswordMaterial("pw", saltBytes, TEST_ITERATIONS);
    const blob = await encryptVault(kek, {
      providers: {},
      edgar: { name: "Eve", email: "eve@example.com" },
    });
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
    vi.spyOn(apiModule, "updateVaultRequest").mockResolvedValue({ updated: true });
    const fetchMock = vi.fn(async () => {
      return new Response(JSON.stringify({ cleared: true }), { status: 200 });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    await loginUser("pat", "pw");

    render(<EdgarIdentityCard />);
    await waitFor(() => {
      expect(screen.getByText(/saved to your vault/i)).toBeInTheDocument();
    });

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /clear identity/i }));

    await waitFor(() => {
      expect(
        screen.getByRole("form", { name: /edgar identity registration/i }),
      ).toBeInTheDocument();
    });
    const deleteCall = fetchMock.mock.calls.find(
      (c) =>
        String((c as unknown[])[0]).endsWith("/session/edgar") &&
        ((c as unknown[])[1] as RequestInit)?.method === "DELETE",
    );
    expect(deleteCall).toBeDefined();
  });
});
