// Provider-key workflow (functional).
//
// Complements the page-level `tests/security/provider-settings.test.tsx`
// (save / validate-ok / remove / clear-all) by driving the `ProviderKeyRow`
// component directly through the operator-facing edit interactions that
// the page suite does not exercise:
//
//   - the "Replace" affordance + masked-tail status for a row that
//     already holds a key;
//   - a rejected validation verdict (`valid: false`) surfacing the
//     amber "Rejected" badge — distinct from a transient `ApiError`;
//   - the Validate button being inert until a key is stored;
//   - the masked tail fully redacting a short key (`mask_secret` parity);
//   - the edit-then-cancel path discarding the draft without a write;
//   - a save against a locked vault surfacing the error without losing
//     the entered draft.
//
// These are component-level (no page mount, no vault unlock), so they
// isolate the row's own state machine.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ProviderKeyRow } from "@/components/provider-key-row";
import { resetLocalState } from "@/lib/user-vault";
import type { ProviderInfo } from "@/lib/api-types";

const originalFetch = globalThis.fetch;

const OPENAI: ProviderInfo = {
  name: "openai",
  surface: "llm",
  supports_upstream_routing: false,
};

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  // The row's Save path writes through the user vault; reset it so each
  // test starts with a locked vault (no KEK, no cleartext).
  resetLocalState();
  vi.restoreAllMocks();
});

describe("ProviderKeyRow — stored-key affordances", () => {
  it("shows Replace + Remove and a masked tail when a key is stored", () => {
    render(<ProviderKeyRow provider={OPENAI} storedKey="sk-abcd12345678" />);

    // A stored row offers Replace (not "Add key") and Remove.
    expect(
      screen.getByRole("button", { name: /replace/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /remove/i })).toBeInTheDocument();

    // The status chip shows only the first4…last4 masked tail, never the
    // full key.
    const status = screen.getByTestId("provider-key-status");
    expect(status.textContent).toMatch(/Stored/);
    expect(status.textContent).toContain("sk-a");
    expect(status.textContent).toContain("5678");
    expect(document.body.textContent ?? "").not.toContain("sk-abcd12345678");
  });

  it("fully redacts a key shorter than eight characters", () => {
    render(<ProviderKeyRow provider={OPENAI} storedKey="short" />);
    const status = screen.getByTestId("provider-key-status");
    // mask_secret parity: <8 chars → eight bullets, never the cleartext.
    expect(status.textContent).toContain("••••••••");
    expect(document.body.textContent ?? "").not.toContain("short");
  });

  it("disables Validate until a key is stored", () => {
    render(<ProviderKeyRow provider={OPENAI} storedKey={undefined} />);
    expect(screen.getByRole("button", { name: /validate/i })).toBeDisabled();
    // A keyless row offers "Add key", not Replace/Remove.
    expect(screen.getByRole("button", { name: /add key/i })).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /remove/i }),
    ).not.toBeInTheDocument();
  });
});

describe("ProviderKeyRow — validation verdict", () => {
  it("renders a Rejected badge when the backend reports valid:false", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/providers/validate") {
        return new Response(
          JSON.stringify({ valid: false, provider: "openai", surface: "llm" }),
          { status: 200 },
        );
      }
      return new Response("{}", { status: 200 });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<ProviderKeyRow provider={OPENAI} storedKey="sk-storedkey12345" />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /validate/i }));

    await waitFor(() => {
      expect(screen.getByText(/^Rejected$/)).toBeInTheDocument();
    });
    // A valid:false verdict is NOT an error — no alert is raised.
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    // The candidate key rode the JSON body (never the URL) under the
    // stored value.
    const call = fetchMock.mock.calls.find(
      (c) => (c[0] as string) === "/api/admin/providers/validate",
    ) as unknown as [string, RequestInit];
    expect(call[0]).not.toContain("sk-storedkey12345");
    expect(JSON.parse(call[1].body as string)).toMatchObject({
      provider: "openai",
      api_key: "sk-storedkey12345", // pragma: allowlist secret
      surface: "llm",
    });
  });
});

describe("ProviderKeyRow — edit interactions", () => {
  it("discards the draft when the edit form is cancelled (no write)", async () => {
    render(<ProviderKeyRow provider={OPENAI} storedKey={undefined} />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /add key/i }));
    const input = screen.getByLabelText(/openai API key/i);
    await user.type(input, "sk-DISCARDME12345");

    // The in-form Cancel button (not the outer remove) tears down edit
    // mode without persisting.
    await user.click(screen.getByRole("button", { name: /^cancel$/i }));
    expect(screen.queryByLabelText(/openai API key/i)).not.toBeInTheDocument();
    // Still a keyless row — nothing was saved.
    expect(screen.getByRole("button", { name: /add key/i })).toBeInTheDocument();
  });

  it("surfaces a save failure against a locked vault without losing the draft", async () => {
    render(<ProviderKeyRow provider={OPENAI} storedKey={undefined} />);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: /add key/i }));
    await user.type(
      screen.getByLabelText(/openai API key/i),
      "sk-LOCKEDVAULT12345",
    );
    // The vault is locked (no login) → setProviderKey rejects with
    // VaultLockedError; the row surfaces it as an alert and keeps the
    // edit form open so the operator can retry after re-authenticating.
    await user.click(screen.getByRole("button", { name: /^save$/i }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent ?? "").not.toBe("");
    expect(screen.getByLabelText(/openai API key/i)).toBeInTheDocument();
  });
});
