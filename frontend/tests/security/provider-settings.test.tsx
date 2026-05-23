// Provider Settings page + ProviderKeyRow.
//
// Asserts:
//   - the page loads /api/admin/providers/ and renders each entry
//   - "Save" writes to sessionStorage and clears the password input
//   - "Validate" POSTs to /api/admin/providers/validate, body carries
//     the key but the UI never echoes it back
//   - "Remove" wipes the stored key
//   - "Clear all keys" wipes every namespaced key
//   - the key value is never rendered into the DOM at any point after
//     save — only the masked tail

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

vi.mock("next/navigation", async () => {
  const actual = await vi.importActual<Record<string, unknown>>(
    "next/navigation",
  );
  return {
    ...actual,
    usePathname: () => "/providers",
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      back: vi.fn(),
      refresh: vi.fn(),
      prefetch: vi.fn(),
    }),
  };
});

import ProviderSettingsPage from "@/app/(app)/providers/page";
import { clearProviderKeys, loadProviderKeys } from "@/lib/provider-keys";

const originalFetch = globalThis.fetch;

const CATALOGUE = {
  providers: [
    { name: "openai", surface: "llm", supports_upstream_routing: false },
    { name: "anthropic", surface: "llm", supports_upstream_routing: false },
  ],
  total: 2,
};

beforeEach(() => {
  window.sessionStorage.clear();
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
  clearProviderKeys();
});

function mockCatalogue(): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url === "/api/admin/providers/") {
      return new Response(JSON.stringify(CATALOGUE), { status: 200 });
    }
    return new Response("{}", { status: 200 });
  });
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

describe("ProviderSettingsPage", () => {
  it("loads the catalogue and renders one row per provider", async () => {
    mockCatalogue();
    render(<ProviderSettingsPage />);

    await screen.findByText("openai");
    expect(screen.getByText("anthropic")).toBeInTheDocument();
    // No stored keys yet — both rows say "No key".
    const statuses = screen.getAllByTestId("provider-key-status");
    expect(statuses.map((node) => node.textContent)).toEqual([
      "No key",
      "No key",
    ]);
  });

  it("Save writes to sessionStorage and clears the input field", async () => {
    mockCatalogue();
    render(<ProviderSettingsPage />);

    await screen.findByText("openai");

    const user = userEvent.setup();
    // First "Add key" button — the openai row.
    await user.click(screen.getAllByRole("button", { name: /add key/i })[0]!);
    const input = screen.getByLabelText(/openai API key/i);
    await user.type(input, "sk-PASSWORDSECRET12345"); // pragma: allowlist secret
    await user.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => {
      expect(loadProviderKeys()).toEqual({
        openai: "sk-PASSWORDSECRET12345",
      });
    });
    // Input is gone (edit mode closed). Masked tail surfaces in the status.
    expect(screen.queryByLabelText(/openai API key/i)).not.toBeInTheDocument();
    const statuses = screen.getAllByTestId("provider-key-status");
    expect(statuses[0]?.textContent).toMatch(/Stored/);
    expect(statuses[0]?.textContent).toMatch(/sk-P/);
    expect(statuses[0]?.textContent).toMatch(/2345/);
    // The cleartext key is never rendered in full.
    expect(document.body.textContent ?? "").not.toContain(
      "sk-PASSWORDSECRET12345",
    );
  });

  it("Validate POSTs to /providers/validate with key in body, surfaces verdict", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/providers/") {
        return new Response(JSON.stringify(CATALOGUE), { status: 200 });
      }
      if (url === "/api/admin/providers/validate") {
        return new Response(
          JSON.stringify({
            valid: true,
            provider: "openai",
            surface: "llm",
          }),
          { status: 200 },
        );
      }
      return new Response("{}", { status: 200 });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<ProviderSettingsPage />);
    await screen.findByText("openai");
    const user = userEvent.setup();

    await user.click(screen.getAllByRole("button", { name: /add key/i })[0]!);
    await user.type(
      screen.getByLabelText(/openai API key/i),
      "sk-VALIDATEME12345", // pragma: allowlist secret
    );
    await user.click(screen.getByRole("button", { name: /^save$/i }));

    await user.click(
      screen.getAllByRole("button", { name: /validate/i })[0]!,
    );

    await waitFor(() => {
      expect(screen.getByText(/Validated/)).toBeInTheDocument();
    });
    const validateCall = fetchMock.mock.calls.find((call) =>
      String((call as unknown[])[0]).endsWith("/providers/validate"),
    ) as unknown as [string, RequestInit] | undefined;
    expect(validateCall).toBeDefined();
    const init = validateCall![1];
    expect(JSON.parse(init.body as string)).toMatchObject({
      provider: "openai",
      api_key: "sk-VALIDATEME12345",
      surface: "llm",
    });
  });

  it("surfaces a 502 validation error without echoing the key", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/providers/") {
        return new Response(JSON.stringify(CATALOGUE), { status: 200 });
      }
      return new Response(
        JSON.stringify({
          error: "provider_error",
          message: "The upstream provider returned an error.",
          hint: "Inspect the audit log; do not rotate the key on a non-auth error.",
        }),
        { status: 502 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<ProviderSettingsPage />);
    await screen.findByText("openai");
    const user = userEvent.setup();

    await user.click(screen.getAllByRole("button", { name: /add key/i })[0]!);
    await user.type(
      screen.getByLabelText(/openai API key/i),
      "sk-DOOMEDVALUE12345", // pragma: allowlist secret
    );
    await user.click(screen.getByRole("button", { name: /^save$/i }));
    await user.click(screen.getAllByRole("button", { name: /validate/i })[0]!);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/upstream provider returned an error/);
    expect(alert.textContent ?? "").not.toContain("sk-DOOMEDVALUE12345");
  });

  it("Remove wipes the stored key for that row", async () => {
    mockCatalogue();
    render(<ProviderSettingsPage />);
    await screen.findByText("openai");

    const user = userEvent.setup();
    await user.click(screen.getAllByRole("button", { name: /add key/i })[0]!);
    await user.type(
      screen.getByLabelText(/openai API key/i),
      "sk-EPHEMERAL00001", // pragma: allowlist secret
    );
    await user.click(screen.getByRole("button", { name: /^save$/i }));
    await waitFor(() => {
      expect(loadProviderKeys()).toEqual({ openai: "sk-EPHEMERAL00001" });
    });

    await user.click(screen.getByRole("button", { name: /remove/i }));
    await waitFor(() => {
      expect(loadProviderKeys()).toEqual({});
    });
  });

  it("Clear all keys nukes every stored key", async () => {
    mockCatalogue();
    render(<ProviderSettingsPage />);
    await screen.findByText("openai");

    const user = userEvent.setup();
    // Save openai.
    await user.click(screen.getAllByRole("button", { name: /add key/i })[0]!);
    await user.type(
      screen.getByLabelText(/openai API key/i),
      "sk-aaaaaaaaaaaa", // pragma: allowlist secret
    );
    await user.click(screen.getByRole("button", { name: /^save$/i }));
    // Save anthropic.
    await user.click(screen.getAllByRole("button", { name: /add key/i })[0]!);
    await user.type(
      screen.getByLabelText(/anthropic API key/i),
      "sk-ant-bbbbbbbbbbbb", // pragma: allowlist secret
    );
    await user.click(screen.getByRole("button", { name: /^save$/i }));
    await waitFor(() => {
      expect(Object.keys(loadProviderKeys())).toHaveLength(2);
    });

    await user.click(screen.getByRole("button", { name: /clear all keys/i }));
    await waitFor(() => {
      expect(loadProviderKeys()).toEqual({});
    });
  });

  it("surfaces a backend catalogue error without crashing", async () => {
    globalThis.fetch = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            error: "backend_unreachable",
            message: "backend did not respond",
          }),
          { status: 502 },
        ),
    ) as unknown as typeof fetch;

    render(<ProviderSettingsPage />);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/backend did not respond/);
  });
});
