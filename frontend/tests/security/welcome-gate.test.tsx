// WelcomeGate UI gate.
//
// Asserts:
//  - children are NOT rendered until the GET probe returns authenticated=true
//  - submitting the form POSTs JSON with `credentials: same-origin`
//  - the form fields are masked (type=password) so over-the-shoulder /
//    screen-share leaks are minimised
//  - the keys never end up in the rendered DOM after a successful submit
//  - autocomplete is disabled on the key inputs

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { WelcomeGate } from "@/components/welcome-gate";

const originalFetch = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

function mockProbe(authenticated: boolean): void {
  (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mockImplementation(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/api/admin/session") && (init?.method ?? "GET") === "GET") {
        return new Response(JSON.stringify({ authenticated }), { status: 200 });
      }
      return new Response("", { status: 200 });
    },
  );
}

describe("WelcomeGate", () => {
  it("does not render children until authenticated", async () => {
    mockProbe(false);
    render(
      <WelcomeGate>
        <div data-testid="protected">protected content</div>
      </WelcomeGate>,
    );
    await waitFor(() => {
      expect(screen.getByRole("form", { name: /operator sign-in/i })).toBeInTheDocument();
    });
    expect(screen.queryByTestId("protected")).not.toBeInTheDocument();
  });

  it("renders children when the probe reports authenticated=true", async () => {
    mockProbe(true);
    render(
      <WelcomeGate>
        <div data-testid="protected">protected content</div>
      </WelcomeGate>,
    );
    await waitFor(() => {
      expect(screen.getByTestId("protected")).toBeInTheDocument();
    });
    expect(screen.queryByRole("form")).not.toBeInTheDocument();
  });

  it("masks both key inputs and disables autocomplete", async () => {
    mockProbe(false);
    render(
      <WelcomeGate>
        <div />
      </WelcomeGate>,
    );
    const apiInput = await screen.findByLabelText(/API key/i);
    const adminInput = await screen.findByLabelText(/Admin key/i);
    expect(apiInput).toHaveAttribute("type", "password");
    expect(adminInput).toHaveAttribute("type", "password");
    expect(apiInput).toHaveAttribute("autocomplete", "off");
    expect(adminInput).toHaveAttribute("autocomplete", "off");
  });

  it("POSTs JSON with same-origin credentials and clears the form on success", async () => {
    const fetchMock = globalThis.fetch as unknown as ReturnType<typeof vi.fn>;
    let probeCount = 0;
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/api/admin/session") && (init?.method ?? "GET") === "GET") {
        probeCount += 1;
        return new Response(
          JSON.stringify({ authenticated: probeCount > 1 }),
          { status: 200 },
        );
      }
      if (url.endsWith("/api/admin/session") && init?.method === "POST") {
        return new Response(JSON.stringify({ ok: true }), { status: 200 });
      }
      return new Response("", { status: 200 });
    });

    render(
      <WelcomeGate>
        <div data-testid="protected">protected</div>
      </WelcomeGate>,
    );

    const user = userEvent.setup();
    const apiInput = await screen.findByLabelText(/API key/i);
    const adminInput = await screen.findByLabelText(/Admin key/i);
    await user.type(apiInput, "live-api-key");
    await user.type(adminInput, "live-admin-key");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(screen.getByTestId("protected")).toBeInTheDocument();
    });

    // Find the POST call.
    const postCall = fetchMock.mock.calls.find(
      (call) =>
        typeof call[0] === "string" &&
        call[0].includes("/api/admin/session") &&
        (call[1] as RequestInit | undefined)?.method === "POST",
    );
    expect(postCall).toBeDefined();
    const init = postCall?.[1] as RequestInit;
    expect(init.credentials).toBe("same-origin");
    expect(init.headers).toMatchObject({ "Content-Type": "application/json" });
    const body = JSON.parse(init.body as string) as { api_key: string; admin_key: string };
    expect(body.api_key).toBe("live-api-key");
    expect(body.admin_key).toBe("live-admin-key");

    // After a successful submit, neither key should appear anywhere in the DOM.
    expect(document.body.innerHTML).not.toContain("live-api-key");
    expect(document.body.innerHTML).not.toContain("live-admin-key");
  });

  it("shows an error message when the POST fails", async () => {
    const fetchMock = globalThis.fetch as unknown as ReturnType<typeof vi.fn>;
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/api/admin/session") && (init?.method ?? "GET") === "GET") {
        return new Response(JSON.stringify({ authenticated: false }), { status: 200 });
      }
      if (url.endsWith("/api/admin/session") && init?.method === "POST") {
        return new Response(
          JSON.stringify({
            error: "invalid_api_key",
            message: "API key was rejected by the backend",
          }),
          { status: 401 },
        );
      }
      return new Response("", { status: 200 });
    });

    render(
      <WelcomeGate>
        <div />
      </WelcomeGate>,
    );

    const user = userEvent.setup();
    await user.type(await screen.findByLabelText(/API key/i), "bad");
    await user.type(await screen.findByLabelText(/Admin key/i), "bad");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(/API key was rejected/);
    });
  });
});
