// EDGAR identity card UI.
//
// Asserts:
//  - submitting calls POST /api/admin/session/edgar with JSON body
//  - on success, the form swaps to "registered" and clears local state
//  - on 400/422 the error message NEVER echoes the offending input value
//  - clear button hits DELETE /api/admin/session/edgar

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { EdgarIdentityCard } from "@/components/edgar-identity-card";

const originalFetch = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("EdgarIdentityCard", () => {
  it("renders the form when no identity is registered yet", () => {
    render(<EdgarIdentityCard />);
    expect(
      screen.getByRole("form", { name: /edgar identity registration/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/full name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
  });

  it("submits the form and swaps to the registered state on success", async () => {
    const fetchMock = vi.fn(async () => {
      return new Response(JSON.stringify({ registered: true }), {
        status: 201,
      });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const onRegistered = vi.fn();
    render(<EdgarIdentityCard onRegistered={onRegistered} />);

    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/full name/i), "Alice Example");
    await user.type(screen.getByLabelText(/email/i), "alice@example.com");
    await user.click(screen.getByRole("button", { name: /register/i }));

    await waitFor(() => {
      expect(
        screen.getByText(/EDGAR identity registered/i),
      ).toBeInTheDocument();
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe("/api/admin/session/edgar");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body as string) as {
      name: string;
      email: string;
    };
    expect(body.name).toBe("Alice Example");
    expect(body.email).toBe("alice@example.com");
    expect(onRegistered).toHaveBeenCalledTimes(1);
  });

  it("clears local form state after a successful submit", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response(JSON.stringify({ registered: true }), {
        status: 201,
      });
    }) as unknown as typeof fetch;

    render(<EdgarIdentityCard />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/full name/i), "Bob Tester");
    await user.type(screen.getByLabelText(/email/i), "bob@example.com");
    await user.click(screen.getByRole("button", { name: /register/i }));

    await waitFor(() => {
      expect(
        screen.getByText(/EDGAR identity registered/i),
      ).toBeInTheDocument();
    });

    // After success, the form is unmounted; values must not linger in
    // the DOM. (Either the registered state is shown OR the inputs are
    // empty — the registered state hides the form entirely.)
    expect(document.body.innerHTML).not.toContain("Bob Tester");
    expect(document.body.innerHTML).not.toContain("bob@example.com");
  });

  it("never echoes the offending value when the backend rejects the input", async () => {
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
    // Submit a deliberately ugly value; the backend rejects with a
    // canned message and the UI must NOT render the value back.
    await user.type(screen.getByLabelText(/full name/i), "BadValueXYZ");
    await user.type(screen.getByLabelText(/email/i), "bad@example.com");
    await user.click(screen.getByRole("button", { name: /register/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(/EDGAR identity failed validation/);
    expect(alert).toHaveTextContent(/control characters/);
    // The offending input value MUST NOT appear in the alert.
    expect(alert.textContent ?? "").not.toContain("BadValueXYZ");
  });

  it("clear button hits DELETE /api/admin/session/edgar", async () => {
    let call = 0;
    const fetchMock = vi.fn(async () => {
      call += 1;
      if (call === 1) {
        return new Response(JSON.stringify({ registered: true }), {
          status: 201,
        });
      }
      return new Response(JSON.stringify({ cleared: true }), { status: 200 });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<EdgarIdentityCard />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/full name/i), "Eve");
    await user.type(screen.getByLabelText(/email/i), "eve@example.com");
    await user.click(screen.getByRole("button", { name: /register/i }));

    await waitFor(() => {
      expect(
        screen.getByText(/EDGAR identity registered/i),
      ).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /clear identity/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });
    const [, init] = fetchMock.mock.calls[1] as unknown as [
      string,
      RequestInit,
    ];
    expect(init.method).toBe("DELETE");
  });
});
