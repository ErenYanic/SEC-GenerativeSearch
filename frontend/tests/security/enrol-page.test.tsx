// Enrolment page.
//
// What this test file pins:
//
//   1. The "your password is the only key" warning renders ABOVE the
//      password field (DOM order is load-bearing — a user must read it
//      before they choose a password).
//   2. Password confirmation mismatch / too-short input is rejected
//      client-side WITHOUT a network call.
//   3. A successful enrolment calls `enrolUser(token, password)` and
//      the PASSWORD never appears in the captured `enrolUserRequest`
//      payload.
//   4. Backend envelopes (401 token invalid, 409 already enrolled)
//      surface verbatim without echoing the token / password.
//   5. A missing token short-circuits with an explanatory error.

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// `next/navigation` — useSearchParams returns the token; the WelcomeGate
// transitively imports nothing from navigation, but the enrol page does.
let mockToken = "valid-token";
vi.mock("next/navigation", async () => {
  const actual = await vi.importActual<Record<string, unknown>>(
    "next/navigation",
  );
  return {
    ...actual,
    useSearchParams: () => new URLSearchParams(`token=${mockToken}`),
  };
});

import EnrolPage from "@/app/enrol/page";
import * as apiModule from "@/lib/api";

const originalFetch = globalThis.fetch;

beforeEach(() => {
  mockToken = "valid-token";
  // WelcomeGate probes GET /api/admin/session on mount; report
  // authenticated so the enrol form renders.
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("/api/admin/session")) {
      return new Response(JSON.stringify({ authenticated: true }), {
        status: 200,
      });
    }
    return new Response("{}", { status: 200 });
  }) as unknown as typeof fetch;
});

afterEach(() => {
  cleanup();
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

async function renderEnrol(): Promise<void> {
  render(<EnrolPage />);
  // Wait for WelcomeGate to resolve its probe + render the form.
  await waitFor(() => {
    expect(screen.getByLabelText("Complete enrolment")).toBeInTheDocument();
  });
}

describe("EnrolPage — warning placement", () => {
  it("renders the password-loss warning ABOVE the password field", async () => {
    await renderEnrol();
    const warning = screen.getByTestId("password-loss-warning");
    const passwordField = screen.getByLabelText("Password");
    // `compareDocumentPosition` returns DOCUMENT_POSITION_FOLLOWING (4)
    // when `passwordField` follows `warning` in document order.
    const position = warning.compareDocumentPosition(passwordField);
    expect(position & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("warns that the password is unrecoverable", async () => {
    await renderEnrol();
    expect(screen.getByTestId("password-loss-warning")).toHaveTextContent(
      /only key/i,
    );
  });
});

describe("EnrolPage — client-side validation", () => {
  it("rejects a password shorter than the minimum without a network call", async () => {
    const enrolSpy = vi.spyOn(apiModule, "enrolUserRequest");
    await renderEnrol();
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "short" },
    });
    fireEvent.change(screen.getByLabelText("Confirm password"), {
      target: { value: "short" },
    });
    fireEvent.click(screen.getByRole("button", { name: /complete enrolment/i }));
    await waitFor(() => {
      expect(screen.getByText(/at least 12 characters/i)).toBeInTheDocument();
    });
    expect(enrolSpy).not.toHaveBeenCalled();
  });

  it("rejects mismatched confirmation without a network call", async () => {
    const enrolSpy = vi.spyOn(apiModule, "enrolUserRequest");
    await renderEnrol();
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "correct-horse-battery" },
    });
    fireEvent.change(screen.getByLabelText("Confirm password"), {
      target: { value: "correct-horse-stapler" },
    });
    fireEvent.click(screen.getByRole("button", { name: /complete enrolment/i }));
    await waitFor(() => {
      expect(screen.getByText(/do not match/i)).toBeInTheDocument();
    });
    expect(enrolSpy).not.toHaveBeenCalled();
  });

  it("rejects a missing token", async () => {
    mockToken = "";
    const enrolSpy = vi.spyOn(apiModule, "enrolUserRequest");
    await renderEnrol();
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "correct-horse-battery" },
    });
    fireEvent.change(screen.getByLabelText("Confirm password"), {
      target: { value: "correct-horse-battery" },
    });
    fireEvent.click(screen.getByRole("button", { name: /complete enrolment/i }));
    await waitFor(() => {
      expect(screen.getByText(/missing its token/i)).toBeInTheDocument();
    });
    expect(enrolSpy).not.toHaveBeenCalled();
  });
});

describe("EnrolPage — submission", () => {
  it("derives client-side + posts; the password never crosses the wire", async () => {
    const enrolSpy = vi
      .spyOn(apiModule, "enrolUserRequest")
      .mockResolvedValue({ enrolled: true, user_id: 9, username: "newcomer" });
    await renderEnrol();
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "correct-horse-battery" },
    });
    fireEvent.change(screen.getByLabelText("Confirm password"), {
      target: { value: "correct-horse-battery" },
    });
    fireEvent.click(screen.getByRole("button", { name: /complete enrolment/i }));

    await waitFor(() => {
      expect(screen.getByLabelText("Enrolment complete")).toBeInTheDocument();
    });
    expect(enrolSpy).toHaveBeenCalledTimes(1);
    const sent = enrolSpy.mock.calls[0]?.[0];
    expect(sent?.token).toBe("valid-token");
    expect(sent?.kdf_algo).toBe("pbkdf2-sha256");
    expect(sent?.salt_m.length).toBe(22);
    expect(sent?.auth_proof.length).toBe(43);
    expect(sent?.vault_iv.length).toBe(16);
    expect(JSON.stringify(enrolSpy.mock.calls)).not.toContain(
      "correct-horse-battery",
    );
  });

  it("surfaces a 409 already-enrolled envelope without echoing the token", async () => {
    vi.spyOn(apiModule, "enrolUserRequest").mockRejectedValue(
      new apiModule.ApiError(
        409,
        "enrolment_already_completed",
        "A user with this username has already enrolled.",
      ),
    );
    await renderEnrol();
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "correct-horse-battery" },
    });
    fireEvent.change(screen.getByLabelText("Confirm password"), {
      target: { value: "correct-horse-battery" },
    });
    fireEvent.click(screen.getByRole("button", { name: /complete enrolment/i }));
    // The password-loss warning is itself role="alert", so we wait for
    // the specific error text rather than any alert node.
    const errorAlert = await screen.findByText(/already enrolled/i);
    expect(errorAlert.textContent).not.toContain("valid-token");
  });

  it("surfaces a 401 invalid-token envelope", async () => {
    vi.spyOn(apiModule, "enrolUserRequest").mockRejectedValue(
      new apiModule.ApiError(
        401,
        "enrolment_token_invalid",
        "Enrolment token is invalid or expired.",
      ),
    );
    await renderEnrol();
    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "correct-horse-battery" },
    });
    fireEvent.change(screen.getByLabelText("Confirm password"), {
      target: { value: "correct-horse-battery" },
    });
    fireEvent.click(screen.getByRole("button", { name: /complete enrolment/i }));
    expect(await screen.findByText(/invalid or expired/i)).toBeInTheDocument();
  });
});
