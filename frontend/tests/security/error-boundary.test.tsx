// App-group error boundary (`src/app/(app)/error.tsx`).
//
// Pins:
//   - the fallback renders with role="alert" so screen readers announce
//     the failure;
//   - the error.message is NEVER rendered into the visible DOM (the
//     "errors never echo input" contract applies to upstream errors
//     too — `fetch` failure messages can carry server-internal paths);
//   - clicking Reset invokes the boundary's `reset` callback;
//   - the boundary surfaces the safe `error.name` + `digest` for
//     operator triage (those are NOT user-controlled).

import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import AppGroupError from "@/app/(app)/error";

describe("(app)/error.tsx — app-group error boundary", () => {
  it("renders with role=alert + assertive aria-live so AT announces the failure", () => {
    const reset = vi.fn();
    const err = Object.assign(new Error("oh no"), { name: "TypeError" });
    render(<AppGroupError error={err} reset={reset} />);
    const alert = screen.getByRole("alert");
    expect(alert).toBeInTheDocument();
    expect(alert.getAttribute("aria-live")).toBe("assertive");
  });

  it("does NOT render the error.message verbatim into the visible DOM", () => {
    const reset = vi.fn();
    const sentinel = "USER-SUPPLIED-SECRET-IN-UPSTREAM-FETCH-MESSAGE";
    const err = Object.assign(new Error(sentinel), { name: "ApiError" });
    render(<AppGroupError error={err} reset={reset} />);
    // The sentinel string MUST NOT appear anywhere on screen.
    expect(screen.queryByText(new RegExp(sentinel))).toBeNull();
    // The safe error.name DOES render for triage.
    expect(screen.getByText(/ApiError/)).toBeInTheDocument();
  });

  it("renders the digest (when present) — for operator triage in the audit log", () => {
    const reset = vi.fn();
    const err = Object.assign(new Error("masked"), {
      name: "Error",
      digest: "abc123",
    });
    render(<AppGroupError error={err} reset={reset} />);
    expect(screen.getByText(/abc123/)).toBeInTheDocument();
  });

  it("invokes reset() when the user clicks Reset", async () => {
    const reset = vi.fn();
    const err = Object.assign(new Error("masked"), { name: "Error" });
    render(<AppGroupError error={err} reset={reset} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /reset/i }));
    expect(reset).toHaveBeenCalledTimes(1);
  });
});
