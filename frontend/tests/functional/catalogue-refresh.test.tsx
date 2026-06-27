// Model-catalogue refresh card (functional).
//
// Component-level coverage of `CatalogueRefreshCard` — the dashboard's
// admin action that triggers `POST /api/providers/catalogue/refresh`
// through the admin proxy. Drives the card's own state machine directly
// (no page mount):
//
//   - a click POSTs to the proxy path and renders the content-free
//     success summary (source + provider/model counts);
//   - the request carries METHOD POST and no request body (source / URL /
//     overlay path are server config, never client input);
//   - a 502 fail-closed verdict surfaces the backend's canned message in
//     an alert, not a crash;
//   - the button is inert while a refresh is in flight.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { CatalogueRefreshCard } from "@/components/catalogue-refresh-card";

const originalFetch = globalThis.fetch;

const REFRESH_PATH = "providers/catalogue/refresh";

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("CatalogueRefreshCard", () => {
  it("POSTs to the proxy and renders the content-free summary", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes(REFRESH_PATH)) {
        return new Response(
          JSON.stringify({
            source: "models_dev",
            source_url: "https://models.dev/api.json",
            provider_count: 7,
            model_count: 42,
          }),
          { status: 200 },
        );
      }
      return new Response("{}", { status: 200 });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<CatalogueRefreshCard />);
    await user.click(screen.getByRole("button", { name: /refresh catalogue/i }));

    const status = await screen.findByRole("status");
    expect(status).toHaveTextContent("models_dev");
    expect(status).toHaveTextContent("7 providers");
    expect(status).toHaveTextContent("42 models");

    // The request is a bare POST through the admin proxy with no body.
    const call = fetchMock.mock.calls.find(([input]) => {
      const url = typeof input === "string" ? input : String(input);
      return url.includes(REFRESH_PATH);
    });
    expect(call).toBeDefined();
    const init = call?.[1];
    expect(init?.method).toBe("POST");
    expect(init?.body ?? null).toBeNull();
  });

  it("surfaces a fail-closed 502 verdict as an alert", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          error: "catalogue_refresh_failed",
          message: "Catalogue refresh fetch failed (ConnectTimeout).",
        }),
        { status: 502 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<CatalogueRefreshCard />);
    await user.click(screen.getByRole("button", { name: /refresh catalogue/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/fetch failed/i);
    // No success summary on a failed refresh.
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("disables the button while a refresh is in flight", async () => {
    const user = userEvent.setup();
    let resolve!: (value: Response) => void;
    const pending = new Promise<Response>((r) => {
      resolve = r;
    });
    globalThis.fetch = vi.fn(() => pending) as unknown as typeof fetch;

    render(<CatalogueRefreshCard />);
    const button = screen.getByRole("button", { name: /refresh catalogue/i });
    await user.click(button);

    await waitFor(() => expect(button).toBeDisabled());

    resolve(
      new Response(
        JSON.stringify({
          source: "models_dev",
          source_url: "https://models.dev/api.json",
          provider_count: 1,
          model_count: 1,
        }),
        { status: 200 },
      ),
    );
    await waitFor(() => expect(button).not.toBeDisabled());
  });
});
