// ARIA / loading-state discipline across the (app) pages.
//
// Cross-cutting checks that the loading + streaming surfaces follow the
// same accessibility contract:
//
//   - Dashboard + Filings render the shared `Skeleton` (`role="status"`
//     + `aria-busy`) on initial load, not a bare text node.
//   - Ingest's task-progress card carries `aria-live="polite"` so screen
//     readers track step / status changes during polling.
//   - The ingest status badge carries an `aria-label` that names the
//     status (because the visual badge alone is colour-coded only).
//
// We do NOT re-test the page's happy-path here — those live in
// `pages.test.tsx`; this file is the ARIA / discipline lock.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("next/navigation", async () => {
  const actual = await vi.importActual<Record<string, unknown>>(
    "next/navigation",
  );
  return {
    ...actual,
    usePathname: () => "/dashboard",
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      back: vi.fn(),
      refresh: vi.fn(),
      prefetch: vi.fn(),
    }),
  };
});

import DashboardPage from "@/app/(app)/dashboard/page";
import FilingsPage from "@/app/(app)/filings/page";
import IngestPage from "@/app/(app)/ingest/page";

const originalFetch = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("Dashboard — loading skeleton", () => {
  it("renders a Skeleton placeholder while the status fetch is pending", () => {
    // A fetch that never resolves leaves the page in the loading state
    // long enough for us to assert the placeholder.
    globalThis.fetch = vi.fn(() => new Promise(() => undefined)) as unknown as typeof fetch;
    render(<DashboardPage />);
    const region = screen.getByRole("status");
    expect(region.getAttribute("aria-busy")).toBe("true");
    expect(region.getAttribute("aria-live")).toBe("polite");
    // The Skeleton's accessible label.
    expect(screen.getByText("Loading deployment status…")).toBeInTheDocument();
  });
});

describe("Filings — loading skeleton", () => {
  it("renders a Skeleton placeholder while the list fetch is pending", () => {
    globalThis.fetch = vi.fn(() => new Promise(() => undefined)) as unknown as typeof fetch;
    render(<FilingsPage />);
    expect(screen.getByText("Loading filings…")).toBeInTheDocument();
  });
});

describe("Ingest — task-progress ARIA contract", () => {
  it("the task progress card is wrapped in an aria-live=polite region", async () => {
    // Submit returns a task id; the polling effect then keeps querying
    // until terminal — we cut the test short by returning a 'completed'
    // status on the very first poll so the card renders deterministically.
    let pollCalls = 0;
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/ingest/add") || url.includes("/ingest/batch")) {
        return new Response(
          JSON.stringify({
            task_id: "task-aria-123",
            status: "running",
            websocket_url: "ws://x",
          }),
          { status: 200 },
        );
      }
      if (url.includes("/ingest/tasks/")) {
        pollCalls += 1;
        return new Response(
          JSON.stringify({
            task_id: "task-aria-123",
            status: "completed",
            tickers: ["AAPL"],
            form_types: ["10-K"],
            progress: {
              current_ticker: null,
              current_form_type: null,
              step_label: "finished",
              step_index: 5,
              step_total: 5,
              filings_done: 1,
              filings_total: 1,
              filings_skipped: 0,
              filings_failed: 0,
            },
            results: [],
            error: null,
            started_at: null,
            completed_at: null,
          }),
          { status: 200 },
        );
      }
      // Fallback for any unmatched call (defensive — surface as a
      // failing assertion instead of throwing into the await).
      return new Response("{}", { status: 200 });
    }) as unknown as typeof fetch;

    render(<IngestPage />);
    const { default: userEvent } = await import("@testing-library/user-event");
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/tickers/i), "AAPL");
    await user.click(screen.getByRole("button", { name: /submit ingest/i }));

    // Wait for the submit fetch to land — without this the assertion
    // races React's post-submit setState pass.
    await waitFor(() => {
      const callCount = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls
        .length;
      expect(callCount).toBeGreaterThan(0);
    });

    // The TaskProgressCard renders after setTask. Find the heading by
    // text (happy-dom does not always compute aria-labelledby into an
    // accessible name) and walk up to the section that carries
    // aria-live.
    // The h2 renders `Task {task_id.slice(0, 8)}…` — first 8 chars of
    // "task-aria-123" = "task-ari".
    const heading = await screen.findByText(/Task task-ari…/i);
    const region = heading.closest("section");
    expect(region).not.toBeNull();
    expect(region?.getAttribute("aria-live")).toBe("polite");
    expect(region?.getAttribute("aria-atomic")).toBe("false");

    // Status badge MUST carry an aria-label so a non-visual reader
    // gets the status word — the visual badge is colour-coded only.
    const badge = await screen.findByLabelText(/Task status:/i);
    expect(badge).toBeInTheDocument();

    // Polling tick assertion is intentionally NOT exercised here — the
    // page uses a 2 s `POLL_INTERVAL_MS` which would force this ARIA-
    // focused test to either burn a real-clock wait or wire fake
    // timers. The poll-lifecycle contract is exercised in
    // `pages.test.tsx` instead.
    void pollCalls;
  });
});

describe("Dashboard + Filings — Skeleton appears once per page (no fallback text)", () => {
  it("Dashboard's loading region is the Skeleton (not the old 'Loading status…' text)", () => {
    globalThis.fetch = vi.fn(() => new Promise(() => undefined)) as unknown as typeof fetch;
    render(<DashboardPage />);
    // The pre-13.7 placeholder was a plain `<p>Loading status…</p>`.
    // After the refactor, the visible loading hint is the sr-only
    // skeleton label — assert the old node is gone so a partial
    // revert surfaces here.
    expect(screen.queryByText(/Loading status…/i)).toBeNull();
    expect(screen.getByText(/Loading deployment status…/i)).toBeInTheDocument();
  });

  it("Filings's loading region is the Skeleton (not the old 'Loading filings…' <p>)", () => {
    globalThis.fetch = vi.fn(() => new Promise(() => undefined)) as unknown as typeof fetch;
    render(<FilingsPage />);
    // The Skeleton renders the label inside an sr-only span; the
    // `getByRole("status")` already pinned the region. Here we just
    // pin that the label still matches the operator-facing copy.
    const region = screen.getByRole("status");
    expect(region.textContent).toContain("Loading filings…");
  });
});

describe("Skeleton — multi-row independence", () => {
  // Sanity check: rendering two skeletons in different regions does
  // NOT collapse them into one status announcement (each carries its
  // own role=status node). This is the foundation for using the
  // component on both the dashboard status panel AND a future panel
  // on the same route without screen-reader cross-talk.
  it("two Skeletons in the same DOM mount as independent status regions", async () => {
    const { Skeleton } = await import("@/components/skeleton");
    render(
      <>
        <Skeleton rows={2} label="Loading A…" />
        <Skeleton rows={3} label="Loading B…" />
      </>,
    );
    const regions = screen.getAllByRole("status");
    expect(regions.length).toBe(2);
    expect(screen.getByText("Loading A…")).toBeInTheDocument();
    expect(screen.getByText("Loading B…")).toBeInTheDocument();
  });
});
