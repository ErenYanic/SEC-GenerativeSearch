// Chat page streaming mechanics (functional).
//
// The security suite (`tests/security/chat-page.test.tsx`) pins the
// privacy-load-bearing invariants (history round-trip, cancellation
// never commits). This file covers the operator-experience path where
// answer-delta dispatches are batched and the in-progress answer must
// still assemble correctly from many small deltas.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

vi.mock("next/navigation", async () => {
  const actual = await vi.importActual<Record<string, unknown>>(
    "next/navigation",
  );
  return {
    ...actual,
    usePathname: () => "/chat",
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      back: vi.fn(),
      refresh: vi.fn(),
      prefetch: vi.fn(),
    }),
  };
});

import ChatPage from "@/app/(app)/chat/page";
import {
  emptyProvidersResponse,
  finalFrame,
  pendingStreamingResponse,
  sseFrame,
} from "../security/_sse-harness";

const originalFetch = globalThis.fetch;

const SAMPLE_PLAN = {
  raw_query: "Apple AI risk",
  detected_language: "en",
  query_en: "Apple AI risk",
  tickers: ["AAPL"],
  form_types: ["10-K"],
  date_range: null,
  intent: "lookup",
  suggested_answer_mode: "concise",
};

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("ChatPage — streaming mechanics", () => {
  it("assembles the in-progress answer from many single-character deltas (F9 coalescing)", async () => {
    const partial = "Apple discloses AI risk";
    const pending = pendingStreamingResponse();
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/rag/plan") {
        return new Response(
          JSON.stringify({
            plan: SAMPLE_PLAN,
            provider: "openai",
            model: "gpt-test",
          }),
          { status: 200 },
        );
      }
      if (url === "/api/admin/providers/") {
        return emptyProvidersResponse();
      }
      return pending.response;
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<ChatPage />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/your message/i), "tell me about AAPL");
    await user.click(screen.getByRole("button", { name: /^Send$/i }));

    pending.release(partial.split("").map((ch) => sseFrame("delta", { text: ch })));

    // The in-progress pending turn (before `final` commits it) reflects
    // the full, in-order concatenation of every pushed delta.
    await waitFor(() => {
      expect(screen.getByText(partial)).toBeInTheDocument();
    });

    // Let the stream finish cleanly so React can unmount without a
    // dangling reader.
    pending.release([finalFrame(partial)]);
    await waitFor(() => {
      expect(screen.getAllByText(partial).length).toBeGreaterThan(0);
    });
  });
});
