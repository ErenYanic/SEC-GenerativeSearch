// Chat page (`src/app/(app)/chat/page.tsx`).
//
// Mounts the multi-turn chat surface in isolation and asserts the
// Phase-13.5 contract:
//
//   - Each user message runs `POST /api/admin/rag/plan` then
//     `POST /api/admin/rag/stream`; the raw query travels in the body
//     of BOTH calls, never on the URL.
//   - After a clean `final` event, the turn is committed to the
//     transcript; the next message includes the prior turn in the
//     `history` array on the stream body.
//   - Cancelling an in-flight stream (the Cancel button — the
//     keyboard-driven mirror of the CLI's first Ctrl-C) does NOT
//     commit the turn to history. A subsequent send carries an empty
//     history.
//   - The Clear history button drops the transcript.
//   - The browser-tab is the only home of the history — no
//     `localStorage` / `sessionStorage` writes leak from the surface
//     (asserted by the static `storage-discipline` regression already).

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

function sseFrame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function streamingResponse(frames: string[]): Response {
  const encoder = new TextEncoder();
  let i = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i >= frames.length) {
        controller.close();
        return;
      }
      controller.enqueue(encoder.encode(frames[i] as string));
      i += 1;
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function pendingStreamingResponse(): {
  response: Response;
  release: (frames: string[]) => void;
} {
  // A stream the test can hold open until it explicitly releases the
  // frames. Used to simulate a cancel mid-flight.
  let pull: ((frames: string[]) => void) | null = null;
  let queued: string[] = [];
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      // Buffer frames; close on empty after release.
      if (queued.length > 0) {
        controller.enqueue(encoder.encode(queued.shift() as string));
        return;
      }
      pull = (frames) => {
        queued.push(...frames);
        pull = null;
        if (queued.length === 0) {
          controller.close();
        } else {
          controller.enqueue(encoder.encode(queued.shift() as string));
        }
      };
    },
  });
  const response = new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
  function release(frames: string[]): void {
    if (pull !== null) {
      pull(frames);
    } else {
      queued.push(...frames);
    }
  }
  return { response, release };
}

function finalFrame(answer: string): string {
  return sseFrame("final", {
    answer,
    provider: "openai",
    model: "gpt-test",
    prompt_version: "v1",
    token_usage: { input_tokens: 1, output_tokens: 2, total_tokens: 3 },
    latency_seconds: 0.1,
    streamed: true,
    refused: false,
  });
}

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("ChatPage — happy path", () => {
  it("sends a message, streams the answer, and renders the assistant turn", async () => {
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
      return streamingResponse([
        sseFrame("delta", { text: "Apple " }),
        sseFrame("delta", { text: "answer." }),
        finalFrame("Apple answer."),
      ]);
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<ChatPage />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/your message/i), "tell me about AAPL");
    await user.click(screen.getByRole("button", { name: /^Send$/i }));

    await waitFor(() => {
      expect(screen.getByText("Apple answer.")).toBeInTheDocument();
    });
    // The user message is rendered in the transcript.
    expect(screen.getByText("tell me about AAPL")).toBeInTheDocument();
    // Both plan + stream were hit; query is in the body, never on the URL.
    const planCall = fetchMock.mock.calls.find(
      (c) => (c[0] as string) === "/api/admin/rag/plan",
    );
    expect(planCall).toBeDefined();
    const streamCall = fetchMock.mock.calls.find(
      (c) => (c[0] as string) === "/api/admin/rag/stream",
    );
    expect(streamCall).toBeDefined();
    expect(streamCall![0]).not.toContain("AAPL");
  });

  it("forwards the prior turn as history on the next message", async () => {
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
      return streamingResponse([
        sseFrame("delta", { text: "first answer." }),
        finalFrame("first answer."),
      ]);
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<ChatPage />);
    const user = userEvent.setup();
    const input = screen.getByLabelText(/your message/i);
    await user.type(input, "first question");
    await user.click(screen.getByRole("button", { name: /^Send$/i }));
    await waitFor(() => {
      expect(screen.getByText("first answer.")).toBeInTheDocument();
    });

    // Reset the fetch mock for the second turn (we want a clean
    // sequence of plan + stream for the follow-up).
    fetchMock.mockClear();

    await user.type(input, "second question");
    await user.click(screen.getByRole("button", { name: /^Send$/i }));

    await waitFor(() => {
      const streamCall = fetchMock.mock.calls.find(
        (c) => (c[0] as string) === "/api/admin/rag/stream",
      );
      expect(streamCall).toBeDefined();
    });
    const streamCall = fetchMock.mock.calls.find(
      (c) => (c[0] as string) === "/api/admin/rag/stream",
    );
    const [, init] = streamCall as unknown as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      history: Array<{ query: string; answer: string }>;
    };
    expect(body.history).toEqual([
      { query: "first question", answer: "first answer." },
    ]);
  });
});

describe("ChatPage — cancellation semantics", () => {
  it("cancelling an in-flight turn does NOT commit it to history", async () => {
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
      // The page calls listProviders() on mount;
      // serve it an empty catalogue so the consume-once `pending`
      // body is reserved for the SSE stream call below.
      if (url === "/api/admin/providers/") {
        return new Response(
          JSON.stringify({ providers: [], total: 0 }),
          { status: 200 },
        );
      }
      return pending.response;
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<ChatPage />);
    const user = userEvent.setup();
    const input = screen.getByLabelText(/your message/i);
    await user.type(input, "doomed question");
    await user.click(screen.getByRole("button", { name: /^Send$/i }));

    // Wait for the stream to open and surface a partial delta so we
    // know cancellation interrupts an actually-running stream.
    pending.release([sseFrame("delta", { text: "partial " })]);
    await waitFor(() => {
      expect(
        screen.getByText("doomed question"),
      ).toBeInTheDocument();
    });

    // Cancel mid-flight.
    await user.click(screen.getByRole("button", { name: /^Cancel$/i }));

    // The transcript has no assistant turn; only the in-flight (now
    // dropped) bubble is gone.
    expect(screen.queryByText(/partial/)).not.toBeInTheDocument();

    // Send a follow-up; the body must carry an EMPTY history because
    // the cancelled turn was never committed.
    fetchMock.mockClear();
    const followup = pendingStreamingResponse();
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
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
        return new Response(
          JSON.stringify({ providers: [], total: 0 }),
          { status: 200 },
        );
      }
      return followup.response;
    });

    await user.type(input, "fresh question");
    await user.click(screen.getByRole("button", { name: /^Send$/i }));

    await waitFor(() => {
      const streamCall = fetchMock.mock.calls.find(
        (c) => (c[0] as string) === "/api/admin/rag/stream",
      );
      expect(streamCall).toBeDefined();
    });
    const streamCall = fetchMock.mock.calls.find(
      (c) => (c[0] as string) === "/api/admin/rag/stream",
    );
    const [, init] = streamCall as unknown as [string, RequestInit];
    const body = JSON.parse(init.body as string) as {
      history: Array<{ query: string; answer: string }>;
    };
    expect(body.history).toEqual([]);
    // Let the second stream finish cleanly so React can unmount.
    followup.release([finalFrame("fresh answer.")]);
  });

  it("clears history on demand", async () => {
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
      return streamingResponse([finalFrame("first answer.")]);
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<ChatPage />);
    const user = userEvent.setup();
    const input = screen.getByLabelText(/your message/i);
    await user.type(input, "first question");
    await user.click(screen.getByRole("button", { name: /^Send$/i }));
    await waitFor(() => {
      expect(screen.getByText("first answer.")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /clear history/i }));
    expect(screen.queryByText("first answer.")).not.toBeInTheDocument();
    expect(screen.queryByText("first question")).not.toBeInTheDocument();
  });
});

describe("ChatPage — error contract", () => {
  it("surfaces a pre-stream 4xx without echoing the user input", async () => {
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
      return new Response(
        JSON.stringify({
          error: "provider_unauthorized",
          message: "The upstream provider rejected the supplied API key.",
        }),
        { status: 401 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<ChatPage />);
    const user = userEvent.setup();
    await user.type(
      screen.getByLabelText(/your message/i),
      "SENSITIVE-QUERY-XYZ",
    );
    await user.click(screen.getByRole("button", { name: /^Send$/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/rejected the supplied API key/);
    expect(alert.textContent ?? "").not.toContain("SENSITIVE-QUERY-XYZ");
    // Pre-stream errors are do-not-retry — no Retry button rendered.
    expect(
      screen.queryByRole("button", { name: /retry/i }),
    ).not.toBeInTheDocument();
  });
});
