// RAG streaming mechanics + failure states (functional).
//
// The security suite (`tests/security/rag-page.test.tsx`) already pins
// the privacy-load-bearing streaming invariants (query-in-body, in-stream
// error retry affordance, pre-stream 4xx do-not-retry). This functional
// suite covers the operator-experience paths it leaves open:
//
//   - heartbeat frames interleaved with deltas are tolerated and never
//     corrupt the assembled answer;
//   - a `final` event with `refused: true` renders the refusal notice
//     instead of an answer body (no model output to show);
//   - a transport failure (the fetch itself rejecting, not an HTTP
//     envelope) is treated as maybe-retry with a Retry affordance;
//   - clicking Retry after an in-stream error re-runs generation and
//     clears the error on success;
//   - cancelling mid-stream returns the surface to idle without
//     surfacing an error or a half-built answer.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

vi.mock("next/navigation", async () => {
  const actual = await vi.importActual<Record<string, unknown>>(
    "next/navigation",
  );
  return {
    ...actual,
    usePathname: () => "/rag",
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      back: vi.fn(),
      refresh: vi.fn(),
      prefetch: vi.fn(),
    }),
  };
});

import RagPage from "@/app/(app)/rag/page";
import {
  emptyProvidersResponse,
  finalFrame,
  pendingStreamingResponse,
  sseFrame,
  streamingResponse,
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

function planResponse(): Response {
  return new Response(
    JSON.stringify({ plan: SAMPLE_PLAN, provider: "openai", model: "gpt-test" }),
    { status: 200 },
  );
}

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

// Drive the page from a blank textarea to a planned-and-generating
// state. Returns once the Generate click has fired.
async function planThenGenerate(
  user: ReturnType<typeof userEvent.setup>,
): Promise<void> {
  await user.type(screen.getByLabelText(/question/i), "Apple AI risk");
  await user.click(screen.getByRole("button", { name: /^Plan$/i }));
  await screen.findByText("AAPL");
  await user.click(screen.getByRole("button", { name: /generate answer/i }));
}

describe("RagPage — streaming mechanics", () => {
  it("tolerates interleaved heartbeat frames and assembles the full answer", async () => {
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/rag/plan") {
        return planResponse();
      }
      if (url === "/api/admin/providers/") {
        return emptyProvidersResponse();
      }
      return streamingResponse([
        sseFrame("heartbeat", {}),
        sseFrame("delta", { text: "Apple " }),
        sseFrame("heartbeat", {}),
        sseFrame("delta", { text: "discloses AI risk." }),
        sseFrame("heartbeat", {}),
        finalFrame("Apple discloses AI risk."),
      ]);
    }) as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await planThenGenerate(user);

    await waitFor(() => {
      expect(screen.getByLabelText(/^answer$/i).textContent ?? "").toContain(
        "Apple discloses AI risk.",
      );
    });
    // The heartbeats never leaked a visible token into the answer body.
    expect(screen.getByLabelText(/^answer$/i).textContent ?? "").not.toContain(
      "heartbeat",
    );
  });

  it("assembles the in-progress answer from many single-character deltas (F9 coalescing)", async () => {
    // Assert on the live streaming state before `final` replaces the
    // answer wholesale so the batched path is exercised directly.
    const partial = "Apple discloses AI risk";
    const pending = pendingStreamingResponse();
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/rag/plan") {
        return planResponse();
      }
      if (url === "/api/admin/providers/") {
        return emptyProvidersResponse();
      }
      return pending.response;
    }) as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await planThenGenerate(user);

    pending.release(partial.split("").map((ch) => sseFrame("delta", { text: ch })));

    await waitFor(() => {
      expect(screen.getByLabelText(/^answer$/i).textContent ?? "").toContain(
        partial,
      );
    });

    // Let the stream finish cleanly so React can unmount without a
    // dangling reader.
    pending.release([finalFrame(partial)]);
  });

  it("renders the refusal notice when the orchestrator refuses (final.refused)", async () => {
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/rag/plan") {
        return planResponse();
      }
      if (url === "/api/admin/providers/") {
        return emptyProvidersResponse();
      }
      return streamingResponse([
        sseFrame("final", {
          answer: "",
          provider: "openai",
          model: "gpt-test",
          prompt_version: "v1",
          token_usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
          latency_seconds: 0.0,
          streamed: true,
          refused: true,
        }),
      ]);
    }) as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await planThenGenerate(user);

    await waitFor(() => {
      expect(screen.getByText(/orchestrator\s+refused/i)).toBeInTheDocument();
    });
    // No Retry affordance: a refusal is a settled answer, not an error.
    expect(
      screen.queryByRole("button", { name: /retry/i }),
    ).not.toBeInTheDocument();
  });
});

describe("RagPage — failure states", () => {
  it("treats a transport failure as maybe-retry with a Retry affordance", async () => {
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/rag/plan") {
        return planResponse();
      }
      if (url === "/api/admin/providers/") {
        return emptyProvidersResponse();
      }
      // The fetch itself rejects (TCP reset / DNS failure) — NOT an HTTP
      // envelope. This is the non-ApiError branch of handleGenerate.
      throw new TypeError("Failed to fetch");
    }) as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await planThenGenerate(user);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/Lost contact with the answer stream/i);
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("retry after an in-stream error re-runs generation and clears the error", async () => {
    let streamCalls = 0;
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/rag/plan") {
        return planResponse();
      }
      if (url === "/api/admin/providers/") {
        return emptyProvidersResponse();
      }
      streamCalls += 1;
      if (streamCalls === 1) {
        // First attempt fails mid-stream (maybe-retry).
        return streamingResponse([
          sseFrame("delta", { text: "partial " }),
          sseFrame("error", {
            error: "provider_unavailable",
            message: "Upstream rate-limited.",
            hint: "Retry after a short backoff.",
          }),
        ]);
      }
      // Retry succeeds cleanly.
      return streamingResponse([
        sseFrame("delta", { text: "Recovered answer." }),
        finalFrame("Recovered answer."),
      ]);
    }) as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await planThenGenerate(user);

    // First attempt surfaces the in-stream error + Retry button.
    const retry = await screen.findByRole("button", { name: /retry/i });
    await user.click(retry);

    await waitFor(() => {
      expect(screen.getByLabelText(/^answer$/i).textContent ?? "").toContain(
        "Recovered answer.",
      );
    });
    // The error alert is gone after the successful retry.
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(streamCalls).toBe(2);
  });

  it("cancelling mid-stream returns to idle without an error or partial answer", async () => {
    const pending = pendingStreamingResponse();
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/rag/plan") {
        return planResponse();
      }
      if (url === "/api/admin/providers/") {
        return emptyProvidersResponse();
      }
      return pending.response;
    }) as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await planThenGenerate(user);

    // Surface a partial delta so we know an actual in-flight stream is
    // interrupted.
    pending.release([sseFrame("delta", { text: "partial answer " })]);
    await waitFor(() => {
      expect(screen.getByLabelText(/^answer$/i).textContent ?? "").toContain(
        "partial answer",
      );
    });

    await user.click(screen.getByRole("button", { name: /^cancel$/i }));

    // handleCancel resets generation to idle → the AnswerCard unmounts.
    await waitFor(() => {
      expect(screen.queryByLabelText(/^answer$/i)).not.toBeInTheDocument();
    });
    expect(screen.queryByText(/partial answer/)).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
