// RAG page (`src/app/(app)/rag/page.tsx`).
//
// Mounts the page in isolation and asserts:
//   - the plan request hits `/api/admin/rag/plan` with the raw query in
//     the body (never on the URL)
//   - the planned chips render exactly the backend response (no input
//     splicing)
//   - a streaming generation builds the answer from `delta` events and
//     populates the source panel from `citation` events
//   - in-stream `error` events surface a retry affordance without
//     wiping the partial answer
//   - a pre-stream 400 surfaces as a do-not-retry alert and never
//     opens the reader

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

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("RagPage — planning", () => {
  it("submits the query to /api/admin/rag/plan via POST body (never the URL)", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL) => {
      return new Response(
        JSON.stringify({
          plan: SAMPLE_PLAN,
          provider: "openai",
          model: "gpt-test",
        }),
        { status: 200 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/question/i), "Apple AI risk");
    await user.click(screen.getByRole("button", { name: /^Plan$/i }));

    await waitFor(() => {
      // Find the rag/plan call (the page also fires listProviders on
      // mount; we don't care about the order, only that the plan call
      // carries the body and never the URL).
      const planCall = fetchMock.mock.calls.find(
        (call) => (call[0] as string) === "/api/admin/rag/plan",
      );
      expect(planCall).toBeDefined();
    });
    const planCall = fetchMock.mock.calls.find(
      (call) => (call[0] as string) === "/api/admin/rag/plan",
    ) as unknown as [string, RequestInit];
    const [url, init] = planCall;
    expect(url).toBe("/api/admin/rag/plan");
    expect(url).not.toContain("Apple");
    expect(init.body).toContain("Apple AI risk");
  });

  it("renders the planned chips from the backend response", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          plan: SAMPLE_PLAN,
          provider: "openai",
          model: "gpt-test",
        }),
        { status: 200 },
      );
    }) as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/question/i), "Apple AI risk");
    await user.click(screen.getByRole("button", { name: /^Plan$/i }));

    await waitFor(() => {
      expect(screen.getByText("AAPL")).toBeInTheDocument();
    });
    expect(screen.getByText("10-K")).toBeInTheDocument();
    // Both `openai` chips render in their own labels; ambiguous matchers
    // are fine — we only need to assert presence.
    expect(screen.getAllByText("openai").length).toBeGreaterThan(0);
  });

  it("surfaces a 400 plan failure without echoing the input", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          error: "provider_key_required",
          message: "No API key resolved for provider 'openai'.",
          hint: "Register the key on the Providers page.",
        }),
        { status: 400 },
      );
    }) as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/question/i), "SECRET-INPUT");
    await user.click(screen.getByRole("button", { name: /^Plan$/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/No API key resolved/);
    expect(alert.textContent ?? "").not.toContain("SECRET-INPUT");
  });
});

describe("RagPage — generation streaming", () => {
  it("streams delta + citation + final events into the answer + source panel", async () => {
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
        sseFrame("delta", { text: "Apple cited [1] " }),
        sseFrame("delta", { text: "AI risk." }),
        sseFrame("citation", {
          chunk_id: "c1",
          ticker: "AAPL",
          form_type: "10-K",
          filing_date: "2024-09-30",
          accession_number: "0000320193-23-000077",
          section_path: "Risk Factors",
          text_span: "AI risk text.",
          similarity: 0.9,
          display_index: 1,
        }),
        sseFrame("final", {
          answer: "Apple cited [1] AI risk.",
          provider: "openai",
          model: "gpt-test",
          prompt_version: "v1",
          token_usage: {
            input_tokens: 10,
            output_tokens: 20,
            total_tokens: 30,
          },
          latency_seconds: 0.4,
          streamed: true,
          refused: false,
        }),
      ]);
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/question/i), "Apple AI risk");
    await user.click(screen.getByRole("button", { name: /^Plan$/i }));
    await screen.findByText("AAPL");
    await user.click(screen.getByRole("button", { name: /generate answer/i }));

    await waitFor(() => {
      expect(screen.getByText(/Sources \(1\)/)).toBeInTheDocument();
    });
    // The streamed answer text appears.
    expect(screen.getByLabelText(/^answer$/i).textContent ?? "").toContain(
      "Apple cited",
    );
    // The source panel renders the citation chunk.
    expect(screen.getByText("AI risk text.")).toBeInTheDocument();
    expect(screen.getByText(/Risk Factors/)).toBeInTheDocument();

    // The stream call must POST and never include the query in the URL.
    const streamCall = fetchMock.mock.calls.find(
      (call) => (call[0] as string) === "/api/admin/rag/stream",
    );
    expect(streamCall).toBeDefined();
    const [streamUrl, streamInit] = streamCall as unknown as [
      string,
      RequestInit,
    ];
    expect(streamUrl).toBe("/api/admin/rag/stream");
    expect(streamUrl).not.toContain("Apple");
    expect(streamInit.method).toBe("POST");
  });

  it("surfaces an in-stream error event with a retry affordance and preserves the partial answer", async () => {
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
        sseFrame("delta", { text: "partial answer " }),
        sseFrame("error", {
          error: "provider_unavailable",
          message: "Upstream rate-limited.",
          hint: "Retry after a short backoff.",
        }),
      ]);
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/question/i), "x");
    await user.click(screen.getByRole("button", { name: /^Plan$/i }));
    await screen.findByText("AAPL");
    await user.click(screen.getByRole("button", { name: /generate answer/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/Upstream rate-limited/);
    // Partial answer kept.
    expect(screen.getByLabelText(/^answer$/i).textContent ?? "").toContain(
      "partial answer",
    );
    // Retryable error → Retry button is offered.
    expect(
      screen.getByRole("button", { name: /retry/i }),
    ).toBeInTheDocument();
  });

  it("surfaces a pre-stream 4xx HTTP envelope without echoing the query", async () => {
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

    render(<RagPage />);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText(/question/i), "TOPSECRETQUERY");
    await user.click(screen.getByRole("button", { name: /^Plan$/i }));
    await screen.findByText("AAPL");
    await user.click(screen.getByRole("button", { name: /generate answer/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/rejected the supplied API key/);
    expect(alert.textContent ?? "").not.toContain("TOPSECRETQUERY");
    // Pre-stream failures are do-not-retry — no retry button is shown.
    expect(
      screen.queryByRole("button", { name: /retry/i }),
    ).not.toBeInTheDocument();
  });
});

describe("RagPage — ModelPicker wiring", () => {
  it("forwards picker provider/model/routing_hints into the stream POST body", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/providers/") {
        return new Response(
          JSON.stringify({
            providers: [
              { name: "openai", surface: "llm", supports_upstream_routing: false },
              {
                name: "openrouter",
                surface: "llm",
                supports_upstream_routing: true,
              },
            ],
            total: 2,
          }),
          { status: 200 },
        );
      }
      if (url === "/api/admin/rag/plan") {
        return new Response(
          JSON.stringify({
            plan: SAMPLE_PLAN,
            provider: "openrouter",
            model: "openai/gpt-test",
          }),
          { status: 200 },
        );
      }
      return streamingResponse([
        sseFrame("final", {
          answer: "ok.",
          provider: "openrouter",
          model: "openai/gpt-test",
          prompt_version: "v1",
          token_usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
          latency_seconds: 0.1,
          streamed: true,
          refused: false,
        }),
      ]);
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();

    // Wait for the catalogue fetch to settle so the picker renders the
    // routing-capable provider option.
    await waitFor(() => {
      expect(
        screen.getByRole("option", { name: /openrouter \(routing\)/i }),
      ).toBeInTheDocument();
    });

    // Pick provider + model.
    await user.selectOptions(
      screen.getByLabelText(/^provider$/i),
      "openrouter",
    );
    await user.type(screen.getByLabelText(/model slug/i), "openai/gpt-test");
    // The routing-hint UI MUST render only after the openrouter choice.
    await user.type(
      screen.getByLabelText(/preferred upstream order/i),
      "anthropic",
    );

    // Fire the plan + generate flow.
    await user.type(screen.getByLabelText(/question/i), "topic");
    await user.click(screen.getByRole("button", { name: /^Plan$/i }));
    await screen.findByText("AAPL");
    await user.click(screen.getByRole("button", { name: /generate answer/i }));

    await waitFor(() => {
      const streamCall = fetchMock.mock.calls.find(
        (c) => (c[0] as string) === "/api/admin/rag/stream",
      );
      expect(streamCall).toBeDefined();
    });
    const streamCall = fetchMock.mock.calls.find(
      (c) => (c[0] as string) === "/api/admin/rag/stream",
    ) as unknown as [string, RequestInit];
    const body = JSON.parse(streamCall[1].body as string) as {
      provider: string;
      model: string;
      routing_hints?: { order?: string[] };
    };
    expect(body.provider).toBe("openrouter");
    expect(body.model).toBe("openai/gpt-test");
    expect(body.routing_hints?.order).toEqual(["anthropic"]);
  });

  it("does NOT attach routing_hints when the provider is non-OpenRouter", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/providers/") {
        return new Response(
          JSON.stringify({
            providers: [
              { name: "openai", surface: "llm", supports_upstream_routing: false },
            ],
            total: 1,
          }),
          { status: 200 },
        );
      }
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
        sseFrame("final", {
          answer: "ok.",
          provider: "openai",
          model: "gpt-test",
          prompt_version: "v1",
          token_usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
          latency_seconds: 0.1,
          streamed: true,
          refused: false,
        }),
      ]);
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await waitFor(() => {
      expect(
        screen.getByRole("option", { name: "openai" }),
      ).toBeInTheDocument();
    });
    await user.selectOptions(screen.getByLabelText(/^provider$/i), "openai");
    // Routing UI MUST NOT render for a non-routing provider — the
    // capability gating is the SPA's side of the backend's fail-closed
    // guard against `invalid_flag_combination`.
    expect(
      screen.queryByLabelText(/preferred upstream order/i),
    ).not.toBeInTheDocument();

    await user.type(screen.getByLabelText(/question/i), "topic");
    await user.click(screen.getByRole("button", { name: /^Plan$/i }));
    await screen.findByText("AAPL");
    await user.click(screen.getByRole("button", { name: /generate answer/i }));

    await waitFor(() => {
      const streamCall = fetchMock.mock.calls.find(
        (c) => (c[0] as string) === "/api/admin/rag/stream",
      );
      expect(streamCall).toBeDefined();
    });
    const streamCall = fetchMock.mock.calls.find(
      (c) => (c[0] as string) === "/api/admin/rag/stream",
    ) as unknown as [string, RequestInit];
    const body = JSON.parse(streamCall[1].body as string) as Record<
      string,
      unknown
    >;
    // The picker never attached `routing_hints` (capability gated), so
    // the body MUST NOT carry the field.  This is the SPA's contribution
    // to the load-bearing invariant that the backend's
    // `invalid_flag_combination` guard never fires under normal SPA use.
    expect(body).not.toHaveProperty("routing_hints");
  });
});
