// Typed backend client (`src/lib/api.ts`).
//
// Asserts:
//  - all calls route through /api/admin/* — never the raw backend URL
//  - non-2xx responses surface as ApiError without echoing input
//  - request bodies are JSON-encoded; GETs carry no body
//  - 204 No Content does not error on `.json()`

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  apiFetchWithProviderKeys,
  clearEdgarIdentity,
  deleteFiling,
  getStatus,
  listFilings,
  listProviders,
  planRagQuery,
  registerEdgarIdentity,
  streamRagAnswer,
  submitIngestAdd,
  validateProvider,
} from "@/lib/api";
import type { QueryPlanSchema } from "@/lib/api-types";
import {
  clearProviderKeys,
  setProviderKey,
} from "@/lib/provider-keys";

const originalFetch = globalThis.fetch;

interface Captured {
  url: string;
  init: RequestInit;
}

function recordingFetch(response: Response): {
  fetchMock: ReturnType<typeof vi.fn>;
  calls: Captured[];
} {
  const calls: Captured[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({
      url: typeof input === "string" ? input : input.toString(),
      init: init ?? {},
    });
    return response.clone();
  });
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  return { fetchMock, calls };
}

beforeEach(() => {
  globalThis.fetch = vi.fn();
  window.sessionStorage.clear();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
  clearProviderKeys();
});

describe("api client routing", () => {
  it("routes every call through /api/admin/* (never the raw backend)", async () => {
    const { calls } = recordingFetch(
      new Response(JSON.stringify({}), { status: 200 }),
    );
    await getStatus();
    await listFilings({ ticker: "AAPL" });
    await registerEdgarIdentity({ name: "Doe", email: "d@example.com" });
    for (const call of calls) {
      expect(call.url.startsWith("/api/admin/")).toBe(true);
    }
  });

  it("uses same-origin credentials and no-store cache for every request", async () => {
    const { calls } = recordingFetch(
      new Response(JSON.stringify({}), { status: 200 }),
    );
    await getStatus();
    expect(calls[0]?.init.credentials).toBe("same-origin");
    expect(calls[0]?.init.cache).toBe("no-store");
  });

  it("JSON-encodes POST bodies with Content-Type: application/json", async () => {
    const { calls } = recordingFetch(
      new Response(JSON.stringify({ registered: true }), { status: 201 }),
    );
    await registerEdgarIdentity({ name: "Alice", email: "a@example.com" });
    const init = calls[0]?.init;
    expect(init?.method).toBe("POST");
    const headers = new Headers(init?.headers as HeadersInit);
    expect(headers.get("content-type")).toBe("application/json");
    expect(init?.body).toBe(
      JSON.stringify({ name: "Alice", email: "a@example.com" }),
    );
  });

  it("does NOT attach a Content-Type header on GET requests", async () => {
    const { calls } = recordingFetch(
      new Response(JSON.stringify({ filings: [], total: 0 }), { status: 200 }),
    );
    await listFilings();
    const headers = new Headers(calls[0]?.init.headers as HeadersInit);
    expect(headers.get("content-type")).toBeNull();
    expect(calls[0]?.init.body).toBeUndefined();
  });

  it("returns a typed body on 204 No Content without throwing", async () => {
    recordingFetch(new Response(null, { status: 204 }));
    // delete-by-accession returns the envelope; even an empty 204 should
    // resolve. We exercise via deleteFiling which returns a typed shape.
    const result = await deleteFiling("0000320193-23-000077");
    expect(result).toBeUndefined();
  });
});

describe("api client error contract", () => {
  it("raises ApiError with the backend's canned message — never the operator input", async () => {
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

    let raised: unknown;
    try {
      // Submit a deliberately ugly value; assert it does NOT appear in
      // the surfaced error message.
      await registerEdgarIdentity({
        name: "BadName",
        email: "badEmail",
      });
    } catch (exc) {
      raised = exc;
    }
    expect(raised).toBeInstanceOf(ApiError);
    const err = raised as ApiError;
    expect(err.status).toBe(400);
    expect(err.code).toBe("invalid_edgar_identity");
    expect(err.message).toBe("EDGAR identity failed validation.");
    expect(err.message).not.toContain("BadName");
    expect(err.message).not.toContain("badEmail");
    // The hint surfaces the validator's explanation but is field-agnostic.
    expect(err.hint).toBe("name must not contain control characters");
    expect(err.hint).not.toContain("BadName");
  });

  it("falls back to a generic message when the body is not JSON", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response("plain text body", { status: 500 });
    }) as unknown as typeof fetch;
    let raised: unknown;
    try {
      await getStatus();
    } catch (exc) {
      raised = exc;
    }
    expect(raised).toBeInstanceOf(ApiError);
    const err = raised as ApiError;
    expect(err.status).toBe(500);
    expect(err.message).toMatch(/status 500/);
    // Generic fallback — never echoes the response body.
    expect(err.message).not.toContain("plain text body");
  });

  it("preserves trailing slash on filings list and target path", async () => {
    const { calls } = recordingFetch(
      new Response(JSON.stringify({ filings: [], total: 0 }), { status: 200 }),
    );
    await listFilings();
    expect(calls[0]?.url).toBe("/api/admin/filings/");
  });

  it("URL-encodes accession numbers in path parameters", async () => {
    const { calls } = recordingFetch(
      new Response(JSON.stringify({}), { status: 200 }),
    );
    await deleteFiling("0000320193-23-000077");
    expect(calls[0]?.url).toBe(
      "/api/admin/filings/0000320193-23-000077",
    );
  });

  it("DELETE EDGAR identity goes through session/edgar with no body", async () => {
    const { calls } = recordingFetch(
      new Response(JSON.stringify({ cleared: true }), { status: 200 }),
    );
    await clearEdgarIdentity();
    expect(calls[0]?.url).toBe("/api/admin/session/edgar");
    expect(calls[0]?.init.method).toBe("DELETE");
    expect(calls[0]?.init.body).toBeUndefined();
  });

  it("ingest add submits JSON body to /api/admin/ingest/add", async () => {
    const { calls } = recordingFetch(
      new Response(
        JSON.stringify({
          task_id: "abc",
          status: "pending",
          websocket_url: "/ws/ingest/abc",
        }),
        { status: 202 },
      ),
    );
    await submitIngestAdd({
      tickers: ["AAPL"],
      form_types: ["10-K"],
      count: 1,
    });
    expect(calls[0]?.url).toBe("/api/admin/ingest/add");
    expect(calls[0]?.init.method).toBe("POST");
  });
});

describe("provider catalogue + validation", () => {
  it("listProviders calls GET /api/admin/providers/", async () => {
    const { calls } = recordingFetch(
      new Response(
        JSON.stringify({ providers: [], total: 0 }),
        { status: 200 },
      ),
    );
    await listProviders();
    expect(calls[0]?.url).toBe("/api/admin/providers/");
    expect(calls[0]?.init.method ?? "GET").toBe("GET");
  });

  it("validateProvider posts the body and attaches X-Provider-Key-* headers from the store", async () => {
    setProviderKey("openai", "sk-LONGENOUGHKEY"); // pragma: allowlist secret
    const { calls } = recordingFetch(
      new Response(
        JSON.stringify({ valid: true, provider: "openai", surface: "llm" }),
        { status: 200 },
      ),
    );
    const verdict = await validateProvider({
      provider: "openai",
      api_key: "sk-CANDIDATE", // pragma: allowlist secret
      surface: "llm",
    });
    expect(verdict.valid).toBe(true);
    const init = calls[0]?.init;
    expect(calls[0]?.url).toBe("/api/admin/providers/validate");
    expect(init?.method).toBe("POST");
    const headers = new Headers(init?.headers as HeadersInit);
    expect(headers.get("X-Provider-Key-openai")).toBe("sk-LONGENOUGHKEY");
    // Body carries the candidate, not the stored key — the route uses
    // the body as the canonical signal, headers only carry the audit
    // lineage. Tenants can validate a freshly-typed key.
    expect(init?.body).toContain("sk-CANDIDATE");
  });

  it("apiFetchWithProviderKeys attaches browser provider keys", async () => {
    setProviderKey("anthropic", "sk-ant-PROPAGATETHIS"); // pragma: allowlist secret
    const { calls } = recordingFetch(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    await apiFetchWithProviderKeys("rag/plan", {
      method: "POST",
      body: JSON.stringify({ query: "anything" }),
    });
    const headers = new Headers(calls[0]?.init.headers as HeadersInit);
    expect(headers.get("X-Provider-Key-anthropic")).toBe(
      "sk-ant-PROPAGATETHIS",
    );
  });

  it("default apiFetch path (e.g. getStatus) does NOT attach provider keys", async () => {
    setProviderKey("openai", "sk-SHOULDNOTLEAK"); // pragma: allowlist secret
    const { calls } = recordingFetch(
      new Response(JSON.stringify({}), { status: 200 }),
    );
    await getStatus();
    const headers = new Headers(calls[0]?.init.headers as HeadersInit);
    expect(headers.get("X-Provider-Key-openai")).toBeNull();
  });

  it("validateProvider surfaces a 502 ProviderError as ApiError without echoing the key", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          error: "provider_error",
          message: "The upstream provider returned an error.",
        }),
        { status: 502 },
      );
    }) as unknown as typeof fetch;
    let raised: unknown;
    try {
      await validateProvider({
        provider: "openai",
        api_key: "sk-SHOULDNOTAPPEAR", // pragma: allowlist secret
      });
    } catch (exc) {
      raised = exc;
    }
    expect(raised).toBeInstanceOf(ApiError);
    const err = raised as ApiError;
    expect(err.status).toBe(502);
    expect(err.message).not.toContain("sk-SHOULDNOTAPPEAR");
  });
});

// ---------------------------------------------------------------------------
// RAG plan + stream
// ---------------------------------------------------------------------------

const SAMPLE_PLAN: QueryPlanSchema = {
  raw_query: "test query",
  detected_language: "en",
  query_en: "test query",
  tickers: ["AAPL"],
  form_types: ["10-K"],
  date_range: null,
  intent: "lookup",
  suggested_answer_mode: "concise",
};

function sseFrame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function readableStreamFromChunks(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let i = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i >= chunks.length) {
        controller.close();
        return;
      }
      controller.enqueue(encoder.encode(chunks[i] as string));
      i += 1;
    },
  });
}

describe("planRagQuery", () => {
  it("POSTs the query in the body, never on the URL, and attaches provider keys", async () => {
    setProviderKey("openai", "sk-PLANLINEAGE"); // pragma: allowlist secret
    const { calls } = recordingFetch(
      new Response(
        JSON.stringify({
          plan: SAMPLE_PLAN,
          provider: "openai",
          model: "gpt-test",
        }),
        { status: 200 },
      ),
    );
    const res = await planRagQuery({ query: "How did AAPL describe AI risk?" });
    expect(res.plan.tickers).toEqual(["AAPL"]);
    const init = calls[0]?.init;
    expect(calls[0]?.url).toBe("/api/admin/rag/plan");
    expect(init?.method).toBe("POST");
    // The raw query MUST appear in the body — never on the URL.
    expect(calls[0]?.url).not.toContain("How did AAPL");
    expect(init?.body).toContain("How did AAPL");
    const headers = new Headers(init?.headers as HeadersInit);
    expect(headers.get("X-Provider-Key-openai")).toBe("sk-PLANLINEAGE");
  });

  it("raises ApiError on 400 provider_key_required without echoing the query", async () => {
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
    let raised: unknown;
    try {
      await planRagQuery({ query: "SECRET-QUERY-VALUE" });
    } catch (exc) {
      raised = exc;
    }
    expect(raised).toBeInstanceOf(ApiError);
    const err = raised as ApiError;
    expect(err.code).toBe("provider_key_required");
    expect(err.message).not.toContain("SECRET-QUERY-VALUE");
  });
});

describe("streamRagAnswer", () => {
  it("parses delta / citation / final events and never puts the query on the URL", async () => {
    const upstream = new Response(
      readableStreamFromChunks([
        sseFrame("delta", { text: "Hello " }),
        sseFrame("delta", { text: "world." }),
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
          answer: "Hello world.",
          provider: "openai",
          model: "gpt-test",
          prompt_version: "v1",
          token_usage: {
            input_tokens: 1,
            output_tokens: 2,
            total_tokens: 3,
          },
          latency_seconds: 0.1,
          streamed: true,
          refused: false,
        }),
      ]),
      {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      },
    );
    const fetchMock = vi.fn(async () => upstream);
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const deltas: string[] = [];
    let citationCount = 0;
    let finalSeen = false;
    await streamRagAnswer(
      {
        plan: SAMPLE_PLAN,
        mode: "concise",
      },
      {
        onDelta: (t) => deltas.push(t),
        onCitation: () => {
          citationCount += 1;
        },
        onFinal: () => {
          finalSeen = true;
        },
      },
    );
    expect(deltas.join("")).toBe("Hello world.");
    expect(citationCount).toBe(1);
    expect(finalSeen).toBe(true);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe("/api/admin/rag/stream");
    expect(url).not.toContain(SAMPLE_PLAN.raw_query);
    expect(init.method).toBe("POST");
    const headers = new Headers(init.headers as HeadersInit);
    expect(headers.get("Accept")).toBe("text/event-stream");
  });

  it("raises ApiError on a pre-stream 4xx and never opens the reader", async () => {
    globalThis.fetch = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          error: "unknown_provider",
          message: "Unknown provider 'nope'.",
        }),
        { status: 400 },
      );
    }) as unknown as typeof fetch;
    let raised: unknown;
    try {
      await streamRagAnswer({ plan: SAMPLE_PLAN }, {});
    } catch (exc) {
      raised = exc;
    }
    expect(raised).toBeInstanceOf(ApiError);
    const err = raised as ApiError;
    expect(err.status).toBe(400);
    expect(err.code).toBe("unknown_provider");
  });

  it("dispatches in-stream error events to onError (maybe-retry) without rejecting", async () => {
    const upstream = new Response(
      readableStreamFromChunks([
        sseFrame("delta", { text: "partial " }),
        sseFrame("error", {
          error: "provider_unavailable",
          message: "Upstream rate-limited.",
          hint: "Retry after a short backoff.",
        }),
      ]),
      { status: 200, headers: { "content-type": "text/event-stream" } },
    );
    globalThis.fetch = vi.fn(
      async () => upstream,
    ) as unknown as typeof fetch;

    let captured: { error: string; message: string; hint?: string } | null = null;
    await expect(
      streamRagAnswer(
        { plan: SAMPLE_PLAN },
        {
          onError: (e) => {
            captured = e;
          },
        },
      ),
    ).resolves.toBeUndefined();
    expect(captured).not.toBeNull();
    expect(captured!.error).toBe("provider_unavailable");
  });

  it("delivers heartbeat events to onHeartbeat", async () => {
    const upstream = new Response(
      readableStreamFromChunks([
        sseFrame("heartbeat", {}),
        sseFrame("heartbeat", {}),
      ]),
      { status: 200, headers: { "content-type": "text/event-stream" } },
    );
    globalThis.fetch = vi.fn(
      async () => upstream,
    ) as unknown as typeof fetch;
    let beats = 0;
    await streamRagAnswer(
      { plan: SAMPLE_PLAN },
      {
        onHeartbeat: () => {
          beats += 1;
        },
      },
    );
    expect(beats).toBe(2);
  });

  it("handles partial frame buffering across chunk boundaries", async () => {
    // Split the same frame across two read() calls to exercise the
    // line-buffering seam.
    const full = sseFrame("delta", { text: "split" });
    const half = Math.floor(full.length / 2);
    const upstream = new Response(
      readableStreamFromChunks([full.slice(0, half), full.slice(half)]),
      { status: 200, headers: { "content-type": "text/event-stream" } },
    );
    globalThis.fetch = vi.fn(
      async () => upstream,
    ) as unknown as typeof fetch;
    const deltas: string[] = [];
    await streamRagAnswer(
      { plan: SAMPLE_PLAN },
      { onDelta: (t) => deltas.push(t) },
    );
    expect(deltas.join("")).toBe("split");
  });
});
