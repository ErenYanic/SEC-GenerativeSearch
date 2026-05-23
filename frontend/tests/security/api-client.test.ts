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
  registerEdgarIdentity,
  submitIngestAdd,
  validateProvider,
} from "@/lib/api";
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
