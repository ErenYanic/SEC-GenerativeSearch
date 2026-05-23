// Typed browser-side client for the SEC-GenerativeSearch backend.
//
// Every call routes through the Next.js admin proxy at `/api/admin/*`.
// The proxy strips inbound auth headers and injects the operator's
// server-held keys; the browser never sees `X-API-Key` / `X-Admin-Key`.
//
// Error contract
// --------------
// Non-2xx responses surface as `ApiError`. The error message NEVER
// contains user-supplied input (form values, query strings) — it is
// either the backend's canned `message` or a generic fallback. Pages
// that need to render fields-with-errors must surface a field-agnostic
// notice; the offending value is intentionally not propagated.

import type {
  AnswerMode,
  CitationSchema,
  ConversationTurnSchema,
  EdgarIdentityRegisterResponse,
  FilingListResponse,
  IngestTaskResponse,
  ProviderListResponse,
  ProviderValidateResponse,
  QueryPlanSchema,
  RagPlanResponse,
  RagStreamFinalPayload,
  StatusResponse,
  TaskListResponse,
  TaskStatusResponse,
} from "@/lib/api-types";
import { providerKeyHeaders } from "@/lib/provider-keys";

/** Error raised on any non-2xx response from the admin proxy. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly hint?: string;

  constructor(status: number, code: string, message: string, hint?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.hint = hint;
  }
}

interface ErrorEnvelope {
  error?: string;
  message?: string;
  hint?: string;
}

const PROXY_PREFIX = "/api/admin/";

function buildProxyUrl(path: string): string {
  // Strip a leading `/api/` from the caller so both `/api/filings/` and
  // `filings/` work. A leading `/` without `api/` is treated as a raw
  // backend path. Trailing slashes are preserved — FastAPI is sensitive.
  let rel = path;
  if (rel.startsWith("/api/")) {
    rel = rel.slice("/api/".length);
  } else if (rel.startsWith("/")) {
    rel = rel.slice(1);
  }
  return PROXY_PREFIX + rel;
}

interface ApiFetchOptions extends RequestInit {
  /**
   * Attach `X-Provider-Key-{provider}` headers from the browser-side
   * `sessionStorage` store. Default `false` — most endpoints are
   * server-side admin proxy routes that do not need a downstream
   * provider call. Set to `true` for the RAG / search / validation
   * paths that actually invoke an upstream provider so the backend
   * resolver picks the per-request tier first.
   */
  attachProviderKeys?: boolean;
}

async function apiFetch<T>(
  path: string,
  init: ApiFetchOptions = {},
): Promise<T> {
  const { attachProviderKeys, ...rest } = init;
  const providerHeaders =
    attachProviderKeys === true ? providerKeyHeaders() : {};
  const res = await fetch(buildProxyUrl(path), {
    credentials: "same-origin",
    cache: "no-store",
    ...rest,
    headers: {
      Accept: "application/json",
      ...(rest.body !== undefined && rest.body !== null
        ? { "Content-Type": "application/json" }
        : {}),
      ...providerHeaders,
      ...(rest.headers ?? {}),
    },
  });
  if (res.status === 204) {
    return undefined as T;
  }
  const text = await res.text();
  let body: unknown = null;
  if (text !== "") {
    try {
      body = JSON.parse(text);
    } catch {
      // Body was not JSON — fall through to status-only handling.
    }
  }
  if (!res.ok) {
    const env =
      body !== null && typeof body === "object"
        ? (body as ErrorEnvelope)
        : {};
    throw new ApiError(
      res.status,
      typeof env.error === "string" ? env.error : "request_failed",
      typeof env.message === "string"
        ? env.message
        : `Request failed with status ${res.status}`,
      typeof env.hint === "string" ? env.hint : undefined,
    );
  }
  return body as T;
}

// ---------------------------------------------------------------------------
// Status — read-tier deployment snapshot
// ---------------------------------------------------------------------------

export function getStatus(): Promise<StatusResponse> {
  return apiFetch<StatusResponse>("status/");
}

// ---------------------------------------------------------------------------
// EDGAR identity — per-session SEC name + email
// ---------------------------------------------------------------------------

export interface EdgarIdentityRequestBody {
  name: string;
  email: string;
}

export function registerEdgarIdentity(
  body: EdgarIdentityRequestBody,
): Promise<EdgarIdentityRegisterResponse> {
  return apiFetch<EdgarIdentityRegisterResponse>("session/edgar", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function clearEdgarIdentity(): Promise<{ cleared: boolean }> {
  return apiFetch<{ cleared: boolean }>("session/edgar", {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Filings — list / detail / delete
// ---------------------------------------------------------------------------

export interface FilingsListParams {
  ticker?: string;
  form_type?: string;
  limit?: number;
  offset?: number;
}

export function listFilings(
  params: FilingsListParams = {},
): Promise<FilingListResponse> {
  const qs = new URLSearchParams();
  if (params.ticker !== undefined) qs.set("ticker", params.ticker);
  if (params.form_type !== undefined) qs.set("form_type", params.form_type);
  if (params.limit !== undefined) qs.set("limit", String(params.limit));
  if (params.offset !== undefined) qs.set("offset", String(params.offset));
  const query = qs.toString();
  return apiFetch<FilingListResponse>(
    `filings/${query !== "" ? `?${query}` : ""}`,
  );
}

export function deleteFiling(accession: string): Promise<{
  accession_number: string;
  chunks_deleted: number;
}> {
  return apiFetch(`filings/${encodeURIComponent(accession)}`, {
    method: "DELETE",
  });
}

export function deleteFilingsByIds(
  accessionNumbers: string[],
): Promise<{
  filings_deleted: number;
  chunks_deleted: number;
  not_found: string[];
}> {
  return apiFetch("filings/delete-by-ids", {
    method: "POST",
    body: JSON.stringify({ accession_numbers: accessionNumbers }),
  });
}

// ---------------------------------------------------------------------------
// Ingest — submit task, poll status
// ---------------------------------------------------------------------------

export interface IngestRequestBody {
  tickers: string[];
  form_types: string[];
  count_mode?: "latest" | "per_form" | "total";
  count?: number;
  year?: number;
  start_date?: string;
  end_date?: string;
}

export function submitIngestAdd(
  body: IngestRequestBody,
): Promise<IngestTaskResponse> {
  return apiFetch<IngestTaskResponse>("ingest/add", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function submitIngestBatch(
  body: IngestRequestBody,
): Promise<IngestTaskResponse> {
  return apiFetch<IngestTaskResponse>("ingest/batch", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getIngestTask(taskId: string): Promise<TaskStatusResponse> {
  return apiFetch<TaskStatusResponse>(
    `ingest/tasks/${encodeURIComponent(taskId)}`,
  );
}

export function listIngestTasks(): Promise<TaskListResponse> {
  return apiFetch<TaskListResponse>("ingest/tasks");
}

export function cancelIngestTask(
  taskId: string,
): Promise<{ task_id: string; status: string }> {
  return apiFetch(`ingest/tasks/${encodeURIComponent(taskId)}`, {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Providers — catalogue + key validation
// ---------------------------------------------------------------------------

export function listProviders(): Promise<ProviderListResponse> {
  return apiFetch<ProviderListResponse>("providers/");
}

export interface ProviderValidateRequestBody {
  provider: string;
  api_key: string;
  surface?: "llm" | "embedding" | "reranker";
  model?: string;
}

/**
 * Round-trip a candidate provider key against the upstream provider via
 * `POST /api/providers/validate`. Returns the verdict as
 * `ProviderValidateResponse.valid` — `false` is reserved for an
 * explicit auth rejection; transient failures propagate as `ApiError`
 * 502 / 503 so callers do not interpret a network blip as "wrong key".
 *
 * The key is sent in the JSON body, never on a URL or query string.
 * It is also attached as `X-Provider-Key-{provider}` so the backend's
 * audit-log entry pins lineage to the per-request header tier — this
 * is the documented happy path when a tenant validates their own key.
 */
export function validateProvider(
  body: ProviderValidateRequestBody,
): Promise<ProviderValidateResponse> {
  return apiFetch<ProviderValidateResponse>("providers/validate", {
    method: "POST",
    body: JSON.stringify(body),
    attachProviderKeys: true,
  });
}

// ---------------------------------------------------------------------------
// Provider-key propagation — used by RAG / search flows
// ---------------------------------------------------------------------------

/**
 * Helper for downstream callers that need to attach the browser-tier
 * provider keys to a single request. Wraps `apiFetch` with
 * `attachProviderKeys: true`. Exposed so future RAG / search modules
 * can plug into the same propagation pipeline without re-implementing
 * the header build.
 */
export function apiFetchWithProviderKeys<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  return apiFetch<T>(path, { ...init, attachProviderKeys: true });
}

// ---------------------------------------------------------------------------
// RAG — plan + streaming generation
// ---------------------------------------------------------------------------

export interface RagPlanRequestBody {
  query: string;
  provider?: string;
  model?: string;
}

/**
 * Run query-understanding for the editable chip UI. The raw query
 * travels in the body — never the URL — so it does not land in proxy
 * access logs. `X-Provider-Key-*` headers attach so the backend's
 * resolver chain hits the per-request tier first.
 */
export function planRagQuery(
  body: RagPlanRequestBody,
): Promise<RagPlanResponse> {
  return apiFetch<RagPlanResponse>("rag/plan", {
    method: "POST",
    body: JSON.stringify(body),
    attachProviderKeys: true,
  });
}

export interface RagStreamRequestBody {
  plan: QueryPlanSchema;
  provider?: string;
  model?: string;
  mode?: AnswerMode;
  max_output_tokens?: number;
  /**
   * Prior `Q:/A:` turns from the browser-tab chat surface.
   * The orchestrator renders these as a history block before retrieval
   * and generation; retrieved chunks and citations from prior turns are
   * intentionally not on the wire.
   */
  history?: ConversationTurnSchema[];
}

/**
 * Callback set for `streamRagAnswer`. Every callback fires on the
 * browser's main thread inside an async iterator — keep their bodies
 * non-blocking.
 *
 * - `onDelta` carries one streamed text fragment per call. Callers
 *   append these to render the incrementally-built answer.
 * - `onCitation` fires for each source chunk the model leaned on; the
 *   citation panel renders these as they arrive between the last delta
 *   and the final event.
 * - `onFinal` fires once with the fully-assembled answer and
 *   traceability (provider, model, token usage, refused-flag).
 * - `onError` fires when the SSE stream itself emits an `error` event
 *   after the response is already open — maybe-retry semantics.
 * - `onHeartbeat` is optional; the page can use it to gate a
 *   "still working" spinner.
 */
export interface RagStreamHandlers {
  onDelta?: (text: string) => void;
  onCitation?: (citation: CitationSchema) => void;
  onFinal?: (payload: RagStreamFinalPayload) => void;
  onError?: (error: { error: string; message: string; hint?: string }) => void;
  onHeartbeat?: () => void;
}

interface StreamEventFrame {
  event: string;
  data: string;
}

function parseSseFrames(buffer: string): {
  frames: StreamEventFrame[];
  remainder: string;
} {
  // SSE frames are delimited by a blank line. The backend always emits
  // `event:` then `data:` then a blank line — but we still parse
  // defensively so a stray comment line cannot derail the loop.
  const frames: StreamEventFrame[] = [];
  let cursor = 0;
  while (true) {
    const boundary = buffer.indexOf("\n\n", cursor);
    if (boundary === -1) {
      break;
    }
    const block = buffer.slice(cursor, boundary);
    cursor = boundary + 2;
    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) {
        // SSE comment — ignore.
        continue;
      }
      if (line.startsWith("event:")) {
        eventName = line.slice("event:".length).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice("data:".length).trim());
      }
    }
    frames.push({ event: eventName, data: dataLines.join("\n") });
  }
  return { frames, remainder: buffer.slice(cursor) };
}

/**
 * Open the RAG SSE stream and dispatch each event to the supplied
 * handlers. Returns when the server closes the stream or the supplied
 * `AbortSignal` fires.
 *
 * Error contract
 * --------------
 * - Pre-stream HTTP errors (unknown provider / 400 / 401 / 5xx) surface
 *   as `ApiError` so the caller can treat them as do-not-retry. The
 *   error envelope shape mirrors `apiFetch`.
 * - In-stream `error` events fire `onError` and the promise resolves
 *   normally — callers treat these as maybe-retry.
 * - `AbortError` from the supplied signal propagates as a rejection
 *   with `name === "AbortError"` so the caller can distinguish "user
 *   cancelled" from "backend failed".
 *
 * EventSource is NOT used because (a) it only supports GET and our
 * route is POST (query in body, never URL — backend contract), and
 * (b) EventSource cannot attach custom headers, so the
 * `X-Provider-Key-*` audit-log lineage would be lost.
 */
export async function streamRagAnswer(
  body: RagStreamRequestBody,
  handlers: RagStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const providerHeaders = providerKeyHeaders();
  const res = await fetch(buildProxyUrl("rag/stream"), {
    method: "POST",
    credentials: "same-origin",
    cache: "no-store",
    signal,
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
      ...providerHeaders,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    // Pre-stream error — read the JSON envelope and raise as ApiError.
    const text = await res.text();
    let env: ErrorEnvelope = {};
    if (text !== "") {
      try {
        const parsed = JSON.parse(text) as unknown;
        if (parsed !== null && typeof parsed === "object") {
          env = parsed as ErrorEnvelope;
        }
      } catch {
        // Body was not JSON — fall through to status-only envelope.
      }
    }
    throw new ApiError(
      res.status,
      typeof env.error === "string" ? env.error : "request_failed",
      typeof env.message === "string"
        ? env.message
        : `Request failed with status ${res.status}`,
      typeof env.hint === "string" ? env.hint : undefined,
    );
  }

  if (res.body === null) {
    // No body — backend signalled an empty stream (degenerate case).
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const { frames, remainder } = parseSseFrames(buffer);
      buffer = remainder;
      for (const frame of frames) {
        dispatchFrame(frame, handlers);
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // Reader may already be released if the response was cancelled.
    }
  }
}

function dispatchFrame(
  frame: StreamEventFrame,
  handlers: RagStreamHandlers,
): void {
  if (frame.event === "heartbeat") {
    handlers.onHeartbeat?.();
    return;
  }
  let payload: unknown = null;
  if (frame.data !== "") {
    try {
      payload = JSON.parse(frame.data);
    } catch {
      // Malformed frame — silently drop. The backend's contract is to
      // emit one JSON object per data line; a parse failure here is a
      // backend bug we do not want to escalate at the call site.
      return;
    }
  }
  if (payload === null || typeof payload !== "object") {
    return;
  }
  const record = payload as Record<string, unknown>;
  switch (frame.event) {
    case "delta": {
      const text = record.text;
      if (typeof text === "string" && handlers.onDelta !== undefined) {
        handlers.onDelta(text);
      }
      return;
    }
    case "citation": {
      handlers.onCitation?.(record as unknown as CitationSchema);
      return;
    }
    case "final": {
      handlers.onFinal?.(record as unknown as RagStreamFinalPayload);
      return;
    }
    case "error": {
      const errorCode = typeof record.error === "string" ? record.error : "stream_error";
      const message =
        typeof record.message === "string"
          ? record.message
          : "The stream ended with an error.";
      const hint = typeof record.hint === "string" ? record.hint : undefined;
      handlers.onError?.({ error: errorCode, message, hint });
      return;
    }
    default:
      // Unknown event — ignore (forward-compatible with future events).
      return;
  }
}
