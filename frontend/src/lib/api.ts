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
  EdgarIdentityRegisterResponse,
  FilingListResponse,
  IngestTaskResponse,
  ProviderListResponse,
  ProviderValidateResponse,
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
