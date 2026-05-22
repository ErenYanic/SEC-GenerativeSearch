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
  StatusResponse,
  TaskListResponse,
  TaskStatusResponse,
} from "@/lib/api-types";

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

async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const res = await fetch(buildProxyUrl(path), {
    credentials: "same-origin",
    cache: "no-store",
    ...init,
    headers: {
      Accept: "application/json",
      ...(init.body !== undefined && init.body !== null
        ? { "Content-Type": "application/json" }
        : {}),
      ...(init.headers ?? {}),
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
