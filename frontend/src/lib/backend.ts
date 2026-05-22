// Backend-base-URL resolution for the server-side Next.js route handlers.
//
// MUST NOT be reachable from the client bundle — the proxy needs the
// canonical (private) backend URL. Browsers reach the backend only via the
// Next.js proxy at `/api/admin/...`, never directly.

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

export function getBackendBaseUrl(): string {
  const raw = process.env.SEC_API_BASE_URL ?? "";
  const url = raw === "" ? DEFAULT_BACKEND_URL : raw;
  // Trim trailing slash so callers can prepend `/api/...` unconditionally.
  return url.replace(/\/+$/, "");
}

/** Build an absolute backend URL from a relative API path.
 *
 * Path MUST start with `/api/` — anything else is rejected. Without this
 * gate, a buggy caller could be tricked into hitting an arbitrary URL
 * (open-redirect style; not applicable to the proxy but kept defensive).
 */
export function buildBackendUrl(apiPath: string): string {
  if (!apiPath.startsWith("/api/")) {
    throw new Error(`backend path must start with /api/, got: ${apiPath}`);
  }
  return `${getBackendBaseUrl()}${apiPath}`;
}
