// Single source of truth for the SPA origin's security headers.
//
// Identical strict CSP in dev and prod, with only two dev-only carve-outs
// (`'unsafe-eval'` for Next HMR and `ws://localhost:*` in `connect-src`). No
// `'unsafe-inline'` in any environment. Trusted Types required for `script`.
//
// The header set MUST mirror the backend `SecurityHeadersMiddleware`
// (see src/sec_generative_search/api/middleware.py). When updating one,
// update the other in the same commit.

/** Build a strict CSP using the per-request nonce.
 *
 * `isDevelopment` MUST be derived from `process.env.NODE_ENV === "development"`
 * at the call site — never accept it from a request header, query string,
 * or runtime config (an attacker who flips the flag could relax the policy).
 */
export function buildContentSecurityPolicy(
  nonce: string,
  isDevelopment: boolean,
): string {
  // Dev-only carve-outs — both gated on isDevelopment.
  const scriptDevExtras = isDevelopment ? " 'unsafe-eval'" : "";
  const connectDevExtras = isDevelopment ? " ws://localhost:* http://localhost:*" : "";

  // The directives below are the production policy. Read top-to-bottom as
  // an explicit allow-list — everything not named here is denied.
  const directives: string[] = [
    "default-src 'self'",
    `script-src 'self' 'nonce-${nonce}'${scriptDevExtras}`,
    `style-src 'self' 'nonce-${nonce}'`,
    "img-src 'self' data:",
    "font-src 'self'",
    `connect-src 'self'${connectDevExtras}`,
    "frame-ancestors 'none'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "require-trusted-types-for 'script'",
    "upgrade-insecure-requests",
  ];

  return directives.join("; ");
}

/** Static security headers that do not depend on the per-request nonce.
 *
 * Mirrors the backend `SecurityHeadersMiddleware` directive-for-directive.
 * Header names use the canonical capitalisation; HTTP headers are
 * case-insensitive but consistent casing makes diffs readable.
 */
export function staticSecurityHeaders(): Record<string, string> {
  return {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": [
      "accelerometer=()",
      "camera=()",
      "geolocation=()",
      "gyroscope=()",
      "magnetometer=()",
      "microphone=()",
      "payment=()",
      "usb=()",
    ].join(", "),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
  };
}

/** Generate a 128-bit base64-encoded nonce using Web Crypto.
 *
 * Web Crypto is available in the Next middleware (Edge) runtime and in
 * Node 22 globals. We never use Math.random() for security tokens.
 */
export function generateNonce(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  // Base64 without padding fits the CSP nonce alphabet.
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replace(/=+$/, "");
}
