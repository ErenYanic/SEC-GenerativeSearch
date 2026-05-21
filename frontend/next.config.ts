import type { NextConfig } from "next";

// Per-request CSP nonces live in middleware.ts, not here — config-level
// `headers()` cannot mint a per-request value. Static security headers that
// never need a nonce are still set in middleware too, so the whole header
// surface stays in one auditable place.
const config: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  productionBrowserSourceMaps: false,
  experimental: {
    // Trusted Types is enabled via the CSP header in middleware; this flag
    // tells Next to opt its hydration sinks into a default policy at build.
    // See src/app/layout.tsx for the client-side default policy installer.
  },
};

export default config;
