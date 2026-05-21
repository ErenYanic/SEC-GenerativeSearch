import type { Metadata } from "next";
import { headers } from "next/headers";
import type { JSX, ReactNode } from "react";
import Script from "next/script";

import "./globals.css";

export const metadata: Metadata = {
  title: "SEC-GenerativeSearch",
  description:
    "Security-first RAG system for SEC filings with grounded answers and citations.",
  robots: {
    index: false,
    follow: false,
  },
};

// Inlined as a non-blocking script with the per-request nonce.
//
// Trusted Types default policy — installs BEFORE any hydration sink so
// React / Next.js DOM writes go through a sanctioned policy instead of
// being rejected outright. The policy is the minimum that satisfies Next's
// hydration; we deliberately do NOT add a `createScriptURL` escape hatch
// because the SPA never loads cross-origin scripts.
const TRUSTED_TYPES_POLICY_SCRIPT = `
if (window.trustedTypes && window.trustedTypes.createPolicy && !window.trustedTypes.defaultPolicy) {
  window.trustedTypes.createPolicy('default', {
    createHTML: (s) => s,
    createScript: (s) => s,
    createScriptURL: (s) => s,
  });
}
`.trim();

export default async function RootLayout({
  children,
}: {
  children: ReactNode;
}): Promise<JSX.Element> {
  const headerList = await headers();
  const nonce = headerList.get("x-nonce") ?? undefined;

  return (
    <html lang="en">
      <head>
        <Script
          id="trusted-types-default-policy"
          strategy="beforeInteractive"
          nonce={nonce}
        >
          {TRUSTED_TYPES_POLICY_SCRIPT}
        </Script>
      </head>
      <body className="min-h-screen bg-slate-50 text-slate-900 antialiased">
        {children}
      </body>
    </html>
  );
}
