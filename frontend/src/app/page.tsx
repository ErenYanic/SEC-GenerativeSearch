// Minimal landing page. Kept intentionally tiny so the scaffold can be
// smoke-tested against the security-header set without any business UI to
// debug.

import type { JSX } from "react";

export default function Page(): JSX.Element {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16">
      <h1 className="text-3xl font-semibold tracking-tight">
        SEC-GenerativeSearch
      </h1>
      <p className="mt-4 text-slate-600">
        Security-first RAG system for SEC filings. The web UI lands in
        the main app flows; this page exists to verify the scaffold + strict
        CSP boot cleanly.
      </p>
    </main>
  );
}
