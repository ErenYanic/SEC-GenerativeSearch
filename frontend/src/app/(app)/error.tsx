"use client";

// App-group error boundary (Next.js 16 convention).
//
// Any uncaught render error inside an `(app)/...` page lands here. The
// fallback renders the standard `AppShell` chrome (so navigation stays
// reachable) plus a recoverable error card with a Reset action that
// re-mounts the subtree via `reset()`.
//
// What the error message MUST NOT contain:
//   - the offending input value (queries, ticker symbols, EDGAR
//     identity) — the "errors never echo input" contract carries over;
//   - the raw `Error.message` from upstream `fetch` failures (those can
//     leak server-internal paths). We surface a generic operator-facing
//     message and a Reset affordance instead.
//
// Why a route-group boundary (not root): the root `app/layout.tsx`
// installs the Trusted Types policy via a nonced inline script — if
// the boundary lived there, the policy would re-install on every
// crash-recover cycle, which is unnecessary and complicates the CSP
// invariant tests. Scoping the boundary to `(app)` keeps the
// authenticated tree resilient without touching the root layout.

import type { JSX } from "react";

interface ErrorBoundaryProps {
  // Next.js passes the caught Error here. We intentionally do NOT
  // render `error.message` in the visible UI — Tier-3 user input may
  // appear in upstream `fetch` error messages and the page-level
  // contract is field-agnostic. The error name is safe (e.g. "TypeError",
  // "ApiError") and helpful for operator triage in the support flow.
  error: Error & { digest?: string };
  reset: () => void;
}

export default function AppGroupError({
  error,
  reset,
}: ErrorBoundaryProps): JSX.Element {
  return (
    <div
      role="alert"
      aria-live="assertive"
      className="space-y-4 rounded-lg border border-red-200 bg-red-50 p-6 shadow"
    >
      <h1 className="text-xl font-semibold tracking-tight text-red-900">
        Something went wrong.
      </h1>
      <p className="text-sm text-red-800">
        The page hit an unrecoverable error and was unmounted. Resetting
        re-renders the page; if the failure persists, sign out and back
        in. The operator audit log carries the failure under{" "}
        <span className="font-mono">
          {error.name}
          {error.digest !== undefined ? ` (${error.digest})` : ""}
        </span>
        .
      </p>
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => {
            reset();
          }}
          className="rounded bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-500"
        >
          Reset
        </button>
      </div>
    </div>
  );
}
