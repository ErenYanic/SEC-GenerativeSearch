// Skeleton — visual placeholder rendered during initial loads.
//
// One shared primitive so every page that shows a loading state uses
// the same shape + animation. Backed by a `role="status"` wrapper so
// screen readers announce a non-blocking "loading" without speaking
// individual placeholder bars. The visual bars themselves carry
// `aria-hidden` so the announcement happens exactly once.
//
// Why a primitive (and not just inline divs per page): the strict CSP
// posture + Tailwind v4 means every shimmer / pulse animation lives in
// a stylesheet, not an inline style. Centralising the markup also gives
// `pages.test.tsx` a single `getByRole("status", { name: /loading…/i })`
// hook to assert against without coupling to per-page DOM shape.

import type { JSX } from "react";

export interface SkeletonProps {
  /** Number of placeholder bars to render. Default 3. */
  rows?: number;
  /** Optional accessible label announced once. */
  label?: string;
}

export function Skeleton({
  rows = 3,
  label = "Loading…",
}: SkeletonProps): JSX.Element {
  return (
    <div role="status" aria-busy="true" aria-live="polite" className="space-y-2">
      <span className="sr-only">{label}</span>
      {Array.from({ length: rows }).map((_, idx) => (
        <div
          key={idx}
          aria-hidden="true"
          className="h-3 w-full animate-pulse rounded bg-slate-200"
        />
      ))}
    </div>
  );
}
