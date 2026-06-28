// Small presentation-only formatting helpers shared across pages.
//
// Kept dependency-free and side-effect-free so it can be imported from any
// client component without pulling in a formatting library.

/**
 * Format a USD figure for display. Per-request RAG cost estimates are
 * fractions of a cent, so we allow up to six fraction digits while keeping
 * the leading two for readability. Used for both the per-turn / per-request
 * cost and the chat session-total surfaces.
 *
 * The figure is always an *estimate*, never a final bill — callers surface it
 * with that disclaimer and render `—` for an unknown-cost model.
 */
export function formatUsd(value: number): string {
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 6,
  });
}
