"use client";

// Model-catalogue refresh card.
//
// One operator action: trigger the admin-gated
// `POST /api/providers/catalogue/refresh`. The backend fetches the
// configured upstream metadata source (models.dev / LiteLLM), re-validates
// it as untrusted input, and writes an additive overlay merged over the
// bundled baseline; the handling worker reloads it in place.
//
// Trust boundary
// --------------
// The refresh is admin-tier — the admin proxy injects the operator's
// server-held `X-Admin-Key`. This card is rendered only when the status
// snapshot reports `is_admin`, so a non-admin never sees a button that would
// 403. The response is content-free (source + counts only); there is no
// per-token cost, model slug, credential, or filesystem path to leak.

import { useCallback, useState, type JSX } from "react";

import { ApiError, refreshModelCatalogue } from "@/lib/api";
import type { CatalogueRefreshResponse } from "@/lib/api-types";

type RefreshState =
  | { kind: "idle" }
  | { kind: "refreshing" }
  | { kind: "done"; report: CatalogueRefreshResponse }
  | { kind: "error"; message: string };

export function CatalogueRefreshCard(): JSX.Element {
  const [state, setState] = useState<RefreshState>({ kind: "idle" });

  const handleRefresh = useCallback(() => {
    setState((prev) => (prev.kind === "refreshing" ? prev : { kind: "refreshing" }));
    void (async () => {
      try {
        const report = await refreshModelCatalogue();
        setState({ kind: "done", report });
      } catch (exc) {
        // Field-agnostic: surface the backend's canned message or a generic
        // fallback — never user-supplied input (there is none on this route).
        const message =
          exc instanceof ApiError
            ? exc.message
            : "Could not reach the backend.";
        setState({ kind: "error", message });
      }
    })();
  }, []);

  const refreshing = state.kind === "refreshing";

  return (
    <section
      aria-labelledby="catalogue-refresh-heading"
      className="rounded-lg border border-slate-200 bg-white p-6 shadow"
    >
      <h2
        id="catalogue-refresh-heading"
        className="mb-2 text-lg font-semibold tracking-tight"
      >
        Model catalogue
      </h2>
      <p className="mb-4 text-sm text-slate-600">
        Fetch the latest model capabilities and pricing from the configured
        upstream source and merge them over the bundled baseline. The refresh
        is validated and fail-closed: a bad upstream never degrades the
        catalogue. Effective immediately on this server; the rest of a fleet
        picks it up on its next restart.
      </p>

      <button
        type="button"
        onClick={handleRefresh}
        disabled={refreshing}
        className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
      >
        {refreshing ? "Refreshing…" : "Refresh catalogue"}
      </button>

      {state.kind === "done" ? (
        <p
          className="mt-4 rounded border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800"
          role="status"
        >
          Refreshed from{" "}
          <span className="font-medium">{state.report.source}</span>:{" "}
          {state.report.provider_count.toLocaleString()} providers,{" "}
          {state.report.model_count.toLocaleString()} models.
        </p>
      ) : null}
      {state.kind === "error" ? (
        <p
          className="mt-4 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
          role="alert"
        >
          {state.message}
        </p>
      ) : null}
    </section>
  );
}
