"use client";

// Dashboard — operator landing surface.
//
// Pulls `GET /api/status/` (the only read-tier endpoint exposed on the
// admin proxy that requires no further set-up) and surfaces the
// deployment-profile-dependent metadata operators most often need: the
// embedder identity (the seal that ChromaDB enforces), the storage
// filing count, and whether persistent credential storage is enabled.
//
// The EDGAR identity card lives here so operators on B/C deployments
// (where `API_EDGAR_SESSION_REQUIRED=true`) can register before
// triggering an ingest.

import { useEffect, useState, type JSX } from "react";

import { CatalogueRefreshCard } from "@/components/catalogue-refresh-card";
import { EdgarIdentityCard } from "@/components/edgar-identity-card";
import { Skeleton } from "@/components/skeleton";
import { ApiError, getStatus } from "@/lib/api";
import type { StatusResponse } from "@/lib/api-types";

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; data: StatusResponse }
  | { kind: "error"; message: string };

export default function DashboardPage(): JSX.Element {
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const data = await getStatus();
        if (!cancelled) {
          setState({ kind: "ready", data });
        }
      } catch (exc) {
        if (cancelled) {
          return;
        }
        const message =
          exc instanceof ApiError
            ? exc.message
            : "Could not reach the backend.";
        setState({ kind: "error", message });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Dashboard</h1>
        <p className="text-sm text-slate-600">
          Deployment snapshot and per-session set-up.
        </p>
      </header>

      <section
        aria-labelledby="status-heading"
        className="rounded-lg border border-slate-200 bg-white p-6 shadow"
      >
        <h2
          id="status-heading"
          className="mb-4 text-lg font-semibold tracking-tight"
        >
          Deployment status
        </h2>
        {state.kind === "loading" ? (
          <Skeleton rows={4} label="Loading deployment status…" />
        ) : null}
        {state.kind === "error" ? (
          <p
            className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
            role="alert"
          >
            {state.message}
          </p>
        ) : null}
        {state.kind === "ready" ? (
          <StatusGrid data={state.data} />
        ) : null}
      </section>

      {state.kind === "ready" && state.data.is_admin ? (
        <CatalogueRefreshCard />
      ) : null}

      <EdgarIdentityCard heading="EDGAR identity (per-session)" />
    </div>
  );
}

function StatusGrid({ data }: { data: StatusResponse }): JSX.Element {
  const rows: Array<[string, string]> = [
    ["Version", data.version],
    ["Profile", data.deployment_profile],
    ["Embedding provider", data.embedding_provider],
    ["Embedding model", data.embedding_model],
    ["Storage filings", data.storage_filings.toLocaleString()],
    ["Admin tier", data.is_admin ? "yes" : "no"],
    [
      "Persistent credentials",
      data.persist_provider_credentials ? "enabled" : "disabled",
    ],
  ];
  return (
    <dl className="grid grid-cols-1 gap-x-8 gap-y-3 sm:grid-cols-2">
      {rows.map(([label, value]) => (
        <div key={label} className="flex flex-col">
          <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
            {label}
          </dt>
          <dd className="text-sm text-slate-900">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
