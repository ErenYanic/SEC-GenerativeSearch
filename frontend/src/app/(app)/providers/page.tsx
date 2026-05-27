"use client";

// Lists every registered provider surfaced by `GET /api/providers/` and
// lets the operator add, validate, or remove a per-provider API key.
// Keys are held in browser `sessionStorage` only; they NEVER reach the
// Next.js server (the admin proxy holds only the operator's API and
// admin keys). Downstream RAG / search calls attach the keys as
// `X-Provider-Key-{provider}` headers via `apiFetchWithProviderKeys`.
//
// Browser-tier trust boundary
// ---------------------------
// `sessionStorage` is bounded by the lifetime of the browser tab. We
// deliberately do NOT use `localStorage` (would persist across browser
// restarts and tabs). The ESLint config bans direct `localStorage`
// access outside this module's allow-list; a regression test pins the
// rule in place.

import {
  useCallback,
  useEffect,
  useState,
  useSyncExternalStore,
  type JSX,
} from "react";

import { ProviderKeyRow } from "@/components/provider-key-row";
import { ApiError, listProviders } from "@/lib/api";
import type { ProviderInfo } from "@/lib/api-types";
import {
  clearProviderKeys,
  loadProviderKeys,
  subscribe as subscribeKeys,
  type ProviderKeyMap,
} from "@/lib/provider-keys";

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; providers: ProviderInfo[] }
  | { kind: "error"; message: string };

const EMPTY_KEY_MAP: ProviderKeyMap = Object.freeze({});

export default function ProviderSettingsPage(): JSX.Element {
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  // `useSyncExternalStore` keeps the page in lockstep with sessionStorage
  // mutations — including the ones triggered by sibling rows after a save
  // or remove. `getServerSnapshot` returns an immutable empty map so SSR
  // never reads from a window that does not exist (the file is `"use
  // client"` so React still rehydrates on the client).
  const storedKeys = useSyncExternalStore(
    subscribeKeys,
    loadProviderKeys,
    () => EMPTY_KEY_MAP,
  );

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const data = await listProviders();
        if (!cancelled) {
          setState({ kind: "ready", providers: data.providers });
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

  const [clearing, setClearing] = useState(false);

  const handleClearAll = useCallback(() => {
    if (clearing) {
      return;
    }
    setClearing(true);
    void clearProviderKeys()
      .catch(() => {
        // The vault upload may fail (e.g. session expired); the row
        // subscription keeps the UI in sync regardless, so we swallow
        // the rejection rather than surface a transient banner here.
      })
      .finally(() => {
        setClearing(false);
      });
  }, [clearing]);

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">
          Provider settings
        </h1>
        <p className="text-sm text-slate-600">
          Per-provider API keys are encrypted in your browser under your
          password and stored in your account vault. The server keeps only
          the ciphertext — it never sees your keys in cleartext.
        </p>
      </header>

      <section
        aria-labelledby="provider-list-heading"
        className="space-y-4"
      >
        <div className="flex items-center justify-between">
          <h2
            id="provider-list-heading"
            className="text-lg font-semibold tracking-tight"
          >
            Registered providers
          </h2>
          {Object.keys(storedKeys).length > 0 ? (
            <button
              type="button"
              onClick={handleClearAll}
              disabled={clearing}
              className="rounded border border-red-300 px-3 py-1 text-sm text-red-700 hover:bg-red-50 disabled:opacity-50"
            >
              {clearing ? "Clearing…" : "Clear all keys"}
            </button>
          ) : null}
        </div>

        {state.kind === "loading" ? (
          <p className="text-sm text-slate-500" role="status">
            Loading provider catalogue…
          </p>
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
          state.providers.length === 0 ? (
            <p className="text-sm text-slate-500">
              No providers are registered on this deployment.
            </p>
          ) : (
            <ul className="space-y-3">
              {state.providers.map((entry) => (
                <ProviderKeyRow
                  key={`${entry.name}::${entry.surface}`}
                  provider={entry}
                  storedKey={storedKeys[entry.name]}
                />
              ))}
            </ul>
          )
        ) : null}
      </section>
    </div>
  );
}
