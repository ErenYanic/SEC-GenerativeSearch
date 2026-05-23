"use client";

// One row in the Provider Settings list: add / validate / remove a key.
//
// The key value is held in local React state for the lifetime of the
// edit interaction only. As soon as the operator clicks "Save", the
// value is written to `sessionStorage` via `setProviderKey` and the
// local state is cleared so the cleartext is not retained in a closure
// that a later XSS could read.
//
// We never echo the key back into the DOM after save. The summary line
// shows only the masked tail (first / last four characters) — mirroring
// the backend's `mask_secret` audit-log shape but rendered in the
// browser, never round-tripped through the server.

import {
  useCallback,
  useId,
  useState,
  type FormEvent,
  type JSX,
} from "react";

import { ApiError, validateProvider } from "@/lib/api";
import {
  removeProviderKey,
  setProviderKey,
} from "@/lib/provider-keys";
import type { ProviderInfo } from "@/lib/api-types";

export interface ProviderKeyRowProps {
  provider: ProviderInfo;
  /** Current stored key, or `undefined` if none is set. */
  storedKey: string | undefined;
}

type Verdict = { kind: "idle" } | { kind: "ok" } | { kind: "rejected" };

export function ProviderKeyRow({
  provider,
  storedKey,
}: ProviderKeyRowProps): JSX.Element {
  const [editing, setEditing] = useState(false);
  const [draftKey, setDraftKey] = useState("");
  const [busy, setBusy] = useState<"idle" | "saving" | "validating">("idle");
  const [verdict, setVerdict] = useState<Verdict>({ kind: "idle" });
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fieldId = useId();

  const reset = useCallback((): void => {
    setEditing(false);
    setDraftKey("");
    setBusy("idle");
    setVerdict({ kind: "idle" });
    setErrorMessage(null);
  }, []);

  const handleSave = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (busy !== "idle") {
        return;
      }
      setBusy("saving");
      setErrorMessage(null);
      try {
        setProviderKey(provider.name, draftKey);
        // Wipe the cleartext copy from the closure immediately.
        setDraftKey("");
        setEditing(false);
      } catch (exc) {
        setErrorMessage(
          exc instanceof Error
            ? exc.message
            : "Could not save the provider key.",
        );
      } finally {
        setBusy("idle");
      }
    },
    [busy, draftKey, provider.name],
  );

  const handleValidate = useCallback(async () => {
    if (busy !== "idle") {
      return;
    }
    if (storedKey === undefined && draftKey === "") {
      setErrorMessage("Enter or save a key before validating.");
      return;
    }
    setBusy("validating");
    setVerdict({ kind: "idle" });
    setErrorMessage(null);
    try {
      const candidate = draftKey !== "" ? draftKey : storedKey;
      if (candidate === undefined) {
        return;
      }
      const result = await validateProvider({
        provider: provider.name,
        api_key: candidate,
        surface: provider.surface as "llm" | "embedding" | "reranker",
      });
      setVerdict({ kind: result.valid ? "ok" : "rejected" });
    } catch (exc) {
      // Surface the backend's canned message; never echo the candidate
      // key value back into the UI.
      if (exc instanceof ApiError) {
        setErrorMessage(
          exc.hint !== undefined && exc.hint !== ""
            ? `${exc.message} ${exc.hint}`
            : exc.message,
        );
      } else {
        setErrorMessage("Validation request failed — backend unreachable.");
      }
    } finally {
      setBusy("idle");
    }
  }, [busy, draftKey, provider.name, provider.surface, storedKey]);

  const handleRemove = useCallback(() => {
    try {
      removeProviderKey(provider.name);
    } catch {
      // Removal is best-effort; nothing to surface.
    }
    reset();
  }, [provider.name, reset]);

  const tail = storedKey !== undefined ? maskTail(storedKey) : null;

  return (
    <li className="flex flex-col gap-3 rounded border border-slate-200 bg-white p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="space-y-0.5">
          <p className="text-sm font-semibold text-slate-900">
            {provider.name}
          </p>
          <p className="text-xs uppercase tracking-wide text-slate-500">
            {provider.surface}
            {provider.supports_upstream_routing ? " · routing-capable" : ""}
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          {storedKey !== undefined ? (
            <span
              className="rounded bg-emerald-50 px-2 py-0.5 font-medium text-emerald-700"
              data-testid="provider-key-status"
            >
              Stored · {tail}
            </span>
          ) : (
            <span
              className="rounded bg-slate-100 px-2 py-0.5 font-medium text-slate-600"
              data-testid="provider-key-status"
            >
              No key
            </span>
          )}
          {verdict.kind === "ok" ? (
            <span
              className="rounded bg-emerald-50 px-2 py-0.5 font-medium text-emerald-700"
              role="status"
            >
              Validated
            </span>
          ) : null}
          {verdict.kind === "rejected" ? (
            <span
              className="rounded bg-amber-50 px-2 py-0.5 font-medium text-amber-700"
              role="status"
            >
              Rejected
            </span>
          ) : null}
        </div>
      </div>

      {editing ? (
        <form
          onSubmit={handleSave}
          className="space-y-2"
          aria-label={`Set ${provider.name} key`}
        >
          <label
            htmlFor={fieldId}
            className="block text-sm font-medium text-slate-700"
          >
            {provider.name} API key
          </label>
          <input
            id={fieldId}
            type="password"
            value={draftKey}
            onChange={(event) => {
              setDraftKey(event.target.value);
            }}
            autoComplete="off"
            spellCheck={false}
            required
            maxLength={4096}
            className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
          />
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="submit"
              disabled={busy !== "idle"}
              className="rounded bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {busy === "saving" ? "Saving…" : "Save"}
            </button>
            <button
              type="button"
              onClick={() => {
                setDraftKey("");
                setEditing(false);
                setErrorMessage(null);
              }}
              className="rounded border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100"
            >
              Cancel
            </button>
          </div>
        </form>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => {
              setEditing(true);
              setErrorMessage(null);
              setVerdict({ kind: "idle" });
            }}
            className="rounded border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100"
          >
            {storedKey !== undefined ? "Replace" : "Add key"}
          </button>
          <button
            type="button"
            onClick={() => {
              void handleValidate();
            }}
            disabled={busy !== "idle" || storedKey === undefined}
            className="rounded border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100 disabled:opacity-50"
          >
            {busy === "validating" ? "Validating…" : "Validate"}
          </button>
          {storedKey !== undefined ? (
            <button
              type="button"
              onClick={handleRemove}
              className="rounded border border-red-300 px-3 py-1.5 text-sm text-red-700 hover:bg-red-50"
            >
              Remove
            </button>
          ) : null}
        </div>
      )}

      {errorMessage !== null ? (
        <p
          className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
          role="alert"
        >
          {errorMessage}
        </p>
      ) : null}
    </li>
  );
}

// Show only the first and last four characters of a key, padded with
// dots in the middle. Mirrors the backend's `mask_secret` envelope so
// operators can correlate the audit log entry with the visible row.
// Values under eight characters are fully redacted, matching the
// `core/security.py::mask_secret` contract — never surface a short key.
function maskTail(value: string): string {
  if (value.length < 8) {
    return "•".repeat(8);
  }
  return `${value.slice(0, 4)}…${value.slice(-4)}`;
}
