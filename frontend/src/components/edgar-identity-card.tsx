"use client";

// EDGAR identity registration card.
//
// Surfaces `POST /api/session/edgar` for B/C deployments where
// `API_EDGAR_SESSION_REQUIRED=true`. The backend validates name and
// email shape (control characters rejected) and the route NEVER echoes
// the offending value back; this component mirrors that contract by
// surfacing a field-agnostic notice on error rather than highlighting
// the user's input.
//
// Lifecycle
// ---------
// The identity is held in process memory on the backend, keyed by the
// browser's `session_id` cookie (minted at login by the admin session
// route). Clearing the form here also clears the on-server entry via
// `DELETE /api/session/edgar`.

import {
  useCallback,
  useId,
  useState,
  type FormEvent,
  type JSX,
} from "react";

import {
  ApiError,
  clearEdgarIdentity,
  registerEdgarIdentity,
} from "@/lib/api";

type CardState = "idle" | "submitting" | "registered" | "error";

export interface EdgarIdentityCardProps {
  /** Title rendered above the form. Defaults to a generic heading. */
  heading?: string;
  /** Called once the identity is successfully registered. */
  onRegistered?: () => void;
}

export function EdgarIdentityCard({
  heading = "EDGAR identity",
  onRegistered,
}: EdgarIdentityCardProps): JSX.Element {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [state, setState] = useState<CardState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const nameId = useId();
  const emailId = useId();

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (state === "submitting") {
        return;
      }
      setState("submitting");
      setErrorMessage(null);
      try {
        await registerEdgarIdentity({ name, email });
        setState("registered");
        // Clear local copies — the server now holds the identity, and we
        // do not want the values lingering in React state where a
        // future XSS could read them.
        setName("");
        setEmail("");
        onRegistered?.();
      } catch (exc) {
        setState("error");
        if (exc instanceof ApiError) {
          // The backend's `message` is canned and field-agnostic; the
          // optional `hint` is also field-agnostic per the schema
          // contract (`validate_edgar_name` / `validate_edgar_email`
          // never embed the offending value). Surface both verbatim.
          setErrorMessage(
            exc.hint !== undefined && exc.hint !== ""
              ? `${exc.message} ${exc.hint}`
              : exc.message,
          );
        } else {
          setErrorMessage("Network error — backend unreachable");
        }
      }
    },
    [email, name, onRegistered, state],
  );

  const handleClear = useCallback(async () => {
    try {
      await clearEdgarIdentity();
    } catch {
      // Idempotent on the backend; surface nothing on transient error.
    }
    setState("idle");
    setErrorMessage(null);
    setName("");
    setEmail("");
  }, []);

  return (
    <section
      className="rounded-lg border border-slate-200 bg-white p-6 shadow"
      aria-labelledby={`${nameId}-heading`}
    >
      <header className="space-y-1">
        <h2
          id={`${nameId}-heading`}
          className="text-lg font-semibold tracking-tight"
        >
          {heading}
        </h2>
        <p className="text-sm text-slate-600">
          Required by SEC EDGAR for every fetch. The name and email are
          held in process memory only and cleared on sign-out.
        </p>
      </header>

      {state === "registered" ? (
        <div className="mt-4 space-y-3">
          <p
            className="rounded border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-800"
            role="status"
          >
            EDGAR identity registered for this session.
          </p>
          <button
            type="button"
            onClick={() => {
              void handleClear();
            }}
            className="rounded border border-slate-300 px-3 py-1 text-sm text-slate-700 hover:bg-slate-100"
          >
            Clear identity
          </button>
        </div>
      ) : (
        <form
          onSubmit={(event) => {
            void handleSubmit(event);
          }}
          className="mt-4 space-y-4"
          aria-label="EDGAR identity registration"
        >
          <div className="space-y-1">
            <label
              htmlFor={nameId}
              className="block text-sm font-medium text-slate-700"
            >
              Full name
            </label>
            <input
              id={nameId}
              type="text"
              value={name}
              onChange={(event) => {
                setName(event.target.value);
              }}
              autoComplete="off"
              required
              maxLength={128}
              className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
            />
          </div>

          <div className="space-y-1">
            <label
              htmlFor={emailId}
              className="block text-sm font-medium text-slate-700"
            >
              Email
            </label>
            <input
              id={emailId}
              type="email"
              value={email}
              onChange={(event) => {
                setEmail(event.target.value);
              }}
              autoComplete="off"
              required
              maxLength={254}
              className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
            />
          </div>

          {errorMessage !== null ? (
            <p
              className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
              role="alert"
            >
              {errorMessage}
            </p>
          ) : null}

          <button
            type="submit"
            disabled={state === "submitting"}
            className="rounded bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {state === "submitting" ? "Registering…" : "Register"}
          </button>
        </form>
      )}
    </section>
  );
}
