"use client";

// Enrolment companion page.
//
// The operator mints a single-use enrolment token via the admin user-
// management UI and shares the resulting `/enrol?token=…` link with the
// user out-of-band. The user opens it here, sets a password (twice),
// and the browser derives everything client-side before POSTing.
//
// Route placement
// ---------------
// `/enrol` lives OUTSIDE the `(app)` route group on purpose:
//   - It must NOT be wrapped by `LoginGate` — the user has no account
//     yet, so the login form would trap them before they could enrol.
//   - It IS wrapped (locally) by `WelcomeGate` so the deployment's
//     shared admin session exists; the enrol POST rides the admin proxy
//     which injects the server-held `X-API-Key`.
//
// Security contract
// -----------------
//   - The password is derived into PBKDF2/HKDF material client-side and
//     NEVER crosses the wire — only `auth_proof` + the ciphertext blob.
//   - A conspicuous "your password is the only key" warning renders
//     ABOVE the password field (the test net pins its position).
//   - The token rides the URL query string; it is single-use and signed,
//     so a replay after the first successful enrolment 409s server-side.

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  Suspense,
  useCallback,
  useId,
  useState,
  type FormEvent,
  type JSX,
} from "react";

import { WelcomeGate } from "@/components/welcome-gate";
import { ApiError } from "@/lib/api";
import { enrolUser } from "@/lib/user-vault";

const MIN_PASSWORD_LENGTH = 12;

type EnrolState =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "enrolled"; username: string }
  | { kind: "error"; message: string };

function EnrolForm(): JSX.Element {
  const searchParams = useSearchParams();
  const token = searchParams.get("token") ?? "";

  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [state, setState] = useState<EnrolState>({ kind: "idle" });

  const passwordId = useId();
  const confirmId = useId();

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (state.kind === "submitting") {
        return;
      }
      if (token === "") {
        setState({
          kind: "error",
          message: "Enrolment link is missing its token. Ask your operator to re-issue it.",
        });
        return;
      }
      if (password.length < MIN_PASSWORD_LENGTH) {
        setState({
          kind: "error",
          message: `Password must be at least ${MIN_PASSWORD_LENGTH} characters.`,
        });
        return;
      }
      if (password !== confirm) {
        setState({ kind: "error", message: "Passwords do not match." });
        return;
      }
      setState({ kind: "submitting" });
      // Capture + clear React state immediately. The derivation runs on
      // the captured const; the tab never re-snapshots the cleartext.
      const pwd = password;
      setPassword("");
      setConfirm("");
      try {
        const result = await enrolUser(token, pwd);
        setState({ kind: "enrolled", username: result.username });
      } catch (exc) {
        if (exc instanceof ApiError) {
          // Backend envelopes are field-agnostic: enrolment_token_invalid
          // (401), enrolment_already_completed / username_exists (409),
          // user_tier_disabled (503). Surface verbatim — never echo the
          // token or password back.
          setState({
            kind: "error",
            message:
              exc.hint !== undefined && exc.hint !== ""
                ? `${exc.message} ${exc.hint}`
                : exc.message,
          });
        } else {
          setState({
            kind: "error",
            message: "Network error — backend unreachable.",
          });
        }
      }
    },
    [confirm, password, state.kind, token],
  );

  if (state.kind === "enrolled") {
    return (
      <section
        className="w-full max-w-md space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow"
        aria-label="Enrolment complete"
      >
        <h1 className="text-lg font-semibold tracking-tight">
          Enrolment complete
        </h1>
        <p
          className="rounded border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-800"
          role="status"
        >
          Your account is ready. Sign in with your username and the password
          you just chose.
        </p>
        <Link
          href="/"
          className="inline-block rounded bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-800"
        >
          Go to sign-in
        </Link>
      </section>
    );
  }

  return (
    <form
      onSubmit={(event) => {
        void handleSubmit(event);
      }}
      className="w-full max-w-md space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow"
      aria-label="Complete enrolment"
    >
      <header className="space-y-1">
        <h1 className="text-lg font-semibold tracking-tight">
          Set your password
        </h1>
        <p className="text-sm text-slate-600">
          This password encrypts your provider keys and EDGAR identity. The
          server never sees it.
        </p>
      </header>

      {/* Conspicuous, load-bearing warning ABOVE the password field. */}
      <p
        className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm font-semibold text-amber-900"
        role="alert"
        data-testid="password-loss-warning"
      >
        Your password is the only key to your vault. If you lose it, your
        stored keys cannot be recovered — write it down somewhere safe.
      </p>

      <div className="space-y-1">
        <label
          htmlFor={passwordId}
          className="block text-sm font-medium text-slate-700"
        >
          Password
        </label>
        <input
          id={passwordId}
          type="password"
          value={password}
          onChange={(event) => {
            setPassword(event.target.value);
          }}
          autoComplete="new-password"
          required
          className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
        />
      </div>

      <div className="space-y-1">
        <label
          htmlFor={confirmId}
          className="block text-sm font-medium text-slate-700"
        >
          Confirm password
        </label>
        <input
          id={confirmId}
          type="password"
          value={confirm}
          onChange={(event) => {
            setConfirm(event.target.value);
          }}
          autoComplete="new-password"
          required
          className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
        />
      </div>

      {state.kind === "error" ? (
        <p
          className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
          role="alert"
        >
          {state.message}
        </p>
      ) : null}

      <button
        type="submit"
        disabled={state.kind === "submitting"}
        className="w-full rounded bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
      >
        {state.kind === "submitting" ? "Enrolling…" : "Complete enrolment"}
      </button>
    </form>
  );
}

export default function EnrolPage(): JSX.Element {
  return (
    <WelcomeGate>
      <main className="flex min-h-screen items-center justify-center px-6">
        <Suspense
          fallback={
            <p className="text-sm text-slate-500" role="status">
              Loading enrolment…
            </p>
          }
        >
          <EnrolForm />
        </Suspense>
      </main>
    </WelcomeGate>
  );
}
