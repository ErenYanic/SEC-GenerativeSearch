"use client";

// User-tier login gate.
//
// Sits INSIDE `WelcomeGate` — operator authentication mints the admin
// session cookie first; this component then layers per-user vault
// unlock on top.
//
// Lifecycle
// ---------
//   - Mount: probe user-tier availability by hitting
//     `/api/auth/login-params?username=__probe__`. A 503
//     `user_tier_disabled` envelope means the deployment has no
//     SQLCipher pepper — the user tier is not in play; render children
//     unchanged (Scenario A without enrolled users).
//   - Locked: render the username + password form. Submission derives
//     `auth_proof` + KEK in the tab, POSTs `/api/auth/login`, decrypts
//     the vault, and hydrates the in-memory cache. The password React
//     state is wiped the moment `loginUser` resolves.
//   - Unlocked: render children. The unlocked status is derived from
//     the user-vault subscription at render time — no shadow state.
//
// What never crosses the wire
// ---------------------------
// The password is consumed once by `loginUser` (which feeds it into
// PBKDF2 client-side) and then dropped — both via the cleared React
// state AND by the function never holding a long-lived reference. The
// derived KEK is held by `user-vault.ts` only.

import Link from "next/link";
import {
  useCallback,
  useEffect,
  useId,
  useState,
  useSyncExternalStore,
  type FormEvent,
  type JSX,
  type ReactNode,
} from "react";

import { ApiError, loginParamsRequest } from "@/lib/api";
import {
  isUnlocked,
  loginUser,
  snapshot,
  subscribe,
} from "@/lib/user-vault";

// `unlocked` is derived from the vault subscription at render time,
// so this enum carries only the states that come from the probe.
type ProbeStatus = "probing" | "user_tier_disabled" | "locked";

interface LoginGateProps {
  children: ReactNode;
}

function getSnapshot(): boolean {
  return isUnlocked();
}

export function LoginGate({ children }: LoginGateProps): JSX.Element {
  // Subscribe to vault state so a successful login (or sign-out from
  // AppShell) re-renders without prop drilling. The bool comes from
  // `isUnlocked()` and flips synchronously with module-level state.
  const vaultUnlocked = useSyncExternalStore(
    (listener) => subscribe(() => listener()),
    getSnapshot,
    () => false,
  );

  const [probe, setProbe] = useState<ProbeStatus>("probing");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const usernameFieldId = useId();
  const passwordFieldId = useId();

  // ---------------------------------------------------------------------
  // Probe: figure out whether the user tier is even available.
  // ---------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        // Username is the literal "__probe__"; the backend returns a
        // deterministic decoy salt for unknown usernames, so a 200 is
        // the "tier enabled" signal regardless of any real user.
        await loginParamsRequest("__probe__");
        if (!cancelled) {
          setProbe("locked");
        }
      } catch (exc) {
        if (cancelled) {
          return;
        }
        if (exc instanceof ApiError && exc.status === 503) {
          setProbe("user_tier_disabled");
          return;
        }
        // Any other failure — surface as locked + carry an error so
        // the operator can see what is wrong. We refuse to silently
        // pass through on a network error because that would mask a
        // misconfigured deployment.
        setProbe("locked");
        setErrorMessage(
          exc instanceof ApiError
            ? exc.message
            : "Could not reach the backend.",
        );
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (submitting) {
        return;
      }
      setSubmitting(true);
      setErrorMessage(null);
      // Capture password into a local const, then clear React state
      // immediately so the React render tree never re-snapshots the
      // raw password value after this point. The local const is
      // garbage-collected at function exit.
      const pwd = password;
      setPassword("");
      try {
        await loginUser(username, pwd);
        // `useSyncExternalStore` picks up the vault unlock on the next
        // tick; we also clear the username field so a screenshot taken
        // on the dashboard does not leak it back into the gate state.
        setUsername("");
      } catch (exc) {
        if (exc instanceof ApiError) {
          // The backend's `login_refused` envelope is intentionally
          // opaque (no enumeration of unknown user vs wrong proof vs
          // locked). Surface it verbatim — never echo the username.
          setErrorMessage(
            exc.hint !== undefined && exc.hint !== ""
              ? `${exc.message} ${exc.hint}`
              : exc.message,
          );
        } else {
          setErrorMessage("Network error — backend unreachable");
        }
      } finally {
        setSubmitting(false);
      }
    },
    [password, submitting, username],
  );

  if (probe === "user_tier_disabled" || vaultUnlocked) {
    return <>{children}</>;
  }

  if (probe === "probing") {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-sm text-slate-500" role="status">
          Checking authentication…
        </p>
      </div>
    );
  }

  // probe === "locked"
  return (
    <main className="flex min-h-screen items-center justify-center px-6">
      <form
        onSubmit={(event) => {
          void handleSubmit(event);
        }}
        className="w-full max-w-md space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow"
        aria-label="User sign-in"
      >
        <header className="space-y-1">
          <h1 className="text-lg font-semibold tracking-tight">Sign in</h1>
          <p className="text-sm text-slate-600">
            Your password unlocks your provider keys + EDGAR identity in
            this tab. The server never sees the password.
          </p>
        </header>

        <div className="space-y-1">
          <label
            htmlFor={usernameFieldId}
            className="block text-sm font-medium text-slate-700"
          >
            Username
          </label>
          <input
            id={usernameFieldId}
            type="text"
            value={username}
            onChange={(event) => {
              setUsername(event.target.value);
            }}
            autoComplete="username"
            required
            maxLength={64}
            className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
          />
        </div>

        <div className="space-y-1">
          <label
            htmlFor={passwordFieldId}
            className="block text-sm font-medium text-slate-700"
          >
            Password
          </label>
          <input
            id={passwordFieldId}
            type="password"
            value={password}
            onChange={(event) => {
              setPassword(event.target.value);
            }}
            autoComplete="current-password"
            required
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
          disabled={submitting}
          className="w-full rounded bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
        >
          {submitting ? "Signing in…" : "Sign in"}
        </button>

        <p className="text-center text-xs text-slate-500">
          Have an enrolment link?{" "}
          <Link href="/enrol" className="underline hover:text-slate-700">
            Complete enrolment
          </Link>
        </p>
      </form>
    </main>
  );
}

// Re-export for tests so they can read the vault snapshot without
// reaching into the user-vault module.
export const _internals = { snapshot } as const;
