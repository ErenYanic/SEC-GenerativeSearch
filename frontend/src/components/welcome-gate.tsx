"use client";

// Operator-facing login gate.
//
// Renders a two-field form (API key + admin key) when there is no admin
// session. Posts the keys to `/api/admin/session` which validates them and
// mints an HttpOnly cookie server-side. Once authenticated, children
// render. The component never reads or writes either key to anywhere the
// browser tab can read it back — the only place either key exists is in
// the HTTPS POST payload and the server-side session map.

import {
  useCallback,
  useEffect,
  useId,
  useState,
  type FormEvent,
  type JSX,
  type ReactNode,
} from "react";

type GateStatus = "loading" | "authenticated" | "unauthenticated";

interface ErrorEnvelope {
  error?: string;
  message?: string;
  hint?: string;
}

interface WelcomeGateProps {
  children: ReactNode;
}

export function WelcomeGate({ children }: WelcomeGateProps): JSX.Element {
  const [status, setStatus] = useState<GateStatus>("loading");
  const [apiKey, setApiKey] = useState("");
  const [adminKey, setAdminKey] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const apiKeyFieldId = useId();
  const adminKeyFieldId = useId();

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch("/api/admin/session", {
          method: "GET",
          credentials: "same-origin",
          cache: "no-store",
        });
        if (cancelled) {
          return;
        }
        if (!res.ok) {
          setStatus("unauthenticated");
          return;
        }
        const body = (await res.json()) as { authenticated?: boolean };
        setStatus(body.authenticated === true ? "authenticated" : "unauthenticated");
      } catch {
        if (!cancelled) {
          setStatus("unauthenticated");
        }
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
      try {
        const res = await fetch("/api/admin/session", {
          method: "POST",
          credentials: "same-origin",
          cache: "no-store",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: apiKey, admin_key: adminKey }),
        });
        if (res.ok) {
          // Wipe local copies as soon as we have the cookie. The keys live
          // exclusively in the server-side session map from now on.
          setApiKey("");
          setAdminKey("");
          setStatus("authenticated");
          return;
        }
        let payload: ErrorEnvelope = {};
        try {
          payload = (await res.json()) as ErrorEnvelope;
        } catch {
          // Body was not JSON — fall through to generic message.
        }
        setErrorMessage(payload.message ?? `Login failed (HTTP ${res.status})`);
      } catch {
        setErrorMessage("Network error — backend unreachable");
      } finally {
        setSubmitting(false);
      }
    },
    [adminKey, apiKey, submitting],
  );

  const handleLogout = useCallback(async () => {
    try {
      await fetch("/api/admin/session", {
        method: "DELETE",
        credentials: "same-origin",
        cache: "no-store",
      });
    } catch {
      // Ignore — the cookie is HttpOnly so we cannot clear it client-side
      // anyway. The server clears its session-map entry; on the next
      // load the GET probe will return authenticated=false.
    }
    setStatus("unauthenticated");
  }, []);

  if (status === "loading") {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-sm text-slate-500" role="status">
          Checking session…
        </p>
      </div>
    );
  }

  if (status === "authenticated") {
    return (
      <>
        <div className="flex justify-end px-6 pt-4">
          <button
            type="button"
            onClick={() => {
              void handleLogout();
            }}
            className="rounded border border-slate-300 px-3 py-1 text-sm text-slate-700 hover:bg-slate-100"
          >
            Sign out
          </button>
        </div>
        {children}
      </>
    );
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-6">
      <form
        onSubmit={(event) => {
          void handleSubmit(event);
        }}
        className="w-full max-w-md space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow"
        aria-label="Operator sign-in"
      >
        <header className="space-y-1">
          <h1 className="text-lg font-semibold tracking-tight">
            Operator sign-in
          </h1>
          <p className="text-sm text-slate-600">
            Provide the API and admin keys for this deployment. Keys are held
            server-side and never persisted in the browser.
          </p>
        </header>

        <div className="space-y-1">
          <label
            htmlFor={apiKeyFieldId}
            className="block text-sm font-medium text-slate-700"
          >
            API key
          </label>
          <input
            id={apiKeyFieldId}
            type="password"
            value={apiKey}
            onChange={(event) => {
              setApiKey(event.target.value);
            }}
            autoComplete="off"
            required
            className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
          />
        </div>

        <div className="space-y-1">
          <label
            htmlFor={adminKeyFieldId}
            className="block text-sm font-medium text-slate-700"
          >
            Admin key
          </label>
          <input
            id={adminKeyFieldId}
            type="password"
            value={adminKey}
            onChange={(event) => {
              setAdminKey(event.target.value);
            }}
            autoComplete="off"
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
      </form>
    </main>
  );
}
