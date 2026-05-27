// Layout for every authenticated page. The route group `(app)` does not
// affect URLs — `/dashboard`, `/filings`, `/ingest` remain at the root —
// but co-locates the two gates + shell once.
//
// Auth tiers (outer → inner):
//   - `WelcomeGate` — operator tier. Validates `API_KEY` / `API_ADMIN_KEY`
//     against the backend and mints an HttpOnly admin-session cookie
//     (server-side store; the keys never reach the browser).
//   - `LoginGate` — user tier. Once user-tier is enabled
//     on the backend (SQLCipher + `API_AUTH_PEPPER` set), this gate
//     requires per-user username + password. The password derives a
//     KEK client-side and unlocks the per-user encrypted vault. When
//     user-tier is disabled at the backend, `LoginGate` is a no-op.
//   - `AppShell` — nav bar + sign-out around the page content.

import type { JSX, ReactNode } from "react";

import { AppShell } from "@/components/app-shell";
import { LoginGate } from "@/components/login-gate";
import { WelcomeGate } from "@/components/welcome-gate";

export default function AuthenticatedLayout({
  children,
}: {
  children: ReactNode;
}): JSX.Element {
  return (
    <WelcomeGate>
      <LoginGate>
        <AppShell>{children}</AppShell>
      </LoginGate>
    </WelcomeGate>
  );
}
