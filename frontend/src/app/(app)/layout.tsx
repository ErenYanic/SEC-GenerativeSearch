// Layout for every authenticated page. The route group `(app)` does not
// affect URLs — `/dashboard`, `/filings`, `/ingest` remain at the root —
// but co-locates the gate + shell once.
//
// `WelcomeGate` checks the admin session via the server-side probe; while
// the probe is in flight, the gate renders a status message. Once
// authenticated, `AppShell` paints the navigation bar around the page
// content.

import type { JSX, ReactNode } from "react";

import { AppShell } from "@/components/app-shell";
import { WelcomeGate } from "@/components/welcome-gate";

export default function AuthenticatedLayout({
  children,
}: {
  children: ReactNode;
}): JSX.Element {
  return (
    <WelcomeGate>
      <AppShell>{children}</AppShell>
    </WelcomeGate>
  );
}
