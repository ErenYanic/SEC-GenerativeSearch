"use client";

// Application shell: top navigation bar shared by every authenticated page.
//
// `WelcomeGate` renders the shell only once the operator is authenticated,
// so this component is never visible to anonymous callers. Sign-out lives
// here (in the nav bar) rather than on `WelcomeGate` so unauthenticated
// pages do not carry an unnecessary button.

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useState, type JSX, type ReactNode } from "react";

import { clearProviderKeys } from "@/lib/provider-keys";

interface NavItem {
  href: string;
  label: string;
}

const NAV_ITEMS: NavItem[] = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/filings", label: "Filings" },
  { href: "/ingest", label: "Ingest" },
  { href: "/providers", label: "Providers" },
];

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps): JSX.Element {
  const pathname = usePathname();
  const [signingOut, setSigningOut] = useState(false);

  const handleSignOut = useCallback(async () => {
    if (signingOut) {
      return;
    }
    setSigningOut(true);
    // Drop every per-provider key in this tab before tearing down the
    // server session. `sessionStorage` would clear on tab close anyway,
    // but explicit clearing closes the window during the redirect.
    try {
      clearProviderKeys();
    } catch {
      // Storage may be unavailable in a degraded browser; sign-out
      // continues regardless.
    }
    try {
      await fetch("/api/admin/session", {
        method: "DELETE",
        credentials: "same-origin",
        cache: "no-store",
      });
    } catch {
      // Best-effort: the HttpOnly cookies will expire on their own.
    }
    // Reload to re-trigger the WelcomeGate probe; the server-side store
    // entry is already revoked.
    window.location.assign("/");
  }, [signingOut]);

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white">
        <nav
          aria-label="Primary"
          className="mx-auto flex max-w-6xl items-center justify-between gap-6 px-6 py-3"
        >
          <div className="flex items-center gap-6">
            <Link
              href="/dashboard"
              className="text-sm font-semibold tracking-tight text-slate-900"
            >
              SEC-GenerativeSearch
            </Link>
            <ul className="flex items-center gap-1">
              {NAV_ITEMS.map((item) => {
                const active =
                  pathname === item.href ||
                  pathname?.startsWith(`${item.href}/`);
                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      aria-current={active ? "page" : undefined}
                      className={
                        "rounded px-3 py-1 text-sm " +
                        (active === true
                          ? "bg-slate-900 text-white"
                          : "text-slate-700 hover:bg-slate-100")
                      }
                    >
                      {item.label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
          <button
            type="button"
            onClick={() => {
              void handleSignOut();
            }}
            disabled={signingOut}
            className="rounded border border-slate-300 px-3 py-1 text-sm text-slate-700 hover:bg-slate-100 disabled:opacity-50"
          >
            {signingOut ? "Signing out…" : "Sign out"}
          </button>
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
    </div>
  );
}
