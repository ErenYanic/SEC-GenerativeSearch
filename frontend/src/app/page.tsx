import type { JSX } from "react";

import { WelcomeGate } from "@/components/welcome-gate";

export default function Page(): JSX.Element {
  return (
    <WelcomeGate>
      <main className="mx-auto max-w-3xl px-6 py-16">
        <h1 className="text-3xl font-semibold tracking-tight">
          SEC-GenerativeSearch
        </h1>
        <p className="mt-4 text-slate-600">
          Operator console. The main app surfaces render here under the same
          authentication gate.
        </p>
      </main>
    </WelcomeGate>
  );
}
