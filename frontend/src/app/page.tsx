import { redirect } from "next/navigation";

// The landing surface is the Dashboard — wrapped in `WelcomeGate` via the
// `(app)` route group's layout. Redirecting from `/` keeps the auth gate
// in one location and avoids rendering the operator console twice (once
// here and once under the route group).
export default function Page(): never {
  redirect("/dashboard");
}
