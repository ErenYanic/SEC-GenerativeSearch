# SEC-GenerativeSearch — Frontend

Next.js 16 + TypeScript + Tailwind v4 SPA. The app starts with a minimal
scaffold and grows into the main product flows from there.

## Toolchain

- **Node**: 22 LTS (see `engines.node` in `package.json`).
- **Package manager**: pnpm 11.x, pinned via the `packageManager` field.
  Contributors with Corepack get the right pnpm version automatically:
  `corepack enable` then any `pnpm` command bootstraps `pnpm@11.2.2`.
  Without Corepack, install pnpm globally (`npm i -g pnpm@11.2.2`).

## Daily commands

```bash
pnpm install                 # one-off: materialise node_modules/
pnpm dev                     # start the dev server on :3000 (webpack)
pnpm test                    # run the Vitest suite (incl. security regression tests)
pnpm lint                    # run ESLint with the XSS-hygiene rule set
pnpm typecheck               # run tsc --noEmit
pnpm build                   # production build (Turbopack)
pnpm start                   # serve the production build on :3000
pnpm audit:ci                # mirror the CI gate locally
```

> **`pnpm dev` runs webpack, not Turbopack, by design.** Next.js 16.2.6's
> Turbopack dev mode drops middleware-set response headers during SSR — a
> known regression that breaks the strict-CSP boot we depend on. The
> `--webpack` flag opts the dev server into the proven runtime; production
> (`next build` / `next start`) uses Turbopack-compiled middleware and is
> unaffected. Revisit when Turbopack closes the gap.

## Security posture

The scaffold ships the load-bearing security controls from day one. The
load-bearing surface is:

- `middleware.ts` writes the per-request CSP nonce + full security-header
  set on every response.
- `src/lib/security-headers.ts` is the single source of truth for the CSP
  directives and the static header set. Mirror changes here with the
  backend `SecurityHeadersMiddleware`.
- `src/app/layout.tsx` installs a Trusted Types default policy via a
  nonced inline script before any hydration sink.
- `eslint.config.mjs` locks in React XSS-hygiene rules
  (`react/no-danger`, `no-eval`, `no-script-url`, …) — never drop one
  without rewriting `tests/security/eslint-rules.test.ts`.
- `tests/security/` is the regression net for every invariant above.
  CI runs them on every PR.

## What this scaffold does NOT do yet

- No application pages yet — only a minimal landing page.
- No `@/lib/api` HTTP client yet.
- No `sessionStorage` provider-key store yet.
