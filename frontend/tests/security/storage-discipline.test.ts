// Storage discipline.
//
// The browser tab is the only place per-provider keys live in
// cleartext. `sessionStorage` closes when the tab closes; `localStorage`
// persists across browser restarts and across tabs and would
// dramatically widen the exfiltration window if an injected script ever
// reached the SPA. Two regression nets:
//
//   1. Static — scan every `src/` source file (excluding the sanctioned
//      `provider-keys.ts` and code comments) for any `localStorage` /
//      `window.localStorage` token. A drive-by edit that introduces a
//      stash trips this immediately.
//   2. Config — `eslint.config.mjs` ships `no-restricted-globals` and
//      `no-restricted-properties` rules banning `localStorage`.

import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const srcRoot = path.resolve(repoRoot, "src");

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stat = statSync(full);
    if (stat.isDirectory()) {
      out.push(...walk(full));
    } else if (/\.(ts|tsx|js|mjs)$/.test(entry)) {
      out.push(full);
    }
  }
  return out;
}

// Strip line and block comments so a "do NOT use localStorage" docstring
// does not trip the audit. Cheap regex pass is adequate; a real
// AST-based scan is overkill for a regression net.
function stripComments(src: string): string {
  let out = src.replace(/\/\*[\s\S]*?\*\//g, "");
  out = out.replace(/(^|[^:])\/\/.*$/gm, "$1");
  return out;
}

describe("localStorage ban", () => {
  it("no production source file references `localStorage` (provider keys live in sessionStorage)", () => {
    const offenders: string[] = [];
    for (const file of walk(srcRoot)) {
      const raw = readFileSync(file, "utf-8");
      const code = stripComments(raw);
      if (/\blocalStorage\b/.test(code)) {
        offenders.push(path.relative(repoRoot, file));
      }
    }
    expect(offenders).toEqual([]);
  });

  it("eslint.config.mjs declares no-restricted-globals against localStorage", () => {
    const cfg = readFileSync(
      path.resolve(repoRoot, "eslint.config.mjs"),
      "utf-8",
    );
    expect(cfg).toContain(`"no-restricted-globals"`);
    expect(cfg).toMatch(/name:\s*"localStorage"/);
  });

  it("eslint.config.mjs declares no-restricted-properties against window.localStorage", () => {
    const cfg = readFileSync(
      path.resolve(repoRoot, "eslint.config.mjs"),
      "utf-8",
    );
    expect(cfg).toContain(`"no-restricted-properties"`);
    expect(cfg).toMatch(/object:\s*"window"[\s\S]+?property:\s*"localStorage"/);
  });
});

describe("provider-keys is the sole sanctioned sessionStorage seam", () => {
  it("`sessionStorage` only appears in `src/lib/provider-keys.ts`", () => {
    const offenders: string[] = [];
    for (const file of walk(srcRoot)) {
      const raw = readFileSync(file, "utf-8");
      const code = stripComments(raw);
      if (/\bsessionStorage\b/.test(code)) {
        const rel = path.relative(repoRoot, file);
        if (rel !== path.join("src", "lib", "provider-keys.ts")) {
          offenders.push(rel);
        }
      }
    }
    expect(offenders).toEqual([]);
  });
});
