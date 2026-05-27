// Storage discipline.
//
// Per-user provider keys + EDGAR identity now live in the in-memory
// vault snapshot (`user-vault.ts`) while the user is logged in; the
// ciphertext is persisted SERVER-side in `users.ciphertext_vault`. The
// browser never writes either secret to disk-backed storage. Pre-13.11
// the keys lived in `sessionStorage`; post-13.11 NO production source
// file touches `sessionStorage` OR `localStorage` at all. Two regression
// nets:
//
//   1. Static — scan every `src/` source file (excluding code comments)
//      for any `localStorage` / `sessionStorage` token. A drive-by edit
//      that introduces a browser-storage stash trips this immediately.
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

describe("no browser-storage seam (vault is in-memory + server)", () => {
  it("no production source file references `sessionStorage`", () => {
    const offenders: string[] = [];
    for (const file of walk(srcRoot)) {
      const raw = readFileSync(file, "utf-8");
      const code = stripComments(raw);
      if (/\bsessionStorage\b/.test(code)) {
        offenders.push(path.relative(repoRoot, file));
      }
    }
    expect(offenders).toEqual([]);
  });
});
