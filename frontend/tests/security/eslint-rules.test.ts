// Assert the React XSS-hygiene lint rules are wired into eslint.config.mjs.
// Without this, a future drive-by config edit could silently drop the rule
// set and reopen the XSS sink class.

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const eslintConfigPath = path.resolve(__dirname, "../../eslint.config.mjs");
const eslintSource = readFileSync(eslintConfigPath, "utf-8");

describe("ESLint XSS-hygiene rules", () => {
  it("forbids dangerouslySetInnerHTML via react/no-danger", () => {
    expect(eslintSource).toContain(`"react/no-danger": "error"`);
  });

  it("forbids javascript: URLs in JSX via react/jsx-no-script-url", () => {
    expect(eslintSource).toContain(`"react/jsx-no-script-url": "error"`);
  });

  it("blocks eval()", () => {
    expect(eslintSource).toContain(`"no-eval": "error"`);
  });

  it("blocks implied eval (setTimeout('...'))", () => {
    expect(eslintSource).toContain(`"no-implied-eval": "error"`);
  });

  it("blocks new Function() factory", () => {
    expect(eslintSource).toContain(`"no-new-func": "error"`);
  });

  it("blocks javascript: URLs anywhere via no-script-url", () => {
    expect(eslintSource).toContain(`"no-script-url": "error"`);
  });

  it("bans localStorage via no-restricted-globals", () => {
    expect(eslintSource).toContain(`"no-restricted-globals"`);
    expect(eslintSource).toMatch(/name:\s*"localStorage"/);
  });

  it("bans window.localStorage via no-restricted-properties", () => {
    expect(eslintSource).toContain(`"no-restricted-properties"`);
    expect(eslintSource).toMatch(
      /object:\s*"window"[\s\S]+?property:\s*"localStorage"/,
    );
  });
});
