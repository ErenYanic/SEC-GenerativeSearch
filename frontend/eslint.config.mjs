// Flat-config consumer for ESLint 9 + Next.js 16. `eslint-config-next`
// already exports a flat-config array, so we spread it directly — the
// older `FlatCompat`/`extends` bridge tripped a circular-ref validation
// bug under the new Next config shape.

import nextConfig from "eslint-config-next";
import nextCoreWebVitals from "eslint-config-next/core-web-vitals";

const config = [
  ...nextConfig,
  ...nextCoreWebVitals,
  {
    // XSS-hygiene lock-in. These rules ship from day one so an injected
    // script cannot ride on a forgotten `dangerouslySetInnerHTML` or a
    // stray `eval`. Update `tests/security/eslint-rules.test.ts` in
    // lockstep with any change here — the regression test asserts the rule
    // set is present.
    rules: {
      "react/no-danger": "error",
      "react/jsx-no-script-url": "error",
      "react/jsx-no-target-blank": [
        "error",
        { allowReferrer: false, enforceDynamicLinks: "always" },
      ],
      "no-eval": "error",
      "no-implied-eval": "error",
      "no-new-func": "error",
      "no-script-url": "error",
    },
  },
  {
    // Tests load DOM globals via happy-dom and may use deliberate-XSS
    // fixtures to assert the sanitiser. The lint rules stay strict
    // outside this folder.
    files: ["tests/**/*.{ts,tsx}"],
    rules: {
      "react/no-danger": "off",
    },
  },
  {
    ignores: [".next/**", "node_modules/**", "out/**", "coverage/**"],
  },
];

export default config;
