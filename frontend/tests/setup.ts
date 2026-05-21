import "@testing-library/jest-dom/vitest";

// happy-dom does not expose Web Crypto by default; install the Node
// global so `crypto.getRandomValues` works in security-header tests.
import { webcrypto } from "node:crypto";

if (!globalThis.crypto) {
  // @ts-expect-error — webcrypto type differs slightly from lib.dom.
  globalThis.crypto = webcrypto;
}
