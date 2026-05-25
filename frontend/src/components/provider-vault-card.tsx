"use client";

// Surfaces the three observable states of `provider-keys.ts`:
//   - plain                          → "Enable high-security mode" form
//   - encrypted + locked             → "Unlock" passphrase prompt
//   - encrypted + unlocked           → status panel with Lock / Disable
//                                       buttons + remaining-key count
//
// The passphrase NEVER leaves this component: it is held in a React
// state ref for the duration of the input interaction, fed into
// `enableEncryption()` / `unlock()`, then wiped from state. Input
// boxes are `type="password"` + `autoComplete="off"` so the browser's
// own password manager does not stash it.

import {
  useCallback,
  useId,
  useState,
  useSyncExternalStore,
  type FormEvent,
  type JSX,
} from "react";

import {
  disableEncryption,
  enableEncryption,
  InvalidPassphraseError,
  isEncrypted,
  isUnlocked,
  lock,
  loadProviderKeys,
  subscribe as subscribeKeys,
  unlock,
  type ProviderKeyMap,
} from "@/lib/provider-keys";

const EMPTY_KEY_MAP: ProviderKeyMap = Object.freeze({});

// Loose lower bound so the operator can't pick a 4-character "x" the
// way they might pick an API key passphrase. We deliberately do NOT
// enforce a complexity rule — false promise of strength + bad UX.
const MIN_PASSPHRASE_LENGTH = 8;

type ViewState =
  | { kind: "plain" }
  | { kind: "locked" }
  | { kind: "unlocked"; count: number };

function viewState(snap: ProviderKeyMap): ViewState {
  if (!isEncrypted()) {
    return { kind: "plain" };
  }
  if (!isUnlocked()) {
    return { kind: "locked" };
  }
  return { kind: "unlocked", count: Object.keys(snap).length };
}

export function ProviderVaultCard(): JSX.Element {
  const snapshot = useSyncExternalStore(
    subscribeKeys,
    loadProviderKeys,
    () => EMPTY_KEY_MAP,
  );
  const state = viewState(snapshot);

  return (
    <section
      aria-labelledby="vault-heading"
      className="rounded border border-slate-200 bg-white p-4"
    >
      <header className="space-y-1">
        <h2
          id="vault-heading"
          className="text-base font-semibold tracking-tight"
        >
          High-security mode
        </h2>
        <p className="text-xs text-slate-500">
          Encrypt provider keys at rest with a per-tab passphrase.
          Browser extensions and storage-stealer malware see only
          ciphertext until you unlock.
        </p>
      </header>

      {state.kind === "plain" ? (
        <EnablePanel />
      ) : state.kind === "locked" ? (
        <UnlockPanel />
      ) : (
        <UnlockedPanel count={state.count} />
      )}
    </section>
  );
}

function EnablePanel(): JSX.Element {
  const passphraseId = useId();
  const confirmId = useId();
  const [passphrase, setPassphrase] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (busy) {
        return;
      }
      setErrorMessage(null);
      if (passphrase.length < MIN_PASSPHRASE_LENGTH) {
        setErrorMessage(
          `Passphrase must be at least ${MIN_PASSPHRASE_LENGTH} characters.`,
        );
        return;
      }
      if (passphrase !== confirm) {
        setErrorMessage("Passphrase confirmation does not match.");
        return;
      }
      setBusy(true);
      try {
        await enableEncryption(passphrase);
        // Wipe the passphrase from React state the moment the vault
        // is sealed; the derived CryptoKey lives in module memory now.
        setPassphrase("");
        setConfirm("");
      } catch (exc) {
        setErrorMessage(
          exc instanceof Error
            ? exc.message
            : "Could not enable high-security mode.",
        );
      } finally {
        setBusy(false);
      }
    },
    [busy, confirm, passphrase],
  );

  return (
    <form
      onSubmit={(event) => {
        void handleSubmit(event);
      }}
      className="mt-4 space-y-3"
      aria-label="Enable high-security mode"
    >
      <div className="space-y-1">
        <label
          htmlFor={passphraseId}
          className="block text-sm font-medium text-slate-700"
        >
          Passphrase
        </label>
        <input
          id={passphraseId}
          type="password"
          value={passphrase}
          onChange={(event) => {
            setPassphrase(event.target.value);
          }}
          autoComplete="off"
          spellCheck={false}
          required
          minLength={MIN_PASSPHRASE_LENGTH}
          maxLength={1024}
          className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
        />
      </div>
      <div className="space-y-1">
        <label
          htmlFor={confirmId}
          className="block text-sm font-medium text-slate-700"
        >
          Confirm passphrase
        </label>
        <input
          id={confirmId}
          type="password"
          value={confirm}
          onChange={(event) => {
            setConfirm(event.target.value);
          }}
          autoComplete="off"
          spellCheck={false}
          required
          minLength={MIN_PASSPHRASE_LENGTH}
          maxLength={1024}
          className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
        />
      </div>
      <button
        type="submit"
        disabled={busy}
        className="rounded bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
      >
        {busy ? "Sealing vault…" : "Enable high-security mode"}
      </button>
      {errorMessage !== null ? (
        <p
          className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
          role="alert"
        >
          {errorMessage}
        </p>
      ) : null}
    </form>
  );
}

function UnlockPanel(): JSX.Element {
  const fieldId = useId();
  const [passphrase, setPassphrase] = useState("");
  const [busy, setBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (busy) {
        return;
      }
      setBusy(true);
      setErrorMessage(null);
      try {
        await unlock(passphrase);
        setPassphrase("");
      } catch (exc) {
        // Map any decrypt failure to a single canned message; we do not
        // want to disclose whether the vault is malformed or the
        // passphrase is wrong.
        if (exc instanceof InvalidPassphraseError) {
          setErrorMessage(
            "Passphrase is incorrect. The vault stays sealed.",
          );
        } else {
          setErrorMessage(
            exc instanceof Error
              ? exc.message
              : "Could not unlock the vault.",
          );
        }
      } finally {
        setBusy(false);
      }
    },
    [busy, passphrase],
  );

  return (
    <form
      onSubmit={(event) => {
        void handleSubmit(event);
      }}
      className="mt-4 space-y-3"
      aria-label="Unlock provider-key vault"
    >
      <p className="text-sm text-slate-600">
        Vault sealed. Enter the passphrase to access stored keys.
      </p>
      <div className="space-y-1">
        <label
          htmlFor={fieldId}
          className="block text-sm font-medium text-slate-700"
        >
          Passphrase
        </label>
        <input
          id={fieldId}
          type="password"
          value={passphrase}
          onChange={(event) => {
            setPassphrase(event.target.value);
          }}
          autoComplete="off"
          spellCheck={false}
          required
          maxLength={1024}
          className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
        />
      </div>
      <button
        type="submit"
        disabled={busy}
        className="rounded bg-slate-900 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
      >
        {busy ? "Unlocking…" : "Unlock"}
      </button>
      {errorMessage !== null ? (
        <p
          className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
          role="alert"
        >
          {errorMessage}
        </p>
      ) : null}
    </form>
  );
}

function UnlockedPanel({ count }: { count: number }): JSX.Element {
  const [busy, setBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleLock = useCallback(() => {
    setErrorMessage(null);
    lock();
  }, []);

  const handleDisable = useCallback(async () => {
    if (busy) {
      return;
    }
    setBusy(true);
    setErrorMessage(null);
    try {
      await disableEncryption();
    } catch (exc) {
      setErrorMessage(
        exc instanceof Error
          ? exc.message
          : "Could not disable high-security mode.",
      );
    } finally {
      setBusy(false);
    }
  }, [busy]);

  return (
    <div className="mt-4 space-y-3">
      <p
        className="text-sm text-emerald-700"
        role="status"
        data-testid="vault-unlocked-status"
      >
        Vault unlocked · {count} {count === 1 ? "key" : "keys"} stored.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={handleLock}
          className="rounded border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100"
        >
          Lock now
        </button>
        <button
          type="button"
          onClick={() => {
            void handleDisable();
          }}
          disabled={busy}
          className="rounded border border-amber-300 px-3 py-1.5 text-sm text-amber-700 hover:bg-amber-50 disabled:opacity-50"
        >
          {busy ? "Disabling…" : "Disable high-security mode"}
        </button>
      </div>
      {errorMessage !== null ? (
        <p
          className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
          role="alert"
        >
          {errorMessage}
        </p>
      ) : null}
    </div>
  );
}
