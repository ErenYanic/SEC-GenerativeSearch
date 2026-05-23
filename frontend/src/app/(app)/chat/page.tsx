"use client";

// Chat — multi-turn RAG over the ingested filing index.
//
// The page mirrors the CLI's `sec-rag rag chat` semantics:
//
//   - Conversation history is kept **in tab memory only** (a React
//     useState array of `ChatTurn`). It is never written to
//     `sessionStorage` / `localStorage`, never persisted server-side, and
//     dies on tab close — matching the project's "chat history
//     persistence is out of scope by design" stance.
//   - Each user turn runs `POST /api/rag/plan` then `POST /api/rag/stream`
//     against the backend, with the prior turns supplied as `history`
//     in the stream body. The orchestrator strips retrieved-chunk +
//     citation content from prior turns at the route boundary (only
//     `{query, answer}` round-trip), so a follow-up never re-injects
//     the prior chunk text into the prompt.
//   - The first Escape (or Cancel button) while a stream is in flight
//     aborts via `AbortController` and discards the in-flight turn —
//     it is NOT committed to history. A second Escape while idle
//     navigates back via `history.back()` (the page-level equivalent
//     of the CLI's second-Ctrl-C → exit 130 contract). The turn-commit
//     happens only when the SSE `final` event arrives without
//     cancellation.
//
// Why a separate page (not folded into /rag): the chat surface owns a
// long-lived conversation state and a different keybinding contract
// (Escape vs. Ctrl-C); folding it into the single-shot Ask page would
// collapse two distinct UX flows.

import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type JSX,
  type KeyboardEvent,
} from "react";

import { ApiError, planRagQuery, streamRagAnswer } from "@/lib/api";
import type {
  CitationSchema,
  ConversationTurnSchema,
  QueryPlanSchema,
  RagStreamFinalPayload,
} from "@/lib/api-types";

// One committed conversation turn. Mirrors the wire-tier
// `ConversationTurnSchema` for the `{query, answer}` round-trip plus
// extra UI-only fields (citations, traceability) that never leave the
// tab.
interface ChatTurn {
  id: string;
  query: string;
  answer: string;
  citations: CitationSchema[];
  plan: QueryPlanSchema;
  provider: string;
  model: string;
  tokenTotal: number;
  refused: boolean;
}

type InFlight =
  | { kind: "idle" }
  | { kind: "planning"; query: string }
  | {
      kind: "streaming";
      query: string;
      plan: QueryPlanSchema;
      planProvider: string;
      planModel: string;
      answer: string;
      citations: CitationSchema[];
    }
  | {
      kind: "error";
      message: string;
      hint?: string;
      retryable: boolean;
      // Snapshot of the in-flight payload for a retry.
      query: string;
      plan: QueryPlanSchema | null;
    };

// History bound on the wire is 10 turns (matches `RAG_CHAT_HISTORY_TURNS`
// default + `ConversationTurnSchema.history` `max_length`).
const HISTORY_MAX_TURNS = 10;
// Mirrors the backend's `ConversationTurnSchema.answer` `max_length`. We
// trim before sending so a long answer does not collapse a follow-up
// request with a 422.
const HISTORY_ANSWER_MAX_CHARS = 4096;

export default function ChatPage(): JSX.Element {
  const [draft, setDraft] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [state, setState] = useState<InFlight>({ kind: "idle" });
  const abortRef = useRef<AbortController | null>(null);
  const draftId = useId();

  // Abort any in-flight stream on unmount — the SPA's only consumer
  // lives here while this page is mounted.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const inFlight = state.kind === "planning" || state.kind === "streaming";

  // Recent turns serialised for the wire. Truncated to the last
  // `HISTORY_MAX_TURNS` and answer field clipped to mirror the backend
  // bound — keeps the 64 KiB body cap honoured even after several
  // back-and-forths.
  const historyForWire: ConversationTurnSchema[] = useMemo(() => {
    const slice = turns.slice(-HISTORY_MAX_TURNS);
    return slice.map((turn) => ({
      query: turn.query,
      answer:
        turn.answer.length > HISTORY_ANSWER_MAX_CHARS
          ? turn.answer.slice(0, HISTORY_ANSWER_MAX_CHARS)
          : turn.answer,
    }));
  }, [turns]);

  const runTurn = useCallback(
    async (query: string) => {
      // Plan first; the backend rejects a raw query at /rag/stream by
      // contract (chat does not bypass the plan gate).
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setState({ kind: "planning", query });
      let planResponse: Awaited<ReturnType<typeof planRagQuery>>;
      try {
        planResponse = await planRagQuery({ query });
      } catch (exc) {
        if (controller.signal.aborted) {
          // Cancelled mid-plan — drop the turn silently.
          return;
        }
        const message =
          exc instanceof ApiError ? exc.message : "Could not plan the question.";
        const hint = exc instanceof ApiError ? exc.hint : undefined;
        setState({
          kind: "error",
          message,
          hint,
          retryable: false,
          query,
          plan: null,
        });
        return;
      }

      setState({
        kind: "streaming",
        query,
        plan: planResponse.plan,
        planProvider: planResponse.provider,
        planModel: planResponse.model,
        answer: "",
        citations: [],
      });

      let answer = "";
      const citations: CitationSchema[] = [];
      let final: RagStreamFinalPayload | null = null;
      let streamError: { message: string; hint?: string } | null = null;

      try {
        await streamRagAnswer(
          {
            plan: planResponse.plan,
            history: historyForWire,
          },
          {
            onDelta: (text) => {
              answer += text;
              setState((prev) => {
                if (prev.kind !== "streaming") {
                  return prev;
                }
                return { ...prev, answer };
              });
            },
            onCitation: (citation) => {
              citations.push(citation);
              setState((prev) => {
                if (prev.kind !== "streaming") {
                  return prev;
                }
                return { ...prev, citations: [...citations] };
              });
            },
            onFinal: (payload) => {
              final = payload;
            },
            onError: (event) => {
              streamError = { message: event.message, hint: event.hint };
            },
          },
          controller.signal,
        );
      } catch (exc) {
        if (controller.signal.aborted) {
          // First-Escape / Cancel — the turn is NOT committed to history.
          return;
        }
        if (exc instanceof ApiError) {
          setState({
            kind: "error",
            message: exc.message,
            hint: exc.hint,
            retryable: false,
            query,
            plan: planResponse.plan,
          });
          return;
        }
        setState({
          kind: "error",
          message: "Lost contact with the answer stream.",
          retryable: true,
          query,
          plan: planResponse.plan,
        });
        return;
      }

      if (controller.signal.aborted) {
        return;
      }

      if (streamError !== null) {
        setState({
          kind: "error",
          message: (streamError as { message: string }).message,
          hint: (streamError as { hint?: string }).hint,
          retryable: true,
          query,
          plan: planResponse.plan,
        });
        return;
      }

      // Commit the turn ONLY on a clean final event. A cancelled or
      // error path returns above without touching `turns`, mirroring
      // the CLI's "cancelled turns are never committed to history"
      // contract.
      const settled: ChatTurn = {
        id:
          typeof crypto !== "undefined" && "randomUUID" in crypto
            ? crypto.randomUUID()
            : `${Date.now().toString()}-${Math.random().toString(36).slice(2)}`,
        query,
        answer: final !== null ? (final as RagStreamFinalPayload).answer : answer,
        citations,
        plan: planResponse.plan,
        provider:
          final !== null
            ? (final as RagStreamFinalPayload).provider
            : planResponse.provider,
        model:
          final !== null
            ? (final as RagStreamFinalPayload).model
            : planResponse.model,
        tokenTotal:
          final !== null
            ? (final as RagStreamFinalPayload).token_usage.total_tokens
            : 0,
        refused: final !== null ? (final as RagStreamFinalPayload).refused : false,
      };
      setTurns((prev) => [...prev, settled]);
      setState({ kind: "idle" });
      abortRef.current = null;
    },
    [historyForWire],
  );

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const trimmed = draft.trim();
      if (trimmed.length === 0 || inFlight) {
        return;
      }
      setDraft("");
      void runTurn(trimmed);
    },
    [draft, inFlight, runTurn],
  );

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState({ kind: "idle" });
  }, []);

  const handleClear = useCallback(() => {
    setTurns([]);
    setState({ kind: "idle" });
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const handleRetry = useCallback(() => {
    if (state.kind !== "error") {
      return;
    }
    void runTurn(state.query);
  }, [state, runTurn]);

  // Keybinding: Escape mirrors the CLI's Ctrl-C contract.
  //   - First Escape while streaming → cancel the in-flight turn.
  //   - Escape while idle → navigate back (the page-level equivalent of
  //     the CLI's second-Ctrl-C → exit 130).
  const onKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key !== "Escape") {
        return;
      }
      if (inFlight) {
        event.preventDefault();
        handleCancel();
        return;
      }
      if (state.kind === "idle") {
        event.preventDefault();
        // Best-effort: if the user landed here via an external link
        // `history.back()` falls through; we never rely on it as the
        // only exit affordance — the AppShell nav remains visible.
        if (typeof window !== "undefined") {
          window.history.back();
        }
      }
    },
    [inFlight, state.kind, handleCancel],
  );

  return (
    <div className="space-y-6" onKeyDown={onKeyDown}>
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">Chat</h1>
          <p className="text-sm text-slate-600">
            Multi-turn conversation over the ingested SEC filings.
            History lives in this tab only — closing the tab or pressing
            Escape twice clears it. Each turn re-retrieves fresh chunks.
          </p>
        </div>
        <button
          type="button"
          onClick={handleClear}
          disabled={turns.length === 0 && state.kind === "idle"}
          className="rounded border border-slate-300 px-3 py-1 text-sm text-slate-700 hover:bg-slate-100 disabled:opacity-50"
        >
          Clear history
        </button>
      </header>

      <section
        aria-label="Conversation"
        className="space-y-4"
      >
        {turns.length === 0 && state.kind === "idle" ? (
          <p className="rounded border border-dashed border-slate-300 bg-white px-4 py-6 text-center text-sm text-slate-500">
            Ask anything about the filings you have ingested. Follow-up
            turns can reference earlier answers.
          </p>
        ) : null}
        <ol className="space-y-4">
          {turns.map((turn, index) => (
            <TurnCard key={turn.id} turn={turn} index={index + 1} />
          ))}
          {state.kind === "planning" || state.kind === "streaming" ? (
            <PendingTurn state={state} index={turns.length + 1} />
          ) : null}
        </ol>
        {state.kind === "error" ? (
          <ErrorPanel
            message={state.message}
            hint={state.hint}
            retryable={state.retryable}
            onRetry={handleRetry}
          />
        ) : null}
      </section>

      <form
        onSubmit={handleSubmit}
        aria-label="Send a chat message"
        className="space-y-3 rounded-lg border border-slate-200 bg-white p-4 shadow"
      >
        <label
          htmlFor={draftId}
          className="block text-sm font-medium text-slate-700"
        >
          Your message
        </label>
        <textarea
          id={draftId}
          value={draft}
          onChange={(event) => {
            setDraft(event.target.value);
          }}
          rows={3}
          maxLength={1024}
          placeholder="e.g. Summarise Apple's most recent AI risk disclosures."
          required
          className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
        />
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
          <span>
            History on the wire: {historyForWire.length.toString()} /{" "}
            {HISTORY_MAX_TURNS.toString()} turns. Press Escape to cancel
            the current turn; Escape again to leave the page.
          </span>
          <div className="flex items-center gap-2">
            {inFlight ? (
              <button
                type="button"
                onClick={handleCancel}
                className="rounded border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100"
              >
                Cancel
              </button>
            ) : null}
            <button
              type="submit"
              disabled={inFlight || draft.trim().length === 0}
              className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
            >
              {state.kind === "planning"
                ? "Planning…"
                : state.kind === "streaming"
                  ? "Streaming…"
                  : "Send"}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}

function TurnCard({
  turn,
  index,
}: {
  turn: ChatTurn;
  index: number;
}): JSX.Element {
  const indexById = useMemo(() => {
    const map = new Map<number, CitationSchema>();
    for (const citation of turn.citations) {
      if (citation.display_index > 0) {
        map.set(citation.display_index, citation);
      }
    }
    return map;
  }, [turn.citations]);

  return (
    <li
      className="rounded-lg border border-slate-200 bg-white shadow"
      aria-label={`Turn ${index.toString()}`}
    >
      <div className="border-b border-slate-100 px-4 py-3">
        <p className="text-xs uppercase tracking-wide text-slate-500">
          You
        </p>
        <p className="mt-1 whitespace-pre-wrap text-sm text-slate-800">
          {turn.query}
        </p>
      </div>
      <div className="px-4 py-3">
        <p className="text-xs uppercase tracking-wide text-slate-500">
          Assistant
        </p>
        {turn.refused ? (
          <p
            className="mt-1 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800"
            role="status"
          >
            No filings matched the plan; the orchestrator refused without
            calling the model.
          </p>
        ) : (
          <AnswerBody answer={turn.answer} citations={indexById} turnId={turn.id} />
        )}
        <p className="mt-2 text-xs text-slate-500">
          <span className="font-mono">{turn.provider}</span>
          {turn.model !== "" ? (
            <>
              {" / "}
              <span className="font-mono">{turn.model}</span>
            </>
          ) : null}
          {turn.tokenTotal > 0
            ? ` — ${turn.tokenTotal.toString()} tokens`
            : null}
        </p>
        {turn.citations.length > 0 ? (
          <SourcePanel citations={turn.citations} turnId={turn.id} />
        ) : null}
      </div>
    </li>
  );
}

function PendingTurn({
  state,
  index,
}: {
  state: Extract<InFlight, { kind: "planning" } | { kind: "streaming" }>;
  index: number;
}): JSX.Element {
  const answer = state.kind === "streaming" ? state.answer : "";
  const indexById = useMemo(() => {
    const map = new Map<number, CitationSchema>();
    if (state.kind === "streaming") {
      for (const citation of state.citations) {
        if (citation.display_index > 0) {
          map.set(citation.display_index, citation);
        }
      }
    }
    return map;
  }, [state]);

  return (
    <li
      className="rounded-lg border border-slate-200 bg-white shadow"
      aria-label={`Turn ${index.toString()} (pending)`}
    >
      <div className="border-b border-slate-100 px-4 py-3">
        <p className="text-xs uppercase tracking-wide text-slate-500">
          You
        </p>
        <p className="mt-1 whitespace-pre-wrap text-sm text-slate-800">
          {state.query}
        </p>
      </div>
      <div className="px-4 py-3">
        <p className="text-xs uppercase tracking-wide text-slate-500">
          Assistant
        </p>
        <p
          role="status"
          aria-live="polite"
          className="mt-1 text-xs uppercase tracking-wide text-slate-500"
        >
          {state.kind === "planning" ? "Planning…" : "Streaming…"}
        </p>
        {answer.length > 0 ? (
          <AnswerBody answer={answer} citations={indexById} turnId="pending" />
        ) : null}
      </div>
    </li>
  );
}

function ErrorPanel({
  message,
  hint,
  retryable,
  onRetry,
}: {
  message: string;
  hint?: string;
  retryable: boolean;
  onRetry: () => void;
}): JSX.Element {
  return (
    <div className="space-y-2 rounded-lg border border-red-200 bg-red-50 p-4">
      <p className="text-sm text-red-700" role="alert">
        {message}
        {hint !== undefined ? ` ${hint}` : ""}
      </p>
      {retryable ? (
        <div className="flex justify-end">
          <button
            type="button"
            onClick={onRetry}
            className="rounded border border-red-300 bg-white px-3 py-1.5 text-sm text-red-700 hover:bg-red-100"
          >
            Retry
          </button>
        </div>
      ) : null}
    </div>
  );
}

// Inline `[N]` markers turn into clickable chips anchored at the turn-
// scoped source panel below. Plain-text rendering only — same posture
// as the Ask page (no Markdown / HTML sink).
function AnswerBody({
  answer,
  citations,
  turnId,
}: {
  answer: string;
  citations: Map<number, CitationSchema>;
  turnId: string;
}): JSX.Element {
  const segments = useMemo(() => splitAnswerByCitation(answer), [answer]);
  return (
    <p className="mt-1 whitespace-pre-wrap text-sm leading-relaxed text-slate-800">
      {segments.map((segment, idx) => {
        if (segment.kind === "text") {
          return <span key={idx}>{segment.text}</span>;
        }
        const citation = citations.get(segment.index);
        const label = `[${segment.index.toString()}]`;
        const title =
          citation !== undefined
            ? `${citation.ticker} ${citation.form_type} ${citation.filing_date}`
            : "unmatched citation";
        return (
          <a
            key={idx}
            href={`#citation-${turnId}-${segment.index.toString()}`}
            title={title}
            className="mx-0.5 rounded bg-slate-100 px-1 py-0.5 font-mono text-xs text-slate-700 hover:bg-slate-200"
          >
            {label}
          </a>
        );
      })}
    </p>
  );
}

function SourcePanel({
  citations,
  turnId,
}: {
  citations: CitationSchema[];
  turnId: string;
}): JSX.Element {
  return (
    <details className="mt-3 rounded border border-slate-200 bg-slate-50">
      <summary className="cursor-pointer px-3 py-2 text-xs font-medium text-slate-700">
        Sources ({citations.length.toString()})
      </summary>
      <ol className="space-y-2 px-3 pb-3">
        {citations.map((citation, idx) => (
          <li
            key={`${citation.chunk_id}-${idx.toString()}`}
            id={`citation-${turnId}-${citation.display_index.toString()}`}
            className="rounded border border-slate-200 bg-white p-2 text-xs text-slate-700"
          >
            <p className="font-mono">
              [{citation.display_index.toString()}] {citation.ticker}{" "}
              {citation.form_type} {citation.filing_date}{" "}
              <span className="text-slate-500">
                {citation.accession_number}
              </span>
            </p>
            <p className="mt-1 text-slate-500">{citation.section_path}</p>
            <p className="mt-1 whitespace-pre-wrap text-slate-800">
              {citation.text_span}
            </p>
          </li>
        ))}
      </ol>
    </details>
  );
}

type Segment =
  | { kind: "text"; text: string }
  | { kind: "citation"; index: number };

const CITATION_RE = /\[(\d+)\]/g;

function splitAnswerByCitation(answer: string): Segment[] {
  const out: Segment[] = [];
  let cursor = 0;
  CITATION_RE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = CITATION_RE.exec(answer)) !== null) {
    if (match.index > cursor) {
      out.push({ kind: "text", text: answer.slice(cursor, match.index) });
    }
    const indexNum = Number.parseInt(match[1] ?? "0", 10);
    if (Number.isFinite(indexNum) && indexNum > 0) {
      out.push({ kind: "citation", index: indexNum });
    } else {
      out.push({ kind: "text", text: match[0] });
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < answer.length) {
    out.push({ kind: "text", text: answer.slice(cursor) });
  }
  return out;
}
