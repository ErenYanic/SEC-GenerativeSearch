"use client";

// Ask — RAG over the ingested filing index.
//
// The page binds two backend endpoints in sequence:
//   1. `POST /api/rag/plan` runs query-understanding and returns the
//      editable `QueryPlanSchema`. The chip UI renders the parsed
//      tickers, form types, date range, intent, and suggested mode so
//      the operator can confirm or edit before spending the generation
//      budget. The raw query travels in the body — never the URL — so
//      it does not land in proxy access logs.
//   2. `POST /api/rag/stream` streams the answer over Server-Sent
//      Events. `delta` events build the answer text; `citation` events
//      populate the source panel; `final` carries the assembled
//      payload + traceability; `error` surfaces in-stream failures
//      (maybe-retry). Pre-stream HTTP errors (400 / 401 / 5xx) raise
//      `ApiError` and are presented as do-not-retry.
//
// Why the split: the human-in-the-loop edit step is enforced at the
// API shape (generation routes accept a plan, not a raw query). The UI
// mirrors that — the operator approves the plan before generation
// fires.
//
// Why a single page (no separate "search" route): the RAG surface
// subsumes the legacy keyword search; the citations panel and the
// chunk-level provenance match the same data shape that `POST
// /api/search` would have returned, with the answer summary on top.

import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type JSX,
} from "react";

import {
  ApiError,
  listProviders,
  planRagQuery,
  streamRagAnswer,
  type RagStreamRequestBody,
} from "@/lib/api";
import type {
  AnswerMode,
  CitationSchema,
  ProviderInfo,
  QueryPlanSchema,
  RagStreamFinalPayload,
} from "@/lib/api-types";
import { ModelPicker, type ModelPickerValue } from "@/components/model-picker";
import { formatUsd } from "@/lib/format";

const ANSWER_MODES: AnswerMode[] = [
  "concise",
  "analytical",
  "extractive",
  "comparative",
];

// Advanced retrieval-tuning knobs. Held as raw input strings so an
// empty field means "use the deployment default" — only a non-empty,
// finitely-parsed value rides the request. Bounds are re-enforced by
// the backend (caps 0..50, over-fetch 1..10), so the inputs are UX
// convenience, not a trust boundary.
interface RetrievalTuning {
  maxPerSection: string;
  maxPerFiling: string;
  rerankOverFetch: string;
}

const EMPTY_TUNING: RetrievalTuning = {
  maxPerSection: "",
  maxPerFiling: "",
  rerankOverFetch: "",
};

type RetrievalTuningBody = Pick<
  RagStreamRequestBody,
  "max_per_section" | "max_per_filing" | "rerank_over_fetch_factor"
>;

function parseTuning(value: RetrievalTuning): RetrievalTuningBody {
  const out: RetrievalTuningBody = {};
  const section = Number.parseInt(value.maxPerSection, 10);
  if (value.maxPerSection.trim() !== "" && Number.isFinite(section)) {
    out.max_per_section = section;
  }
  const filing = Number.parseInt(value.maxPerFiling, 10);
  if (value.maxPerFiling.trim() !== "" && Number.isFinite(filing)) {
    out.max_per_filing = filing;
  }
  const factor = Number.parseInt(value.rerankOverFetch, 10);
  if (value.rerankOverFetch.trim() !== "" && Number.isFinite(factor)) {
    out.rerank_over_fetch_factor = factor;
  }
  return out;
}

type GenerationStreamState =
  | { kind: "idle" }
  | { kind: "streaming"; answer: string; heartbeat: number }
  | {
      kind: "done";
      answer: string;
      final: RagStreamFinalPayload;
    }
  | {
      kind: "error";
      answer: string;
      error: { error: string; message: string; hint?: string };
      retryable: boolean;
    };

interface PlanError {
  message: string;
  hint?: string;
}

export default function RagPage(): JSX.Element {
  const [query, setQuery] = useState("");
  const [planning, setPlanning] = useState(false);
  const [planError, setPlanError] = useState<PlanError | null>(null);
  const [plan, setPlan] = useState<QueryPlanSchema | null>(null);
  const [planMeta, setPlanMeta] = useState<{
    provider: string;
    model: string;
  } | null>(null);
  const [modeOverride, setModeOverride] = useState<AnswerMode | null>(null);
  const [generation, setGeneration] = useState<GenerationStreamState>({
    kind: "idle",
  });
  const [citations, setCitations] = useState<CitationSchema[]>([]);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [pickerValue, setPickerValue] = useState<ModelPickerValue>({
    provider: "",
    model: "",
  });
  const [tuning, setTuning] = useState<RetrievalTuning>(EMPTY_TUNING);
  const abortRef = useRef<AbortController | null>(null);

  const queryId = useId();

  // Cancel any in-flight stream when the page unmounts to avoid
  // dangling readers (the SPA's only consumer of an SSE stream lives
  // here).
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  // Load the provider catalogue once. Failure is non-fatal: the picker
  // simply offers the empty `(default)` provider option and the request
  // falls back to settings.llm.default_provider on the backend. The
  // catalogue carries `supports_upstream_routing` so the ModelPicker
  // can gate the OpenRouter routing UI client-side. We defend against a
  // malformed payload (no `providers` array) so a test or proxy quirk
  // never collapses the page render.
  useEffect(() => {
    let cancelled = false;
    void listProviders()
      .then((response) => {
        if (!cancelled && Array.isArray(response?.providers)) {
          setProviders(response.providers);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const handlePlanSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (planning || query.trim().length === 0) {
        return;
      }
      // Cancel any prior in-flight generation — re-planning supersedes
      // the previous answer.
      abortRef.current?.abort();
      abortRef.current = null;
      setPlanning(true);
      setPlanError(null);
      setPlan(null);
      setPlanMeta(null);
      setCitations([]);
      setGeneration({ kind: "idle" });
      try {
        const response = await planRagQuery({
          query: query.trim(),
          // Plan + generate share the same provider so the audit log
          // ties them together; `routing_hints` is intentionally NOT
          // sent on /plan — the CLI doesn't either, and OpenRouter's
          // routing block has no semantics for the query-understanding
          // request shape.
          ...(pickerValue.provider !== ""
            ? { provider: pickerValue.provider }
            : {}),
          ...(pickerValue.model !== "" ? { model: pickerValue.model } : {}),
        });
        setPlan(response.plan);
        setPlanMeta({ provider: response.provider, model: response.model });
        setModeOverride(response.plan.suggested_answer_mode);
      } catch (exc) {
        const message =
          exc instanceof ApiError ? exc.message : "Could not plan the query.";
        const hint = exc instanceof ApiError ? exc.hint : undefined;
        setPlanError({ message, hint });
      } finally {
        setPlanning(false);
      }
    },
    [planning, query, pickerValue.provider, pickerValue.model],
  );

  const handleGenerate = useCallback(async () => {
    if (plan === null) {
      return;
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setCitations([]);
    setGeneration({ kind: "streaming", answer: "", heartbeat: 0 });
    let answerSoFar = "";
    try {
      await streamRagAnswer(
        {
          plan,
          mode: modeOverride ?? undefined,
          ...(pickerValue.provider !== ""
            ? { provider: pickerValue.provider }
            : {}),
          ...(pickerValue.model !== "" ? { model: pickerValue.model } : {}),
          ...(pickerValue.routing_hints !== undefined
            ? { routing_hints: pickerValue.routing_hints }
            : {}),
          ...parseTuning(tuning),
        },
        {
          onDelta: (text) => {
            answerSoFar += text;
            setGeneration((prev) => {
              if (prev.kind !== "streaming") {
                return prev;
              }
              return { ...prev, answer: answerSoFar };
            });
          },
          onCitation: (citation) => {
            setCitations((prev) => [...prev, citation]);
          },
          onFinal: (final) => {
            setGeneration({
              kind: "done",
              answer: final.answer,
              final,
            });
          },
          onError: (error) => {
            setGeneration({
              kind: "error",
              answer: answerSoFar,
              error,
              retryable: true,
            });
          },
          onHeartbeat: () => {
            setGeneration((prev) => {
              if (prev.kind !== "streaming") {
                return prev;
              }
              return { ...prev, heartbeat: prev.heartbeat + 1 };
            });
          },
        },
        controller.signal,
      );
    } catch (exc) {
      if (controller.signal.aborted) {
        // User-initiated abort — do not surface as an error.
        return;
      }
      if (exc instanceof ApiError) {
        setGeneration({
          kind: "error",
          answer: answerSoFar,
          error: {
            error: exc.code,
            message: exc.message,
            hint: exc.hint,
          },
          retryable: false,
        });
        return;
      }
      setGeneration({
        kind: "error",
        answer: answerSoFar,
        error: {
          error: "stream_failed",
          message: "Lost contact with the answer stream.",
        },
        retryable: true,
      });
    }
  }, [plan, modeOverride, pickerValue, tuning]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setGeneration({ kind: "idle" });
  }, []);

  const generating = generation.kind === "streaming";

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Ask</h1>
        <p className="text-sm text-slate-600">
          Pose a natural-language question over the ingested SEC filings.
          Review the parsed plan before generating an answer; sources
          appear as you stream.
        </p>
      </header>

      <form
        onSubmit={(event) => {
          void handlePlanSubmit(event);
        }}
        aria-label="Plan the RAG query"
        className="space-y-3 rounded-lg border border-slate-200 bg-white p-6 shadow"
      >
        <label
          htmlFor={queryId}
          className="block text-sm font-medium text-slate-700"
        >
          Question
        </label>
        <textarea
          id={queryId}
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
          }}
          rows={3}
          maxLength={1024}
          placeholder="e.g. How did Apple describe AI risk in its most recent 10-K?"
          required
          className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
        />
        <ModelPicker
          providers={providers}
          value={pickerValue}
          onChange={setPickerValue}
          disabled={planning || generating}
        />
        <RetrievalTuningControls
          value={tuning}
          onChange={setTuning}
          disabled={planning || generating}
        />
        {planError !== null ? (
          <p
            className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
            role="alert"
          >
            {planError.message}
            {planError.hint !== undefined ? ` ${planError.hint}` : ""}
          </p>
        ) : null}
        <div className="flex items-center justify-end gap-2">
          <button
            type="submit"
            disabled={planning || query.trim().length === 0}
            className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {planning ? "Planning…" : "Plan"}
          </button>
        </div>
      </form>

      {plan !== null && planMeta !== null ? (
        <PlanCard
          plan={plan}
          meta={planMeta}
          modeOverride={modeOverride}
          onModeChange={setModeOverride}
          onGenerate={() => {
            void handleGenerate();
          }}
          onCancel={handleCancel}
          generating={generating}
        />
      ) : null}

      {generation.kind !== "idle" ? (
        <AnswerCard
          state={generation}
          citations={citations}
          onRetry={() => {
            void handleGenerate();
          }}
        />
      ) : null}

      {citations.length > 0 ? <SourcePanel citations={citations} /> : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Retrieval tuning — advanced diversity + over-fetch controls (7.5.bis)
// ---------------------------------------------------------------------------

// Collapsed by default so the common path stays uncluttered. Each input
// is blank → "use deployment default"; a value rides the stream body and
// is re-bounded by the backend. The over-fetch knob is only meaningful
// once a reranker is wired on the backend, which the note calls out.
function RetrievalTuningControls({
  value,
  onChange,
  disabled,
}: {
  value: RetrievalTuning;
  onChange: (next: RetrievalTuning) => void;
  disabled: boolean;
}): JSX.Element {
  return (
    <details className="rounded border border-slate-200 bg-slate-50 p-3">
      <summary className="cursor-pointer text-xs font-medium uppercase tracking-wide text-slate-500">
        Retrieval tuning (advanced)
      </summary>
      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <TuningField
          label="Max per section"
          hint="0 disables"
          min={0}
          max={50}
          value={value.maxPerSection}
          disabled={disabled}
          onChange={(next) => {
            onChange({ ...value, maxPerSection: next });
          }}
        />
        <TuningField
          label="Max per filing"
          hint="0 disables"
          min={0}
          max={50}
          value={value.maxPerFiling}
          disabled={disabled}
          onChange={(next) => {
            onChange({ ...value, maxPerFiling: next });
          }}
        />
        <TuningField
          label="Rerank over-fetch"
          hint="1 disables"
          min={1}
          max={10}
          value={value.rerankOverFetch}
          disabled={disabled}
          onChange={(next) => {
            onChange({ ...value, rerankOverFetch: next });
          }}
        />
      </div>
      <p className="mt-2 text-xs text-slate-500">
        Leave blank to use the deployment defaults. Diversity caps bound
        how many chunks from one section or filing may appear; over-fetch
        only applies when a reranker is configured.
      </p>
    </details>
  );
}

function TuningField({
  label,
  hint,
  min,
  max,
  value,
  disabled,
  onChange,
}: {
  label: string;
  hint: string;
  min: number;
  max: number;
  value: string;
  disabled: boolean;
  onChange: (next: string) => void;
}): JSX.Element {
  const id = useId();
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-xs font-medium text-slate-600">
        {label}
      </label>
      <input
        id={id}
        type="number"
        inputMode="numeric"
        min={min}
        max={max}
        step={1}
        value={value}
        disabled={disabled}
        placeholder="default"
        onChange={(event) => {
          onChange(event.target.value);
        }}
        className="w-full rounded border border-slate-300 px-2 py-1 text-sm focus:border-slate-500 focus:outline-none disabled:opacity-50"
      />
      <span className="text-[11px] text-slate-400">{hint}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Plan card — editable chips + mode toggle
// ---------------------------------------------------------------------------

function PlanCard({
  plan,
  meta,
  modeOverride,
  onModeChange,
  onGenerate,
  onCancel,
  generating,
}: {
  plan: QueryPlanSchema;
  meta: { provider: string; model: string };
  modeOverride: AnswerMode | null;
  onModeChange: (mode: AnswerMode) => void;
  onGenerate: () => void;
  onCancel: () => void;
  generating: boolean;
}): JSX.Element {
  return (
    <section
      aria-labelledby="plan-heading"
      className="rounded-lg border border-slate-200 bg-white p-6 shadow space-y-4"
    >
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="plan-heading" className="text-lg font-semibold tracking-tight">
          Query plan
        </h2>
        <p className="text-xs text-slate-500">
          Planned by{" "}
          <span className="font-mono">{meta.provider}</span>
          {meta.model !== "" ? (
            <>
              {" / "}
              <span className="font-mono">{meta.model}</span>
            </>
          ) : null}
        </p>
      </header>

      <dl className="grid grid-cols-1 gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
        <Field label="Detected language">
          <span className="font-mono">{plan.detected_language}</span>
        </Field>
        <Field label="Intent">{plan.intent || "—"}</Field>
        <Field label="Tickers">
          <ChipList items={plan.tickers} empty="any" />
        </Field>
        <Field label="Form types">
          <ChipList items={plan.form_types} empty="any" />
        </Field>
        <Field label="Date range">
          {plan.date_range !== null ? (
            <span className="font-mono">
              {plan.date_range[0]} → {plan.date_range[1]}
            </span>
          ) : (
            <span className="text-slate-500">any</span>
          )}
        </Field>
        <Field label="Suggested mode">
          <span className="font-mono">{plan.suggested_answer_mode}</span>
        </Field>
      </dl>

      <fieldset className="space-y-2">
        <legend className="text-xs font-medium uppercase tracking-wide text-slate-500">
          Answer mode
        </legend>
        <div className="flex flex-wrap gap-2" role="radiogroup">
          {ANSWER_MODES.map((mode) => {
            const active = (modeOverride ?? plan.suggested_answer_mode) === mode;
            return (
              <button
                key={mode}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => {
                  onModeChange(mode);
                }}
                className={
                  "rounded-full border px-3 py-1 text-xs " +
                  (active
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-300 text-slate-700 hover:bg-slate-100")
                }
              >
                {mode}
              </button>
            );
          })}
        </div>
      </fieldset>

      <div className="flex justify-end gap-2">
        {generating ? (
          <button
            type="button"
            onClick={onCancel}
            className="rounded border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100"
          >
            Cancel
          </button>
        ) : null}
        <button
          type="button"
          onClick={onGenerate}
          disabled={generating}
          className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
        >
          {generating ? "Generating…" : "Generate answer"}
        </button>
      </div>
    </section>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <div className="flex flex-col gap-1">
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
        {label}
      </dt>
      <dd className="text-sm text-slate-900">{children}</dd>
    </div>
  );
}

function ChipList({
  items,
  empty,
}: {
  items: string[];
  empty: string;
}): JSX.Element {
  if (items.length === 0) {
    return <span className="text-slate-500">{empty}</span>;
  }
  return (
    <ul className="flex flex-wrap gap-1">
      {items.map((item) => (
        <li
          key={item}
          className="rounded-full bg-slate-100 px-2 py-0.5 font-mono text-xs text-slate-700"
        >
          {item}
        </li>
      ))}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// Answer card — streamed text with citation chip rendering
// ---------------------------------------------------------------------------

function AnswerCard({
  state,
  citations,
  onRetry,
}: {
  state: GenerationStreamState;
  citations: CitationSchema[];
  onRetry: () => void;
}): JSX.Element {
  const answer = useMemo(() => {
    if (state.kind === "streaming") {
      return state.answer;
    }
    if (state.kind === "done") {
      return state.answer;
    }
    if (state.kind === "error") {
      return state.answer;
    }
    return "";
  }, [state]);

  const refused = state.kind === "done" && state.final.refused;
  const indexById = useMemo(() => {
    const map = new Map<number, CitationSchema>();
    for (const citation of citations) {
      if (citation.display_index > 0) {
        map.set(citation.display_index, citation);
      }
    }
    return map;
  }, [citations]);

  return (
    <section
      aria-labelledby="answer-heading"
      className="rounded-lg border border-slate-200 bg-white p-6 shadow space-y-3"
    >
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="answer-heading" className="text-lg font-semibold tracking-tight">
          Answer
        </h2>
        {state.kind === "streaming" ? (
          <span
            role="status"
            className="text-xs uppercase tracking-wide text-slate-500"
          >
            Streaming…
          </span>
        ) : null}
        {state.kind === "done" ? (
          <span className="text-xs text-slate-500">
            <span className="font-mono">{state.final.provider}</span> /{" "}
            <span className="font-mono">{state.final.model}</span>
            {" — "}
            {state.final.token_usage.total_tokens.toString()} tokens
            {state.final.estimated_cost_usd !== undefined ? (
              <>
                {" — "}
                <span className="font-mono" title="Estimated cost — not a final bill">
                  {state.final.estimated_cost_usd !== null
                    ? `~${formatUsd(state.final.estimated_cost_usd)}`
                    : "—"}
                </span>
              </>
            ) : null}
            {" in "}
            {state.final.latency_seconds.toFixed(1)} s
          </span>
        ) : null}
      </header>

      {refused ? (
        <p
          className="rounded border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800"
          role="status"
        >
          No filings in the index matched this plan; the orchestrator
          refused without calling the model. Edit the chips, ingest
          more filings, or relax the date range.
        </p>
      ) : (
        // aria-live=polite + aria-atomic=false: screen readers track new
        // tokens as they stream in without re-announcing the whole answer
        // on every delta. Same shape as the chat surface's pending bubble.
        <div aria-live="polite" aria-atomic="false">
          <AnswerBody answer={answer} citations={indexById} />
        </div>
      )}

      {state.kind === "error" ? (
        <div className="space-y-2">
          <p
            className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
            role="alert"
          >
            {state.error.message}
            {state.error.hint !== undefined ? ` ${state.error.hint}` : ""}
          </p>
          {state.retryable ? (
            <div className="flex justify-end">
              <button
                type="button"
                onClick={onRetry}
                className="rounded border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100"
              >
                Retry
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

// `AnswerBody` splits the streamed answer on inline `[N]` markers and
// renders each as a citation chip linked back to the source panel.
// Plain-text rendering only — there is no Markdown / HTML sink to
// defend against; the strict CSP + Trusted Types posture would refuse
// anything richer anyway.
function AnswerBody({
  answer,
  citations,
}: {
  answer: string;
  citations: Map<number, CitationSchema>;
}): JSX.Element {
  const segments = useMemo(() => splitAnswerByCitation(answer), [answer]);
  return (
    <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-800">
      {segments.map((segment, index) => {
        if (segment.kind === "text") {
          return <span key={index}>{segment.text}</span>;
        }
        const citation = citations.get(segment.index);
        const label = `[${segment.index.toString()}]`;
        const title =
          citation !== undefined
            ? `${citation.ticker} ${citation.form_type} ${citation.filing_date}`
            : "unmatched citation";
        return (
          <a
            key={index}
            href={`#citation-${segment.index.toString()}`}
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

type Segment =
  | { kind: "text"; text: string }
  | { kind: "citation"; index: number };

const CITATION_RE = /\[(\d+)\]/g;

function splitAnswerByCitation(answer: string): Segment[] {
  const out: Segment[] = [];
  let cursor = 0;
  // `matchAll` is unavailable on older `lib.dom` targets we still
  // ship in tsconfig; the global-regex `exec` loop is portable.
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

// ---------------------------------------------------------------------------
// Source panel — one card per citation
// ---------------------------------------------------------------------------

function SourcePanel({
  citations,
}: {
  citations: CitationSchema[];
}): JSX.Element {
  return (
    <section
      aria-labelledby="sources-heading"
      className="rounded-lg border border-slate-200 bg-white p-6 shadow"
    >
      <h2
        id="sources-heading"
        className="text-lg font-semibold tracking-tight"
      >
        Sources ({citations.length.toString()})
      </h2>
      <ol className="mt-4 space-y-3">
        {citations.map((citation, index) => (
          <li
            key={`${citation.chunk_id}-${index.toString()}`}
            id={`citation-${citation.display_index.toString()}`}
            className="rounded border border-slate-200 bg-slate-50 p-3"
          >
            <header className="flex flex-wrap items-baseline justify-between gap-2 text-xs text-slate-600">
              <span className="font-mono">
                [{citation.display_index.toString()}]{" "}
                {citation.ticker} {citation.form_type}{" "}
                {citation.filing_date}
              </span>
              <span className="font-mono">{citation.accession_number}</span>
            </header>
            <p className="mt-1 text-xs text-slate-500">
              {citation.section_path}
            </p>
            <p className="mt-2 whitespace-pre-wrap text-sm text-slate-800">
              {citation.text_span}
            </p>
          </li>
        ))}
      </ol>
    </section>
  );
}
