"use client";

// Ingest — submit an EDGAR ingestion task and follow its progress.
//
// Why polling, not WebSocket
// --------------------------
// The backend WebSocket `/ws/ingest/{task_id}` requires either an
// `X-API-Key` header or a first-message JSON auth carrying the API key.
// The browser holds NEITHER — both keys live in the Next.js
// server-side session map and are injected on every proxy hop. There is
// no browser path to reach `/ws/...` without leaking the API key into
// page JS. Polling `GET /api/ingest/tasks/{id}` through the existing
// admin proxy keeps the key server-side.
//
// The `_evict_stale_locked` sweep that `TaskManager.get_task` runs on
// every poll already serves the operator-UI use case.

import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type FormEvent,
  type JSX,
} from "react";

import {
  ApiError,
  cancelIngestTask,
  getIngestTask,
  submitIngestAdd,
  submitIngestBatch,
} from "@/lib/api";
import type { TaskStatusResponse } from "@/lib/api-types";

type FormState = {
  tickers: string;
  formTypes: string;
  count: string;
};

const INITIAL_FORM: FormState = {
  tickers: "",
  formTypes: "10-K",
  count: "1",
};

const TERMINAL_STATES = new Set([
  "completed",
  "failed",
  "cancelled",
]);

const POLL_INTERVAL_MS = 2000;

export default function IngestPage(): JSX.Element {
  const [form, setForm] = useState<FormState>(INITIAL_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [task, setTask] = useState<TaskStatusResponse | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const tickersId = useId();
  const formTypesId = useId();
  const countId = useId();

  const stopPolling = useCallback(() => {
    if (pollTimer.current !== null) {
      clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  // Polling effect — keyed on the active task id + status. Whenever a
  // non-terminal task is in state, schedule one poll. After each poll
  // resolves, `setTask` updates `task.status`; the effect re-runs and
  // either schedules the next poll or stops (terminal state).
  //
  // This is the canonical React 19 idiom for self-driven polling: no
  // recursive closures, no ref writes during render, and cleanup runs
  // automatically when the task changes or the component unmounts.
  const taskId = task?.task_id ?? null;
  const taskStatus = task?.status ?? null;
  useEffect(() => {
    if (taskId === null || taskStatus === null) {
      return;
    }
    if (TERMINAL_STATES.has(taskStatus)) {
      return;
    }
    let cancelled = false;
    pollTimer.current = setTimeout(() => {
      void (async () => {
        if (cancelled) {
          return;
        }
        try {
          const next = await getIngestTask(taskId);
          if (!cancelled) {
            setTask(next);
          }
        } catch (exc) {
          if (cancelled) {
            return;
          }
          // 404 here = task expired or this session lost ownership;
          // surface a terminal generic error and stop polling.
          const message =
            exc instanceof ApiError
              ? exc.message
              : "Lost contact with the task.";
          setTask((prev) =>
            prev !== null
              ? { ...prev, status: "failed", error: message }
              : null,
          );
        }
      })();
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      if (pollTimer.current !== null) {
        clearTimeout(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [taskId, taskStatus]);

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (submitting) {
        return;
      }
      stopPolling();
      setSubmitting(true);
      setSubmitError(null);
      try {
        const tickers = form.tickers
          .split(/[\s,]+/)
          .map((t) => t.trim().toUpperCase())
          .filter((t) => t.length > 0);
        const formTypes = form.formTypes
          .split(/[\s,]+/)
          .map((t) => t.trim().toUpperCase())
          .filter((t) => t.length > 0);
        const count = Number.parseInt(form.count, 10);
        const body = {
          tickers,
          form_types: formTypes,
          count: Number.isFinite(count) && count > 0 ? count : 1,
        };
        const response =
          tickers.length === 1
            ? await submitIngestAdd(body)
            : await submitIngestBatch(body);
        setTask({
          task_id: response.task_id,
          status: response.status,
          tickers,
          form_types: formTypes,
          progress: {
            current_ticker: null,
            current_form_type: null,
            step_label: "queued",
            step_index: 0,
            step_total: 5,
            filings_done: 0,
            filings_total: 0,
            filings_skipped: 0,
            filings_failed: 0,
          },
          results: [],
          error: null,
          started_at: null,
          completed_at: null,
        });
        // The effect keyed on [taskId, taskStatus] takes over from here
        // and drives the polling loop until the task reaches a terminal
        // state.
      } catch (exc) {
        const message =
          exc instanceof ApiError
            ? `${exc.message}${exc.hint !== undefined ? ` ${exc.hint}` : ""}`
            : "Could not submit the ingest task.";
        setSubmitError(message);
      } finally {
        setSubmitting(false);
      }
    },
    [form.count, form.formTypes, form.tickers, stopPolling, submitting],
  );

  const handleCancel = useCallback(async () => {
    if (task === null || cancelling) {
      return;
    }
    setCancelling(true);
    try {
      await cancelIngestTask(task.task_id);
    } catch {
      // Cooperative cancel — the worker observes the cancel_event between
      // pipeline steps. Failures here are transient; we keep polling.
    } finally {
      setCancelling(false);
    }
  }, [cancelling, task]);

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Ingest</h1>
        <p className="text-sm text-slate-600">
          Submit an EDGAR ingestion task and watch progress. Requires an
          active EDGAR identity (register one on the Dashboard).
        </p>
      </header>

      <form
        onSubmit={(event) => {
          void handleSubmit(event);
        }}
        className="grid grid-cols-1 gap-4 rounded-lg border border-slate-200 bg-white p-6 shadow sm:grid-cols-3"
        aria-label="Ingest task submission"
      >
        <div className="sm:col-span-2 space-y-1">
          <label
            htmlFor={tickersId}
            className="block text-sm font-medium text-slate-700"
          >
            Tickers
          </label>
          <input
            id={tickersId}
            type="text"
            value={form.tickers}
            onChange={(event) => {
              setForm({ ...form, tickers: event.target.value });
            }}
            placeholder="AAPL, MSFT"
            required
            className="w-full rounded border border-slate-300 px-3 py-2 text-sm uppercase focus:border-slate-500 focus:outline-none"
          />
          <p className="text-xs text-slate-500">
            Comma- or whitespace-separated. One ticker uses the `/add`
            route; multiple use `/batch`.
          </p>
        </div>
        <div className="space-y-1">
          <label
            htmlFor={countId}
            className="block text-sm font-medium text-slate-700"
          >
            Count per (ticker, form)
          </label>
          <input
            id={countId}
            type="number"
            min={1}
            max={500}
            value={form.count}
            onChange={(event) => {
              setForm({ ...form, count: event.target.value });
            }}
            className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
          />
        </div>
        <div className="sm:col-span-3 space-y-1">
          <label
            htmlFor={formTypesId}
            className="block text-sm font-medium text-slate-700"
          >
            Form types
          </label>
          <input
            id={formTypesId}
            type="text"
            value={form.formTypes}
            onChange={(event) => {
              setForm({ ...form, formTypes: event.target.value });
            }}
            placeholder="10-K, 10-Q"
            required
            className="w-full rounded border border-slate-300 px-3 py-2 text-sm uppercase focus:border-slate-500 focus:outline-none"
          />
        </div>
        {submitError !== null ? (
          <p
            className="sm:col-span-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
            role="alert"
          >
            {submitError}
          </p>
        ) : null}
        <div className="sm:col-span-3 flex justify-end">
          <button
            type="submit"
            disabled={submitting}
            className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-50"
          >
            {submitting ? "Submitting…" : "Submit ingest"}
          </button>
        </div>
      </form>

      {task !== null ? (
        <TaskProgressCard
          task={task}
          cancelling={cancelling}
          onCancel={() => {
            void handleCancel();
          }}
        />
      ) : null}
    </div>
  );
}

function TaskProgressCard({
  task,
  cancelling,
  onCancel,
}: {
  task: TaskStatusResponse;
  cancelling: boolean;
  onCancel: () => void;
}): JSX.Element {
  const terminal = TERMINAL_STATES.has(task.status);
  return (
    <section
      aria-labelledby="task-progress-heading"
      className="rounded-lg border border-slate-200 bg-white p-6 shadow"
    >
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h2
          id="task-progress-heading"
          className="text-lg font-semibold tracking-tight"
        >
          Task {task.task_id.slice(0, 8)}…
        </h2>
        <span
          className={
            "rounded px-2 py-0.5 text-xs font-medium uppercase tracking-wide " +
            statusClass(task.status)
          }
        >
          {task.status}
        </span>
      </header>
      <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
        <Cell label="Step" value={task.progress.step_label || "—"} />
        <Cell
          label="Filings"
          value={`${task.progress.filings_done.toString()} / ${task.progress.filings_total.toString()}`}
        />
        <Cell
          label="Skipped"
          value={task.progress.filings_skipped.toString()}
        />
        <Cell label="Failed" value={task.progress.filings_failed.toString()} />
      </dl>
      {task.error !== null ? (
        <p
          className="mt-4 rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
          role="alert"
        >
          {task.error}
        </p>
      ) : null}
      {task.results.length > 0 ? (
        <details className="mt-4 text-sm">
          <summary className="cursor-pointer text-slate-700">
            Results ({task.results.length})
          </summary>
          <ul className="mt-2 space-y-1 text-xs text-slate-600">
            {task.results.map((result) => (
              <li key={result.accession_number} className="font-mono">
                {result.ticker} {result.form_type} {result.accession_number} —{" "}
                {result.chunk_count} chunk{result.chunk_count === 1 ? "" : "s"}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
      {!terminal ? (
        <div className="mt-4 flex justify-end">
          <button
            type="button"
            onClick={onCancel}
            disabled={cancelling}
            className="rounded border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-100 disabled:opacity-50"
          >
            {cancelling ? "Cancelling…" : "Cancel"}
          </button>
        </div>
      ) : null}
    </section>
  );
}

function Cell({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div className="flex flex-col">
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">
        {label}
      </dt>
      <dd className="text-sm text-slate-900">{value}</dd>
    </div>
  );
}

function statusClass(status: string): string {
  if (status === "completed") {
    return "bg-green-100 text-green-800";
  }
  if (status === "failed") {
    return "bg-red-100 text-red-800";
  }
  if (status === "cancelled" || status === "cancelling") {
    return "bg-amber-100 text-amber-800";
  }
  return "bg-slate-100 text-slate-700";
}
