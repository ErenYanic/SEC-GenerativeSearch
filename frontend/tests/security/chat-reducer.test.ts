// Pure unit tests for the chat page's in-flight reducer
// (`src/app/(app)/chat/reducer.ts`).
//
// The reducer is the load-bearing transition table for the chat surface
// — pinning the contracts here as pure-function assertions catches a
// transition regression long before it surfaces as a flaky integration
// test on `chat-page.test.tsx`.

import { describe, expect, it } from "vitest";

import {
  INITIAL_STATE,
  reducer,
  type InFlightState,
} from "@/app/(app)/chat/reducer";
import type { CitationSchema, QueryPlanSchema } from "@/lib/api-types";

const PLAN: QueryPlanSchema = {
  raw_query: "Apple AI risk",
  detected_language: "en",
  query_en: "Apple AI risk",
  tickers: ["AAPL"],
  form_types: ["10-K"],
  date_range: null,
  intent: "lookup",
  suggested_answer_mode: "concise",
};

function streaming(
  override: Partial<Extract<InFlightState, { kind: "streaming" }>> = {},
): Extract<InFlightState, { kind: "streaming" }> {
  return {
    kind: "streaming",
    query: "q",
    plan: PLAN,
    planProvider: "openai",
    planModel: "gpt-test",
    answer: "",
    citations: [],
    ...override,
  };
}

const CITATION: CitationSchema = {
  chunk_id: "c1",
  ticker: "AAPL",
  form_type: "10-K",
  filing_date: "2024-09-30",
  accession_number: "0000320193-23-000077",
  section_path: "Risk Factors",
  text_span: "AI risk text.",
  similarity: 0.9,
  display_index: 1,
};

describe("chat reducer — initial state", () => {
  it("starts in idle", () => {
    expect(INITIAL_STATE).toEqual({ kind: "idle" });
  });
});

describe("chat reducer — happy-path transitions", () => {
  it("idle → planning on START_PLAN", () => {
    const next = reducer(INITIAL_STATE, { type: "START_PLAN", query: "topic" });
    expect(next).toEqual({ kind: "planning", query: "topic" });
  });

  it("planning → streaming on PLAN_OK; clears answer + citations", () => {
    const next = reducer(
      { kind: "planning", query: "topic" },
      {
        type: "PLAN_OK",
        query: "topic",
        plan: PLAN,
        planProvider: "openai",
        planModel: "gpt-test",
      },
    );
    expect(next.kind).toBe("streaming");
    if (next.kind !== "streaming") {
      throw new Error("unreachable");
    }
    expect(next.answer).toBe("");
    expect(next.citations).toEqual([]);
    expect(next.planProvider).toBe("openai");
  });

  it("STREAM_DELTA appends to answer string in order", () => {
    const a = reducer(streaming(), { type: "STREAM_DELTA", text: "Apple " });
    const b = reducer(a, { type: "STREAM_DELTA", text: "cited [1]" });
    expect(b.kind).toBe("streaming");
    if (b.kind !== "streaming") {
      throw new Error("unreachable");
    }
    expect(b.answer).toBe("Apple cited [1]");
  });

  it("STREAM_CITATION appends to citations array in order", () => {
    const next = reducer(streaming(), {
      type: "STREAM_CITATION",
      citation: CITATION,
    });
    if (next.kind !== "streaming") {
      throw new Error("unreachable");
    }
    expect(next.citations).toEqual([CITATION]);
  });
});

describe("chat reducer — failure transitions", () => {
  it("planning → error (do-not-retry) on PLAN_FAILED", () => {
    const next = reducer(
      { kind: "planning", query: "q" },
      {
        type: "PLAN_FAILED",
        query: "q",
        message: "auth failed",
        hint: "rotate key",
      },
    );
    expect(next).toEqual({
      kind: "error",
      message: "auth failed",
      hint: "rotate key",
      retryable: false,
      query: "q",
      plan: null,
    });
  });

  it("streaming → error (retryable) on STREAM_ERROR; reuses the plan", () => {
    const next = reducer(streaming(), {
      type: "STREAM_ERROR",
      message: "lost connection",
      retryable: true,
    });
    expect(next).toMatchObject({
      kind: "error",
      message: "lost connection",
      retryable: true,
      plan: PLAN,
    });
  });
});

describe("chat reducer — late-event idempotency (load-bearing)", () => {
  // The async stream can fire `onDelta` AFTER the user has cancelled or
  // an error has terminated the in-flight turn. The reducer MUST drop
  // these so the page does not flip back to streaming mid-cancel.

  it("STREAM_DELTA after CANCEL is dropped silently", () => {
    const cancelled = reducer(streaming({ answer: "partial " }), {
      type: "CANCEL",
    });
    const late = reducer(cancelled, { type: "STREAM_DELTA", text: "tail" });
    expect(late).toEqual(INITIAL_STATE);
  });

  it("STREAM_CITATION after STREAM_ERROR is dropped silently", () => {
    const errored = reducer(streaming(), {
      type: "STREAM_ERROR",
      message: "err",
      retryable: true,
    });
    const late = reducer(errored, {
      type: "STREAM_CITATION",
      citation: CITATION,
    });
    expect(late).toBe(errored);
  });

  it("STREAM_ERROR with no open stream is ignored", () => {
    const next = reducer(INITIAL_STATE, {
      type: "STREAM_ERROR",
      message: "ghost",
      retryable: false,
    });
    expect(next).toBe(INITIAL_STATE);
  });
});

describe("chat reducer — CANCEL / RESET", () => {
  it("CANCEL collapses any state back to idle", () => {
    expect(reducer(streaming(), { type: "CANCEL" })).toEqual(INITIAL_STATE);
    expect(
      reducer({ kind: "planning", query: "q" }, { type: "CANCEL" }),
    ).toEqual(INITIAL_STATE);
  });

  it("RESET collapses any state back to idle (including error)", () => {
    const errored: InFlightState = {
      kind: "error",
      message: "boom",
      retryable: true,
      query: "q",
      plan: PLAN,
    };
    expect(reducer(errored, { type: "RESET" })).toEqual(INITIAL_STATE);
  });
});

describe("chat reducer — defensive transitions", () => {
  // The reducer MUST treat the state object as immutable. Mutation
  // would let a late onDelta observer write into a previous render's
  // closure-captured state and leak streaming data into a settled
  // turn. Pinning identity ensures every successful transition
  // produces a brand-new object.
  it("STREAM_DELTA produces a new state object (not a mutation)", () => {
    const before = streaming({ answer: "x" });
    const after = reducer(before, { type: "STREAM_DELTA", text: "y" });
    expect(after).not.toBe(before);
    expect(before.kind === "streaming" && before.answer).toBe("x");
  });

  it("late events that drop silently return the SAME reference", () => {
    // For dropped events (post-cancel / wrong-state) the reducer
    // returns `state` as-is. React relies on reference equality to
    // short-circuit a render; pinning this contract here means a
    // future change that accidentally returned `{...state}` would
    // surface in tests, not in a wasted React render burst.
    const idle = INITIAL_STATE;
    const after = reducer(idle, { type: "STREAM_DELTA", text: "ghost" });
    expect(after).toBe(idle);
  });

  it("PLAN_OK from an unexpected prior state still transitions (forward-only)", () => {
    // Defensive but deliberate: even if a stale plan response lands
    // after a CANCEL, taking it forward to streaming is harmless —
    // the page's AbortController has already discarded the network
    // body, so `STREAM_DELTA` follow-ups would be no-ops. The
    // reducer doesn't gate on `state.kind === "planning"`.
    const idle = INITIAL_STATE;
    const after = reducer(idle, {
      type: "PLAN_OK",
      query: "q",
      plan: PLAN,
      planProvider: "openai",
      planModel: "x",
    });
    expect(after.kind).toBe("streaming");
  });
});
