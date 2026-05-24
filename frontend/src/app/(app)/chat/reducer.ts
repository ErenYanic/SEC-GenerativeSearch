// Pure reducer for the chat page's in-flight state machine.
//
// Why a pure reducer (and not just `useState` calls):
//   - The state transitions are non-trivial (planning → streaming with
//     interim delta/citation events → terminal final OR mid-stream error).
//     A centralised reducer makes the transition table the load-bearing
//     artefact and the page itself a thin dispatcher.
//   - Pure-function transitions are unit-testable without a DOM, which
//     is the cheapest possible test surface for the contract.
//   - The reducer cannot ship effects (no `setTimeout`, no `fetch`); side
//     effects (network, abort) stay in the page where the AbortController
//     and the history committer already live. This matches the React 19
//     idiom: reducer for state, effects in the component.
//
// What it does NOT do:
//   - It does not own the commit-to-history step — that is a parent-
//     scope side effect gated on `state.kind === "streaming"` + a clean
//     `final` SSE event arriving. The reducer only models the in-flight
//     transition, not the persisted transcript.

import type { CitationSchema, QueryPlanSchema } from "@/lib/api-types";

export type InFlightState =
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
      query: string;
      plan: QueryPlanSchema | null;
    };

/**
 * Action vocabulary. Every action is a pure data record; nothing here
 * carries a function reference or a `Promise` so the reducer stays
 * trivially serialisable for debugging.
 */
export type InFlightAction =
  | { type: "START_PLAN"; query: string }
  | { type: "PLAN_FAILED"; message: string; hint?: string; query: string }
  | {
      type: "PLAN_OK";
      query: string;
      plan: QueryPlanSchema;
      planProvider: string;
      planModel: string;
    }
  | { type: "STREAM_DELTA"; text: string }
  | { type: "STREAM_CITATION"; citation: CitationSchema }
  | {
      type: "STREAM_ERROR";
      message: string;
      hint?: string;
      retryable: boolean;
    }
  | { type: "CANCEL" }
  | { type: "RESET" };

export const INITIAL_STATE: InFlightState = { kind: "idle" };

/**
 * Pure reducer. Unknown transitions return the state unchanged — the
 * page does not depend on the reducer raising for an out-of-order
 * action (e.g. a late `STREAM_DELTA` arriving after a `CANCEL` is
 * dropped silently, which is the correct behaviour for an async stream
 * the parent already torn down).
 */
export function reducer(
  state: InFlightState,
  action: InFlightAction,
): InFlightState {
  switch (action.type) {
    case "START_PLAN":
      return { kind: "planning", query: action.query };

    case "PLAN_FAILED":
      // Pre-stream 4xx — do-not-retry. The page reports the error and
      // surfaces no Retry button.
      return {
        kind: "error",
        message: action.message,
        hint: action.hint,
        retryable: false,
        query: action.query,
        plan: null,
      };

    case "PLAN_OK":
      // Transitioning to streaming clears any prior partial state.
      return {
        kind: "streaming",
        query: action.query,
        plan: action.plan,
        planProvider: action.planProvider,
        planModel: action.planModel,
        answer: "",
        citations: [],
      };

    case "STREAM_DELTA":
      if (state.kind !== "streaming") {
        // Late delta after cancel/error — drop silently.
        return state;
      }
      return { ...state, answer: state.answer + action.text };

    case "STREAM_CITATION":
      if (state.kind !== "streaming") {
        return state;
      }
      return { ...state, citations: [...state.citations, action.citation] };

    case "STREAM_ERROR":
      // In-stream `error` SSE event — maybe-retry. The plan is reused
      // so the parent can re-fire the stream without re-planning.
      if (state.kind !== "streaming") {
        // No stream open — ignore.
        return state;
      }
      return {
        kind: "error",
        message: action.message,
        hint: action.hint,
        retryable: action.retryable,
        query: state.query,
        plan: state.plan,
      };

    case "CANCEL":
    case "RESET":
      return INITIAL_STATE;

    default:
      return state;
  }
}
