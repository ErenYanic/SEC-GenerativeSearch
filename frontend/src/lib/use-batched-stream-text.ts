"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";

// Coalesces high-frequency text chunks (SSE `delta` events, which arrive
// near token-by-token from the LLM SDKs) into at most one flush per
// animation frame, instead of one React state update per chunk. Shared by
// the chat and Ask pages, whose streaming handlers are otherwise identical.
//
// Ordering + cancellation are both preserved: `push` appends to an
// in-order buffer and `flush` always receives the buffer in arrival
// order; `reset` drops any buffered-but-unflushed text and cancels a
// pending frame, so a stale flush from a superseded/cancelled turn can
// never land on a later one. Call `reset` before starting a new stream
// and on cancel — the hook itself has no notion of "turn" or "cancel",
// callers own that.
export function useBatchedStreamText(onFlush: (batched: string) => void): {
  push: (chunk: string) => void;
  reset: () => void;
} {
  const bufferRef = useRef("");
  const frameRef = useRef<number | null>(null);
  const onFlushRef = useRef(onFlush);
  useEffect(() => {
    onFlushRef.current = onFlush;
  }, [onFlush]);

  const cancelPendingFrame = useCallback(() => {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
  }, []);

  const push = useCallback(
    (chunk: string) => {
      bufferRef.current += chunk;
      if (frameRef.current === null) {
        frameRef.current = requestAnimationFrame(() => {
          frameRef.current = null;
          const batched = bufferRef.current;
          bufferRef.current = "";
          if (batched.length > 0) {
            onFlushRef.current(batched);
          }
        });
      }
    },
    [],
  );

  const reset = useCallback(() => {
    cancelPendingFrame();
    bufferRef.current = "";
  }, [cancelPendingFrame]);

  // Unmount safety net — the page's own cleanup effect should already
  // call `reset`, but a pending frame must never fire into an unmounted
  // page's dispatcher.
  useEffect(() => cancelPendingFrame, [cancelPendingFrame]);

  // `push`/`reset` are individually stable (useCallback, empty/stable
  // deps); memoise the wrapper object too so callers can put it in a
  // dependency array without defeating their own memoisation.
  return useMemo(() => ({ push, reset }), [push, reset]);
}
