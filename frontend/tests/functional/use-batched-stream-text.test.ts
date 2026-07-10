// `useBatchedStreamText` (`src/lib/use-batched-stream-text.ts`).
//
// Coalesces SSE `delta` chunks into at most one flush per animation
// frame, shared by the chat and Ask pages. The invariant under test is
// correctness, not timing: many small pushes must still concatenate, in
// order, into the full text once flushed — asserted via `waitFor` (no
// fake timers, no render-count assertion) so the test does not pin an
// animation-frame cadence.

import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useBatchedStreamText } from "@/lib/use-batched-stream-text";

describe("useBatchedStreamText", () => {
  it("coalesces many small pushes into fewer, in-order flushes", async () => {
    const flushed: string[] = [];
    const { result } = renderHook(() =>
      useBatchedStreamText((batched) => {
        flushed.push(batched);
      }),
    );

    const chunks = ["A", "p", "p", "l", "e", " ", "1", "0", "-", "K"];
    act(() => {
      for (const chunk of chunks) {
        result.current.push(chunk);
      }
    });

    await waitFor(() => {
      expect(flushed.join("")).toBe("Apple 10-K");
    });
    // Ten single-character pushes coalesced into (almost certainly) one
    // frame — strictly fewer flushes than pushes either way.
    expect(flushed.length).toBeLessThan(chunks.length);
  });

  it("assembles the correct text across multiple animation frames", async () => {
    const flushed: string[] = [];
    const { result } = renderHook(() =>
      useBatchedStreamText((batched) => {
        flushed.push(batched);
      }),
    );

    act(() => {
      result.current.push("first ");
    });
    await waitFor(() => {
      expect(flushed.join("")).toBe("first ");
    });

    act(() => {
      result.current.push("second");
    });
    await waitFor(() => {
      expect(flushed.join("")).toBe("first second");
    });
  });

  it("reset drops buffered-but-unflushed text and cancels the pending frame", async () => {
    const onFlush = vi.fn();
    const { result } = renderHook(() => useBatchedStreamText(onFlush));

    act(() => {
      result.current.push("discarded");
      result.current.reset();
    });

    // Give any (incorrectly still-scheduled) frame a chance to fire.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(onFlush).not.toHaveBeenCalled();

    // The hook is still usable after a reset.
    act(() => {
      result.current.push("kept");
    });
    await waitFor(() => {
      expect(onFlush).toHaveBeenCalledWith("kept");
    });
  });
});
