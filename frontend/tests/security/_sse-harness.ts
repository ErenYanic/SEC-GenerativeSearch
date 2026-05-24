// Shared SSE / fetch-mock harness for the streaming-page test suite.
//
// The three primitives here used to live duplicated across
// `rag-page.test.tsx` and `chat-page.test.tsx`. Centralising them:
//
//   - keeps the on-the-wire SSE shape (`event: …\ndata: …\n\n`) authored
//     in exactly one place, so a stray drift away from RFC framing
//     surfaces in every test that uses the harness;
//   - lets the `pendingStreamingResponse` cancel-mid-flight helper grow
//     once with first-class typing rather than per-page;
//   - provides a `finalFrame` builder so the boilerplate-y
//     `provider/model/token_usage` payload for happy-path final events
//     is not re-derived per test.
//
// This module is test-only — never imported from `src/`.

/** Encode one SSE frame per RFC: `event: <name>\ndata: <json>\n\n`. */
export function sseFrame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * Construct a Response whose body emits the supplied frames in order
 * and closes. Each call builds a fresh `ReadableStream` — safe to call
 * inside a routing fetch mock that returns a new response per request.
 */
export function streamingResponse(frames: string[]): Response {
  const encoder = new TextEncoder();
  let i = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i >= frames.length) {
        controller.close();
        return;
      }
      controller.enqueue(encoder.encode(frames[i] as string));
      i += 1;
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

/**
 * A streaming Response the test can hold open and release frames into
 * on its own schedule. The returned `release` function pushes additional
 * frames into the body; subsequent `pull` requests deliver them. Used
 * for cancel-mid-flight tests where the timing of the next frame is the
 * point of the test.
 *
 * The body is consume-once — only one fetch call should land on this
 * response. Route the mock by URL so the parallel `listProviders()`
 * fetch (or any other on-mount call) does not eat it.
 */
export function pendingStreamingResponse(): {
  response: Response;
  release: (frames: string[]) => void;
} {
  let pull: ((frames: string[]) => void) | null = null;
  let queued: string[] = [];
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (queued.length > 0) {
        controller.enqueue(encoder.encode(queued.shift() as string));
        return;
      }
      pull = (frames) => {
        queued.push(...frames);
        pull = null;
        if (queued.length === 0) {
          controller.close();
        } else {
          controller.enqueue(encoder.encode(queued.shift() as string));
        }
      };
    },
  });
  const response = new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
  function release(frames: string[]): void {
    if (pull !== null) {
      pull(frames);
    } else {
      queued.push(...frames);
    }
  }
  return { response, release };
}

/**
 * Build a happy-path `final` SSE frame with the default provider /
 * model / token-usage payload most tests do not exercise. Use the
 * lower-level `sseFrame("final", …)` directly when a test needs to
 * pin a non-default value (e.g. an OpenRouter run).
 */
export function finalFrame(answer: string): string {
  return sseFrame("final", {
    answer,
    provider: "openai",
    model: "gpt-test",
    prompt_version: "v1",
    token_usage: { input_tokens: 1, output_tokens: 2, total_tokens: 3 },
    latency_seconds: 0.1,
    streamed: true,
    refused: false,
  });
}

/**
 * Mirror of `ProviderListResponse` for tests that mount a page whose
 * `useEffect(listProviders)` would otherwise race the consumable
 * `pendingStreamingResponse` body. Use as the `/api/admin/providers/`
 * branch of a routed fetch mock.
 */
export function emptyProvidersResponse(): Response {
  return new Response(JSON.stringify({ providers: [], total: 0 }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}
