// Citation views (functional).
//
// The streamed answer renders inline `[N]` markers as in-text anchor
// chips that jump to a per-citation card in the source panel. This suite
// pins that rendering contract — distinct from the security suite's
// "the source panel shows the citation chunk" assertion:
//
//   - each `[N]` becomes an <a> whose href targets `#citation-N` and
//     whose title carries the matched source's ticker/form/date;
//   - a marker with no matching citation renders as an "unmatched
//     citation" chip rather than crashing or fabricating a source;
//   - the source panel lists one card per citation in arrival order,
//     each anchored by its display index and carrying the accession,
//     section path, and text span.
//
// Plain-text rendering only — there is NO Markdown / HTML sink, so the
// chips are the sole place markers become interactive.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

vi.mock("next/navigation", async () => {
  const actual = await vi.importActual<Record<string, unknown>>(
    "next/navigation",
  );
  return {
    ...actual,
    usePathname: () => "/rag",
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      back: vi.fn(),
      refresh: vi.fn(),
      prefetch: vi.fn(),
    }),
  };
});

import RagPage from "@/app/(app)/rag/page";
import { emptyProvidersResponse, sseFrame, streamingResponse } from "../security/_sse-harness";

const originalFetch = globalThis.fetch;

const SAMPLE_PLAN = {
  raw_query: "Apple AI risk",
  detected_language: "en",
  query_en: "Apple AI risk",
  tickers: ["AAPL"],
  form_types: ["10-K"],
  date_range: null,
  intent: "lookup",
  suggested_answer_mode: "concise",
};

function planResponse(): Response {
  return new Response(
    JSON.stringify({ plan: SAMPLE_PLAN, provider: "openai", model: "gpt-test" }),
    { status: 200 },
  );
}

function citation(overrides: Record<string, unknown>): Record<string, unknown> {
  return {
    chunk_id: "c1",
    ticker: "AAPL",
    form_type: "10-K",
    filing_date: "2024-09-30",
    accession_number: "0000320193-24-000001",
    section_path: "Risk Factors",
    text_span: "AI risk discussion.",
    similarity: 0.9,
    display_index: 1,
    ...overrides,
  };
}

function finalWith(answer: string): string {
  return sseFrame("final", {
    answer,
    provider: "openai",
    model: "gpt-test",
    prompt_version: "v1",
    token_usage: { input_tokens: 10, output_tokens: 20, total_tokens: 30 },
    latency_seconds: 0.3,
    streamed: true,
    refused: false,
  });
}

beforeEach(() => {
  globalThis.fetch = vi.fn();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

async function planThenGenerate(
  user: ReturnType<typeof userEvent.setup>,
): Promise<void> {
  await user.type(screen.getByLabelText(/question/i), "Apple AI risk");
  await user.click(screen.getByRole("button", { name: /^Plan$/i }));
  await screen.findByText("AAPL");
  await user.click(screen.getByRole("button", { name: /generate answer/i }));
}

describe("RagPage — citation chip rendering", () => {
  it("links each in-text [N] marker to its source card with a descriptive title", async () => {
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/rag/plan") {
        return planResponse();
      }
      if (url === "/api/admin/providers/") {
        return emptyProvidersResponse();
      }
      return streamingResponse([
        sseFrame("delta", { text: "Per [1] and [2], Apple flags AI risk." }),
        sseFrame("citation", citation({ chunk_id: "c1", display_index: 1 })),
        sseFrame(
          "citation",
          citation({
            chunk_id: "c2",
            display_index: 2,
            section_path: "MD&A",
            accession_number: "0000320193-24-000002",
            text_span: "Second source span.",
          }),
        ),
        finalWith("Per [1] and [2], Apple flags AI risk."),
      ]);
    }) as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await planThenGenerate(user);

    await waitFor(() => {
      expect(screen.getByText(/Sources \(2\)/)).toBeInTheDocument();
    });

    // Each marker is an anchor jumping to the matching source card.
    const chip1 = screen.getByRole("link", { name: "[1]" });
    const chip2 = screen.getByRole("link", { name: "[2]" });
    expect(chip1).toHaveAttribute("href", "#citation-1");
    expect(chip2).toHaveAttribute("href", "#citation-2");
    // The matched source's metadata rides the title attribute.
    expect(chip1).toHaveAttribute("title", "AAPL 10-K 2024-09-30");

    // The source panel renders one card per citation, in arrival order,
    // each anchored by its display index.
    const sources = screen.getByRole("heading", { name: /Sources \(2\)/ })
      .parentElement as HTMLElement;
    const items = within(sources).getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveAttribute("id", "citation-1");
    expect(items[1]).toHaveAttribute("id", "citation-2");
    expect(within(items[1]!).getByText(/Second source span\./)).toBeInTheDocument();
    expect(within(items[1]!).getByText(/MD&A/)).toBeInTheDocument();
  });

  it("renders an unmatched marker as an inert 'unmatched citation' chip", async () => {
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url === "/api/admin/rag/plan") {
        return planResponse();
      }
      if (url === "/api/admin/providers/") {
        return emptyProvidersResponse();
      }
      return streamingResponse([
        // Only citation [1] arrives; the answer also references [9], which
        // has no backing source.
        sseFrame("delta", { text: "Grounded [1] but dangling [9]." }),
        sseFrame("citation", citation({ display_index: 1 })),
        finalWith("Grounded [1] but dangling [9]."),
      ]);
    }) as unknown as typeof fetch;

    render(<RagPage />);
    const user = userEvent.setup();
    await planThenGenerate(user);

    await waitFor(() => {
      expect(screen.getByText(/Sources \(1\)/)).toBeInTheDocument();
    });

    // The grounded marker resolves to its source; the dangling one is
    // labelled unmatched and resolves to no fabricated metadata.
    expect(screen.getByRole("link", { name: "[1]" })).toHaveAttribute(
      "title",
      "AAPL 10-K 2024-09-30",
    );
    expect(screen.getByRole("link", { name: "[9]" })).toHaveAttribute(
      "title",
      "unmatched citation",
    );
    // Only the single real source is listed — the [9] marker invented
    // nothing in the panel.
    const sources = screen.getByRole("heading", { name: /Sources \(1\)/ })
      .parentElement as HTMLElement;
    expect(within(sources).getAllByRole("listitem")).toHaveLength(1);
  });
});
