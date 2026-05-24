// Skeleton primitive (`src/components/skeleton.tsx`).
//
// The shared loading placeholder used by Dashboard + Filings (Phase
// 13.7). The tests pin the ARIA contract — one screen-reader
// announcement per render, decorative bars hidden — so screen readers
// announce "loading…" once instead of speaking each individual bar.

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { Skeleton } from "@/components/skeleton";

describe("Skeleton — ARIA contract", () => {
  it("renders a single status region with aria-busy", () => {
    render(<Skeleton />);
    const region = screen.getByRole("status");
    expect(region.getAttribute("aria-busy")).toBe("true");
    expect(region.getAttribute("aria-live")).toBe("polite");
  });

  it("announces a default 'Loading…' label via .sr-only span", () => {
    render(<Skeleton />);
    // The label lives inside the status region; the visual bars are
    // aria-hidden so AT does not enumerate them.
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("renders a custom label when supplied", () => {
    render(<Skeleton label="Loading filings…" />);
    expect(screen.getByText("Loading filings…")).toBeInTheDocument();
  });

  it("hides the decorative bars from AT (aria-hidden=true)", () => {
    const { container } = render(<Skeleton rows={3} />);
    const bars = container.querySelectorAll('[aria-hidden="true"]');
    expect(bars.length).toBe(3);
  });

  it("renders the requested number of bars (default 3)", () => {
    const { container, rerender } = render(<Skeleton />);
    expect(container.querySelectorAll('[aria-hidden="true"]').length).toBe(3);
    rerender(<Skeleton rows={7} />);
    expect(container.querySelectorAll('[aria-hidden="true"]').length).toBe(7);
  });
});
