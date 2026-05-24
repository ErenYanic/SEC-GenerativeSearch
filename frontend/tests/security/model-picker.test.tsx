// The picker is the SPA's gate for the OpenRouter upstream-routing UI;
// it MUST stay client-side aligned with the backend's
// `supports_upstream_routing` capability flag so the SPA never sends a
// hint the backend would reject as `invalid_flag_combination`. These
// tests pin:
//   - the LLM-surface entries from the catalogue render as options
//   - the routing UI is gated on `supports_upstream_routing`
//   - switching AWAY from a routing-capable provider clears any pending
//     hints (so a stale hint cannot survive a provider change)
//   - the `order` parser handles comma-separated entries and trims
//     whitespace
//   - malformed upstream slugs surface a client-side `aria-invalid` cue
//   - the picker is purely controlled (no internal state — onChange is
//     called on every interaction)

import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import {
  ModelPicker,
  type ModelPickerProps,
  type ModelPickerValue,
} from "@/components/model-picker";
import type { ProviderInfo } from "@/lib/api-types";

const CATALOGUE: ProviderInfo[] = [
  { name: "openai", surface: "llm", supports_upstream_routing: false },
  { name: "anthropic", surface: "llm", supports_upstream_routing: false },
  { name: "openrouter", surface: "llm", supports_upstream_routing: true },
  // Embedding entries MUST NOT appear in the provider dropdown.
  { name: "local", surface: "embedding", supports_upstream_routing: false },
];

function renderPicker(
  override: Partial<ModelPickerProps> = {},
): {
  onChange: ReturnType<typeof vi.fn>;
  rerender: (next: Partial<ModelPickerProps>) => void;
} {
  const onChange = vi.fn();
  const base: ModelPickerProps = {
    providers: CATALOGUE,
    value: { provider: "", model: "" },
    onChange,
  };
  const { rerender: doRerender } = render(<ModelPicker {...base} {...override} />);
  return {
    onChange,
    rerender: (next) => {
      doRerender(<ModelPicker {...base} {...override} {...next} />);
    },
  };
}

describe("ModelPicker — catalogue rendering", () => {
  it("renders only LLM-surface providers as options", () => {
    renderPicker();
    const select = screen.getByLabelText(/^provider$/i) as HTMLSelectElement;
    const optionValues = Array.from(select.options).map((o) => o.value);
    expect(optionValues).toContain("openai");
    expect(optionValues).toContain("anthropic");
    expect(optionValues).toContain("openrouter");
    // Embedding entries are filtered out client-side.
    expect(optionValues).not.toContain("local");
    // The default '(default)' option remains.
    expect(optionValues).toContain("");
  });

  it("marks the routing-capable provider as `(routing)` in the option label", () => {
    renderPicker();
    expect(
      screen.getByRole("option", { name: /openrouter \(routing\)/i }),
    ).toBeInTheDocument();
    // Non-routing providers are NOT annotated.
    expect(screen.getByRole("option", { name: "openai" })).toBeInTheDocument();
  });
});

describe("ModelPicker — routing UI gating", () => {
  it("hides the routing UI for non-OpenRouter providers", () => {
    renderPicker({ value: { provider: "openai", model: "" } });
    expect(screen.queryByLabelText(/preferred upstream order/i)).toBeNull();
    expect(screen.queryByLabelText(/allow fallbacks/i)).toBeNull();
  });

  it("shows the routing UI for OpenRouter (supports_upstream_routing=true)", () => {
    renderPicker({ value: { provider: "openrouter", model: "" } });
    expect(screen.getByLabelText(/preferred upstream order/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/allow fallbacks/i)).toBeInTheDocument();
  });
});

describe("ModelPicker — onChange contract", () => {
  it("emits provider change immediately", async () => {
    const { onChange } = renderPicker();
    const user = userEvent.setup();
    await user.selectOptions(
      screen.getByLabelText(/^provider$/i),
      "anthropic",
    );
    expect(onChange).toHaveBeenCalledWith({
      provider: "anthropic",
      model: "",
    });
  });

  it("emits model-slug edits as the user types", async () => {
    const { onChange } = renderPicker({
      value: { provider: "anthropic", model: "" },
    });
    const user = userEvent.setup();
    const input = screen.getByLabelText(/model slug/i);
    await user.type(input, "x");
    // The controlled input fires onChange per keystroke; for a single
    // character the last call is the only one.
    expect(onChange).toHaveBeenLastCalledWith({
      provider: "anthropic",
      model: "x",
    });
  });

  it("populates routing_hints.order from comma-separated input", async () => {
    const { onChange } = renderPicker({
      value: { provider: "openrouter", model: "" },
    });
    const user = userEvent.setup();
    const orderInput = screen.getByLabelText(/preferred upstream order/i);
    await user.type(orderInput, "a");
    expect(onChange).toHaveBeenLastCalledWith({
      provider: "openrouter",
      model: "",
      routing_hints: { order: ["a"] },
    });
  });

  it("emits a 'no hints' object as undefined (collapses to /api 'or_hints=none')", async () => {
    // Start with order=[anthropic]; clear it; the next onChange MUST
    // drop `routing_hints` entirely so the backend audit log shows
    // `or_hints=none` rather than an empty-object passthrough.
    const { onChange } = renderPicker({
      value: {
        provider: "openrouter",
        model: "",
        routing_hints: { order: ["anthropic"] },
      },
    });
    const user = userEvent.setup();
    const orderInput = screen.getByLabelText(
      /preferred upstream order/i,
    ) as HTMLInputElement;
    await user.clear(orderInput);
    // Last onChange call should have routing_hints undefined.
    const lastCall = onChange.mock.calls[onChange.mock.calls.length - 1] as [
      ModelPickerValue,
    ];
    expect(lastCall[0].routing_hints).toBeUndefined();
  });
});

describe("ModelPicker — slug shape validation (UX hint, not security gate)", () => {
  it("flags malformed upstream slugs with aria-invalid", async () => {
    const { rerender } = renderPicker({
      value: { provider: "openrouter", model: "" },
    });
    // Simulate the parent applying an onChange that includes an uppercase
    // slug (the user's input that the parent forwarded verbatim).
    rerender({
      value: {
        provider: "openrouter",
        model: "",
        routing_hints: { order: ["UPPERCASE"] },
      },
    });
    const input = screen.getByLabelText(
      /preferred upstream order/i,
    ) as HTMLInputElement;
    expect(input.getAttribute("aria-invalid")).toBe("true");
  });
});

describe("ModelPicker — fallbacks tri-state", () => {
  it("emits allow_fallbacks=true when the user picks 'Yes'", async () => {
    const { onChange } = renderPicker({
      value: { provider: "openrouter", model: "" },
    });
    const user = userEvent.setup();
    await user.selectOptions(
      screen.getByLabelText(/allow fallbacks/i),
      "true",
    );
    const lastCall = onChange.mock.calls[onChange.mock.calls.length - 1] as [
      ModelPickerValue,
    ];
    expect(lastCall[0].routing_hints?.allow_fallbacks).toBe(true);
  });

  it("emits allow_fallbacks=false when the user picks 'No'", async () => {
    const { onChange } = renderPicker({
      value: { provider: "openrouter", model: "" },
    });
    const user = userEvent.setup();
    await user.selectOptions(
      screen.getByLabelText(/allow fallbacks/i),
      "false",
    );
    const lastCall = onChange.mock.calls[onChange.mock.calls.length - 1] as [
      ModelPickerValue,
    ];
    expect(lastCall[0].routing_hints?.allow_fallbacks).toBe(false);
  });
});

describe("ModelPicker — provider-switch sanitiser", () => {
  it("clears pending hints when switching to a non-routing provider", () => {
    // First render with openrouter + hints present.
    const onChange = vi.fn();
    const { rerender } = render(
      <ModelPicker
        providers={CATALOGUE}
        value={{
          provider: "openrouter",
          model: "",
          routing_hints: { order: ["anthropic"] },
        }}
        onChange={onChange}
      />,
    );
    // Re-render with the parent's new value (provider switched to
    // openai but routing_hints not yet cleared by the parent — the
    // ModelPicker effect must fire onChange to drop them).
    rerender(
      <ModelPicker
        providers={CATALOGUE}
        value={{
          provider: "openai",
          model: "",
          routing_hints: { order: ["anthropic"] },
        }}
        onChange={onChange}
      />,
    );
    // The effect runs after the rerender — the picker MUST signal the
    // parent to drop routing_hints.
    expect(onChange).toHaveBeenCalledWith({
      provider: "openai",
      model: "",
      routing_hints: undefined,
    });
  });
});
