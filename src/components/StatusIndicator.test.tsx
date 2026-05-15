import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { StatusIndicator, type Status } from "./StatusIndicator";

/**
 * The status dot drives both the in-app sidebar indicator and (via a
 * mirrored Rust color table) the system tray icon. Lock the colors and
 * the human-readable labels down so a careless edit can't silently break
 * the "no one in frame" UX.
 */

const expectations: Record<
  Status,
  { color: string; label: string; pulse: boolean }
> = {
  normal: { color: "bg-green-500", label: "Monitoring", pulse: true },
  alert: { color: "bg-red-500", label: "Alert", pulse: true },
  paused: { color: "bg-yellow-500", label: "Paused", pulse: false },
  offline: { color: "bg-gray-500", label: "Offline", pulse: false },
  absent: { color: "bg-gray-700", label: "No one in frame", pulse: false },
};

describe("StatusIndicator", () => {
  for (const [status, want] of Object.entries(expectations) as [
    Status,
    { color: string; label: string; pulse: boolean },
  ][]) {
    it(`renders ${status} with the right color, label, and pulse`, () => {
      const { container, getByText } = render(
        <StatusIndicator status={status} />,
      );
      expect(getByText(want.label)).toBeTruthy();
      const dot = container.querySelector(`.${want.color}`);
      expect(dot).not.toBeNull();
      const pulseEl = container.querySelector(".animate-ping");
      if (want.pulse) {
        expect(pulseEl).not.toBeNull();
      } else {
        expect(pulseEl).toBeNull();
      }
    });
  }

  it("renders the count badge when count is provided", () => {
    const { getByText } = render(
      <StatusIndicator status="normal" count={42} />,
    );
    expect(getByText("42")).toBeTruthy();
  });

  it("renders 0 as a count rather than hiding the badge", () => {
    const { getByText } = render(
      <StatusIndicator status="normal" count={0} />,
    );
    expect(getByText("0")).toBeTruthy();
  });

  it("omits the count badge when count is undefined", () => {
    const { container } = render(<StatusIndicator status="normal" />);
    const tabularNum = container.querySelector(".tabular-nums");
    expect(tabularNum).toBeNull();
  });
});
