import { describe, it, expect } from "vitest";
import {
  followLiveView,
  shouldResetView,
  advanceView,
  clampTimeWindow,
  clampScWindow,
  fullView,
  type View,
  type CaptureId,
} from "./view";

describe("followLiveView", () => {
  it("returns full extent when prev is null (first data)", () => {
    expect(followLiveView(null, 0, 10)).toEqual({ tMin: 0, tMax: 10 });
    expect(followLiveView(null, 5, 15)).toEqual({ tMin: 5, tMax: 15 });
  });

  it("slides forward preserving duration, tracking the right edge", () => {
    const prev = { tMin: 5, tMax: 7 }; // duration 2
    // data grew from [0,10] to [0,10.5]: window slides by 0.5
    expect(followLiveView(prev, 0, 10.5)).toEqual({ tMin: 8.5, tMax: 10.5 });
    // next poll [0,11]: slides another 0.5
    expect(followLiveView({ tMin: 8.5, tMax: 10.5 }, 0, 11)).toEqual({ tMin: 9, tMax: 11 });
  });

  it("slides by exactly the amount of new time", () => {
    const prev = { tMin: 8, tMax: 10 }; // duration 2, right edge at 10
    const res = followLiveView(prev, 0, 10.3); // 0.3s of new data
    expect(res.tMax - 10).toBeCloseTo(0.3, 10); // right edge moved by new time
    expect(res.tMin - 8).toBeCloseTo(0.3, 10); // left edge moved by the same
    expect(res.tMax - res.tMin).toBeCloseTo(2, 10); // duration preserved
  });

  it("returns full extent when duration >= span (follow-live at full extent)", () => {
    const prev = { tMin: 0, tMax: 10 }; // duration 10
    expect(followLiveView(prev, 0, 10)).toEqual({ tMin: 0, tMax: 10 }); // ===
    expect(followLiveView(prev, 0, 8)).toEqual({ tMin: 0, tMax: 8 }); // duration > span
  });

  it("never shows time beyond the data (clamps when window wider than data)", () => {
    // window duration 5, but data span only 3 → can't fit, show full
    const prev = { tMin: 0, tMax: 5 };
    expect(followLiveView(prev, 2, 5)).toEqual({ tMin: 2, tMax: 5 });
  });

  it("handles a sliding window (tMin grows, tMax grows)", () => {
    const prev = { tMin: 8, tMax: 10 }; // duration 2
    // window slid: data was [0,10], now [1,11]
    expect(followLiveView(prev, 1, 11)).toEqual({ tMin: 9, tMax: 11 });
  });

  it("does not snap back to full extent once zoomed", () => {
    // user zoomed to a 2s window; follow-live must keep it a 2s window that
    // tracks forward, not reset to the full extent.
    const prev = { tMin: 5, tMax: 7 };
    const res = followLiveView(prev, 0, 12); // data grew to 12s
    expect(res).toEqual({ tMin: 10, tMax: 12 });
    expect(res.tMax - res.tMin).toBe(2);
  });
});

describe("shouldResetView", () => {
  const prev: CaptureId = { filename: "a.dat", numSubcarriers: 242, tMin: 0, tMax: 10 };

  it("returns false for same capture with more frames appended", () => {
    expect(shouldResetView(prev, { filename: "a.dat", numSubcarriers: 242, tMin: 0, tMax: 12 })).toBe(false);
  });

  it("returns false for a sliding window (tMin grows, tMax grows)", () => {
    expect(shouldResetView(prev, { filename: "a.dat", numSubcarriers: 242, tMin: 1, tMax: 11 })).toBe(false);
  });

  it("returns false when no frames changed", () => {
    expect(shouldResetView(prev, { filename: "a.dat", numSubcarriers: 242, tMin: 0, tMax: 10 })).toBe(false);
  });

  it("returns true for subcarrier-count change", () => {
    expect(shouldResetView(prev, { filename: "a.dat", numSubcarriers: 248, tMin: 0, tMax: 10 })).toBe(true);
  });

  it("returns true when the time axis went backwards (truncation)", () => {
    expect(shouldResetView(prev, { filename: "a.dat", numSubcarriers: 242, tMin: 0, tMax: 8 })).toBe(true);
  });

  it("returns true when tMin went backwards (different file)", () => {
    expect(shouldResetView(prev, { filename: "a.dat", numSubcarriers: 242, tMin: -1, tMax: 10 })).toBe(true);
  });
});

describe("advanceView", () => {
  const view: View = { tMin: 5, tMax: 7, scMin: -10, scMax: 10 };
  const data = { tMin: 0, tMax: 10.5 };

  it("frozen: appending frames does not change the view (same reference)", () => {
    const result = advanceView(view, data, false);
    expect(result).toBe(view); // bit-identical, not a copy
  });

  it("live: appending frames slides the window preserving duration and sc band", () => {
    const result = advanceView(view, data, true);
    expect(result.tMin).toBe(8.5);
    expect(result.tMax).toBe(10.5);
    expect(result.scMin).toBe(-10);
    expect(result.scMax).toBe(10);
    expect(result.tMax - result.tMin).toBe(2); // duration preserved
  });

  it("a zoom followed by three polls leaves the view bit-identical (frozen)", () => {
    let v: View = { tMin: 3, tMax: 4, scMin: -5, scMax: 5 };
    const polls = [
      { tMin: 0, tMax: 10.1 },
      { tMin: 0, tMax: 10.2 },
      { tMin: 0, tMax: 10.3 },
    ];
    for (const p of polls) {
      v = advanceView(v, p, false);
    }
    expect(v).toEqual({ tMin: 3, tMax: 4, scMin: -5, scMax: 5 });
  });

  it("live across three polls slides by the total new time", () => {
    let v: View = { tMin: 8, tMax: 10, scMin: -5, scMax: 5 }; // duration 2
    v = advanceView(v, { tMin: 0, tMax: 10.3 }, true); // -> [8.3, 10.3]
    v = advanceView(v, { tMin: 0, tMax: 10.6 }, true); // -> [8.6, 10.6]
    v = advanceView(v, { tMin: 0, tMax: 10.9 }, true); // -> [8.9, 10.9]
    expect(v.tMax - v.tMin).toBeCloseTo(2, 10);
    expect(v.tMin).toBeCloseTo(8.9, 10);
    expect(v.tMax).toBeCloseTo(10.9, 10);
    expect(v.scMin).toBe(-5);
    expect(v.scMax).toBe(5);
  });

  it("truncation is decided by shouldResetView, not advanceView", () => {
    // advanceView never resets on its own; the caller checks shouldResetView.
    const v: View = { tMin: 5, tMax: 7, scMin: -10, scMax: 10 };
    const truncated = { tMin: 0, tMax: 6 };
    expect(shouldResetView(
      { filename: "a.dat", numSubcarriers: 242, tMin: 0, tMax: 10 },
      { filename: "a.dat", numSubcarriers: 242, tMin: 0, tMax: 6 },
    )).toBe(true);
    // advanceView frozen leaves it alone; advanceView live would slide, but the
    // caller resets instead because shouldResetView returned true.
    expect(advanceView(v, truncated, false)).toBe(v);
  });
});

describe("fullView", () => {
  it("returns the data extent as the view", () => {
    expect(fullView({ filename: "a.dat", numSubcarriers: 242, tMin: 1, tMax: 9, scMin: -121, scMax: 120 }))
      .toEqual({ tMin: 1, tMax: 9, scMin: -121, scMax: 120 });
  });
});

describe("clampTimeWindow", () => {
  it("returns full extent when window is wider than data", () => {
    expect(clampTimeWindow(-5, 20, 0, 10)).toEqual({ tMin: 0, tMax: 10 });
  });

  it("shifts right when the left edge is below data", () => {
    expect(clampTimeWindow(-2, 4, 0, 10)).toEqual({ tMin: 0, tMax: 6 });
  });

  it("shifts left when the right edge is above data", () => {
    expect(clampTimeWindow(8, 14, 0, 10)).toEqual({ tMin: 4, tMax: 10 });
  });

  it("leaves a fully-contained window unchanged", () => {
    expect(clampTimeWindow(2, 8, 0, 10)).toEqual({ tMin: 2, tMax: 8 });
  });

  it("falls back to full for degenerate or inverted input", () => {
    expect(clampTimeWindow(5, 5, 0, 10)).toEqual({ tMin: 0, tMax: 10 });
    expect(clampTimeWindow(8, 2, 0, 10)).toEqual({ tMin: 0, tMax: 10 });
  });
});

describe("clampScWindow", () => {
  it("clamps a band wider than the data to full", () => {
    expect(clampScWindow(-200, 200, -121, 120)).toEqual({ scMin: -121, scMax: 120 });
  });

  it("shifts right when below the band", () => {
    expect(clampScWindow(-200, -180, -121, 120)).toEqual({ scMin: -121, scMax: -101 });
  });

  it("shifts left when above the band", () => {
    expect(clampScWindow(100, 200, -121, 120)).toEqual({ scMin: 20, scMax: 120 });
  });

  it("leaves a contained band unchanged", () => {
    expect(clampScWindow(-10, 10, -121, 120)).toEqual({ scMin: -10, scMax: 10 });
  });

  it("falls back to full for degenerate input", () => {
    expect(clampScWindow(5, 5, -121, 120)).toEqual({ scMin: -121, scMax: 120 });
    expect(clampScWindow(10, -10, -121, 120)).toEqual({ scMin: -121, scMax: 120 });
  });
});

describe("shouldResetView: capture file identity", () => {
  const base = { filename: "a.dat", numSubcarriers: 242, tMin: 10, tMax: 20 };

  it("resets when the file changes even if its time range sits inside the old one", () => {
    // The component is not remounted when the user edits the path, so a frozen
    // view would otherwise survive the switch and sit on a range the new
    // capture has no data for -- a blank plot with no explanation.
    expect(
      shouldResetView(base, { ...base, filename: "b.dat", tMin: 12, tMax: 18 }),
    ).toBe(true);
  });

  it("resets when a longer capture is opened whose window starts later", () => {
    // Both bounds move forward, so every time-based check passes; only the
    // filename distinguishes these.
    expect(
      shouldResetView(base, { filename: "b.dat", numSubcarriers: 242, tMin: 500, tMax: 600 }),
    ).toBe(true);
  });

  it("does not reset while the same file grows", () => {
    expect(shouldResetView(base, { ...base, tMin: 11, tMax: 25 })).toBe(false);
    expect(shouldResetView(base, { ...base, tMax: 21 })).toBe(false);
  });
});

describe("shouldResetView: filter change", () => {
  const base = { filename: "a.dat", numSubcarriers: 242, tMin: 0, tMax: 100, mimo: "all", sourceMac: "all" };

  it("resets when MIMO filter changes from all to 2x2", () => {
    expect(shouldResetView(base, { ...base, mimo: "2x2", tMin: 10, tMax: 90 })).toBe(true);
  });

  it("resets when MIMO filter changes from 2x2 to all (widening)", () => {
    const narrowed = { ...base, mimo: "2x2", tMin: 10, tMax: 90 };
    // Widening: tMin shrinks (10→0), tMax grows (90→100). Without the filter
    // check, tMin<prev.tMin would catch this — but the filter check fires
    // first and is the semantically correct reason.
    expect(shouldResetView(narrowed, base)).toBe(true);
  });

  it("resets when source MAC changes even if extent is identical", () => {
    // Two MACs both span [0, 100]: tMin/tMax unchanged, so the time checks
    // pass. Without the filter check, the view and ampScaleRef would stay
    // locked to the old MAC's data — wrong colors, stale zoom.
    const macA = { ...base, sourceMac: "aa:bb:cc:dd:ee:ff" };
    const macB = { ...base, sourceMac: "11:22:33:44:55:66" };
    expect(shouldResetView(macA, macB)).toBe(true);
  });

  it("does not reset when filter is unchanged (live poll with filter active)", () => {
    // Same filter, t_max grew (new frames matching filter arrived): no reset,
    // follow-live slides the view.
    expect(shouldResetView(base, { ...base, tMax: 110 })).toBe(false);
  });

  it("does not reset when both filter fields are undefined", () => {
    // Backwards compatibility: CaptureId without filter fields (e.g. older
    // callers) still works — undefined !== undefined is false.
    const noFilter = { filename: "a.dat", numSubcarriers: 242, tMin: 0, tMax: 100 };
    expect(shouldResetView(noFilter, { ...noFilter, tMax: 110 })).toBe(false);
  });
});
