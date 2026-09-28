import { describe, expect, it } from "vitest";

import {
  bandPath,
  clampWindow,
  formatTime,
  linePath,
  linearScale,
  runs,
  ticks,
  zoomWindow,
} from "./series";

describe("linearScale", () => {
  it("maps the domain onto the range", () => {
    const s = linearScale([0, 10], [0, 100]);
    expect(s(0)).toBe(0);
    expect(s(5)).toBe(50);
    expect(s(10)).toBe(100);
  });

  it("inverts when the range does, as a y axis does", () => {
    const s = linearScale([0, 1], [200, 0]);
    expect(s(0)).toBe(200);
    expect(s(1)).toBe(0);
  });

  it("survives a zero-width domain instead of drawing NaN", () => {
    const s = linearScale([3, 3], [0, 100]);
    expect(s(3)).toBe(50);
    expect(Number.isFinite(s(99))).toBe(true);
  });
});

describe("linePath", () => {
  const x = linearScale([0, 4], [0, 400]);
  const y = linearScale([0, 1], [100, 0]);

  it("draws one subpath through finite values", () => {
    const path = linePath([0, 1, 2], [0, 0.5, 1], x, y);
    expect(path.match(/M/g)).toHaveLength(1);
    expect(path.match(/L/g)).toHaveLength(2);
  });

  it("breaks the line at a null instead of joining across it", () => {
    // The whole point: a joined line asserts a value the data never gave, and
    // on a presence panel the gap is exactly where the claim matters.
    const path = linePath([0, 1, 2, 3], [0.2, null, null, 0.8], x, y);
    expect(path.match(/M/g)).toHaveLength(2);
    expect(path).not.toContain("L");
  });

  it("treats a NaN that survived serialisation as a break", () => {
    const path = linePath([0, 1, 2], [0.2, NaN, 0.8], x, y);
    expect(path.match(/M/g)).toHaveLength(2);
  });

  it("returns an empty path when nothing is finite", () => {
    expect(linePath([0, 1], [null, null], x, y)).toBe("");
  });
});

describe("clampWindow", () => {
  it("leaves a window that already fits", () => {
    expect(clampWindow([10, 20], [0, 100], 1)).toEqual([10, 20]);
  });

  it("slides a window back inside instead of cropping it", () => {
    // A pan that hits the end should stop, not shrink the view in the reader's
    // hands.
    expect(clampWindow([95, 115], [0, 100], 1)).toEqual([80, 100]);
    expect(clampWindow([-30, -10], [0, 100], 1)).toEqual([0, 20]);
  });

  it("refuses to go narrower than the floor", () => {
    expect(clampWindow([50, 50.01], [0, 100], 0.5)).toEqual([50, 50.5]);
  });

  it("refuses to go wider than the limit", () => {
    expect(clampWindow([-50, 150], [0, 100], 1)).toEqual([0, 100]);
  });
});

describe("zoomWindow", () => {
  const limit: [number, number] = [0, 100];

  it("keeps the anchored time under the cursor", () => {
    const [t0, t1] = zoomWindow([0, 100], 0.25, 0.5, limit, 1);
    // 25 s was a quarter in and must still be a quarter in.
    expect(t0 + 0.25 * (t1 - t0)).toBeCloseTo(25);
    expect(t1 - t0).toBeCloseTo(50);
  });

  it("zooms out to the limit and no further", () => {
    expect(zoomWindow([40, 60], 0.5, 100, limit, 1)).toEqual([0, 100]);
  });

  it("stops at the floor when zooming in", () => {
    const out = zoomWindow([40, 60], 0.5, 1e-6, limit, 0.5);
    expect(out[1] - out[0]).toBeCloseTo(0.5);
    expect(out[0]).toBeCloseTo(49.75);
  });

  it("treats an out-of-range anchor as an edge", () => {
    expect(zoomWindow([0, 100], 5, 0.5, limit, 1)).toEqual(
      zoomWindow([0, 100], 1, 0.5, limit, 1),
    );
  });
});

describe("bandPath", () => {
  const x = linearScale([0, 4], [0, 400]);
  const y = linearScale([-1, 1], [100, 0]);

  it("closes one filled region over a run of columns", () => {
    const path = bandPath([0, 1, 2], [-0.2, -0.3, -0.1], [0.2, 0.4, 0.1], x, y);
    expect(path.match(/M/g)).toHaveLength(1);
    expect(path.match(/Z/g)).toHaveLength(1);
  });

  it("splits the band at a blank column instead of filling across it", () => {
    const path = bandPath(
      [0, 1, 2, 3, 4],
      [-0.2, -0.2, null, -0.2, -0.2],
      [0.2, 0.2, null, 0.2, 0.2],
      x,
      y,
    );
    expect(path.match(/Z/g)).toHaveLength(2);
  });

  it("skips a lone column, which has no width to fill", () => {
    expect(bandPath([0], [-0.2], [0.2], x, y)).toBe("");
  });

  it("treats a NaN edge as a break", () => {
    const path = bandPath([0, 1, 2], [-0.2, NaN, -0.2], [0.2, 0.2, 0.2], x, y);
    expect(path).toBe("");
  });
});

describe("runs", () => {
  it("collapses equal neighbours into one block", () => {
    const out = runs([0, 1, 2, 3], ["a", "a", "b", "b"]);
    expect(out.map((r) => r.value)).toEqual(["a", "b"]);
  });

  it("splits at every change, including a repeat later on", () => {
    const out = runs([0, 1, 2], ["a", "b", "a"]);
    expect(out.map((r) => r.value)).toEqual(["a", "b", "a"]);
  });

  it("puts boundaries between window centres, not on them", () => {
    // A verdict describes its window, not the instant at its centre, so the
    // block has to reach halfway to each neighbour.
    const out = runs([10, 20, 30], ["a", "b", "b"]);
    expect(out[0].t0).toBe(5);
    expect(out[0].t1).toBe(15);
    expect(out[1].t0).toBe(15);
    expect(out[1].t1).toBe(35);
  });

  it("covers the whole span with no holes between blocks", () => {
    const out = runs([0, 2, 4, 6], ["a", "b", "b", "c"]);
    for (let i = 1; i < out.length; i++) {
      expect(out[i].t0).toBe(out[i - 1].t1);
    }
  });

  it("handles the degenerate inputs a live poll can produce", () => {
    expect(runs([], [])).toEqual([]);
    expect(runs([7], ["a"])).toEqual([{ value: "a", t0: 7, t1: 7 }]);
  });
});

describe("ticks", () => {
  it("lands on round numbers", () => {
    expect(ticks(0, 1, 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
  });

  it("does not accumulate floating point drift", () => {
    for (const t of ticks(0, 1, 5)) {
      expect(String(t).length).toBeLessThan(6);
    }
  });

  it("refuses a degenerate or inverted domain", () => {
    expect(ticks(1, 1)).toEqual([]);
    expect(ticks(5, 2)).toEqual([]);
    expect(ticks(NaN, 1)).toEqual([]);
  });
});

describe("formatTime", () => {
  it("uses hours on a capture-length span", () => {
    expect(formatTime(3720, 3600)).toBe("1:02");
  });

  it("uses minutes and seconds on a minutes-long span", () => {
    expect(formatTime(125, 120)).toBe("2:05");
  });

  it("keeps sub-second precision when zoomed in", () => {
    expect(formatTime(1.234, 1)).toBe("1.23s");
  });
});
