import { describe, expect, it } from "vitest";

import { formatTime, linePath, linearScale, runs, ticks } from "./series";

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
