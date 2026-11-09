import { describe, expect, it } from "vitest";

import type { RealtimeRateRow } from "./api";
import { EMPTY_RT, alarmDots, captureSpans, firstAtOrAfter, mergeRealtime, nearestRow, sinceOf, yTop } from "./realtime";

const sent = (t: number[], rateT: number[] = []) => ({
  t,
  x: t.map((v) => v / 100),
  src: t.map(() => 0),
  rate_t: rateT,
  rate: rateT.map(() => 43),
});

describe("mergeRealtime", () => {
  it("appends only what is newer, per series, and keeps the last keepS seconds", () => {
    let h = mergeRealtime(EMPTY_RT, sent([10, 10.25, 10.5], [9, 10]), 11, 3600);
    expect(sinceOf(h)).toBe(10);
    h = mergeRealtime(h, sent([10.25, 10.5, 10.75], [10, 11]), 12, 3600);
    expect(h.t).toEqual([10, 10.25, 10.5, 10.75]);
    expect(h.rateT).toEqual([9, 10, 11]);
    h = mergeRealtime(h, sent([11, 11.25]), 13, 2.5);
    expect(h.t).toEqual([10.5, 10.75, 11, 11.25]);
    expect(h.rateT).toEqual([11]);
  });

  it("starts from everything the server keeps", () => {
    expect(sinceOf(EMPTY_RT)).toBe(0);
  });
});

describe("helpers", () => {
  it("finds the first value at or after", () => {
    expect(firstAtOrAfter([1, 2, 3], 2)).toBe(1);
    expect(firstAtOrAfter([1, 2, 3], 9)).toBe(3);
    expect(firstAtOrAfter([], 1)).toBe(0);
  });

  it("leaves room above the threshold and the peak", () => {
    expect(yTop([1, 1.1, null], 2.14)).toBeCloseTo(3.21);
    expect(yTop([1, 12], 2.14)).toBeCloseTo(12.96);
    expect(yTop([500], 2.14)).toBe(60);
  });

  it("joins capture-file hops into spans", () => {
    const t = [0, 0.25, 0.5, 0.75, 1, 3, 3.25];
    expect(captureSpans(t, [0, 1, 1, 1, 0, 1, 1], 0.25)).toEqual([
      { t0: 0, t1: 0.75 },
      { t0: 2.75, t1: 3.25 },
    ]);
  });

  it("marks alarms on their hop", () => {
    expect(alarmDots([1, 1.25, 1.5], [1, 5, 2], [{ t: 1.25 }])).toEqual([null, 5, null]);
  });

  it("picks the measured rate closest on a log scale", () => {
    const table = [43, 21.5, 10.8].map((rate_hz) => ({ rate_hz }) as RealtimeRateRow);
    expect(nearestRow(table, 42.6)).toBe(0);
    expect(nearestRow(table, 16)).toBe(1);
    expect(nearestRow(table, null)).toBe(-1);
  });
});
