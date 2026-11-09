// The Realtime tab's client-side history and the small pieces of arithmetic it draws with. Pure, so they are tested
// without a server (realtime.test.ts).

import type { RealtimeRateRow, RealtimeStatus } from "./api";

export interface RtSeries {
  t: number[];
  x: (number | null)[];
  src: number[];
  rateT: number[];
  rate: number[];
}

export const EMPTY_RT: RtSeries = { t: [], x: [], src: [], rateT: [], rate: [] };

const last = (xs: number[]) => (xs.length ? xs[xs.length - 1] : -Infinity);

/** Where the next poll starts: the older of the two series' last times (a second's frame count is sent once the
 *  second has closed, after the values in it), 0 for everything the server keeps. Duplicates are dropped on merge. */
export function sinceOf(h: RtSeries): number {
  if (!h.t.length && !h.rateT.length) return 0;
  return Math.min(h.t.length ? last(h.t) : Infinity, h.rateT.length ? last(h.rateT) : Infinity);
}

/** Append what the server sent after what is held; drop anything older than `keepS` before `now`. */
export function mergeRealtime(
  prev: RtSeries,
  s: Pick<RealtimeStatus, "t" | "x" | "src" | "rate_t" | "rate">,
  now: number,
  keepS: number,
): RtSeries {
  const lt = last(prev.t);
  const lr = last(prev.rateT);
  const t = prev.t.slice();
  const x = prev.x.slice();
  const src = prev.src.slice();
  s.t.forEach((v, i) => {
    if (v > lt) {
      t.push(v);
      x.push(s.x[i]);
      src.push(s.src[i]);
    }
  });
  const rateT = prev.rateT.slice();
  const rate = prev.rate.slice();
  s.rate_t.forEach((v, i) => {
    if (v > lr) {
      rateT.push(v);
      rate.push(s.rate[i]);
    }
  });
  const cut = now - keepS;
  const a = firstAtOrAfter(t, cut);
  const b = firstAtOrAfter(rateT, cut);
  return { t: t.slice(a), x: x.slice(a), src: src.slice(a), rateT: rateT.slice(b), rate: rate.slice(b) };
}

/** Index of the first value >= v in an ascending array (its length when none). */
export function firstAtOrAfter(xs: number[], v: number): number {
  let lo = 0;
  let hi = xs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Top of the y axis: room above the threshold and the highest value on screen, within reason. */
export function yTop(xs: (number | null)[], threshold: number, cap = 60): number {
  let hi = 0;
  for (const v of xs) if (v != null && Number.isFinite(v) && v > hi) hi = v;
  return Math.min(cap, Math.max(3, threshold * 1.5, hi * 1.08));
}

/** Spans whose values were read from a running capture's file (src 1), joined across one missing hop. */
export function captureSpans(t: number[], src: number[], hop: number): { t0: number; t1: number }[] {
  const out: { t0: number; t1: number }[] = [];
  for (let i = 0; i < t.length; i++) {
    if (src[i] !== 1) continue;
    const prev = out[out.length - 1];
    if (prev && t[i] - prev.t1 <= 2.01 * hop) prev.t1 = t[i];
    else out.push({ t0: t[i] - hop, t1: t[i] });
  }
  return out;
}

/** Alarm markers on the value series: the value at an alarm's time, null elsewhere. */
export function alarmDots(t: number[], x: (number | null)[], alarms: { t: number }[]): (number | null)[] {
  const out: (number | null)[] = new Array(t.length).fill(null);
  for (const a of alarms) {
    const i = firstAtOrAfter(t, a.t - 1e-3);
    if (i < t.length && Math.abs(t[i] - a.t) < 1e-2) out[i] = x[i];
  }
  return out;
}

/** The measured row closest to a frame rate (on a log scale: 21.5 Hz is as far from 43 as 10.8 is from 21.5). */
export function nearestRow(table: RealtimeRateRow[], rateHz: number | null): number {
  if (rateHz == null || !(rateHz > 0) || !table.length) return -1;
  let best = 0;
  for (let i = 1; i < table.length; i++) {
    if (Math.abs(Math.log(table[i].rate_hz / rateHz)) < Math.abs(Math.log(table[best].rate_hz / rateHz))) best = i;
  }
  return best;
}
