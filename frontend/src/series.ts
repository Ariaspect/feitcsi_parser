// Pure geometry for the presence panel's line charts and verdict strip.
// Kept out of the component so it is unit-testable without a DOM, the same
// split view.ts and render.ts use for the heatmaps.

export interface Scale {
  (value: number): number;
}

/** Map a data domain onto a pixel range. Linear, clamped nowhere — callers
 *  crop with the SVG viewBox rather than by folding points onto the edge,
 *  which would draw a false flat line along the boundary. */
export function linearScale(
  domain: [number, number],
  range: [number, number],
): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  // A zero-width domain has no meaningful mapping; put everything at the
  // middle rather than dividing by zero and drawing NaN paths.
  if (span === 0) return () => (r0 + r1) / 2;
  return (value: number) => r0 + ((value - d0) / span) * (r1 - r0);
}

/** SVG path for a series that may be missing values.
 *
 * A `null` is a window with no answer, and it must read as one. Joining
 * across it draws a straight line through the gap, which on a presence panel
 * is a line the data never claimed; substituting zero is worse still, since
 * zero score is precisely the "empty room" reading. So a null ends the
 * current subpath and the next finite value starts a new one, leaving a
 * visible break.
 *
 * Non-finite numbers are treated as null: a NaN that survived serialisation
 * should break the line, not poison the path string.
 */
export function linePath(
  xs: number[],
  ys: (number | null)[],
  x: Scale,
  y: Scale,
): string {
  let path = "";
  let open = false;
  for (let i = 0; i < xs.length && i < ys.length; i++) {
    const v = ys[i];
    if (v === null || v === undefined || !Number.isFinite(v)) {
      open = false;
      continue;
    }
    const px = x(xs[i]);
    const py = y(v);
    if (!Number.isFinite(px) || !Number.isFinite(py)) {
      open = false;
      continue;
    }
    path += `${open ? "L" : "M"}${px.toFixed(2)} ${py.toFixed(2)}`;
    open = true;
  }
  return path;
}

export interface Run<T> {
  value: T;
  t0: number;
  t1: number;
}

/** Collapse a per-window series into contiguous runs of equal value.
 *
 * The verdict strip is drawn as blocks, and one rectangle per window would be
 * hundreds of adjacent fills that seam visibly against each other at most
 * zoom levels.
 *
 * Boundaries sit at the midpoint between neighbouring window centres, because
 * a window centre is what the backend reports and a verdict is about the
 * window, not the instant. The outer edges extend by half the neighbouring
 * spacing so the strip covers the range it describes instead of stopping
 * half a window short at each end.
 */
export function runs<T>(times: number[], values: T[]): Run<T>[] {
  const n = Math.min(times.length, values.length);
  if (n === 0) return [];
  if (n === 1) return [{ value: values[0], t0: times[0], t1: times[0] }];

  const edge = (i: number): number => (times[i] + times[i + 1]) / 2;
  const first = times[0] - (times[1] - times[0]) / 2;
  const last = times[n - 1] + (times[n - 1] - times[n - 2]) / 2;

  const out: Run<T>[] = [];
  let start = first;
  for (let i = 0; i < n; i++) {
    const end = i === n - 1 ? last : edge(i);
    const previous = out[out.length - 1];
    if (previous && previous.value === values[i]) {
      previous.t1 = end;
    } else {
      out.push({ value: values[i], t0: start, t1: end });
    }
    start = end;
  }
  return out;
}

/** Round tick values for an axis, at or below *count* of them.
 *
 * Steps are 1/2/5 x a power of ten, so labels land on numbers a reader can
 * hold — 0.25, not 0.2857. */
export function ticks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return [];
  const rough = (max - min) / Math.max(1, count);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const normalised = rough / magnitude;
  const step =
    (normalised >= 5 ? 5 : normalised >= 2 ? 2 : 1) * magnitude;

  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step / 1e6; v += step) {
    // Re-round: repeated addition of 0.1 drifts into 0.30000000000000004.
    out.push(Number((Math.round(v / step) * step).toPrecision(12)));
  }
  return out;
}

/** Seconds formatted for a time axis, adapting to the span on screen.
 *
 * A four-hour capture labelled in bare seconds is unreadable, and a
 * two-second zoom labelled in whole minutes has no labels at all. */
export function formatTime(seconds: number, span: number): string {
  if (!Number.isFinite(seconds)) return "";
  const sign = seconds < 0 ? "-" : "";
  const s = Math.abs(seconds);
  if (span >= 600) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return `${sign}${h}:${String(m).padStart(2, "0")}`;
  }
  if (span >= 20) {
    const m = Math.floor(s / 60);
    const rest = Math.floor(s % 60);
    return `${sign}${m}:${String(rest).padStart(2, "0")}`;
  }
  return `${sign}${s.toFixed(span >= 2 ? 1 : 2)}s`;
}
