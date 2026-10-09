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

/** Fit a window inside `limit`, never narrower than `minSpan` and never wider
 *  than `limit` itself, by clamping the span and then sliding it in.
 *
 *  Slid rather than cropped: a pan that runs into the end of the capture should
 *  stop moving, not shrink the view the reader is holding.
 */
export function clampWindow(
  window: [number, number],
  limit: [number, number],
  minSpan: number,
): [number, number] {
  const outer = Math.max(0, limit[1] - limit[0]);
  const span = Math.min(Math.max(window[1] - window[0], minSpan), outer || minSpan);
  let t0 = window[0];
  if (t0 + span > limit[1]) t0 = limit[1] - span;
  if (t0 < limit[0]) t0 = limit[0];
  return [t0, t0 + span];
}

/** Zoom a window about a fraction of its own width, keeping that point fixed.
 *
 *  `anchor` is where the cursor sits, 0 at the left edge and 1 at the right, so
 *  the time under the pointer does not move while the wheel turns — the only
 *  zoom that feels like the view is being scaled rather than jumped.
 *  `factor` above 1 widens (zooms out), below 1 narrows.
 */
export function zoomWindow(
  window: [number, number],
  anchor: number,
  factor: number,
  limit: [number, number],
  minSpan: number,
): [number, number] {
  const span = window[1] - window[0];
  const a = Math.min(1, Math.max(0, anchor));
  const at = window[0] + a * span;
  // The span is clamped BEFORE the window is positioned. The other order lets
  // clampWindow widen the window by moving its right edge, which walks the
  // anchored time out to the left edge exactly when the wheel hits the floor.
  const outer = Math.max(0, limit[1] - limit[0]);
  const next = Math.min(Math.max(span * factor, minSpan), outer || minSpan);
  return clampWindow([at - a * next, at - a * next + next], limit, minSpan);
}

/** SVG path for a filled band between a low and a high series.
 *
 * Drawn for a decimated per-frame signal, where one column spans many frames
 * and a line through the column medians would hide the single frame that
 * moved — which on a frame-to-frame detector is the whole signal. The band is
 * the column's extremes, so a one-frame spike stays visible at any zoom.
 *
 * A column missing either edge closes the current band and starts a new one
 * after it, for the same reason `linePath` breaks: a band drawn across a
 * dropout would fill area the data never measured.
 */
export function bandPath(
  xs: number[],
  los: (number | null)[],
  his: (number | null)[],
  x: Scale,
  y: Scale,
): string {
  let path = "";
  let run: { px: number; lo: number; hi: number }[] = [];

  const flush = () => {
    // A band needs two columns to have any width; a lone column is left to
    // the median line rather than drawn as a zero-width sliver.
    if (run.length >= 2) {
      const top = run.map((p) => `${p.px.toFixed(2)} ${p.hi.toFixed(2)}`);
      const bottom = run
        .slice()
        .reverse()
        .map((p) => `${p.px.toFixed(2)} ${p.lo.toFixed(2)}`);
      path += `M${top.join("L")}L${bottom.join("L")}Z`;
    }
    run = [];
  };

  for (let i = 0; i < xs.length; i++) {
    const lo = los[i];
    const hi = his[i];
    if (
      lo === null || hi === null || lo === undefined || hi === undefined ||
      !Number.isFinite(lo) || !Number.isFinite(hi)
    ) {
      flush();
      continue;
    }
    const px = x(xs[i]);
    const pLo = y(lo);
    const pHi = y(hi);
    if (!Number.isFinite(px) || !Number.isFinite(pLo) || !Number.isFinite(pHi)) {
      flush();
      continue;
    }
    run.push({ px, lo: pLo, hi: pHi });
  }
  flush();
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
