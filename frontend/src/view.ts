// Pure view-state logic for the heatmap, kept out of the component so it can be
// unit-tested without a DOM.
//
// The heatmap is a live-polling UI: App.tsx refetches every `refresh_ms` and
// passes a brand-new `timeSeconds` array on every poll. The view (the visible
// time/subcarrier window) must therefore live in a ref and survive across
// renders — it is never rebuilt from array identity. These functions decide
// how the view moves in response to new data and when it must be reset.

/** The current visible window. Persisted in a ref across renders. */
export interface View {
  tMin: number;
  tMax: number;
  scMin: number;
  scMax: number;
}

/**
 * Capture identity — what distinguishes "the same capture, growing" from "a
 * different capture". Only a change here is grounds for resetting the view.
 * Array identity, length, and new frames appended are NOT identity changes.
 */
export interface CaptureId {
  /** Name of the file the data came from. Taken from the snapshot rather than
   * from the path input, so it changes in lockstep with the data instead of on
   * every keystroke while the user is still typing a path. Two captures with
   * the same basename in different directories are indistinguishable here; the
   * time and subcarrier checks below are what catch that case. */
  filename: string;
  numSubcarriers: number;
  tMin: number;
  tMax: number;
  /** MIMO filter ('all' or 'NxM'). A filter change is a view-identity change:
   * the visible data window and amplitude distribution both shift, so the
   * view and the locked color scale must reset. */
  mimo?: string | null;
  /** Source MAC filter ('all' or a MAC string). Same rationale as mimo. */
  sourceMac?: string | null;
}

/** Full-extent view for a fresh capture. */
export function fullView(data: CaptureId & { scMin: number; scMax: number }): View {
  return { tMin: data.tMin, tMax: data.tMax, scMin: data.scMin, scMax: data.scMax };
}

/**
 * Slide the time window forward to keep the newest packet at the right edge,
 * preserving the window's DURATION. Clamps so the window never shows time
 * beyond the data. Returns just the time portion; the caller preserves the
 * subcarrier band.
 *
 * - prev null (first data): full extent [newTMin, newTMax].
 * - duration <= 0 or >= span (window at least as wide as data): full extent.
 *   The >= span case matters for follow-live at full extent: as long as the
 *   data is no wider than the window, show everything; once data outgrows the
 *   window, the window starts tracking the right edge at its fixed duration.
 * - otherwise: tMax = newTMax, tMin = newTMax - duration. Because duration <=
 *   span here, tMin >= newTMin, so the left edge never crosses the data start.
 */
export function followLiveView(
  prev: { tMin: number; tMax: number } | null,
  newTMin: number,
  newTMax: number,
): { tMin: number; tMax: number } {
  if (!prev) {
    return { tMin: newTMin, tMax: newTMax };
  }
  const duration = prev.tMax - prev.tMin;
  const span = newTMax - newTMin;
  if (!(duration > 0) || duration >= span) {
    return { tMin: newTMin, tMax: newTMax };
  }
  const tMax = newTMax;
  const tMin = tMax - duration; // >= newTMin because duration <= span
  return { tMin, tMax };
}

/**
 * Decide whether the view should be reset to full extent because the capture
 * identity changed. Returns true for a subcarrier-count change or a time axis
 * that went backwards (truncation or a different file). Returns false for a
 * normal append or a sliding window where tMin/tMax only grow.
 *
 * The live poll is a sliding window of the last `maxPackets` packets, so tMin
 * normally only increases as old frames age out. tMin or tMax decreasing is
 * the signal that the file was truncated or swapped.
 */
export function shouldResetView(prev: CaptureId, next: CaptureId): boolean {
  // A different file is a different capture even when its time range happens to
  // sit inside the old one. The component is not remounted when the user edits
  // the path, so without this check a frozen view can survive the switch and
  // land on a range the new capture has no data for -- a blank plot with no
  // indication of why.
  if (next.filename !== prev.filename) return true;
  if (next.numSubcarriers !== prev.numSubcarriers) return true;
  // A filter change shifts the visible data window and amplitude distribution.
  // The time-range checks below only catch narrowing (tMin/tMax shrink); a
  // filter that widens or shifts to a same-extent different MAC would not
  // trigger a reset, leaving a stale view and a locked color scale bound to
  // the old filter's data.
  if (next.mimo !== prev.mimo) return true;
  if (next.sourceMac !== prev.sourceMac) return true;
  if (next.tMax < prev.tMax) return true; // newest packet went backwards — truncation
  if (next.tMin < prev.tMin) return true; // start went backwards — different file
  return false;
}

/**
 * Advance the view for a new data arrival.
 *
 * - Frozen (followLive false): return prev unchanged — same reference, so a
 *   frozen view is bit-identical across any number of polls. This is the
 *   property that makes "zoom then poll" not destroy the view.
 * - Live (followLive true): slide the time window forward via followLiveView,
 *   preserving the subcarrier band.
 *
 * Never resets; reset is the caller's job, driven by shouldResetView.
 */
export function advanceView(
  prev: View,
  data: { tMin: number; tMax: number },
  followLive: boolean,
): View {
  if (!followLive) return prev;
  const { tMin, tMax } = followLiveView(prev, data.tMin, data.tMax);
  return { tMin, tMax, scMin: prev.scMin, scMax: prev.scMax };
}

/**
 * Clamp a time window to the data extent, never inverting. A window wider than
 * the data collapses to full extent; a window shifted past either edge is
 * shifted back inside. Used as a safety net after d3-zoom derives a visible
 * domain.
 */
export function clampTimeWindow(
  tMin: number,
  tMax: number,
  dataTMin: number,
  dataTMax: number,
): { tMin: number; tMax: number } {
  const span = dataTMax - dataTMin;
  let lo = tMin;
  let hi = tMax;
  if (!(hi > lo)) {
    return { tMin: dataTMin, tMax: dataTMax };
  }
  if (hi - lo >= span) {
    return { tMin: dataTMin, tMax: dataTMax };
  }
  if (lo < dataTMin) {
    const d = dataTMin - lo;
    lo += d;
    hi += d;
  }
  if (hi > dataTMax) {
    const d = hi - dataTMax;
    lo -= d;
    hi -= d;
  }
  if (lo < dataTMin) lo = dataTMin;
  return { tMin: lo, tMax: hi };
}

/**
 * Clamp a subcarrier band to the full band, never inverting. Mirrors
 * clampTimeWindow for the subcarrier axis, which is zoomed via shift+wheel
 * (hand-rolled, not d3-zoom) and clamped to the physical band.
 */
export function clampScWindow(
  scMin: number,
  scMax: number,
  bandMin: number,
  bandMax: number,
): { scMin: number; scMax: number } {
  const span = bandMax - bandMin;
  let lo = scMin;
  let hi = scMax;
  if (!(hi > lo)) {
    return { scMin: bandMin, scMax: bandMax };
  }
  if (hi - lo >= span) {
    return { scMin: bandMin, scMax: bandMax };
  }
  if (lo < bandMin) {
    const d = bandMin - lo;
    lo += d;
    hi += d;
  }
  if (hi > bandMax) {
    const d = hi - bandMax;
    lo -= d;
    hi -= d;
  }
  if (lo < bandMin) lo = bandMin;
  return { scMin: lo, scMax: hi };
}
