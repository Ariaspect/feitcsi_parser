export interface Meta {
  filename: string;
  chipset: string;
  bandwidth: string | number;
  num_subcarriers: number;
  total_frames: number;
  t_min: number;
  t_max: number;
  num_rx: number;
  num_tx: number;
}

export interface Filters {
  mimo_modes: string[];
  source_macs: string[];
}

export interface CaptureFile {
  filename: string;
  path: string;
  size_bytes: number;
  mtime: number;
}

export async function fetchCaptures(signal?: AbortSignal): Promise<CaptureFile[]> {
  const res = await fetch("/api/captures", { signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return (await res.json()) as CaptureFile[];
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/** Shorten *text* to *max* characters, eliding from the middle.
 *
 * End-truncation is the CSS default and the wrong choice for capture names:
 * these differ in their *tails* (`csi_20260813_030001.dat` vs
 * `csi_20260813_120001.dat`), so cutting the end throws away the only part
 * that identifies the file, extension included. */
export function middleTruncate(text: string, max: number): string {
  if (max <= 1) return text.slice(0, Math.max(0, max));
  if (text.length <= max) return text;
  const keep = max - 1; // one char for the ellipsis
  const head = Math.ceil(keep / 2);
  const tail = keep - head;
  return `${text.slice(0, head)}\u2026${tail > 0 ? text.slice(text.length - tail) : ""}`;
}

/** Fit a capture name into *max* characters, keeping the basename whole.
 *
 * Captures may be nested (`/api/captures` reports a path relative to
 * `captures/`), and the directories are the disposable part — the file itself
 * is what identifies the capture. So the directory prefix is elided first, and
 * only a basename that cannot fit on its own is truncated. */
export function truncateCaptureName(name: string, max: number): string {
  if (name.length <= max) return name;

  const cut = name.lastIndexOf("/");
  if (cut < 0) return middleTruncate(name, max);

  const base = name.slice(cut + 1);
  // "…/" costs 2 characters; below that the directory cannot be shown at all.
  if (base.length + 2 >= max) return middleTruncate(base, max);
  return `${middleTruncate(name.slice(0, cut), max - base.length - 1)}/${base}`;
}

function filterParams(mimo?: string | null, sourceMac?: string | null): string {
  let p = "";
  if (mimo && mimo !== "all") p += `&mimo=${encodeURIComponent(mimo)}`;
  if (sourceMac && sourceMac !== "all") p += `&source_mac=${encodeURIComponent(sourceMac)}`;
  return p;
}

export async function fetchMeta(
  path: string,
  signal?: AbortSignal,
  mimo?: string | null,
  sourceMac?: string | null,
): Promise<Meta> {
  const url = `/api/meta?path=${encodeURIComponent(path)}${filterParams(mimo, sourceMac)}`;
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return (await res.json()) as Meta;
}

export async function fetchFilters(
  path: string,
  signal?: AbortSignal,
): Promise<Filters> {
  const url = `/api/filters?path=${encodeURIComponent(path)}`;
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return (await res.json()) as Filters;
}

export interface Tile {
  grid: Float32Array; // length = height * width, row-major, row 0 = highest subcarrier
  width: number;
  height: number;
  /** The window THIS tile covers. Columns are quantised to a fixed lattice,
   *  so the backend snaps the requested window outwards and reports what it
   *  actually served -- always containing the request, never equal to it by
   *  luck. Draw against these, not against what was asked for: assuming the
   *  request came back verbatim shifts the image by up to one column and
   *  brings back the crawling the lattice removed. */
  t0: number;
  t1: number;
  /** Seconds per column, and the lattice level it came from. A pan or a live
   *  poll at the same level keeps every column it already had. */
  dt: number;
  level: number;
  captureTMin: number;
  captureTMax: number; // the whole capture's extent, NOT this tile's window
  framesDecoded: number;
  totalInRange: number;
  exact: boolean;
  /** False when a correction metric had no absolute orientation to anchor
   *  to, so this tile's polarity is not comparable with another view's. */
  anchored: boolean;
  vmin: number;
  vmax: number;
  pLow: number; // 1st percentile of finite values — robust scale for amplitude
  pHigh: number; // 99th percentile — amplitude locks to this, not vmin/vmax
}

export type Metric =
  | "amplitude"
  | "phase"
  | "csi_ratio_amplitude"
  | "csi_ratio_phase"
  // Derived phase views. Unwrapped values are no longer angles on a circle,
  // so they take a sequential palette and an auto-fitted scale, not TWILIGHT
  // and a fixed [-pi, pi].
  | "phase_unwrapped"
  | "phase_detrended"
  | "csi_ratio_phase_unwrapped"
  // Swap-corrected ratio: same units and ranges as the uncorrected pair,
  // with frames whose rx streams arrived exchanged put back the right way up.
  | "csi_ratio_phase_corrected"
  | "csi_ratio_amplitude_corrected"
  // Unwrapped along time on the corrected ratio: accumulated phase, so it
  // leaves [-pi, pi] and takes a fitted scale like the amplitude metrics.
  | "csi_ratio_phase_time_unwrapped"
  // Delay-domain view of the raw channel (rx0/tx0), not the ratio:
  // abs(IFFT(amplitude, phase)) per frame. Row 0 is not a subcarrier here,
  // it is a delay tap, fftshifted onto the same centred axis the other
  // panels use -- but unlike the ratio's own CIR, the peak is not
  // zero-referenced (no CFO/SFO cancellation for a single channel), so it
  // sits off-centre by this capture's own uncalibrated timing offset.
  | "csi_cir";

export async function fetchTile(
  path: string,
  t0: number,
  t1: number,
  width: number,
  metric: Metric,
  signal?: AbortSignal,
  mimo?: string | null,
  sourceMac?: string | null,
  interpolate?: boolean,
): Promise<Tile> {
  const url =
    `/api/tile?path=${encodeURIComponent(path)}` +
    `&t0=${t0}&t1=${t1}&width=${width}&metric=${metric}` +
    filterParams(mimo, sourceMac) +
    // Omit when true: that is the backend's own default, and every existing
    // caller that never heard of this parameter must keep building the same
    // URL it always has.
    (interpolate === false ? "&interpolate=false" : "");
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  // The backend writes little-endian float32 explicitly (grid.astype("<f4")).
  // Every platform this runs on is little-endian, so a direct Float32Array view
  // over the ArrayBuffer is correct without a byte-swap.
  const grid = new Float32Array(await res.arrayBuffer());
  const h = res.headers;
  return {
    grid,
    width: parseInt(h.get("X-Tile-Width") ?? "0", 10),
    height: parseInt(h.get("X-Tile-Height") ?? "0", 10),
    // Fall back to the requested window for a backend older than the
    // lattice, which served exactly what it was asked for.
    t0: parseFloat(h.get("X-Tile-T0") ?? String(t0)),
    t1: parseFloat(h.get("X-Tile-T1") ?? String(t1)),
    dt: parseFloat(h.get("X-Tile-DT") ?? "0"),
    level: parseInt(h.get("X-Tile-Level") ?? "0", 10),
    captureTMin: parseFloat(h.get("X-Capture-TMin") ?? "0"),
    captureTMax: parseFloat(h.get("X-Capture-TMax") ?? "0"),
    framesDecoded: parseInt(h.get("X-Tile-Frames") ?? "0", 10),
    totalInRange: parseInt(h.get("X-Tile-Total") ?? "0", 10),
    exact: h.get("X-Tile-Exact") === "1",
    // Absent header means an older backend that always anchored implicitly.
    anchored: h.get("X-Tile-Anchored") !== "0",
    vmin: parseFloat(h.get("X-Tile-VMin") ?? "0"),
    vmax: parseFloat(h.get("X-Tile-VMax") ?? "0"),
    pLow: parseFloat(h.get("X-Tile-PLow") ?? "0"),
    pHigh: parseFloat(h.get("X-Tile-PHigh") ?? "0"),
  };
}

/** Metrics /api/doppler accepts. Both are real-valued series: amplitude, and
 *  the time-unwrapped ratio phase. Raw wrapped phase is deliberately absent —
 *  its 2π jumps are broadband steps that would dominate an FFT and read as
 *  motion that is not there. */
export type DopplerMetric = "amplitude" | "csi_ratio_phase_time_unwrapped";

export interface DopplerTile extends Tile {
  /** The capture's own median frame rate over the frames in range — not a
   *  function of any requested width. */
  fs: number;
  /** Nyquist, fs/2. The axis runs 0..fMax and is one-sided: real input means
   *  the sign of the Doppler shift is not recoverable, so approaching and
   *  receding motion are indistinguishable. Motion above fMax aliases. */
  fMax: number;
  win: number;
  hop: number;
  winSeconds: number;
}

export async function fetchDoppler(
  path: string,
  t0: number,
  t1: number,
  metric: DopplerMetric,
  winSeconds: number,
  signal?: AbortSignal,
  mimo?: string | null,
  sourceMac?: string | null,
  interpolate?: boolean,
): Promise<DopplerTile> {
  const url =
    `/api/doppler?path=${encodeURIComponent(path)}` +
    `&t0=${t0}&t1=${t1}&metric=${metric}&win_seconds=${winSeconds}` +
    filterParams(mimo, sourceMac) +
    (interpolate === false ? "&interpolate=false" : "");
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  const grid = new Float32Array(await res.arrayBuffer());
  const h = res.headers;
  return {
    grid,
    width: parseInt(h.get("X-Doppler-Width") ?? "0", 10),
    height: parseInt(h.get("X-Doppler-Height") ?? "0", 10),
    // Column centres, not the requested range: a column is centred on its
    // window, so the first sits half a window inside t0. Labelling the axis
    // from the request would draw every column half a window too early.
    t0: parseFloat(h.get("X-Doppler-ColT0") ?? "0"),
    t1: parseFloat(h.get("X-Doppler-ColT1") ?? "0"),
    // Doppler columns are STFT windows, laid out by the hop the backend
    // chose, not by the tile lattice — so there is no level to report and dt
    // is read off the columns themselves.
    dt: 0,
    level: 0,
    captureTMin: parseFloat(h.get("X-Capture-TMin") ?? "0"),
    captureTMax: parseFloat(h.get("X-Capture-TMax") ?? "0"),
    framesDecoded: parseInt(h.get("X-Doppler-Frames") ?? "0", 10),
    totalInRange: parseInt(h.get("X-Doppler-Frames") ?? "0", 10),
    // Columns are windows, never stride-sampled frames, so there is no
    // inexact case to report and nothing to anchor against another view.
    exact: true,
    anchored: true,
    vmin: parseFloat(h.get("X-Tile-VMin") ?? "0"),
    vmax: parseFloat(h.get("X-Tile-VMax") ?? "0"),
    pLow: parseFloat(h.get("X-Tile-PLow") ?? "0"),
    pHigh: parseFloat(h.get("X-Tile-PHigh") ?? "0"),
    fs: parseFloat(h.get("X-Doppler-Fs") ?? "0"),
    fMax: parseFloat(h.get("X-Doppler-FMax") ?? "0"),
    win: parseInt(h.get("X-Doppler-Win") ?? "0", 10),
    hop: parseInt(h.get("X-Doppler-Hop") ?? "0", 10),
    winSeconds: parseFloat(h.get("X-Doppler-WinSeconds") ?? "0"),
  };
}

/** The per-subcarrier signal the presence detector runs on. `complex` is the
 *  default and the one to trust: amplitude and phase have complementary
 *  Fresnel blind spots, so a chest invisible in one shows in the other, and
 *  keeping the ratio complex avoids having to choose. The real channels are
 *  diagnostic — which one carries the signal says where the subject is
 *  sitting relative to the antennas. */
export type PresenceChannel = "complex" | "phase" | "magnitude";

/** One verdict per analysis window.
 *
 *  `unknown` exists so that absence is never claimed for free. A window
 *  assembled mostly from samples interpolated across a capture dropout comes
 *  out flat, and flat scores exactly like an empty room — and so does every
 *  window when no empty-room reference was given at all, because `empty`
 *  means "matched a room known to be empty" and there is nothing to match. */
export type PresenceState = "unknown" | "moving" | "present" | "empty";

export interface PresenceParams {
  channel: PresenceChannel;
  window_seconds: number;
  hop_seconds: number;
  rate_band_rpm: [number, number];
  bandpass_hz: [number, number];
  motion_frac_lo: number;
  motion_frac_hi: number;
  tonality_flat_lo: number;
  tonality_flat_hi: number;
  max_gap_fraction: number;
  smooth_windows: number;
  present_threshold: number;
  baseline_dev_k: number;
  motion_ratio_hi: number;
}

/** What the empty-room reference range measured, or `null` when none was
 *  given. `devP95` is how far that room's own windows strayed from its
 *  profile — the unit `baselineDev` is judged in — and `motionFloor` is its
 *  fractional-motion noise floor, which is never zero. */
export interface PresenceReference {
  devP95: number;
  motionFloor: number;
  nWindows: number;
}

/** Series are aligned with `timeS` and hold `null` where a window has no
 *  answer — a break in the line, never a zero. */
export interface Presence {
  timeS: number[];
  state: PresenceState[];
  score: (number | null)[];
  periodicity: (number | null)[];
  tonality: (number | null)[];
  motionGate: (number | null)[];
  motionLevel: (number | null)[];
  /** Fractional motion as a multiple of the reference room's own floor.
   *  Dimensionless, so one threshold works across radios; `null` throughout
   *  when no reference was given. */
  motionRatio: (number | null)[];
  /** How far this window's channel sits from the empty-room profile, in dB.
   *  The only evidence that can see a motionless occupant: every other series
   *  here is mean-removed, and a body parked in a room is a mean. */
  baselineDev: (number | null)[];
  /** Whether the breathing score cleared `present_threshold` here. Evidence
   *  only — it does not decide occupancy, and `rateRpm` is `null` where it is
   *  false. */
  breathing: boolean[];
  rateRpm: (number | null)[];
  unknown: boolean[];
  /** The capture's own median frame rate, not a function of any width. */
  fsHz: number;
  win: number;
  hop: number;
  /** What the window actually was, which is not what was asked for once a
   *  zoom is narrower than the requested window and it gets clamped. */
  windowSeconds: number;
  /** The slowest rate this window length can actually resolve. Above the
   *  requested floor means slower breathing is out of reach here. */
  rpmFloorEff: number;
  /** `baselineDev` above this is an occupant. `null` without a reference. */
  baselineDevThreshold: number | null;
  reference: PresenceReference | null;
  framesUsed: number;
  framesWithoutRatio: number;
  captureTMin: number;
  captureTMax: number;
  params: PresenceParams;
  warnings: string[];
}

export interface PresenceOptions {
  channel?: PresenceChannel;
  windowSeconds?: number;
  hopSeconds?: number;
  rpmLo?: number;
  rpmHi?: number;
  presentThreshold?: number;
  motionFracLo?: number;
  motionFracHi?: number;
  /** A stretch of capture known to be empty. Both ends or neither. Without
   *  it the detector reports motion but never absence. */
  refT0?: number | null;
  refT1?: number | null;
  /** Capture holding the reference range; defaults to the analysed one. */
  refPath?: string | null;
  baselineDevK?: number;
  motionRatioHi?: number;
  mimo?: string | null;
  sourceMac?: string | null;
  interpolate?: boolean;
}

export async function fetchPresence(
  path: string,
  t0: number,
  t1: number,
  options: PresenceOptions = {},
  signal?: AbortSignal,
): Promise<Presence> {
  const {
    channel = "complex",
    windowSeconds = 30,
    hopSeconds = 1,
    rpmLo = 9,
    rpmHi = 30,
    presentThreshold = 0.25,
    motionFracLo = 0.1,
    motionFracHi = 0.25,
    refT0,
    refT1,
    refPath,
    baselineDevK = 3,
    motionRatioHi = 2,
    mimo,
    sourceMac,
    interpolate,
  } = options;

  const url =
    `/api/presence?path=${encodeURIComponent(path)}` +
    `&t0=${t0}&t1=${t1}&channel=${channel}` +
    `&window_seconds=${windowSeconds}&hop_seconds=${hopSeconds}` +
    `&rpm_lo=${rpmLo}&rpm_hi=${rpmHi}&present_threshold=${presentThreshold}` +
    `&motion_frac_lo=${motionFracLo}&motion_frac_hi=${motionFracHi}` +
    `&baseline_dev_k=${baselineDevK}&motion_ratio_hi=${motionRatioHi}` +
    (refT0 != null && refT1 != null ? `&ref_t0=${refT0}&ref_t1=${refT1}` : "") +
    (refPath ? `&ref_path=${encodeURIComponent(refPath)}` : "") +
    filterParams(mimo, sourceMac) +
    (interpolate === false ? "&interpolate=false" : "");

  const res = await fetch(url, { signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  const body = await res.json();
  return {
    timeS: body.time_s,
    state: body.state,
    score: body.score,
    periodicity: body.periodicity,
    tonality: body.tonality,
    motionGate: body.motion_gate,
    motionLevel: body.motion_level,
    motionRatio: body.motion_ratio,
    baselineDev: body.baseline_dev,
    breathing: body.breathing,
    rateRpm: body.rate_rpm,
    unknown: body.unknown,
    fsHz: body.fs_hz,
    win: body.win,
    hop: body.hop,
    windowSeconds: body.window_seconds,
    rpmFloorEff: body.rpm_floor_eff,
    baselineDevThreshold: body.baseline_dev_threshold ?? null,
    reference: body.reference
      ? {
          devP95: body.reference.dev_p95,
          motionFloor: body.reference.motion_floor,
          nWindows: body.reference.n_windows,
        }
      : null,
    framesUsed: body.frames_used,
    framesWithoutRatio: body.frames_without_ratio,
    captureTMin: body.t_min,
    captureTMax: body.t_max,
    params: body.params,
    warnings: body.warnings ?? [],
  };
}
