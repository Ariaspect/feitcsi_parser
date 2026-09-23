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
  /** Conditions the capture was recorded under, as far as they were recorded.
   *  Absent rather than null when unknown, so "not recorded" is distinguishable
   *  from a recorded blank. `scenario` falls back to what the camera saw
   *  (empty / partial / occupied) when nothing was declared. */
  room?: string;
  configuration?: string;
  scenario?: string;
  subject?: string;
  activity?: string;
  facing?: string;
  distance_m?: number;
  /** Camera-measured occupied fraction, present only with the scenario
   *  fallback above. */
  occupancy?: number;
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
  /** True when the receiver's per-gain-state amplitude distortion was removed
   *  from this tile. False means the values are exactly as decoded: the
   *  toggle is off, the metric is one the correction does not touch, the
   *  capture is not MediaTek, or its gain never stepped. */
  agcCorrected: boolean;
  /** How many gain states carried a correction. 0 when agcCorrected is false. */
  agcStates: number;
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
  // The vendored parser's own planes, produced by its functions rather than
  // derived from ours: RSSI-restored absolute amplitude, and the phase and
  // magnitude of its conjugate-across-rx feature. Bins its occupancy rule
  // rejects arrive as NaN, so they render blank rather than as measurements.
  | "lg_amplitude"
  | "lg_conj_phase"
  | "lg_conj_amplitude"
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
  agc?: boolean,
): Promise<Tile> {
  const url =
    `/api/tile?path=${encodeURIComponent(path)}` +
    `&t0=${t0}&t1=${t1}&width=${width}&metric=${metric}` +
    filterParams(mimo, sourceMac) +
    // Omit when true: that is the backend's own default, and every existing
    // caller that never heard of this parameter must keep building the same
    // URL it always has.
    (interpolate === false ? "&interpolate=false" : "") +
    (agc === false ? "&agc=false" : "");
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
    // Absent header means a backend older than the correction, which never
    // applied one -- so "not corrected" is the honest reading, not "unknown".
    agcCorrected: h.get("X-Tile-Agc") === "1",
    agcStates: parseInt(h.get("X-Tile-AgcStates") ?? "0", 10),
    vmin: parseFloat(h.get("X-Tile-VMin") ?? "0"),
    vmax: parseFloat(h.get("X-Tile-VMax") ?? "0"),
    pLow: parseFloat(h.get("X-Tile-PLow") ?? "0"),
    pHigh: parseFloat(h.get("X-Tile-PHigh") ?? "0"),
  };
}

/** Metrics /api/doppler accepts. `csi_ratio_complex` is the complex ratio
 *  itself and is the one to read a room from: it is the only one whose
 *  Doppler is *signed*, and it never unwraps anything. The other two are real
 *  series, so their spectra are conjugate-symmetric and approaching motion
 *  lands on the same row as receding. Raw wrapped phase is deliberately
 *  absent — its 2π jumps are broadband steps that read as motion that is not
 *  there. */
export type DopplerMetric =
  | "csi_ratio_complex"
  | "amplitude"
  | "csi_ratio_phase_time_unwrapped";

export interface DopplerTile extends Tile {
  /** The capture's own median frame rate over the frames in range — not a
   *  function of any requested width. */
  fs: number;
  /** Bottom of the frequency axis. 0 for the real metrics, whose spectra are
   *  conjugate-symmetric — approaching and receding motion land on the same
   *  row there. About -fs/2 for `csi_ratio_complex`, where the sign survives
   *  and the two directions are separate rows. */
  fMin: number;
  /** Top of the frequency axis, Nyquist. Motion above it aliases: at 5 GHz a
   *  1 m/s movement sits near 33 Hz, far above any capture here. */
  fMax: number;
  win: number;
  hop: number;
  winSeconds: number;
  /** The fixed colour range this panel should use, in dB. Doppler values are
   *  normalised by the taper's coherent gain and served in dB, so the same
   *  number means the same motion in every capture. Auto-fitting instead is
   *  what made two captures of the same room look different: one spanning
   *  0..0.2 in raw magnitude and another 0..7.0, both stretched across the
   *  whole ramp. */
  scaleMin: number;
  scaleMax: number;
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
    scaleMin: parseFloat(h.get("X-Doppler-ScaleMin") ?? "-55"),
    scaleMax: parseFloat(h.get("X-Doppler-ScaleMax") ?? "-15"),
    // The Doppler path runs on the CSI ratio, which divides the gain out.
    agcCorrected: false,
    agcStates: 0,
    vmin: parseFloat(h.get("X-Tile-VMin") ?? "0"),
    vmax: parseFloat(h.get("X-Tile-VMax") ?? "0"),
    pLow: parseFloat(h.get("X-Tile-PLow") ?? "0"),
    pHigh: parseFloat(h.get("X-Tile-PHigh") ?? "0"),
    fs: parseFloat(h.get("X-Doppler-Fs") ?? "0"),
    fMin: parseFloat(h.get("X-Doppler-FMin") ?? "0"),
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
 *  given. `devScale` is how far that room's own windows typically strayed from
 *  its profile — the unit `baselineDev` is judged in — and `motionFloor` is its
 *  fractional-motion noise floor, which is never zero. `devP95` is the same
 *  quantity at the 95th percentile, which the threshold used until 20260914;
 *  a `devP95` far above `devScale` means the reference range never settled. */
export interface PresenceReference {
  devScale: number;
  devP95: number;
  motionFloor: number;
  nWindows: number;
  /** How many known-empty stretches were pooled. More than one means the
   *  verdict measures distance to the NEAREST empty state, not to their
   *  average — which is what keeps a changed room from reading as occupied. */
  nRanges?: number;
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
  /** Known-empty stretches. Several are meaningful, not redundant: a room
   *  that changed while the capture ran is empty in more than one way, and a
   *  reference holding only the first calls every later empty window
   *  occupied. Sent as repeated query parameters, paired in order. */
  refT0?: number | number[] | null;
  refT1?: number | number[] | null;
  /** Capture holding the reference range; defaults to the analysed one. */
  refPath?: string | null;
  baselineDevK?: number;
  motionRatioHi?: number;
  mimo?: string | null;
  sourceMac?: string | null;
  interpolate?: boolean;
}

/** Pair up the reference ranges as repeated `ref_t0`/`ref_t1` parameters.
 *  A bare `${array}` would join with commas and the server would reject it. */
function refParams(
  t0: number | number[] | null | undefined,
  t1: number | number[] | null | undefined,
): string {
  if (t0 == null || t1 == null) return "";
  const starts = Array.isArray(t0) ? t0 : [t0];
  const ends = Array.isArray(t1) ? t1 : [t1];
  if (starts.length !== ends.length) return "";
  return starts
    .map((s, i) => `&ref_t0=${s}&ref_t1=${ends[i]}`)
    .join("");
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
    refParams(refT0, refT1) +
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
          devScale: body.reference.dev_scale ?? body.reference.dev_p95,
          devP95: body.reference.dev_p95,
          motionFloor: body.reference.motion_floor,
          nWindows: body.reference.n_windows,
          nRanges: body.reference.n_ranges,
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

/** Ground truth recorded beside a capture by the labelled-run wrapper. */
export interface LabelPresence {
  timeS: number[];
  present: boolean[];
  maxConf: number[];
  roi: number[] | null;
  model: string | null;
}

export interface LabelPhase {
  label: string;
  t0: number;
  t1: number;
}

export interface Labels {
  /** Per-frame webcam detections, or null when no `_cv.json` sits beside the
   *  capture. Times are relative to the capture's first sample. */
  present: LabelPresence | null;
  /** The run's INTENDED protocol, not observed truth — measured transitions
   *  have run 2-18 s late. Where the two disagree, `present` is the evidence. */
  phases: LabelPhase[] | null;
  position?: string | null;
  source: string | null;
  captureStartUtcEpoch: number | null;
}

export async function fetchLabels(
  path: string,
  signal?: AbortSignal,
): Promise<Labels> {
  const url = `/api/labels?path=${encodeURIComponent(path)}`;
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`labels: ${res.status}`);
  return res.json();
}

/** Output of the vendored MT7921 parser's processing over a capture. */
export interface LgParse {
  frames: number;
  framesInFile: number;
  nrx: number;
  ntx: number;
  nsub: number;
  /** Subcarriers their occupancy rule judges to carry data. Fewer than ours
   *  keeps: they take bins measured non-zero, we interpolate pilots and DC. */
  activeBins: number[];
  occupancy: number[];
  rawSpectrum: (number | null)[];
  /** Amplitude after their RSSI-based AGC restoration, so absolute dBm rather
   *  than the chip's relative scale. */
  agcSpectrum: (number | null)[];
  /** Lag-1 phase coherence per candidate feature. `conj_rx` is their default;
   *  `conj_tx` is the axis this project divides along. The gap between them is
   *  the substantive disagreement between the two parsers. */
  coherence: Record<string, number>;
  rssiMean: number;
  twoStreamFrames: number;
  peer: string | null;
  macCensus: [string, number][];
  tMin: number;
  tMax: number;
  toolVersion: string;
}

export async function fetchLgParse(
  path: string,
  maxFrames = 4096,
  signal?: AbortSignal,
): Promise<LgParse> {
  const url = `/api/lgparse?path=${encodeURIComponent(path)}&max_frames=${maxFrames}`;
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`lgparse: ${res.status}`);
  return res.json();
}

/** What the LG on-board detector said when replayed over a capture. */
export interface LgDetect {
  events: { kind: "+" | "-"; t: number }[];
  intervals: { t0: number; t1: number }[];
  records: number;
  parseFailures: number;
  duration: number;
  threshold: number;
  absenceDuration: number;
  /** The interpreter it ran under. NumPy 1.x is not incidental: under NumPy 2
   *  its TLV walk desynchronises at the first CSI field, silently. */
  numpy: string;
  python: string;
  cached: boolean;
  truth: {
    timeS: number[];
    present: boolean[];
    tp: number; fp: number; fn: number; tn: number;
    /** Empty frames within `marginSeconds` of a transition, scored in
     *  neither direction. */
    excluded: number;
    marginSeconds: number;
    accuracy: number;
    precision: number;
    recall: number;
    /** What it would score by always saying "present" — the bar to clear. */
    baseRate: number;
  } | null;
}

export async function fetchLgDetect(
  path: string,
  threshold = 26,
  absence = 10,
  signal?: AbortSignal,
): Promise<LgDetect> {
  const url =
    `/api/lgdetect?path=${encodeURIComponent(path)}` +
    `&threshold=${threshold}&absence=${absence}`;
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `lgdetect: ${res.status}`);
  }
  return res.json();
}

/** One cell-count set, plus the rates derived from it. Nulls where a rate has
 *  no denominator — a capture with no empty window has no specificity, and
 *  saying 0% would be a different claim from saying "not measured". */
export interface Confusion {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  total: number;
  /** Cells the camera covered but could not vouch for: empty frames within
   *  the margin of a transition, or a span neither empty nor occupied. Scored
   *  in neither direction, and reported so the discard is visible. */
  excluded: number;
  accuracy: number | null;
  recall: number | null;
  specificity: number | null;
  precision: number | null;
}

/** Both detectors resampled onto one grid and scored against the camera.
 *
 *  `ours` is null when the capture could not be calibrated — the reference has
 *  to come from other camera-empty captures, never this one's own stretches,
 *  and there have to be at least two. `calibrationNote` says which condition
 *  failed. `lg` is always present: it compares each frame to the one before and
 *  so needs no reference at all, which is exactly the trade the two make. */
export interface Phase1 {
  path: string;
  gridSeconds: number;
  /** Seconds of the empty label next to each transition that were not scored. */
  marginSeconds: number;
  timeS: number[];
  groundTruth: { timeS: number[]; present: boolean[] };
  calibrated: boolean;
  calibrationNote?: string;
  /** Present when the reference pool sits further from the capture than the
   *  default window allows. Nothing else detects a pool from a different room,
   *  so a widened window is the one thing worth saying out loud. */
  referenceWarning?: string;
  ours: {
    present: boolean[];
    threshold: number;
    devScale: number;
    /** Median pairwise distance between the pooled references. Small means they
     *  describe one room; large means the pool spans two and its scale would
     *  measure the gap between them. */
    poolSpread: number;
    /** How far the capture's quietest window still sits from the pool. */
    minDeviation: number;
    /** That distance in thresholds. Diagnostic only: it does NOT separate a
     *  usable calibration from a broken one. Measured over 22 working
     *  calibrations it spans 0.09–5.65, while two known-broken ones read 4.94
     *  and 5.20 — fully inside that range, because the numerator also carries
     *  how occupied the capture is and the denominator how tight the pool is. */
    applicability: number | null;
    /** Hours between this capture and its nearest reference. This is the real
     *  guard against a pool from a different room, so it is shown. */
    referenceAgeH: number;
    references: string[];
    confusion: Confusion;
  } | null;
  lg: {
    present: boolean[];
    threshold: number;
    absence: number;
    events: number;
    confusion: Confusion;
  };
}

export async function fetchPhase1(
  path: string,
  opts: { grid?: number; k?: number; lgThreshold?: number; lgAbsence?: number } = {},
  signal?: AbortSignal,
): Promise<Phase1> {
  const { grid = 1, k = 3, lgThreshold = 26, lgAbsence = 10 } = opts;
  const url =
    `/api/phase1?path=${encodeURIComponent(path)}` +
    `&grid=${grid}&k=${k}&lg_threshold=${lgThreshold}&lg_absence=${lgAbsence}`;
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `phase1: ${res.status}`);
  }
  return res.json();
}

/** One window of the FarSense replay in full, for the detail panels. */
export interface FarSenseDetail {
  index: number;
  startS: number;
  tS: number[];
  /** Subcarrier with the largest BNR in this window, in capture bin numbers. */
  bestSc: number;
  /** Its projection axis, radians from the real axis. */
  bestTheta: number;
  /** Its smoothed, mean-removed ratio over the window — the arc of Fig. 11. */
  iq: [number, number][];
  /** Its respiration pattern: the projection onto `bestTheta`. */
  pattern: (number | null)[];
  /** BNR-weighted sum of the selected subcarriers' autocorrelations, by lag. */
  acf: (number | null)[];
  lagLo: number;
  lagHi: number;
  /** Where the first in-band peak was read, or null when there was none. */
  lag: number | null;
  scIndex: number[];
  bnr: (number | null)[];
  theta: (number | null)[];
  selected: boolean[];
}

export interface FarSenseParams {
  window_seconds: number;
  hop_seconds: number;
  band_rpm: [number, number];
  n_theta: number;
  fft_size: number;
  keep_fraction: number;
  savgol_seconds: number;
  savgol_order: number;
  highpass_hz: number;
  motion_frac_hi: number;
  max_gap_fraction: number;
  min_peak: number;
}

/** FarSense (Zeng et al. 2019) replayed over a range. Series are aligned
 *  with `timeS`; `null` is a window with no answer. `rpm` is null wherever
 *  the window was not stationary, had no in-band autocorrelation peak, or
 *  the peak fell below `min_peak`. */
export interface FarSense {
  timeS: number[];
  stationary: boolean[];
  motionLevel: (number | null)[];
  unknown: boolean[];
  rpm: (number | null)[];
  lag: (number | null)[];
  /** Height of the first peak of the combined autocorrelation, in units of
   *  summed BNR (the paper's Eq. 11, unnormalised). */
  acfPeak: (number | null)[];
  /** The same divided by the summed BNR: the weighted mean autocorrelation at
   *  the peak lag, in -1..1. The number to judge a rate by. */
  acfPeakNorm: (number | null)[];
  bnrMax: (number | null)[];
  nSelected: number[];
  bestSc: number[];
  bestTheta: (number | null)[];
  /** Capture bin number of each row of `bnrMap`. */
  scIndex: number[];
  /** BNR per live subcarrier (rows) per window (columns). */
  bnrMap: (number | null)[][];
  /** Multiply a BNR by this to put it on 0..1, where 1 is a pure tone. */
  bnrNormFactor: number;
  /** The best subcarrier's respiration pattern, stitched across windows and
   *  scaled to unit deviation, on the sample grid. */
  patternT: number[];
  pattern: (number | null)[];
  detail: FarSenseDetail | null;
  win: number;
  hop: number;
  windowSeconds: number;
  fsHz: number;
  lagLo: number;
  lagHi: number;
  params: FarSenseParams;
  framesUsed: number;
  framesWithoutRatio: number;
  captureTMin: number;
  captureTMax: number;
}

export interface FarSenseOptions {
  windowSeconds?: number;
  hopSeconds?: number;
  rpmLo?: number;
  rpmHi?: number;
  nTheta?: number;
  keepFraction?: number;
  savgolSeconds?: number;
  savgolOrder?: number;
  /** Zero-phase high-pass before smoothing, Hz. 0 is the paper. */
  highpassHz?: number;
  motionFracHi?: number;
  minPeak?: number;
  /** A time in seconds; the window nearest it comes back under `detail`. */
  detailT?: number | null;
  mimo?: string | null;
  sourceMac?: string | null;
  interpolate?: boolean;
}

export async function fetchFarSense(
  path: string,
  t0: number,
  t1: number,
  options: FarSenseOptions = {},
  signal?: AbortSignal,
): Promise<FarSense> {
  const {
    windowSeconds = 10,
    hopSeconds = 1,
    rpmLo = 10,
    rpmHi = 30,
    nTheta = 200,
    keepFraction = 0.65,
    savgolSeconds = 1.0,
    savgolOrder = 3,
    highpassHz = 0,
    motionFracHi = 0.25,
    minPeak = 0.2,
    detailT,
    mimo,
    sourceMac,
    interpolate,
  } = options;

  const url =
    `/api/farsense?path=${encodeURIComponent(path)}` +
    `&t0=${t0}&t1=${t1}` +
    `&window_seconds=${windowSeconds}&hop_seconds=${hopSeconds}` +
    `&rpm_lo=${rpmLo}&rpm_hi=${rpmHi}&n_theta=${nTheta}` +
    `&keep_fraction=${keepFraction}` +
    `&savgol_seconds=${savgolSeconds}&savgol_order=${savgolOrder}` +
    `&highpass_hz=${highpassHz}` +
    `&motion_frac_hi=${motionFracHi}&min_peak=${minPeak}` +
    (detailT != null ? `&detail_t=${detailT}` : "") +
    filterParams(mimo, sourceMac) +
    (interpolate === false ? "&interpolate=false" : "");

  const res = await fetch(url, { signal });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `farsense: ${res.status}`);
  }
  const body = await res.json();
  const d = body.detail;
  return {
    timeS: body.time_s,
    stationary: body.stationary,
    motionLevel: body.motion_level,
    unknown: body.unknown,
    rpm: body.rpm,
    lag: body.lag,
    acfPeak: body.acf_peak,
    acfPeakNorm: body.acf_peak_norm,
    bnrMax: body.bnr_max,
    nSelected: body.n_selected,
    bestSc: body.best_sc,
    bestTheta: body.best_theta,
    scIndex: body.sc_index,
    bnrMap: body.bnr_map,
    bnrNormFactor: body.bnr_norm_factor,
    patternT: body.pattern_t,
    pattern: body.pattern,
    detail: d
      ? {
          index: d.index,
          startS: d.start_s,
          tS: d.t_s,
          bestSc: d.best_sc,
          bestTheta: d.best_theta,
          iq: d.iq,
          pattern: d.pattern,
          acf: d.acf,
          lagLo: d.lag_lo,
          lagHi: d.lag_hi,
          lag: d.lag ?? null,
          scIndex: d.sc_index,
          bnr: d.bnr,
          theta: d.theta,
          selected: d.selected,
        }
      : null,
    win: body.win,
    hop: body.hop,
    windowSeconds: body.window_seconds,
    fsHz: body.fs_hz,
    lagLo: body.lag_lo,
    lagHi: body.lag_hi,
    params: body.params,
    framesUsed: body.frames_used,
    framesWithoutRatio: body.frames_without_ratio,
    captureTMin: body.t_min,
    captureTMax: body.t_max,
  };
}

/** One second of the calibration-free detector. `moving` and `breathing`
 *  are the evidence; `held` is presence kept alive by a hold running out
 *  from evidence (after any, before breathing); `bridged` is a gap whose
 *  holds met from both sides; `empty` is the room once nothing has
 *  happened for that long. */
export type HybridState = "unknown" | "moving" | "breathing" | "held" | "bridged" | "empty";

export interface HybridParams {
  use_amplitude: boolean;
  hold_seconds: number;
  burst_seconds: number;
  motion_rel: number;
  motion_abs: number;
  amp_rel: number;
  amp_abs: number;
  floor_percentile: number;
  breath_min_peak: number;
  breath_persist_seconds: number;
  breath_rate_tol: number;
  sparse_fraction: number;
  sparse_window_seconds: number;
  breath_min_fraction: number;
  breath_window_seconds: number;
  breath_highpass_hz: number;
  max_gap_fraction: number;
  rpm_lo: number;
  rpm_hi: number;
  n_theta: number;
  fft_size: number;
  keep_fraction: number;
  savgol_seconds: number;
  savgol_order: number;
  /** FarSense stationary gate; null when off. */
  motion_frac_hi: number | null;
  positive_only: boolean;
  motion_floor: number | null;
  lead_hold: boolean;
  bridge_bursts: "off" | "run" | "any";
}

export interface HybridConfusion extends Confusion {
  /** Fraction of scored seconds the camera called occupied — what always
   *  saying "present" would score. */
  baseRate: number | null;
  marginSeconds: number;
}

export interface Hybrid {
  timeS: number[];
  present: boolean[];
  state: HybridState[];
  unknown: boolean[];
  /** Per-second median |Δr|/|r|; null where the second was a dropout. */
  motionRatio: (number | null)[];
  /** Per-second median |ΔA| in dB on the raw amplitude. */
  motionAmp: (number | null)[];
  burst: boolean[];
  breathing: boolean[];
  breathPeak: (number | null)[];
  breathRpm: (number | null)[];
  /** The range's own quiet level and the threshold derived from it. */
  ratioFloor: number | null;
  ratioThreshold: number | null;
  ampFloor: number | null;
  ampThreshold: number | null;
  /** Why the breathing channel could not run, when it could not. */
  breathNote: string | null;
  fsHz: number;
  params: HybridParams;
  framesUsed: number;
  framesWithoutRatio: number;
  captureTMin: number;
  captureTMax: number;
  /** Where the floor came from: 'own' (this range's 20th percentile) or 'explicit'. */
  floorScope: string;
  truth: { timeS: number[]; present: boolean[] } | null;
  /** Why the capture's labels are not scored, when its sidecar says so. */
  truthExcluded: string | null;
  confusion: HybridConfusion | null;
}

export interface HybridOptions {
  useAmplitude?: boolean;
  holdS?: number;
  burstS?: number;
  motionRel?: number;
  motionAbs?: number;
  ampRel?: number;
  ampAbs?: number;
  floorPct?: number;
  breathMinPeak?: number;
  breathPersistS?: number;
  breathRateTol?: number;
  /** Sparse breathing: fraction of windows within `sparseWindow` seconds that must clear the peak threshold; 0 = off. */
  sparseFraction?: number;
  sparseWindow?: number;
  breathWindow?: number;
  breathHighpass?: number;
  /** FarSense search knobs, same meaning as on the FarSense tab. */
  rpmLo?: number;
  rpmHi?: number;
  nTheta?: number;
  fftSize?: number;
  keepFraction?: number;
  savgolSeconds?: number;
  savgolOrder?: number;
  /** The paper's stationary gate; null or undefined leaves it off. */
  motionFracHi?: number | null;
  positiveOnly?: boolean;
  maxGapFraction?: number;
  /** An explicit quiet |Δr|/|r| level in place of the range's own 20th
   *  percentile. Omit for the range's own. */
  motionFloor?: number | null;
  /** Breathing also holds presence `holdS` before it, and a gap whose
   *  holds meet is present throughout. On by default. */
  leadHold?: boolean;
  /** Fill the whole stretch between two bursts when breathing lies between
   *  them: 'run' needs a breathing run, 'any' a single qualifying window. */
  bridgeBursts?: "off" | "run" | "any";
  marginS?: number;
  mimo?: string | null;
  sourceMac?: string | null;
  interpolate?: boolean;
}

export async function fetchHybrid(
  path: string,
  t0: number,
  t1: number,
  options: HybridOptions = {},
  signal?: AbortSignal,
): Promise<Hybrid> {
  const {
    useAmplitude = false,
    holdS = 20,
    burstS = 2,
    motionRel = 2,
    motionAbs = 0.1,
    ampRel = 2,
    ampAbs = 0.5,
    floorPct = 20,
    breathMinPeak = 0.2,
    breathPersistS = 10,
    breathRateTol = 3,
    sparseFraction = 0,
    sparseWindow = 60,
    breathWindow = 10,
    breathHighpass = 0,
    rpmLo = 10,
    rpmHi = 30,
    nTheta = 200,
    fftSize = 8192,
    keepFraction = 0.65,
    savgolSeconds = 1.0,
    savgolOrder = 3,
    motionFracHi,
    positiveOnly = true,
    maxGapFraction = 0.5,
    motionFloor,
    leadHold = true,
    bridgeBursts = "off",
    marginS = 5,
    mimo,
    sourceMac,
    interpolate,
  } = options;

  const url =
    `/api/hybrid?path=${encodeURIComponent(path)}` +
    `&t0=${t0}&t1=${t1}` +
    `&use_amplitude=${useAmplitude}&hold_s=${holdS}&burst_s=${burstS}` +
    `&motion_rel=${motionRel}&motion_abs=${motionAbs}` +
    `&amp_rel=${ampRel}&amp_abs=${ampAbs}&floor_pct=${floorPct}` +
    `&breath_min_peak=${breathMinPeak}&breath_persist_s=${breathPersistS}` +
    `&breath_rate_tol=${breathRateTol}&breath_window=${breathWindow}` +
    `&sparse_fraction=${sparseFraction}&sparse_window=${sparseWindow}` +
    `&breath_highpass=${breathHighpass}&margin_s=${marginS}` +
    `&rpm_lo=${rpmLo}&rpm_hi=${rpmHi}&n_theta=${nTheta}&fft_size=${fftSize}` +
    `&keep_fraction=${keepFraction}&savgol_seconds=${savgolSeconds}&savgol_order=${savgolOrder}` +
    (motionFracHi != null ? `&motion_frac_hi=${motionFracHi}` : "") +
    `&positive_only=${positiveOnly}&max_gap_fraction=${maxGapFraction}` +
    (motionFloor != null ? `&motion_floor=${motionFloor}` : "") +
    `&lead_hold=${leadHold}&bridge_bursts=${bridgeBursts}` +
    filterParams(mimo, sourceMac) +
    (interpolate === false ? "&interpolate=false" : "");

  const res = await fetch(url, { signal });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `hybrid: ${res.status}`);
  }
  const body = await res.json();
  const c = body.confusion;
  return {
    timeS: body.time_s,
    present: body.present,
    state: body.state,
    unknown: body.unknown,
    motionRatio: body.motion_ratio,
    motionAmp: body.motion_amp,
    burst: body.burst,
    breathing: body.breathing,
    breathPeak: body.breath_peak,
    breathRpm: body.breath_rpm,
    ratioFloor: body.ratio_floor ?? null,
    ratioThreshold: body.ratio_threshold ?? null,
    ampFloor: body.amp_floor ?? null,
    ampThreshold: body.amp_threshold ?? null,
    breathNote: body.breath_note ?? null,
    fsHz: body.fs_hz,
    params: body.params,
    framesUsed: body.frames_used,
    framesWithoutRatio: body.frames_without_ratio,
    captureTMin: body.t_min,
    captureTMax: body.t_max,
    floorScope: body.floor_scope ?? "own",
    truth: body.truth ? { timeS: body.truth.time_s, present: body.truth.present } : null,
    truthExcluded: body.truth_excluded ?? null,
    confusion: c
      ? {
          tp: c.tp, fp: c.fp, fn: c.fn, tn: c.tn, total: c.total, excluded: c.excluded,
          accuracy: c.accuracy, recall: c.recall, specificity: c.specificity, precision: c.precision,
          baseRate: c.base_rate ?? null,
          marginSeconds: c.margin_s,
        }
      : null,
  };
}
