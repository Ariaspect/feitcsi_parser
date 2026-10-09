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

// --------------------------------------------------------------------------- #
//  Frame-to-frame amplitude step (/api/frame-diff)
// --------------------------------------------------------------------------- #

/** One column of the decimated per-step series. Columns are equal in time, and
 *  each carries a median and the extremes it spans — see `bandPath` for why the
 *  extremes are what gets drawn. */
export type FrameDiffSignal = "amplitude" | "ratio_amp" | "ratio_complex";

export interface FrameDiff {
  /** Which series was differenced. */
  signal: FrameDiffSignal;
  /** Column centres, on the capture's clock. */
  timeS: number[];
  /** Median of the signed fold: + the array brightening, − fading. */
  signed: (number | null)[];
  /** The column's extremes of the signed fold. A single frame survives here. */
  signedLo: (number | null)[];
  signedHi: (number | null)[];
  /** Median of |d|, which cannot cancel when subcarriers disagree in sign.
   *  This is `hybrid.amplitude_diff` on the bounded axis, exactly. */
  magnitude: (number | null)[];
  magnitudeHi: (number | null)[];
  /** The MEAN fold, and its modulus: what the subcarriers agree about, with
   *  what they disagree about cancelled. The channel's common mode. */
  common: (number | null)[];
  commonHi: (number | null)[];
  /** Frames behind each column, blanked ones included. */
  count: number[];
  binSeconds: number;
  /** False when the range held fewer steps than columns asked for, so every
   *  column is one frame pair and the envelope is the value itself. */
  decimated: boolean;
  framesUsed: number;
  /** Frames in range the uniformity rule removed: another transmitter, another
   *  MIMO mode, or a narrower bandwidth. A step between two frame shapes is
   *  bookkeeping, not motion. */
  framesDropped: number;
  framesDroppedNarrow: number;
  /** The frame set actually used, so the panel never has to assume. */
  sourceMac: string | null;
  mimo: [number, number] | null;
  selectionNote: string;
  nSubcarriers: number;
  captureTMin: number;
  captureTMax: number;
  summary: {
    steps: number;
    stepsMeasured: number;
    /** Subcarriers a typical step was folded over — the array width less the
     *  guard band and the dead bins. */
    liveMedian: number;
    commonMedian: number | null;
    nBridged: number;
    /** Frame pairs that crossed a reported gain state. Always counted; blanked
     *  only when `gateGain` was asked for. 84-100% of the loudest 1% of steps
     *  are these, so the number is owed to the reader either way. */
    nGainCrossed: number;
    gainGated: boolean;
    gapLimit: number;
    median: number | null;
    p99: number | null;
    max: number | null;
    /** The step in the units it was measured in: dB for the amplitude signals,
     *  degrees of rotation for the complex one, where a step IS an angle. */
    nativeUnit: string;
    medianNative: number | null;
    maxNative: number | null;
  };
}

export interface FrameDiffOptions {
  signal?: FrameDiffSignal;
  /** Blank the frame pairs that cross a gain state. Off by default. */
  gateGain?: boolean;
  maxPoints?: number;
  mimo?: string | null;
  sourceMac?: string | null;
  interpolate?: boolean;
}

export async function fetchFrameDiff(
  path: string,
  t0: number,
  t1: number,
  options: FrameDiffOptions = {},
  signal?: AbortSignal,
): Promise<FrameDiff> {
  const {
    signal: series = "amplitude", gateGain = false, maxPoints = 2000,
    mimo, sourceMac, interpolate,
  } = options;

  const url =
    `/api/frame-diff?path=${encodeURIComponent(path)}` +
    `&t0=${t0}&t1=${t1}` +
    `&max_points=${maxPoints}&signal=${series}` +
    (gateGain ? "&gate_gain=true" : "") +
    filterParams(mimo, sourceMac) +
    (interpolate === false ? "&interpolate=false" : "");

  const res = await fetch(url, { signal });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `frame diff: ${res.status}`);
  }
  const body = await res.json();
  const s = body.summary;
  return {
    timeS: body.time_s,
    signed: body.signed,
    signedLo: body.signed_lo,
    signedHi: body.signed_hi,
    magnitude: body.magnitude,
    magnitudeHi: body.magnitude_hi,
    common: body.common,
    commonHi: body.common_hi,
    count: body.count,
    binSeconds: body.bin_seconds,
    decimated: body.decimated,
    framesUsed: body.frames_used,
    framesDropped: body.frames_dropped,
    framesDroppedNarrow: body.frames_dropped_narrow,
    sourceMac: body.source_mac,
    mimo: body.mimo,
    selectionNote: body.selection_note,
    signal: body.signal,
    nSubcarriers: body.n_subcarriers,
    captureTMin: body.capture_t_min,
    captureTMax: body.capture_t_max,
    summary: {
      steps: s.steps,
      stepsMeasured: s.steps_measured,
      liveMedian: s.live_median,
      commonMedian: s.common_median,
      nBridged: s.n_bridged,
      nGainCrossed: s.n_gain_crossed,
      gainGated: s.gain_gated,
      gapLimit: s.gap_limit,
      median: s.median,
      p99: s.p99,
      max: s.max,
      nativeUnit: s.native_unit,
      medianNative: s.median_native,
      maxNative: s.max_native,
    },
  };
}

// --------------------------------------------------------------------------- #
//  Hybrid 2 (/api/hybrid2): the complex frame step for motion, FarSense for breath
// --------------------------------------------------------------------------- #

export interface Hybrid2 {
  timeS: number[];
  present: boolean[];
  state: HybridState[];
  unknown: boolean[];
  /** Per-second median of the ratio-complex frame step. The motion channel. */
  motion: (number | null)[];
  /** Hybrid 1's |Δr|/|r| on the same axis, for reference only — it decides
   *  nothing here. Exactly twice `motion` before the fold and the frame set
   *  differ, so the two are close but not equal. */
  motionReference: (number | null)[];
  burst: boolean[];
  breathing: boolean[];
  breathPeak: (number | null)[];
  breathRpm: (number | null)[];
  /** The range's own quiet level, and the threshold built from it. A range
   *  occupied throughout has no quiet stretch — its own floor IS the occupant,
   *  which is the detector's known blind spot rather than a bug. */
  floor: number | null;
  threshold: number | null;
  floorScope: "own" | "explicit";
  signal: string;
  selectionNote: string;
  lagSeconds: number;
  lagFrames: number;
  gainGated: boolean;
  nGainCrossed: number;
  breathNote: string | null;
  fsHz: number;
  framesUsed: number;
  framesDropped: number;
  tMin: number;
  tMax: number;
  truth: { timeS: number[]; present: boolean[] } | null;
  truthExcluded: string | null;
  confusion: HybridConfusion | null;
  /** One present/empty call for the whole range from fixed thresholds: P90 of
   *  the 2 s-lag step, else a run of FarSense peaks. See docs/hybrid2.md. */
  rangeVerdict: {
    present: boolean;
    by: "motion" | "breathing" | null;
    motionP90: number | null;
    breathRun: number;
    seconds: number;
    thresholds: { lagSeconds: number; motionP90: number; breathPeak: number; breathRun: number };
  };
  /** The same rule applied every second to the trailing `windowSeconds`:
   *  what a live system would show. */
  rangeSeries: { present: boolean[]; state: Hybrid2RangeState[]; windowSeconds: number };
  rangeConfusion: HybridConfusion | null;
}

export type Hybrid2RangeState = "present:motion" | "present:breathing" | "empty" | "unknown";

export interface Hybrid2Options {
  lagS?: number;
  gateGain?: boolean;
  holdS?: number;
  burstS?: number;
  motionRel?: number;
  motionAbs?: number;
  floorPct?: number;
  motionFloor?: number | null;
  breathMinPeak?: number;
  breathPersistS?: number;
  breathWindow?: number;
  rpmLo?: number;
  rpmHi?: number;
  leadHold?: boolean;
  marginS?: number;
  mimo?: string | null;
  sourceMac?: string | null;
  interpolate?: boolean;
  /** Trailing window, seconds, for the per-second range-rule series. */
  rangeWindow?: number;
}

export async function fetchHybrid2(
  path: string,
  t0: number,
  t1: number,
  options: Hybrid2Options = {},
  signal?: AbortSignal,
): Promise<Hybrid2> {
  const {
    lagS = 0, gateGain = false, holdS = 20, burstS = 2,
    motionRel = 2, motionAbs = 0.05, floorPct = 20, motionFloor,
    breathMinPeak = 0.2, breathPersistS = 10, breathWindow = 10,
    rpmLo = 10, rpmHi = 30, leadHold = true, marginS = 5, rangeWindow = 60,
    mimo, sourceMac, interpolate,
  } = options;

  const url =
    `/api/hybrid2?path=${encodeURIComponent(path)}` +
    `&range_window=${rangeWindow}` +
    `&t0=${t0}&t1=${t1}&lag_s=${lagS}&hold_s=${holdS}&burst_s=${burstS}` +
    `&motion_rel=${motionRel}&motion_abs=${motionAbs}&floor_pct=${floorPct}` +
    `&breath_min_peak=${breathMinPeak}&breath_persist_s=${breathPersistS}` +
    `&breath_window=${breathWindow}&rpm_lo=${rpmLo}&rpm_hi=${rpmHi}` +
    `&margin_s=${marginS}` +
    (gateGain ? "&gate_gain=true" : "") +
    (leadHold ? "" : "&lead_hold=false") +
    (motionFloor != null ? `&motion_floor=${motionFloor}` : "") +
    filterParams(mimo, sourceMac) +
    (interpolate === false ? "&interpolate=false" : "");

  const res = await fetch(url, { signal });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `hybrid2: ${res.status}`);
  }
  const body = await res.json();
  const rc = body.range_confusion;
  const c = body.confusion;
  return {
    timeS: body.time_s,
    present: body.present,
    state: body.state,
    unknown: body.unknown,
    motion: body.motion,
    motionReference: body.motion_reference,
    burst: body.burst,
    breathing: body.breathing,
    breathPeak: body.breath_peak,
    breathRpm: body.breath_rpm,
    floor: body.floor,
    threshold: body.threshold,
    floorScope: body.floor_scope,
    signal: body.signal,
    selectionNote: body.selection_note,
    rangeVerdict: {
      present: Boolean(body.range_verdict?.present),
      by: body.range_verdict?.by ?? null,
      motionP90: body.range_verdict?.motion_p90 ?? null,
      breathRun: body.range_verdict?.breath_run ?? 0,
      seconds: body.range_verdict?.seconds ?? 0,
      thresholds: {
        lagSeconds: body.range_verdict?.thresholds?.lag_seconds ?? 2,
        motionP90: body.range_verdict?.thresholds?.motion_p90 ?? 0.035,
        breathPeak: body.range_verdict?.thresholds?.breath_peak ?? 0.25,
        breathRun: body.range_verdict?.thresholds?.breath_run ?? 5,
      },
    },
    lagSeconds: body.lag_seconds,
    lagFrames: body.lag_frames,
    gainGated: body.gain_gated,
    nGainCrossed: body.n_gain_crossed,
    breathNote: body.breath_note,
    fsHz: body.fs_hz,
    framesUsed: body.frames_used,
    framesDropped: body.frames_dropped,
    tMin: body.t_min,
    tMax: body.t_max,
    truth: body.truth ? { timeS: body.truth.time_s, present: body.truth.present } : null,
    truthExcluded: body.truth_excluded,
    confusion: c
      ? {
          tp: c.tp, fp: c.fp, fn: c.fn, tn: c.tn, total: c.total,
          excluded: c.excluded, accuracy: c.accuracy, recall: c.recall,
          specificity: c.specificity, precision: c.precision,
          baseRate: c.base_rate, marginSeconds: c.margin_s,
        }
      : null,
    rangeConfusion: rc
      ? {
          tp: rc.tp, fp: rc.fp, fn: rc.fn, tn: rc.tn, total: rc.total,
          excluded: rc.excluded, accuracy: rc.accuracy, recall: rc.recall,
          specificity: rc.specificity, precision: rc.precision,
          baseRate: rc.base_rate, marginSeconds: rc.margin_s,
        }
      : null,
    rangeSeries: {
      present: body.range_series?.present ?? [],
      state: body.range_series?.state ?? [],
      windowSeconds: body.range_series?.window_seconds ?? 60,
    },
  };
}

// --------------------------------------------------------------------------- #
//  Classifier (/api/classifier): the one-minute verdict's feature bank
// --------------------------------------------------------------------------- #

/** One entry of the bank, as the backend describes it. The tab renders these,
 *  so a feature added by a later test needs no new column code here. */
export interface ClassifierFeature {
  key: string;
  label: string;
  /** Which test of the programme produced it ("range rule", "1", "context", …). */
  test: string;
  /** "in rule" for the range rule's inputs, "candidate" for what the tests
   *  have produced since, "context" for link-state readings beside them. */
  status: string;
  /** A measured operating point to draw as a guide, or null. Not a verdict. */
  reference: number | null;
  axis: [number, number] | null;
  decimals: number;
  description: string;
  /** What the column decides or feeds: "decides PRESENT", "input of P(occupied)", "context", … */
  role: string;
  /** The one-line answers the test gave: what this column separates, and what it does not. */
  separates: string;
  notSeparates: string;
}

/** One step of the verdict, with the evidence behind it. */
export interface ClassifierDecision {
  step: string;
  test: string;
  key: string;
  text: string;
  evidence: string;
}

/** One row of the compounding table: what each test separated. */
export interface ClassifierStep {
  step: string;
  feature: string;
  separates: string;
  notSeparates: string;
}

export interface ClassifierUnit {
  t0: number;
  t1: number;
  nWindows: number;
  nSeconds: number;
  nFrames: number;
  /** Fraction of the unit's camera frames with a person; null without a camera. */
  cameraOccupancy: number | null;
  cameraFrames: number;
  /** One value per `ClassifierFeature.key`. */
  values: Record<string, number | null>;
}

export interface Classifier {
  /** ψ̂ per autocorrelation window, on the ratio amplitude and phase. */
  acf: {
    timeS: number[];
    amp: (number | null)[];
    phase: (number | null)[];
    windowFrames: number;
    windowSeconds: number | null;
    /** −1/T, where white noise sits. */
    nullMean: number;
  };
  /** The lag step the range rule reads, per second. */
  step: { timeS: number[]; level: (number | null)[]; lagSeconds: number; lagFrames: number };
  units: ClassifierUnit[];
  unitSeconds: number;
  fsHz: number | null;
  nSubcarriers: number;
  framesUsed: number;
  framesDropped: number;
  selectionNote: string;
  features: ClassifierFeature[];
  decision: ClassifierDecision[];
  steps: ClassifierStep[];
  truth: { timeS: number[]; present: boolean[] } | null;
  truthExcluded: string | null;
}

export interface ClassifierOptions {
  unitS?: number;
  acfFrames?: number;
  lagS?: number;
  mimo?: string | null;
  sourceMac?: string | null;
  interpolate?: boolean;
}

export async function fetchClassifier(
  path: string,
  t0: number,
  t1: number,
  options: ClassifierOptions = {},
  signal?: AbortSignal,
): Promise<Classifier> {
  const { unitS = 60, acfFrames = 84, lagS = 2, mimo, sourceMac, interpolate } = options;
  const url =
    `/api/classifier?path=${encodeURIComponent(path)}` +
    `&t0=${t0}&t1=${t1}&unit_s=${unitS}&acf_frames=${acfFrames}&lag_s=${lagS}` +
    filterParams(mimo, sourceMac) +
    (interpolate === false ? "&interpolate=false" : "");

  const res = await fetch(url, { signal });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `classifier: ${res.status}`);
  }
  const body = await res.json();
  const features: ClassifierFeature[] = (body.features ?? []).map((f: Record<string, unknown>) => ({
    key: f.key as string,
    label: f.label as string,
    test: String(f.test),
    status: f.status as string,
    reference: (f.reference as number | null) ?? null,
    axis: (f.axis as [number, number] | null) ?? null,
    decimals: (f.decimals as number) ?? 3,
    description: (f.description as string) ?? "",
    role: (f.role as string) ?? "",
    separates: (f.separates as string) ?? "",
    notSeparates: (f.not_separates as string) ?? "",
  }));
  const decision: ClassifierDecision[] = (body.decision ?? []).map((d: Record<string, unknown>) => ({
    step: String(d.step), test: String(d.test), key: String(d.key), text: String(d.text ?? ""), evidence: String(d.evidence ?? ""),
  }));
  const steps: ClassifierStep[] = (body.steps ?? []).map((d: Record<string, unknown>) => ({
    step: String(d.step), feature: String(d.feature ?? ""), separates: String(d.separates ?? ""), notSeparates: String(d.not_separates ?? ""),
  }));
  const units: ClassifierUnit[] = (body.units ?? []).map((u: Record<string, unknown>) => ({
    t0: u.t0 as number,
    t1: u.t1 as number,
    nWindows: (u.n_windows as number) ?? 0,
    nSeconds: (u.n_seconds as number) ?? 0,
    nFrames: (u.n_frames as number) ?? 0,
    cameraOccupancy: (u.camera_occupancy as number | null) ?? null,
    cameraFrames: (u.camera_frames as number) ?? 0,
    values: Object.fromEntries(features.map((f) => [f.key, (u[f.key] as number | null) ?? null])),
  }));
  return {
    acf: {
      timeS: body.acf.time_s,
      amp: body.acf.amp,
      phase: body.acf.phase,
      windowFrames: body.acf.window_frames,
      windowSeconds: body.acf.window_seconds ?? null,
      nullMean: body.acf.null_mean,
    },
    step: {
      timeS: body.step.time_s,
      level: body.step.level,
      lagSeconds: body.step.lag_seconds,
      lagFrames: body.step.lag_frames,
    },
    units,
    unitSeconds: body.unit_seconds,
    fsHz: body.fs_hz ?? null,
    nSubcarriers: body.n_subcarriers,
    framesUsed: body.frames_used,
    framesDropped: body.frames_dropped,
    selectionNote: body.selection_note,
    features,
    decision,
    steps,
    truth: body.truth ? { timeS: body.truth.time_s, present: body.truth.present } : null,
    truthExcluded: body.truth_excluded ?? null,
  };
}

// --------------------------------------------------------------------------- //
//  AF8 classifier (/api/af8) -- the ML tab                                    //
// --------------------------------------------------------------------------- //

/** The eight features, in the order the model's weights expect. */
export type Af8Feature =
  | "A_slope3" | "A_r025_2" | "A_r5_2" | "A_p90"
  | "F_pkmax" | "F_pkmed" | "F_run" | "F_rpm_sd" | "C3_revisit";

/** The model's arithmetic for one window, one entry per feature. */
export interface Af8Model {
  features: Af8Feature[];
  value: number[];            // after imputation
  imputed: boolean[];         // true where the feature was missing
  mean: number[];
  scale: number[];
  coef: number[];
  z: number[];
  contribution: number[];     // coef * z
  intercept: number;
  logit: number;
}

export interface Af8 {
  file: string;
  window: number;
  windows: [number, number][];
  windowS: [number, number];
  fsHz: number;
  subcarriers: number;
  seconds: number;
  pPerson: number;
  label: 0 | 1;
  threshold: number;
  features: Record<Af8Feature, number | null>;
  model: Af8Model;
  motion: {
    lags: number[];           // every gap D was measured at (s)
    D: number[];              // median step at each gap
    shapeLags: number[];      // the three the model reads
    fitIntercept: number;     // log D = slope * log tau + fitIntercept
    cells: number[];          // 1 s cell centres for the step lines
    step: Record<string, (number | null)[]>;   // per-second median step, by gap
    p90Time: number[];
    p90Values: (number | null)[];
  };
  revisit: {
    lags: number[];           // 10..30 s, 1 s apart
    D: number[];              // median step at each lag, grid thinned to ~10 Hz
    runningMax: number[];     // max of D up to each lag
    drop: number[];           // (runningMax - D) / runningMax; C3 is the largest
    at: number | null;        // the lag of the largest drop
  };
  breath: {
    time: number[];           // FarSense window centres
    peak: (number | null)[];
    rpm: (number | null)[];
    good: boolean[];          // peak >= threshold
    threshold: number;
    run: { windows: number; t0: number; t1: number } | null;
    rpmMean: number | null;
  };
  camera: { time: number[]; present: boolean[]; fraction: number | null } | null;
}

export async function fetchAf8(path: string, window: number, signal?: AbortSignal): Promise<Af8> {
  const res = await fetch(`/api/af8?path=${encodeURIComponent(path)}&window=${window}`, { signal });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `af8: ${res.status}`);
  }
  const b = await res.json();
  const nums = (xs: (number | null)[]) => xs.map((v) => (v == null ? NaN : v));
  return {
    file: b.file,
    window: b.window,
    windows: b.windows,
    windowS: b.window_s,
    fsHz: b.fs_hz,
    subcarriers: b.subcarriers,
    seconds: b.seconds,
    pPerson: b.p_person,
    label: b.label,
    threshold: b.threshold,
    features: b.features,
    model: {
      features: b.model.features,
      value: nums(b.model.value),
      imputed: b.model.imputed,
      mean: nums(b.model.mean),
      scale: nums(b.model.scale),
      coef: nums(b.model.coef),
      z: nums(b.model.z),
      contribution: nums(b.model.contribution),
      intercept: b.model.intercept,
      logit: b.model.logit,
    },
    motion: {
      lags: b.motion.lags,
      D: b.motion.D,
      shapeLags: b.motion.shape_lags,
      fitIntercept: b.motion.fit_intercept,
      cells: b.motion.cells,
      step: b.motion.step,
      p90Time: nums(b.motion.p90_time),
      p90Values: b.motion.p90_values,
    },
    revisit: {
      lags: b.revisit.lags,
      D: nums(b.revisit.D),
      runningMax: nums(b.revisit.running_max),
      drop: nums(b.revisit.drop),
      at: b.revisit.at,
    },
    breath: {
      time: nums(b.breath.time),
      peak: b.breath.peak,
      rpm: b.breath.rpm,
      good: b.breath.good,
      threshold: b.breath.threshold,
      run: b.breath.run,
      rpmMean: b.breath.rpm_mean,
    },
    camera: b.camera,
  };
}
