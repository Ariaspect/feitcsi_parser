import { useEffect, useMemo, useRef, useState } from "react";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  fetchFarSense,
  fetchLabels,
  type FarSense as FarSenseData,
  type Labels,
  type Meta,
} from "./api";
import { Chart, CHART_MARGIN } from "./Chart";
import { VIRIDIS, buildLut, lutIndex } from "./colormap";
import { formatTime, linearScale, runs } from "./series";
import type { TimeLink } from "./timelink";

// Same cap as the presence tab: the whole range in view is decoded at full
// rate, and ten minutes holds hundreds of 12 s windows.
const DEFAULT_SPAN_SECONDS = 600;
const STRIP = 14;

export interface FarSenseProps {
  path: string;
  meta: Meta;
  timeLink?: TimeLink;
  mimo: string;
  sourceMac: string;
  interpolate: boolean;
  dark: boolean;
}

function NumberField({
  id,
  label,
  value,
  onChange,
  min,
  max,
  step,
  width = "w-20",
  title,
}: {
  id: string;
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  width?: string;
  title?: string;
}) {
  return (
    <div className="flex items-center gap-2" title={title}>
      <Label htmlFor={id} className="text-[10px] text-muted-foreground uppercase tracking-wide">
        {label}
      </Label>
      <Input
        id={id}
        type="number"
        min={min}
        max={max}
        step={step}
        className={width}
        value={value}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v) && v >= min && v <= max) onChange(v);
        }}
      />
    </div>
  );
}

/** The I/Q trajectory of one subcarrier's ratio over one window -- the arc of
 *  the paper's Fig. 11 -- with the projection axis the sweep chose. */
function ArcPanel({
  iq,
  theta,
  size,
  dark,
}: {
  iq: [number, number][];
  theta: number;
  size: number;
  dark: boolean;
}) {
  const extent = Math.max(1e-9, ...iq.map(([a, b]) => Math.max(Math.abs(a), Math.abs(b))));
  const pad = 14;
  const half = size / 2 - pad;
  const sx = (v: number) => size / 2 + (v / extent) * half;
  const sy = (v: number) => size / 2 - (v / extent) * half;
  const grid = dark ? "#2a2f37" : "#e6e9ee";
  const text = dark ? "#8b95a3" : "#6b7480";
  const path = iq
    .map(([a, b], i) => `${i === 0 ? "M" : "L"}${sx(a).toFixed(1)} ${sy(b).toFixed(1)}`)
    .join("");
  const ax = Math.cos(theta) * half;
  const ay = Math.sin(theta) * half;
  return (
    <svg width={size} height={size} role="img" aria-label="ratio in the complex plane">
      <line x1={pad} x2={size - pad} y1={size / 2} y2={size / 2} stroke={grid} />
      <line x1={size / 2} x2={size / 2} y1={pad} y2={size - pad} stroke={grid} />
      <line
        x1={size / 2 - ax}
        x2={size / 2 + ax}
        y1={size / 2 + ay}
        y2={size / 2 - ay}
        stroke="#c7a02f"
        strokeWidth={1}
        strokeDasharray="4 3"
      />
      <path d={path} fill="none" stroke="#2f6fed" strokeWidth={1} strokeOpacity={0.55} />
      {iq.map(([a, b], i) => (
        <circle key={i} cx={sx(a)} cy={sy(b)} r={1.4} fill="#2f6fed" fillOpacity={0.5} />
      ))}
      {iq.length > 0 && (
        <circle cx={sx(iq[0][0])} cy={sy(iq[0][1])} r={3} fill="#d97a2f" />
      )}
      <text x={size - pad} y={size / 2 - 4} textAnchor="end" fontSize={9} fill={text}>
        I
      </text>
      <text x={size / 2 + 4} y={pad + 8} fontSize={9} fill={text}>
        Q
      </text>
      <text x={pad} y={size - 4} fontSize={9} fill={text}>
        ±{extent.toExponential(1)} · axis θ={((theta * 180) / Math.PI).toFixed(0)}°
      </text>
    </svg>
  );
}

/** BNR per subcarrier per window, on the shared time axis. Rows are the
 *  live subcarriers in capture-bin order; brighter is more of the window's
 *  energy in one in-band tone. */
function BnrStrip({
  data,
  domain,
  width,
  height,
  dark,
}: {
  data: FarSenseData;
  domain: [number, number];
  width: number;
  height: number;
  dark: boolean;
}) {
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const inner = Math.max(1, width - CHART_MARGIN.left - CHART_MARGIN.right);
  const x = linearScale(domain, [0, inner]);
  const text = dark ? "#8b95a3" : "#6b7480";

  useEffect(() => {
    const node = canvas.current;
    if (!node) return;
    const ctx = node.getContext("2d");
    if (!ctx) return;
    const rows = data.bnrMap.length;
    const cols = data.timeS.length;
    ctx.clearRect(0, 0, node.width, node.height);
    if (!rows || !cols) return;
    const image = new ImageData(cols, rows);
    const u32 = new Uint32Array(image.data.buffer);
    const lut = buildLut(VIRIDIS);
    const scale = data.bnrNormFactor;
    for (let r = 0; r < rows; r++) {
      const row = data.bnrMap[r];
      for (let c = 0; c < cols; c++) {
        const v = row[c];
        const li = v === null ? -1 : lutIndex(v * scale, 0, 0.5);
        u32[r * cols + c] = li < 0 ? 0 : lut[li];
      }
    }
    const off = document.createElement("canvas");
    off.width = cols;
    off.height = rows;
    off.getContext("2d")?.putImageData(image, 0, 0);
    const hop = data.hop / data.fsHz;
    const x0 = CHART_MARGIN.left + x(data.timeS[0] - hop / 2);
    const x1 = CHART_MARGIN.left + x(data.timeS[cols - 1] + hop / 2);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, Math.round(x0), 0, Math.max(1, Math.round(x1 - x0)), node.height);
  }, [data, domain, width, height, x]);

  return (
    <div className="relative">
      <canvas ref={canvas} width={width} height={height} style={{ display: "block" }} />
      <div
        className="absolute left-0 top-0 text-[9px]"
        style={{ color: text, writingMode: "vertical-rl", transform: "rotate(180deg)", height }}
      >
        subcarrier
      </div>
    </div>
  );
}

export function FarSense({
  path,
  meta,
  timeLink,
  mimo,
  sourceMac,
  interpolate,
  dark,
}: FarSenseProps) {
  const [windowSeconds, setWindowSeconds] = useState(10);
  const [rpmLo, setRpmLo] = useState(10);
  const [rpmHi, setRpmHi] = useState(30);
  const [keepFraction, setKeepFraction] = useState(0.6);
  const [nTheta, setNTheta] = useState(200);
  const [savgolSeconds, setSavgolSeconds] = useState(1.0);
  const [highpassHz, setHighpassHz] = useState(0);
  const [motionFracHi, setMotionFracHi] = useState(0.25);
  const [minPeak, setMinPeak] = useState(0.2);
  const [detailT, setDetailT] = useState<number | null>(null);
  const [data, setData] = useState<FarSenseData | null>(null);
  const [labels, setLabels] = useState<Labels | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);

  const [range, setRange] = useState<[number, number]>(() => [
    Math.max(meta.t_min, meta.t_max - DEFAULT_SPAN_SECONDS),
    meta.t_max,
  ]);

  const holder = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(880);
  useEffect(() => {
    const node = holder.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.max(320, Math.floor(entry.contentRect.width)));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!timeLink) return;
    return timeLink.subscribe((w) => {
      setRange(([t0, t1]) =>
        w.tMin === t0 && w.tMax === t1 ? [t0, t1] : [w.tMin, w.tMax],
      );
    });
  }, [timeLink]);

  useEffect(() => {
    const controller = new AbortController();
    setLabels(null);
    fetchLabels(path, controller.signal)
      .then(setLabels)
      .catch(() => setLabels(null));
    return () => controller.abort();
  }, [path]);

  const options = useMemo(
    () => ({
      windowSeconds,
      rpmLo,
      rpmHi,
      keepFraction,
      nTheta,
      savgolSeconds,
      highpassHz,
      motionFracHi,
      minPeak,
      mimo,
      sourceMac,
      interpolate,
    }),
    [
      windowSeconds, rpmLo, rpmHi, keepFraction, nTheta, savgolSeconds, highpassHz,
      motionFracHi, minPeak, mimo, sourceMac, interpolate,
    ],
  );

  // The main request carries whatever window was picked last, so a parameter
  // change keeps the detail panels on the same window. A pick on its own goes
  // through the second effect and replaces only `detail`.
  const detailRef = useRef<number | null>(null);
  detailRef.current = detailT;
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetchFarSense(path, range[0], range[1], { ...options, detailT: detailRef.current }, controller.signal)
      .then((result) => {
        setData(result);
        setError(null);
        if (result.detail === null && result.timeS.length) {
          // Open on the last stationary window, which is what the paper's GUI
          // would be showing at the end of this range.
          let pick = result.timeS.length - 1;
          for (let i = result.timeS.length - 1; i >= 0; i--) {
            if (result.stationary[i] && !result.unknown[i]) { pick = i; break; }
          }
          setDetailT(result.timeS[pick]);
        }
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setData(null);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [path, range, options]);

  const detailIndex = useMemo(() => {
    if (!data || detailT === null || !data.timeS.length) return null;
    let best = 0;
    for (let i = 1; i < data.timeS.length; i++) {
      if (Math.abs(data.timeS[i] - detailT) < Math.abs(data.timeS[best] - detailT)) best = i;
    }
    return best;
  }, [data, detailT]);

  useEffect(() => {
    if (!data || detailT === null || detailIndex === null) return;
    if (data.detail && data.detail.index === detailIndex) return;
    const controller = new AbortController();
    setDetailLoading(true);
    fetchFarSense(path, range[0], range[1], { ...options, detailT }, controller.signal)
      .then((result) => {
        setData((prev) => (prev ? { ...prev, detail: result.detail } : result));
      })
      .catch(() => undefined)
      .finally(() => {
        if (!controller.signal.aborted) setDetailLoading(false);
      });
    return () => controller.abort();
    // `data` is read, not depended on: a new main result resets the detail
    // through its own request, and listing it here would double every fetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detailT, detailIndex, path, range, options]);

  const domain = useMemo<[number, number]>(() => {
    if (!data || data.timeS.length === 0) return range;
    return [data.timeS[0], data.timeS[data.timeS.length - 1]];
  }, [data, range]);

  const innerWidth = Math.max(1, width - CHART_MARGIN.left - CHART_MARGIN.right);
  const stripX = linearScale(domain, [0, innerWidth]);
  const clampX = (t: number) => Math.max(0, Math.min(innerWidth, stripX(t)));

  const truthRuns = useMemo(
    () => (labels?.present ? runs(labels.present.timeS, labels.present.present) : []),
    [labels],
  );
  const stationaryRuns = useMemo(
    () =>
      data
        ? runs(
            data.timeS,
            data.timeS.map((_, i) =>
              data.unknown[i] ? "unknown" : data.stationary[i] ? "still" : "moving",
            ),
          )
        : [],
    [data],
  );

  // Rate dots coloured by the normalised peak behind them: a rate read off a
  // peak of 0.02 and one read off 0.4 must not look alike.
  const peakColors = useMemo(() => {
    if (!data) return [];
    const lut = buildLut(VIRIDIS);
    return data.acfPeakNorm.map((v) => {
      const li = v === null ? -1 : lutIndex(v, 0, 0.5);
      if (li < 0) return "transparent";
      const packed = lut[li];
      const r = packed & 0xff;
      const g = (packed >> 8) & 0xff;
      const b = (packed >> 16) & 0xff;
      return `rgb(${r},${g},${b})`;
    });
  }, [data]);

  const tally = useMemo(() => {
    if (!data) return null;
    const still = data.stationary.filter((s, i) => s && !data.unknown[i]).length;
    const rated = data.rpm.filter((v) => v !== null).length;
    const rates = data.rpm.filter((v): v is number => v !== null).sort((a, b) => a - b);
    const median = rates.length ? rates[Math.floor(rates.length / 2)] : null;
    return { still, rated, median };
  }, [data]);

  const cam = dark ? "#4f7fd4" : "#3f6fc4";
  const vacant = dark ? "#2a2f37" : "#e6e9ee";
  const stillColor = "#2f9c6f";
  const movingColor = "#d97a2f";
  const detail = data?.detail ?? null;
  const detailWindowT = detail ? detail.tS.map((t) => t - detail.tS[0]) : [];
  const acfNorm = detail
    ? detail.acf.map((v) => (v === null || detail.acf[0] === null || detail.acf[0] === 0 ? null : v / (detail.acf[0] as number)))
    : [];
  const acfLagS = detail ? detail.acf.map((_, k) => k / (data?.fsHz ?? 1)) : [];
  const detailPattern = detail
    ? (() => {
        const vals = detail.pattern.filter((v): v is number => v !== null);
        const mean = vals.reduce((a, b) => a + b, 0) / Math.max(1, vals.length);
        const sd = Math.sqrt(
          vals.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, vals.length),
        ) || 1;
        return detail.pattern.map((v) => (v === null ? null : (v - mean) / sd));
      })()
    : [];

  return (
    <div className="space-y-4" ref={holder}>
      <div className="flex flex-wrap items-center gap-3">
        <NumberField id="fs-win" label="Window (s)" value={windowSeconds} onChange={setWindowSeconds} min={4} max={120}
          title="Longer: a steady rhythm stands out more against noise and the rate is finer, but it assumes the rate holds for the whole window and a seated person shows only about half a window after sitting down. Shorter: reacts faster and tolerates a wandering rate, but peaks are noisier and the slow edge of the band fits only a breath or two." />
        <NumberField id="fs-lo" label="rpm" value={rpmLo} onChange={setRpmLo} min={1} max={119} width="w-16"
          title="Narrower band: fewer chances for noise to pass as a peak and the empty room stops piling up at the band edge, but a real rate outside it is missed. Wider: catches unusual rates at the cost of more false peaks; the slow edge must still fit in the window or the tab refuses." />
        <span className="text-[11px] text-muted-foreground">–</span>
        <NumberField id="fs-hi" label="" value={rpmHi} onChange={setRpmHi} min={2} max={120} width="w-16" />
        <NumberField id="fs-keep" label="Keep ≥ ×best" value={keepFraction} onChange={setKeepFraction} min={0} max={1} step={0.05}
          title="Higher (towards 1): only the very best subcarriers vote — a sharper peak when a chest is there, jumpier from window to window. Lower (towards 0): every subcarrier votes and the ones that barely see the chest dilute the peak (measured: 0.215 → 0.119 at 0)." />
        <NumberField id="fs-theta" label="θ steps" value={nTheta} onChange={setNTheta} min={2} max={720} step={2}
          title="More: a finer search for the direction the breath moves the ratio along; above ~50 nothing changes but the compute. Fewer: below ~10 the direction can be missed and the peak falls." />
        <NumberField id="fs-sg" label="S-G (s)" value={savgolSeconds} onChange={setSavgolSeconds} min={0} max={5} step={0.1}
          title="Longer: a smoother pattern with less per-packet jitter, but fast breaths start to flatten (1–1.5 s begins to eat 30 rpm). Shorter or 0: the raw packets — noisier pattern, nothing removed." />
        <NumberField id="fs-hp" label="High-pass (Hz)" value={highpassHz} onChange={setHighpassHz} min={0} max={2} step={0.05}
          title="Higher: removes slower wander (drift, a person settling) so it cannot pass as a slow breath; above ~0.15 it starts eating 10 rpm breathing itself. 0: nothing removed — drift leaks into the slow edge of the band." />
        <NumberField id="fs-motion" label="Motion above" value={motionFracHi} onChange={setMotionFracHi} min={0.01} max={2} step={0.01}
          title="Lower: stricter — fidgeting counts as motion, so fewer windows show a rate but those that do are cleaner. Higher: looser — walking can pass as stationary and produce junk rates." />
        <NumberField id="fs-peak" label="Min peak" value={minPeak} onChange={setMinPeak} min={-1} max={1} step={0.05}
          title="Higher: only rates with a clear rhythm behind them are drawn (an empty room sits near 0 ± 0.1, an occupant 0.2–0.6), so fewer dots and fewer false ones. 0: a number whenever the window is stationary, empty room included." />
        {data && (
          <span className="text-[11px] text-muted-foreground">
            {data.fsHz.toFixed(1)} Hz · {data.timeS.length} windows · {data.windowSeconds.toFixed(1)} s each ·
            lags {data.lagLo}–{data.lagHi} · {data.scIndex.length} subcarriers
            {loading && " · refreshing"}
          </span>
        )}
      </div>

      {error ? (
        <div className="text-muted-foreground p-8 text-sm">No result for this range — {error}</div>
      ) : !data ? (
        <div className="text-muted-foreground p-8 text-sm">
          {loading ? "Running the sweep over this range…" : "Nothing to analyse yet."}
        </div>
      ) : data.timeS.length === 0 ? (
        <div className="text-muted-foreground p-8 text-sm">
          This range holds no complete window — widen it, or shorten the window length.
        </div>
      ) : (
        <>
          {/* 1. camera truth, 2. stationary / non-stationary — the paper's
              "human status", which is the only gate it has. */}
          <div>
            <svg width={width} height={STRIP * 2 + 8} role="img" aria-label="camera and stationary strips">
              <g transform={`translate(${CHART_MARGIN.left},2)`}>
                <rect x={0} width={innerWidth} y={0} height={STRIP} fill={vacant} />
                {truthRuns.map((r, i) => {
                  const a = clampX(r.t0);
                  const b = clampX(r.t1);
                  return r.value && b > a ? (
                    <rect key={`t${i}`} x={a} width={Math.max(1, b - a)} y={0} height={STRIP} fill={cam} />
                  ) : null;
                })}
                <text x={-6} y={STRIP - 3} textAnchor="end" fontSize={9} fill="currentColor">
                  camera
                </text>
                <rect x={0} width={innerWidth} y={STRIP + 4} height={STRIP} fill={vacant} />
                {stationaryRuns.map((r, i) => {
                  const a = clampX(r.t0);
                  const b = clampX(r.t1);
                  if (b <= a || r.value === "unknown") return null;
                  return (
                    <rect key={`s${i}`} x={a} width={Math.max(1, b - a)} y={STRIP + 4} height={STRIP}
                      fill={r.value === "still" ? stillColor : movingColor}>
                      <title>
                        {r.value} · {formatTime(r.t0, domain[1] - domain[0])} – {formatTime(r.t1, domain[1] - domain[0])}
                      </title>
                    </rect>
                  );
                })}
                <text x={-6} y={STRIP * 2 + 1} textAnchor="end" fontSize={9} fill="currentColor">
                  status
                </text>
              </g>
            </svg>
            <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
              <span><span style={{ color: cam }}>■</span> person in frame</span>
              <span><span style={{ color: stillColor }}>■</span> stationary · {tally?.still}</span>
              <span><span style={{ color: movingColor }}>■</span> non-stationary (no rate shown)</span>
              {!labels?.present && <span>no camera sidecar beside this capture</span>}
            </div>
          </div>

          <Chart
            width={width}
            times={data.patternT}
            domain={domain}
            yDomain={[-3.5, 3.5]}
            yLabel="respiration pattern (σ)"
            dark={dark}
            height={130}
            series={[{ values: data.pattern, color: "#2f6fed", width: 1, label: "pattern" }]}
            marker={detailT}
            onPick={setDetailT}
          />
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            The best-BNR subcarrier&apos;s projection, window by window, stitched and scaled to
            unit deviation — the trace the paper&apos;s GUI scrolls. Click anywhere to open that
            window below.
          </p>

          <Chart
            width={width}
            times={data.timeS}
            domain={domain}
            yDomain={[rpmLo, rpmHi]}
            yLabel="rate (rpm)"
            dark={dark}
            height={130}
            series={[{ values: data.rpm, color: dark ? "#3a4048" : "#c9cfd8", width: 1, label: "rate" }]}
            dots={{ values: data.rpm, colors: peakColors, radius: 2.2, label: "rate" }}
            marker={detailT}
            onPick={setDetailT}
          />
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            {tally && (
              <>
                A rate in {tally.rated} of {tally.still} stationary windows
                {tally.median !== null && <>, median {tally.median.toFixed(1)} rpm</>}.{" "}
              </>
            )}
            Dots are coloured by the normalised peak the rate was read from (dark = near 0,
            yellow = 0.5 or more). The paper shows a number whenever the target is stationary;
            raise <b>Min peak</b> to hide the ones with nothing behind them.
          </p>

          <Chart
            width={width}
            times={data.timeS}
            domain={domain}
            yDomain={[-0.3, 1]}
            yLabel="evidence"
            dark={dark}
            height={120}
            series={[
              { values: data.acfPeakNorm, color: "#7b5ea7", width: 1.6, label: "peak (weighted mean ACF at lag)" },
              { values: data.bnrMax.map((v) => (v === null ? null : v * data.bnrNormFactor)), color: "#c7a02f", label: "best BNR ×N/8192" },
            ]}
            guides={minPeak > 0 ? [{ value: minPeak, color: "#d62728", label: "min peak" }] : []}
            marker={detailT}
            onPick={setDetailT}
          />
          <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
            <span style={{ color: "#7b5ea7" }}>peak: the combined autocorrelation at the lag read, normalised to −1..1</span>
            <span style={{ color: "#c7a02f" }}>best BNR: fraction of a subcarrier&apos;s energy in its strongest in-band tone (1 = pure tone)</span>
          </div>

          <BnrStrip data={data} domain={domain} width={width} height={120} dark={dark} />
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            BNR of every subcarrier (rows, capture order) in every window (columns), 0 dark to
            0.5 yellow on the pure-tone scale. A chest lights a band of neighbouring rows
            together; noise lights single cells.
          </p>

          {detail && (
            <div className="space-y-2">
              <div className="text-[11px] text-muted-foreground">
                Window at {formatTime(detail.tS[0] + data.windowSeconds / 2, domain[1] - domain[0])} · best subcarrier{" "}
                <b>{detail.bestSc}</b> · {detail.selected.filter(Boolean).length} of {detail.selected.length} subcarriers combined
                {detail.lag !== null ? (
                  <> · first peak at lag {detail.lag} = {(detail.lag / data.fsHz).toFixed(2)} s → {(60 * data.fsHz / detail.lag).toFixed(1)} rpm (integer lag)</>
                ) : (
                  <> · no in-band peak</>
                )}
                {detailLoading && " · loading"}
              </div>
              <div className="flex flex-wrap gap-4 items-start">
                <div>
                  <ArcPanel iq={detail.iq} theta={detail.bestTheta} size={220} dark={dark} />
                  <div className="text-[10px] text-muted-foreground w-[220px] leading-relaxed">
                    Its ratio in the complex plane over the window, mean removed; orange dot is
                    the first sample, dashed line the projection axis. A chest traces an arc
                    (Fig. 11); noise fills a blob.
                  </div>
                </div>
                <div className="flex-1 min-w-[280px] space-y-2">
                  <Chart
                    width={Math.max(280, width - 220 - 16 - 32)}
                    times={detailWindowT}
                    domain={[0, detailWindowT.length ? detailWindowT[detailWindowT.length - 1] : 1]}
                    yDomain={[-3.5, 3.5]}
                    yLabel="pattern in window (σ)"
                    xMode="offset"
                    dark={dark}
                    height={110}
                    series={[{ values: detailPattern, color: "#2f6fed", width: 1.2, label: "pattern" }]}
                  />
                  <Chart
                    width={Math.max(280, width - 220 - 16 - 32)}
                    times={acfLagS}
                    domain={[0, acfLagS.length ? acfLagS[Math.min(acfLagS.length - 1, detail.lagHi + Math.round(data.fsHz))] : 1]}
                    yDomain={[-1, 1]}
                    yLabel="combined autocorrelation vs lag (s)"
                    xMode="offset"
                    dark={dark}
                    height={110}
                    series={[{ values: acfNorm, color: "#7b5ea7", width: 1.4, label: "acf" }]}
                    guides={[{ value: 0, color: dark ? "#3a4048" : "#c9cfd8", label: "zero" }]}
                    marker={detail.lag !== null ? detail.lag / data.fsHz : null}
                  />
                  <div className="text-[10px] text-muted-foreground leading-relaxed">
                    Lags {detail.lagLo}–{detail.lagHi} ({(detail.lagLo / data.fsHz).toFixed(1)}–
                    {(detail.lagHi / data.fsHz).toFixed(1)} s) are the band; the dashed line is the
                    first local maximum inside it, which is the rate.
                  </div>
                </div>
              </div>
            </div>
          )}

          <p className="text-[11px] text-muted-foreground leading-relaxed">
            A step-for-step copy of FarSense (Zeng et al., IMWUT 2019) on this capture&apos;s
            CSI ratio: Savitzky-Golay smoothing, then for every subcarrier a sweep of {nTheta}{" "}
            projection axes in the complex plane keeping the one whose strongest in-band FFT
            bin holds the largest share of the window&apos;s energy (BNR), then a BNR-weighted
            sum of the autocorrelations of the subcarriers within {keepFraction}× of the best,
            read at its first in-band peak. Two things differ from the paper by necessity: the
            ratio here is the AP&apos;s two transmit chains at one receive chain rather than two
            receive antennas (the same per-packet phase divides out either way), and the
            peak lag is refined between samples because at {data.fsHz.toFixed(0)} Hz one lag is
            up to 1.5 rpm. <b>High-pass</b> and <b>Min peak</b> are additions, both off by
            default.
          </p>
        </>
      )}
    </div>
  );
}
