import { useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { fetchHybrid, type Hybrid as HybridData, type HybridState, type Meta } from "./api";
import { Chart, CHART_MARGIN } from "./Chart";
import { VIRIDIS, buildLut, lutIndex } from "./colormap";
import { formatTime, linearScale, runs } from "./series";
import type { TimeLink } from "./timelink";

const DEFAULT_SPAN_SECONDS = 600;
const STRIP = 16;

const STATE_LABEL: Record<HybridState, string> = {
  moving: "moving",
  breathing: "breathing",
  held: "held",
  bridged: "bridged",
  empty: "empty",
  unknown: "no data",
};

function stateColor(state: HybridState, dark: boolean): string {
  switch (state) {
    case "moving":
      return "#d97a2f";
    case "breathing":
      return "#2f6fed";
    case "held":
      return dark ? "#2c4a7a" : "#b9cdf0";
    case "bridged":
      return dark ? "#3d5f95" : "#95b3e6";
    case "empty":
      return dark ? "#2b3038" : "#e6e9ee";
    case "unknown":
      return "url(#hybrid-no-data)";
  }
}

function NumberField({
  id, label, value, onChange, min, max, step, width = "w-20", title,
}: {
  id: string; label: string; value: number; onChange: (v: number) => void;
  min: number; max: number; step?: number; width?: string; title?: string;
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

function SectionLabel({ children }: { children: string }) {
  return (
    <span className="w-full text-[10px] font-semibold uppercase tracking-wide text-foreground/70 sm:w-auto sm:min-w-[7.5rem]">
      {children}
    </span>
  );
}

function rate(v: number | null): string {
  return v === null ? "—" : `${(v * 100).toFixed(1)}%`;
}

export interface HybridProps {
  path: string;
  meta: Meta;
  timeLink?: TimeLink;
  mimo: string;
  sourceMac: string;
  interpolate: boolean;
  dark: boolean;
}

/** The calibration-free detector: motion opens presence, breathing keeps it
 *  open, nothing keeps it open without fresh evidence. Scored live against
 *  the camera with the same margin the Phase 1 panel uses. */
export function Hybrid({ path, meta, timeLink, mimo, sourceMac, interpolate, dark }: HybridProps) {
  const [holdS, setHoldS] = useState(20);
  const [burstS, setBurstS] = useState(2);
  const [motionRel, setMotionRel] = useState(2);
  const [motionAbs, setMotionAbs] = useState(0.1);
  const [useAmplitude, setUseAmplitude] = useState(false);
  const [leadHold, setLeadHold] = useState(true);
  const [breathMinPeak, setBreathMinPeak] = useState(0.2);
  const [breathPersistS, setBreathPersistS] = useState(15);
  const [breathRateTol, setBreathRateTol] = useState(3);
  const [breathWindow, setBreathWindow] = useState(15);
  const [breathHighpass, setBreathHighpass] = useState(0);
  const [rpmLo, setRpmLo] = useState(10);
  const [rpmHi, setRpmHi] = useState(30);
  const [keepFraction, setKeepFraction] = useState(0.6);
  const [nTheta, setNTheta] = useState(200);
  const [fftSize, setFftSize] = useState(8192);
  const [savgolSeconds, setSavgolSeconds] = useState(1.0);
  const [savgolOrder, setSavgolOrder] = useState(3);
  const [motionFracHi, setMotionFracHi] = useState<number | null>(null);
  const [positiveOnly, setPositiveOnly] = useState(true);
  const [maxGapFraction, setMaxGapFraction] = useState(0.5);
  const [motionFloor, setMotionFloor] = useState<number | null>(null);
  const [marginS, setMarginS] = useState(5);
  const [data, setData] = useState<HybridData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

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
      setRange(([t0, t1]) => (w.tMin === t0 && w.tMax === t1 ? [t0, t1] : [w.tMin, w.tMax]));
    });
  }, [timeLink]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetchHybrid(
      path, range[0], range[1],
      {
        holdS, burstS, motionRel, motionAbs, useAmplitude, leadHold, breathMinPeak, breathPersistS,
        breathRateTol, breathWindow, breathHighpass, rpmLo, rpmHi, keepFraction, nTheta, fftSize,
        savgolSeconds, savgolOrder, motionFracHi, positiveOnly, maxGapFraction,
        motionFloor, marginS, mimo, sourceMac, interpolate,
      },
      controller.signal,
    )
      .then((result) => { setData(result); setError(null); })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setData(null);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [
    path, range, holdS, burstS, motionRel, motionAbs, useAmplitude, leadHold, breathMinPeak,
    breathPersistS, breathRateTol, breathWindow, breathHighpass, rpmLo, rpmHi, keepFraction, nTheta,
    fftSize, savgolSeconds, savgolOrder, motionFracHi, positiveOnly, maxGapFraction,
    motionFloor, marginS, mimo, sourceMac, interpolate,
  ]);

  const domain = useMemo<[number, number]>(() => {
    if (!data || data.timeS.length === 0) return range;
    return [data.timeS[0] - 0.5, data.timeS[data.timeS.length - 1] + 0.5];
  }, [data, range]);

  const innerWidth = Math.max(1, width - CHART_MARGIN.left - CHART_MARGIN.right);
  const x = linearScale(domain, [0, innerWidth]);
  const clampX = (t: number) => Math.max(0, Math.min(innerWidth, x(t)));

  const stateRuns = useMemo(() => (data ? runs(data.timeS, data.state) : []), [data]);
  const truthRuns = useMemo(
    () => (data?.truth ? runs(data.truth.timeS, data.truth.present) : []),
    [data],
  );
  const tally = useMemo(() => {
    const counts: Record<HybridState, number> = { moving: 0, breathing: 0, held: 0, bridged: 0, empty: 0, unknown: 0 };
    for (const s of data?.state ?? []) counts[s] += 1;
    return counts;
  }, [data]);

  const motionCeiling = useMemo(() => {
    const finite = (data?.motionRatio ?? []).filter((v): v is number => v !== null);
    const thr = data?.ratioThreshold ?? 0.1;
    const p95 = finite.length ? [...finite].sort((a, b) => a - b)[Math.floor(finite.length * 0.95)] : thr;
    return Math.max(thr * 1.5, p95 * 1.3, 0.05);
  }, [data]);
  const ampCeiling = useMemo(() => {
    const finite = (data?.motionAmp ?? []).filter((v): v is number => v !== null);
    const thr = data?.ampThreshold ?? 0.5;
    const p95 = finite.length ? [...finite].sort((a, b) => a - b)[Math.floor(finite.length * 0.95)] : thr;
    return Math.max(thr * 1.5, p95 * 1.3, 0.5);
  }, [data]);

  // Every FarSense window's rate, coloured by the normalised peak behind it,
  // as on the FarSense tab; the line on top is where the hybrid counted it.
  const peakColors = useMemo(() => {
    if (!data) return [];
    const lut = buildLut(VIRIDIS);
    return data.breathPeak.map((v) => {
      const li = v === null ? -1 : lutIndex(v, 0, 0.5);
      if (li < 0) return "transparent";
      const packed = lut[li];
      return `rgb(${packed & 0xff},${(packed >> 8) & 0xff},${(packed >> 16) & 0xff})`;
    });
  }, [data]);

  const cam = dark ? "#4f7fd4" : "#3f6fc4";
  const vacant = dark ? "#2a2f37" : "#e6e9ee";
  const c = data?.confusion ?? null;

  return (
    <div className="space-y-4" ref={holder}>
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-3">
          <SectionLabel>Hybrid detection</SectionLabel>
          <NumberField id="hy-hold" label="Hold (s)" value={holdS} onChange={setHoldS} min={0} max={600}
            title="Presence is kept this long after the last evidence (and, with lead hold, before breathing), then dropped" />
          <NumberField id="hy-burst" label="Burst (s)" value={burstS} onChange={setBurstS} min={1} max={60}
            title="Consecutive seconds above the motion threshold that count as a burst" />
          <NumberField id="hy-rel" label="Motion ×floor" value={motionRel} onChange={setMotionRel} min={1} max={100} step={0.5}
            title="Motion threshold as a multiple of the range's own quiet level (20th percentile)" />
          <NumberField id="hy-abs" label="min" value={motionAbs} onChange={setMotionAbs} min={0} max={10} step={0.01} width="w-16"
            title="…but never below this |Δr|/|r|" />
          <div className="flex items-center gap-2"
            title="The quiet level is this range's own 20th percentile of the per-second motion; type a value to use instead">
            <Label htmlFor="hy-floor" className="text-[10px] text-muted-foreground uppercase tracking-wide">Floor</Label>
            <Input id="hy-floor" type="number" min={0} max={10} step={0.005} className="w-20" placeholder="auto"
              value={motionFloor ?? ""}
              onChange={(e) => {
                const raw = e.target.value;
                if (raw === "") { setMotionFloor(null); return; }
                const v = Number(raw);
                if (Number.isFinite(v) && v >= 0 && v <= 10) setMotionFloor(v);
              }} />
          </div>
          <Button
            variant={useAmplitude ? "default" : "outline"}
            size="sm"
            className="h-7 px-2 text-[11px]"
            title="Also count bursts on the raw amplitude frame-diff channel (sees shadowing; also sees gain steps)"
            onClick={() => setUseAmplitude((v) => !v)}
          >
            amplitude {useAmplitude ? "on" : "off"}
          </Button>
          <Button
            variant={leadHold ? "default" : "outline"}
            size="sm"
            className="h-7 px-2 text-[11px]"
            title="Breathing also holds presence for the hold length before it; where that meets the hold running out of earlier evidence, the whole gap is present (bridged). Off: presence only runs forward from evidence."
            onClick={() => setLeadHold((v) => !v)}
          >
            lead hold {leadHold ? "on" : "off"}
          </Button>
          <NumberField id="hy-peak" label="Breath peak ≥" value={breathMinPeak} onChange={setBreathMinPeak} min={-1} max={1} step={0.05}
            title="Normalised FarSense peak a window needs to count as breathing (an empty room sits near 0 ± 0.1, an occupant 0.2–0.6)" />
          <NumberField id="hy-persist" label="for (s)" value={breathPersistS} onChange={setBreathPersistS} min={1} max={120} width="w-16"
            title="…through this many seconds of consecutive windows, 80% of which must qualify and agree on the rate" />
          <NumberField id="hy-tol" label="± rpm" value={breathRateTol} onChange={setBreathRateTol} min={0} max={60} step={0.5} width="w-16"
            title="How far a window's rate may sit from the run's median and still agree" />
          <NumberField id="hy-margin" label="Label margin (s)" value={marginS} onChange={setMarginS} min={0} max={60} width="w-16"
            title="Empty camera frames within this many seconds of a transition are not scored" />
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <SectionLabel>FarSense</SectionLabel>
          <NumberField id="hy-win" label="Window (s)" value={breathWindow} onChange={setBreathWindow} min={4} max={120}
            title="Longer: a steady rhythm stands out more and the rate is finer, but it assumes the rate holds for the whole window and evidence sits at the window's centre. Shorter: reacts faster and tolerates a wandering rate, but peaks are noisier." />
          <NumberField id="hy-lo" label="rpm" value={rpmLo} onChange={setRpmLo} min={1} max={119} width="w-16"
            title="Narrower band: fewer chances for noise to pass as a peak, but a real rate outside it is missed. Wider: catches unusual rates at the cost of more false peaks; the slow edge must fit in the window." />
          <NumberField id="hy-hi" label="–" value={rpmHi} onChange={setRpmHi} min={2} max={120} width="w-16" />
          <NumberField id="hy-keep" label="Keep ≥ ×best" value={keepFraction} onChange={setKeepFraction} min={0} max={1} step={0.05}
            title="Higher (towards 1): only the very best subcarriers vote — a sharper peak when a chest is there, jumpier window to window. Lower: every subcarrier votes and the ones that barely see the chest dilute the peak." />
          <NumberField id="hy-theta" label="θ steps" value={nTheta} onChange={setNTheta} min={2} max={720} step={2}
            title="More: a finer search for the direction the breath moves the ratio along; above ~50 nothing changes but the compute. Fewer: below ~10 the direction can be missed and the peak falls." />
          <NumberField id="hy-fft" label="FFT" value={fftSize} onChange={setFftSize} min={64} max={65536} step={64} width="w-20"
            title="Zero-padded spectrum size for the BNR that picks the projection angle and the subcarriers; finer bins above ~2048 change little" />
          <NumberField id="hy-sg" label="S-G (s)" value={savgolSeconds} onChange={setSavgolSeconds} min={0} max={5} step={0.1} width="w-16"
            title="Longer: a smoother pattern with less per-packet jitter, but fast breaths flatten (1–1.5 s begins to eat 30 rpm). 0: the raw packets." />
          <NumberField id="hy-sgo" label="order" value={savgolOrder} onChange={setSavgolOrder} min={1} max={7} width="w-14"
            title="Savitzky-Golay polynomial order; higher keeps sharper features through the smoothing" />
          <NumberField id="hy-hp" label="High-pass (Hz)" value={breathHighpass} onChange={setBreathHighpass} min={0} max={2} step={0.05}
            title="Higher: removes slower wander (drift, a person settling) so it cannot pass as a slow breath; above ~0.15 it starts eating 10 rpm breathing itself. 0: nothing removed." />
          <div className="flex items-center gap-2"
            title="The paper's stationary gate: a window whose median fractional change is above this reports no peak. Empty: off — every window reports and the burst rule above decides what motion means. Lower: stricter, fewer windows rated but cleaner.">
            <Label htmlFor="hy-gate" className="text-[10px] text-muted-foreground uppercase tracking-wide">Motion gate</Label>
            <Input id="hy-gate" type="number" min={0.01} max={5} step={0.01} className="w-16" placeholder="off"
              value={motionFracHi ?? ""}
              onChange={(e) => {
                const raw = e.target.value;
                if (raw === "") { setMotionFracHi(null); return; }
                const v = Number(raw);
                if (Number.isFinite(v) && v > 0 && v <= 5) setMotionFracHi(v);
              }} />
          </div>
          <NumberField id="hy-gap" label="Max gap" value={maxGapFraction} onChange={setMaxGapFraction} min={0.05} max={1} step={0.05} width="w-16"
            title="A second or window more than this fraction interpolated across dropouts reports nothing (no data)" />
          <Button
            variant={positiveOnly ? "default" : "outline"}
            size="sm"
            className="h-7 px-2 text-[11px]"
            title="On: the breath is the first POSITIVE local maximum of the autocorrelation — a negative first maximum is a wiggle on the slope out of the trough, not a period. Off: the paper's literal first local maximum."
            onClick={() => setPositiveOnly((v) => !v)}
          >
            first positive peak {positiveOnly ? "on" : "off"}
          </Button>
          {data && (
            <span className="text-[11px] text-muted-foreground tabular-nums">
              {data.fsHz.toFixed(1)} Hz · floor {data.ratioFloor?.toFixed(3) ?? "—"} ({data.floorScope}) → threshold{" "}
              {data.ratioThreshold?.toFixed(3) ?? "—"}
              {useAmplitude && data.ampThreshold !== null && <> · amp {data.ampFloor?.toFixed(2)} → {data.ampThreshold.toFixed(2)} dB</>}
              {loading && " · refreshing"}
            </span>
          )}
        </div>
      </div>

      {error ? (
        <div className="text-muted-foreground p-8 text-sm">No verdict for this range — {error}</div>
      ) : !data ? (
        <div className="text-muted-foreground p-8 text-sm">
          {loading ? "Analysing this range…" : "Nothing to analyse yet."}
        </div>
      ) : (
        <>
          <div>
            <svg width={width} height={STRIP * 2 + 8} role="img" aria-label="camera and verdict strips">
              <defs>
                <pattern id="hybrid-no-data" width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <rect width={6} height={6} fill={dark ? "#1b1f25" : "#f2f4f7"} />
                  <line x1={0} y1={0} x2={0} y2={6} stroke={dark ? "#3a424d" : "#c9cfd8"} strokeWidth={2} />
                </pattern>
              </defs>
              <g transform={`translate(${CHART_MARGIN.left},2)`}>
                <rect x={0} width={innerWidth} y={0} height={STRIP} fill={vacant} />
                {truthRuns.map((r, i) => {
                  const a = clampX(r.t0); const b = clampX(r.t1);
                  return r.value && b > a ? (
                    <rect key={`t${i}`} x={a} width={Math.max(1, b - a)} y={0} height={STRIP} fill={cam} />
                  ) : null;
                })}
                <text x={-6} y={STRIP - 4} textAnchor="end" fontSize={9} fill="currentColor">camera</text>
                {stateRuns.map((r, i) => {
                  const a = clampX(r.t0); const b = clampX(r.t1);
                  if (b <= a) return null;
                  return (
                    <rect key={`s${i}`} x={a} width={Math.max(1, b - a)} y={STRIP + 4} height={STRIP} fill={stateColor(r.value, dark)}>
                      <title>{STATE_LABEL[r.value]} · {formatTime(r.t0, domain[1] - domain[0])} – {formatTime(r.t1, domain[1] - domain[0])}</title>
                    </rect>
                  );
                })}
                <text x={-6} y={STRIP * 2} textAnchor="end" fontSize={9} fill="currentColor">verdict</text>
              </g>
            </svg>
            <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
              <span><span style={{ color: cam }}>■</span> person in frame</span>
              {(["moving", "breathing", "held", "bridged", "empty", "unknown"] as HybridState[]).map((s) => (
                <span key={s} className="flex items-center gap-1.5">
                  <svg width={10} height={10}><rect width={10} height={10} fill={stateColor(s, dark)} /></svg>
                  {STATE_LABEL[s]} · {tally[s]}
                </span>
              ))}
              {!data.truth && <span>no camera sidecar beside this capture</span>}
            </div>
          </div>

          {c && (
            <div className="flex flex-wrap items-start gap-6">
              <table className="text-[11px] tabular-nums">
                <tbody>
                  <tr><td className="w-24" /><td className="px-3 py-0.5 text-muted-foreground" colSpan={2}>camera</td></tr>
                  <tr><td /><td className="px-3 py-0.5 text-muted-foreground">occupied</td><td className="px-3 py-0.5 text-muted-foreground">empty</td></tr>
                  <tr><td className="py-0.5 pr-2 text-right text-muted-foreground">said present</td><td className="px-3 py-0.5">{c.tp}</td><td className="px-3 py-0.5">{c.fp}</td></tr>
                  <tr><td className="py-0.5 pr-2 text-right text-muted-foreground">said empty</td><td className="px-3 py-0.5">{c.fn}</td><td className="px-3 py-0.5">{c.tn}</td></tr>
                </tbody>
              </table>
              <div className="text-[11px] text-muted-foreground tabular-nums leading-relaxed">
                <div>
                  accuracy <b className="text-foreground">{rate(c.accuracy)}</b> · recall {rate(c.recall)} · specificity{" "}
                  {rate(c.specificity)} · balanced{" "}
                  <b className="text-foreground">
                    {c.recall !== null && c.specificity !== null ? rate((c.recall + c.specificity) / 2) : "—"}
                  </b>
                </div>
                <div>
                  always-present baseline {rate(c.baseRate)} · {c.excluded} s not scored (empty, within {c.marginSeconds} s of a transition)
                </div>
                <div>No empty-room reference was used: thresholds come from this range&apos;s own quiet floor.</div>
              </div>
            </div>
          )}

          <Chart
            width={width}
            times={data.timeS}
            domain={domain}
            yDomain={[0, motionCeiling]}
            yLabel="motion |Δr|/|r| (per-second median)"
            dark={dark}
            height={120}
            series={[{ values: data.motionRatio, color: "#d97a2f", width: 1.4, label: "ratio motion" }]}
            guides={data.ratioThreshold !== null ? [{ value: data.ratioThreshold, color: "#d62728", label: "burst" }] : []}
          />
          {useAmplitude && (
            <Chart
              width={width}
              times={data.timeS}
              domain={domain}
              yDomain={[0, ampCeiling]}
              yLabel="motion |ΔA| dB (raw amplitude)"
              dark={dark}
              height={110}
              series={[{ values: data.motionAmp, color: "#c7a02f", width: 1.4, label: "amp motion" }]}
              guides={data.ampThreshold !== null ? [{ value: data.ampThreshold, color: "#d62728", label: "burst" }] : []}
            />
          )}
          <Chart
            width={width}
            times={data.timeS}
            domain={domain}
            yDomain={[-0.3, 1]}
            yLabel="breathing peak (FarSense, normalised)"
            dark={dark}
            height={120}
            series={[{ values: data.breathPeak, color: "#7b5ea7", width: 1.4, label: "peak" }]}
            guides={[{ value: breathMinPeak, color: "#d62728", label: "min peak" }]}
          />
          <Chart
            width={width}
            times={data.timeS}
            domain={domain}
            yDomain={[Math.max(0, rpmLo - 2), rpmHi + 3]}
            yLabel="FarSense rate (rpm): every window as a dot, line where breathing counted"
            dark={dark}
            height={120}
            series={[{ values: data.breathRpm.map((v, i) => (data.breathing[i] ? v : null)), color: "#2f6fed", width: 1.8, label: "counted" }]}
            dots={{ values: data.breathRpm, colors: peakColors, radius: 2, label: "rate" }}
          />
          <p className="text-[11px] text-muted-foreground">
            Dots are coloured by the normalised peak the rate was read from (dark = near 0, yellow = 0.5 or more), as on the FarSense tab.
          </p>
          {data.breathNote && (
            <p className="text-[11px] text-muted-foreground">Breathing channel off for this range: {data.breathNote}</p>
          )}
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            A <b>burst</b> is {burstS} s or more of per-second motion above {motionRel}× this range&apos;s own
            quiet level (its 20th percentile), never below {motionAbs}. <b>Breathing</b> is the FarSense peak
            ({rpmLo}–{rpmHi} rpm, {breathWindow} s windows, {breathHighpass} Hz high-pass) at or above {breathMinPeak}
            through {breathPersistS} s of consecutive windows whose rates agree within ±{breathRateTol} rpm — neighbouring windows share almost all their samples, so a short run
            proves nothing. Either opens presence; presence then holds {holdS} s past the last evidence and drops.
            A moved chair is a burst followed by nothing; a fan is motion with no breathing and no bursts — the
            first is handled, the second is not yet, for want of a capture to test on.
          </p>
        </>
      )}
    </div>
  );
}
