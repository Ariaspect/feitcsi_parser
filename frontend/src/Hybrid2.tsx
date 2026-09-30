import { useEffect, useMemo, useRef, useState } from "react";

import { ChartBusy } from "@/components/ChartBusy";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  fetchHybrid2,
  type Hybrid2 as Hybrid2Data,
  type HybridState,
  type Hybrid2RangeState,
  type Meta,
} from "./api";
import { Chart, CHART_MARGIN } from "./Chart";
import { formatTime, linearScale, runs } from "./series";
import type { TimeLink } from "./timelink";

const DEFAULT_SPAN_SECONDS = 600;
const STRIP = 16;

// Fixed, like the frame step's own panel: this axis is dimensionless, so one
// scale means two captures can be read against each other. The threshold and
// the floor are drawn on it, which is the whole point of the panel.
const MOTION_AXIS = 0.3;

const STATE_LABEL: Record<HybridState, string> = {
  moving: "moving",
  breathing: "breathing",
  held: "held",
  bridged: "bridged",
  empty: "empty",
  unknown: "no data",
};

const RANGE_LABEL: Record<Hybrid2RangeState, string> = {
  "present:motion": "present (motion)",
  "present:breathing": "present (breathing)",
  empty: "empty",
  unknown: "no data",
};

function rangeColor(state: Hybrid2RangeState, dark: boolean): string {
  switch (state) {
    case "present:motion":
      return "#d97a2f";
    case "present:breathing":
      return "#2f6fed";
    case "empty":
      return dark ? "#2b3038" : "#e6e9ee";
    case "unknown":
      return "url(#hybrid2-no-data)";
  }
}

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
      return "url(#hybrid2-no-data)";
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
        id={id} type="number" min={min} max={max} step={step} className={width}
        value={value}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v) && v >= min && v <= max) onChange(v);
        }}
      />
    </div>
  );
}

export interface Hybrid2Props {
  path: string;
  meta: Meta;
  timeLink?: TimeLink;
  mimo: string;
  sourceMac: string;
  interpolate: boolean;
  dark: boolean;
}

export function Hybrid2({
  path, meta, timeLink, mimo, sourceMac, interpolate, dark,
}: Hybrid2Props) {
  const [lagS, setLagS] = useState(0);
  const [holdS, setHoldS] = useState(20);
  const [burstS, setBurstS] = useState(2);
  const [motionRel, setMotionRel] = useState(2);
  const [motionAbs, setMotionAbs] = useState(0.05);
  const [floorPct, setFloorPct] = useState(20);
  const [breathMinPeak, setBreathMinPeak] = useState(0.2);
  const [breathPersistS, setBreathPersistS] = useState(10);
  const [leadHold, setLeadHold] = useState(true);
  const [gateGain, setGateGain] = useState(false);
  // Which rule the verdict strip draws: the per-second hybrid (own floor,
  // holds) or the fixed-threshold range rule applied to a trailing window.
  const [verdictMode, setVerdictMode] = useState<"hybrid" | "range">("hybrid");
  const [rangeWindow, setRangeWindow] = useState(60);

  const [data, setData] = useState<Hybrid2Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

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
    setBusy(true);
    fetchHybrid2(
      path, range[0], range[1],
      {
        lagS, holdS, burstS, motionRel, motionAbs, floorPct,
        breathMinPeak, breathPersistS, leadHold, gateGain, rangeWindow,
        mimo, sourceMac, interpolate,
      },
      controller.signal,
    )
      .then((result) => { setData(result); setError(null); })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setData(null);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [
    path, range, lagS, holdS, burstS, motionRel, motionAbs, floorPct,
    breathMinPeak, breathPersistS, leadHold, gateGain, rangeWindow, mimo, sourceMac,
    interpolate,
  ]);

  const domain = useMemo<[number, number]>(() => {
    if (!data || data.timeS.length === 0) return range;
    return [data.timeS[0], data.timeS[data.timeS.length - 1]];
  }, [data, range]);

  const stateRuns = useMemo(
    () => (data ? runs<string>(data.timeS, verdictMode === "range" ? data.rangeSeries.state : data.state) : []),
    [data, verdictMode],
  );
  const labelOf = (v: string) => (verdictMode === "range" ? RANGE_LABEL[v as Hybrid2RangeState] : STATE_LABEL[v as HybridState]);
  const colorOf = (v: string) => (verdictMode === "range" ? rangeColor(v as Hybrid2RangeState, dark) : stateColor(v as HybridState, dark));
  const conf = data ? (verdictMode === "range" ? data.rangeConfusion : data.confusion) : null;
  const truthRuns = useMemo(
    () => (data?.truth ? runs(data.truth.timeS, data.truth.present) : []),
    [data],
  );

  const innerWidth = Math.max(1, width - CHART_MARGIN.left - CHART_MARGIN.right);
  const x = linearScale(domain, [0, innerWidth]);
  const clampX = (t: number) => Math.min(innerWidth, Math.max(0, x(t)));
  const vacant = dark ? "#2a2f37" : "#e6e9ee";
  const cam = dark ? "#4f7fd4" : "#3f6fc4";
  const rate = (v: number | null) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

  return (
    <div className="space-y-4" ref={holder}>
      <div className="flex flex-wrap items-center gap-3">
        <NumberField
          id="h2-lag" label="Lag (s)" value={lagS} onChange={setLagS}
          min={0} max={10} step={0.5}
          title="Difference frames this far apart instead of adjacent ones. 0 keeps the adjacent pair. Measured within a capture against its own empty seconds, small movement goes from 19% of its seconds detected at one frame to 88% at 2 s, while a still occupant stays near chance — the lag, not the statistic, is what makes small motion visible."
        />
        <NumberField
          id="h2-hold" label="Hold (s)" value={holdS} onChange={setHoldS}
          min={0} max={600}
          title="Presence runs this long past the last evidence."
        />
        <NumberField
          id="h2-burst" label="Burst (s)" value={burstS} onChange={setBurstS}
          min={1} max={60}
          title="Consecutive seconds above the motion threshold that open presence."
        />
        <NumberField
          id="h2-rel" label="× floor" value={motionRel} onChange={setMotionRel}
          min={1} max={100} step={0.5}
          title="Motion threshold as a multiple of the range's own quiet level."
        />
        <NumberField
          id="h2-abs" label="Min level" value={motionAbs} onChange={setMotionAbs}
          min={0} max={1} step={0.01}
          title="The threshold never falls below this. 0.05 is hybrid 1's 0.10 halved, because this metric is exactly half of |Δr|/|r|."
        />
        <NumberField
          id="h2-floor" label="Floor pct" value={floorPct} onChange={setFloorPct}
          min={0} max={100}
          title="Percentile of the per-second level taken as the range's quiet floor. A range occupied throughout has no quiet stretch — its own percentile IS the occupant, and this detector cannot see past that."
        />
        <NumberField
          id="h2-peak" label="Breath peak" value={breathMinPeak} onChange={setBreathMinPeak}
          min={-1} max={1} step={0.05}
          title="Normalised FarSense autocorrelation peak a window needs to count as breathing."
        />
        <NumberField
          id="h2-persist" label="Persist (s)" value={breathPersistS} onChange={setBreathPersistS}
          min={1} max={120}
          title="Seconds of consecutive qualifying windows that must agree on the rate before breathing counts."
        />
        <Button
          variant={leadHold ? "default" : "outline"} size="sm"
          className="h-7 px-2 text-[11px]"
          title="Breathing also holds presence for the hold length BEFORE it — the person was already there while the first window filled."
          onClick={() => setLeadHold((v) => !v)}
        >
          lead hold {leadHold ? "on" : "off"}
        </Button>
        <Button
          variant={gateGain ? "default" : "outline"} size="sm"
          className="h-7 px-2 text-[11px]"
          title="Blank frame pairs that cross a reported receiver gain state. On the ratio this changes little (the common gain divides out); it is here because the frame step offers it."
          onClick={() => setGateGain((v) => !v)}
        >
          gain gate {gateGain ? "on" : "off"}
        </Button>
        {data && (
          <span className="text-[11px] text-muted-foreground">
            <span
              className={`mr-2 rounded px-1.5 py-0.5 font-semibold ${data.rangeVerdict.present ? "bg-[#2f6fed] text-white" : "bg-[#6b7280] text-white"}`}
              title={`Range verdict, fixed thresholds: present when P90 of the ${data.rangeVerdict.thresholds.lagSeconds} s-lag step > ${data.rangeVerdict.thresholds.motionP90}, else when the FarSense peak ≥ ${data.rangeVerdict.thresholds.breathPeak} for ≥ ${data.rangeVerdict.thresholds.breathRun} consecutive windows. Chosen on 235 one-minute captures (docs/hybrid2.md).`}
            >
              range: {data.rangeVerdict.present ? "present" : "empty"}
              {data.rangeVerdict.by && ` (${data.rangeVerdict.by})`}
              {" · "}P90 {data.rangeVerdict.motionP90?.toFixed(4) ?? "—"} · breath run {data.rangeVerdict.breathRun}
            </span>
            {data.fsHz.toFixed(1)} Hz · {data.selectionNote}
            {data.lagFrames > 1 && ` · lag ${data.lagFrames} frames`}
          </span>
        )}
      </div>

      {error ? (
        <div className="text-muted-foreground p-8 text-sm">
          No verdict for this range — {error}
        </div>
      ) : !data ? (
        <ChartBusy busy empty height={150} label="running the detector" />
      ) : (
        <>
          <ChartBusy busy={busy} height={STRIP * 2 + 8} label="running">
            <svg width={width} height={STRIP * 2 + 8} role="img" aria-label="camera and verdict strips">
              <defs>
                <pattern id="hybrid2-no-data" width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
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
                    <rect key={`s${i}`} x={a} width={Math.max(1, b - a)} y={STRIP + 4} height={STRIP} fill={colorOf(r.value)}>
                      <title>
                        {labelOf(r.value)} · {formatTime(r.t0, domain[1] - domain[0])} – {formatTime(r.t1, domain[1] - domain[0])}
                      </title>
                    </rect>
                  );
                })}
                <text x={-6} y={STRIP * 2} textAnchor="end" fontSize={9} fill="currentColor">{verdictMode === "range" ? "range" : "verdict"}</text>
              </g>
            </svg>
          </ChartBusy>

          <div className="flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
            <Button
              variant={verdictMode === "range" ? "default" : "outline"}
              size="sm"
              className="h-6 px-2 text-[11px]"
              title="Draw the verdict strip from the fixed-threshold range rule (P90 of the 2 s-lag step > 0.035, else a run of 5 FarSense peaks ≥ 0.25), applied every second to the trailing window — what a live system would show. Off: the per-second hybrid with its own floor and holds."
              onClick={() => setVerdictMode((m) => (m === "range" ? "hybrid" : "range"))}
            >
              verdict: {verdictMode === "range" ? "range rule" : "hybrid"}
            </Button>
            {verdictMode === "range" && (
              <NumberField id="h2-rwin" label="window (s)" value={rangeWindow} onChange={setRangeWindow} min={5} max={600} width="w-16"
                title="Trailing seconds the range rule sees at each second. 60 is what the one-minute captures were scored with; shorter exits sooner after the person leaves" />
            )}
            {(verdictMode === "range"
              ? (["present:motion", "present:breathing", "empty", "unknown"] as string[])
              : (["moving", "breathing", "held", "bridged", "empty", "unknown"] as string[])
            ).map((s) => (
              <span key={s} className="flex items-center gap-1.5">
                <svg width={10} height={10}><rect width={10} height={10} fill={colorOf(s)} /></svg>
                {labelOf(s)}
              </span>
            ))}
          </div>

          <ChartBusy busy={busy} height={150} label="running">
            <Chart
              width={width} times={data.timeS} domain={domain}
              yDomain={[0, MOTION_AXIS]} height={150} dark={dark}
              yLabel="motion · ratio-complex frame step"
              series={[
                { values: data.motion, color: "#2f7c5c", width: 1.6, label: "frame step" },
                { values: data.motionReference, color: "#9aa5b1", width: 1, label: "|Δr|/|r| (reference)", dashed: true },
              ]}
              guides={[
                ...(data.threshold != null
                  ? [{ value: data.threshold, color: "#d62728", label: "threshold" }] : []),
                ...(data.floor != null
                  ? [{ value: data.floor, color: "#9aa5b1", label: "floor" }] : []),
              ]}
            />
          </ChartBusy>

          <ChartBusy busy={busy} height={130} label="running">
            <Chart
              width={width} times={data.timeS} domain={domain}
              yDomain={[-0.2, 1]} height={130} dark={dark}
              yLabel="breathing · FarSense peak"
              series={[{ values: data.breathPeak, color: "#2f6fed", width: 1.4, label: "peak" }]}
              guides={[{ value: breathMinPeak, color: "#d62728", label: "min peak" }]}
            />
          </ChartBusy>

          <p className="text-[11px] text-muted-foreground leading-relaxed tabular-nums">
            floor {data.floor?.toFixed(5) ?? "—"} ({data.floorScope}) · threshold{" "}
            {data.threshold?.toFixed(5) ?? "—"} · {data.framesUsed} frames
            {data.framesDropped > 0 && `, ${data.framesDropped} of another shape dropped`}
            {data.nGainCrossed > 0 &&
              ` · ${data.nGainCrossed} pairs cross a gain state${data.gainGated ? " (blanked)" : ""}`}
            {data.breathNote && ` · breathing: ${data.breathNote}`}
          </p>

          {conf && (
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              {verdictMode === "range" ? `range rule (${data.rangeSeries.windowSeconds} s window): ` : "hybrid: "}
              accuracy <b className="text-foreground">{rate(conf.accuracy)}</b> ·
              recall {rate(conf.recall)} · specificity{" "}
              {rate(conf.specificity)} · precision {rate(conf.precision)} ·
              always-present baseline {rate(conf.baseRate)}
              {conf.excluded > 0 &&
                ` · ${conf.excluded} s not scored (within ${conf.marginSeconds} s of a camera transition)`}
            </p>
          )}
          {data.truthExcluded && (
            <p className="text-[11px] text-red-500">
              This capture is flagged {data.truthExcluded} — it yields no truth, so nothing above is scored.
            </p>
          )}

          <p className="text-[11px] text-muted-foreground leading-relaxed">
            <b>Motion</b> is the ratio-complex frame step of the Motion &amp; presence
            panel, reduced to a per-second median and thresholded against the
            range&apos;s <b>own</b> quiet level. It sees motion big or small and
            nothing else — measured within a capture, walking clears its empty
            seconds 4.56× while a still occupant reads 0.94×, correctly, because a
            still body is not moving. <b>Breathing</b> is FarSense, unchanged from
            Hybrid 1, and is what covers the still case; that is the whole reason
            this is a hybrid rather than a threshold.
            <br />
            The known blind spot is the floor, not the metric: a range occupied
            throughout has no quiet stretch, so its own percentile <i>is</i> the
            occupant. Measured on the six 09-21 sitting captures the own-floor rule
            recalls 0 % where a session floor recalls 100 %. Set <b>Min level</b> or
            an explicit floor when you know the link&apos;s idle level.
          </p>
        </>
      )}
    </div>
  );
}
