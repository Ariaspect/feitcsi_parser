import { useEffect, useMemo, useRef, useState } from "react";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  fetchPresence,
  type Meta,
  type Presence as PresenceData,
  type PresenceChannel,
  type PresenceState,
} from "./api";
import { formatTime, linePath, linearScale, runs, ticks } from "./series";
import type { TimeLink } from "./timelink";

// Longest stretch analysed before the user asks for more. Presence needs
// full-rate samples -- an autocorrelation cannot be fed stride-sampled frames
// without destroying the very periodicity it is looking for -- so the whole
// range in view is decoded, and a four-hour capture opened whole would decode
// every frame it holds. Ten minutes is long enough to hold dozens of windows
// and short enough to come back promptly.
const DEFAULT_SPAN_SECONDS = 600;

const STATE_LABEL: Record<PresenceState, string> = {
  present: "still occupant",
  moving: "motion",
  empty: "empty",
  unknown: "no data",
};

function stateColor(state: PresenceState, dark: boolean): string {
  switch (state) {
    case "present":
      return "#2f6fed";
    case "moving":
      return "#d97a2f";
    case "empty":
      return dark ? "#2b3038" : "#e6e9ee";
    case "unknown":
      // Never a flat fill: absence of data has to look different from a
      // verdict, or the strip quietly asserts an empty room across a dropout.
      return "url(#presence-no-data)";
  }
}

interface ChartSeries {
  values: (number | null)[];
  color: string;
  width?: number;
  label: string;
  dashed?: boolean;
}

interface ChartProps {
  times: number[];
  domain: [number, number];
  yDomain: [number, number];
  series: ChartSeries[];
  guides?: { value: number; color: string; label: string }[];
  height?: number;
  yLabel: string;
  dark: boolean;
  width: number;
}

// Top margin holds the axis label clear of the highest tick label; at 8 the
// two sit on the same line and overprint each other.
const MARGIN = { top: 20, right: 12, bottom: 18, left: 44 };

function Chart({
  times,
  domain,
  yDomain,
  series,
  guides = [],
  height = 120,
  yLabel,
  dark,
  width,
}: ChartProps) {
  const inner = {
    w: Math.max(1, width - MARGIN.left - MARGIN.right),
    h: Math.max(1, height - MARGIN.top - MARGIN.bottom),
  };
  const x = linearScale(domain, [0, inner.w]);
  const y = linearScale(yDomain, [inner.h, 0]);
  const grid = dark ? "#2a2f37" : "#e6e9ee";
  const text = dark ? "#8b95a3" : "#6b7480";
  const span = domain[1] - domain[0];

  return (
    <svg width={width} height={height} role="img" aria-label={yLabel}>
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        {ticks(yDomain[0], yDomain[1], 4).map((v) => (
          <g key={v}>
            <line x1={0} x2={inner.w} y1={y(v)} y2={y(v)} stroke={grid} strokeWidth={1} />
            <text x={-6} y={y(v)} dy="0.32em" textAnchor="end" fontSize={9} fill={text}>
              {v}
            </text>
          </g>
        ))}
        {ticks(domain[0], domain[1], 6).map((v) => (
          <text
            key={v}
            x={x(v)}
            y={inner.h + 12}
            textAnchor="middle"
            fontSize={9}
            fill={text}
          >
            {formatTime(v, span)}
          </text>
        ))}
        {guides.map((g) => (
          <line
            key={g.label}
            x1={0}
            x2={inner.w}
            y1={y(g.value)}
            y2={y(g.value)}
            stroke={g.color}
            strokeWidth={1}
            strokeDasharray="4 3"
          />
        ))}
        {series.map((s) => (
          <path
            key={s.label}
            d={linePath(times, s.values, x, y)}
            fill="none"
            stroke={s.color}
            strokeWidth={s.width ?? 1}
            strokeDasharray={s.dashed ? "3 2" : undefined}
            strokeLinejoin="round"
          />
        ))}
        <text x={-MARGIN.left + 2} y={-8} fontSize={9} fill={text}>
          {yLabel}
        </text>
      </g>
    </svg>
  );
}

export interface PresenceProps {
  path: string;
  meta: Meta;
  timeLink?: TimeLink;
  mimo: string;
  sourceMac: string;
  interpolate: boolean;
  dark: boolean;
}

export function Presence({
  path,
  meta,
  timeLink,
  mimo,
  sourceMac,
  interpolate,
  dark,
}: PresenceProps) {
  const [channel, setChannel] = useState<PresenceChannel>("complex");
  const [windowSeconds, setWindowSeconds] = useState(12);
  const [threshold, setThreshold] = useState(0.25);
  const [motionFracHi, setMotionFracHi] = useState(0.25);
  const [data, setData] = useState<PresenceData | null>(null);
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

  // Follow the shared time axis, so zooming a heatmap moves this panel with
  // it. Publishing back is deliberately not done: this panel has no zoom of
  // its own to broadcast, and echoing a window it was just handed would loop.
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
    setLoading(true);
    fetchPresence(
      path,
      range[0],
      range[1],
      {
        channel,
        windowSeconds,
        presentThreshold: threshold,
        motionFracHi,
        mimo,
        sourceMac,
        interpolate,
      },
      controller.signal,
    )
      .then((result) => {
        setData(result);
        setError(null);
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
  }, [
    path, range, channel, windowSeconds, threshold, motionFracHi,
    mimo, sourceMac, interpolate,
  ]);

  const domain = useMemo<[number, number]>(() => {
    if (!data || data.timeS.length === 0) return range;
    return [data.timeS[0], data.timeS[data.timeS.length - 1]];
  }, [data, range]);

  const stateRuns = useMemo(
    () => (data ? runs(data.timeS, data.state) : []),
    [data],
  );

  const tally = useMemo(() => {
    const counts: Record<PresenceState, number> = {
      present: 0, moving: 0, empty: 0, unknown: 0,
    };
    for (const s of data?.state ?? []) counts[s] += 1;
    return counts;
  }, [data]);

  const motionCeiling = useMemo(() => {
    const finite = (data?.motionLevel ?? []).filter(
      (v): v is number => v !== null && Number.isFinite(v),
    );
    // Scaled to the data, with the gross-motion threshold always in frame so
    // the trace can be read against the line that classifies it.
    return Math.max(motionFracHi * 1.3, ...finite.map((v) => v * 1.2), 0.05);
  }, [data, motionFracHi]);

  const stripHeight = 26;
  const innerWidth = Math.max(1, width - MARGIN.left - MARGIN.right);
  const stripX = linearScale(domain, [0, innerWidth]);

  return (
    <div className="space-y-4" ref={holder}>
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <Label className="text-[10px] text-muted-foreground uppercase tracking-wide">
            Channel
          </Label>
          <Select value={channel} onValueChange={(v) => setChannel(v as PresenceChannel)}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="complex">complex</SelectItem>
              <SelectItem value="phase">phase</SelectItem>
              <SelectItem value="magnitude">magnitude</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Label htmlFor="presence-win" className="text-[10px] text-muted-foreground uppercase tracking-wide">
            Window (s)
          </Label>
          <Input
            id="presence-win"
            type="number"
            min={4}
            max={120}
            className="w-20"
            value={windowSeconds}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v) && v >= 4 && v <= 120) setWindowSeconds(v);
            }}
          />
        </div>

        <div className="flex items-center gap-2">
          <Label htmlFor="presence-thr" className="text-[10px] text-muted-foreground uppercase tracking-wide">
            Present above
          </Label>
          <Input
            id="presence-thr"
            type="number"
            min={0}
            max={1}
            step={0.05}
            className="w-20"
            value={threshold}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v) && v >= 0 && v <= 1) setThreshold(v);
            }}
          />
        </div>

        <div className="flex items-center gap-2">
          <Label htmlFor="presence-motion" className="text-[10px] text-muted-foreground uppercase tracking-wide">
            Motion above
          </Label>
          <Input
            id="presence-motion"
            type="number"
            min={0.01}
            max={2}
            step={0.01}
            className="w-20"
            value={motionFracHi}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isFinite(v) && v >= 0.01 && v <= 2) setMotionFracHi(v);
            }}
          />
        </div>

        {data && (
          <span className="text-[11px] text-muted-foreground">
            {data.fsHz.toFixed(1)} Hz · {data.timeS.length} windows ·{" "}
            {data.windowSeconds.toFixed(1)} s each · floor{" "}
            {data.rpmFloorEff.toFixed(0)} rpm
            {loading && " · refreshing"}
          </span>
        )}
      </div>

      {error ? (
        <div className="text-muted-foreground p-8 text-sm">
          No verdict for this range — {error}
        </div>
      ) : !data ? (
        // "Nothing here" and "not asked yet" are different claims, and on a
        // presence panel the first one is a verdict. A range this size can
        // take seconds to decode at full rate, and saying the room is empty
        // for that whole time is the failure this panel exists to avoid.
        <div className="text-muted-foreground p-8 text-sm">
          {loading ? "Analysing this range…" : "Nothing to analyse yet."}
        </div>
      ) : data.timeS.length === 0 ? (
        <div className="text-muted-foreground p-8 text-sm">
          This range holds no complete analysis window — widen it, or shorten
          the window length.
        </div>
      ) : (
        <>
          <div>
            <svg width={width} height={stripHeight + 20}>
              <defs>
                {/* Absence of data must not look like a verdict. */}
                <pattern
                  id="presence-no-data"
                  width={6}
                  height={6}
                  patternUnits="userSpaceOnUse"
                  patternTransform="rotate(45)"
                >
                  <rect width={6} height={6} fill={dark ? "#1b1f25" : "#f2f4f7"} />
                  <line
                    x1={0} y1={0} x2={0} y2={6}
                    stroke={dark ? "#3a424d" : "#c9cfd8"}
                    strokeWidth={2}
                  />
                </pattern>
              </defs>
              <g transform={`translate(${MARGIN.left},0)`}>
                {stateRuns.map((run, i) => {
                  // A run reaches half a window past the outermost centres,
                  // but the domain stops at those centres -- so the first and
                  // last blocks are clipped to the plot area. Left as-is the
                  // strip sits offset from the traces below it, and reading
                  // one against the other is the whole point of stacking them.
                  const x0 = Math.max(0, Math.min(innerWidth, stripX(run.t0)));
                  const x1 = Math.max(0, Math.min(innerWidth, stripX(run.t1)));
                  if (x1 <= x0) return null;
                  return (
                  <rect
                    key={`${run.t0}-${i}`}
                    x={x0}
                    width={Math.max(1, x1 - x0)}
                    y={0}
                    height={stripHeight}
                    fill={stateColor(run.value, dark)}
                  >
                    <title>
                      {STATE_LABEL[run.value]} · {formatTime(run.t0, domain[1] - domain[0])}
                      {" – "}
                      {formatTime(run.t1, domain[1] - domain[0])}
                    </title>
                  </rect>
                  );
                })}
              </g>
            </svg>
            <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
              {(["present", "moving", "empty", "unknown"] as PresenceState[]).map((s) => (
                <span key={s} className="flex items-center gap-1.5">
                  <svg width={10} height={10}>
                    <rect width={10} height={10} fill={stateColor(s, dark)} />
                  </svg>
                  {STATE_LABEL[s]} · {tally[s]}
                </span>
              ))}
            </div>
          </div>

          <Chart
            width={width}
            times={data.timeS}
            domain={domain}
            yDomain={[0, motionCeiling]}
            yLabel="motion |Δr|/|r|"
            dark={dark}
            series={[
              { values: data.motionLevel, color: "#d97a2f", width: 1.6, label: "motion" },
            ]}
            guides={[{ value: motionFracHi, color: "#d62728", label: "gross motion" }]}
          />

          <Chart
            width={width}
            times={data.timeS}
            domain={domain}
            yDomain={[0, 1]}
            yLabel="breathing score"
            dark={dark}
            height={150}
            series={[
              { values: data.periodicity, color: "#9aa5b1", label: "periodicity" },
              { values: data.tonality, color: "#c7a02f", label: "tonality" },
              { values: data.motionGate, color: "#2f9c6f", label: "motion gate", dashed: true },
              { values: data.score, color: "#2f6fed", width: 2, label: "score" },
            ]}
            guides={[{ value: threshold, color: "#d62728", label: "threshold" }]}
          />

          <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
            <span style={{ color: "#2f6fed" }}>score</span>
            <span style={{ color: "#9aa5b1" }}>periodicity</span>
            <span style={{ color: "#c7a02f" }}>tonality</span>
            <span style={{ color: "#2f9c6f" }}>motion gate</span>
            <span>= score is their product</span>
          </div>

          {tally.present === 0 && (
            <p className="text-[11px] text-muted-foreground">
              No window in this range was called a still occupant, so the rate
              axis below is empty by construction — not a chart that failed to
              draw.
            </p>
          )}

          <Chart
            width={width}
            times={data.timeS}
            domain={domain}
            yDomain={[data.params.rate_band_rpm[0], data.params.rate_band_rpm[1]]}
            yLabel="rate (rpm)"
            dark={dark}
            height={110}
            series={[
              {
                // Drawn only where a still occupant was actually claimed. A
                // rate is the lag of the largest autocorrelation peak inside
                // the band, and that exists in every window whether or not
                // anything was breathing -- plotting it unconditionally shows
                // a confident breathing rate for an empty room.
                values: data.rateRpm.map((v, i) =>
                  data.state[i] === "present" ? v : null,
                ),
                color: "#2f6fed",
                width: 1.6,
                label: "rate",
              },
            ]}
          />

          {data.warnings.length > 0 && (
            <ul className="text-[11px] text-muted-foreground leading-relaxed list-disc pl-4">
              {data.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}

          <p className="text-[11px] text-muted-foreground leading-relaxed">
            Built on the <b>swap-corrected</b> CSI ratio: uncorrected, 1.2% of
            frame-to-frame steps exceed π, and a π step is a broadband impulse
            with energy inside the respiration band — it would read as
            breathing in an empty room. A still occupant is decided on{" "}
            <b>periodicity</b>, not on energy: score = periodicity × tonality ×
            motion gate, and each term is a veto. Hatched stretches are windows
            more than half interpolated across a capture dropout; they report{" "}
            <b>no data</b> rather than “empty”, because a bridged hole is flat
            and flat scores exactly like an absent occupant.{" "}
            {sourceMac === "all" && (
              <>
                <b>Pick a single source MAC.</b> Two transmitters interleaved
                are two different channels, and alternating between them reads
                as movement that is not there — measured at 0.53 mixed against
                0.37–0.47 per transmitter on capture.dat.
              </>
            )}
          </p>
        </>
      )}
    </div>
  );
}
