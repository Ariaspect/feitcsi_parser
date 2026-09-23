import { useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  fetchMotionSignal,
  type Meta,
  type MotionSignal as MotionSignalData,
  type MotionSignalMode,
} from "./api";
import { Chart, CHART_MARGIN, type ChartBand } from "./Chart";
import { formatTime, linearScale, runs } from "./series";
import type { TimeLink } from "./timelink";

const DEFAULT_SPAN_SECONDS = 600;
const STRIP = 16;

const MODE_LABEL: Record<MotionSignalMode, string> = {
  label: "label-based (what was measured)",
  free: "label-free (deployable)",
};

const MODE_SHORT: Record<MotionSignalMode, string> = {
  label: "label-based",
  free: "label-free",
};

const VAR_COLOR = "#0d8a94";
const LAG_COLOR = "#c2701d";
const THRESHOLD_COLOR = "#d62728";

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

/** Symmetric-ish limits with a little headroom, from the finite values only. */
function extent(values: (number | null)[], pad = 0.08, floor = 1): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v === null || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [-floor, floor];
  if (hi - lo < 1e-9) return [lo - floor, hi + floor];
  const m = (hi - lo) * pad;
  return [lo - m, hi + m];
}

/** Contiguous spans where `flag` is true, as time bands for the chart. */
function bandsOf(times: number[], flag: boolean[], half: number): ChartBand[] {
  return runs(times, flag)
    .filter((r) => r.value)
    .map((r) => ({ t0: r.t0 - half, t1: r.t1 + half }));
}

export interface MotionSignalProps {
  path: string;
  meta: Meta;
  timeLink?: TimeLink;
  mimo: string;
  sourceMac: string;
  interpolate: boolean;
  dark: boolean;
}

/** The amplitude-motion signal the stage-1/2 experiment settled on: the
 *  ratio's variance and lag-1 autocorrelation per 2 s window, normalised
 *  against this capture's own quiet level, combined by fixed weights.
 *
 *  Not a presence detector — no hold, no breathing, no second chance. It is
 *  the scalar meant to be handed to a back-end classifier, drawn so the
 *  windows it gets right and the windows it misses are both visible. */
export function MotionSignal({ path, meta, timeLink, mimo, sourceMac, interpolate, dark }: MotionSignalProps) {
  const [windowS, setWindowS] = useState(2);
  const [hopS, setHopS] = useState(0.5);
  const [highpassHz, setHighpassHz] = useState(0.3);
  const [marginS, setMarginS] = useState(5);
  const [featureMode, setFeatureMode] = useState<MotionSignalMode>("label");
  const [data, setData] = useState<MotionSignalData | null>(null);
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
    fetchMotionSignal(
      path, range[0], range[1],
      { windowS, hopS, highpassHz, marginS, mimo, sourceMac, interpolate },
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
  }, [path, range, windowS, hopS, highpassHz, marginS, mimo, sourceMac, interpolate]);

  const domain = useMemo<[number, number]>(() => {
    if (!data || data.timeS.length === 0) return range;
    const half = data.windowSeconds / 2;
    return [data.timeS[0] - half, data.timeS[data.timeS.length - 1] + half];
  }, [data, range]);

  const innerWidth = Math.max(1, width - CHART_MARGIN.left - CHART_MARGIN.right);
  const x = linearScale(domain, [0, innerWidth]);
  const clampX = (t: number) => Math.max(0, Math.min(innerWidth, x(t)));

  const truthRuns = useMemo(
    () => (data?.truth ? runs(data.truth.timeS, data.truth.present) : []),
    [data],
  );

  // The feature panels show one normalisation at a time; when there is no
  // camera, `label` has nothing to centre on, so follow what is available.
  const shown: MotionSignalMode = data && !data.truth ? "free" : featureMode;
  const feat = data?.modes[shown] ?? null;

  const varDomain = useMemo(() => extent(feat?.variance ?? []), [feat]);
  const lagDomain = useMemo(() => extent(feat?.lag1 ?? []), [feat]);

  const cam = dark ? "#4f7fd4" : "#3f6fc4";
  const vacant = dark ? "#2a2f37" : "#e6e9ee";
  const ink = dark ? "#e8eaed" : "#1b1f25";
  const half = (data?.windowSeconds ?? 2) / 2;

  return (
    <div className="space-y-4" ref={holder}>
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-3">
          <SectionLabel>Motion signal</SectionLabel>
          <NumberField id="ms-win" label="Window (s)" value={windowS} onChange={setWindowS} min={0.25} max={60} step={0.25}
            title="Feature window. The experiment fixed 2 s: long enough for a 0.3 Hz high-pass to have something to remove, short enough that a two-second fidget is not averaged into a five-second calm." />
          <NumberField id="ms-hop" label="Hop (s)" value={hopS} onChange={setHopS} min={0.05} max={30} step={0.05}
            title="Step between windows. At 0.5 s with a 2 s window, neighbouring windows share three quarters of their samples — they are not independent evidence." />
          <NumberField id="ms-hp" label="High-pass (Hz)" value={highpassHz} onChange={setHighpassHz} min={0.01} max={5} step={0.05}
            title="DC removal before the variance. Lag-1 is deliberately taken before this filter: a high-pass leaves neighbouring noise samples anticorrelated, and lag-1 would read that as structure." />
          <NumberField id="ms-margin" label="Label margin (s)" value={marginS} onChange={setMarginS} min={0} max={60} width="w-16"
            title="Empty camera frames within this many seconds of a transition are not scored" />
          <Button
            variant="outline"
            size="sm"
            className="h-7 px-2 text-[11px]"
            disabled={!data?.truth}
            title="Which normalisation the two feature panels below are drawn in. Both score panels are always shown."
            onClick={() => setFeatureMode((v) => (v === "label" ? "free" : "label"))}
          >
            features: {MODE_SHORT[shown]}
          </Button>
        </div>
      </div>

      {error ? (
        <div className="text-muted-foreground p-8 text-sm">No signal for this range — {error}</div>
      ) : !data || !feat ? (
        <div className="text-muted-foreground p-8 text-sm">
          {loading ? "Extracting this range…" : "Nothing to extract yet."}
        </div>
      ) : (
        <>
          <div>
            <svg width={width} height={STRIP + 6} role="img" aria-label="camera truth">
              <g transform={`translate(${CHART_MARGIN.left},2)`}>
                <rect x={0} width={innerWidth} y={0} height={STRIP} fill={vacant} />
                {truthRuns.map((r, i) => {
                  const a = clampX(r.t0); const b = clampX(r.t1);
                  return r.value && b > a ? (
                    <rect key={`t${i}`} x={a} width={Math.max(1, b - a)} y={0} height={STRIP} fill={cam}>
                      <title>occupied · {formatTime(r.t0, domain[1] - domain[0])} – {formatTime(r.t1, domain[1] - domain[0])}</title>
                    </rect>
                  ) : null;
                })}
                <text x={-6} y={STRIP - 4} textAnchor="end" fontSize={9} fill="currentColor">camera</text>
              </g>
            </svg>
            <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
              <span><span style={{ color: cam }}>■</span> person in frame</span>
              <span>
                RATIO · {data.streams} subcarriers · {data.fsHz.toFixed(1)} Hz · {data.timeS.length} windows of{" "}
                {data.nSamples} samples
              </span>
              {data.inCorpus && (
                <span className="text-amber-700 dark:text-amber-400">
                  this capture is in the 29 the weights were fitted on — these are training numbers
                </span>
              )}
              {!data.truth && (data.truthExcluded
                ? <span className="text-amber-700 dark:text-amber-400">not scored — excluded from evaluation: {data.truthExcluded}</span>
                : <span>no camera sidecar beside this capture</span>)}
            </div>
          </div>

          <Chart
            width={width}
            times={data.timeS}
            domain={domain}
            yDomain={varDomain}
            yLabel={`variance (${MODE_SHORT[shown]} z)`}
            dark={dark}
            height={110}
            series={[{ values: feat.variance, color: VAR_COLOR, width: 1.1, label: "variance" }]}
            guides={[{ value: 0, color: dark ? "#3a424d" : "#c9cfd8", label: "quiet" }]}
          />
          <Chart
            width={width}
            times={data.timeS}
            domain={domain}
            yDomain={lagDomain}
            yLabel={`lag-1 autocorrelation (${MODE_SHORT[shown]} z)`}
            dark={dark}
            height={110}
            series={[{ values: feat.lag1, color: LAG_COLOR, width: 1.1, label: "lag1" }]}
            guides={[{ value: 0, color: dark ? "#3a424d" : "#c9cfd8", label: "quiet" }]}
          />

          {(["label", "free"] as MotionSignalMode[]).map((mode) => {
            const m = data.modes[mode];
            const c = m.confusion;
            const dom = extent(m.score);
            return (
              <div key={mode} className="space-y-1">
                <Chart
                  width={width}
                  times={data.timeS}
                  domain={domain}
                  yDomain={[Math.min(dom[0], m.threshold - 0.3), Math.max(dom[1], m.threshold + 0.3)]}
                  yLabel={`score · ${MODE_LABEL[mode]}`}
                  dark={dark}
                  height={130}
                  series={[{ values: m.score, color: ink, width: 1.1, label: `score-${mode}` }]}
                  guides={[{ value: m.threshold, color: THRESHOLD_COLOR, label: "threshold" }]}
                  bands={bandsOf(data.timeS, m.present, half)}
                  bandColor={THRESHOLD_COLOR}
                />
                <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[11px] text-muted-foreground tabular-nums">
                  {c ? (
                    <>
                      <span>
                        recall <b className="text-foreground">{rate(c.recall)}</b> · specificity{" "}
                        <b className="text-foreground">{rate(c.specificity)}</b> · balanced{" "}
                        <b className="text-foreground">
                          {c.recall !== null && c.specificity !== null ? rate((c.recall + c.specificity) / 2) : "—"}
                        </b>{" "}
                        · accuracy {rate(c.accuracy)}
                      </span>
                      <span>
                        {c.tp}/{c.fp}/{c.fn}/{c.tn} tp/fp/fn/tn · always-present baseline {rate(c.baseRate)} ·{" "}
                        {c.excluded} windows not scored
                      </span>
                    </>
                  ) : (
                    <span>not scored — no camera labels for this range</span>
                  )}
                  <span>
                    weights var {m.coefficients.variance.toFixed(3)} · lag1 {m.coefficients.lag1.toFixed(3)} ·
                    threshold {m.threshold.toFixed(3)}
                  </span>
                  {m.note && <span className="text-amber-700 dark:text-amber-400">{m.note}</span>}
                </div>
              </div>
            );
          })}

          <p className="text-[11px] text-muted-foreground leading-relaxed">
            The source is <b>RATIO</b>, the magnitude of the complex ratio along the AP&apos;s transmit chains —
            the same grid the presence detector rides on, so a change of decode cannot read as motion.
            Each {data.windowSeconds} s window gives two numbers, reduced over {data.streams} subcarriers by the
            median: the <b>variance</b> of the {data.highpassHz} Hz high-passed window, and the{" "}
            <b>lag-1 autocorrelation</b> of the window <i>before</i> that filter. Both are then divided by this
            capture&apos;s own scale — nothing here is comparable across captures before that, because empty-room
            variance spans 36× between links and pooling captures measures the link rather than the room.
          </p>
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            The two score panels differ only in where that scale comes from. <b>Label-based</b> centres on the
            windows the camera calls empty: it is what every number in the experiment was measured with, and it
            cannot ship. <b>Label-free</b> uses this capture&apos;s own 10th and 40th percentiles, no labels — the
            pair has to sit low, because at 20th/80th the occupant of a capture 40% occupied is inside the upper
            percentile and scales their own signal away (walking recall fell from 100% to 25.8% that way).
            The weights are fixed constants fitted once over the 29 marked captures with the threshold at 90%
            specificity on that corpus; they are not re-fitted on what you are looking at. Note the weights
            themselves: lag-1 carries nearly all of the discrimination, and variance survives mostly as a tie-break.
          </p>
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            This is a signal, not a verdict. There is no hold, no breathing channel and no second chance — a
            seated person who stops moving between phone taps drops below the threshold within a window, which is
            why the seated-with-phone captures land near 60% recall while walking sits at 100%. The Hybrid tab is
            where that gap is covered; here it is left visible on purpose.
          </p>
        </>
      )}
    </div>
  );
}
