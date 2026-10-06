import { useEffect, useMemo, useRef, useState } from "react";

import { ChartBusy } from "@/components/ChartBusy";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  fetchClassifier,
  type Classifier as ClassifierData,
  type ClassifierFeature,
  type ClassifierUnit,
  type Meta,
} from "./api";
import { Chart, CHART_MARGIN } from "./Chart";
import { formatTime, linearScale, runs } from "./series";
import type { TimeLink } from "./timelink";

const DEFAULT_SPAN_SECONDS = 600;
const STRIP = 16;

// Both axes fixed, as on the Hybrid 2 tab: dimensionless quantities, so one
// scale means two captures can be read against each other. The step's is the
// Hybrid 2 motion axis; ψ̂ is bounded by ±1 and sits at −1/T when nothing moves.
const STEP_AXIS: [number, number] = [0, 0.3];
const ACF_AXIS: [number, number] = [-0.2, 1];

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

/** A feature's value against its reference, for the unit strip and the table:
 *  over, under, or no reference to compare with. */
function standing(f: ClassifierFeature, v: number | null): "over" | "under" | "none" {
  if (v == null || f.reference == null) return "none";
  return v > f.reference ? "over" : "under";
}

function fmt(f: ClassifierFeature, v: number | null): string {
  if (v == null) return "—";
  return v.toFixed(f.decimals);
}

export interface ClassifierProps {
  path: string;
  meta: Meta;
  timeLink?: TimeLink;
  mimo: string;
  sourceMac: string;
  interpolate: boolean;
  dark: boolean;
}

export function Classifier({
  path, meta, timeLink, mimo, sourceMac, interpolate, dark,
}: ClassifierProps) {
  const [unitS, setUnitS] = useState(60);
  const [acfFrames, setAcfFrames] = useState(84);
  const [lagS, setLagS] = useState(2);

  const [data, setData] = useState<ClassifierData | null>(null);
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
    fetchClassifier(
      path, range[0], range[1],
      { unitS, acfFrames, lagS, mimo, sourceMac, interpolate },
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
  }, [path, range, unitS, acfFrames, lagS, mimo, sourceMac, interpolate]);

  const domain = useMemo<[number, number]>(() => {
    if (!data || data.units.length === 0) return range;
    return [data.units[0].t0, data.units[data.units.length - 1].t1];
  }, [data, range]);

  const truthRuns = useMemo(
    () => (data?.truth ? runs(data.truth.timeS, data.truth.present) : []),
    [data],
  );

  // The features that carry a reference are the ones a strip can be drawn
  // for; the rest are read in the table.
  const guided = useMemo(
    () => (data ? data.features.filter((f) => f.reference != null) : []),
    [data],
  );

  const innerWidth = Math.max(1, width - CHART_MARGIN.left - CHART_MARGIN.right);
  const x = linearScale(domain, [0, innerWidth]);
  const clampX = (t: number) => Math.min(innerWidth, Math.max(0, x(t)));
  const vacant = dark ? "#2a2f37" : "#e6e9ee";
  const cam = dark ? "#4f7fd4" : "#3f6fc4";
  const over = "#d97a2f";
  const under = dark ? "#2b3038" : "#e6e9ee";
  const stripHeight = STRIP * (1 + guided.length) + 4 * guided.length + 4;
  const span = domain[1] - domain[0];
  const occupancyLabel = (u: ClassifierUnit) =>
    u.cameraOccupancy == null ? "no camera" : `camera ${(u.cameraOccupancy * 100).toFixed(0)} % occupied`;

  return (
    <div className="space-y-4" ref={holder}>
      <div className="flex flex-wrap items-center gap-3">
        <NumberField
          id="cl-unit" label="Unit (s)" value={unitS} onChange={setUnitS}
          min={5} max={600} step={5}
          title="The range is cut into equal units nearest this length; one call per unit is the application's verdict. A one-minute capture is one unit."
        />
        <NumberField
          id="cl-acf" label="ACF window (frames)" value={acfFrames} onChange={setAcfFrames}
          min={8} max={4000} step={4} width="w-24"
          title="WiDetect's T. 84 frames is ~2 s at 42 Hz. Fixed in frames rather than seconds because the no-motion distribution of ψ̂ is a function of the sample count: mean −1/T."
        />
        <NumberField
          id="cl-lag" label="Step lag (s)" value={lagS} onChange={setLagS}
          min={0} max={10} step={0.5}
          title="Lag of the frame step. 2 s is what the range rule reads, so the step P90 here is the rule's P90."
        />
        {data && (
          <span className="text-[11px] text-muted-foreground">
            {data.units.length} unit{data.units.length === 1 ? "" : "s"} of {data.unitSeconds.toFixed(1)} s ·{" "}
            {data.fsHz?.toFixed(1) ?? "—"} Hz · {data.nSubcarriers} subcarriers · ψ̂ window {data.acf.windowSeconds?.toFixed(2) ?? "—"} s ·{" "}
            {data.selectionNote}
          </span>
        )}
      </div>

      {error ? (
        <div className="text-muted-foreground p-8 text-sm">
          No features for this range — {error}
        </div>
      ) : !data ? (
        <ChartBusy busy empty height={150} label="computing the feature bank" />
      ) : (
        <>
          <ChartBusy busy={busy} height={stripHeight} label="computing">
            <svg width={width} height={stripHeight} role="img" aria-label="camera and per-unit feature strips">
              <g transform={`translate(${CHART_MARGIN.left},2)`}>
                <rect x={0} width={innerWidth} y={0} height={STRIP} fill={vacant} />
                {truthRuns.map((r, i) => {
                  const a = clampX(r.t0); const b = clampX(r.t1);
                  return r.value && b > a ? (
                    <rect key={`t${i}`} x={a} width={Math.max(1, b - a)} y={0} height={STRIP} fill={cam} />
                  ) : null;
                })}
                <text x={-6} y={STRIP - 4} textAnchor="end" fontSize={9} fill="currentColor">camera</text>
                {guided.map((f, row) => {
                  const y = (STRIP + 4) * (row + 1);
                  return (
                    <g key={f.key}>
                      {data.units.map((u, i) => {
                        const a = clampX(u.t0); const b = clampX(u.t1);
                        const s = standing(f, u.values[f.key]);
                        return (
                          <rect
                            key={i} x={a} width={Math.max(1, b - a - 1)} y={y} height={STRIP}
                            fill={s === "over" ? over : under}
                            stroke={dark ? "#1b1f25" : "#ffffff"} strokeWidth={1}
                          >
                            <title>
                              {f.label} {fmt(f, u.values[f.key])} {s === "over" ? ">" : "≤"} {f.reference}
                              {" · "}{formatTime(u.t0, span)} – {formatTime(u.t1, span)} · {occupancyLabel(u)}
                            </title>
                          </rect>
                        );
                      })}
                      <text x={-6} y={y + STRIP - 4} textAnchor="end" fontSize={9} fill="currentColor">
                        {f.label.replace(" · ", " ")}
                      </text>
                    </g>
                  );
                })}
              </g>
            </svg>
          </ChartBusy>
          <p className="text-[11px] text-muted-foreground">
            One cell per unit, filled where the feature is over its reference. The references are
            measured operating points (docs/one_minute_classifier.md), not the verdict — the
            verdict rule is what the remaining tests decide.
          </p>

          <ChartBusy busy={busy} height={150} label="computing">
            <Chart
              width={width} times={data.acf.timeS} domain={domain}
              yDomain={ACF_AXIS} height={150} dark={dark}
              yLabel={`ψ̂ · lag-1 autocorrelation, ${data.acf.windowFrames}-frame windows`}
              series={[
                { values: data.acf.amp, color: "#7a4fd4", width: 1.5, label: "ratio amplitude" },
                { values: data.acf.phase, color: "#c08a2e", width: 1.2, label: "ratio phase", dashed: true },
              ]}
              guides={[
                ...guided.filter((f) => f.key === "acf_amp_p90").map((f) => ({
                  value: f.reference as number, color: "#d62728", label: "reference (P90w)",
                })),
                { value: data.acf.nullMean, color: "#9aa5b1", label: "null −1/T" },
              ]}
            />
          </ChartBusy>

          <ChartBusy busy={busy} height={150} label="computing">
            <Chart
              width={width} times={data.step.timeS} domain={domain}
              yDomain={STEP_AXIS} height={150} dark={dark}
              yLabel={`motion · frame step, ${data.step.lagSeconds} s lag`}
              series={[{ values: data.step.level, color: "#2f7c5c", width: 1.6, label: "per-second median" }]}
              guides={guided.filter((f) => f.key === "step_p90").map((f) => ({
                value: f.reference as number, color: "#d62728", label: "range rule (P90)",
              }))}
            />
          </ChartBusy>

          <div className="overflow-x-auto">
            <table className="w-full text-[11px] tabular-nums">
              <thead>
                <tr className="text-muted-foreground">
                  <th className="px-2 py-1 text-left font-medium">unit</th>
                  <th className="px-2 py-1 text-right font-medium" title="Fraction of camera frames in the unit with a person">camera</th>
                  {data.features.map((f) => (
                    <th
                      key={f.key}
                      className="px-2 py-1 text-right font-medium whitespace-nowrap"
                      title={`${f.description}${f.reference != null ? ` Reference ${f.reference}.` : ""} (${f.status}, test ${f.test})`}
                    >
                      {f.label}
                      {f.reference != null && <span className="text-muted-foreground/70"> › {f.reference}</span>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.units.map((u, i) => (
                  <tr key={i} className="border-t border-border/50">
                    <td className="px-2 py-1 whitespace-nowrap">
                      {formatTime(u.t0, span)} – {formatTime(u.t1, span)}
                      <span className="text-muted-foreground"> · {u.nWindows} w</span>
                    </td>
                    <td className="px-2 py-1 text-right">
                      {u.cameraOccupancy == null ? "—" : `${(u.cameraOccupancy * 100).toFixed(0)} %`}
                    </td>
                    {data.features.map((f) => {
                      const s = standing(f, u.values[f.key]);
                      return (
                        <td
                          key={f.key}
                          className={`px-2 py-1 text-right ${s === "over" ? "font-semibold text-[#d97a2f]" : ""}`}
                        >
                          {fmt(f, u.values[f.key])}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {data.truthExcluded && (
            <p className="text-[11px] text-red-500">
              This capture is flagged {data.truthExcluded} — its camera column is shown but is not truth.
            </p>
          )}

          <p className="text-[11px] text-muted-foreground leading-relaxed">
            <b>The feature bank for the one-minute verdict.</b> Every column is a scalar per unit;
            the bank grows as the test programme produces features that transfer. <b>Step P90</b> is
            the range rule&apos;s motion input — sharp on its own link (AUC 0.996, still sitter
            against empty) and at chance across the 09-16/17 link-state shift, because its empty
            floor moves 10× with the link. <b>ψ̂</b> (test 1, WiDetect) is the lag-1 autocorrelation
            of the detrended ratio level: receiver noise, clean or ten times noisier, is white at
            the frame rate and lands it at −1/T either way, so its empty floor transfers where the
            step&apos;s does not (−0.009 on 09-30, +0.022 / +0.008 on 09-16/17). What moves it is
            slow coherent variation in 0.1–2 Hz — a sitter&apos;s micro-motion, a robot, or
            whatever moved near the link on the camera-empty mornings of 09-30 — and the ratio&apos;s
            subcarriers move together (F_eff ≈ 2), so the null is ten times wider than the paper&apos;s
            formula and the reference is an empirical operating point: 90 % specificity at home,
            96 % on the noisy link, 56 % recall of still sitters. A feature beside the step, not a
            replacement for it.
          </p>
        </>
      )}
    </div>
  );
}
