import { useEffect, useId, useRef, useState } from "react";

import { ChartBusy } from "@/components/ChartBusy";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fetchFrameDiff, type FrameDiff, type FrameDiffSignal } from "./api";
import {
  bandPath,
  clampWindow,
  formatTime,
  linePath,
  linearScale,
  ticks,
  zoomWindow,
} from "./series";

// Narrowest window the frame step's wheel may reach. At 42 Hz half a second is
// about 20 frame pairs, which is as far in as a per-frame trace has anything
// left to resolve -- past it the panel is drawing the same steps wider.
const MIN_STEP_SPAN_SECONDS = 0.5;

// How long the frame step waits before refetching a window the reader is still
// moving. Long enough that one drag is one or two requests rather than one per
// pointer move, short enough to feel like the panel is keeping up.
const STEP_FETCH_DEBOUNCE_MS = 180;

// The frame step's y axis, the same on every capture and every zoom. Scaling
// it to the data made two captures incomparable at a glance, which is the one
// thing this panel is for -- and the axis is dimensionless, so a fixed one
// means something. +-0.5 rather than the bounded +-1 because nothing measured
// comes near the bound: over six captures across three signals the worst
// clipping is 0.063% of steps on one of them, and zero on fifteen of eighteen.
const STEP_AXIS = 0.5;

interface ChartSeries {
  values: (number | null)[];
  color: string;
  width?: number;
  label: string;
  dashed?: boolean;
}

/** A filled region between two series. Drawn for a decimated per-frame signal,
 *  where the extremes of a column are the signal and its median is not. */
interface ChartBand {
  lo: (number | null)[];
  hi: (number | null)[];
  color: string;
  label: string;
  opacity?: number;
}

interface ChartProps {
  times: number[];
  domain: [number, number];
  yDomain: [number, number];
  series: ChartSeries[];
  bands?: ChartBand[];
  guides?: { value: number; color: string; label: string }[];
  height?: number;
  yLabel: string;
  dark: boolean;
  width: number;
  /** Given, the plot pans on drag and zooms on the wheel, reporting the window
   *  it wants. The caller owns the window, so a chart is never zoomed to
   *  something the caller cannot fetch. */
  onWindow?: (next: [number, number]) => void;
  /** Outer bound the window may not leave. Required with `onWindow`. */
  limit?: [number, number];
  /** Narrowest window the wheel may reach. Below a frame or two there is
   *  nothing left to resolve. */
  minSpan?: number;
}

// Top margin holds the axis label clear of the highest tick label; at 8 the
// two sit on the same line and overprint each other.
const MARGIN = { top: 20, right: 12, bottom: 18, left: 44 };

function ZoomChart({
  times,
  domain,
  yDomain,
  series,
  bands = [],
  guides = [],
  height = 120,
  yLabel,
  dark,
  width,
  onWindow,
  limit,
  minSpan = 0.5,
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

  // One clip per instance. Without it a series whose times reach past the
  // domain draws outside the plot box -- which is what the frame step does by
  // construction, since its steps cover the whole decoded range while the
  // window-centre series above it start half a window in.
  const clipId = `plot-${useId()}`;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef<{ clientX: number; from: [number, number] } | null>(null);
  const interactive = Boolean(onWindow && limit);

  // Wheel zoom needs preventDefault, and React's onWheel is passive, so the
  // listener is attached by hand. Anchored on the cursor: see zoomWindow.
  useEffect(() => {
    const node = svgRef.current;
    if (!node || !onWindow || !limit) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = node.getBoundingClientRect();
      const anchor = (event.clientX - rect.left - MARGIN.left) / inner.w;
      // Up/away zooms in. The exponent keeps a trackpad's many small deltas and
      // a mouse's few large ones on the same scale.
      const factor = Math.exp(event.deltaY * 0.002);
      onWindow(zoomWindow(domain, anchor, factor, limit, minSpan));
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, [onWindow, limit, domain, inner.w, minSpan]);

  const onPointerDown = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!interactive || event.button !== 0) return;
    drag.current = { clientX: event.clientX, from: [domain[0], domain[1]] };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const onPointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const held = drag.current;
    if (!held || !onWindow || !limit) return;
    // Measured from where the drag started, not from the last event, so the
    // window cannot accumulate drift over a long drag.
    const moved = ((event.clientX - held.clientX) / inner.w)
      * (held.from[1] - held.from[0]);
    onWindow(clampWindow(
      [held.from[0] - moved, held.from[1] - moved], limit, minSpan,
    ));
  };
  const endDrag = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!drag.current) return;
    drag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      role="img"
      aria-label={yLabel}
      style={interactive ? { cursor: "ew-resize", touchAction: "none" } : undefined}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      <defs>
        <clipPath id={clipId}>
          {/* A little headroom above and below so a 1.8 px stroke sitting on
              the top gridline is not shaved in half by its own clip. */}
          <rect x={0} y={-2} width={inner.w} height={inner.h + 4} />
        </clipPath>
      </defs>
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
        <g clipPath={`url(#${clipId})`}>
          {/* Under the guides and the lines: the band is context, and a guide
              hidden behind a fill is a threshold the reader cannot place. */}
          {bands.map((b) => (
            <path
              key={b.label}
              d={bandPath(times, b.lo, b.hi, x, y)}
              fill={b.color}
              fillOpacity={b.opacity ?? 0.25}
              stroke="none"
            />
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
        </g>
        <text x={-MARGIN.left + 2} y={-8} fontSize={9} fill={text}>
          {yLabel}
        </text>
      </g>
    </svg>
  );
}

export interface FrameStepProps {
  path: string;
  /** The range the host panel analyses: the window a zoom may not leave. */
  range: [number, number];
  /** What the host's own charts span, so the trace lines up with them until
   *  the reader zooms. */
  domain: [number, number];
  width: number;
  mimo: string;
  sourceMac: string;
  interpolate: boolean;
  dark: boolean;
}

/** The frame-to-frame step, (a - a')/(a + a'), on its own pan/zoom axis.
 *
 *  Its own request and its own panel: it is evidence rather than a vote, so it
 *  must not be able to slow the host's verdict down or fail it. */
export function FrameStep({
  path,
  range,
  domain,
  width,
  mimo,
  sourceMac,
  interpolate,
  dark,
}: FrameStepProps) {
  // Off by default: it was measured after the panel shipped, and a measurement
  // does not get to change what the panel showed. See docs/frame_step.md.
  const [gateGain, setGateGain] = useState(false);
  const [stepSignal, setStepSignal] = useState<FrameDiffSignal>("ratio_complex");

  // The frame step's own view. `null` follows the panel, so the trace sits on
  // the shared time axis with the charts above it until the reader zooms; a
  // zoom then REFETCHES over the narrower window, because decimation happens
  // server-side and stretching existing columns would magnify nothing. At full
  // zoom each column is one frame pair, which is the only way a per-frame
  // signal can actually be read.
  const [stepWindow, setStepWindow] = useState<[number, number] | null>(null);
  const [step, setStep] = useState<FrameDiff | null>(null);
  const [stepError, setStepError] = useState<string | null>(null);
  // Set the moment anything changes, not when the request finally goes: the
  // fetch is debounced, so between the change and the request there is a
  // stretch where the chart is stale and nothing would say so.
  const [stepBusy, setStepBusy] = useState(false);

  // A zoom belongs to the range it was taken in. Keeping it across a pan of the
  // shared axis would leave the panel showing a window the reader did not pick.
  useEffect(() => setStepWindow(null), [path, range]);

  useEffect(() => {
    setStepBusy(true);
    const [from, to] = stepWindow ?? range;
    const controller = new AbortController();
    // Debounced, because a drag sets the window on every pointer move and each
    // fetch decodes a range server-side. The chart follows the pointer the whole
    // time on the data it already has -- stretched, and clipped at the edges --
    // and the finer columns arrive once the pointer settles.
    const timer = setTimeout(() => {
      fetchFrameDiff(
        path,
        from,
        to,
        {
          signal: stepSignal,
          gateGain,
          // One column per pixel. Fewer would average away the single frame this
          // signal exists to show; more would be columns the panel cannot draw.
          maxPoints: Math.max(200, Math.min(4000, width)),
          mimo,
          sourceMac,
          interpolate,
        },
        controller.signal,
      )
        .then((result) => {
          setStep(result);
          setStepError(null);
        })
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          setStep(null);
          setStepError(err instanceof Error ? err.message : String(err));
        })
        .finally(() => {
          if (!controller.signal.aborted) setStepBusy(false);
        });
    }, STEP_FETCH_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [
    path, range, stepWindow, stepSignal, gateGain, width, mimo, sourceMac,
    interpolate,
  ]);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <Label className="text-[10px] text-muted-foreground uppercase tracking-wide">
          Frame step
        </Label>
        <Select
          value={stepSignal}
          onValueChange={(v) => setStepSignal(v as FrameDiffSignal)}
        >
          <SelectTrigger className="w-36 h-7 text-[11px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="amplitude">raw amplitude</SelectItem>
            <SelectItem value="ratio_amp">ratio |r|</SelectItem>
            <SelectItem value="ratio_complex">ratio complex</SelectItem>
          </SelectContent>
        </Select>
        <Button
          variant={gateGain ? "default" : "outline"}
          size="sm"
          className="h-7 px-2 text-[11px]"
          title="Blank the frame pairs that cross a reported receiver gain state, the way a pair spanning a dropout is blanked. Measured: costs 12-52% of the steps, halves the tail (p99 down ~50%), and changes neither the empty/occupied separation nor the 15.5x spread of the empty level across captures."
          onClick={() => setGateGain((v) => !v)}
        >
          gain gate {gateGain ? "on" : "off"}
        </Button>
        {stepWindow && (
          <Button
            variant="outline"
            size="sm"
            className="h-7 px-2 text-[11px]"
            title="Back to the panel's own range, on the shared time axis"
            onClick={() => setStepWindow(null)}
          >
            reset zoom
          </Button>
        )}
      </div>

      {stepError !== null ? (
        <p className="text-[11px] text-red-500">frame step: {stepError}</p>
      ) : step === null ? (
        <ChartBusy busy empty height={150} label="decoding">
          <div />
        </ChartBusy>
      ) : (
        <>
          <ChartBusy busy={stepBusy} height={150} label="decoding">
            <ZoomChart
              width={width}
              times={step.timeS}
              // The shared axis until the reader zooms, then its own. Drag
              // pans, the wheel zooms about the cursor, and both are
              // bounded by the panel's range because that is what the
              // detector above was computed over.
              domain={stepWindow ?? domain}
              limit={range}
              onWindow={setStepWindow}
              minSpan={MIN_STEP_SPAN_SECONDS}
              yDomain={[-STEP_AXIS, STEP_AXIS]}
              yLabel="frame step (a−a′)/(a+a′)"
              dark={dark}
              height={150}
              bands={[
                {
                  lo: step.signedLo,
                  hi: step.signedHi,
                  color: "#5a8f7b",
                  label: "signed envelope",
                  opacity: dark ? 0.32 : 0.22,
                },
              ]}
              series={[
                { values: step.signed, color: "#2f7c5c", width: 1.4, label: "signed median" },
                // The median of |d| is the quantity the Hybrid tab's
                // amplitude channel reports, exactly. The column peak is
                // drawn beside it only when columns hold more than one
                // frame pair -- otherwise it is the same line twice.
                { values: step.magnitude, color: "#a34a8f", width: 1.4, label: "|step| median" },
                ...(step.decimated
                  ? [{
                      values: step.magnitudeHi,
                      color: "#a34a8f",
                      width: 1,
                      label: "|step| peak",
                      dashed: true,
                    }]
                  : []),
              ]}
            />
          </ChartBusy>
          <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
            <span style={{ color: "#2f7c5c" }}>signed median</span>
            <span style={{ color: "#5a8f7b" }}>
              {step.decimated
                ? `envelope of ${step.binSeconds.toFixed(2)} s columns`
                : "one column per frame pair"}
            </span>
            <span style={{ color: "#a34a8f" }}>
              |step| median{step.decimated && " (dashed: column peak)"}
            </span>
            <span>
              axis ±{STEP_AXIS} — fixed, so captures compare
              {step.summary.max !== null && (
                <>
                  {" "}· peak {(step.summary.maxNative ?? 0).toFixed(1)}
                  {step.summary.nativeUnit}
                </>
              )}
            </span>
            <span>
              {stepWindow ? (
                <>
                  zoomed to {stepWindow[0].toFixed(1)}–
                  {stepWindow[1].toFixed(1)} s (
                  {((range[1] - range[0]) / (stepWindow[1] - stepWindow[0])).toFixed(0)}×,
                  off the shared axis) · drag to pan, wheel to zoom
                </>
              ) : (
                "drag to pan · wheel to zoom"
              )}
            </span>
          </div>
          <p className="text-[11px] text-muted-foreground leading-relaxed tabular-nums">
            {/* The signal the SERVER used, not the one the control
                says: the two silently disagreed once, and a panel that
                names what it drew cannot do that again. */}
            <b>{step.signal.replace("_", " ")}</b> ·{" "}
            {step.summary.stepsMeasured} of {step.summary.steps} steps
            measured over {step.nSubcarriers} subcarriers ·{" "}
            <b>{step.selectionNote}</b>
            {step.framesDropped > 0 && (
              <> · {step.framesDropped} frames of another shape dropped</>
            )}
            {step.summary.nBridged > 0 && (
              <>
                {" "}· {step.summary.nBridged} dropped for spanning a gap
                wider than {step.summary.gapLimit.toFixed(2)} s
              </>
            )}
            {" "}· {step.summary.nGainCrossed} cross a gain state
            {step.summary.gainGated ? " (blanked)" : " (kept)"}
            {step.summary.median !== null && (
              <>
                {" "}· median {step.summary.median.toFixed(4)} (
                {(step.summary.medianNative ?? 0).toFixed(2)}
                {step.summary.nativeUnit}), peak{" "}
                {(step.summary.max ?? 0).toFixed(3)} (
                {(step.summary.maxNative ?? 0).toFixed(1)}
                {step.summary.nativeUnit})
              </>
            )}
          </p>
          <dl className="text-[11px] text-muted-foreground leading-relaxed grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
            <dt className="font-medium text-foreground">Fold</dt>
            <dd>
              {step.signal === "ratio_complex" ? (
                <>
                  One step per subcarrier per frame pair on the{" "}
                  <b>complex ratio</b>, then the <b>median</b> across the{" "}
                  {step.summary.liveMedian} live ones (of {step.nSubcarriers}).
                  A complex difference has a direction, not a sign, so the
                  traces are its decomposition:{" "}
                  <span style={{ color: "#2f7c5c" }}>signed</span> is the
                  radial part <b>(|r|−|r′|)/(|r|+|r′|)</b>, exactly what the
                  ratio |r| channel shows, and{" "}
                  <span style={{ color: "#a34a8f" }}>|step|</span> is the total{" "}
                  <b>|r−r′|/(|r|+|r′|)</b>. The total is never below the
                  radial, so <b>the gap between them is the phase
                  rotation</b> — a body at fixed range walks the ratio round a
                  circle at constant magnitude, which the radial part alone
                  cannot see. At constant magnitude the total reads back as an
                  angle, {step.summary.nativeUnit} above.
                </>
              ) : (
                <>
                  One <b>d</b> per subcarrier per frame pair,{" "}
                  <b>(a<sub>t</sub> − a<sub>t−1</sub>)/(a<sub>t</sub> + a<sub>t−1</sub>)</b>{" "}
                  on {step.signal === "ratio_amp" ? "the ratio's" : "the raw"}{" "}
                  amplitude, then the <b>median</b> across the{" "}
                  {step.summary.liveMedian} live ones (of {step.nSubcarriers}) —
                  twice. <span style={{ color: "#2f7c5c" }}>Signed</span> keeps
                  the direction and so can cancel: a body brightens some
                  subcarriers and fades others, which reads near zero with a
                  wide envelope. <span style={{ color: "#a34a8f" }}>|step|</span>{" "}
                  folds the magnitudes and cannot.
                </>
              )}{" "}
              Median, not mean — two subcarriers moving 40 dB leave it at zero,
              which is the point. The <b>mean</b> (common-mode) fold is still
              computed and still in the payload, but is no longer drawn: over
              twelve empty captures its floor spans 48–73× between captures
              against 10× for the median fold, so two captures cannot be read
              against one another on it. Bins that are dead or in the guard band are
              dropped first — that is the {step.nSubcarriers} −{" "}
              {step.summary.liveMedian} missing above — and a pair with fewer
              than 4 live ones reports nothing rather than a median of two.
            </dd>

            <dt className="font-medium text-foreground">AGC correction</dt>
            <dd>
              <b>Off</b> — matching the board and the Hybrid tab.{" "}
              {step.signal !== "amplitude" && (
                <>
                  On this signal a gain step mostly cannot reach it — the
                  common gain divides out of the ratio — and measured, a
                  crossing reads 1.02× the same-state level on empty and
                  still captures against 4.4–5.4× on the raw amplitude. Not
                  always: on a walking capture the ratio still reads 3.4× at a
                  crossing, because there the gain is changing <i>because</i>{" "}
                  the person is moving.{" "}
                </>
              )}
              Measured over 21 captures on the raw amplitude, applying the per-gain-state
              correction leaves the median within 2 % and raises the 99th
              percentile by <b>12–125 %</b>: it is a per-frame, shape-only
              correction, so differencing two differently-corrected frames
              adds a step that was not there. The crossings are counted
              instead —{" "}
              {step.summary.nGainCrossed} of {step.summary.steps} pairs here (
              {((step.summary.nGainCrossed / Math.max(1, step.summary.steps)) * 100).toFixed(0)}
              %) — and <b>gain gate</b> blanks them
              {step.summary.gainGated ? ", which is on" : ", currently off"}.
            </dd>
          </dl>
        </>
      )}
    </div>
  );
}
