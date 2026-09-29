import { useEffect, useId, useMemo, useRef, useState } from "react";

import { ChartBusy, Spinner } from "@/components/ChartBusy";
import { Button } from "@/components/ui/button";
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
  fetchFrameDiff,
  fetchLabels,
  fetchPresence,
  type FrameDiff,
  type FrameDiffSignal,
  type Labels,
  type Meta,
  type Presence as PresenceData,
  type PresenceChannel,
  type PresenceState,
} from "./api";
import {
  bandPath,
  clampWindow,
  formatTime,
  linePath,
  linearScale,
  runs,
  ticks,
  zoomWindow,
} from "./series";
import type { TimeLink } from "./timelink";

// Longest stretch analysed before the user asks for more. Presence needs
// full-rate samples -- an autocorrelation cannot be fed stride-sampled frames
// without destroying the very periodicity it is looking for -- so the whole
// range in view is decoded, and a four-hour capture opened whole would decode
// every frame it holds. Ten minutes is long enough to hold dozens of windows
// and short enough to come back promptly.
const DEFAULT_SPAN_SECONDS = 600;

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

function Chart({
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
  const [windowSeconds, setWindowSeconds] = useState(30);
  // The empty-room reference. Named by the operator rather than derived,
  // because a reference taken from recent history absorbs an occupant who
  // sits still and then reports the room as empty precisely while it is not.
  const [reference, setReference] = useState<[number, number][] | null>(null);
  // ...except when the capture states its own. A labelled run records the
  // stretch it was empty for, and requiring the operator to re-declare that by
  // hand leaves the offset trace blank on exactly the captures built to have
  // one. Seeded once per capture, and only while the operator has not chosen:
  // any manual pick outranks the protocol, because the protocol is what was
  // intended and the operator may have seen that it did not happen.
  // Fetched once per capture and kept, so the reference can be re-derived on
  // demand rather than only at load. Depends on `path` alone: listing
  // `reference` here would abort this request every time the operator picked a
  // range, and the current pick is read through a ref instead.
  const [labels, setLabels] = useState<Labels | null>(null);
  const referenceRef = useRef<[number, number][] | null>(null);
  referenceRef.current = reference;
  useEffect(() => {
    const controller = new AbortController();
    setLabels(null);
    fetchLabels(path, controller.signal)
      .then(setLabels)
      .catch(() => setLabels(null));
    return () => controller.abort();
  }, [path]);

  // The stretch this capture is known to have been empty for.
  //
  // Taken from the CAMERA where there is one, not from the protocol: the
  // protocol says what was intended and the detections say what happened, and
  // on the labelled runs so far the occupant arrived 2-12 s after the phase
  // said they would. Ending the reference at the first detection rather than
  // at the nominal boundary keeps those seconds out of it.
  //
  // The LEADING empty stretch only, never a trailing one. On both labelled
  // runs the channel did not return to its starting state once the occupant
  // left -- 0.55 dB to 2.92 on one, 0.40 to 1.60 on the other -- so the room
  // at the end is not the room at the start whatever the protocol called it,
  // and averaging the two would calibrate against a room that no longer
  // exists. MARGIN backs off a little further, because the approach to the
  // chair is motion that the camera sees late.
  // EVERY stretch the camera saw nobody in, not just the first.
  //
  // A room does not return to its starting state once an occupant has been in
  // it: measured on both labelled runs, a reference built from the leading
  // stretch alone calls 100% of the trailing empty windows occupied. Pooling
  // the stretches fixes that outright -- 0% false, with the occupant still
  // found in 98% and 100% of windows -- because the verdict then measures
  // distance to the NEAREST empty state rather than to one arbitrary state or
  // to an average matching neither.
  //
  // MIN_RUN: a lone detection is flicker, not an arrival, and a lone miss is
  // flicker, not a departure -- 20260904_193228 carries exactly one dropped
  // frame mid-sit, 0.81 confidence before it and 0.89 after.
  //
  // MARGIN is a grace period either side of every transition, and it exists
  // because the camera bounds the OCCUPANT, not the MOTION. A person walks to
  // the chair before the detector first boxes them and away from it after the
  // last box, and on the approach they are also nearer the transmitter than
  // they will be once seated -- so the seconds adjacent to a transition carry
  // the largest channel disturbance in the run while the camera still reads
  // nobody. Folding those into a reference is precisely backwards: it
  // calibrates "empty" against the loudest motion present. Five seconds is
  // what the walk to and from the chair costs at this room's geometry; the
  // detections themselves cannot recover it, since the evidence is missing
  // from the frames rather than merely uncertain in them.
  const emptyRefs = useMemo<[number, number][]>(() => {
    const MARGIN = 5;
    const MIN_RUN = 3;
    const present = labels?.present;
    if (present && present.timeS.length) {
      const flags = present.present;
      const held = flags.map((_, i) =>
        i + MIN_RUN <= flags.length
          ? flags.slice(i, i + MIN_RUN).every(Boolean)
          : flags[i],
      );
      const out: [number, number][] = [];
      let start: number | null = present.timeS[0];
      for (let i = 0; i < held.length; i++) {
        if (held[i] && start !== null) {
          const end = present.timeS[i] - MARGIN;
          if (end > start) out.push([Math.max(0, start), end]);
          start = null;
        } else if (!flags[i] && start === null) {
          // Symmetric: only reopen once absence has held, so one dropped
          // frame mid-sit does not split the occupied stretch in two.
          const quiet = flags.slice(i, i + MIN_RUN);
          if (quiet.length === MIN_RUN && quiet.every((v) => !v)) {
            start = present.timeS[i] + MARGIN;
          }
        }
      }
      if (start !== null) {
        const end = present.timeS[present.timeS.length - 1];
        if (end > start) out.push([Math.max(0, start), end]);
      }
      if (out.length) return out;
    }
    return (labels?.phases ?? [])
      .filter((ph) => ph.label === "empty" && ph.t1 > ph.t0)
      .map((ph) => [ph.t0, ph.t1] as [number, number]);
  }, [labels]);

  // Seed it once, so a labelled capture opens with the offset trace already
  // populated instead of the "no reference" placeholder. A manual pick
  // outranks this: the protocol is intent, and the operator may have watched
  // it not happen.
  useEffect(() => {
    if (emptyRefs.length && !referenceRef.current) setReference(emptyRefs);
  }, [emptyRefs]);
  const [threshold, setThreshold] = useState(0.25);
  const [motionFracHi, setMotionFracHi] = useState(0.25);
  // The frame-to-frame amplitude step, fetched separately from the detector.
  // Its own request because it is a different decode (raw amplitude, not the
  // ratio grid) and because it is evidence rather than a vote: nothing in the
  // verdict above depends on it, so it must not be able to slow the verdict
  // down or fail it.
  const [showStep, setShowStep] = useState(true);
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
        refT0: reference?.map(([a]) => a) ?? null,
        refT1: reference?.map(([, b]) => b) ?? null,
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
    path, range, channel, windowSeconds, threshold, motionFracHi, reference,
    mimo, sourceMac, interpolate,
  ]);

  // A zoom belongs to the range it was taken in. Keeping it across a pan of the
  // shared axis would leave the panel showing a window the reader did not pick.
  useEffect(() => setStepWindow(null), [path, range]);

  useEffect(() => {
    if (!showStep) {
      setStep(null);
      setStepError(null);
      setStepBusy(false);
      return;
    }
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
    path, range, stepWindow, showStep, stepSignal, gateGain, width, mimo,
    sourceMac, interpolate,
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

  const devCeiling = useMemo(() => {
    const finite = (data?.baselineDev ?? []).filter(
      (v): v is number => v !== null && Number.isFinite(v),
    );
    const threshold = data?.baselineDevThreshold ?? 0;
    return Math.max(threshold * 1.6, ...finite.map((v) => v * 1.2), 0.5);
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

        <div className="flex items-center gap-2">
          <Button
            variant={showStep ? "default" : "outline"}
            size="sm"
            className="h-7 px-2 text-[11px]"
            title="The frame-to-frame amplitude step, as (a_t - a_t-1)/(a_t + a_t-1) — the raw dB difference on a bounded -1..1 axis. Evidence only; it does not vote on the verdict above."
            onClick={() => setShowStep((v) => !v)}
          >
            frame step {showStep ? "on" : "off"}
          </Button>
          {showStep && (
            <>
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
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          <Label className="text-[10px] text-muted-foreground uppercase tracking-wide">
            Empty reference
          </Label>
          {reference?.length ? (
            <>
              <span className="text-[11px] tabular-nums">
                {reference
                  .map(([a, b]) => `${a.toFixed(1)}–${b.toFixed(1)}`)
                  .join(", ")}{" "}
                s
              </span>
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2 text-[11px]"
                onClick={() => setReference(null)}
              >
                Clear
              </Button>
            </>
          ) : (
            <span className="text-[11px] text-muted-foreground">none</span>
          )}
          <Button
            variant="outline"
            size="sm"
            className="h-7 px-2 text-[11px]"
            onClick={() => setReference([[range[0], range[1]]])}
          >
            Use this view
          </Button>
          {emptyRefs.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2 text-[11px]"
              title={`Set the reference to every stretch the camera saw nobody in: ${emptyRefs
                .map(([a, b]) => `${a.toFixed(1)}-${b.toFixed(1)}`)
                .join(", ")} s`}
              onClick={() => setReference(emptyRefs)}
            >
              Use empty label{emptyRefs.length > 1 ? `s (${emptyRefs.length})` : ""}
            </Button>
          )}
        </div>

        {data && (
          <span className="text-[11px] text-muted-foreground">
            {data.fsHz.toFixed(1)} Hz · {data.timeS.length} windows ·{" "}
            {data.windowSeconds.toFixed(1)} s each · floor{" "}
            {data.rpmFloorEff.toFixed(0)} rpm
            {loading && (
              <span className="ml-1 inline-flex items-center gap-1 align-middle">
                <Spinner /> refreshing
              </span>
            )}
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
        <div className="text-muted-foreground text-sm">
          {loading ? (
            <ChartBusy busy empty height={150} label="analysing this range" />
          ) : (
            <div className="p-8">Nothing to analyse yet.</div>
          )}
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

          {data.baselineDevThreshold === null ? (
            <div className="rounded-md border border-dashed px-3 py-2 text-[11px] text-muted-foreground leading-relaxed">
              <b>No empty-room reference.</b> Absence is a claim, and there is
              nothing here to measure it against — so every window that is not
              moving reads <b>no data</b> rather than “empty”, and a motionless
              occupant cannot be seen at all. Park the view on a stretch you
              know was empty and press <b>Use this view</b>.
            </div>
          ) : (
            <ChartBusy busy={loading} height={150} label="analysing">
              <Chart
                width={width}
                times={data.timeS}
                domain={domain}
                yDomain={[0, devCeiling]}
                yLabel="channel offset (dB)"
                dark={dark}
                height={150}
                series={[
                  { values: data.baselineDev, color: "#7b5ea7", width: 1.8, label: "offset from empty" },
                ]}
                guides={[
                  {
                    value: data.baselineDevThreshold,
                    color: "#d62728",
                    label: "occupied",
                  },
                ]}
              />
            </ChartBusy>
          )}

          {data.reference && (
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              Reference: {data.reference.nWindows} windows, wandering{" "}
              {data.reference.devScale.toFixed(2)} dB on their own (median×2;
              p95 {data.reference.devP95.toFixed(2)}) — so
              “occupied” is {data.params.baseline_dev_k}× that, or{" "}
              {(data.baselineDevThreshold ?? 0).toFixed(2)} dB. Motion floor{" "}
              {data.reference.motionFloor.toFixed(3)}, and gross motion is{" "}
              {data.params.motion_ratio_hi}× it.
            </p>
          )}

          <ChartBusy busy={loading} height={120} label="analysing">
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
          </ChartBusy>

          {showStep && step === null && stepError === null && (
            <ChartBusy busy empty height={150} label="decoding">
              <div />
            </ChartBusy>
          )}

          {showStep && (stepError !== null || step !== null) && (
            <div className="space-y-1">
              {stepError !== null ? (
                <p className="text-[11px] text-red-500">
                  frame step: {stepError}
                </p>
              ) : step !== null && (
                <>
                  <ChartBusy busy={stepBusy} height={150} label="decoding">
                    <Chart
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
                </>
              )}
            </div>
          )}

          {showStep && step !== null && (
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
          )}

          <ChartBusy busy={loading} height={150} label="analysing">
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
          </ChartBusy>

          <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
            <span style={{ color: "#2f6fed" }}>score</span>
            <span style={{ color: "#9aa5b1" }}>periodicity</span>
            <span style={{ color: "#c7a02f" }}>tonality</span>
            <span style={{ color: "#2f9c6f" }}>motion gate</span>
            <span>= score is their product · evidence only, does not decide</span>
          </div>

          {!data.breathing.some(Boolean) && (
            <p className="text-[11px] text-muted-foreground">
              No window in this range carried a believable chest, so the rate
              axis below is empty by construction — not a chart that failed to
              draw.
            </p>
          )}

          <ChartBusy busy={loading} height={110} label="analysing">
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
                    data.breathing[i] ? v : null,
                  ),
                  color: "#2f6fed",
                  width: 1.6,
                  label: "rate",
                },
              ]}
            />
          </ChartBusy>

          {data.warnings.length > 0 && (
            <ul className="text-[11px] text-muted-foreground leading-relaxed list-disc pl-4">
              {data.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}

          <p className="text-[11px] text-muted-foreground leading-relaxed">
            Built on the <b>raw</b> CSI ratio, with no swap correction: what
            the panel reads is what the NIC delivered. That is not free — 1.2%
            of frame-to-frame steps exceed π uncorrected, and a π step is a
            broadband impulse with energy inside the respiration band, so it
            can read as motion or as breathing in an empty room. Check the
            ratio panels' corrected views before trusting a verdict on a
            capture full of swaps. A still occupant is decided on the{" "}
            <b>channel offset</b> from a known-empty reference, because a
            motionless body does not modulate the channel, it displaces it —
            and every other series here is mean-removed, so the displacement is
            the one thing they cannot see. The breathing score is reported
            alongside as evidence and does not vote: measured on a walk-in /
            sit-still / walk-out capture, periodicity ran <i>higher</i> in the
            empty room (0.102) than with an occupant (0.076). Hatched stretches
            are windows more than half interpolated across a capture dropout;
            they report <b>no data</b> rather than “empty”, because a bridged
            hole is flat and flat scores exactly like an absent occupant.{" "}
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
