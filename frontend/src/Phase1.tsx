import { useEffect, useRef, useState } from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fetchPhase1, type Confusion, type Phase1 as Phase1Data } from "./api";

const STRIP = 16;
const GAP = 5;
const PADDING = { left: 96, right: 12 };

// The values the LG script was tuned between -- every `threshold = ...` line in
// its source, kept so the sweep here is over the author's own candidates rather
// than a range invented later. 26 is the one it ships with.
const LG_THRESHOLDS = [10, 15, 17, 18, 19, 20, 21, 23, 26, 27, 28, 30, 40, 45, 50];

// 1 s is the camera's rate: there is no ground truth finer, so a smaller grid
// would be comparing the detectors to interpolation. Longer grids are offered
// because averaging is the one knob that trades latency for stability.
const GRIDS = [1, 2, 5, 10, 30];

function rate(v: number | null): string {
  return v === null ? "—" : `${(v * 100).toFixed(1)}%`;
}

function Matrix({ title, c, accent }: { title: string; c: Confusion; accent: string }) {
  return (
    <div className="space-y-1">
      <div className="text-[11px] font-medium" style={{ color: accent }}>
        {title}
      </div>
      <table className="text-[11px] tabular-nums">
        <tbody>
          <tr>
            <td className="w-24" />
            <td className="px-3 py-0.5 text-muted-foreground" colSpan={2}>
              ground truth
            </td>
          </tr>
          <tr>
            <td />
            <td className="px-3 py-0.5 text-muted-foreground">occupied</td>
            <td className="px-3 py-0.5 text-muted-foreground">empty</td>
          </tr>
          <tr>
            <td className="py-0.5 pr-2 text-right text-muted-foreground">
              said present
            </td>
            <td className="px-3 py-0.5">{c.tp}</td>
            <td className="px-3 py-0.5">{c.fp}</td>
          </tr>
          <tr>
            <td className="py-0.5 pr-2 text-right text-muted-foreground">
              said empty
            </td>
            <td className="px-3 py-0.5">{c.fn}</td>
            <td className="px-3 py-0.5">{c.tn}</td>
          </tr>
        </tbody>
      </table>
      <div className="text-[11px] text-muted-foreground tabular-nums">
        acc {rate(c.accuracy)} · recall {rate(c.recall)} · spec{" "}
        {rate(c.specificity)} · prec {rate(c.precision)}
      </div>
    </div>
  );
}

/** Runs of equal value, so a 300-point boolean draws as a handful of rects. */
function runs(times: number[], values: boolean[]) {
  const out: { t0: number; t1: number; value: boolean }[] = [];
  for (let i = 0; i < values.length; i++) {
    const last = out[out.length - 1];
    if (last && last.value === values[i]) last.t1 = times[i];
    else out.push({ t0: times[i], t1: times[i], value: values[i] });
  }
  return out;
}

export function Phase1({ path, dark }: { path: string; dark: boolean }) {
  const [grid, setGrid] = useState(1);
  const [lgThreshold, setLgThreshold] = useState(26);
  const [data, setData] = useState<Phase1Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [width, setWidth] = useState(720);
  const box = useRef<HTMLDivElement | null>(null);

  // Sized from the container, never from window.innerWidth: an SVG wider than
  // its column reflows the page, and the reflow re-triggers the heatmaps'
  // ResizeObserver, which aborts their in-flight tile fetches.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const w = entry.contentRect.width;
      if (w > 0) setWidth(Math.max(320, w));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetchPhase1(path, { grid, lgThreshold }, controller.signal)
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((e) => {
        if (e.name === "AbortError") return;
        setError(String(e.message ?? e));
        setData(null);
        setLoading(false);
      });
    return () => controller.abort();
  }, [path, grid, lgThreshold]);

  const cam = dark ? "#93c5fd" : "#2563eb";
  const oursColor = dark ? "#6ee7b7" : "#059669";
  const lgColor = dark ? "#fca5a5" : "#dc2626";
  const vacant = dark ? "#1f2937" : "#e5e7eb";

  const inner = Math.max(1, width - PADDING.left - PADDING.right);
  const tMax = data ? Math.max(...data.timeS, 1) : 1;
  const x = (t: number) => (t / tMax) * inner;

  const truthRuns = data
    ? runs(data.groundTruth.timeS, data.groundTruth.present)
    : [];
  const oursRuns = data?.ours ? runs(data.timeS, data.ours.present) : [];
  const lgRuns = data ? runs(data.timeS, data.lg.present) : [];

  const lanes: { label: string; color: string; runs: typeof truthRuns }[] = [
    { label: "ground truth", color: cam, runs: truthRuns },
    { label: "ours", color: oursColor, runs: oursRuns },
    { label: `LG (${lgThreshold} dB)`, color: lgColor, runs: lgRuns },
  ];

  return (
    <div ref={box} className="space-y-3">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">
            Grid (s)
          </Label>
          <Select value={String(grid)} onValueChange={(v) => setGrid(Number(v))}>
            <SelectTrigger className="w-24">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {GRIDS.map((g) => (
                <SelectItem key={g} value={String(g)}>
                  {g} s
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">
            LG threshold
          </Label>
          <Select
            value={String(lgThreshold)}
            onValueChange={(v) => setLgThreshold(Number(v))}
          >
            <SelectTrigger className="w-28">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {LG_THRESHOLDS.map((t) => (
                <SelectItem key={t} value={String(t)}>
                  {t} dB
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {loading && (
          <span className="text-[11px] text-muted-foreground">scoring…</span>
        )}
      </div>

      {error && (
        <div className="text-[11px] text-destructive leading-relaxed">{error}</div>
      )}

      {data && (
        <>
          <svg
            width={width}
            height={STRIP * 3 + GAP * 2 + 8}
            role="img"
            aria-label="ground truth against both detectors"
          >
            <g transform={`translate(${PADDING.left},4)`}>
              {lanes.map((lane, li) => {
                const y = li * (STRIP + GAP);
                return (
                  <g key={lane.label}>
                    <rect x={0} width={inner} y={y} height={STRIP} fill={vacant} />
                    {lane.runs.map((r, i) =>
                      r.value ? (
                        <rect
                          key={i}
                          x={x(r.t0)}
                          width={Math.max(1, x(r.t1) - x(r.t0))}
                          y={y}
                          height={STRIP}
                          fill={lane.color}
                        />
                      ) : null,
                    )}
                    <text
                      x={-8}
                      y={y + STRIP - 4}
                      textAnchor="end"
                      fontSize={10}
                      fill="currentColor"
                    >
                      {lane.label}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>

          {!data.calibrated && (
            <div className="text-[11px] leading-relaxed rounded border-l-2 border-amber-500 bg-amber-500/10 px-3 py-2">
              <span className="font-medium">Ours is not scored here.</span>{" "}
              {data.calibrationNote} — the reference has to come from other
              captures the camera saw as empty, never from this one&apos;s own
              empty stretches, and one such capture is not enough: a single
              capture reports how far it wanders from itself (~0.04 dB
              overnight), not how far two captures sit apart (~0.2 dB even ten
              minutes later). A threshold built on the former runs about 5×
              too low. LG needs no reference at all, so its side still scores —
              that difference is the trade the two approaches make.
            </div>
          )}

          {data.referenceWarning && (
            <div className="text-[11px] leading-relaxed rounded border-l-2 border-amber-500 bg-amber-500/10 px-3 py-2">
              <span className="font-medium">Do not trust these numbers.</span>{" "}
              {data.referenceWarning}.
            </div>
          )}

          {data.ours && (
            <div className="text-[11px] text-muted-foreground leading-relaxed rounded border-l-2 border-muted px-3 py-2">
              <span className="font-medium text-foreground">
                Whether these references describe this room is not checked, and
                cannot be.
              </span>{" "}
              The pool is screened for agreeing with itself, but a pool can
              agree with itself perfectly and still be the wrong room: five
              captures from one morning agree to 0.16 dB and put 100% of a
              capture eight days later above their threshold. Only the{" "}
              {data.ours.referenceAgeH.toFixed(1)} h between this capture and
              its nearest reference stands against that. A numeric test was
              tried and does not exist — measured over 22 working calibrations,
              the capture&apos;s distance from its pool ranges 0.09–5.65×
              threshold while two known-broken ones read 4.94× and 5.20×,
              because a capture occupied end to end is far from any empty
              reference for honest reasons. Read the matrices knowing that.
            </div>
          )}

          <div className="flex flex-wrap gap-10">
            {data.ours && (
              <Matrix title="ours" c={data.ours.confusion} accent={oursColor} />
            )}
            <Matrix
              title={`LG, ${lgThreshold} dB`}
              c={data.lg.confusion}
              accent={lgColor}
            />
          </div>

          <p className="text-[11px] text-muted-foreground leading-relaxed">
            Both verdicts resampled to {data.gridSeconds}s, the camera&apos;s own
            rate — ours reduces a window to one verdict, LG&apos;s holds a state
            between movement events, so neither is comparable until both sit on
            one grid. Windows the camera cannot call unambiguously are dropped
            rather than guessed. LG fired {data.lg.events} events.
            {data.ours &&
              ` Ours calibrated against ${data.ours.references.length} empty captures agreeing within ${data.ours.poolSpread.toFixed(2)} dB, threshold ${data.ours.threshold.toFixed(3)} dB, nearest reference ${data.ours.referenceAgeH.toFixed(1)} h away.`}
          </p>
        </>
      )}
    </div>
  );
}
