import { useEffect, useMemo, useRef, useState } from "react";

import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fetchLgDetect, type LgDetect } from "./api";
import { runs } from "./series";
import type { TimeLink, TimeWindow } from "./timelink";

const PADDING = { left: 74, right: 70 };
const STRIP = 16;

// The values the author was tuning between: every `threshold = ...` line in
// csi_dump_parsing.py, commented and uncommented. 26.0 is the one left active,
// so it is the default here. Offering exactly this set keeps the tab a way of
// reproducing their sweep rather than inventing a different one.
const THRESHOLDS = [
  5, 10, 15, 17, 18, 19, 20, 21, 23, 26, 27, 28, 30, 40, 45, 50, 52, 55, 60, 100,
];

/** The LG on-board detector's verdict, replayed over this capture.
 *
 * Three things, in the order a claim has to be read: what was true, what it
 * said, and how often those agreed.
 */
export function LgDetector({
  path,
  captureTMin,
  captureTMax,
  timeLink,
  dark,
}: {
  path: string;
  captureTMin: number;
  captureTMax: number;
  timeLink?: TimeLink;
  dark: boolean;
}) {
  const [threshold, setThreshold] = useState(26);
  const [data, setData] = useState<LgDetect | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const holder = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(880);
  const [domain, setDomain] = useState<[number, number]>([captureTMin, captureTMax]);

  useEffect(() => {
    const controller = new AbortController();
    setBusy(true);
    setError(null);
    fetchLgDetect(path, threshold, 10, controller.signal)
      .then(setData)
      .catch((e) => {
        if (e.name !== "AbortError") {
          setError(String(e.message ?? e));
          setData(null);
        }
      })
      .finally(() => setBusy(false));
    return () => controller.abort();
  }, [path, threshold]);

  useEffect(() => setDomain([captureTMin, captureTMax]), [captureTMin, captureTMax]);

  useEffect(() => {
    const node = holder.current;
    if (!node) return;
    const ro = new ResizeObserver(([e]) =>
      setWidth(Math.max(320, Math.floor(e.contentRect.width))),
    );
    ro.observe(node);
    return () => ro.disconnect();
  }, [data]);

  useEffect(() => {
    if (!timeLink) return;
    return timeLink.subscribe((w: TimeWindow) => setDomain([w.tMin, w.tMax]));
  }, [timeLink]);

  const truthRuns = useMemo(
    () => (data?.truth ? runs(data.truth.timeS, data.truth.present) : []),
    [data],
  );

  const inner = Math.max(1, width - PADDING.left - PADDING.right);
  const [t0, t1] = domain;
  const span = t1 - t0 || 1;
  const x = (t: number) => Math.max(0, Math.min(inner, ((t - t0) / span) * inner));

  const said = dark ? "#c98b3a" : "#b8762a";
  const cam = dark ? "#4f7fd4" : "#3f6fc4";
  const vacant = dark ? "#2a2f37" : "#e6e9ee";
  const line = dark ? "#3a4048" : "#dfe3e9";
  const t = data?.truth ?? null;

  return (
    <div ref={holder} className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
          Threshold
        </Label>
        <Select
          value={String(threshold)}
          onValueChange={(v) => setThreshold(Number(v))}
        >
          <SelectTrigger className="h-7 w-28 text-[11px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {THRESHOLDS.map((v) => (
              <SelectItem key={v} value={String(v)} className="text-[11px]">
                {v} dB{v === 26 ? " (shipped)" : ""}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {busy && <span className="text-[11px] text-muted-foreground">running…</span>}
        {data && !busy && (
          <span className="text-[11px] text-muted-foreground">
            {data.records} records · NumPy {data.numpy} · Python {data.python}
            {data.cached ? " · cached" : ""}
            {data.parseFailures > 0 && ` · ${data.parseFailures} parse failures`}
          </span>
        )}
      </div>

      {error && (
        <div className="rounded-md border border-dashed px-3 py-2 text-[11px] leading-relaxed">
          {error}
        </div>
      )}

      {busy && !data && (
        <div className="text-[11px] text-muted-foreground leading-relaxed">
          A threshold it has not been run at before walks every record in pure
          Python — about 20 s per five minutes of capture. Cached afterwards.
        </div>
      )}

      {data && (
        <>
          {/* 1. ground truth, 2. what the detector said — truth first, so the
              detector is read against it rather than the other way round. */}
          <svg
            width={width}
            height={STRIP * 2 + 26}
            role="img"
            aria-label="ground truth against detector verdict"
          >
            <g transform={`translate(${PADDING.left},4)`}>
              <rect x={0} width={inner} y={0} height={STRIP} fill={vacant} />
              {truthRuns.map((r, i) => {
                const a = x(r.t0);
                const b = x(r.t1);
                return r.value && b > a ? (
                  <rect key={`t${i}`} x={a} width={Math.max(1, b - a)} y={0}
                        height={STRIP} fill={cam} />
                ) : null;
              })}
              <text x={-8} y={STRIP - 4} textAnchor="end" fontSize={10} fill="currentColor">
                ground truth
              </text>

              <rect x={0} width={inner} y={STRIP + 4} height={STRIP} fill={vacant} />
              {data.intervals.map((iv, i) => {
                const a = x(iv.t0);
                const b = x(iv.t1);
                return b > a ? (
                  <rect key={`d${i}`} x={a} width={Math.max(1, b - a)}
                        y={STRIP + 4} height={STRIP} fill={said} />
                ) : null;
              })}
              <text x={-8} y={STRIP * 2} textAnchor="end" fontSize={10} fill="currentColor">
                detector
              </text>
            </g>
          </svg>

          <p className="text-[11px] text-muted-foreground leading-relaxed">
            <span style={{ color: cam }}>■</span> a person was in frame ·{" "}
            <span style={{ color: said }}>■</span> it reported present ·{" "}
            {data.events.filter((e) => e.kind === "+").length} “+” and{" "}
            {data.events.filter((e) => e.kind === "-").length} “−” events,
            absence declared after {data.absenceDuration}s without movement.
          </p>

          {/* 3. the confusion matrix. A single accuracy hides which way it is
              wrong, and this detector is wrong asymmetrically: it finds the
              occupant and also calls the empty room occupied. */}
          {t ? (
            <div className="space-y-2">
              <div className="text-[11px] text-muted-foreground">
                Confusion matrix, one cell per labelled frame
              </div>
              <table className="text-[11px] tabular-nums">
                <tbody>
                  <tr>
                    <td className="w-28" />
                    <td className="px-3 py-1 text-muted-foreground" colSpan={2}>
                      ground truth
                    </td>
                  </tr>
                  <tr>
                    <td />
                    <td className="px-3 py-1 text-muted-foreground">occupied</td>
                    <td className="px-3 py-1 text-muted-foreground">empty</td>
                  </tr>
                  <tr>
                    <td className="py-1 pr-2 text-right text-muted-foreground">
                      said present
                    </td>
                    <td className="px-3 py-1 font-medium" style={{ background: line }}>
                      {t.tp} <span className="text-muted-foreground">TP</span>
                    </td>
                    <td className="px-3 py-1 font-medium" style={{ background: line }}>
                      {t.fp} <span className="text-muted-foreground">FP</span>
                    </td>
                  </tr>
                  <tr>
                    <td className="py-1 pr-2 text-right text-muted-foreground">
                      said empty
                    </td>
                    <td className="px-3 py-1 font-medium" style={{ background: line }}>
                      {t.fn} <span className="text-muted-foreground">FN</span>
                    </td>
                    <td className="px-3 py-1 font-medium" style={{ background: line }}>
                      {t.tn} <span className="text-muted-foreground">TN</span>
                    </td>
                  </tr>
                </tbody>
              </table>
              <div className="flex flex-wrap gap-x-5 text-[11px]">
                <span>
                  accuracy{" "}
                  <b
                    className={
                      t.accuracy > t.baseRate ? "text-foreground" : "text-red-500"
                    }
                  >
                    {t.accuracy.toFixed(2)}
                  </b>
                </span>
                <span className="text-muted-foreground">
                  precision {t.precision.toFixed(2)}
                </span>
                <span className="text-muted-foreground">
                  recall {t.recall.toFixed(2)}
                </span>
                <span className="text-muted-foreground">
                  always-present baseline {t.baseRate.toFixed(2)}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                The baseline is what always answering “present” would score, and
                accuracy below it means the detector has said nothing — shown in
                red for that reason. It is a frame-to-frame amplitude-difference
                trigger, so it answers “is something changing”: a still occupant
                and a changed room look alike to it, which is why the errors
                land mostly in FP.
              </p>
            </div>
          ) : (
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              No <code>_cv.json</code> beside this capture, so there is nothing
              to score against — the strip above is the detector&apos;s claim
              with nothing to check it.
            </p>
          )}
        </>
      )}
    </div>
  );
}
