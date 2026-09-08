import { useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { fetchLgDetect, type LgDetect } from "./api";
import { runs } from "./series";
import type { TimeLink, TimeWindow } from "./timelink";

const PADDING = { left: 60, right: 70 };
const STRIP = 16;

/** The LG on-board detector's verdict, replayed over this capture.
 *
 * Its own code decides; nothing here reimplements it. The two strips are the
 * whole point: what it said, against what the camera saw.
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
  const [pending, setPending] = useState(26);
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
        if (e.name !== "AbortError") { setError(String(e.message ?? e)); setData(null); }
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

  return (
    <div ref={holder} className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">
          Threshold (dB)
        </Label>
        <Input
          className="h-7 w-20 text-[11px]"
          type="number"
          value={pending}
          onChange={(e) => setPending(Number(e.target.value))}
        />
        <Button
          variant="outline"
          size="sm"
          className="h-7 px-2 text-[11px]"
          disabled={busy || pending === threshold}
          onClick={() => setThreshold(pending)}
        >
          {busy ? "running…" : "Run"}
        </Button>
        {data && (
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
        <div className="text-[11px] text-muted-foreground">
          replaying — a first run walks every record in pure Python and takes
          around 20 s per five minutes of capture; the result is cached after.
        </div>
      )}

      {data && (
        <>
          <svg width={width} height={STRIP * 2 + 30} role="img"
               aria-label="detector verdict against camera">
            <g transform={`translate(${PADDING.left},4)`}>
              <rect x={0} width={inner} y={0} height={STRIP} fill={vacant} />
              {data.intervals.map((iv, i) => {
                const a = x(iv.t0), b = x(iv.t1);
                return b > a ? (
                  <rect key={`d${i}`} x={a} width={Math.max(1, b - a)} y={0}
                        height={STRIP} fill={said} />
                ) : null;
              })}
              <text x={-8} y={STRIP - 4} textAnchor="end" fontSize={10} fill="currentColor">
                detector
              </text>

              {truthRuns.map((r, i) => {
                const a = x(r.t0), b = x(r.t1);
                return b > a ? (
                  <rect key={`t${i}`} x={a} width={Math.max(1, b - a)}
                        y={STRIP + 3} height={STRIP}
                        fill={r.value ? cam : vacant} />
                ) : null;
              })}
              {data.truth && (
                <text x={-8} y={STRIP * 2 - 1} textAnchor="end" fontSize={10}
                      fill="currentColor">camera</text>
              )}
            </g>
          </svg>

          <p className="text-[11px] text-muted-foreground leading-relaxed">
            <span style={{ color: said }}>■</span> it reported present ·{" "}
            <span style={{ color: cam }}>■</span> a person was in frame ·{" "}
            {data.events.filter((e) => e.kind === "+").length} “+” and{" "}
            {data.events.filter((e) => e.kind === "-").length} “−” events, with
            absence declared after {data.absenceDuration}s without movement.
          </p>

          {data.truth && (
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              Against the camera: accuracy{" "}
              <b className="text-foreground">{data.truth.accuracy.toFixed(2)}</b>,
              precision {data.truth.precision.toFixed(2)}, recall{" "}
              {data.truth.recall.toFixed(2)} (tp {data.truth.tp}, fp{" "}
              {data.truth.fp}, fn {data.truth.fn}, tn {data.truth.tn}).{" "}
              Always answering “present” would score{" "}
              {data.truth.baseRate.toFixed(2)} — the bar this has to clear to
              have said anything. It is a frame-to-frame amplitude-difference
              trigger, so it answers “is something changing”, and a still
              occupant and a changed room look alike to it.
            </p>
          )}
        </>
      )}
    </div>
  );
}
