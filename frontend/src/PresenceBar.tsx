import { useEffect, useMemo, useState } from "react";

import { fetchLabels, type Labels } from "./api";
import { runs } from "./series";
import type { TimeLink, TimeWindow } from "./timelink";

// Matches Heatmap's PADDING so the strip sits directly under the plot area and
// reads against it column for column. Nothing enforces that agreement, so if
// the heatmap's margins move this must move with them.
const PADDING = { right: 70, left: 60 };
const STRIP_HEIGHT = 14;

interface Props {
  path: string;
  captureTMin: number;
  captureTMax: number;
  timeLink?: TimeLink;
  dark: boolean;
}

/** Webcam ground truth for a labelled capture, as a strip under the heatmap.
 *
 * Only renders for captures that carry a `_cv.json`; ordinary captures show
 * nothing rather than an empty frame, because most captures have no labels and
 * a permanently blank strip would just be furniture.
 *
 * The detections are the evidence and the protocol phases are the intent: the
 * two are drawn as separate rows precisely so a disagreement is visible rather
 * than reconciled away. On the first labelled runs every transition landed
 * late, by 2 s at best and 18 s at worst.
 */
export function PresenceBar({
  path,
  captureTMin,
  captureTMax,
  timeLink,
  dark,
}: Props) {
  const [labels, setLabels] = useState<Labels | null>(null);
  const [width, setWidth] = useState(900);
  const [domain, setDomain] = useState<[number, number]>([
    captureTMin,
    captureTMax,
  ]);

  useEffect(() => {
    const controller = new AbortController();
    fetchLabels(path, controller.signal)
      .then(setLabels)
      .catch(() => setLabels(null));
    return () => controller.abort();
  }, [path]);

  useEffect(() => {
    setDomain([captureTMin, captureTMax]);
  }, [captureTMin, captureTMax]);

  // Follow the shared time window so panning the heatmap pans the strip. The
  // publisher's identity is ignored: this view never publishes, so it cannot
  // be the source of its own update.
  useEffect(() => {
    if (!timeLink) return;
    return timeLink.subscribe((w: TimeWindow) => setDomain([w.tMin, w.tMax]));
  }, [timeLink]);

  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth > 0 ? window.innerWidth : 900);
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const presenceRuns = useMemo(
    () =>
      labels?.present
        ? runs(labels.present.timeS, labels.present.present)
        : [],
    [labels],
  );

  if (!labels?.present) return null;

  const inner = Math.max(1, width - PADDING.left - PADDING.right);
  const [t0, t1] = domain;
  const span = t1 - t0 || 1;
  const x = (t: number) => ((t - t0) / span) * inner;
  const clamp = (v: number) => Math.max(0, Math.min(inner, v));

  const occupied = dark ? "#4f7fd4" : "#3f6fc4";
  const vacant = dark ? "#2a2f37" : "#e6e9ee";
  const phaseFill: Record<string, string> = {
    sitting: dark ? "#7b5ea7" : "#8f74ba",
    empty: dark ? "#333940" : "#dfe3e9",
  };

  return (
    <div className="text-[11px] text-muted-foreground">
      <svg width={width} height={STRIP_HEIGHT * 2 + 26} role="img"
           aria-label="webcam ground truth">
        <g transform={`translate(${PADDING.left},4)`}>
          {presenceRuns.map((run, i) => {
            const a = clamp(x(run.t0));
            const b = clamp(x(run.t1));
            if (b <= a) return null;
            return (
              <rect key={`p${i}`} x={a} width={Math.max(1, b - a)} y={0}
                    height={STRIP_HEIGHT}
                    fill={run.value ? occupied : vacant} />
            );
          })}
          <text x={-8} y={STRIP_HEIGHT - 3} textAnchor="end"
                fill="currentColor" fontSize={10}>camera</text>

          {(labels.phases ?? []).map((ph, i) => {
            const a = clamp(x(ph.t0));
            const b = clamp(x(ph.t1));
            if (b <= a) return null;
            return (
              <rect key={`f${i}`} x={a} width={Math.max(1, b - a)}
                    y={STRIP_HEIGHT + 2} height={STRIP_HEIGHT}
                    fill={phaseFill[ph.label] ?? vacant} opacity={0.75} />
            );
          })}
          {labels.phases && (
            <text x={-8} y={STRIP_HEIGHT * 2 - 1} textAnchor="end"
                  fill="currentColor" fontSize={10}>protocol</text>
          )}
        </g>
      </svg>
      <p className="leading-relaxed">
        <span style={{ color: occupied }}>■</span> person detected in frame
        {labels.phases && (
          <>
            {" · "}
            <span style={{ color: phaseFill.sitting }}>■</span> intended
            “sitting” phase
          </>
        )}
        {labels.position && <> · position <b>{labels.position}</b></>}
        {" · "}from <code>{labels.source}</code>. The camera row is measured;
        the protocol row is what was asked for. They disagree at every
        transition, and where they do the camera is the evidence.
      </p>
    </div>
  );
}
