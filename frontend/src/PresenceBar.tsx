import { useEffect, useMemo, useRef, useState } from "react";

import { fetchLabels, type Labels } from "./api";
import { runs } from "./series";
import type { TimeLink, TimeWindow } from "./timelink";

// Matches Heatmap's PADDING so the strip sits under the plot area and reads
// against it column for column. Nothing enforces that agreement, so if the
// heatmap's margins move this must move with them.
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
 * Only renders for captures carrying a `_cv.json`; ordinary captures show
 * nothing rather than a permanently blank strip.
 *
 * Width comes from a ResizeObserver on its own container, never from
 * `window.innerWidth`: an SVG wider than the column it sits in overflows, and
 * the reflow that causes retriggers the heatmap's ResizeObserver, which
 * redraws and aborts its in-flight tile requests. The visible symptom is
 * NS_BINDING_ABORTED on every pan and heatmaps that never finish loading.
 */
export function PresenceBar({
  path,
  captureTMin,
  captureTMax,
  timeLink,
  dark,
}: Props) {
  const [labels, setLabels] = useState<Labels | null>(null);
  const holder = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(880);
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

  useEffect(() => {
    const node = holder.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.max(320, Math.floor(entry.contentRect.width)));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [labels]);

  // Follow the shared window; never publish. This strip has no zoom of its own
  // to broadcast, and echoing a window it was handed would loop.
  useEffect(() => {
    if (!timeLink) return;
    return timeLink.subscribe((w: TimeWindow) => setDomain([w.tMin, w.tMax]));
  }, [timeLink]);

  const presenceRuns = useMemo(
    () =>
      labels?.present ? runs(labels.present.timeS, labels.present.present) : [],
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

  return (
    <div ref={holder} className="w-full text-[11px] text-muted-foreground">
      <svg
        width={width}
        height={STRIP_HEIGHT + 6}
        role="img"
        aria-label="webcam person detections"
      >
        <g transform={`translate(${PADDING.left},3)`}>
          {presenceRuns.map((run, i) => {
            const a = clamp(x(run.t0));
            const b = clamp(x(run.t1));
            if (b <= a) return null;
            return (
              <rect
                key={`p${i}`}
                x={a}
                width={Math.max(1, b - a)}
                y={0}
                height={STRIP_HEIGHT}
                fill={run.value ? occupied : vacant}
              />
            );
          })}
          <text
            x={-8}
            y={STRIP_HEIGHT - 3}
            textAnchor="end"
            fill="currentColor"
            fontSize={10}
          >
            camera
          </text>
        </g>
      </svg>
      <p className="leading-relaxed">
        <span style={{ color: occupied }}>■</span> person detected in frame
        {labels.position && (
          <>
            {" · "}position <b>{labels.position}</b>
          </>
        )}
        {" · "}from <code>{labels.source}</code>
      </p>
    </div>
  );
}
