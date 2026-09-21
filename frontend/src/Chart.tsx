import { useRef } from "react";

import { formatTime, linePath, linearScale, ticks } from "./series";

/** A line on the chart. `null` values break the line rather than drawing
 *  through the gap -- see `series.linePath`. */
export interface ChartSeries {
  values: (number | null)[];
  color: string;
  width?: number;
  label: string;
  dashed?: boolean;
}

export interface ChartGuide {
  value: number;
  color: string;
  label: string;
}

/** Dots drawn on top of the lines, one per time, each with its own colour --
 *  a rate coloured by how much evidence stood behind it. */
export interface ChartDots {
  values: (number | null)[];
  colors: string[];
  radius?: number;
  label: string;
}

export interface ChartProps {
  times: number[];
  domain: [number, number];
  yDomain: [number, number];
  series: ChartSeries[];
  guides?: ChartGuide[];
  dots?: ChartDots;
  height?: number;
  yLabel: string;
  /** Label the x axis in seconds within a window (`"offset"`) rather than
   *  as capture time; used for the per-window detail panels. */
  xMode?: "time" | "offset";
  /** A vertical line at this time -- the window a detail panel is showing. */
  marker?: number | null;
  /** Called with the time under a click, so a chart can pick a window. */
  onPick?: (t: number) => void;
  dark: boolean;
  width: number;
}

// Top margin holds the axis label clear of the highest tick label; at 8 the
// two sit on the same line and overprint each other.
export const CHART_MARGIN = { top: 20, right: 12, bottom: 18, left: 44 };

/** The presence panels' line chart, as a component of its own so a second
 *  tab can draw on it. Same geometry as `Presence.tsx`'s, plus dots, a marker
 *  and a click that reports the time under the pointer. */
export function Chart({
  times,
  domain,
  yDomain,
  series,
  guides = [],
  dots,
  height = 120,
  yLabel,
  xMode = "time",
  marker,
  onPick,
  dark,
  width,
}: ChartProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const inner = {
    w: Math.max(1, width - CHART_MARGIN.left - CHART_MARGIN.right),
    h: Math.max(1, height - CHART_MARGIN.top - CHART_MARGIN.bottom),
  };
  const x = linearScale(domain, [0, inner.w]);
  const y = linearScale(yDomain, [inner.h, 0]);
  const grid = dark ? "#2a2f37" : "#e6e9ee";
  const text = dark ? "#8b95a3" : "#6b7480";
  const span = domain[1] - domain[0];

  const pick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!onPick || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const px = e.clientX - rect.left - CHART_MARGIN.left;
    if (px < 0 || px > inner.w) return;
    onPick(domain[0] + (px / inner.w) * span);
  };

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      role="img"
      aria-label={yLabel}
      onClick={pick}
      style={onPick ? { cursor: "crosshair" } : undefined}
    >
      <g transform={`translate(${CHART_MARGIN.left},${CHART_MARGIN.top})`}>
        {ticks(yDomain[0], yDomain[1], 4).map((v) => (
          <g key={v}>
            <line x1={0} x2={inner.w} y1={y(v)} y2={y(v)} stroke={grid} strokeWidth={1} />
            <text x={-6} y={y(v)} dy="0.32em" textAnchor="end" fontSize={9} fill={text}>
              {v}
            </text>
          </g>
        ))}
        {ticks(domain[0], domain[1], 6).map((v) => (
          <text key={v} x={x(v)} y={inner.h + 12} textAnchor="middle" fontSize={9} fill={text}>
            {xMode === "time" ? formatTime(v, span) : `${v}s`}
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
        {dots &&
          dots.values.map((v, i) =>
            v === null || !Number.isFinite(v) ? null : (
              <circle
                key={`${dots.label}-${i}`}
                cx={x(times[i])}
                cy={y(v)}
                r={dots.radius ?? 2}
                fill={dots.colors[i]}
              />
            ),
          )}
        {marker != null && Number.isFinite(marker) && marker >= domain[0] && marker <= domain[1] && (
          <line
            x1={x(marker)}
            x2={x(marker)}
            y1={0}
            y2={inner.h}
            stroke={dark ? "#e8eaed" : "#1b1f25"}
            strokeWidth={1}
            strokeDasharray="2 2"
            pointerEvents="none"
          />
        )}
        <text x={-CHART_MARGIN.left + 2} y={-8} fontSize={9} fill={text}>
          {yLabel}
        </text>
      </g>
    </svg>
  );
}
