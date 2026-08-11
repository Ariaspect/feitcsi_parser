import { useEffect, useRef } from "react";
import uPlot from "uplot";

interface HeatmapProps {
  timeSeconds: number[];
  matrix: number[][];
  subcarrierCount: number;
  minValue: number;
  maxValue: number;
  title: string;
  colorLabel: string;
  palette?: string[];
  height?: number;
}

const DEFAULT_PALETTE = [
  "#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89",
  "#35b779", "#6ece58", "#b5de2b", "#fde725",
];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function colorFor(value: number, min: number, max: number, palette: string[]): string {
  if (!Number.isFinite(value)) return "#000000";
  const t = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  const scaled = t * (palette.length - 1);
  const i = Math.floor(scaled);
  const f = scaled - i;
  if (i >= palette.length - 1) return palette[palette.length - 1];
  const c1 = hexToRgb(palette[i]);
  const c2 = hexToRgb(palette[i + 1]);
  const r = Math.round(lerp(c1[0], c2[0], f));
  const g = Math.round(lerp(c1[1], c2[1], f));
  const b = Math.round(lerp(c1[2], c2[2], f));
  return `rgb(${r},${g},${b})`;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ];
}

export function Heatmap({
  timeSeconds,
  matrix,
  subcarrierCount,
  minValue,
  maxValue,
  title,
  colorLabel,
  palette = DEFAULT_PALETTE,
  height = 400,
}: HeatmapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const width = container.clientWidth;

    // Subcarrier y-axis: -N/2 .. +N/2-1
    const halfN = Math.floor(subcarrierCount / 2);

    // uPlot expects data as (n+1) arrays: [x, y1, y2, ...]
    // For heatmap we pass only x (time) and use draw hook to render matrix.
    // We pass a single dummy y series to satisfy uPlot's data shape.
    const dummyY = new Array(timeSeconds.length).fill(0);

    const data: uPlot.AlignedData = [timeSeconds, dummyY] as unknown as uPlot.AlignedData;

    const opts: uPlot.Options = {
      width,
      height,
      title,
      series: [
        { label: "Time (s)" },
        { label: "subcarrier", show: false },
      ],
      scales: {
        x: { time: false },
        y: { min: -halfN, max: halfN - 1 },
      },
      axes: [
        {},
        { label: "Subcarrier bin" },
      ],
      hooks: {
        draw: [
          (u: uPlot) => {
            const ctx = u.ctx;
            if (!ctx) return;
            // Plot area
            const left = u.bbox.left;
            const top = u.bbox.top;
            const w = u.bbox.width;
            const h = u.bbox.height;
            const nFrames = timeSeconds.length;
            if (nFrames === 0 || subcarrierCount === 0) return;

            const tMin = timeSeconds[0];
            const tMax = timeSeconds[nFrames - 1];
            const tRange = Math.max(tMax - tMin, 1e-9);

            // Map time → x pixel
            const timeToX = (t: number) =>
              left + ((t - tMin) / tRange) * w;

            // Map subcarrier index → y pixel (top = +N/2-1, bottom = -N/2)
            const scToY = (sc: number) =>
              top + ((halfN - 1 - sc) / (subcarrierCount - 1 || 1)) * h;

            const cellW = w / Math.max(nFrames, 1);
            const cellH = h / Math.max(subcarrierCount, 1);

            // Draw cells
            for (let fi = 0; fi < nFrames; fi++) {
              const x = timeToX(timeSeconds[fi]);
              const row = matrix[fi];
              for (let si = 0; si < subcarrierCount; si++) {
                const v = row[si];
                const y = scToY(si);
                ctx.fillStyle = colorFor(v, minValue, maxValue, palette);
                ctx.fillRect(x, y, cellW + 1, cellH + 1);
              }
            }

            // Color bar on right
            const barW = 14;
            const barX = left + w + 8;
            const barH = h;
            const grad = ctx.createLinearGradient(0, top + barH, 0, top);
            for (let i = 0; i < palette.length; i++) {
              grad.addColorStop(i / (palette.length - 1), palette[i]);
            }
            ctx.fillStyle = grad;
            ctx.fillRect(barX, top, barW, barH);
            ctx.strokeStyle = "#888";
            ctx.strokeRect(barX, top, barW, barH);
            ctx.fillStyle = "#000";
            ctx.font = "10px sans-serif";
            ctx.textAlign = "left";
            ctx.fillText(maxValue.toFixed(1), barX + barW + 4, top + 10);
            ctx.fillText(minValue.toFixed(1), barX + barW + 4, top + barH);
            ctx.save();
            ctx.translate(barX + barW + 4, top + barH / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = "center";
            ctx.fillText(colorLabel, 0, 0);
            ctx.restore();
          },
        ],
      },
    };

    const plot = new uPlot(opts, data, container);
    plotRef.current = plot;

    return () => {
      plot.destroy();
      plotRef.current = null;
    };
  }, [timeSeconds, matrix, subcarrierCount, minValue, maxValue, title, colorLabel, palette, height]);

  return <div ref={containerRef} style={{ width: "100%" }} />;
}
