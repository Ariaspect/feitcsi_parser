import { useEffect, useRef } from "react";
import { VIRIDIS, buildLut } from "./colormap";
import { renderToImageData, subcarrierSourceRect, type Aggregation } from "./render";

interface HeatmapProps {
  timeSeconds: number[];
  matrix: number[][];
  subcarrierCount: number;
  minValue: number;
  maxValue: number;
  title: string;
  colorLabel: string;
  palette?: readonly string[];
  aggregation?: Aggregation;
  height?: number;
}

const PADDING = { top: 30, right: 70, bottom: 40, left: 60 };

interface View {
  tMin: number;
  tMax: number;
  scMin: number;
  scMax: number;
}

function nearestFrameIndex(times: number[], t: number): number {
  let lo = 0, hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(times[lo - 1] - t) < Math.abs(times[lo] - t)) return lo - 1;
  return lo;
}

export function Heatmap({
  timeSeconds,
  matrix,
  subcarrierCount,
  minValue,
  maxValue,
  title,
  colorLabel,
  palette = VIRIDIS,
  aggregation = "max",
  height = 400,
}: HeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<View | null>(null);
  const hoverRef = useRef<{ t: number; sc: number; v: number } | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const imageDataRef = useRef<ImageData | null>(null);

  const halfN = Math.floor(subcarrierCount / 2);

  useEffect(() => {
    if (timeSeconds.length === 0) return;
    viewRef.current = {
      tMin: timeSeconds[0],
      tMax: timeSeconds[timeSeconds.length - 1],
      scMin: -halfN,
      scMax: halfN - 1,
    };
  }, [timeSeconds, halfN]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const cssW = container.clientWidth;
      const cssH = height;
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);

      const plot = {
        x: PADDING.left,
        y: PADDING.top,
        w: cssW - PADDING.left - PADDING.right,
        h: cssH - PADDING.top - PADDING.bottom,
      };

      const view = viewRef.current;
      if (!view || timeSeconds.length === 0 || subcarrierCount === 0) {
        ctx.fillStyle = "#888";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("No data", cssW / 2, cssH / 2);
        return;
      }

      const tToX = (t: number) =>
        plot.x + ((t - view.tMin) / (view.tMax - view.tMin || 1e-9)) * plot.w;
      const scToY = (sc: number) =>
        plot.y + ((view.scMax - sc) / (view.scMax - view.scMin || 1)) * plot.h;

      const tRange = view.tMax - view.tMin || 1e-9;
      const scRange = view.scMax - view.scMin || 1;

      // --- Render heatmap data via LUT + ImageData blit ---
      const widthPx = Math.max(1, Math.round(plot.w));

      if (!offscreenRef.current) {
        offscreenRef.current = document.createElement("canvas");
      }
      const offscreen = offscreenRef.current;
      offscreen.width = widthPx;
      offscreen.height = subcarrierCount;
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) return;

      const lut = buildLut(palette, 256);
      const imageData = renderToImageData({
        matrix,
        timeSeconds,
        subcarrierCount,
        view: { tMin: view.tMin, tMax: view.tMax },
        widthPx,
        lut,
        min: minValue,
        max: maxValue,
        aggregation,
        target: imageDataRef.current ?? undefined,
      });
      imageDataRef.current = imageData;
      offCtx.putImageData(imageData, 0, 0);

      // Crop to the visible subcarrier band and stretch it into the plot area.
      const { srcY, srcH } = subcarrierSourceRect(subcarrierCount, view.scMin, view.scMax);

      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offscreen, 0, srcY, widthPx, srcH, plot.x, plot.y, plot.w, plot.h);
      // --- End heatmap data render ---

      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);

      ctx.fillStyle = "#000";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const xTicks = 6;
      for (let i = 0; i <= xTicks; i++) {
        const t = view.tMin + (i / xTicks) * tRange;
        const x = tToX(t);
        ctx.fillText(t.toFixed(2), x, plot.y + plot.h + 6);
        ctx.beginPath();
        ctx.moveTo(x, plot.y + plot.h);
        ctx.lineTo(x, plot.y + plot.h + 4);
        ctx.stroke();
      }

      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      const yTicks = 6;
      for (let i = 0; i <= yTicks; i++) {
        const sc = view.scMin + (i / yTicks) * scRange;
        const y = scToY(sc);
        ctx.fillText(sc.toString(), plot.x - 6, y);
        ctx.beginPath();
        ctx.moveTo(plot.x, y);
        ctx.lineTo(plot.x - 4, y);
        ctx.stroke();
      }

      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText("Time (s)", plot.x + plot.w / 2, cssH - 4);
      ctx.save();
      ctx.translate(12, plot.y + plot.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Subcarrier bin", 0, 0);
      ctx.restore();

      const barW = 14;
      const barX = plot.x + plot.w + 16;
      const grad = ctx.createLinearGradient(0, plot.y + plot.h, 0, plot.y);
      for (let i = 0; i < palette.length; i++) {
        grad.addColorStop(i / (palette.length - 1), palette[i]);
      }
      ctx.fillStyle = grad;
      ctx.fillRect(barX, plot.y, barW, plot.h);
      ctx.strokeStyle = "#888";
      ctx.strokeRect(barX, plot.y, barW, plot.h);
      ctx.fillStyle = "#000";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(maxValue.toFixed(1), barX + barW + 4, plot.y + 4);
      ctx.fillText(minValue.toFixed(1), barX + barW + 4, plot.y + plot.h - 4);
      ctx.save();
      ctx.translate(barX + barW + 30, plot.y + plot.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText(colorLabel, 0, 0);
      ctx.restore();

      ctx.fillStyle = "#000";
      ctx.font = "bold 13px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(title, plot.x, 6);

      const hover = hoverRef.current;
      if (hover) {
        const x = tToX(hover.t);
        const y = scToY(hover.sc);
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, plot.y);
        ctx.lineTo(x, plot.y + plot.h);
        ctx.moveTo(plot.x, y);
        ctx.lineTo(plot.x + plot.w, y);
        ctx.stroke();
        ctx.fillStyle = "rgba(0,0,0,0.85)";
        const txt = `t=${hover.t.toFixed(3)}s  sc=${hover.sc}  v=${hover.v.toFixed(2)}`;
        ctx.font = "11px monospace";
        const tw = ctx.measureText(txt).width;
        ctx.fillRect(x + 8, y - 22, tw + 12, 18);
        ctx.fillStyle = "#fff";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(txt, x + 14, y - 13);
      }
    };

    draw();

    const plot = {
      x: PADDING.left,
      y: PADDING.top,
      w: container.clientWidth - PADDING.left - PADDING.right,
      h: height - PADDING.top - PADDING.bottom,
    };

    const xToT = (x: number) => {
      const v = viewRef.current!;
      return v.tMin + ((x - plot.x) / plot.w) * (v.tMax - v.tMin);
    };
    const yToSc = (y: number) => {
      const v = viewRef.current!;
      return v.scMax - ((y - plot.y) / plot.h) * (v.scMax - v.scMin);
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const v = viewRef.current!;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const tAt = xToT(mx);
      const scAt = yToSc(my);
      const factor = e.deltaY > 0 ? 1.2 : 1 / 1.2;
      const newTRange = (v.tMax - v.tMin) * factor;
      const newScRange = (v.scMax - v.scMin) * factor;
      const tRatio = (tAt - v.tMin) / (v.tMax - v.tMin || 1);
      const scRatio = (v.scMax - scAt) / (v.scMax - v.scMin || 1);
      v.tMin = tAt - tRatio * newTRange;
      v.tMax = tAt + (1 - tRatio) * newTRange;
      v.scMin = scAt - (1 - scRatio) * newScRange;
      v.scMax = scAt + scRatio * newScRange;
      hoverRef.current = null;
      draw();
    };

    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    const onDown = (e: MouseEvent) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    };
    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      if (dragging) {
        const v = viewRef.current!;
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        const dt = -(dx / plot.w) * (v.tMax - v.tMin);
        const dsc = (dy / plot.h) * (v.scMax - v.scMin);
        v.tMin += dt; v.tMax += dt;
        v.scMin += dsc; v.scMax += dsc;
        lastX = e.clientX;
        lastY = e.clientY;
        hoverRef.current = null;
        draw();
      } else if (mx >= plot.x && mx <= plot.x + plot.w && my >= plot.y && my <= plot.y + plot.h) {
        const t = xToT(mx);
        const sc = Math.round(yToSc(my));
        const fi = nearestFrameIndex(timeSeconds, t);
        const si = sc + halfN;
        if (fi >= 0 && fi < matrix.length && si >= 0 && si < subcarrierCount) {
          hoverRef.current = { t: timeSeconds[fi], sc, v: matrix[fi][si] };
        } else {
          hoverRef.current = null;
        }
        draw();
      } else if (hoverRef.current) {
        hoverRef.current = null;
        draw();
      }
    };
    const onUp = () => { dragging = false; };
    const onLeave = () => { hoverRef.current = null; draw(); };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("mouseleave", onLeave);

    const onResize = () => draw();
    window.addEventListener("resize", onResize);

    return () => {
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      canvas.removeEventListener("mouseleave", onLeave);
      window.removeEventListener("resize", onResize);
    };
  }, [timeSeconds, matrix, subcarrierCount, minValue, maxValue, title, colorLabel, palette, aggregation, height, halfN]);

  return <div ref={containerRef} style={{ width: "100%" }}><canvas ref={canvasRef} /></div>;
}
