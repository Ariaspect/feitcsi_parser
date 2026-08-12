import { useEffect, useRef } from "react";
import { select } from "d3-selection";
import { zoom, zoomIdentity, type D3ZoomEvent, type ZoomBehavior } from "d3-zoom";
import { scaleLinear, type ScaleLinear } from "d3-scale";
import { VIRIDIS, buildLut } from "./colormap";
import { renderToImageData, subcarrierSourceRect, type Aggregation } from "./render";
import {
  advanceView,
  clampScWindow,
  clampTimeWindow,
  fullView,
  shouldResetView,
  type CaptureId,
  type View,
} from "./view";

interface HeatmapProps {
  /** Identifies which capture the data came from, so switching files resets
   * the view. The component is never remounted on a path change. */
  filename: string;
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

interface PlotRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Live geometry written by draw() and read by mouse handlers. Holds the
 * current plot rect and the screen<->data conversions for the current view, so
 * handlers never close over a stale render's `plot` or `view`. */
interface Geometry {
  plot: PlotRect;
  tToX: (t: number) => number;
  xToT: (x: number) => number;
  scToY: (sc: number) => number;
  yToSc: (y: number) => number;
}

/** Everything draw() needs from the latest render, mirrored into a ref so the
 * mount-once effect's draw/listeners never close over stale props. */
interface PropsMirror {
  filename: string;
  timeSeconds: number[];
  matrix: number[][];
  subcarrierCount: number;
  halfN: number;
  minValue: number;
  maxValue: number;
  title: string;
  colorLabel: string;
  palette: readonly string[];
  aggregation: Aggregation;
  height: number;
}

function nearestFrameIndex(times: readonly number[], t: number): number {
  let lo = 0,
    hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(times[lo - 1] - t) < Math.abs(times[lo] - t)) return lo - 1;
  return lo;
}

export function Heatmap({
  filename,
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

  // --- Persistent view state (survives polls, never rebuilt from identity) ---
  const viewRef = useRef<View | null>(null);
  const followLiveRef = useRef(true);
  const captureRef = useRef<CaptureId | null>(null);

  // --- Live state for draw + handlers, all ref-backed to avoid stale closures
  const hoverRef = useRef<{ t: number; sc: number; v: number } | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const imageDataRef = useRef<ImageData | null>(null);
  const geometryRef = useRef<Geometry | null>(null);
  const propsRef = useRef<PropsMirror | null>(null);

  // --- Bridges between the mount-once effect and the per-render effect ---
  const drawRef = useRef<(() => void) | null>(null);
  const syncZoomRef = useRef<(() => void) | null>(null);
  const baseScaleRef = useRef<ScaleLinear<number, number> | null>(null);
  const zoomBehaviorRef = useRef<ZoomBehavior<HTMLCanvasElement, unknown> | null>(null);
  const programmaticRef = useRef(false);

  const halfN = Math.floor(subcarrierCount / 2);

  // Mirror latest props into a ref on every render so the stable draw/listen
  // path always sees fresh data without re-subscribing.
  propsRef.current = {
    filename,
    timeSeconds,
    matrix,
    subcarrierCount,
    halfN,
    minValue,
    maxValue,
    title,
    colorLabel,
    palette,
    aggregation,
    height,
  };

  // -------------------------------------------------------------------------
  // Effect 1 (runs every render): view lifecycle + cheap redraw.
  // Decides whether to reset, slide (follow-live), or freeze the view, then
  // redraws and re-syncs the d3 transform. Does NOT register DOM listeners.
  // Declared before Effect 2 so on the first mount it runs first and the view
  // is set before the mount effect's initial draw.
  // -------------------------------------------------------------------------
  useEffect(() => {
    const props = propsRef.current;
    if (!props) return;
    if (props.timeSeconds.length === 0) {
      // No data this poll: show the empty state without touching the view, so
      // a transient empty frame doesn't reset a frozen or live window.
      drawRef.current?.();
      return;
    }

    const newTMin = props.timeSeconds[0];
    const newTMax = props.timeSeconds[props.timeSeconds.length - 1];
    const newCapture: CaptureId = {
      filename: props.filename,
      numSubcarriers: props.subcarrierCount,
      tMin: newTMin,
      tMax: newTMax,
    };

    const prevCapture = captureRef.current;
    if (!prevCapture || shouldResetView(prevCapture, newCapture)) {
      // Fresh capture or identity change (subcarrier count or backwards time):
      // full extent, follow live. This is the ONLY path that resets the view.
      viewRef.current = fullView({
        ...newCapture,
        scMin: -props.halfN,
        scMax: props.halfN - 1,
      });
      followLiveRef.current = true;
    } else if (viewRef.current) {
      // Same capture: slide if live, freeze if not. advanceView returns the
      // same reference when frozen, so a frozen view is bit-identical across
      // any number of polls.
      viewRef.current = advanceView(
        viewRef.current,
        { tMin: newTMin, tMax: newTMax },
        followLiveRef.current,
      );
    }
    captureRef.current = newCapture;

    // The base time scale's domain must reflect the current data extent before
    // we resync the d3 transform, otherwise rescaleX would map to stale data.
    if (baseScaleRef.current) {
      baseScaleRef.current.domain([newTMin, newTMax]);
    }

    drawRef.current?.();
    syncZoomRef.current?.();
  });

  // -------------------------------------------------------------------------
  // Effect 2 (mount once): register DOM listeners, ResizeObserver, and d3-zoom.
  // Defines draw() and syncZoomToView(), storing them in refs. All handlers
  // read from refs so they never close over a stale render's geometry or view.
  // -------------------------------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const sel = select(canvas);

    // --- draw: the single render path. Reads everything from refs. ---
    const draw = () => {
      const ctx = canvas.getContext("2d");
      const props = propsRef.current;
      if (!ctx || !props) return;

      const dpr = window.devicePixelRatio || 1;
      const cssW = container.clientWidth;
      const cssH = props.height;
      if (cssW === 0 || cssH === 0) return;
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);

      const plot: PlotRect = {
        x: PADDING.left,
        y: PADDING.top,
        w: cssW - PADDING.left - PADDING.right,
        h: cssH - PADDING.top - PADDING.bottom,
      };

      const view = viewRef.current;
      const {
        timeSeconds: ts,
        matrix: mx,
        subcarrierCount: nSc,
        minValue: mn,
        maxValue: mxVal,
        title: ttl,
        colorLabel: clbl,
        palette: pal,
        aggregation: agg,
      } = props;

      if (!view || ts.length === 0 || nSc === 0) {
        ctx.fillStyle = "#888";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("No data", cssW / 2, cssH / 2);
        geometryRef.current = null;
        return;
      }

      const tRange = view.tMax - view.tMin || 1e-9;
      const scRange = view.scMax - view.scMin || 1;
      const tToX = (t: number) => plot.x + ((t - view.tMin) / tRange) * plot.w;
      const xToT = (x: number) => view.tMin + ((x - plot.x) / plot.w) * tRange;
      const scToY = (sc: number) =>
        plot.y + ((view.scMax - sc) / scRange) * plot.h;
      const yToSc = (y: number) =>
        view.scMax - ((y - plot.y) / plot.h) * scRange;

      // Publish live geometry for the mouse handlers.
      geometryRef.current = { plot, tToX, xToT, scToY, yToSc };

      // Keep the base time scale in sync: domain from the data extent, range
      // from the current plot width. d3-zoom rescales this to derive the
      // visible window.
      if (baseScaleRef.current) {
        baseScaleRef.current
          .domain([ts[0], ts[ts.length - 1]])
          .range([0, plot.w]);
      }

      // --- Heatmap data via LUT + ImageData blit (phase 1) ---
      const widthPx = Math.max(1, Math.round(plot.w));
      if (!offscreenRef.current) {
        offscreenRef.current = document.createElement("canvas");
      }
      const offscreen = offscreenRef.current;
      offscreen.width = widthPx;
      offscreen.height = nSc;
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) return;

      const lut = buildLut(pal, 256);
      const imageData = renderToImageData({
        matrix: mx,
        timeSeconds: ts,
        subcarrierCount: nSc,
        view: { tMin: view.tMin, tMax: view.tMax },
        widthPx,
        lut,
        min: mn,
        max: mxVal,
        aggregation: agg,
        target: imageDataRef.current ?? undefined,
      });
      imageDataRef.current = imageData;
      offCtx.putImageData(imageData, 0, 0);

      const { srcY, srcH } = subcarrierSourceRect(nSc, view.scMin, view.scMax);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offscreen, 0, srcY, widthPx, srcH, plot.x, plot.y, plot.w, plot.h);

      // --- Frame ---
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);

      // --- Time axis ticks via d3-scale (nice, stable round numbers) ---
      const tScale = scaleLinear()
        .domain([view.tMin, view.tMax])
        .range([plot.x, plot.x + plot.w]);
      const tFmt = tScale.tickFormat(6);
      ctx.fillStyle = "#000";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (const t of tScale.ticks(6)) {
        const x = tScale(t);
        ctx.fillText(tFmt(t), x, plot.y + plot.h + 6);
        ctx.beginPath();
        ctx.moveTo(x, plot.y + plot.h);
        ctx.lineTo(x, plot.y + plot.h + 4);
        ctx.stroke();
      }

      // --- Subcarrier axis ticks via d3-scale ---
      const scScale = scaleLinear()
        .domain([view.scMin, view.scMax])
        .range([plot.y + plot.h, plot.y]);
      const scFmt = scScale.tickFormat(6);
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (const sc of scScale.ticks(6)) {
        const y = scScale(sc);
        ctx.fillText(scFmt(sc), plot.x - 6, y);
        ctx.beginPath();
        ctx.moveTo(plot.x, y);
        ctx.lineTo(plot.x - 4, y);
        ctx.stroke();
      }

      // --- Axis labels ---
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText("Time (s)", plot.x + plot.w / 2, cssH - 4);
      ctx.save();
      ctx.translate(12, plot.y + plot.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Subcarrier bin", 0, 0);
      ctx.restore();

      // --- Colorbar ---
      const barW = 14;
      const barX = plot.x + plot.w + 16;
      const grad = ctx.createLinearGradient(0, plot.y + plot.h, 0, plot.y);
      for (let i = 0; i < pal.length; i++) {
        grad.addColorStop(i / (pal.length - 1), pal[i]);
      }
      ctx.fillStyle = grad;
      ctx.fillRect(barX, plot.y, barW, plot.h);
      ctx.strokeStyle = "#888";
      ctx.strokeRect(barX, plot.y, barW, plot.h);
      ctx.fillStyle = "#000";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(mxVal.toFixed(1), barX + barW + 4, plot.y + 4);
      ctx.fillText(mn.toFixed(1), barX + barW + 4, plot.y + plot.h - 4);
      ctx.save();
      ctx.translate(barX + barW + 30, plot.y + plot.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText(clbl, 0, 0);
      ctx.restore();

      // --- Title ---
      ctx.fillStyle = "#000";
      ctx.font = "bold 13px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(ttl, plot.x, 6);

      // --- Follow-live / frozen indicator ---
      const live = followLiveRef.current;
      ctx.font = "11px sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "top";
      ctx.fillStyle = live ? "#2a7" : "#d33";
      ctx.fillText(
        live ? "● live" : "⏸ frozen — double-click to resume",
        cssW - PADDING.right,
        8,
      );

      // --- Hover crosshair + tooltip ---
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
    drawRef.current = draw;

    // --- Base time scale + d3-zoom (time axis only) ---
    const baseTimeScale = scaleLinear();
    baseScaleRef.current = baseTimeScale;

    const zoomBehavior: ZoomBehavior<HTMLCanvasElement, unknown> = zoom<
      HTMLCanvasElement,
      unknown
    >()
      .scaleExtent([1, 10000])
      // Exclude shift so shift+wheel / shift+drag fall through to the hand-rolled
      // subcarrier handlers. Keep ctrl+wheel (trackpad pinch) for d3.
      .filter(
        (event: unknown) => {
          const e = event as WheelEvent | MouseEvent | TouchEvent;
          const ctrl = "ctrlKey" in e && e.ctrlKey;
          const shift = "shiftKey" in e && e.shiftKey;
          const button = "button" in e ? e.button : 0;
          return (!ctrl || e.type === "wheel") && !button && !shift;
        },
      )
      .on("zoom", (event: D3ZoomEvent<HTMLCanvasElement, unknown>) => {
        // Programmatic resync: keep the flag quiet (no followLive change, no
        // redraw — the caller draws).
        if (programmaticRef.current) {
          programmaticRef.current = false;
          return;
        }
        const base = baseScaleRef.current;
        const props = propsRef.current;
        if (!base || !props || props.timeSeconds.length === 0 || !viewRef.current) return;
        const [d0, d1] = base.domain();
        if (!(d1 > d0)) return;
        const [visTMin, visTMax] = event.transform.rescaleX(base).domain();
        const clamped = clampTimeWindow(visTMin, visTMax, d0, d1);
        viewRef.current = {
          ...viewRef.current,
          tMin: clamped.tMin,
          tMax: clamped.tMax,
        };
        followLiveRef.current = false;
        hoverRef.current = null;
        draw();
      });
    zoomBehaviorRef.current = zoomBehavior;
    sel.call(zoomBehavior);
    // Disable d3-zoom's default double-click-zoom-in; we use dblclick to reset.
    sel.on("dblclick.zoom", null);

    // --- syncZoomToView: re-sync d3's transform to the current viewRef.
    // Called after data updates and resizes so d3's internal transform stays
    // aligned with the view (which may have moved by follow-live or been
    // frozen while the base scale's domain changed underneath it). ---
    const syncZoomToView = () => {
      const zb = zoomBehaviorRef.current;
      const base = baseScaleRef.current;
      const props = propsRef.current;
      const v = viewRef.current;
      if (!zb || !base || !props || !v || props.timeSeconds.length === 0) return;
      const [d0, d1] = base.domain();
      const [r0, r1] = base.range();
      if (!(d1 > d0) || !(r1 > r0)) return;
      const p1 = base(v.tMin);
      const p2 = base(v.tMax);
      if (!(p2 > p1)) return;
      const k = (r1 - r0) / (p2 - p1);
      const x = r0 - p1 * k;
      const newTransform = zoomIdentity.translate(x, 0).scale(k);
      // Keep the translateExtent pinned to the data extent in base-pixel space
      // so panning can't wander off into empty space beyond the data. Only the X
      // (time) extent is meaningful — d3's Y transform is ignored — so Y is left
      // unbounded to avoid spurious clamping in narrow containers.
      zb.translateExtent([
        [r0, -1e9],
        [r1, 1e9],
      ]);
      programmaticRef.current = true;
      try {
        sel.call(zb.transform, newTransform);
      } finally {
        // If d3 dispatched the event the handler already cleared the flag; if
        // it skipped (transform unchanged) clear it here.
        programmaticRef.current = false;
      }
    };
    syncZoomRef.current = syncZoomToView;

    // --- Double-click: reset to full extent and re-enable follow-live. ---
    const onDoubleClick = () => {
      const props = propsRef.current;
      if (!props || props.timeSeconds.length === 0) return;
      const tMin = props.timeSeconds[0];
      const tMax = props.timeSeconds[props.timeSeconds.length - 1];
      viewRef.current = {
        tMin,
        tMax,
        scMin: -props.halfN,
        scMax: props.halfN - 1,
      };
      followLiveRef.current = true;
      hoverRef.current = null;
      syncZoomToView();
      draw();
    };
    canvas.addEventListener("dblclick", onDoubleClick);

    // --- Subcarrier zoom (shift+wheel), hand-rolled, clamped to the band. ---
    const onShiftWheel = (e: WheelEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();
      const props = propsRef.current;
      const geo = geometryRef.current;
      if (!props || !geo || !viewRef.current) return;
      const v = viewRef.current;
      const rect = canvas.getBoundingClientRect();
      const my = e.clientY - rect.top;
      const scAt = geo.yToSc(my);
      const factor = e.deltaY > 0 ? 1.2 : 1 / 1.2;
      const newRange = (v.scMax - v.scMin) * factor;
      const scRatio = (v.scMax - scAt) / (v.scMax - v.scMin || 1);
      const scMin = scAt - (1 - scRatio) * newRange;
      const scMax = scAt + scRatio * newRange;
      const clamped = clampScWindow(scMin, scMax, -props.halfN, props.halfN - 1);
      viewRef.current = { ...v, scMin: clamped.scMin, scMax: clamped.scMax };
      hoverRef.current = null;
      draw();
    };
    canvas.addEventListener("wheel", onShiftWheel, { passive: false });

    // --- Subcarrier pan (shift+drag), hand-rolled, clamped to the band. ---
    let shiftDragging = false;
    let lastShiftY = 0;
    const onShiftDown = (e: MouseEvent) => {
      if (!e.shiftKey) return;
      shiftDragging = true;
      lastShiftY = e.clientY;
    };
    const onShiftMove = (e: MouseEvent) => {
      if (!shiftDragging) return;
      const props = propsRef.current;
      const geo = geometryRef.current;
      if (!props || !geo || !viewRef.current) return;
      const v = viewRef.current;
      const dy = e.clientY - lastShiftY;
      const dsc = (dy / geo.plot.h) * (v.scMax - v.scMin);
      const clamped = clampScWindow(
        v.scMin + dsc,
        v.scMax + dsc,
        -props.halfN,
        props.halfN - 1,
      );
      viewRef.current = { ...v, scMin: clamped.scMin, scMax: clamped.scMax };
      lastShiftY = e.clientY;
      draw();
    };
    const onShiftUp = () => {
      shiftDragging = false;
    };
    canvas.addEventListener("mousedown", onShiftDown);
    window.addEventListener("mousemove", onShiftMove);
    window.addEventListener("mouseup", onShiftUp);

    // --- Hover crosshair + tooltip (skipped while any button is held, i.e.
    // during a d3 pan or a shift+drag pan). ---
    const onMove = (e: MouseEvent) => {
      if (e.buttons !== 0) return;
      const props = propsRef.current;
      const geo = geometryRef.current;
      if (!props || !geo || !viewRef.current) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      if (
        mx < geo.plot.x ||
        mx > geo.plot.x + geo.plot.w ||
        my < geo.plot.y ||
        my > geo.plot.y + geo.plot.h
      ) {
        if (hoverRef.current) {
          hoverRef.current = null;
          draw();
        }
        return;
      }
      const t = geo.xToT(mx);
      const sc = Math.round(geo.yToSc(my));
      const fi = nearestFrameIndex(props.timeSeconds, t);
      const si = sc + props.halfN;
      if (fi >= 0 && fi < props.matrix.length && si >= 0 && si < props.subcarrierCount) {
        hoverRef.current = { t: props.timeSeconds[fi], sc, v: props.matrix[fi][si] };
      } else {
        hoverRef.current = null;
      }
      draw();
    };
    const onLeave = () => {
      if (hoverRef.current) {
        hoverRef.current = null;
        draw();
      }
    };
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);

    // --- ResizeObserver: redraw when the container's flex layout reflows
    // (sidebar toggle, scrollbar appearing, font load), not just on window
    // resize. Also resync d3 since the plot width changed. ---
    const ro = new ResizeObserver(() => {
      draw();
      syncZoomToView();
    });
    ro.observe(container);

    // Initial render.
    draw();
    syncZoomToView();

    return () => {
      ro.disconnect();
      sel.on(".zoom", null);
      canvas.removeEventListener("dblclick", onDoubleClick);
      canvas.removeEventListener("wheel", onShiftWheel);
      canvas.removeEventListener("mousedown", onShiftDown);
      window.removeEventListener("mousemove", onShiftMove);
      window.removeEventListener("mouseup", onShiftUp);
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
    };
    // Mount once: listeners and d3-zoom are registered exactly once per mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <canvas ref={canvasRef} />
    </div>
  );
}
