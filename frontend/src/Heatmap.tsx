import { useEffect, useRef } from "react";
import { select } from "d3-selection";
import { zoom, zoomIdentity, type D3ZoomEvent, type ZoomBehavior } from "d3-zoom";
import { scaleLinear, type ScaleLinear } from "d3-scale";
import { VIRIDIS, buildLut } from "./colormap";
import { renderTileToImageData, subcarrierSourceRect, tileSourceRect } from "./render";
import { fetchTile, type Tile, type Metric } from "./api";
import { type TimeLink } from "./timelink";
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
  path: string;
  metric: Metric;
  filename: string;
  numSubcarriers: number;
  captureTMin: number;
  captureTMax: number;
  /** Fixed color scale bounds, for a metric whose range is known a priori --
   * phase is always [-π, π]. Omit for amplitude, whose range is a property of
   * the capture: the scale then locks to the first tile (see ampScaleRef),
   * falling back to the current tile's own range until it does. */
  minValue?: number;
  maxValue?: number;
  title: string;
  colorLabel: string;
  palette?: readonly string[];
  height?: number;
  /** Link between stacked heatmaps' time axes. When the user zooms or resets
   * one plot, the other mirrors the time window (but not the subcarrier band).
   * Omit for a standalone heatmap. */
  timeLink?: TimeLink;
  /** MIMO filter passed to /api/tile. 'all' or 'NxM'. */
  mimo?: string | null;
  /** Source MAC filter passed to /api/tile. 'all' or a MAC string. */
  sourceMac?: string | null;
  /** Dark mode: canvas text/axes/crosshair colors adapt. */
  dark?: boolean;
}

const PADDING = { top: 30, right: 70, bottom: 40, left: 60 };

/** True when the color scale must be fitted to the data instead of being
 * given a priori. Wrapped phase knows its range is [-π, π] and passes both
 * bounds; amplitude and the unwrapped phase views do not, and lock to their
 * first tile's percentile band instead. Deriving this from the bounds rather
 * than from a metric name list means a new metric gets the right behaviour by
 * saying what it knows about its own range, not by being added here. */
function autoScales(props: { minValue?: number; maxValue?: number }): boolean {
  return props.minValue === undefined && props.maxValue === undefined;
}

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
  path: string;
  metric: Metric;
  filename: string;
  numSubcarriers: number;
  halfN: number;
  captureTMin: number;
  captureTMax: number;
  minValue?: number;
  maxValue?: number;
  title: string;
  colorLabel: string;
  palette: readonly string[];
  height: number;
  timeLink?: TimeLink;
  mimo?: string | null;
  sourceMac?: string | null;
  dark?: boolean;
}

interface TileEntry {
  tile: Tile;
  seq: number;
}

/** Fetch key — if none of these changed, the current tile is still valid. */
interface FetchKey {
  path: string;
  metric: string;
  t0: number;
  t1: number;
  width: number;
  mimo?: string | null;
  sourceMac?: string | null;
}

export function Heatmap({
  path,
  metric,
  filename,
  numSubcarriers,
  captureTMin,
  captureTMax,
  minValue,
  maxValue,
  title,
  colorLabel,
  palette = VIRIDIS,
  height = 400,
  timeLink,
  mimo,
  sourceMac,
  dark,
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

  // Stable identity for TimeLink: a subscriber ignores publishes that carry
  // its own source so a link-driven d3 resync is not mistaken for user input.
  const sourceRef = useRef({});

  // --- Tile fetch state ---
  const tileRef = useRef<TileEntry | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const seqRef = useRef(0);
  const debounceTimerRef = useRef<number | null>(null);
  const lastFetchKeyRef = useRef<FetchKey | null>(null);
  const requestTileRef = useRef<(() => void) | null>(null);
  // Color scale for metrics without a-priori bounds (amplitude, and the
  // unwrapped phase views), locked to the first tile's percentile bounds
  // (1st/99th) for a given capture. The raw min/max is dominated by outliers
  // — 98.5% of values fall in [40, 60] while extrema span [7.7, 84.7] — so a
  // min/max scale compresses the visible structure into one narrow slice of
  // the colormap. Percentile bounds expose that structure. Values outside the
  // band clamp to the end colors via lutIndex. Cleared on capture identity
  // change (see Effect 1's reset path).
  const ampScaleRef = useRef<{ lo: number; hi: number } | null>(null);
  // Last metric drawn, so a metric switch can drop scale and tile state that
  // was fitted to the previous one. Null until the first render completes.
  const metricRef = useRef<Metric | null>(null);

  const halfN = Math.floor(numSubcarriers / 2);

  // Mirror latest props into a ref on every render so the stable draw/listen
  // path always sees fresh data without re-subscribing.
  propsRef.current = {
    path,
    metric,
    filename,
    numSubcarriers,
    halfN,
    captureTMin,
    captureTMax,
    minValue,
    maxValue,
    title,
    colorLabel,
    palette,
    height,
    timeLink,
    mimo,
    sourceMac,
    dark,
  };

  // -------------------------------------------------------------------------
  // Effect 1 (runs every render): view lifecycle + cheap redraw + tile request.
  // Decides whether to reset, slide (follow-live), or freeze the view, then
  // redraws and re-syncs the d3 transform. Does NOT register DOM listeners.
  // Declared before Effect 2 so on the first mount it runs first and the view
  // is set before the mount effect's initial draw.
  // -------------------------------------------------------------------------
  useEffect(() => {
    const props = propsRef.current;
    if (!props) return;
    if (props.numSubcarriers === 0 || !(props.captureTMax > props.captureTMin)) {
      // Empty capture: show the empty state without touching the view.
      drawRef.current?.();
      return;
    }

    const newCapture: CaptureId = {
      filename: props.filename,
      numSubcarriers: props.numSubcarriers,
      tMin: props.captureTMin,
      tMax: props.captureTMax,
      mimo: props.mimo,
      sourceMac: props.sourceMac,
    };

    // A metric switch (e.g. the detrend toggle) keeps the capture and the
    // view, but the locked color scale and the on-screen tile belong to the
    // old metric — unwrapped phase spans tens of radians where detrended
    // spans a few, so reusing the lock would paint the new data as one flat
    // color. The view deliberately survives: the point of the toggle is to
    // compare the same window both ways.
    if (metricRef.current !== props.metric) {
      metricRef.current = props.metric;
      ampScaleRef.current = null;
      tileRef.current = null;
      imageDataRef.current = null;
      lastFetchKeyRef.current = null;
    }

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
      // Stale tile and locked scale belong to the old capture — drop both so
      // the new capture's first tile re-locks the scale and doesn't blit
      // foreign data through tileSourceRect.
      ampScaleRef.current = null;
      tileRef.current = null;
      imageDataRef.current = null;
      lastFetchKeyRef.current = null;
    } else if (viewRef.current) {
      // Same capture: slide if live, freeze if not. advanceView returns the
      // same reference when frozen, so a frozen view is bit-identical across
      // any number of polls.
      viewRef.current = advanceView(
        viewRef.current,
        { tMin: props.captureTMin, tMax: props.captureTMax },
        followLiveRef.current,
      );
    }
    captureRef.current = newCapture;

    // The base time scale's domain must reflect the current data extent before
    // we resync the d3 transform, otherwise rescaleX would map to stale data.
    if (baseScaleRef.current) {
      baseScaleRef.current.domain([props.captureTMin, props.captureTMax]);
    }

    drawRef.current?.();
    syncZoomRef.current?.();
    requestTileRef.current?.();
  });

  // -------------------------------------------------------------------------
  // Effect 2 (mount once): register DOM listeners, ResizeObserver, d3-zoom,
  // and the tile fetch loop. Defines draw(), syncZoomToView(), and
  // requestTile(). All handlers read from refs so they never close over a
  // stale render's geometry or view.
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

      const c = props.dark
        ? { text: "#e5e7eb", muted: "#9ca3af", border: "#3f3f46", crosshair: "#1a1a1a", tooltipBg: "rgba(255,255,255,0.9)", tooltipText: "#0a0a0a" }
        : { text: "#000000", muted: "#888888", border: "#888888", crosshair: "#ffffff", tooltipBg: "rgba(0,0,0,0.85)", tooltipText: "#ffffff" };

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
      const tileEntry = tileRef.current;

      if (!view || props.numSubcarriers === 0 || !(view.tMax > view.tMin)) {
        ctx.fillStyle = c.muted;
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

      // Publish live geometry for the mouse handlers -- and for the tile fetch,
      // which reads plot.w to size its request. Geometry is a function of the
      // view and the canvas, NOT of the data, so it must be published before
      // the no-tile bail below. Nulling it there instead deadlocks the whole
      // component: no tile means no geometry, no geometry means the fetch
      // returns early, and the tile that would break the cycle never arrives.
      geometryRef.current = { plot, tToX, xToT, scToY, yToSc };

      // Keep the base time scale in sync: domain from the data extent, range
      // from the current plot width. d3-zoom rescales this to derive the
      // visible window.
      if (baseScaleRef.current) {
        baseScaleRef.current
          .domain([props.captureTMin, props.captureTMax])
          .range([0, plot.w]);
      }

      if (!tileEntry) {
        // Geometry is live, the first tile is still in flight. Frame the plot
        // so the layout doesn't jump when it lands.
        ctx.strokeStyle = c.text;
        ctx.lineWidth = 1;
        ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);
        ctx.fillStyle = c.muted;
        ctx.font = "13px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("Loading…", plot.x + plot.w / 2, plot.y + plot.h / 2);
        ctx.fillStyle = c.text;
        ctx.font = "bold 13px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(props.title, plot.x, 6);
        return;
      }

      const tile = tileEntry.tile;

      // --- Color scale ---
      // A metric with an a-priori range (phase) uses it and never moves.
      // Amplitude's range is a property of the capture, so it locks to the
      // first tile's percentile bounds (1st/99th) and stays there -- the raw
      // min/max is outlier-dominated and compresses the visible structure.
      // Rescaling per window would also make identical data look different at
      // different zoom levels.
      //
      // The fallback is the current tile's own vmin/vmax, not a sentinel. A
      // degenerate first tile (pHigh === pLow AND vmax === vmin) leaves the
      // lock unset, and NaN bounds would send lutIndex to NaN, index the LUT
      // with it, and paint the entire plot transparent with "NaN" on the
      // colorbar -- a blank screen with nothing to explain it.
      const locked = autoScales(props) ? ampScaleRef.current : null;
      const min = locked ? locked.lo : props.minValue ?? tile.vmin;
      const max = locked ? locked.hi : props.maxValue ?? tile.vmax;

      // --- Heatmap data via tile + LUT ---
      const lut = buildLut(props.palette, 256);
      const imageData = renderTileToImageData({
        grid: tile.grid,
        width: tile.width,
        height: tile.height,
        lut,
        min,
        max,
        target: imageDataRef.current ?? undefined,
      });
      imageDataRef.current = imageData;

      if (!offscreenRef.current) {
        offscreenRef.current = document.createElement("canvas");
      }
      const offscreen = offscreenRef.current;
      if (offscreen.width !== tile.width || offscreen.height !== tile.height) {
        offscreen.width = tile.width;
        offscreen.height = tile.height;
      }
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) return;
      offCtx.putImageData(imageData, 0, 0);

      // Blit the tile through tileSourceRect (x-axis) combined with
      // subcarrierSourceRect (y-axis) in a single drawImage. tileSourceRect
      // maps the tile's time range onto the view's, so a stale tile from the
      // last fetch is stretched into place instantly and sharpens when the
      // fresh one arrives.
      const srcRect = tileSourceRect(
        { t0: tile.t0, t1: tile.t1, width: tile.width },
        { tMin: view.tMin, tMax: view.tMax },
      );
      if (srcRect) {
        const { srcY, srcH } = subcarrierSourceRect(
          props.numSubcarriers,
          view.scMin,
          view.scMax,
        );
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(
          offscreen,
          srcRect.sx, srcY, srcRect.sw, srcH,
          plot.x + srcRect.dx0 * plot.w, plot.y,
          (srcRect.dx1 - srcRect.dx0) * plot.w, plot.h,
        );
      }

      // --- Frame ---
      ctx.strokeStyle = c.text;
      ctx.lineWidth = 1;
      ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);

      // --- Time axis ticks via d3-scale (nice, stable round numbers) ---
      const tScale = scaleLinear()
        .domain([view.tMin, view.tMax])
        .range([plot.x, plot.x + plot.w]);
      const tFmt = tScale.tickFormat(6);
      ctx.fillStyle = c.text;
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
      for (let i = 0; i < props.palette.length; i++) {
        grad.addColorStop(i / (props.palette.length - 1), props.palette[i]);
      }
      ctx.fillStyle = grad;
      ctx.fillRect(barX, plot.y, barW, plot.h);
      ctx.strokeStyle = c.border;
      ctx.strokeRect(barX, plot.y, barW, plot.h);
      ctx.fillStyle = c.text;
      ctx.font = "10px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(max.toFixed(1), barX + barW + 4, plot.y + 4);
      ctx.fillText(min.toFixed(1), barX + barW + 4, plot.y + plot.h - 4);
      ctx.save();
      ctx.translate(barX + barW + 30, plot.y + plot.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText(props.colorLabel, 0, 0);
      ctx.restore();

      // --- Title ---
      ctx.fillStyle = c.text;
      ctx.font = "bold 13px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(props.title, plot.x, 6);

      // --- Follow-live / frozen + stride-sampled indicator ---
      // exact === false means the tile was stride-sampled because the range
      // exceeded the decode budget. A sampled max-hold can miss transients;
      // silence would be a lie, so the indicator says so.
      // anchored === false means the ratio was NOT corrected: correction
      // needs one transmitter's own frame sequence, and on `all` consecutive
      // frames come from different senders, so it is skipped rather than
      // applied at ~1/20th the effect and reported as done.
      const live = followLiveRef.current;
      const sampled = tileEntry.tile.exact === false;
      const uncorrected = tileEntry.tile.anchored === false;
      ctx.font = "11px sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "top";
      let indicator = live ? "● live" : "⏸ frozen — double-click to resume";
      if (sampled) {
        indicator += "  ⚠ sampled — zoom in for exact";
      }
      if (uncorrected) {
        indicator += "  ⚠ uncorrected — select a transmitter";
      }
      ctx.fillStyle =
        uncorrected || sampled ? "#d80" : live ? "#2a7" : "#d33";
      ctx.fillText(indicator, cssW - PADDING.right, 8);

      // --- Hover crosshair + tooltip ---
      const hover = hoverRef.current;
      if (hover) {
        const x = tToX(hover.t);
        const y = scToY(hover.sc);
        ctx.strokeStyle = c.crosshair;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, plot.y);
        ctx.lineTo(x, plot.y + plot.h);
        ctx.moveTo(plot.x, y);
        ctx.lineTo(plot.x + plot.w, y);
        ctx.stroke();
        ctx.fillStyle = c.tooltipBg;
        const txt = `t=${hover.t.toFixed(3)}s  sc=${hover.sc}  v=${hover.v.toFixed(2)}`;
        ctx.font = "11px monospace";
        const tw = ctx.measureText(txt).width;
        ctx.fillRect(x + 8, y - 22, tw + 12, 18);
        ctx.fillStyle = c.tooltipText;
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
        if (!base || !props || props.numSubcarriers === 0 || !viewRef.current) return;
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
        requestTileRef.current?.();
        // Broadcast the time window to linked plots. Subcarrier zoom stays
        // per-plot and is not part of the message.
        const link = props.timeLink;
        if (link) {
          link.publish(sourceRef.current, { tMin: clamped.tMin, tMax: clamped.tMax }, false);
        }
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
      if (!zb || !base || !props || !v || props.numSubcarriers === 0) return;
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

    // --- Tile fetch: trailing-debounced, one AbortController per request,
    // monotonically sequenced so stale responses are dropped. ---
    const doFetchTile = async () => {
      const props = propsRef.current;
      const view = viewRef.current;
      const geo = geometryRef.current;
      if (!props || !view || !geo) return;
      if (props.numSubcarriers === 0 || !(view.tMax > view.tMin)) return;

      const t0 = view.tMin;
      const t1 = view.tMax;
      // Requested width = the plot's pixel width, so the server never returns
      // more columns than there are pixels to display them.
      const width = Math.max(1, Math.round(geo.plot.w));

      // Skip if nothing changed since the last fetch — a subcarrier-only
      // change (shift+wheel, shift+drag) reaches here via Effect 1 but has
      // the same time window, so it correctly does not refetch.
      const lastKey = lastFetchKeyRef.current;
      if (
        lastKey &&
        lastKey.path === props.path &&
        lastKey.metric === props.metric &&
        lastKey.t0 === t0 &&
        lastKey.t1 === t1 &&
        lastKey.width === width &&
        lastKey.mimo === props.mimo &&
        lastKey.sourceMac === props.sourceMac
      ) {
        return;
      }
      lastFetchKeyRef.current = {
        path: props.path,
        metric: props.metric,
        t0,
        t1,
        width,
        mimo: props.mimo,
        sourceMac: props.sourceMac,
      };

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      const seq = ++seqRef.current;

      try {
        const tile = await fetchTile(
          props.path,
          t0,
          t1,
          width,
          props.metric,
          controller.signal,
          props.mimo,
          props.sourceMac,
        );
        // Drop stale responses — a newer request may have been issued while
        // this one was in flight. Aborting alone is not sufficient: the fetch
        // can still resolve between abort() and the abort taking effect.
        if (seq < seqRef.current) return;
        if (controller.signal.aborted) return;

        // Lock amplitude scale to the first tile with a meaningful range.
        // Prefer percentile bounds (robust to outliers); fall back to extrema
        // if the percentile band is degenerate (pHigh === pLow); leave unset
        // if both are degenerate — draw() then uses the current tile's vmin/vmax.
        if (autoScales(props) && ampScaleRef.current === null) {
          if (tile.pHigh > tile.pLow) {
            ampScaleRef.current = { lo: tile.pLow, hi: tile.pHigh };
          } else if (tile.vmax > tile.vmin) {
            ampScaleRef.current = { lo: tile.vmin, hi: tile.vmax };
          }
        }

        tileRef.current = { tile, seq };
        draw();
      } catch (e) {
        // AbortError is normal control flow during zoom/pan — the user moved
        // on before the previous request finished. Swallow it silently.
        if (e instanceof Error && e.name === "AbortError") return;
        console.error("tile fetch failed:", e);
      }
    };

    const scheduleTileFetch = () => {
      if (debounceTimerRef.current !== null) {
        clearTimeout(debounceTimerRef.current);
      }
      debounceTimerRef.current = window.setTimeout(() => {
        debounceTimerRef.current = null;
        void doFetchTile();
      }, 100);
    };
    requestTileRef.current = scheduleTileFetch;

    // --- Double-click: reset to full extent and re-enable follow-live. ---
    const onDoubleClick = () => {
      const props = propsRef.current;
      if (!props || props.numSubcarriers === 0) return;
      if (!(props.captureTMax > props.captureTMin)) return;
      viewRef.current = {
        tMin: props.captureTMin,
        tMax: props.captureTMax,
        scMin: -props.halfN,
        scMax: props.halfN - 1,
      };
      followLiveRef.current = true;
      hoverRef.current = null;
      syncZoomToView();
      draw();
      scheduleTileFetch();
      const link = props.timeLink;
      if (link) {
        link.publish(
          sourceRef.current,
          { tMin: props.captureTMin, tMax: props.captureTMax },
          true,
        );
      }
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
      // Subcarrier-only change — no tile refetch.
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
    // during a d3 pan or a shift+drag pan). Reads the value straight from the
    // tile grid at the hovered pixel — no frame index search needed. ---
    const onMove = (e: MouseEvent) => {
      if (e.buttons !== 0) return;
      const props = propsRef.current;
      const geo = geometryRef.current;
      const tileEntry = tileRef.current;
      if (!props || !geo || !viewRef.current || !tileEntry) return;
      const tile = tileEntry.tile;
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

      // Map t to a tile column. If t is outside the tile's coverage (the view
      // was panned beyond the last tile), there is no value to show.
      const tileSpan = tile.t1 - tile.t0;
      let col: number;
      if (tileSpan > 0 && t >= tile.t0 && t <= tile.t1) {
        col = Math.floor(((t - tile.t0) / tileSpan) * tile.width);
        col = Math.max(0, Math.min(tile.width - 1, col));
      } else {
        if (hoverRef.current) {
          hoverRef.current = null;
          draw();
        }
        return;
      }

      // Map signed bin sc to a tile row. Row 0 = highest subcarrier index;
      // bin sc lives at row n - 1 - halfN - sc (mirrors subcarrierSourceRect).
      const nSc = props.numSubcarriers;
      const row = nSc - 1 - props.halfN - sc;
      if (row < 0 || row >= nSc) {
        hoverRef.current = null;
      } else {
        const v = tile.grid[row * tile.width + col];
        hoverRef.current = { t, sc, v };
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
    // resize. Also resync d3 and refetch since the plot width changed. ---
    const ro = new ResizeObserver(() => {
      draw();
      syncZoomToView();
      scheduleTileFetch();
    });
    ro.observe(container);

    // --- TimeLink: mirror the time window across stacked heatmaps. ---
    // Subscribes once on mount. On receiving a window from another plot,
    // update only the time half of the view (subcarrier zoom is per-plot),
    // mirror followLive, resync d3, redraw, and refetch. The programmaticRef
    // guard inside syncZoomToView prevents the d3 resync from re-entering the
    // zoom handler and re-publishing — an infinite loop otherwise.
    const link = propsRef.current?.timeLink;
    let unsubTimeLink: (() => void) | null = null;
    if (link) {
      unsubTimeLink = link.subscribe((w, followLive, source) => {
        if (source === sourceRef.current) return; // ignore own publish
        const v = viewRef.current;
        const p = propsRef.current;
        if (!v || !p || p.numSubcarriers === 0) return;
        viewRef.current = { ...v, tMin: w.tMin, tMax: w.tMax };
        followLiveRef.current = followLive;
        hoverRef.current = null;
        syncZoomToView();
        draw();
        scheduleTileFetch();
      });
    }

    // Initial render.
    draw();
    syncZoomToView();
    scheduleTileFetch();

    return () => {
      ro.disconnect();
      unsubTimeLink?.();
      sel.on(".zoom", null);
      canvas.removeEventListener("dblclick", onDoubleClick);
      canvas.removeEventListener("wheel", onShiftWheel);
      canvas.removeEventListener("mousedown", onShiftDown);
      window.removeEventListener("mousemove", onShiftMove);
      window.removeEventListener("mouseup", onShiftUp);
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
      if (debounceTimerRef.current !== null) {
        clearTimeout(debounceTimerRef.current);
      }
      abortRef.current?.abort();
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
