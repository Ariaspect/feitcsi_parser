// Color lookup table and mapping utilities for the heatmap renderer.
// The interpolation math here mirrors the original Heatmap.tsx colorFor() so
// that colors are reproduced exactly, just faster via a precomputed LUT.

const VIRIDIS_STOPS = [
  "#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89",
  "#35b779", "#6ece58", "#b5de2b", "#fde725",
];

export const VIRIDIS: readonly string[] = VIRIDIS_STOPS;

// Cyclic colormap for phase: −π and +π are the same angle, so the colormap
// must start and end on the same color — otherwise every wrap paints a hard
// false edge that looks like a real discontinuity. Sampled from matplotlib's
// twilight; the first and last stops are identical by construction.
const TWILIGHT_STOPS = [
  "#e1d8e2", "#b3c6ce", "#7ba0c2", "#6175ba", "#5d43a4", "#4d176e",
  "#2f1436", "#581547", "#8d2c50", "#b25652", "#c5886c", "#d3bbab",
  "#e1d8e2",
];

export const TWILIGHT: readonly string[] = TWILIGHT_STOPS;

/** Opaque neutral grey — used for pixel columns that contain no frames at all.
 *
 * A gap is a fact about the capture, not a rendering fault, and it has to read
 * that way. Transparent let the canvas background through, which against a
 * dark theme is close enough to the low end of VIRIDIS to pass for a real low
 * value. Neither palette holds an unsaturated mid-grey — VIRIDIS is saturated
 * throughout and TWILIGHT's near-neutrals are tinted and much lighter — so
 * this reads as "no data" against either, and stays distinct from the opaque
 * black of NON_FINITE_COLOR, which is a real db(0) measurement. */
export const NO_DATA_COLOR: number = 0xff6e6e6e; // (a=255, b=0x6e, g=0x6e, r=0x6e) = #6e6e6e

/** Opaque black — used for NaN/Infinity cells, matching the old colorFor("#000000"). */
export const NON_FINITE_COLOR: number = (255 << 24) >>> 0; // (a=255, b=0, g=0, r=0)

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ];
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

const lutCache = new Map<string, Uint32Array>();

/**
 * Build a packed-RGBA LUT (little-endian: (a<<24)|(b<<16)|(g<<8)|r) by linearly
 * interpolating between `stops` in sRGB. Entry i corresponds to normalized
 * position i/(size-1). Default size 256. Cached per (stops, size).
 */
export function buildLut(stops: readonly string[], size: number = 256): Uint32Array {
  const key = stops.join(",") + ":" + size;
  const cached = lutCache.get(key);
  if (cached) return cached;

  const lut = new Uint32Array(size);
  const last = stops.length - 1;

  for (let i = 0; i < size; i++) {
    const t = size <= 1 ? 0 : i / (size - 1);
    const scaled = t * last;
    const idx = Math.floor(scaled);
    const f = scaled - idx;

    let r: number, g: number, b: number;
    if (idx >= last) {
      [r, g, b] = hexToRgb(stops[last]);
    } else {
      const c1 = hexToRgb(stops[idx]);
      const c2 = hexToRgb(stops[idx + 1]);
      r = Math.round(lerp(c1[0], c2[0], f));
      g = Math.round(lerp(c1[1], c2[1], f));
      b = Math.round(lerp(c1[2], c2[2], f));
    }

    lut[i] = ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
  }

  lutCache.set(key, lut);
  return lut;
}

/**
 * Map a data value to an integer LUT index in [0, size-1].
 * Returns -1 for non-finite input. Clamps to [0,1] before mapping.
 */
export function lutIndex(value: number, min: number, max: number, size: number = 256): number {
  if (!Number.isFinite(value)) return -1;
  const t = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  const idx = Math.round(t * (size - 1));
  return Math.max(0, Math.min(size - 1, idx));
}
