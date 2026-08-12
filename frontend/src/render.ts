// Renders pre-aggregated tile grids into an ImageData buffer using a LUT.
// The grid arrives from the backend already aggregated (max-hold for
// amplitude, nearest for phase) and already oriented with row 0 = highest
// subcarrier index — no time search, no aggregation, no row flipping here.

import { NO_DATA_COLOR, NON_FINITE_COLOR, lutIndex } from "./colormap";

/**
 * Source rectangle for blitting the rendered image's subcarrier axis.
 *
 * Row y of the image holds source subcarrier index `n - 1 - y`, whose signed
 * bin is `si - halfN`. Inverting that, bin `sc` lives at row
 * `n - 1 - halfN - sc`. The visible band runs from `scMax` at the top to
 * `scMin` at the bottom, and both ends are inclusive: `scMin` names a row that
 * must be drawn, not the exclusive edge past it. Hence the `+ 1` on the end
 * row -- without it the lowest subcarrier is silently cropped away.
 */
export function subcarrierSourceRect(
  subcarrierCount: number,
  scMin: number,
  scMax: number,
): { srcY: number; srcH: number } {
  const n = subcarrierCount;
  const halfN = Math.floor(n / 2);
  const top = n - 1 - halfN - scMax;
  const bottom = n - 1 - halfN - scMin;

  const srcY = Math.max(0, Math.min(n - 1, top));
  const srcYEnd = Math.max(srcY + 1, Math.min(n, bottom + 1));
  return { srcY, srcH: srcYEnd - srcY };
}

/**
 * Map a tile covering [tile.t0, tile.t1] onto a view showing [view.tMin, view.tMax].
 * Returns the source x-range within the tile and the destination x-range as
 * fractions of the plot width, or null when the two ranges do not overlap.
 *
 * `dx0`/`dx1` are fractions in [0,1] of the plot width; `sx`/`sw` are pixel
 * coordinates in the tile. Partial overlap is clipped on BOTH rects
 * consistently — a tile that covers only the left half of the view draws into
 * only the left half, not stretched across it. This is what lets a stale tile
 * be stretched into place the instant the user zooms, before the fresh tile
 * arrives.
 */
export function tileSourceRect(
  tile: { t0: number; t1: number; width: number },
  view: { tMin: number; tMax: number },
): { sx: number; sw: number; dx0: number; dx1: number } | null {
  const tileSpan = tile.t1 - tile.t0;
  const viewSpan = view.tMax - view.tMin;
  if (!(tileSpan > 0) || !(viewSpan > 0)) return null;

  const overlapT0 = Math.max(tile.t0, view.tMin);
  const overlapT1 = Math.min(tile.t1, view.tMax);
  if (!(overlapT1 > overlapT0)) return null;

  const sx = ((overlapT0 - tile.t0) / tileSpan) * tile.width;
  const sw = ((overlapT1 - overlapT0) / tileSpan) * tile.width;
  const dx0 = (overlapT0 - view.tMin) / viewSpan;
  const dx1 = (overlapT1 - view.tMin) / viewSpan;

  return { sx, sw, dx0, dx1 };
}

export function renderTileToImageData(params: {
  grid: Float32Array;
  width: number;
  height: number;
  lut: Uint32Array;
  min: number;
  max: number;
  target?: ImageData;
}): ImageData {
  const { grid, width, height, lut, min, max, target } = params;
  const lutSize = lut.length;

  let imageData: ImageData;
  if (target && target.width === width && target.height === height) {
    imageData = target;
  } else {
    imageData = new ImageData(width, height);
  }
  const u32 = new Uint32Array(imageData.data.buffer);

  const n = width * height;
  for (let i = 0; i < n; i++) {
    const v = grid[i];
    // NaN = no frames in this column (transparent). -Infinity is real data
    // (db(0)) — opaque black, not missing. The distinction must survive the
    // LUT path because a column of all-db(0) frames is a valid measurement,
    // not a gap.
    if (Number.isNaN(v)) {
      u32[i] = NO_DATA_COLOR;
    } else if (!Number.isFinite(v)) {
      u32[i] = NON_FINITE_COLOR;
    } else {
      const li = lutIndex(v, min, max, lutSize);
      u32[i] = li < 0 ? NON_FINITE_COLOR : lut[li];
    }
  }

  return imageData;
}
