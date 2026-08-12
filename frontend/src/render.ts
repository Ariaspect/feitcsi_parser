// Renders CSI matrix data into an ImageData buffer using a precomputed color LUT.
// The image is built at plot-pixel width (not per-frame width) because timestamps
// are non-uniformly spaced — a uniform-width image would distort the time axis.

import { NO_DATA_COLOR, NON_FINITE_COLOR, lutIndex } from "./colormap";

export interface ColumnView {
  tMin: number;
  tMax: number;
}

export type Aggregation = "max" | "nearest";

/** Binary search: first index i where arr[i] >= x. */
function lowerBound(arr: readonly number[], x: number): number {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

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

export function renderToImageData(params: {
  matrix: number[][];
  timeSeconds: number[];
  subcarrierCount: number;
  view: ColumnView;
  widthPx: number;
  lut: Uint32Array;
  min: number;
  max: number;
  aggregation: Aggregation;
  target?: ImageData;
}): ImageData {
  const {
    matrix,
    timeSeconds,
    subcarrierCount,
    view,
    widthPx,
    lut,
    min,
    max,
    aggregation,
    target,
  } = params;

  const lutSize = lut.length;
  const n = subcarrierCount;

  let imageData: ImageData;
  if (target && target.width === widthPx && target.height === n) {
    imageData = target;
  } else {
    imageData = new ImageData(widthPx, n);
  }
  const u32 = new Uint32Array(imageData.data.buffer);

  const span = view.tMax - view.tMin;

  // Reusable per-column buffer for max-hold.
  const colMax = new Float64Array(n);

  for (let x = 0; x < widthPx; x++) {
    const colStart = view.tMin + (x / widthPx) * span;
    const colEnd = view.tMin + ((x + 1) / widthPx) * span;
    const center = (colStart + colEnd) / 2;
    const isLast = x === widthPx - 1;

    // Binary search for the first frame >= colStart.
    const lo = lowerBound(timeSeconds, colStart);

    // Collect frames in [colStart, colEnd). The last column is inclusive of
    // colEnd (= view.tMax) so the final frame is not dropped by the half-open
    // interval.
    let hi = lo;
    if (isLast) {
      while (hi < timeSeconds.length && timeSeconds[hi] <= colEnd) hi++;
    } else {
      while (hi < timeSeconds.length && timeSeconds[hi] < colEnd) hi++;
    }

    if (lo >= hi) {
      // No frames in this column — fill with transparent NO_DATA_COLOR.
      for (let y = 0; y < n; y++) {
        u32[y * widthPx + x] = NO_DATA_COLOR;
      }
      continue;
    }

    if (aggregation === "max") {
      // Max-hold across frames in the column, per subcarrier.
      colMax.fill(-Infinity);
      for (let fi = lo; fi < hi; fi++) {
        const row = matrix[fi];
        for (let si = 0; si < n; si++) {
          const v = row[si];
          if (Number.isFinite(v) && v > colMax[si]) {
            colMax[si] = v;
          }
        }
      }
      for (let si = 0; si < n; si++) {
        const y = n - 1 - si; // output row 0 = highest subcarrier index (top)
        const cm = colMax[si];
        if (cm === -Infinity) {
          u32[y * widthPx + x] = NON_FINITE_COLOR;
        } else {
          const li = lutIndex(cm, min, max, lutSize);
          u32[y * widthPx + x] = li < 0 ? NON_FINITE_COLOR : lut[li];
        }
      }
    } else {
      // nearest: use the single frame closest to the column center time.
      let bestFi = lo;
      let bestDist = Math.abs(timeSeconds[lo] - center);
      for (let fi = lo + 1; fi < hi; fi++) {
        const d = Math.abs(timeSeconds[fi] - center);
        if (d < bestDist) {
          bestDist = d;
          bestFi = fi;
        }
      }
      const row = matrix[bestFi];
      for (let si = 0; si < n; si++) {
        const y = n - 1 - si; // output row 0 = highest subcarrier index (top)
        const v = row[si];
        if (!Number.isFinite(v)) {
          u32[y * widthPx + x] = NON_FINITE_COLOR;
        } else {
          const li = lutIndex(v, min, max, lutSize);
          u32[y * widthPx + x] = li < 0 ? NON_FINITE_COLOR : lut[li];
        }
      }
    }
  }

  return imageData;
}
