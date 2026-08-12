import { describe, it, expect } from "vitest";
import { buildLut, lutIndex, NO_DATA_COLOR, NON_FINITE_COLOR } from "./colormap";
import { renderToImageData, subcarrierSourceRect } from "./render";

/** Minimal ImageData shim for the node test environment. */
function makeImageData(width: number, height: number): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  return { width, height, data } as unknown as ImageData;
}

function pixel(u32: Uint32Array, x: number, y: number, widthPx: number): number {
  return u32[y * widthPx + x];
}

const STOPS = ["#000000", "#ffffff"];

describe("renderToImageData — row order", () => {
  it("last source subcarrier lands in output row 0", () => {
    const N = 4;
    // subcarrier 0 = value 0 (low), subcarrier 3 = value 100 (high)
    const matrix = [[0, 0, 0, 100]];
    const timeSeconds = [0];
    const view = { tMin: 0, tMax: 10 };
    const widthPx = 10;
    const lut = buildLut(STOPS, 256);
    const target = makeImageData(widthPx, N);
    renderToImageData({
      matrix, timeSeconds, subcarrierCount: N, view, widthPx,
      lut, min: 0, max: 100, aggregation: "max", target,
    });
    const u32 = new Uint32Array(target.data.buffer);
    // col 0 contains the frame (t=0 ∈ [0,1))
    const topRow = pixel(u32, 0, 0, widthPx);        // si = N-1 = 3 → value 100
    const bottomRow = pixel(u32, 0, N - 1, widthPx);  // si = 0 → value 0
    expect(topRow).toBe(lut[lutIndex(100, 0, 100, 256)]);
    expect(bottomRow).toBe(lut[lutIndex(0, 0, 100, 256)]);
  });
});

describe("renderToImageData — column time mapping", () => {
  it("frames at t=0,5,10 land in expected columns", () => {
    const N = 1;
    const matrix = [[100], [50], [0]];
    const timeSeconds = [0, 5, 10];
    const view = { tMin: 0, tMax: 10 };
    const widthPx = 10;
    const lut = buildLut(STOPS, 256);
    const target = makeImageData(widthPx, N);
    renderToImageData({
      matrix, timeSeconds, subcarrierCount: N, view, widthPx,
      lut, min: 0, max: 100, aggregation: "nearest", target,
    });
    const u32 = new Uint32Array(target.data.buffer);
    // col 0 = [0,1) → t=0 → value 100
    // col 5 = [5,6) → t=5 → value 50
    // col 9 (last, inclusive of colEnd) → t=10 → value 0
    expect(pixel(u32, 0, 0, widthPx)).toBe(lut[lutIndex(100, 0, 100, 256)]);
    expect(pixel(u32, 5, 0, widthPx)).toBe(lut[lutIndex(50, 0, 100, 256)]);
    expect(pixel(u32, 9, 0, widthPx)).toBe(lut[lutIndex(0, 0, 100, 256)]);
  });
});

describe("renderToImageData — gap handling", () => {
  it("columns with no frames get NO_DATA_COLOR", () => {
    const N = 1;
    const matrix = [[50]];
    const timeSeconds = [0];
    const view = { tMin: 0, tMax: 10 };
    const widthPx = 10;
    const lut = buildLut(STOPS, 256);
    const target = makeImageData(widthPx, N);
    renderToImageData({
      matrix, timeSeconds, subcarrierCount: N, view, widthPx,
      lut, min: 0, max: 100, aggregation: "max", target,
    });
    const u32 = new Uint32Array(target.data.buffer);
    expect(pixel(u32, 0, 0, widthPx)).not.toBe(NO_DATA_COLOR);
    for (let x = 1; x < widthPx; x++) {
      expect(pixel(u32, x, 0, widthPx)).toBe(NO_DATA_COLOR);
    }
  });
});

describe("renderToImageData — max aggregation", () => {
  it("uses the per-subcarrier maximum across frames in a column", () => {
    const N = 1;
    // both frames in col 0 = [0, 1)
    const matrix = [[30], [70]];
    const timeSeconds = [0.4, 0.6];
    const view = { tMin: 0, tMax: 10 };
    const widthPx = 10;
    const lut = buildLut(STOPS, 256);
    const target = makeImageData(widthPx, N);
    renderToImageData({
      matrix, timeSeconds, subcarrierCount: N, view, widthPx,
      lut, min: 0, max: 100, aggregation: "max", target,
    });
    const u32 = new Uint32Array(target.data.buffer);
    expect(pixel(u32, 0, 0, widthPx)).toBe(lut[lutIndex(70, 0, 100, 256)]);
  });
});

describe("renderToImageData — nearest aggregation", () => {
  it("the frame closest to column center wins", () => {
    const N = 1;
    // col 0 = [0, 1), center = 0.5
    // t=0.3 (dist 0.2, value 30) vs t=0.6 (dist 0.1, value 70) → 70 wins
    const matrix = [[30], [70]];
    const timeSeconds = [0.3, 0.6];
    const view = { tMin: 0, tMax: 10 };
    const widthPx = 10;
    const lut = buildLut(STOPS, 256);
    const target = makeImageData(widthPx, N);
    renderToImageData({
      matrix, timeSeconds, subcarrierCount: N, view, widthPx,
      lut, min: 0, max: 100, aggregation: "nearest", target,
    });
    const u32 = new Uint32Array(target.data.buffer);
    expect(pixel(u32, 0, 0, widthPx)).toBe(lut[lutIndex(70, 0, 100, 256)]);
  });
});

describe("renderToImageData — non-finite input", () => {
  it("NaN maps to NON_FINITE_COLOR", () => {
    const N = 1;
    const matrix = [[NaN]];
    const timeSeconds = [0];
    const view = { tMin: 0, tMax: 10 };
    const widthPx = 10;
    const lut = buildLut(STOPS, 256);
    const target = makeImageData(widthPx, N);
    renderToImageData({
      matrix, timeSeconds, subcarrierCount: N, view, widthPx,
      lut, min: 0, max: 100, aggregation: "max", target,
    });
    const u32 = new Uint32Array(target.data.buffer);
    expect(pixel(u32, 0, 0, widthPx)).toBe(NON_FINITE_COLOR);
  });

  it("all-non-finite column in max aggregation maps to NON_FINITE_COLOR", () => {
    const N = 1;
    const matrix = [[Infinity], [NaN]];
    const timeSeconds = [0.4, 0.6];
    const view = { tMin: 0, tMax: 10 };
    const widthPx = 10;
    const lut = buildLut(STOPS, 256);
    const target = makeImageData(widthPx, N);
    renderToImageData({
      matrix, timeSeconds, subcarrierCount: N, view, widthPx,
      lut, min: 0, max: 100, aggregation: "max", target,
    });
    const u32 = new Uint32Array(target.data.buffer);
    expect(pixel(u32, 0, 0, widthPx)).toBe(NON_FINITE_COLOR);
  });
});

describe("subcarrierSourceRect", () => {
  it("covers every row at the default view of an even-width capture", () => {
    // The default view is scMin = -halfN, scMax = halfN - 1. For even
    // subcarrier counts -- 242 on every real capture -- that names the whole
    // band. Regression guard: an exclusive end row cropped the lowest
    // subcarrier, so only 241 of 242 rows ever reached the canvas.
    for (const n of [2, 8, 64, 242]) {
      const halfN = Math.floor(n / 2);
      const { srcY, srcH } = subcarrierSourceRect(n, -halfN, halfN - 1);
      expect(srcY).toBe(0);
      expect(srcH).toBe(n);
    }
  });

  it("returns one row per bin in the requested band", () => {
    // srcH must equal the inclusive bin count, so the blit neither drops a
    // row nor stretches a phantom one across the plot.
    const n = 242;
    for (const [lo, hi] of [[-121, 120], [-10, 10], [0, 0], [-121, -121], [120, 120]]) {
      expect(subcarrierSourceRect(n, lo, hi).srcH).toBe(hi - lo + 1);
    }
  });

  it("maps a zoomed band to its inclusive row range", () => {
    const n = 242; // halfN = 121, so bin sc sits at row 120 - sc
    expect(subcarrierSourceRect(n, 0, 10)).toEqual({ srcY: 110, srcH: 11 });
    expect(subcarrierSourceRect(n, -5, -5)).toEqual({ srcY: 125, srcH: 1 });
  });

  it("clamps a view wider than the data without inverting", () => {
    const n = 242;
    const { srcY, srcH } = subcarrierSourceRect(n, -500, 500);
    expect(srcY).toBe(0);
    expect(srcH).toBe(n);
  });

  it("never returns an empty rectangle", () => {
    const n = 242;
    for (const [lo, hi] of [[400, 500], [-500, -400], [10, 5]]) {
      const { srcY, srcH } = subcarrierSourceRect(n, lo, hi);
      expect(srcH).toBeGreaterThanOrEqual(1);
      expect(srcY + srcH).toBeLessThanOrEqual(n);
    }
  });
});
