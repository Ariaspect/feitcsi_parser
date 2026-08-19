import { describe, it, expect } from "vitest";
import { buildLut, lutIndex, NO_DATA_COLOR, NON_FINITE_COLOR } from "./colormap";
import {
  renderTileToImageData,
  subcarrierSourceRect,
  tileSourceRect,
} from "./render";

// Shim ImageData for the node test environment. The browser has this as a
// global; node does not unless jsdom is configured.
if (typeof globalThis.ImageData === "undefined") {
  globalThis.ImageData = class {
    width: number;
    height: number;
    data: Uint8ClampedArray;
    constructor(width: number, height: number) {
      this.width = width;
      this.height = height;
      this.data = new Uint8ClampedArray(width * height * 4);
    }
  } as unknown as typeof ImageData;
}

const STOPS = ["#000000", "#ffffff"];

// ---------------------------------------------------------------------------
// tileSourceRect
// ---------------------------------------------------------------------------

describe("tileSourceRect", () => {
  it("exact match: tile spans the whole view", () => {
    const r = tileSourceRect(
      { t0: 0, t1: 10, width: 100 },
      { tMin: 0, tMax: 10 },
    );
    expect(r).toEqual({ sx: 0, sw: 100, dx0: 0, dx1: 1 });
  });

  it("zoomed in: tile is wider than the view (source subset, full destination)", () => {
    // Tile covers [0, 10] in 100 px; view is [2, 8].
    // Overlap [2, 8] → source [20, 80], destination [0, 1].
    const r = tileSourceRect(
      { t0: 0, t1: 10, width: 100 },
      { tMin: 2, tMax: 8 },
    );
    expect(r).toEqual({ sx: 20, sw: 60, dx0: 0, dx1: 1 });
  });

  it("zoomed out: tile is narrower than the view (full source, destination subset)", () => {
    // Tile covers [2, 8] in 60 px; view is [0, 10].
    // Overlap [2, 8] → source [0, 60], destination [0.2, 0.8].
    const r = tileSourceRect(
      { t0: 2, t1: 8, width: 60 },
      { tMin: 0, tMax: 10 },
    );
    expect(r).toEqual({ sx: 0, sw: 60, dx0: 0.2, dx1: 0.8 });
  });

  it("panned: partial overlap on the left (tile covers left part of view)", () => {
    // Tile covers [0, 5]; view is [0, 10]. Overlap [0, 5] → left half.
    const r = tileSourceRect(
      { t0: 0, t1: 5, width: 50 },
      { tMin: 0, tMax: 10 },
    );
    expect(r).toEqual({ sx: 0, sw: 50, dx0: 0, dx1: 0.5 });
  });

  it("panned: partial overlap on the right (tile covers right part of view)", () => {
    // Tile covers [5, 10]; view is [0, 10]. Overlap [5, 10] → right half.
    const r = tileSourceRect(
      { t0: 5, t1: 10, width: 50 },
      { tMin: 0, tMax: 10 },
    );
    expect(r).toEqual({ sx: 0, sw: 50, dx0: 0.5, dx1: 1 });
  });

  it("no overlap: tile entirely before view → null", () => {
    expect(
      tileSourceRect({ t0: 0, t1: 5, width: 50 }, { tMin: 6, tMax: 10 }),
    ).toBeNull();
  });

  it("no overlap: tile entirely after view → null", () => {
    expect(
      tileSourceRect({ t0: 6, t1: 10, width: 50 }, { tMin: 0, tMax: 5 }),
    ).toBeNull();
  });

  it("degenerate tile span (t0 === t1) → null", () => {
    expect(
      tileSourceRect({ t0: 5, t1: 5, width: 10 }, { tMin: 0, tMax: 10 }),
    ).toBeNull();
  });

  it("degenerate view span (tMin === tMax) → null", () => {
    expect(
      tileSourceRect({ t0: 0, t1: 10, width: 100 }, { tMin: 5, tMax: 5 }),
    ).toBeNull();
  });

  it("edge touch (overlapT0 === overlapT1) → null", () => {
    // Tile ends exactly where the view starts — zero-width overlap.
    expect(
      tileSourceRect({ t0: 0, t1: 5, width: 50 }, { tMin: 5, tMax: 10 }),
    ).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// renderTileToImageData
// ---------------------------------------------------------------------------

describe("renderTileToImageData — LUT mapping", () => {
  it("finite values map to the correct LUT entries", () => {
    const W = 2;
    const H = 1;
    const grid = new Float32Array([0, 100]);
    const lut = buildLut(STOPS, 256);
    const target = new ImageData(W, H);
    renderTileToImageData({ grid, width: W, height: H, lut, min: 0, max: 100, target });
    const u32 = new Uint32Array(target.data.buffer);
    expect(u32[0]).toBe(lut[lutIndex(0, 0, 100, 256)]);
    expect(u32[1]).toBe(lut[lutIndex(100, 0, 100, 256)]);
  });

  it("row 0 of the grid lands in row 0 of the image (no flip)", () => {
    // The backend already emits row 0 = highest subcarrier. renderTileToImageData
    // must NOT flip — a flip here would invert the subcarrier axis relative to
    // subcarrierSourceRect, which expects row 0 = top.
    const W = 1;
    const H = 2;
    const grid = new Float32Array([10, 90]);
    const lut = buildLut(STOPS, 256);
    const target = new ImageData(W, H);
    renderTileToImageData({ grid, width: W, height: H, lut, min: 0, max: 100, target });
    const u32 = new Uint32Array(target.data.buffer);
    expect(u32[0]).toBe(lut[lutIndex(10, 0, 100, 256)]); // grid[0] → row 0
    expect(u32[1]).toBe(lut[lutIndex(90, 0, 100, 256)]); // grid[1] → row 1
  });
});

describe("renderTileToImageData — non-finite values", () => {
  it("NaN maps to NO_DATA_COLOR (no data)", () => {
    const grid = new Float32Array([NaN]);
    const lut = buildLut(STOPS, 256);
    const target = new ImageData(1, 1);
    renderTileToImageData({ grid, width: 1, height: 1, lut, min: 0, max: 100, target });
    const u32 = new Uint32Array(target.data.buffer);
    expect(u32[0]).toBe(NO_DATA_COLOR);
  });

  it("-Infinity maps to NON_FINITE_COLOR, NOT NO_DATA_COLOR", () => {
    // -Infinity comes from db(0) — a real measurement, not a gap. It must
    // render as opaque black, not transparent.
    const grid = new Float32Array([-Infinity]);
    const lut = buildLut(STOPS, 256);
    const target = new ImageData(1, 1);
    renderTileToImageData({ grid, width: 1, height: 1, lut, min: 0, max: 100, target });
    const u32 = new Uint32Array(target.data.buffer);
    expect(u32[0]).toBe(NON_FINITE_COLOR);
    expect(u32[0]).not.toBe(NO_DATA_COLOR);
  });

  it("a mix of NaN, -Infinity, and finite in one grid", () => {
    const grid = new Float32Array([42, NaN, -Infinity, 0]);
    const lut = buildLut(STOPS, 256);
    const target = new ImageData(2, 2);
    renderTileToImageData({ grid, width: 2, height: 2, lut, min: 0, max: 100, target });
    const u32 = new Uint32Array(target.data.buffer);
    expect(u32[0]).toBe(lut[lutIndex(42, 0, 100, 256)]);
    expect(u32[1]).toBe(NO_DATA_COLOR);
    expect(u32[2]).toBe(NON_FINITE_COLOR);
    expect(u32[3]).toBe(lut[lutIndex(0, 0, 100, 256)]);
  });
});

describe("renderTileToImageData — target reuse", () => {
  it("same dimensions reuses the target ImageData reference", () => {
    const grid = new Float32Array([50]);
    const lut = buildLut(STOPS, 256);
    const target = new ImageData(1, 1);
    const result = renderTileToImageData({
      grid, width: 1, height: 1, lut, min: 0, max: 100, target,
    });
    expect(result).toBe(target);
  });

  it("different dimensions returns a new ImageData (not the target)", () => {
    const grid = new Float32Array([50]);
    const lut = buildLut(STOPS, 256);
    const target = new ImageData(2, 2);
    const result = renderTileToImageData({
      grid, width: 1, height: 1, lut, min: 0, max: 100, target,
    });
    expect(result).not.toBe(target);
    expect(result.width).toBe(1);
    expect(result.height).toBe(1);
  });

  it("no target creates a new ImageData with the right dimensions", () => {
    const grid = new Float32Array([50, 60, 70, 80]);
    const lut = buildLut(STOPS, 256);
    const result = renderTileToImageData({
      grid, width: 2, height: 2, lut, min: 0, max: 100,
    });
    expect(result.width).toBe(2);
    expect(result.height).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// subcarrierSourceRect (unchanged — still governs the subcarrier blit)
// ---------------------------------------------------------------------------

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
