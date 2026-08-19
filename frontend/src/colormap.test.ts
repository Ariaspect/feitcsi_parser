import { describe, it, expect } from "vitest";
import { VIRIDIS, TWILIGHT, buildLut, lutIndex, NO_DATA_COLOR, NON_FINITE_COLOR } from "./colormap";

function unpack(rgba: number): [number, number, number, number] {
  return [rgba & 0xff, (rgba >>> 8) & 0xff, (rgba >>> 16) & 0xff, (rgba >>> 24) & 0xff];
}

describe("buildLut", () => {
  it("has length equal to size", () => {
    expect(buildLut(VIRIDIS, 256).length).toBe(256);
    expect(buildLut(VIRIDIS, 64).length).toBe(64);
    expect(buildLut(VIRIDIS, 1).length).toBe(1);
  });

  it("endpoints equal the first and last stop exactly", () => {
    const stops = ["#102030", "#aabbcc"];
    const lut = buildLut(stops, 256);
    expect(unpack(lut[0])).toEqual([0x10, 0x20, 0x30, 255]);
    expect(unpack(lut[255])).toEqual([0xaa, 0xbb, 0xcc, 255]);
  });

  it("midpoint between two stops equals the rounded average", () => {
    const lut = buildLut(["#000000", "#ffffff"], 3);
    // LUT[1] is at normalized position 0.5: lerp(0, 255, 0.5) = 127.5 → Math.round = 128
    const [r, g, b, a] = unpack(lut[1]);
    expect(r).toBe(Math.round((0 + 255) / 2));
    expect(g).toBe(Math.round((0 + 255) / 2));
    expect(b).toBe(Math.round((0 + 255) / 2));
    expect(a).toBe(255);
  });

  it("caches by stops+size (same reference)", () => {
    const a = buildLut(VIRIDIS, 256);
    const b = buildLut(VIRIDIS, 256);
    expect(a).toBe(b);
  });
});

describe("lutIndex", () => {
  it("clamps below min to 0", () => {
    expect(lutIndex(-5, 0, 10, 256)).toBe(0);
    expect(lutIndex(0, 0, 10, 256)).toBe(0);
  });

  it("clamps above max to size-1", () => {
    expect(lutIndex(15, 0, 10, 256)).toBe(255);
    expect(lutIndex(10, 0, 10, 256)).toBe(255);
  });

  it("returns -1 for NaN and Infinity", () => {
    expect(lutIndex(NaN, 0, 10, 256)).toBe(-1);
    expect(lutIndex(Infinity, 0, 10, 256)).toBe(-1);
    expect(lutIndex(-Infinity, 0, 10, 256)).toBe(-1);
  });

  it("handles max === min without dividing by zero", () => {
    expect(lutIndex(5, 5, 5, 256)).toBe(0); // (5-5)/(0||1) = 0 → index 0
    expect(lutIndex(6, 5, 5, 256)).toBe(255); // (6-5)/1 = 1 → clamped → 255
  });

  it("maps midpoint to the middle index", () => {
    expect(lutIndex(5, 0, 10, 256)).toBe(128); // round(0.5 * 255) = round(127.5) = 128
  });
});

describe("special colors", () => {
  it("NO_DATA_COLOR is an opaque neutral grey, in neither palette", () => {
    const [r, g, b, a] = unpack(NO_DATA_COLOR);
    expect(a).toBe(255); // opaque: a gap must not read as the canvas behind it
    expect([g, b]).toEqual([r, r]); // unsaturated
    for (const palette of [VIRIDIS, TWILIGHT]) {
      expect(buildLut(palette, 256)).not.toContain(NO_DATA_COLOR);
    }
    expect(NO_DATA_COLOR).not.toBe(NON_FINITE_COLOR);
  });

  it("NON_FINITE_COLOR is opaque black", () => {
    expect(unpack(NON_FINITE_COLOR)).toEqual([0, 0, 0, 255]);
  });
});

// ---------------------------------------------------------------------------
// TWILIGHT cyclic colormap — phase wraps, so the colormap must start and end
// on the same color or every −π/+π wrap paints a false edge.
// ---------------------------------------------------------------------------

describe("TWILIGHT cyclic colormap", () => {
  it("first and last stops are identical", () => {
    expect(TWILIGHT[0]).toBe(TWILIGHT[TWILIGHT.length - 1]);
  });

  it("buildLut produces near-identical RGBA at index 0 and size-1", () => {
    const lut = buildLut(TWILIGHT, 256);
    const [r0, g0, b0] = unpack(lut[0]);
    const [r1, g1, b1] = unpack(lut[255]);
    // Exact equality is expected (both endpoints map to the same stop), but
    // allow a tolerance of 1 so a future LUT rounding change doesn't break the
    // guard while still catching a real cycle break.
    expect(Math.abs(r0 - r1)).toBeLessThanOrEqual(1);
    expect(Math.abs(g0 - g1)).toBeLessThanOrEqual(1);
    expect(Math.abs(b0 - b1)).toBeLessThanOrEqual(1);
  });
});
