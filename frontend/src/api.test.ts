import { describe, it, expect, vi, afterEach } from "vitest";
import { fetchDoppler, fetchMeta, fetchTile } from "./api";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function mockResponse(opts: {
  ok?: boolean;
  status?: number;
  body?: ArrayBuffer;
  json?: unknown;
  headers?: Record<string, string>;
}) {
  const ok = opts.ok ?? true;
  const headers = opts.headers ?? {};
  return {
    ok,
    status: opts.status ?? (ok ? 200 : 500),
    arrayBuffer: () => Promise.resolve(opts.body ?? new ArrayBuffer(0)),
    text: () => Promise.resolve("error body"),
    json: () => Promise.resolve(opts.json ?? {}),
    headers: {
      get: (name: string) => headers[name] ?? null,
    },
  };
}

describe("fetchMeta", () => {
  it("parses the JSON response into a Meta object", async () => {
    const mockMeta = {
      filename: "capture.dat",
      chipset: "Intel AX2xx",
      bandwidth: 80,
      num_subcarriers: 242,
      total_frames: 1000,
      t_min: 0.0,
      t_max: 10.5,
      num_rx: 1,
      num_tx: 1,
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse({ json: mockMeta })));

    const meta = await fetchMeta("captures/capture.dat");
    expect(meta).toEqual(mockMeta);
    expect(meta.filename).toBe("capture.dat");
    expect(meta.num_subcarriers).toBe(242);
    expect(meta.t_max).toBe(10.5);
  });

  it("encodes the path in the URL", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse({ json: {} })));
    await fetchMeta("captures/my file.dat");
    expect(fetch).toHaveBeenCalledTimes(1);
    const url = (fetch as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(url).toContain("path=captures%2Fmy%20file.dat");
  });

  it("passes the signal through to fetch", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse({ json: {} })));
    const controller = new AbortController();
    await fetchMeta("p", controller.signal);
    const init = (fetch as ReturnType<typeof vi.fn>).mock.calls[0][1] as RequestInit;
    expect(init.signal).toBe(controller.signal);
  });

  it("throws on HTTP error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(mockResponse({ ok: false, status: 404 })),
    );
    await expect(fetchMeta("p")).rejects.toThrow("HTTP 404");
  });
});

describe("fetchTile", () => {
  function float32Buf(values: number[]): ArrayBuffer {
    const buf = new ArrayBuffer(values.length * 4);
    new Float32Array(buf).set(values);
    return buf;
  }

  it("decodes the float32 body and parses all headers including exact and capture extent", async () => {
    const body = float32Buf([1.0, 2.0, NaN, -Infinity]);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        mockResponse({
          body,
          headers: {
            "X-Tile-Width": "2",
            "X-Tile-Height": "2",
            "X-Capture-TMin": "0.0",
            "X-Capture-TMax": "10.5",
            "X-Tile-Frames": "100",
            "X-Tile-Total": "200",
            "X-Tile-Exact": "0",
            "X-Tile-VMin": "-90.5",
            "X-Tile-VMax": "-30.2",
            "X-Tile-PLow": "-55.0",
            "X-Tile-PHigh": "-35.0",
          },
        }),
      ),
    );

    const tile = await fetchTile("captures/capture.dat", 0, 10, 2, "amplitude");

    expect(tile.width).toBe(2);
    expect(tile.height).toBe(2);
    expect(tile.t0).toBe(0);
    expect(tile.t1).toBe(10);
    // X-Capture-TMin/TMax are the WHOLE capture's extent, not this tile's.
    expect(tile.captureTMin).toBe(0.0);
    expect(tile.captureTMax).toBe(10.5);
    expect(tile.framesDecoded).toBe(100);
    expect(tile.totalInRange).toBe(200);
    expect(tile.exact).toBe(false);
    expect(tile.vmin).toBe(-90.5);
    expect(tile.vmax).toBe(-30.2);
    expect(tile.pLow).toBe(-55.0);
    expect(tile.pHigh).toBe(-35.0);

    // Body decode: little-endian float32, row-major.
    expect(tile.grid.length).toBe(4);
    expect(tile.grid[0]).toBeCloseTo(1.0);
    expect(tile.grid[1]).toBeCloseTo(2.0);
    expect(Number.isNaN(tile.grid[2])).toBe(true);
    expect(tile.grid[3]).toBe(-Infinity);
  });

  it("exact=true when X-Tile-Exact is '1'", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        mockResponse({
          body: float32Buf([1.0]),
          headers: {
            "X-Tile-Width": "1",
            "X-Tile-Height": "1",
            "X-Tile-Exact": "1",
          },
        }),
      ),
    );
    const tile = await fetchTile("p", 0, 1, 1, "amplitude");
    expect(tile.exact).toBe(true);
  });

  it("builds the URL with path, t0, t1, width, and metric", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        mockResponse({
          body: float32Buf([1.0]),
          headers: { "X-Tile-Width": "1", "X-Tile-Height": "1", "X-Tile-Exact": "1" },
        }),
      ),
    );
    await fetchTile("captures/cap.dat", 1.5, 9.75, 800, "phase");
    const url = (fetch as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(url).toContain("path=captures%2Fcap.dat");
    expect(url).toContain("t0=1.5");
    expect(url).toContain("t1=9.75");
    expect(url).toContain("width=800");
    expect(url).toContain("metric=phase");
  });

  it("passes the signal through to fetch", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        mockResponse({
          body: float32Buf([1.0]),
          headers: { "X-Tile-Width": "1", "X-Tile-Height": "1", "X-Tile-Exact": "1" },
        }),
      ),
    );
    const controller = new AbortController();
    await fetchTile("p", 0, 1, 1, "amplitude", controller.signal);
    const init = (fetch as ReturnType<typeof vi.fn>).mock.calls[0][1] as RequestInit;
    expect(init.signal).toBe(controller.signal);
  });

  it("omits interpolate from the URL when true or unset, matching the backend's own default", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        mockResponse({
          body: float32Buf([1.0]),
          headers: { "X-Tile-Width": "1", "X-Tile-Height": "1", "X-Tile-Exact": "1" },
        }),
      ),
    );
    await fetchTile("p", 0, 1, 1, "amplitude", undefined, null, null, true);
    await fetchTile("p", 0, 1, 1, "amplitude");
    const calls = (fetch as ReturnType<typeof vi.fn>).mock.calls;
    expect(calls[0][0] as string).not.toContain("interpolate");
    expect(calls[1][0] as string).not.toContain("interpolate");
  });

  it("adds interpolate=false to the URL when explicitly disabled", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        mockResponse({
          body: float32Buf([1.0]),
          headers: { "X-Tile-Width": "1", "X-Tile-Height": "1", "X-Tile-Exact": "1" },
        }),
      ),
    );
    await fetchTile("p", 0, 1, 1, "amplitude", undefined, null, null, false);
    const url = (fetch as ReturnType<typeof vi.fn>).mock.calls[0][0] as string;
    expect(url).toContain("interpolate=false");
  });

  it("throws on HTTP error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(mockResponse({ ok: false, status: 500 })),
    );
    await expect(fetchTile("p", 0, 1, 1, "amplitude")).rejects.toThrow("HTTP 500");
  });
});

describe("fetchDoppler", () => {
  function dopplerResponse(width: number, height: number) {
    const grid = new Float32Array(width * height).fill(1);
    return new Response(grid.buffer, {
      status: 200,
      headers: {
        "X-Doppler-Width": String(width),
        "X-Doppler-Height": String(height),
        "X-Doppler-Fs": "17.86",
        "X-Doppler-FMax": "8.93",
        "X-Doppler-Win": String((height - 1) * 2),
        "X-Doppler-Hop": String(height - 1),
        "X-Doppler-WinSeconds": "30",
        "X-Doppler-Frames": "5524",
        "X-Doppler-ColT0": "15",
        "X-Doppler-ColT1": "185",
        "X-Capture-TMin": "0",
        "X-Capture-TMax": "200",
        "X-Tile-VMin": "0",
        "X-Tile-VMax": "10",
        "X-Tile-PLow": "1",
        "X-Tile-PHigh": "9",
      },
    });
  }

  afterEach(() => vi.unstubAllGlobals());

  it("parses the grid and the doppler headers", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => dopplerResponse(20, 129)));
    const tile = await fetchDoppler("c.dat", 0, 200, "amplitude", 30);
    expect(tile.width).toBe(20);
    expect(tile.height).toBe(129);
    expect(tile.grid.length).toBe(20 * 129);
    expect(tile.fs).toBeCloseTo(17.86);
    expect(tile.fMax).toBeCloseTo(8.93);
    // Column centres, not the requested range.
    expect(tile.t0).toBeCloseTo(15);
    expect(tile.t1).toBeCloseTo(185);
  });

  it("sends win_seconds and the filter params", async () => {
    const spy = vi.fn(async () => dopplerResponse(4, 5));
    vi.stubGlobal("fetch", spy);
    await fetchDoppler("c.dat", 1, 2, "amplitude", 45, undefined, "2x2", "aa:bb");
    const url = String(spy.mock.calls[0][0]);
    expect(url).toContain("/api/doppler?");
    expect(url).toContain("win_seconds=45");
    expect(url).toContain("mimo=2x2");
  });

  it("throws with the server status on an error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("too short", { status: 400 })));
    await expect(fetchDoppler("c.dat", 0, 1, "amplitude", 600)).rejects.toThrow(/400/);
  });
});
