export interface Meta {
  filename: string;
  chipset: string;
  bandwidth: string | number;
  num_subcarriers: number;
  total_frames: number;
  t_min: number;
  t_max: number;
  num_rx: number;
  num_tx: number;
}

export async function fetchMeta(
  path: string,
  signal?: AbortSignal,
): Promise<Meta> {
  const url = `/api/meta?path=${encodeURIComponent(path)}`;
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return (await res.json()) as Meta;
}

export interface Tile {
  grid: Float32Array; // length = height * width, row-major, row 0 = highest subcarrier
  width: number;
  height: number;
  t0: number;
  t1: number; // the window THIS tile covers (echo the request)
  captureTMin: number;
  captureTMax: number; // the whole capture's extent, NOT this tile's window
  framesDecoded: number;
  totalInRange: number;
  exact: boolean;
  vmin: number;
  vmax: number;
  pLow: number; // 1st percentile of finite values — robust scale for amplitude
  pHigh: number; // 99th percentile — amplitude locks to this, not vmin/vmax
}

export async function fetchTile(
  path: string,
  t0: number,
  t1: number,
  width: number,
  metric: "amplitude" | "phase",
  signal?: AbortSignal,
): Promise<Tile> {
  const url =
    `/api/tile?path=${encodeURIComponent(path)}` +
    `&t0=${t0}&t1=${t1}&width=${width}&metric=${metric}`;
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  // The backend writes little-endian float32 explicitly (grid.astype("<f4")).
  // Every platform this runs on is little-endian, so a direct Float32Array view
  // over the ArrayBuffer is correct without a byte-swap.
  const grid = new Float32Array(await res.arrayBuffer());
  const h = res.headers;
  return {
    grid,
    width: parseInt(h.get("X-Tile-Width") ?? "0", 10),
    height: parseInt(h.get("X-Tile-Height") ?? "0", 10),
    t0,
    t1,
    captureTMin: parseFloat(h.get("X-Capture-TMin") ?? "0"),
    captureTMax: parseFloat(h.get("X-Capture-TMax") ?? "0"),
    framesDecoded: parseInt(h.get("X-Tile-Frames") ?? "0", 10),
    totalInRange: parseInt(h.get("X-Tile-Total") ?? "0", 10),
    exact: h.get("X-Tile-Exact") === "1",
    vmin: parseFloat(h.get("X-Tile-VMin") ?? "0"),
    vmax: parseFloat(h.get("X-Tile-VMax") ?? "0"),
    pLow: parseFloat(h.get("X-Tile-PLow") ?? "0"),
    pHigh: parseFloat(h.get("X-Tile-PHigh") ?? "0"),
  };
}
