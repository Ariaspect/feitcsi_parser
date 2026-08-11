export interface Snapshot {
  filename: string;
  chipset: string;
  bandwidth: string;
  num_subcarriers: number;
  total_packets: number;
  window_packets: number;
  time_seconds: number[];
  amplitude: number[][];
  phase: number[][];
  amp_min: number;
  amp_max: number;
  phase_min: number;
  phase_max: number;
}

export async function fetchSnapshot(
  path: string,
  maxPackets: number,
): Promise<Snapshot> {
  const url = `/api/snapshot?path=${encodeURIComponent(path)}&max_packets=${maxPackets}`;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return (await res.json()) as Snapshot;
}
