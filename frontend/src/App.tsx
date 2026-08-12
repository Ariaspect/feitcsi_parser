import { useEffect, useState } from "react";
import { fetchSnapshot, type Snapshot } from "./api";
import { Heatmap } from "./Heatmap";

const DEFAULT_PATH = "captures/capture.dat";
const DEFAULT_WINDOW = 200;
const DEFAULT_REFRESH_MS = 300;

export function App() {
  const [path, setPath] = useState(DEFAULT_PATH);
  const [maxPackets, setMaxPackets] = useState(DEFAULT_WINDOW);
  const [refreshMs, setRefreshMs] = useState(DEFAULT_REFRESH_MS);
  const [running, setRunning] = useState(false);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);

  useEffect(() => {
    if (!running) return;
    let cancelled = false;
    let inFlight = false;

    const poll = async () => {
      if (cancelled || inFlight) return;
      inFlight = true;
      try {
        const snap = await fetchSnapshot(path, maxPackets);
        if (!cancelled) {
          setSnapshot(snap);
          setError(null);
          setLastUpdate(Date.now());
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        inFlight = false;
      }
    };

    poll();
    const id = setInterval(poll, refreshMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [running, path, maxPackets, refreshMs]);

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: "1rem", maxWidth: 1400, margin: "0 auto" }}>
      <h1>FeitCSI Realtime Heatmap</h1>

      <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", marginBottom: "1rem", alignItems: "flex-end" }}>
        <label>
          .dat file
          <br />
          <input
            type="text"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            style={{ width: 360, padding: "0.3rem" }}
          />
        </label>
        <label>
          Window (packets)
          <br />
          <input
            type="number"
            min={1}
            max={10000}
            value={maxPackets}
            onChange={(e) => setMaxPackets(Number(e.target.value))}
            style={{ width: 100, padding: "0.3rem" }}
          />
        </label>
        <label>
          Refresh (ms)
          <br />
          <input
            type="number"
            min={50}
            max={10000}
            value={refreshMs}
            onChange={(e) => setRefreshMs(Number(e.target.value))}
            style={{ width: 100, padding: "0.3rem" }}
          />
        </label>
        <button
          onClick={() => setRunning((r) => !r)}
          style={{
            padding: "0.5rem 1.2rem",
            background: running ? "#d33" : "#2a7",
            color: "white",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
            fontWeight: "bold",
          }}
        >
          {running ? "Pause" : "Run realtime"}
        </button>
      </div>

      {error && (
        <div style={{ color: "red", marginBottom: "1rem" }}>
          <b>Error:</b> {error}
        </div>
      )}

      {snapshot && (
        <div style={{ marginBottom: "1rem", color: "#444" }}>
          <b>{snapshot.filename}</b> — {snapshot.chipset} / {snapshot.bandwidth} MHz —
          total: {snapshot.total_packets} packets, window: {snapshot.window_packets},
          subcarriers: {snapshot.num_subcarriers}, amp range: [{snapshot.amp_min.toFixed(1)}, {snapshot.amp_max.toFixed(1)}] dBm
          {lastUpdate && <span>, last update: {new Date(lastUpdate).toLocaleTimeString()}</span>}
        </div>
      )}

      {snapshot && snapshot.window_packets > 0 ? (
        <>
          <Heatmap
            filename={snapshot.filename}
            timeSeconds={snapshot.time_seconds}
            matrix={snapshot.amplitude}
            subcarrierCount={snapshot.num_subcarriers}
            minValue={snapshot.amp_min}
            maxValue={snapshot.amp_max}
            title="FeitCSI — amplitude"
            colorLabel="Amplitude (dBm)"
            height={400}
          />
          <div style={{ height: "1.5rem" }} />
          <Heatmap
            filename={snapshot.filename}
            timeSeconds={snapshot.time_seconds}
            matrix={snapshot.phase}
            subcarrierCount={snapshot.num_subcarriers}
            minValue={snapshot.phase_min}
            maxValue={snapshot.phase_max}
            title="FeitCSI — phase"
            colorLabel="Phase (rad)"
            aggregation="nearest"
            height={400}
          />
        </>
      ) : (
        !error && (
          <div style={{ color: "#888", padding: "2rem" }}>
            Toggle <b>Run realtime</b> to begin. Default file: <code>{DEFAULT_PATH}</code>.
          </div>
        )
      )}
    </div>
  );
}
