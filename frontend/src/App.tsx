import { useEffect, useState } from "react";
import { fetchFilters, fetchMeta, type Filters, type Meta } from "./api";
import { TWILIGHT } from "./colormap";
import { Heatmap } from "./Heatmap";
import { createTimeLink } from "./timelink";

const DEFAULT_PATH = "captures/capture.dat";
const DEFAULT_REFRESH_MS = 300;

export function App() {
  const [path, setPath] = useState(DEFAULT_PATH);
  const [refreshMs, setRefreshMs] = useState(DEFAULT_REFRESH_MS);
  const [running, setRunning] = useState(false);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);
  const [filters, setFilters] = useState<Filters | null>(null);
  const [mimo, setMimo] = useState<string>("all");
  const [sourceMac, setSourceMac] = useState<string>("all");

  // One TimeLink shared by both heatmaps — zooming one mirrors the time axis
  // on the other. Subcarrier zoom stays independent. useState's lazy
  // initializer runs the factory exactly once; useRef(createTimeLink()) would
  // rebuild it on every render (3x/s while polling) only to throw it away.
  const [timeLink] = useState(createTimeLink);

  useEffect(() => {
    let cancelled = false;
    let inFlight = false;

    const poll = async () => {
      if (cancelled || inFlight) return;
      inFlight = true;
      try {
        const m = await fetchMeta(path, undefined, mimo, sourceMac);
        if (!cancelled) {
          setMeta(m);
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

    // Always fetch once on path/refresh change so explore mode (paused) has
    // geometry. In live mode, the interval drives subsequent polls.
    poll();
    if (running) {
      const id = setInterval(poll, refreshMs);
      return () => {
        cancelled = true;
        clearInterval(id);
      };
    }
    return () => {
      cancelled = true;
    };
  }, [running, path, refreshMs, mimo, sourceMac]);

  // Fetch the dropdown option lists whenever the path changes. Independent of
  // the meta poll so a running live view does not refetch the option lists on
  // every refresh.
  useEffect(() => {
    let cancelled = false;
    setFilters(null);
    fetchFilters(path)
      .then((f) => {
        if (!cancelled) setFilters(f);
      })
      .catch(() => {
        // Path errors are surfaced by the meta poll; suppress duplicate.
      });
    return () => {
      cancelled = true;
    };
  }, [path]);

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: "1rem", maxWidth: 1400, margin: "0 auto" }}>
      <h1>FeitCSI Heatmap</h1>

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
        <label>
          MIMO
          <br />
          <select
            value={mimo}
            onChange={(e) => setMimo(e.target.value)}
            disabled={!filters || filters.mimo_modes.length <= 1}
            style={{ padding: "0.3rem", minWidth: 90 }}
          >
            <option value="all">all</option>
            {filters?.mimo_modes.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </label>
        <label>
          Source MAC
          <br />
          <select
            value={sourceMac}
            onChange={(e) => setSourceMac(e.target.value)}
            disabled={!filters || filters.source_macs.length <= 1}
            style={{ padding: "0.3rem", minWidth: 160 }}
          >
            <option value="all">all</option>
            {filters?.source_macs.map((mac) => (
              <option key={mac} value={mac}>{mac}</option>
            ))}
          </select>
        </label>
      </div>

      {error && (
        <div style={{ color: "red", marginBottom: "1rem" }}>
          <b>Error:</b> {error}
        </div>
      )}

      {meta && (
        <div style={{ marginBottom: "1rem", color: "#444" }}>
          <b>{meta.filename}</b> — {meta.chipset} / {meta.bandwidth} MHz —
          frames: {meta.total_frames}, subcarriers: {meta.num_subcarriers},
          time: [{meta.t_min.toFixed(3)}, {meta.t_max.toFixed(3)}] s
          {lastUpdate && <span>, last update: {new Date(lastUpdate).toLocaleTimeString()}</span>}
        </div>
      )}

      {meta && meta.total_frames > 0 ? (
        <>
          <Heatmap
            path={path}
            metric="amplitude"
            filename={meta.filename}
            numSubcarriers={meta.num_subcarriers}
            captureTMin={meta.t_min}
            captureTMax={meta.t_max}
            title="FeitCSI — amplitude"
            colorLabel="Amplitude (dBm)"
            height={400}
            timeLink={timeLink}
            mimo={mimo}
            sourceMac={sourceMac}
          />
          <div style={{ height: "1.5rem" }} />
          <Heatmap
            path={path}
            metric="phase"
            filename={meta.filename}
            numSubcarriers={meta.num_subcarriers}
            captureTMin={meta.t_min}
            captureTMax={meta.t_max}
            minValue={-Math.PI}
            maxValue={Math.PI}
            title="FeitCSI — phase"
            colorLabel="Phase (rad)"
            height={400}
            palette={TWILIGHT}
            timeLink={timeLink}
            mimo={mimo}
            sourceMac={sourceMac}
          />
          <div style={{ height: "1.5rem" }} />
          <Heatmap
            path={path}
            metric="csi_ratio_amplitude"
            filename={meta.filename}
            numSubcarriers={meta.num_subcarriers}
            captureTMin={meta.t_min}
            captureTMax={meta.t_max}
            title="FeitCSI — CSI ratio amplitude (rx1/rx0)"
            colorLabel="Ratio amp (dB)"
            height={400}
            timeLink={timeLink}
            mimo={mimo}
            sourceMac={sourceMac}
          />
          <div style={{ height: "1.5rem" }} />
          <Heatmap
            path={path}
            metric="csi_ratio_phase"
            filename={meta.filename}
            numSubcarriers={meta.num_subcarriers}
            captureTMin={meta.t_min}
            captureTMax={meta.t_max}
            minValue={-Math.PI}
            maxValue={Math.PI}
            title="FeitCSI — CSI ratio phase (rx1/rx0)"
            colorLabel="Ratio phase (rad)"
            height={400}
            palette={TWILIGHT}
            timeLink={timeLink}
            mimo={mimo}
            sourceMac={sourceMac}
          />
        </>
      ) : (
        !error && (
          <div style={{ color: "#888", padding: "2rem" }}>
            Enter a .dat path to explore, or toggle <b>Run realtime</b> for live capture. Default file:{" "}
            <code>{DEFAULT_PATH}</code>.
          </div>
        )
      )}
    </div>
  );
}
