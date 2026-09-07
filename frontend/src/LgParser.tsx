import { useEffect, useMemo, useRef, useState } from "react";

import { fetchLgParse, type LgParse } from "./api";

const PAD = { top: 12, right: 16, bottom: 26, left: 46 };

/** A second parser's reading of the same capture.
 *
 * Everything numerical here is computed by the vendored MT7921 parser's own
 * functions over this project's reader -- see backend/lgproc. It is shown
 * because the two parsers disagree in ways a single number settles, and a
 * disagreement is only useful if it is visible.
 */
export function LgParser({ path, dark }: { path: string; dark: boolean }) {
  const [data, setData] = useState<LgParse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const holder = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(880);

  useEffect(() => {
    const controller = new AbortController();
    setData(null);
    setError(null);
    fetchLgParse(path, 4096, controller.signal)
      .then(setData)
      .catch((e) => {
        if (e.name !== "AbortError") setError(String(e.message ?? e));
      });
    return () => controller.abort();
  }, [path]);

  useEffect(() => {
    const node = holder.current;
    if (!node) return;
    const ro = new ResizeObserver(([entry]) =>
      setWidth(Math.max(320, Math.floor(entry.contentRect.width))),
    );
    ro.observe(node);
    return () => ro.disconnect();
  }, [data]);

  const activeSet = useMemo(
    () => new Set(data?.activeBins ?? []),
    [data],
  );

  if (error) {
    return (
      <div className="rounded-md border border-dashed px-3 py-2 text-[11px]">
        {error}
      </div>
    );
  }
  if (!data) {
    return <div className="text-[11px] text-muted-foreground">reading…</div>;
  }

  const height = 190;
  const iw = Math.max(1, width - PAD.left - PAD.right);
  const ih = height - PAD.top - PAD.bottom;
  const finite = data.agcSpectrum.filter(
    (v): v is number => v !== null && Number.isFinite(v),
  );
  const lo = Math.min(...finite) - 2;
  const hi = Math.max(...finite) + 2;
  const x = (k: number) => (k / Math.max(1, data.nsub - 1)) * iw;
  const y = (v: number) => ih - ((v - lo) / (hi - lo || 1)) * ih;

  const line = (vals: (number | null)[]) => {
    let d = "";
    let pen = false;
    vals.forEach((v, k) => {
      if (v === null || !Number.isFinite(v)) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${x(k).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };

  const grid = dark ? "#3a4048" : "#dfe3e9";
  const rank = Object.entries(data.coherence).sort((a, b) => b[1] - a[1]);

  return (
    <div ref={holder} className="space-y-4">
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-muted-foreground">
        <span>
          <b className="text-foreground">{data.frames}</b> of {data.framesInFile} frames
        </span>
        <span>{data.nrx}×{data.ntx} grid</span>
        <span>{data.nsub} bins</span>
        <span>
          <b className="text-foreground">{data.activeBins.length}</b> active
        </span>
        <span>RSSI {data.rssiMean.toFixed(1)} dBm</span>
        <span>peer <code>{data.peer}</code></span>
        <span>csi_parse {data.toolVersion}</span>
      </div>

      <div>
        <div className="mb-1 text-[11px] text-muted-foreground">
          Lag-1 phase coherence by feature — 1 is a phase that survives frame to
          frame, 0 is noise. <b>conj_rx</b> is their default feature, conjugated
          across receive paths; <b>conj_tx</b> is the axis this project divides
          along. The gap is the disagreement between the two parsers, and it is
          the reason ours does not adopt theirs.
        </div>
        <div className="space-y-1">
          {rank.map(([name, v]) => (
            <div key={name} className="flex items-center gap-2 text-[11px]">
              <span className="w-20 tabular-nums">{name}</span>
              <div className="h-3 flex-1 overflow-hidden rounded-sm" style={{ background: grid }}>
                <div
                  style={{
                    width: `${Math.max(0, Math.min(1, v)) * 100}%`,
                    height: "100%",
                    background:
                      name === "conj_tx" ? "#2f9c6f" : name === "raw" ? "#b04a4a" : "#7b5ea7",
                  }}
                />
              </div>
              <span className="w-12 text-right tabular-nums">{v.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </div>

      <div>
        <div className="mb-1 text-[11px] text-muted-foreground">
          Mean amplitude across the band. <span style={{ color: "#2f6fed" }}>■</span>{" "}
          after their RSSI-based AGC restoration (absolute dBm),{" "}
          <span style={{ color: "#9aa5b1" }}>■</span> as the chip reports it
          (relative — the hardware strips its own gain). Shaded bins are the ones
          their occupancy rule keeps.
        </div>
        <svg width={width} height={height} role="img" aria-label="amplitude spectrum">
          <g transform={`translate(${PAD.left},${PAD.top})`}>
            {Array.from({ length: data.nsub }, (_, k) =>
              activeSet.has(k) ? null : (
                <rect key={k} x={x(k)} width={Math.max(1, iw / data.nsub)} y={0}
                      height={ih} fill={grid} opacity={0.5} />
              ),
            )}
            <path d={line(data.rawSpectrum)} fill="none" stroke="#9aa5b1" strokeWidth={1.2} />
            <path d={line(data.agcSpectrum)} fill="none" stroke="#2f6fed" strokeWidth={1.6} />
            {[lo, (lo + hi) / 2, hi].map((v) => (
              <text key={v} x={-6} y={y(v) + 3} textAnchor="end" fontSize={10} fill="currentColor">
                {v.toFixed(0)}
              </text>
            ))}
            <text x={iw / 2} y={ih + 18} textAnchor="middle" fontSize={10} fill="currentColor">
              subcarrier (raw FFT order — their index is provisional, ours is fftshifted)
            </text>
          </g>
        </svg>
      </div>

      <div className="text-[11px] text-muted-foreground leading-relaxed">
        <b>Transmitters heard:</b>{" "}
        {data.macCensus.map(([mac, n]) => (
          <span key={mac} className="mr-3">
            <code>{mac}</code> {n}
          </span>
        ))}
        <br />
        The CSI engine latches frames it can hear, not only frames addressed to
        us, so a capture can carry a transmitter whose channel has nothing to do
        with the link under test. Their parser filters to the busiest one; this
        project reports it and lets the caller filter.
      </div>
    </div>
  );
}
