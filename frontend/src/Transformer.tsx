import { useEffect, useRef, useState } from "react";

import { ChartBusy } from "@/components/ChartBusy";
import { Button } from "@/components/ui/button";
import { fetchTfm, type Tfm } from "./api";
import { Chart } from "./Chart";
import { AccuracyGrid, NO_PERSON, PERSON, RoomLine, VerdictChip, WindowNav } from "./ModelCommon";

// The Transformer tab (backend/tfm_model.py): transformer alone on 17 per-second channels -- the 15 of the sequence
// models (C1 + C3 lightweight) plus the phase-locked fraction -- judged by the model that never saw the capture's room.

const DESCRIPTION: Record<string, string> = {
  s025: "Median step |r(t) − r(t − 0.25 s)| / (|r(t)| + |r(t − 0.25 s)|) over subcarriers, one value a second: fast motion.",
  s2: "The same step at a 2 s gap.",
  s5: "The same step at a 5 s gap.",
  pk: "FarSense breathing score: first autocorrelation peak of the 10 s window (1 = perfectly repeating); a window every 2 s, held 2 s.",
  rpm: "FarSense breathing rate of the same windows (not calibrated).",
  pk_diff: "FarSense peak on d = r(t) − r(t − T), T the fan's swing period: breathing left once what repeats every T is cancelled.",
  pk_res: "FarSense peak on the residual after the two strongest high-passed (blade) directions are projected out.",
  drift: "1 − |⟨v, v_next⟩| of the leading subcarrier direction of consecutive 2 s blocks: how much the dominant path wanders.",
  ret: "Step at T ÷ step at T/2: a rotating fan comes back to the same state every T (small), a person does not.",
  g15: "2 s step of the signal with the direct AP→TV path removed (delay domain): motion seen in the reflections.",
  bs_all: "Breathing prominence: the strongest breathing line among the 8 leading subcarrier-pattern components (20 s windows, every 4 s).",
  bs_r1: "The same without the strongest component.",
  bs_r2: "The same without the two strongest components.",
  bs_tap: "Breathing prominence on the strongest of the 16 leading CIR taps.",
  bs_f: "Frequency of the strongest breathing line (not calibrated).",
  plf_res:
    "Phase-locked fraction on the residual: how much of the 0.25 s motion energy repeats at the same phase of the swing " +
    "(n ± T, n ± 2T), energy-weighted. A fan's bursts come back at the same head angle (≈ 1); a person's motion does not. A ratio: not calibrated.",
  plf_g: "Phase-locked fraction on the full signal (same definition).",
};

function finite(xs: (number | null)[]): number[] {
  return xs.filter((v): v is number => v != null && Number.isFinite(v));
}

const ok = (v: number | null | undefined): v is number => v != null && Number.isFinite(v);

/** True when some value has no neighbour with a value -- drift (every 2 s) and bs_* (every 4 s) -- so a line that
 *  breaks at every gap would draw nothing. */
function sparse(xs: (number | null)[]): boolean {
  return xs.some((v, i) => ok(v) && !ok(xs[i - 1]) && !ok(xs[i + 1]));
}

/** For display only: straight segments between consecutive values (the model sees the gaps as missing). */
function bridge(xs: (number | null)[]): (number | null)[] {
  const idx = xs.flatMap((v, i) => (ok(v) ? [i] : []));
  const out: (number | null)[] = xs.map(() => null);
  for (let k = 0; k < idx.length; k++) {
    const a = idx[k];
    out[a] = xs[a];
    const b = idx[k + 1];
    if (b == null) break;
    const va = xs[a] as number, vb = xs[b] as number;
    for (let i = a + 1; i < b; i++) out[i] = va + ((vb - va) * (i - a)) / (b - a);
  }
  return out;
}

export function Transformer({ path, dark }: { path: string; dark: boolean }) {
  const [win, setWin] = useState(0);
  const [data, setData] = useState<Tfm | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState<"calibrated" | "raw">("calibrated");

  const holder = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(880);
  useEffect(() => {
    const node = holder.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(320, Math.floor(entry.contentRect.width))));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => setWin(0), [path]);

  useEffect(() => {
    const controller = new AbortController();
    setBusy(true);
    fetchTfm(path, win, controller.signal)
      .then((d) => { setData(d); setError(null); })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setData(null);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [path, win]);

  const cols = width >= 1200 ? 3 : width >= 760 ? 2 : 1;
  const panelWidth = Math.floor((width - 16 * (cols - 1)) / cols);
  const grid = view === "calibrated" && data?.calibrated ? data.calibrated : data?.raw;

  return (
    <div className="space-y-4" ref={holder}>
      <section className="space-y-3 rounded-md border p-3">
        <div className="flex flex-wrap items-baseline gap-2">
          <span className="text-sm font-semibold">Transformer + phase-locked fraction</span>
          <span className="text-[11px] text-muted-foreground">
            17 per-second channels × 61 s · empty-room calibration · 1 token a second, one encoder layer (d 16) · 5 seeds averaged · one model per held-out room
          </span>
          {busy && data && <span className="text-[11px] text-muted-foreground">updating…</span>}
        </div>
        {error ? (
          <div className="text-muted-foreground text-sm">No verdict for this window — {error}</div>
        ) : !data ? (
          <ChartBusy busy empty height={90} label="computing the 17 channels" />
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-3">
              {data.verdict ? <VerdictChip p={data.verdict.p} label={data.verdict.label} truth={data.truth} /> : <span className="text-sm text-muted-foreground">no verdict</span>}
              <WindowNav windows={data.windows} win={win} setWin={setWin} />
              <span className="text-[11px] text-muted-foreground tabular-nums">
                {data.window_s[0].toFixed(0)}–{data.window_s[1].toFixed(0)} s · swing period T {data.swing_period_s == null ? "—" : `${data.swing_period_s.toFixed(1)} s`}
              </span>
            </div>
            <RoomLine room={data.room} trainedOn={data.trained_on} />
            {data.verdict && (
              <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground tabular-nums">
                <span>P per seed:</span>
                {data.verdict.p_seeds.map((p, i) => (
                  <span key={i} className="rounded px-1.5 py-0.5 text-white" style={{ background: p != null && p > data.threshold ? PERSON : NO_PERSON }}>
                    {p == null ? "—" : p.toFixed(3)}
                  </span>
                ))}
                <span>→ mean {data.verdict.p.toFixed(3)} (&gt; {data.threshold} = person)</span>
              </div>
            )}
            <AccuracyGrid
              acc={data.accuracy}
              note={<>Each room held out in turn and judged by the model trained on the other three; scenario C, A and B by the env1 + env2 model (A calibrated with env3's empties, B with its own). Mean over the 5 seeds of each seed's accuracy (ml_lab cy_tfm_plf_table.out). Scenario B has no object minutes.</>}
            />
          </>
        )}
      </section>

      {data && (
        <section className="space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
            <span className="text-sm font-semibold text-foreground">The 17 inputs</span>
            <Button variant={view === "calibrated" ? "default" : "outline"} size="sm" className="h-6 px-2 text-[11px]"
                    disabled={!data.calibrated} onClick={() => setView("calibrated")}>calibrated (model input)</Button>
            <Button variant={view === "raw" ? "default" : "outline"} size="sm" className="h-6 px-2 text-[11px]"
                    onClick={() => setView("raw")}>raw</Button>
            <span>
              calibrated: log on {data.log_channels.join(", ")}; then (x − room's empty median) ÷ empty MAD (shrunk halfway to the training rooms') on every channel but rpm, bs_f, plf_res, plf_g
            </span>
          </div>
          {grid && (
            <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
              {data.channels.map((c, i) => {
                const vals = grid[i];
                const f = finite(vals);
                const lo = f.length ? Math.min(...f) : 0;
                const hi = f.length ? Math.max(...f) : 1;
                const pad = hi > lo ? 0.08 * (hi - lo) : 0.5;
                const times = vals.map((_, k) => data.window_s[0] + k);
                const guides = view === "calibrated" && data.calibrated_channels.includes(c)
                  ? [{ value: 0, color: "#9aa5b1", label: "empty median" }] : [];
                const color = c.startsWith("plf") ? "#b45309" : PERSON;
                const gappy = sparse(vals);
                return (
                  <div key={c} className="space-y-1 min-w-0">
                    <div className="text-xs" title={DESCRIPTION[c]}>
                      <b>{c}</b> <span className="text-muted-foreground">{f.length} of {vals.length} s with a value{gappy ? " · dots = values, dashed = drawn between them (the model sees the gaps as missing)" : ""}</span>
                    </div>
                    <Chart
                      width={panelWidth} dark={dark} height={120} times={times}
                      domain={[data.window_s[0], data.window_s[0] + vals.length - 1]}
                      yDomain={[lo - pad, hi + pad]} yLabel={view === "calibrated" && data.calibrated_channels.includes(c) ? "z vs empty" : c}
                      series={gappy
                        ? [{ values: bridge(vals), color, width: 1, label: c, dashed: true }]
                        : [{ values: vals, color, width: 1.3, label: c }]}
                      dots={gappy ? { values: vals, colors: vals.map(() => color), radius: 2.5, label: c } : undefined}
                      guides={guides}
                    />
                    <div className="text-[11px] text-muted-foreground leading-snug">{DESCRIPTION[c]}</div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
