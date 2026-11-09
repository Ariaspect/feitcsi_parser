import { useEffect, useMemo, useState } from "react";

import { ChartBusy } from "@/components/ChartBusy";
import { Button } from "@/components/ui/button";
import { fetchCalib41, type Calib41 } from "./api";
import { AccuracyGrid, PERSON, RoomLine, VerdictChip, WindowNav } from "./ModelCommon";

// The 41F hidden layer on empty-room-calibrated columns (backend/calib41.py), judged by the model that never saw
// the capture's room. The ML tab's top section.

const GUIDE = "#9aa5b1";
const TOP = 15;

function fmt(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(1) : a >= 1 ? v.toFixed(3) : v.toPrecision(3);
}

function EffectBar({ v, lim, dark }: { v: number | null; lim: number; dark: boolean }) {
  const w = 80;
  const half = w / 2;
  const x0 = v ?? NaN;
  const len = Number.isFinite(x0) ? Math.min(half, (Math.abs(x0) / lim) * half) : 0;
  const x = x0 >= 0 ? half : half - len;
  return (
    <span className="inline-flex items-center gap-1 tabular-nums">
      <svg width={w} height={10} aria-hidden>
        <line x1={half} x2={half} y1={0} y2={10} stroke={dark ? "#4b5563" : "#c3cad3"} />
        <rect x={x} y={2} width={Math.max(len, 0.5)} height={6} rx={1} fill={x0 >= 0 ? PERSON : GUIDE} />
      </svg>
      <span className="w-10 text-right">{Number.isFinite(x0) ? `${x0 >= 0 ? "+" : ""}${x0.toFixed(2)}` : "—"}</span>
    </span>
  );
}

export function Calib41Panel({ path, win, setWin, dark }: { path: string; win: number; setWin: (w: number) => void; dark: boolean }) {
  const [data, setData] = useState<Calib41 | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setBusy(true);
    fetchCalib41(path, win, controller.signal)
      .then((d) => { setData(d); setError(null); })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setData(null);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [path, win]);

  return (
    <section className="space-y-3 rounded-md border p-3">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="text-sm font-semibold">41F model (calibrated)</span>
        <span className="text-[11px] text-muted-foreground">
          11F + aux + Liu (41 columns) · log, minus the room's empty-minute median · one hidden layer (16) · one model per held-out room
        </span>
        {busy && data && <span className="text-[11px] text-muted-foreground">updating…</span>}
      </div>
      {error ? (
        <div className="text-muted-foreground text-sm">No verdict for this window — {error}</div>
      ) : !data ? (
        <ChartBusy busy empty height={90} label="computing the 41 columns" />
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-3">
            {data.verdict ? <VerdictChip p={data.verdict.p} label={data.verdict.label} truth={data.truth} /> : <span className="text-sm text-muted-foreground">no verdict</span>}
            <WindowNav windows={data.windows} win={win} setWin={setWin} />
            <span className="text-[11px] text-muted-foreground tabular-nums">{data.window_s[0].toFixed(0)}–{data.window_s[1].toFixed(0)} s</span>
          </div>
          <RoomLine room={data.room} trainedOn={data.trained_on} />
          {data.effect && <Inputs data={data} dark={dark} />}
          <AccuracyGrid
            acc={data.accuracy}
            note={<>Each room held out in turn and judged by the model trained on the other three, calibrated with its own empty minutes (an empty minute without itself); scenario C by the env1 + env2 model with env3-1's empties. One seed (sklearn, seed 0). ml_lab cy_41hl_strengths / cy_calib4.</>}
          />
        </>
      )}
    </section>
  );
}

function Inputs({ data, dark }: { data: Calib41; dark: boolean }) {
  const [all, setAll] = useState(false);
  const n = data.features.length;
  const effect = data.effect ?? [];
  const lim = useMemo(() => Math.max(0.5, ...effect.map((v) => (v != null && Number.isFinite(v) ? Math.abs(v) : 0))), [effect]);
  const order = useMemo(
    () => Array.from({ length: n }, (_, i) => i).sort((a, b) => Math.abs(effect[b] ?? 0) - Math.abs(effect[a] ?? 0)),
    [effect, n],
  );
  const rows = all ? order : order.slice(0, TOP);
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
        <span className="text-xs font-medium text-foreground">The {n} inputs, sorted by their push on the verdict</span>
        <Button variant="outline" size="sm" className="h-6 px-2 text-[11px]" onClick={() => setAll(!all)}>
          {all ? `top ${TOP}` : `show all ${n}`}
        </Button>
      </div>
      <div className="overflow-x-auto">
        <table className="text-[11px] tabular-nums">
          <thead>
            <tr className="text-muted-foreground">
              <th className="pr-3 text-left font-normal">column</th>
              <th className="pr-3 text-right font-normal">value</th>
              <th className="pr-3 text-right font-normal">empty median</th>
              <th className="pr-3 text-right font-normal">calibrated</th>
              <th className="pr-3 text-right font-normal">z</th>
              <th className="text-left font-normal">push on the logit</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((i) => {
              const f = data.features[i];
              const lab = data.labels[f];
              const logged = data.logged?.[i];
              return (
                <tr key={f} className="border-t border-border/40">
                  <td className="pr-3 py-0.5" title={lab?.description ?? f}>
                    <span className="text-foreground">{lab?.label ?? f}</span>{" "}
                    <span className="text-muted-foreground">{logged ? "· log" : ""}</span>
                  </td>
                  <td className="pr-3 py-0.5 text-right">{fmt(data.value?.[i])}</td>
                  <td className="pr-3 py-0.5 text-right text-muted-foreground">{fmt(data.centre?.[i])}</td>
                  <td className="pr-3 py-0.5 text-right">
                    {data.imputed?.[i] ? <span className="text-muted-foreground" title="missing: the training median is used">missing</span> : fmt(data.calibrated?.[i])}
                  </td>
                  <td className="pr-3 py-0.5 text-right">{fmt(data.z?.[i])}</td>
                  <td className="py-0.5"><EffectBar v={data.effect?.[i] ?? null} lim={lim} dark={dark} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-muted-foreground leading-relaxed">
        calibrated = value (log where marked) minus the room's empty median (same scale). z = after the model's
        standard scaler. Push: how much the logit falls when that one input is set to its training mean — towards
        PERSON (blue, right) or away (grey, left); the inputs interact, so the pushes do not add up to the logit.
      </p>
    </div>
  );
}
