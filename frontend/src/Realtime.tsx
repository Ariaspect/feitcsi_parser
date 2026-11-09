import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  fetchRealtimeStatus,
  setRealtimeThreshold,
  startRealtime,
  stopRealtime,
  type RealtimeStat,
  type RealtimeStatus,
  type RealtimeTiming,
} from "./api";
import { Chart } from "./Chart";
import { formatClock } from "./series";
import {
  EMPTY_RT,
  alarmDots,
  captureSpans,
  firstAtOrAfter,
  mergeRealtime,
  nearestRow,
  sinceOf,
  yTop,
  type RtSeries,
} from "./rtseries";

// The Realtime tab (backend/realtime.py): a movement value every 0.25 s from an always-on CSI stream, beside the
// one-minute presence verdict. It runs on the server that reaches the board, whether or not this tab is open; the
// tab polls what is new once a second and keeps the last hour.

const POLL_MS = 1000;
const KEEP_S = 3600;
const WINDOWS: [number, string][] = [
  [60, "1 min"],
  [300, "5 min"],
  [900, "15 min"],
  [3600, "1 h"],
];
const LINE = "#0f766e";
const ALARM = "#d62728";

const STATE: Record<RealtimeStatus["state"], { label: string; color: string }> = {
  streaming: { label: "streaming", color: "#15803d" },
  capture: { label: "reading a capture", color: "#2f6fed" },
  starting: { label: "starting", color: "#b45309" },
  waiting: { label: "waiting", color: "#b45309" },
  error: { label: "error", color: "#b91c1c" },
  stopped: { label: "stopped", color: "#6b7280" },
};

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function Realtime({ dark }: { dark: boolean }) {
  const [status, setStatus] = useState<RealtimeStatus | null>(null);
  const [hist, setHist] = useState<RtSeries>(EMPTY_RT);
  const [error, setError] = useState<string | null>(null);
  const [win, setWin] = useState(300);
  const [draft, setDraft] = useState<string | null>(null);
  const [acting, setActing] = useState(false);
  const histRef = useRef(hist);
  histRef.current = hist;

  const holder = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(880);
  useEffect(() => {
    const node = holder.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(320, Math.floor(entry.contentRect.width))));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const s = await fetchRealtimeStatus(sinceOf(histRef.current), KEEP_S, controller.signal);
        if (!alive) return;
        setStatus(s);
        setHist((prev) => mergeRealtime(prev, s, s.now, KEEP_S));
        setError(null);
      } catch (err) {
        if (!alive || controller.signal.aborted) return;
        setError(message(err));
      } finally {
        if (alive) timer = window.setTimeout(poll, POLL_MS);
      }
    };
    poll();
    return () => {
      alive = false;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  const act = async (fn: () => Promise<RealtimeStatus>) => {
    setActing(true);
    try {
      const s = await fn();
      setStatus(s);
      setError(null);
    } catch (err) {
      setError(message(err));
    } finally {
      setActing(false);
    }
  };

  const applyThreshold = () => {
    const v = Number(draft);
    if (draft == null || !Number.isFinite(v)) return;
    act(() => setRealtimeThreshold(v)).then(() => setDraft(null));
  };

  const now = status?.now ?? Date.now() / 1000;
  const domain: [number, number] = [now - win, now];
  const a = firstAtOrAfter(hist.t, domain[0]);
  const t = hist.t.slice(a);
  const x = hist.x.slice(a);
  const src = hist.src.slice(a);
  const b = firstAtOrAfter(hist.rateT, domain[0] - 1);
  const rateT = hist.rateT.slice(b).map((v) => v + 0.5);
  const rate = hist.rate.slice(b);
  const thr = status?.threshold ?? 2.14;
  const top = yTop(x, thr);
  const shown = x.map((v) => (v == null ? null : Math.min(v, top)));
  const alarms = (status?.alarms ?? []).filter((al) => al.t >= domain[0]);
  const hop = status?.signal.hop_s ?? 0.25;
  const rateHz = status?.sampling_rate_hz ?? null;
  const row = status ? nearestRow(status.rate_table, rateHz) : -1;
  const measuredAt = status?.rate_table[0]?.rate_hz ?? 43;
  const st = status ? STATE[status.state] : null;
  const rateTop = Math.max(50, ...rate.map((v) => v * 1.1));

  return (
    <div className="space-y-4" ref={holder}>
      <section className="space-y-3 rounded-md border p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold">Real-time movement</span>
          {st && (
            <span className="rounded px-1.5 py-0.5 text-[11px] text-white" style={{ background: st.color }}>
              {st.label}
            </span>
          )}
          <span className="text-[11px] text-muted-foreground">{status?.detail}</span>
          <span className="ml-auto flex gap-1">
            {status?.enabled ? (
              <Button variant="outline" size="sm" className="h-7 px-2 text-xs" disabled={acting}
                      title="End the stream on the board: the stimulus stops and the board is free"
                      onClick={() => act(stopRealtime)}>
                Stop
              </Button>
            ) : (
              <Button size="sm" className="h-7 px-2 text-xs" disabled={acting || !status} onClick={() => act(startRealtime)}>
                Start
              </Button>
            )}
          </span>
        </div>
        {error && <div className="text-sm text-destructive">{error}</div>}

        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs tabular-nums">
          <span title="CSI frames per second the signal used (dominant AP, both transmit chains, full width), mean of the last 5 s">
            sampling rate{" "}
            <b className="text-sm">{rateHz == null ? "—" : `${rateHz.toFixed(1)} Hz`}</b>
            <span className="text-muted-foreground"> · stimulus ping every {status?.ping_interval_s ?? "—"} s</span>
          </span>
          <span className="flex items-center gap-1.5">
            threshold
            <Input type="number" step={0.01} min={0.5} max={100} className="h-7 w-20 text-sm font-semibold"
                   value={draft ?? thr.toFixed(2)} onChange={(e) => setDraft(e.target.value)}
                   onKeyDown={(e) => { if (e.key === "Enter") applyThreshold(); }} />
            {draft != null && (
              <Button size="sm" className="h-7 px-2 text-xs" disabled={acting} onClick={applyThreshold}>set</Button>
            )}
            {status && thr !== status.default_threshold && (
              <Button variant="outline" size="sm" className="h-7 px-2 text-xs" disabled={acting}
                      onClick={() => act(() => setRealtimeThreshold(status.default_threshold))}>
                {status.default_threshold}
              </Button>
            )}
          </span>
          <span>
            alarms in view <b className="text-sm">{alarms.length}</b>
            {alarms.length > 0 && (
              <span className="text-muted-foreground"> · last {formatClock(alarms[alarms.length - 1].t, 60)}</span>
            )}
          </span>
          <span className="text-muted-foreground">{status ? `${status.frames.toLocaleString()} frames` : ""}</span>
          <span className="ml-auto flex gap-1">
            {WINDOWS.map(([s, label]) => (
              <Button key={s} variant={win === s ? "default" : "outline"} size="sm" className="h-6 px-2 text-[11px]"
                      onClick={() => setWin(s)}>
                {label}
              </Button>
            ))}
          </span>
        </div>
        {rateHz != null && Math.abs(Math.log(rateHz / measuredAt)) > Math.log(1.25) && (
          <div className="text-[11px] text-amber-600">
            The default threshold was measured at {measuredAt} Hz; this stream runs at {rateHz.toFixed(1)} Hz — see the
            table below for what the measurement gave at lower rates.
          </div>
        )}

        <Chart
          width={width - 26} dark={dark} height={230} times={t} domain={domain} xMode="clock"
          yDomain={[0, top]} yLabel="self ratio x = 0.5 s step ÷ 1-frame step"
          series={[{ values: shown, color: LINE, width: 1.2, label: "x" }]}
          guides={[
            { value: thr, color: ALARM, label: "threshold" },
            { value: 1, color: "#9aa5b1", label: "empty level" },
          ]}
          bands={captureSpans(t, src, hop)} bandColor="#2f6fed"
          dots={{ values: alarmDots(t, shown, alarms), colors: t.map(() => ALARM), radius: 3, label: "alarms" }}
        />
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
          <span><span style={{ color: LINE }}>━</span> x, every {hop} s</span>
          <span><span style={{ color: ALARM }}>╍</span> threshold {thr.toFixed(2)} (alarms: x at or above it, at most one per {status?.signal.refractory_s ?? 5} s; judged at the threshold in force then)</span>
          <span><span style={{ color: "#9aa5b1" }}>╍</span> 1 = what an empty room reads</span>
          <span><span style={{ color: "#2f6fed", opacity: 0.5 }}>■</span> read from a running capture's file (the board's stream is the capture's then)</span>
        </div>

        <Chart
          width={width - 26} dark={dark} height={100} times={rateT} domain={domain} xMode="clock"
          yDomain={[0, rateTop]} yLabel="sampling rate (frames/s)"
          series={[{ values: rate, color: "#6b7280", width: 1, label: "rate" }]}
          guides={[{ value: measuredAt, color: "#9aa5b1", label: "measured" }]}
        />
      </section>

      {status && (
        <section className="space-y-2 rounded-md border p-3 text-xs">
          {status.metric && (<>
          <div className="text-sm font-semibold">Judging metric: {status.metric.name}</div>
          <p className="leading-snug text-muted-foreground">
            x = median over the last {status.signal.window_s} s of the {status.signal.lag_s} s step ÷ the same median of
            the one-frame step; a value every {status.signal.hop_s} s, an alarm when x ≥ the threshold (at most one per{" "}
            {status.signal.refractory_s} s). Noise moves the ratio as much in one frame as in half a second, so an empty
            room reads about 1; a body moving moves the half-second step far more.
          </p>
          <ul className="list-disc space-y-0.5 pl-5 leading-snug text-muted-foreground">
            {status.metric.family.map((f) => <li key={f}>{f}</li>)}
          </ul>
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <div className="font-semibold text-emerald-700 dark:text-emerald-400">Separates</div>
              <ul className="list-disc space-y-0.5 pl-5 leading-snug">
                {status.metric.separates.map(([what, how]) => (
                  <li key={what}>{what} <span className="text-muted-foreground">— {how}</span></li>
                ))}
              </ul>
            </div>
            <div>
              <div className="font-semibold text-red-700 dark:text-red-400">Does not separate</div>
              <ul className="list-disc space-y-0.5 pl-5 leading-snug">
                {status.metric.does_not.map(([what, how]) => (
                  <li key={what}>{what} <span className="text-muted-foreground">— {how}</span></li>
                ))}
              </ul>
            </div>
          </div>
          <p className="text-muted-foreground">
            Measured offline on 1,097 labelled minutes of four rooms (2026-11-02). By frame rate (the lower rates simulated
            by dropping frames, with a longer window or lag, which this channel does not use):
          </p>
          </>)}
          <div className="overflow-x-auto">
            <table className="w-full text-right tabular-nums">
              <thead className="text-muted-foreground">
                <tr>
                  <th className="text-left font-normal">rate</th>
                  <th className="font-normal">window / lag</th>
                  <th className="font-normal">threshold</th>
                  <th className="font-normal">empty alarms/h</th>
                  <th className="font-normal">big</th>
                  <th className="font-normal">small</th>
                  <th className="font-normal">still person</th>
                  <th className="font-normal">static fan</th>
                  <th className="font-normal">robot</th>
                  <th className="font-normal">entries found</th>
                  <th className="font-normal">median onset vs camera</th>
                </tr>
              </thead>
              <tbody>
                {status.rate_table.map((r, i) => (
                  <tr key={r.rate_hz} className={i === row ? "bg-muted font-semibold" : ""}>
                    <td className="text-left">{r.rate_hz} Hz{i === row ? " ◀ now" : ""}</td>
                    <td>{r.window_s} / {r.lag_s} s</td>
                    <td>{r.threshold.toFixed(2)}</td>
                    <td>{r.empty_alarms_per_h}</td>
                    <td>{r.big} %</td>
                    <td>{r.small} %</td>
                    <td>{r.still} %</td>
                    <td>{r.static_fan} %</td>
                    <td>{r.robot} %</td>
                    <td>{r.entries}</td>
                    <td>{r.onset_s >= 0 ? "+" : ""}{r.onset_s} s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-muted-foreground">
            % = share of labelled minutes in which it fired. Entries: 8 camera-dated natural entries.
          </p>
          <details>
            <summary className="cursor-pointer text-muted-foreground">log ({status.log.length})</summary>
            <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap text-[11px] leading-snug">{status.log.join("\n")}</pre>
          </details>
        </section>
      )}

      {status?.timing && <Timing timing={status.timing} />}
    </div>
  );
}

const TIMING_ROWS: [keyof RealtimeTiming, string, string, number][] = [
  ["per_frame_us", "per frame — total", "µs", 1],
  ["decode_per_frame_us", "per frame — decode (index, CSI ratio)", "µs", 1],
  ["steps_per_frame_us", "per frame — the two steps", "µs", 1],
  ["per_value_us", "per value — the 0.5 s window medians", "µs", 1],
  ["batch_ms", "per batch (one 0.2 s read)", "ms", 2],
  ["frames_per_batch", "frames per batch", "", 0],
  ["latency_s", "latency: value's time → computed", "s", 2],
];

function Timing({ timing }: { timing: RealtimeTiming }) {
  const cols: (keyof RealtimeStat)[] = ["last", "min", "mean", "p50", "p95", "max", "n"];
  return (
    <section className="space-y-2 rounded-md border p-3 text-xs">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="text-sm font-semibold">Computation time</span>
        <span className="text-muted-foreground">
          on {timing.host} (the laptop that reaches the board — lg only forwards), last {timing.window_s / 60} min,{" "}
          {timing.batches.toLocaleString()} batches · CPU {timing.cpu_percent == null ? "—" : `${timing.cpu_percent.toFixed(2)} %`} of one core
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-right tabular-nums">
          <thead className="text-muted-foreground">
            <tr>
              <th className="text-left font-normal" />
              {cols.map((c) => <th key={c} className="font-normal">{c === "mean" ? "avg" : c}</th>)}
            </tr>
          </thead>
          <tbody>
            {TIMING_ROWS.map(([key, label, unit, digits]) => {
              const st = timing[key] as RealtimeStat | null;
              return (
                <tr key={key}>
                  <td className="text-left">{label}{unit ? ` (${unit})` : ""}</td>
                  {cols.map((c) => (
                    <td key={c}>{st == null ? "—" : c === "n" ? st.n.toLocaleString() : st[c].toFixed(digits)}</td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-muted-foreground leading-snug">
        Frames are handled in batches, one per 0.2 s read of the stream; a per-frame time is a batch's time over its
        frames, a per-value time its window medians over its values. Latency counts the stream's transfer, the 0.2 s read
        interval and the computation together.
      </p>
    </section>
  );
}
