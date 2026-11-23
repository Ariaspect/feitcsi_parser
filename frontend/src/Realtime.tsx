import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  cancelRealtimeInstall,
  fetchRealtimeStatus,
  installRealtimeReference,
  startRealtime,
  stopRealtime,
  type PresenceRow,
  type RealtimePresence,
  type RealtimeStatus,
} from "./api";
import { Chart } from "./Chart";
import { formatClock } from "./series";
import { firstAtOrAfter } from "./rtseries";

// The Realtime tab: the 13F / 15F presence verdict every 5 s on the last minute of an always-on CSI stream
// (backend/realtime_presence.py, on the frames backend/realtime.py reads). It runs on the server that reaches the
// board, whether or not this tab is open; the tab polls what is new once a second and keeps the last hour.

const POLL_MS = 1000;
const KEEP_S = 3600;
const WINDOWS: [number, string][] = [
  [300, "5 min"],
  [900, "15 min"],
  [3600, "1 h"],
];
const MODEL_COLORS: Record<string, string> = { seq13pw: "#7c3aed", seq15pwu: "#ea580c" };
const OCCUPIED = "#dc2626";
const EMPTY = "#059669";

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

/** Append judgments newer than what is held; drop those older than `keepS` before `now`. */
function mergeJudged(prev: PresenceRow[], rows: PresenceRow[], now: number, keepS: number): PresenceRow[] {
  const lt = prev.length ? prev[prev.length - 1].t : -Infinity;
  const out = prev.concat(rows.filter((r) => r.t > lt));
  const a = firstAtOrAfter(out.map((r) => r.t), now - keepS);
  return a ? out.slice(a) : out;
}

export function Realtime({ dark }: { dark: boolean }) {
  const [status, setStatus] = useState<RealtimeStatus | null>(null);
  const [judged, setJudged] = useState<PresenceRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [win, setWin] = useState(900);
  const [acting, setActing] = useState(false);
  const judgedRef = useRef(judged);
  judgedRef.current = judged;

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
        const j = judgedRef.current;
        // the movement series is not drawn: ask only for what is new since now
        const s = await fetchRealtimeStatus(Date.now() / 1000, KEEP_S, controller.signal,
                                            j.length ? j[j.length - 1].t : 0);
        if (!alive) return;
        setStatus(s);
        setJudged((prev) => mergeJudged(prev, s.presence?.rows ?? [], s.now, KEEP_S));
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

  const now = status?.now ?? Date.now() / 1000;
  const domain: [number, number] = [now - win, now];
  const st = status ? STATE[status.state] : null;
  const presence = status?.presence;
  const streaming = status?.state === "streaming" || status?.state === "capture";

  return (
    <div className="space-y-4" ref={holder}>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-sm font-semibold">Real-time presence</span>
        {st && (
          <span className="rounded px-1.5 py-0.5 text-[11px] text-white" style={{ background: st.color }}>
            {st.label}
          </span>
        )}
        <span className="text-muted-foreground tabular-nums">
          {status?.sampling_rate_hz == null ? "" : `${status.sampling_rate_hz.toFixed(1)} Hz · `}
          {status?.detail}
        </span>
        <span className="ml-auto flex gap-1">
          {presence && (presence.installing ? (
            <Button variant="outline" size="sm" className="h-7 px-2 text-xs" disabled={acting}
                    onClick={() => act(cancelRealtimeInstall)}>
              Cancel install
            </Button>
          ) : (
            <Button variant={presence.reference ? "outline" : "default"} size="sm" className="h-7 px-2 text-xs"
                    disabled={acting || !streaming}
                    title={`Take the next ${presence.install_minutes} complete minutes as the empty-room reference. The room must stay empty.`}
                    onClick={() => act(installRealtimeReference)}>
              {presence.reference ? "Re-install reference" : "Install reference"}
            </Button>
          ))}
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

      {presence && (
        <Presence presence={presence} rows={judged} now={now} domain={domain} width={width} dark={dark}
                  win={win} setWin={setWin} />
      )}

      {status && (
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground">log ({status.log.length})</summary>
          <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap text-[11px] leading-snug">{status.log.join("\n")}</pre>
        </details>
      )}
    </div>
  );
}

function Presence({ presence, rows, now, domain, width, dark, win, setWin }: {
  presence: RealtimePresence;
  rows: PresenceRow[];
  now: number;
  domain: [number, number];
  width: number;
  dark: boolean;
  win: number;
  setWin: (s: number) => void;
}) {
  const shown = rows.filter((r) => r.t >= domain[0]);
  // a line breaks where judgments stop (stream gap, capture hand-off, no reference)
  const times: number[] = [];
  const vals: Record<string, (number | null)[]> = Object.fromEntries(presence.models.map((m) => [m.key, []]));
  shown.forEach((r, i) => {
    if (i && r.t - shown[i - 1].t > 2.5 * presence.hop_s) {
      times.push((r.t + shown[i - 1].t) / 2);
      presence.models.forEach((m) => vals[m.key].push(null));
    }
    times.push(r.t);
    presence.models.forEach((m) => vals[m.key].push(typeof r[m.key] === "number" ? (r[m.key] as number) : null));
  });
  const last = [...rows].reverse().find((r) => presence.models.some((m) => typeof r[m.key] === "number"));
  const stale = last != null && now - last.t > 4 * presence.hop_s;
  const ref = presence.reference;
  const installs = shown.filter((r) => r.install_minute != null);

  return (
    <section className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-2">
        {presence.models.map((m) => {
          const p = last && typeof last[m.key] === "number" ? (last[m.key] as number) : null;
          const occ = p != null && p > presence.threshold;
          const bg = p == null || stale ? "#6b7280" : occ ? OCCUPIED : EMPTY;
          return (
            <div key={m.key} className="rounded-lg p-4 text-white shadow-sm" style={{ background: bg }}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-lg font-semibold">{m.label}</span>
                <span className="text-sm tabular-nums opacity-90">P {p == null ? "—" : p.toFixed(3)}</span>
              </div>
              <div className="mt-1 text-4xl font-bold tracking-wide">
                {p == null ? "—" : occ ? "OCCUPIED" : "EMPTY"}
              </div>
              <div className="mt-2 h-1.5 w-full rounded bg-white/30">
                <div className="h-1.5 rounded bg-white" style={{ width: `${100 * (p ?? 0)}%` }} />
              </div>
              <div className="mt-2 text-[11px] opacity-90 tabular-nums">
                {last ? `window ending ${formatClock(last.t, 60)}${stale ? " — no new judgment" : ""}` : presence.detail}
              </div>
            </div>
          );
        })}
      </div>

      {presence.installing ? (
        <div className="rounded-md border border-amber-500 bg-amber-50 p-2 text-sm text-amber-800 dark:bg-amber-950 dark:text-amber-200">
          Installing the reference: {presence.install_done} of {presence.install_minutes} minutes taken — keep the room
          empty until it finishes (about {presence.install_minutes + 1} min from the press).
        </div>
      ) : !ref ? (
        <div className="rounded-md border border-amber-500 bg-amber-50 p-2 text-sm text-amber-800 dark:bg-amber-950 dark:text-amber-200">
          No reference yet: with the room empty, press Install reference. Until then nothing is judged.
        </div>
      ) : null}

      <div className="space-y-2 rounded-md border p-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
          {presence.models.map((m) => (
            <span key={m.key}><span style={{ color: MODEL_COLORS[m.key] }}>━</span> {m.label} P</span>
          ))}
          <span><span style={{ color: "#b45309", opacity: 0.5 }}>■</span> install minutes</span>
          <span className="ml-auto flex gap-1">
            {WINDOWS.map(([s, label]) => (
              <Button key={s} variant={win === s ? "default" : "outline"} size="sm" className="h-6 px-2 text-[11px]"
                      onClick={() => setWin(s)}>
                {label}
              </Button>
            ))}
          </span>
        </div>
        <Chart
          width={width - 26} dark={dark} height={200} times={times} domain={domain} xMode="clock"
          yDomain={[0, 1]} yLabel="P(occupied), last minute"
          series={presence.models.map((m) => ({ values: vals[m.key], color: MODEL_COLORS[m.key], width: 1.6, label: m.label }))}
          guides={[{ value: presence.threshold, color: "#9aa5b1", label: "0.5" }]}
          bands={installs.map((r) => ({ t0: r.t - presence.window_s, t1: r.t }))} bandColor="#b45309"
        />
        <div className="space-y-0.5 text-[11px] leading-snug text-muted-foreground">
          <div>
            Every {presence.hop_s} s the last {presence.window_s - 1} s window is judged · trained on{" "}
            {presence.trained_on.join(" + ")} · calibrated on this room&apos;s {presence.install_minutes} install empty
            minutes · occupied when the 5-seed mean P &gt; {presence.threshold}. A person shows within about 5 s of
            arriving; &quot;empty&quot; needs a whole empty minute.
          </div>
          {ref && (
            <div>
              Reference: {ref.minutes} minutes from {formatClock(ref.first_wall[0], 60)}, saved{" "}
              {new Date(ref.saved * 1000).toLocaleString()}.
            </div>
          )}
          {presence.compute_s && (
            <div>
              compute per judgment p50 {presence.compute_s.p50.toFixed(2)} s · max {presence.compute_s.max.toFixed(2)} s
            </div>
          )}
          {presence.errors > 0 && <div className="text-destructive">{presence.errors} errors · {presence.detail}</div>}
        </div>
      </div>
    </section>
  );
}
