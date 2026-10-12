import { useCallback, useEffect, useRef, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { fetchCaptureStatus, startCapture, stopCapture, type CaptureRun, type CaptureStatus } from "./api";

// The server does the work (scripts/capture_mtk_hourly.sh, one run per
// capture); this tab only sends start / stop and shows the server's status.
// One second between polls: the countdown is read from here, and a capture
// is a minute long, so nothing finer is needed.
const POLL_MS = 1000;

const STATE_LABEL: Record<string, string> = {
  idle: "idle",
  countdown: "countdown: leave the room",
  starting: "starting: preflight, reaching the board",
  waking: "waking the board over the serial console",
  running: "capturing",
  finishing: "finishing: archive, upload, labelling",
  between: "waiting before the next capture",
  stopping: "stopping",
};

const OUTCOME_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  running: "default",
  done: "secondary",
  cancelled: "outline",
  failed: "destructive",
  skipped: "destructive",
  aborted: "destructive",
  hung: "destructive",
};

function flag(value: boolean | null | undefined, yes: string, no: string, badIsTrue = false) {
  if (value === null || value === undefined) return <Badge variant="outline">checking…</Badge>;
  const good = badIsTrue ? !value : value;
  return <Badge variant={good ? "secondary" : "destructive"}>{value ? yes : no}</Badge>;
}

function clock(ts: number | null | undefined): string {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleTimeString();
}

export function Capture() {
  const [status, setStatus] = useState<CaptureStatus | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const [duration, setDuration] = useState(60);
  const [delay, setDelay] = useState(10);
  const [count, setCount] = useState(10);
  const [gap, setGap] = useState(0);
  const [room, setRoom] = useState("");
  const [scenario, setScenario] = useState("");
  const [note, setNote] = useState("");
  const [upload, setUpload] = useState(true);
  const [label, setLabel] = useState(true);

  const logRef = useRef<HTMLPreElement>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await fetchCaptureStatus();
      setStatus(s);
      setFetchError(null);
    } catch (e) {
      setFetchError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [status?.log]);

  const busy = status?.busy ?? false;

  const begin = async (n: number) => {
    setPending(true);
    setActionError(null);
    try {
      const s = await startCapture({
        duration_s: duration,
        delay_s: delay,
        count: n,
        gap_s: gap,
        room: room.trim() || undefined,
        scenario: scenario.trim() || undefined,
        note: note.trim() || undefined,
        upload,
        label,
      });
      setStatus(s);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  };

  const stop = async () => {
    setPending(true);
    setActionError(null);
    try {
      setStatus(await stopCapture());
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  };

  const pf = status?.preflight;
  const current = status?.current ?? null;
  const progress =
    current && status?.state === "running" && current.duration_s
      ? Math.min(1, Math.max(0, current.elapsed_s / current.duration_s))
      : null;
  const remaining = status?.countdown_remaining_s ?? null;
  const limits = status?.limits ?? {};
  const lim = (key: string, which: 0 | 1, fallback: number) => limits[key]?.[which] ?? fallback;

  return (
    <div className="space-y-4">
      {fetchError && (
        <Alert variant="destructive">
          <AlertTitle>Cannot reach the capture controller</AlertTitle>
          <AlertDescription>{fetchError}</AlertDescription>
        </Alert>
      )}
      {actionError && (
        <Alert variant="destructive">
          <AlertTitle>Request refused</AlertTitle>
          <AlertDescription>{actionError}</AlertDescription>
        </Alert>
      )}
      {status?.error && (
        <Alert variant="destructive">
          <AlertTitle>Last job ended with an error</AlertTitle>
          <AlertDescription>{status.error}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Start a capture</CardTitle>
            <CardDescription>
              Each capture is one run of <code>capture_mtk_hourly.sh</code>: it reaches the board over the cable
              (serial wake if it sleeps), streams the CSI into <code>captures/</code>, grabs webcam frames, then
              uploads and labels on the collection host when it can reach it. Offline, the files stay here and the
              log says so.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="space-y-1">
                <Label htmlFor="cap-duration">Duration (s)</Label>
                <Input id="cap-duration" type="number" min={lim("duration_s", 0, 10)} max={lim("duration_s", 1, 3300)}
                  value={duration} onChange={(e) => setDuration(Number(e.target.value))} disabled={busy} />
              </div>
              <div className="space-y-1">
                <Label htmlFor="cap-delay">Delay before start (s)</Label>
                <Input id="cap-delay" type="number" min={0} max={lim("delay_s", 1, 3600)}
                  value={delay} onChange={(e) => setDelay(Number(e.target.value))} disabled={busy} />
              </div>
              <div className="space-y-1">
                <Label htmlFor="cap-count">Count (for Collect N)</Label>
                <Input id="cap-count" type="number" min={1} max={lim("count", 1, 200)}
                  value={count} onChange={(e) => setCount(Number(e.target.value))} disabled={busy} />
              </div>
              <div className="space-y-1">
                <Label htmlFor="cap-gap">Gap between captures (s)</Label>
                <Input id="cap-gap" type="number" min={0} max={lim("gap_s", 1, 3600)}
                  value={gap} onChange={(e) => setGap(Number(e.target.value))} disabled={busy} />
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="space-y-1">
                <Label htmlFor="cap-room">Room</Label>
                <Input id="cap-room" placeholder="env2" value={room} onChange={(e) => setRoom(e.target.value)} disabled={busy} />
              </div>
              <div className="space-y-1">
                <Label htmlFor="cap-scenario">Scenario</Label>
                <Input id="cap-scenario" placeholder="empty / static sitting / fan rotating …"
                  value={scenario} onChange={(e) => setScenario(e.target.value)} disabled={busy} />
              </div>
              <div className="space-y-1">
                <Label htmlFor="cap-note">Note</Label>
                <Input id="cap-note" placeholder="anything the files cannot tell later"
                  value={note} onChange={(e) => setNote(e.target.value)} disabled={busy} />
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Room, scenario and note are written to <code>&lt;stamp&gt;_meta.json</code> beside each capture when
              given. They are the ground truth the files cannot recover afterwards.
            </p>

            <div className="flex flex-wrap items-center gap-4 text-sm">
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={upload} onChange={(e) => setUpload(e.target.checked)} disabled={busy} />
                upload to the collection host (lg)
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={label} onChange={(e) => setLabel(e.target.checked)} disabled={busy || !upload} />
                label frames on lg after each capture (adds time between captures)
              </label>
            </div>

            <div className="flex flex-wrap gap-2">
              <Button onClick={() => void begin(1)} disabled={busy || pending}>
                Start 1 capture
              </Button>
              <Button variant="secondary" onClick={() => void begin(count)} disabled={busy || pending || count < 1}>
                Collect {count}
              </Button>
              <Button variant="destructive" onClick={() => void stop()} disabled={!busy || pending}>
                Stop
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Stop ends the current capture and the rest of the batch: every local process of the run is killed as
              one group, the board's stream is killed and its driver disarmed, and the partial file is removed here
              and from lg's <code>live/</code>. A capture that already finished is kept.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Status</CardTitle>
            <CardDescription>
              {status ? STATE_LABEL[status.state] ?? status.state : "…"}
              {status?.job && busy && (
                <>
                  {" "}· capture {Math.min(status.runs.length + (status.state === "countdown" || status.state === "between" ? 1 : 0), status.job.count)} of {status.job.count}
                </>
              )}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {remaining !== null && (status?.state === "countdown" || status?.state === "between") && (
              <div className="text-5xl font-semibold tabular-nums">{Math.ceil(remaining)} s</div>
            )}
            {current && status?.state !== "countdown" && (
              <div className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-mono">{current.stamp ?? `pid ${current.pid}`}</span>
                  <span className="tabular-nums text-muted-foreground">
                    {progress !== null && current.duration_s
                      ? `${Math.floor(current.elapsed_s)} / ${current.duration_s} s`
                      : `${Math.floor(current.elapsed_s)} s`}
                  </span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded bg-muted">
                  <div
                    className="h-full bg-primary transition-[width]"
                    style={{ width: `${Math.round((progress ?? 0) * 100)}%` }}
                  />
                </div>
              </div>
            )}

            <div className="flex flex-wrap gap-2 text-sm">
              <span className="flex items-center gap-1">board {flag(pf?.board_reachable, "answers", "silent")}</span>
              <span className="flex items-center gap-1">
                webcam {flag(pf?.webcam_busy, `busy${pf?.webcam_holder ? ` (${pf.webcam_holder})` : ""}`, "free", true)}
              </span>
              <span className="flex items-center gap-1">lg {flag(pf?.remote_reachable, "reachable", "offline")}</span>
              <span className="flex items-center gap-1">
                capture lock {flag(pf?.capture_lock_busy, "held elsewhere", "free", true)}
              </span>
              {pf?.disk_free_mb !== null && pf?.disk_free_mb !== undefined && (
                <Badge variant={pf.disk_free_mb < 3000 ? "destructive" : "outline"}>
                  {(pf.disk_free_mb / 1000).toFixed(1)} GB free
                </Badge>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              A silent board is woken over the serial console (about three minutes). A busy webcam means the
              capture records CSI without frames. An offline lg means the files wait in <code>{status?.capture_dir}</code>.
            </p>

            {status && status.runs.length > 0 && (
              <table className="w-full text-sm">
                <thead className="text-left text-muted-foreground">
                  <tr>
                    <th className="py-1 pr-2 font-normal">#</th>
                    <th className="py-1 pr-2 font-normal">stamp</th>
                    <th className="py-1 pr-2 font-normal">outcome</th>
                    <th className="py-1 font-normal">result</th>
                  </tr>
                </thead>
                <tbody>
                  {status.runs.map((r: CaptureRun) => (
                    <tr key={r.index} className="border-t border-border/50 align-top">
                      <td className="py-1 pr-2 tabular-nums">{r.index + 1}</td>
                      <td className="py-1 pr-2 font-mono">{r.stamp ?? clock(r.started_at)}</td>
                      <td className="py-1 pr-2">
                        <Badge variant={OUTCOME_VARIANT[r.outcome] ?? "outline"}>{r.outcome}</Badge>
                      </td>
                      <td className="py-1 text-xs text-muted-foreground">
                        {r.summary.replace(/^(DONE|FAIL|SKIP|ABORT)\s+/, "")}
                        {r.sidecar ? ` · ${r.sidecar}` : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Log</CardTitle>
          <CardDescription>
            The capture script's own log (<code>mtk_capture.log</code>) from the moment the job started, plus the
            lines this page added (<code>web:</code>).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <pre ref={logRef} className="max-h-72 overflow-auto rounded bg-muted p-3 text-xs leading-5">
            {status?.log.length ? status.log.join("\n") : "(nothing yet)"}
          </pre>
        </CardContent>
      </Card>
    </div>
  );
}
