import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { fetchCaptureStatus, startCapture, stopCapture, type CaptureRun, type CaptureStatus } from "./api";

// The capture controls live in the toolbar, next to MIMO, always visible
// (user, 2026-10-12: not a tab). <CaptureBar /> is that compact row;
// <CaptureDetails /> is the panel it unfolds under the header, with the
// scenario fields, the per-capture table and the script's log. One provider
// polls the server once a second for both.
//
// The server does the work (scripts/capture_mtk_hourly.sh, one run per
// capture); this file only sends start / stop and shows the server's status.
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

interface Form {
  duration: number;
  delay: number;
  count: number;
  gap: number;
  room: string;
  scenario: string;
  note: string;
  upload: boolean;
  label: boolean;
}

interface CaptureCtx {
  status: CaptureStatus | null;
  fetchError: string | null;
  actionError: string | null;
  pending: boolean;
  busy: boolean;
  form: Form;
  setForm: (patch: Partial<Form>) => void;
  begin: (count: number) => Promise<void>;
  stop: () => Promise<void>;
  open: boolean;
  setOpen: (v: boolean) => void;
}

const Ctx = createContext<CaptureCtx | null>(null);

function useCapture(): CaptureCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("CaptureBar / CaptureDetails must sit inside <CaptureProvider>");
  return ctx;
}

export function CaptureProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<CaptureStatus | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [open, setOpen] = useState(false);
  const [form, setFormState] = useState<Form>({
    duration: 60, delay: 10, count: 10, gap: 0, room: "", scenario: "", note: "", upload: true, label: true,
  });
  const setForm = useCallback((patch: Partial<Form>) => setFormState((f) => ({ ...f, ...patch })), []);

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

  const begin = useCallback(async (count: number) => {
    setPending(true);
    setActionError(null);
    try {
      const s = await startCapture({
        duration_s: form.duration,
        delay_s: form.delay,
        count,
        gap_s: form.gap,
        room: form.room.trim() || undefined,
        scenario: form.scenario.trim() || undefined,
        note: form.note.trim() || undefined,
        upload: form.upload,
        label: form.label,
      });
      setStatus(s);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
      setOpen(true);
    } finally {
      setPending(false);
    }
  }, [form]);

  const stop = useCallback(async () => {
    setPending(true);
    setActionError(null);
    try {
      setStatus(await stopCapture());
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
      setOpen(true);
    } finally {
      setPending(false);
    }
  }, []);

  const value: CaptureCtx = {
    status, fetchError, actionError, pending, busy: status?.busy ?? false, form, setForm, begin, stop, open, setOpen,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

function dot(value: boolean | null | undefined, goodWhen: boolean, title: string) {
  const colour = value === null || value === undefined ? "bg-muted-foreground/40" : value === goodWhen ? "bg-emerald-500" : "bg-red-500";
  return <span key={title} title={title} className={`inline-block h-2 w-2 rounded-full ${colour}`} />;
}

function SmallField({ id, label, children }: { id: string; label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <Label htmlFor={id} className="text-[10px] text-muted-foreground uppercase tracking-wide">{label}</Label>
      {children}
    </div>
  );
}

/** The always-visible row in the toolbar: three numbers, three buttons, one status chip. */
export function CaptureBar() {
  const { status, busy, pending, form, setForm, begin, stop, open, setOpen, fetchError } = useCapture();
  const pf = status?.preflight;
  const cur = status?.current ?? null;
  const remaining = status?.countdown_remaining_s;

  let chip: ReactNode;
  if (fetchError) chip = <Badge variant="destructive" title={fetchError}>controller unreachable</Badge>;
  else if (!status) chip = <Badge variant="outline">…</Badge>;
  else if (status.state === "countdown" || status.state === "between")
    chip = <Badge variant="default" className="tabular-nums">{status.state === "between" ? "next in" : "starts in"} {Math.ceil(remaining ?? 0)} s</Badge>;
  else if (status.state === "running" && cur)
    chip = <Badge variant="default" className="tabular-nums font-mono">{cur.stamp ?? "…"} {Math.floor(cur.elapsed_s)}/{cur.duration_s ?? "?"} s</Badge>;
  else if (status.busy) chip = <Badge variant="default">{STATE_LABEL[status.state] ?? status.state}</Badge>;
  else if (status.error) chip = <Badge variant="destructive" title={status.error}>error</Badge>;
  else if (status.runs.length) chip = <Badge variant="secondary">{status.runs.filter((r) => r.outcome === "done").length}/{status.runs.length} done</Badge>;
  else chip = <Badge variant="outline">idle</Badge>;

  const dots = pf ? (
    <span className="flex items-center gap-1" title="board · webcam · lg · capture lock">
      {dot(pf.board_reachable, true, pf.board_reachable ? "board answers" : "board silent (serial wake on start)")}
      {dot(pf.webcam_busy, false, pf.webcam_busy ? `webcam busy${pf.webcam_holder ? ` (${pf.webcam_holder})` : ""}` : "webcam free")}
      {dot(pf.remote_reachable, true, pf.remote_reachable ? "lg reachable" : "lg offline: files wait locally")}
      {dot(pf.capture_lock_busy, false, pf.capture_lock_busy ? "a capture is running outside this page" : "capture lock free")}
    </span>
  ) : null;

  return (
    <div className="flex items-end gap-1.5">
      <SmallField id="cap-duration" label="Len s">
        <Input id="cap-duration" type="number" min={10} max={3300} className="w-16 h-8" value={form.duration}
          onChange={(e) => setForm({ duration: Number(e.target.value) })} disabled={busy} />
      </SmallField>
      <SmallField id="cap-delay" label="Delay s">
        <Input id="cap-delay" type="number" min={0} max={3600} className="w-16 h-8" value={form.delay}
          onChange={(e) => setForm({ delay: Number(e.target.value) })} disabled={busy} />
      </SmallField>
      <SmallField id="cap-count" label="N">
        <Input id="cap-count" type="number" min={1} max={200} className="w-14 h-8" value={form.count}
          onChange={(e) => setForm({ count: Number(e.target.value) })} disabled={busy} />
      </SmallField>
      <SmallField id="cap-start" label="Collect">
        <div className="flex items-center gap-1">
          <Button id="cap-start" size="sm" className="h-8" onClick={() => void begin(1)} disabled={busy || pending}
            title={`one capture of ${form.duration} s after a ${form.delay} s countdown`}>
            Start
          </Button>
          <Button size="sm" variant="secondary" className="h-8" onClick={() => void begin(form.count)} disabled={busy || pending || form.count < 1}
            title={`${form.count} captures of ${form.duration} s, back to back`}>
            ×{form.count}
          </Button>
          <Button size="sm" variant="destructive" className="h-8" onClick={() => void stop()} disabled={!busy || pending}
            title="stop the current capture and the rest of the batch; the board is disarmed and the partial file removed">
            Stop
          </Button>
          {chip}
          {dots}
          <Button variant="ghost" size="sm" className="h-8 px-2" onClick={() => setOpen(!open)} aria-expanded={open}
            title={open ? "hide capture details" : "scenario, notes, per-capture results, log"}>
            {open ? "▴" : "▾"}
          </Button>
        </div>
      </SmallField>
    </div>
  );
}

/** The panel under the header: scenario fields, status, per-capture table, log. Shown when the bar's ▾ is open. */
export function CaptureDetails() {
  const { status, busy, form, setForm, fetchError, actionError, open } = useCapture();
  const logRef = useRef<HTMLPreElement>(null);
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [status?.log, open]);
  if (!open) return null;

  const pf = status?.preflight;
  const current = status?.current ?? null;
  const progress =
    current && status?.state === "running" && current.duration_s
      ? Math.min(1, Math.max(0, current.elapsed_s / current.duration_s))
      : null;

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
            <CardTitle>Capture settings</CardTitle>
            <CardDescription>
              Each capture is one run of <code>capture_mtk_hourly.sh</code> on {status?.host ?? "the capture laptop"}
              {status?.proxied_via ? <> (this page forwards to it)</> : null}: it reaches the board over the cable
              (serial wake if it sleeps), streams the CSI into <code>captures/</code> at one ping every{" "}
              {status?.ping_interval ?? 0.02} s (about 43 Hz), grabs webcam frames, then uploads and labels on lg
              when it can reach it. Offline, the files stay on the laptop and the log says so.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
              <div className="space-y-1">
                <Label htmlFor="cap-gap">Gap between captures (s)</Label>
                <Input id="cap-gap" type="number" min={0} max={3600} value={form.gap}
                  onChange={(e) => setForm({ gap: Number(e.target.value) })} disabled={busy} />
              </div>
              <div className="space-y-1">
                <Label htmlFor="cap-room">Room</Label>
                <Input id="cap-room" placeholder="env2" value={form.room} onChange={(e) => setForm({ room: e.target.value })} disabled={busy} />
              </div>
              <div className="space-y-1">
                <Label htmlFor="cap-scenario">Scenario</Label>
                <Input id="cap-scenario" placeholder="empty / static sitting / fan rotating …"
                  value={form.scenario} onChange={(e) => setForm({ scenario: e.target.value })} disabled={busy} />
              </div>
              <div className="space-y-1">
                <Label htmlFor="cap-note">Note</Label>
                <Input id="cap-note" placeholder="anything the files cannot tell later"
                  value={form.note} onChange={(e) => setForm({ note: e.target.value })} disabled={busy} />
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Room, scenario and note are written to <code>&lt;stamp&gt;_meta.json</code> beside each capture when
              given. They are the ground truth the files cannot recover afterwards.
            </p>
            <div className="flex flex-wrap items-center gap-4 text-sm">
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={form.upload} onChange={(e) => setForm({ upload: e.target.checked })} disabled={busy} />
                upload to the collection host (lg)
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={form.label} onChange={(e) => setForm({ label: e.target.checked })} disabled={busy || !form.upload} />
                label frames on lg after each capture (adds time between captures)
              </label>
            </div>
            <p className="text-xs text-muted-foreground">
              Stop ends the current capture and the rest of the batch: every local process of the run is killed as
              one group, the board's stream is killed and its driver disarmed, and the partial file is removed on the
              laptop and from lg's <code>live/</code>. A capture that already finished is kept.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Status</CardTitle>
            <CardDescription>
              {status ? STATE_LABEL[status.state] ?? status.state : "…"}
              {status?.job && busy && (
                <> · capture {Math.min(status.runs.length + (status.state === "countdown" || status.state === "between" ? 1 : 0), status.job.count)} of {status.job.count}</>
              )}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
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
                  <div className="h-full bg-primary transition-[width]" style={{ width: `${Math.round((progress ?? 0) * 100)}%` }} />
                </div>
              </div>
            )}
            <div className="flex flex-wrap gap-2 text-sm">
              <span>board {pf?.board_reachable == null ? "…" : pf.board_reachable ? "answers" : "silent"}</span>
              <span>· webcam {pf?.webcam_busy == null ? "…" : pf.webcam_busy ? `busy${pf.webcam_holder ? ` (${pf.webcam_holder})` : ""}` : "free"}</span>
              <span>· lg {pf?.remote_reachable == null ? "…" : pf.remote_reachable ? "reachable" : "offline"}</span>
              <span>· lock {pf?.capture_lock_busy == null ? "…" : pf.capture_lock_busy ? "held elsewhere" : "free"}</span>
              {pf?.disk_free_mb != null && <span>· {(pf.disk_free_mb / 1000).toFixed(1)} GB free</span>}
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
                      <td className="py-1 pr-2 font-mono">{r.stamp ?? new Date(r.started_at * 1000).toLocaleTimeString()}</td>
                      <td className="py-1 pr-2"><Badge variant={OUTCOME_VARIANT[r.outcome] ?? "outline"}>{r.outcome}</Badge></td>
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
            lines the server added (<code>web:</code>).
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
