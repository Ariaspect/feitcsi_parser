import { useCallback, useEffect, useRef, useState } from "react";
import { fetchCaptures, fetchDoppler, fetchFilters, fetchMeta, formatBytes, truncateCaptureName, type CaptureFile, type DopplerMetric, type Filters, type Meta } from "./api";
import { TWILIGHT } from "./colormap";
import { Heatmap } from "./Heatmap";
import { Presence } from "./Presence";
import { pickMimo } from "./filters";
import { createTimeLink } from "./timelink";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { PlayIcon, PauseIcon, SunIcon, MoonIcon, SplineIcon, ArrowLeftRightIcon, ChevronRightIcon } from "lucide-react";

const DEFAULT_PATH = "captures/capture.dat";
const DEFAULT_REFRESH_MS = 300;
// Characters the capture trigger can show at w-72 / text-sm. Every capture
// name in the repo fits whole; longer nested ones lose their middle.
const CAPTURE_NAME_CHARS = 30;
// The dropdown list is allowed to grow past the trigger, so it can afford a
// much longer name before eliding.
const CAPTURE_LIST_CHARS = 44;
const DARK_KEY = "feitcsi-dark";

// Preferred transmitter to show on load. 'all' interleaves every sender in
// the capture, and since two devices here transmit at roughly equal rates,
// consecutive frames come from the same one only ~18% of the time. That is
// harmless for the per-frame panels but makes anything read along the time
// axis meaningless, so a single transmitter is the safer default.
//
// Falls back to 'all' when this address is absent from the loaded capture —
// see the filters effect. A capture from a different rig has different MACs,
// and defaulting to one that is not there would show an empty plot.
const DEFAULT_SOURCE_MAC = "08:bf:b8:95:80:04";

/** A panel that starts folded away.
 *
 * The two secondary views are read occasionally, against the panels that stay
 * open — not on every visit. Folded, they cost a header row instead of a
 * screen of scrolling, and the reader chooses when to spend the space.
 *
 * `keepMounted` is load-bearing: the Heatmap inside subscribes to the shared
 * time link on mount, so unmounting it on fold would leave it at full extent
 * while its neighbours stayed zoomed, and unfolding would show a window that
 * does not match the panel above it. Mounted but display:none it keeps
 * tracking, and Heatmap declines to fetch at zero width, so a folded panel
 * still costs nothing to hold open in the DOM.
 */
function FoldedPanel({
  title,
  hint,
  children,
}: {
  title: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <Collapsible>
      <CollapsibleTrigger className="group flex w-full items-center gap-2 rounded-md border px-3 py-2 text-left transition-colors hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
        <ChevronRightIcon className="size-4 shrink-0 text-muted-foreground transition-transform duration-200 group-data-[panel-open]:rotate-90" />
        <span className="text-sm font-medium">{title}</span>
        <span className="ml-auto hidden truncate text-xs text-muted-foreground sm:inline">{hint}</span>
      </CollapsibleTrigger>
      <CollapsibleContent keepMounted className="pt-4 data-[closed]:hidden">
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
}

/** What a Doppler panel needs to label its frequency axis: how many rows the
 *  server sent, and what the bottom and top rows mean in hertz. */
interface DopplerGeom {
  rows: number;
  fMin: number;
  fMax: number;
}

export function App() {
  const [path, setPath] = useState(DEFAULT_PATH);
  const [refreshMs, setRefreshMs] = useState(DEFAULT_REFRESH_MS);
  const [running, setRunning] = useState(false);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);
  const [filters, setFilters] = useState<Filters | null>(null);
  const [captures, setCaptures] = useState<CaptureFile[] | null>(null);
  const [mimo, setMimo] = useState<string>("all");
  // A default applies until the user overrides it. Without this, picking 'all'
  // deliberately and then loading another capture would snap the selection
  // back to 2x1 and quietly fight the user.
  const mimoTouched = useRef(false);
  const [sourceMac, setSourceMac] = useState<string>(DEFAULT_SOURCE_MAC);
  // Linear interpolation of gaps in both axes: structural nulls (pilots,
  // DC/guard band) along subcarrier, and sampling gaps along time. On by
  // default to match the backend. Off shows the data exactly as decoded off
  // the wire, NaN gaps included -- useful for judging what interpolation is
  // actually doing to a given capture.
  const [interpolate, setInterpolate] = useState<boolean>(true);
  // STFT window, in seconds rather than frames: frame rate runs 5-18 Hz across
  // captures, so a fixed frame count would mean a different physical window on
  // every file. Longer window = finer frequency resolution, fewer columns.
  const [winSeconds, setWinSeconds] = useState<number>(10);
  // Row count and Nyquist are properties of the capture's own frame rate, so
  // they are not known until the first response comes back.
  // Geometry per Doppler panel, because the two no longer share it. A real
  // metric's spectrum is conjugate-symmetric and served one-sided; the complex
  // ratio is served two-sided, so it has about twice the rows and an axis that
  // starts below zero. One `rows` for both would stretch one of them.
  const [dopplerGeom, setDopplerGeom] = useState<{
    complex: DopplerGeom;
    phase: DopplerGeom;
  } | null>(null);
  // Which capture identity the geometry above describes, and whether a probe
  // for it is already in flight. See the effect that fills it in.
  const dopplerGeomKeyRef = useRef<string | null>(null);
  const dopplerGeomInFlightRef = useRef(false);
  // Swap correction on the CSI ratio panels. Some frames arrive with the rx
  // streams exchanged (ratio reciprocal) or the ratio negated (phase +pi);
  // correction puts them back. Off by default so what the panels show is what
  // the NIC delivered: the affected frames read as vertical discontinuities,
  // but nothing has been inferred on the viewer's behalf. Turn it on to hide
  // that artefact once you know it is one. Detection compares each frame
  // against its neighbours, so it needs one transmitter's own sequence -- on
  // `all` the backend declines to act and the panels say so themselves.
  const [swapCorrected, setSwapCorrected] = useState<boolean>(false);
  const [dark, setDark] = useState<boolean>(() => {
    const stored = localStorage.getItem(DARK_KEY);
    return stored === null ? true : stored === "true";
  });

  // What the ratio panels actually show. Correction needs one transmitter's
  // own frame sequence to compare neighbours against, so on `all` it cannot
  // act whatever the toggle says. Deriving metric, panel title and button
  // label from this one flag keeps the three from disagreeing.
  const swapActive = swapCorrected && sourceMac !== "all";

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

  // Learn the spectrogram's row count and frequency ceiling before rendering a
  // panel: Heatmap needs a row count up front, and both fall out of the
  // capture's median frame rate rather than anything the client picks. The
  // panel's own fetch then hits the backend block cache this warmed.
  //
  // The probe belongs to the capture's identity, not to the poll. Keying it on
  // the live extent instead re-ran it every refresh and its cleanup aborted the
  // request in flight; a full-extent Doppler decode takes longer than a poll
  // period, so the geometry never arrived and the Doppler tab stayed empty for
  // as long as the capture was live. Retry while it is unknown, one probe at a
  // time, and stop once it is learned -- neither the row count nor the Nyquist
  // ceiling moves as the capture grows.
  useEffect(() => {
    const key = `${path}|${winSeconds}|${mimo}|${sourceMac}`;
    if (dopplerGeomKeyRef.current !== key) {
      dopplerGeomKeyRef.current = key;
      setDopplerGeom(null);
    }
    if (!meta || meta.total_frames <= 0) return;
    if (dopplerGeom !== null) return;
    // Never abort a probe in flight: it is the expensive request, and the next
    // poll is 300ms away.
    if (dopplerGeomInFlightRef.current) return;
    dopplerGeomInFlightRef.current = true;
    const probe = (m: DopplerMetric) =>
      fetchDoppler(path, meta.t_min, meta.t_max, m, winSeconds,
                   undefined, mimo, sourceMac);
    Promise.all([probe("csi_ratio_complex"), probe("csi_ratio_phase_time_unwrapped")])
      .then(([cx, ph]) => {
        if (dopplerGeomKeyRef.current === key) {
          setDopplerGeom({
            complex: { rows: cx.height, fMin: cx.fMin, fMax: cx.fMax },
            phase: { rows: ph.height, fMin: ph.fMin, fMax: ph.fMax },
          });
        }
      })
      .catch(() => {
        // A range too short for the window is an ordinary state here, not an
        // error worth taking over the page: the panel shows its own message.
        // Leaving the geometry unknown means the next poll retries.
      })
      .finally(() => {
        dopplerGeomInFlightRef.current = false;
      });
  }, [path, winSeconds, mimo, sourceMac, meta, dopplerGeom]);

  const dopplerSource = useCallback(
    (dopplerMetric: DopplerMetric) =>
      (t0: number, t1: number, _width: number, signal: AbortSignal) =>
        // _width is deliberately ignored: column count follows the window and
        // hop, and letting a pixel width drive the sample rate is what
        // manufactures peaks above a capture's own Nyquist.
        fetchDoppler(path, t0, t1, dopplerMetric, winSeconds, signal, mimo, sourceMac, interpolate),
    [path, winSeconds, mimo, sourceMac, interpolate],
  );

  useEffect(() => {
    let cancelled = false;
    setFilters(null);
    fetchFilters(path)
      .then((f) => {
        if (cancelled) return;
        setFilters(f);
        // Keep the selection honest for whatever capture just loaded: a MAC
        // that is not in this file would filter every frame away and draw an
        // empty plot with nothing to explain it. Applies to the default on
        // first load and to a user's pick surviving a path change alike.
        setSourceMac((prev) =>
          prev === "all" || f.source_macs.includes(prev) ? prev : "all",
        );
        // Same honesty check for MIMO, plus a convenience default. See
        // filters.ts for why a default stops applying once the user chooses.
        setMimo((prev) => pickMimo(prev, f.mimo_modes, mimoTouched.current));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [path]);

  useEffect(() => {
    let cancelled = false;
    fetchCaptures()
      .then((c) => {
        if (!cancelled) setCaptures(c);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const toggleDark = () => {
    setDark((d) => {
      const next = !d;
      localStorage.setItem(DARK_KEY, String(next));
      return next;
    });
  };

  const captureItems = (captures ?? []).map((c) => ({
    // The list gets far more room than the trigger (the popup sizes to its
    // content below), so it shows the size too and only elides a name long
    // enough to beat even that.
    label: `${truncateCaptureName(c.filename, CAPTURE_LIST_CHARS)}  (${formatBytes(c.size_bytes)})`,
    value: c.path,
  }));

  const mimoItems = [
    { label: "all", value: "all" },
    ...(filters?.mimo_modes.map((m) => ({ label: m, value: m })) ?? []),
  ];
  const macItems = [
    { label: "all", value: "all" },
    ...(filters?.source_macs.map((mac) => ({ label: mac, value: mac })) ?? []),
  ];

  return (
    <div className={`${dark ? "dark " : ""}font-sans bg-background text-foreground min-h-screen`}>
      <div className="sticky top-0 z-50 bg-background/80 backdrop-blur-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-2.5 flex flex-wrap gap-3 items-center">
          <h1 className="text-base font-bold tracking-tight mr-auto">FeitCSI Heatmap</h1>

          <div className="flex flex-col gap-0.5">
            <Label htmlFor="path" className="text-[10px] text-muted-foreground uppercase tracking-wide">Capture</Label>
            <Select
              value={path}
              onValueChange={(v) => v && setPath(v)}
              items={captureItems}
            >
              <SelectTrigger id="path" className="w-72 h-8" size="sm">
                {/* The trigger formats the selection itself rather than taking
                    the list's label: it drops the size (the list still shows
                    it) to spend the width on the name, and elides from the
                    middle. These captures differ in their tails, so the CSS
                    default of cutting the end throws away the only part that
                    tells them apart. */}
                <SelectValue placeholder={captures === null ? "Loading…" : "Select a .dat file"}>
                  {(value) => {
                    const cap = (captures ?? []).find((c) => c.path === value);
                    if (!cap) {
                      return captures === null ? "Loading…" : "Select a .dat file";
                    }
                    return truncateCaptureName(cap.filename, CAPTURE_NAME_CHARS);
                  }}
                </SelectValue>
              </SelectTrigger>
              {/* The shared popup is pinned to the trigger width
                  (w-(--anchor-width)) with overflow-x-hidden, which clipped
                  every capture label. Widths are set inline rather than by
                  class so they beat that rule outright instead of depending on
                  how twMerge resolves two width utilities. */}
              <SelectContent
                style={{
                  width: "auto",
                  minWidth: "var(--anchor-width)",
                  maxWidth: "min(34rem, calc(100vw - 2rem))",
                }}
              >
                <SelectGroup>
                  {captureItems.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col gap-0.5">
            <Label htmlFor="refresh" className="text-[10px] text-muted-foreground uppercase tracking-wide">ms</Label>
            <Input
              id="refresh"
              type="number"
              min={50}
              max={10000}
              value={refreshMs}
              onChange={(e) => setRefreshMs(Number(e.target.value))}
              className="w-20 h-8"
            />
          </div>

          <Button
            onClick={() => setRunning((r) => !r)}
            variant={running ? "destructive" : "default"}
            size="sm"
          >
            {running ? (
              <>
                <PauseIcon data-icon="inline-start" />
                Pause
              </>
            ) : (
              <>
                <PlayIcon data-icon="inline-start" />
                Live
              </>
            )}
          </Button>

          <div className="flex flex-col gap-0.5">
            <Label htmlFor="mimo" className="text-[10px] text-muted-foreground uppercase tracking-wide">MIMO</Label>
            <Select
              value={mimo}
              onValueChange={(v) => {
                mimoTouched.current = true;
                setMimo(v ?? "all");
              }}
              items={mimoItems}
              disabled={!filters || filters.mimo_modes.length <= 1}
            >
              <SelectTrigger id="mimo" className="w-20 h-8" size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {mimoItems.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col gap-0.5">
            <Label htmlFor="source-mac" className="text-[10px] text-muted-foreground uppercase tracking-wide">MAC</Label>
            <Select
              value={sourceMac}
              onValueChange={(v) => setSourceMac(v ?? "all")}
              items={macItems}
              disabled={!filters || filters.source_macs.length <= 1}
            >
              <SelectTrigger id="source-mac" className="w-44 h-8" size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {macItems.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>

          <Button
            variant={interpolate ? "default" : "outline"}
            size="sm"
            onClick={() => setInterpolate((v) => !v)}
            className="h-8"
            title="Linearly interpolate gaps in subcarrier (structural nulls) and time (sampling gaps)"
          >
            <SplineIcon data-icon="inline-start" />
            Interpolate {interpolate ? "On" : "Off"}
          </Button>

          <Button
            variant={swapActive ? "default" : "outline"}
            size="sm"
            onClick={() => setSwapCorrected((v) => !v)}
            disabled={sourceMac === "all"}
            className="h-8"
            title={
              sourceMac === "all"
                ? "Select a single MAC to correct swaps. Detection compares each frame against its neighbours, and on all senders consecutive frames rarely come from the same one."
                : "Put back frames whose rx streams arrived exchanged, or whose ratio arrived negated. Applies to the CSI ratio panels."
            }
          >
            <ArrowLeftRightIcon data-icon="inline-start" />
            Swap Correction {swapActive ? "On" : "Off"}
          </Button>

          <Button variant="outline" size="icon" onClick={toggleDark} className="h-8 w-8">
            {dark ? <SunIcon /> : <MoonIcon />}
          </Button>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-4 space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {meta && (
          <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
            <Badge variant="secondary">{meta.filename}</Badge>
            <span>{meta.chipset} / {meta.bandwidth} MHz</span>
            <Separator orientation="vertical" className="h-4" />
            <span>{meta.total_frames.toLocaleString()} frames</span>
            <span>{meta.num_subcarriers} subcarriers</span>
            <span>[{meta.t_min.toFixed(3)}, {meta.t_max.toFixed(3)}] s</span>
            {lastUpdate && (
              <>
                <Separator orientation="vertical" className="h-4" />
                <span>{new Date(lastUpdate).toLocaleTimeString()}</span>
              </>
            )}
            {mimo !== "all" && (
              <>
                <Separator orientation="vertical" className="h-4" />
                <Badge variant="outline">MIMO: {mimo}</Badge>
              </>
            )}
            {sourceMac !== "all" && (
              <Badge variant="outline">{sourceMac}</Badge>
            )}
          </div>
        )}

        {meta && meta.total_frames > 0 ? (
          <Tabs defaultValue="channel">
            <TabsList>
              <TabsTrigger value="channel">Channel</TabsTrigger>
              <TabsTrigger value="doppler">Doppler</TabsTrigger>
              <TabsTrigger value="presence">Motion &amp; presence</TabsTrigger>
            </TabsList>

            <TabsContent value="channel">
              <div className="space-y-4">
            <Heatmap
              path={path}
              metric="amplitude"
              filename={meta.filename}
              numSubcarriers={meta.num_subcarriers}
              captureTMin={meta.t_min}
              captureTMax={meta.t_max}
              title="Amplitude"
              colorLabel="Amplitude (dBm)"
              height={320}
              timeLink={timeLink}
              mimo={mimo}
              sourceMac={sourceMac}
              interpolate={interpolate}
              dark={dark}
            />
            <FoldedPanel
              title="Phase (rx0)"
              hint="raw wrapped phase — per-packet offset and slope not removed"
            >
              <Heatmap
                path={path}
                metric="phase"
                filename={meta.filename}
                numSubcarriers={meta.num_subcarriers}
                captureTMin={meta.t_min}
                captureTMax={meta.t_max}
                minValue={-Math.PI}
                maxValue={Math.PI}
                title="Phase"
                colorLabel="Phase (rad)"
                height={320}
                palette={TWILIGHT}
                timeLink={timeLink}
                mimo={mimo}
                sourceMac={sourceMac}
                interpolate={interpolate}
                dark={dark}
              />
            </FoldedPanel>

            <Heatmap
              path={path}
              metric={swapActive ? "csi_ratio_amplitude_corrected" : "csi_ratio_amplitude"}
              filename={meta.filename}
              numSubcarriers={meta.num_subcarriers}
              captureTMin={meta.t_min}
              captureTMax={meta.t_max}
              title={`CSI Ratio Amplitude (rx1/rx0)${swapActive ? " — Swap-Corrected" : ""}`}
              colorLabel="Ratio amp (dB)"
              height={320}
              timeLink={timeLink}
              mimo={mimo}
              sourceMac={sourceMac}
              interpolate={interpolate}
              dark={dark}
            />
            <Heatmap
              path={path}
              metric={swapActive ? "csi_ratio_phase_corrected" : "csi_ratio_phase"}
              filename={meta.filename}
              numSubcarriers={meta.num_subcarriers}
              captureTMin={meta.t_min}
              captureTMax={meta.t_max}
              minValue={-Math.PI}
              maxValue={Math.PI}
              title={`CSI Ratio Phase (rx1/rx0)${swapActive ? " — Swap-Corrected" : ""}`}
              colorLabel="Ratio phase (rad)"
              height={320}
              palette={TWILIGHT}
              timeLink={timeLink}
              mimo={mimo}
              sourceMac={sourceMac}
              interpolate={interpolate}
              dark={dark}
            />

            <FoldedPanel
              title="CSI Ratio Phase — Time-Unwrapped (rx1/rx0)"
              hint="accumulated phase, per segment — not an angle"
            >
              <p className="pb-4 text-xs text-muted-foreground">
                Unwrapped along <i>time</i> on the raw ratio: continuous phase
                accumulated as the channel moves. Restarts after a capture gap
                and anchors each segment at its own start, so the value is phase
                change since that segment began — segments are not comparable
                with one another. Shares the phase palette above for comparison,
                though accumulated phase is not an angle. Unwrapping is the risk
                this view takes: a step it misreads offsets that subcarrier by
                2π for the rest of the capture. On MediaTek the swaps that used
                to cause those are absent, and what is left — 0.027% of
                subcarrier-transitions past 0.9π — is fast motion, not decode.
              </p>
              <Heatmap
                path={path}
                metric="csi_ratio_phase_time_unwrapped"
                filename={meta.filename}
                numSubcarriers={meta.num_subcarriers}
                captureTMin={meta.t_min}
                captureTMax={meta.t_max}
                title="CSI Ratio Phase — Time-Unwrapped (rx1/rx0)"
                colorLabel="Accumulated phase (rad)"
                height={320}
                palette={TWILIGHT}
                timeLink={timeLink}
                mimo={mimo}
                sourceMac={sourceMac}
                interpolate={interpolate}
                dark={dark}
              />
            </FoldedPanel>

            <Separator className="my-2" />
            <p className="text-xs text-muted-foreground">
              Channel impulse response: the raw channel (rx0/tx0), not the
              ratio, inverse-FFT'd from subcarrier into delay. Rows are delay
              taps, not subcarriers — re-centred onto the same middle-of-row
              axis the panels above use, but unlike those, the peak here is
              <i>not</i> zero-delay: a single channel has no CFO/SFO
              cancellation, so it sits off-centre by this capture's own
              uncalibrated timing offset, fairly stable frame to frame.
              Read it for <i>relative</i> delay between echoes — a second,
              smaller peak next to the main one — not absolute time-of-flight
              from the row's centre.
            </p>

            <Heatmap
              path={path}
              metric="csi_cir"
              filename={meta.filename}
              numSubcarriers={meta.num_subcarriers}
              captureTMin={meta.t_min}
              captureTMax={meta.t_max}
              title="Channel Impulse Response — |IFFT| (rx0/tx0)"
              colorLabel="CIR magnitude"
              axisLabel="Delay tap"
              height={320}
              timeLink={timeLink}
              mimo={mimo}
              sourceMac={sourceMac}
              interpolate={interpolate}
              dark={dark}
            />
              </div>
            </TabsContent>

            <TabsContent value="doppler">
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Label htmlFor="win" className="text-[10px] text-muted-foreground uppercase tracking-wide">
                    Window (s)
                  </Label>
                  <Input
                    id="win"
                    type="number"
                    min={1}
                    max={600}
                    className="w-24"
                    value={winSeconds}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      if (Number.isFinite(v) && v >= 1 && v <= 600) setWinSeconds(v);
                    }}
                  />
                  {dopplerGeom && (
                    <span className="text-[11px] text-muted-foreground">
                      resolution {(1 / winSeconds).toFixed(3)} Hz · ceiling{" "}
                      {dopplerGeom.complex.fMax.toFixed(2)} Hz ·{" "}
                      {Math.abs((dopplerGeom.complex.fMax * 0.05754) / 2).toFixed(2)} m/s
                    </span>
                  )}
                </div>

                {dopplerGeom ? (
                  <>
                    <Heatmap
                      path={path}
                      metric="csi_ratio_complex"
                      filename={meta.filename}
                      numSubcarriers={dopplerGeom.complex.rows}
                      captureTMin={meta.t_min}
                      captureTMax={meta.t_max}
                      title="Doppler — complex ratio (signed)"
                      colorLabel="Magnitude"
                      axisLabel="Doppler (Hz)"
                      yDomain={[dopplerGeom.complex.fMin, dopplerGeom.complex.fMax]}
                      source={dopplerSource("csi_ratio_complex")}
                      height={320}
                      timeLink={timeLink}
                      mimo={mimo}
                      sourceMac={sourceMac}
                      interpolate={interpolate}
                      dark={dark}
                    />

                    <Heatmap
                      path={path}
                      metric="csi_ratio_phase_time_unwrapped"
                      filename={meta.filename}
                      numSubcarriers={dopplerGeom.phase.rows}
                      captureTMin={meta.t_min}
                      captureTMax={meta.t_max}
                      title="Doppler — time-unwrapped ratio phase"
                      colorLabel="Magnitude"
                      axisLabel="Doppler (Hz)"
                      yDomain={[dopplerGeom.phase.fMin, dopplerGeom.phase.fMax]}
                      source={dopplerSource("csi_ratio_phase_time_unwrapped")}
                      height={320}
                      timeLink={timeLink}
                      mimo={mimo}
                      sourceMac={sourceMac}
                      interpolate={interpolate}
                      dark={dark}
                    />

                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      The top panel is the <b>complex ratio</b>, so its Doppler is{" "}
                      <b>signed</b>: zero sits at the middle of the axis, and the two
                      halves are the two directions of radial motion. Sign survives
                      here only because this is a ratio — both chains share an
                      oscillator, so the carrier frequency offset divides out and what
                      is left is geometry. It also never unwraps anything, which is
                      what the lower panel cannot avoid: a step it misreads offsets
                      that subcarrier by 2π for the rest of the capture and draws a
                      full-height bar.
                    </p>
                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      The axis stops at this capture&apos;s own Nyquist (±
                      {dopplerGeom.complex.fMax.toFixed(2)} Hz, which is only ±
                      {Math.abs((dopplerGeom.complex.fMax * 0.05754) / 2).toFixed(2)} m/s
                      at 5.21 GHz). A walk is several times that and aliases in as
                      broadband smear rather than as a velocity, so read this for
                      respiration and slow motion, not for gait. Respiration is a{" "}
                      <b>symmetric pair</b> of sidebands about zero, not one peak —{" "}
                      <b>shift + wheel</b> zooms the frequency axis onto it. Blank
                      columns are windows more than half interpolated across a capture
                      dropout; short hiccups are bridged rather than blanked.
                    </p>
                  </>
                ) : (
                  <div className="text-muted-foreground p-8 text-sm">
                    No spectrogram for this range — the window ({winSeconds} s) may be
                    longer than the frames in view. Shorten it, or widen the time range.
                  </div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="presence">
              <Presence
                path={path}
                meta={meta}
                timeLink={timeLink}
                mimo={mimo}
                sourceMac={sourceMac}
                interpolate={interpolate}
                dark={dark}
              />
            </TabsContent>
          </Tabs>
        ) : (
          !error && (
            <div className="text-muted-foreground p-8">
              Enter a .dat path to explore, or toggle <b>Live</b> for realtime capture. Default:{" "}
              <code className="bg-muted px-1.5 py-0.5 rounded text-sm">{DEFAULT_PATH}</code>.
            </div>
          )
        )}
      </div>
    </div>
  );
}
