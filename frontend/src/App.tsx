import { useEffect, useState } from "react";
import { fetchCaptures, fetchFilters, fetchMeta, formatBytes, type CaptureFile, type Filters, type Meta } from "./api";
import { TWILIGHT } from "./colormap";
import { Heatmap } from "./Heatmap";
import { createTimeLink } from "./timelink";
import { Button } from "@/components/ui/button";
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
import { PlayIcon, PauseIcon, SunIcon, MoonIcon } from "lucide-react";

const DEFAULT_PATH = "captures/capture.dat";
const DEFAULT_REFRESH_MS = 300;
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
  const [sourceMac, setSourceMac] = useState<string>(DEFAULT_SOURCE_MAC);
  const [dark, setDark] = useState<boolean>(() => {
    const stored = localStorage.getItem(DARK_KEY);
    return stored === null ? true : stored === "true";
  });

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
    label: `${c.filename}  (${formatBytes(c.size_bytes)})`,
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
                <SelectValue placeholder={captures === null ? "Loading…" : "Select a .dat file"} />
              </SelectTrigger>
              <SelectContent>
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
              onValueChange={(v) => setMimo(v ?? "all")}
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
              height={400}
              timeLink={timeLink}
              mimo={mimo}
              sourceMac={sourceMac}
              dark={dark}
            />
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
              height={400}
              palette={TWILIGHT}
              timeLink={timeLink}
              mimo={mimo}
              sourceMac={sourceMac}
              dark={dark}
            />
            <Heatmap
              path={path}
              metric="csi_ratio_amplitude"
              filename={meta.filename}
              numSubcarriers={meta.num_subcarriers}
              captureTMin={meta.t_min}
              captureTMax={meta.t_max}
              title="CSI Ratio Amplitude (rx1/rx0)"
              colorLabel="Ratio amp (dB)"
              height={400}
              timeLink={timeLink}
              mimo={mimo}
              sourceMac={sourceMac}
              dark={dark}
            />
            <Heatmap
              path={path}
              metric="csi_ratio_phase"
              filename={meta.filename}
              numSubcarriers={meta.num_subcarriers}
              captureTMin={meta.t_min}
              captureTMax={meta.t_max}
              minValue={-Math.PI}
              maxValue={Math.PI}
              title="CSI Ratio Phase (rx1/rx0)"
              colorLabel="Ratio phase (rad)"
              height={400}
              palette={TWILIGHT}
              timeLink={timeLink}
              mimo={mimo}
              sourceMac={sourceMac}
              dark={dark}
            />

            <Separator className="my-2" />
            <p className="text-xs text-muted-foreground">
              Derived views below. Some frames arrive with the rx streams
              exchanged (ratio reciprocal) or the ratio negated (phase +π);
              these put them back. Detection compares each frame against its
              neighbours, so a single Source MAC must be selected — on{" "}
              <code className="bg-muted px-1 py-0.5 rounded">all</code> the
              correction declines to act rather than guessing.
            </p>

            <Heatmap
              path={path}
              metric="csi_ratio_amplitude_corrected"
              filename={meta.filename}
              numSubcarriers={meta.num_subcarriers}
              captureTMin={meta.t_min}
              captureTMax={meta.t_max}
              title="CSI Ratio Amplitude — Swap-Corrected (rx1/rx0)"
              colorLabel="Ratio amp (dB)"
              height={400}
              timeLink={timeLink}
              mimo={mimo}
              sourceMac={sourceMac}
              dark={dark}
            />
            <Heatmap
              path={path}
              metric="csi_ratio_phase_corrected"
              filename={meta.filename}
              numSubcarriers={meta.num_subcarriers}
              captureTMin={meta.t_min}
              captureTMax={meta.t_max}
              minValue={-Math.PI}
              maxValue={Math.PI}
              title="CSI Ratio Phase — Swap-Corrected (rx1/rx0)"
              colorLabel="Ratio phase (rad)"
              height={400}
              palette={TWILIGHT}
              timeLink={timeLink}
              mimo={mimo}
              sourceMac={sourceMac}
              dark={dark}
            />

            <Separator className="my-2" />
            <p className="text-xs text-muted-foreground">
              Unwrapped along <i>time</i> on the corrected ratio: continuous
              phase accumulated as the channel moves. Restarts after a capture
              gap and anchors each segment at its own start, so the value is
              phase change since that segment began — segments are not
              comparable with one another. Shares the phase palette above for
              comparison, though accumulated phase is not an angle.
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
              height={400}
              palette={TWILIGHT}
              timeLink={timeLink}
              mimo={mimo}
              sourceMac={sourceMac}
              dark={dark}
            />

            <Separator className="my-2" />
            <p className="text-xs text-muted-foreground">
              Channel impulse response: the swap-corrected ratio, inverse-FFT'd
              from subcarrier into delay. Rows are delay taps, not
              subcarriers — re-centred so the zero-delay peak sits in the
              middle alongside DC on the panels above, with real echoes to one
              side and the DFT's circular wraparound to the other. A single
              tight peak near centre says the ratio is dominated by one path;
              a capture with real multipath shows a second, smaller peak
              offset from it.
            </p>

            <Heatmap
              path={path}
              metric="csi_ratio_cir"
              filename={meta.filename}
              numSubcarriers={meta.num_subcarriers}
              captureTMin={meta.t_min}
              captureTMax={meta.t_max}
              title="CSI Ratio CIR — |IFFT| (rx1/rx0)"
              colorLabel="CIR magnitude"
              axisLabel="Delay tap"
              height={400}
              timeLink={timeLink}
              mimo={mimo}
              sourceMac={sourceMac}
              dark={dark}
            />
          </div>
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
