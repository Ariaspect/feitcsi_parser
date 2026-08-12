import { useEffect, useState } from "react";
import { fetchFilters, fetchMeta, type Filters, type Meta } from "./api";
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
        if (!cancelled) setFilters(f);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [path]);

  const toggleDark = () => {
    setDark((d) => {
      const next = !d;
      localStorage.setItem(DARK_KEY, String(next));
      return next;
    });
  };

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
            <Label htmlFor="path" className="text-[10px] text-muted-foreground uppercase tracking-wide">.dat file</Label>
            <Input
              id="path"
              type="text"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              className="w-72 h-8"
            />
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
