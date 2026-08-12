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
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PlayIcon, PauseIcon } from "lucide-react";

const DEFAULT_PATH = "captures/capture.dat";
const DEFAULT_REFRESH_MS = 300;

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

  const mimoItems = [
    { label: "all", value: "all" },
    ...(filters?.mimo_modes.map((m) => ({ label: m, value: m })) ?? []),
  ];
  const macItems = [
    { label: "all", value: "all" },
    ...(filters?.source_macs.map((mac) => ({ label: mac, value: mac })) ?? []),
  ];

  return (
    <div className="font-sans p-4 max-w-6xl mx-auto space-y-4">
      <h1 className="text-2xl font-bold tracking-tight">FeitCSI Heatmap</h1>

      <Card>
        <CardHeader>
          <CardTitle>Capture</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4 items-end">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="path">.dat file</Label>
              <Input
                id="path"
                type="text"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                className="w-96"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="refresh">Refresh (ms)</Label>
              <Input
                id="refresh"
                type="number"
                min={50}
                max={10000}
                value={refreshMs}
                onChange={(e) => setRefreshMs(Number(e.target.value))}
                className="w-24"
              />
            </div>

            <Button
              onClick={() => setRunning((r) => !r)}
              variant={running ? "destructive" : "default"}
            >
              {running ? (
                <>
                  <PauseIcon data-icon="inline-start" />
                  Pause
                </>
              ) : (
                <>
                  <PlayIcon data-icon="inline-start" />
                  Run realtime
                </>
              )}
            </Button>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="mimo">MIMO</Label>
              <Select
                value={mimo}
                onValueChange={(v) => setMimo(v ?? "all")}
                items={mimoItems}
                disabled={!filters || filters.mimo_modes.length <= 1}
              >
                <SelectTrigger id="mimo" className="w-24">
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

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="source-mac">Source MAC</Label>
              <Select
                value={sourceMac}
                onValueChange={(v) => setSourceMac(v ?? "all")}
                items={macItems}
                disabled={!filters || filters.source_macs.length <= 1}
              >
                <SelectTrigger id="source-mac" className="w-48">
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
          </div>
        </CardContent>
      </Card>

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
          <span>frames: {meta.total_frames}</span>
          <span>subcarriers: {meta.num_subcarriers}</span>
          <span>time: [{meta.t_min.toFixed(3)}, {meta.t_max.toFixed(3)}] s</span>
          {lastUpdate && (
            <>
              <Separator orientation="vertical" className="h-4" />
              <span>last update: {new Date(lastUpdate).toLocaleTimeString()}</span>
            </>
          )}
        </div>
      )}

      {meta && meta.total_frames > 0 ? (
        <div className="space-y-6">
          <Heatmap
            path={path}
            metric="amplitude"
            filename={meta.filename}
            numSubcarriers={meta.num_subcarriers}
            captureTMin={meta.t_min}
            captureTMax={meta.t_max}
            title="FeitCSI — amplitude"
            colorLabel="Amplitude (dBm)"
            height={400}
            timeLink={timeLink}
            mimo={mimo}
            sourceMac={sourceMac}
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
            title="FeitCSI — phase"
            colorLabel="Phase (rad)"
            height={400}
            palette={TWILIGHT}
            timeLink={timeLink}
            mimo={mimo}
            sourceMac={sourceMac}
          />
          <Heatmap
            path={path}
            metric="csi_ratio_amplitude"
            filename={meta.filename}
            numSubcarriers={meta.num_subcarriers}
            captureTMin={meta.t_min}
            captureTMax={meta.t_max}
            title="FeitCSI — CSI ratio amplitude (rx1/rx0)"
            colorLabel="Ratio amp (dB)"
            height={400}
            timeLink={timeLink}
            mimo={mimo}
            sourceMac={sourceMac}
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
            title="FeitCSI — CSI ratio phase (rx1/rx0)"
            colorLabel="Ratio phase (rad)"
            height={400}
            palette={TWILIGHT}
            timeLink={timeLink}
            mimo={mimo}
            sourceMac={sourceMac}
          />
        </div>
      ) : (
        !error && (
          <div className="text-muted-foreground p-8">
            Enter a .dat path to explore, or toggle <b>Run realtime</b> for live capture. Default file:{" "}
            <code className="bg-muted px-1.5 py-0.5 rounded text-sm">{DEFAULT_PATH}</code>.
          </div>
        )
      )}
    </div>
  );
}
