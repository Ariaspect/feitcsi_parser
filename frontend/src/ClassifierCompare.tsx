import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  fetchClassifierFeatures,
  fetchClassifierItem,
  type CaptureFile,
  type ClassifierFeature,
  type ClassifierItem,
  type ClassifierOptions,
} from "./api";

/** One row of the compare table: a whole capture (t0/t1 null) or one range of one. */
export interface CompareEntry {
  id: string;
  path: string;
  t0: number | null;
  t1: number | null;
}

type Result =
  | { status: "pending" }
  | { status: "done"; item: ClassifierItem }
  | { status: "error"; error: string };

type GroupBy = "gt" | "condition" | "room" | "day" | "capture" | "none";
type Scale = "rows" | "axis";

const STORE_KEY = "classifierCompare.v1";
// The server computes one row at a time; two in flight keeps it busy without
// queueing work the user may remove before it starts.
const CONCURRENCY = 2;
const PICKER_LIMIT = 300;

// The columns a first comparison usually wants: the two scores, the rule's
// inputs, the fan tell, and the Liu columns that separated something on env3.
const DEFAULT_KEYS = [
  "p_occupied", "step_p90", "breath_run20", "breath_peak10", "revisit",
  "liu_ms_mrc", "liu_acf_period_spread", "liu_speed", "liu_coh32", "liu_coh32_breath",
];

// Sequential blue, light -> dark (dataviz reference ramp). The lightest stop is
// near the surface so a weak value recedes; dark mode runs its own steps from
// the dark surface towards light blue, so "strong" is the high-contrast end in
// both themes.
const RAMP_LIGHT = ["#f3f7fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];
const RAMP_DARK = ["#1f242c", "#104281", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"];

function hexToRgb(h: string): [number, number, number] {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function ramp(t: number, dark: boolean): string {
  const stops = (dark ? RAMP_DARK : RAMP_LIGHT).map(hexToRgb);
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const c = stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/** Text over a filled cell: light on the dark end of the light ramp, dark on
 *  the light end of the dark ramp; the theme's own ink below that. */
function inkOn(t: number, dark: boolean): string | undefined {
  if (t < 0.6) return undefined;
  return dark ? "#0b1220" : "#ffffff";
}

function entryId(path: string, t0: number | null, t1: number | null): string {
  return `${path}|${t0 ?? "all"}|${t1 ?? "all"}`;
}

function baseName(path: string): string {
  const b = path.split("/").pop() ?? path;
  return b.replace(/\.(bin|dat)$/, "");
}

function clock(s: number): string {
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
}

function entryLabel(e: CompareEntry): string {
  return e.t0 == null || e.t1 == null ? baseName(e.path) : `${baseName(e.path)} · ${clock(e.t0)}–${clock(e.t1)}`;
}

function median(v: number[]): number | null {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function load<T>(key: keyof Stored, fallback: T): T {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return fallback;
    const v = (JSON.parse(raw) as Partial<Stored>)[key];
    return (v as T) ?? fallback;
  } catch {
    return fallback;
  }
}

interface Stored {
  entries: CompareEntry[];
  keys: string[];
  groupBy: GroupBy;
  scale: Scale;
}

function save(s: Stored): void {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(s));
  } catch {
    /* private window or blocked storage: the table still works, it just forgets */
  }
}

/** The compare table's rows, kept across reloads in this browser. Lives in the
 *  tab so the unit table's "+" can add to it. */
export function useCompareEntries() {
  const [entries, setEntries] = useState<CompareEntry[]>(() => load("entries", [] as CompareEntry[]));
  const add = useCallback((items: { path: string; t0: number | null; t1: number | null }[]) => {
    setEntries((prev) => {
      const seen = new Set(prev.map((e) => e.id));
      const next = [...prev];
      for (const it of items) {
        const id = entryId(it.path, it.t0, it.t1);
        if (!seen.has(id)) { seen.add(id); next.push({ id, ...it }); }
      }
      return next;
    });
  }, []);
  const remove = useCallback((id: string) => setEntries((prev) => prev.filter((e) => e.id !== id)), []);
  const clear = useCallback(() => setEntries([]), []);
  return { entries, add, remove, clear };
}

function gtOfCapture(c: CaptureFile): "person" | "empty" | null {
  if (c.gt === "person" || c.gt === "empty") return c.gt;
  if (c.presence === "occupied") return "person";
  if (c.presence === "empty") return "empty";
  return null;
}

function GtBadge({ gt, title }: { gt: "person" | "empty" | null; title?: string }) {
  if (!gt) return <span className="text-muted-foreground" title={title ?? "no ground truth recorded"}>—</span>;
  return (
    <span
      title={title}
      className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-medium ${
        gt === "person"
          ? "bg-amber-500/15 text-amber-700 dark:text-amber-300"
          : "bg-slate-500/15 text-slate-700 dark:text-slate-300"
      }`}
    >
      {gt}
    </span>
  );
}

const SELECT_CLASS = "h-7 rounded-md border border-border bg-background px-2 text-xs";

export interface ClassifierCompareProps {
  captures: CaptureFile[] | null;
  currentPath: string;
  currentUnits: { t0: number; t1: number }[];
  /** The bank as the tab last received it; fetched on its own when absent. */
  features: ClassifierFeature[] | null;
  options: ClassifierOptions;
  dark: boolean;
  compare: ReturnType<typeof useCompareEntries>;
}

export function ClassifierCompare({
  captures, currentPath, currentUnits, features: bankFromTab, options, dark, compare,
}: ClassifierCompareProps) {
  const { entries, add, remove, clear } = compare;

  // ---- the bank's columns
  const [bankFetched, setBankFetched] = useState<ClassifierFeature[] | null>(null);
  useEffect(() => {
    if (bankFromTab || bankFetched) return;
    const c = new AbortController();
    fetchClassifierFeatures(c.signal).then(setBankFetched).catch(() => {});
    return () => c.abort();
  }, [bankFromTab, bankFetched]);
  const bank = bankFromTab ?? bankFetched ?? [];

  const [keys, setKeys] = useState<string[]>(() => load("keys", DEFAULT_KEYS));
  const [groupBy, setGroupBy] = useState<GroupBy>(() => load("groupBy", "gt" as GroupBy));
  const [scale, setScale] = useState<Scale>(() => load("scale", "rows" as Scale));
  useEffect(() => { save({ entries, keys, groupBy, scale }); }, [entries, keys, groupBy, scale]);

  const columns = useMemo(
    () => bank.filter((f) => keys.includes(f.key)),   // the bank's order, not the click order
    [bank, keys],
  );
  const toggleKey = (k: string) =>
    setKeys((prev) => (prev.includes(k) ? prev.filter((x) => x !== k) : [...prev, k]));

  // Group the picker by the bank's own "test" labels, in the bank's order.
  const featureGroups = useMemo(() => {
    const g: { test: string; items: ClassifierFeature[] }[] = [];
    for (const f of bank) {
      const last = g[g.length - 1];
      if (last && last.test === f.test) last.items.push(f);
      else g.push({ test: f.test, items: [f] });
    }
    return g;
  }, [bank]);

  // ---- computing the rows, a few at a time
  const [results, setResults] = useState<Record<string, Result>>({});
  const optKey = JSON.stringify(options);
  const generation = useRef(0);
  useEffect(() => { generation.current += 1; setResults({}); }, [optKey]);
  useEffect(() => {
    const inflight = Object.values(results).filter((r) => r.status === "pending").length;
    const todo = entries.filter((e) => !results[e.id]).slice(0, Math.max(0, CONCURRENCY - inflight));
    if (!todo.length) return;
    const gen = generation.current;
    setResults((prev) => {
      const n = { ...prev };
      for (const e of todo) n[e.id] = { status: "pending" };
      return n;
    });
    for (const e of todo) {
      fetchClassifierItem(e.path, e.t0, e.t1, options)
        .then((item) => {
          if (gen !== generation.current) return;
          setResults((prev) => (prev[e.id] ? { ...prev, [e.id]: { status: "done", item } } : prev));
        })
        .catch((err: unknown) => {
          if (gen !== generation.current) return;
          const error = err instanceof Error ? err.message : String(err);
          setResults((prev) => (prev[e.id] ? { ...prev, [e.id]: { status: "error", error } } : prev));
        });
    }
    // options is read through optKey
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entries, results, optKey]);

  const done = entries.filter((e) => results[e.id]?.status === "done").length;
  const failed = entries.filter((e) => results[e.id]?.status === "error").length;

  // ---- the capture picker
  const [filter, setFilter] = useState("");
  const [gtFilter, setGtFilter] = useState<"any" | "person" | "empty" | "none">("any");
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const shown = useMemo(() => {
    const q = filter.trim().toLowerCase();
    const terms = q ? q.split(/\s+/) : [];
    return (captures ?? []).filter((c) => {
      const gt = gtOfCapture(c);
      if (gtFilter === "none" ? gt !== null : gtFilter !== "any" && gt !== gtFilter) return false;
      if (!terms.length) return true;
      const hay = `${c.filename} ${c.recorded ?? ""} ${c.room ?? ""} ${c.scenario ?? ""}`.toLowerCase();
      return terms.every((t) => hay.includes(t));
    });
  }, [captures, filter, gtFilter]);
  const shownCapped = shown.slice(0, PICKER_LIMIT);
  const toggleChecked = (p: string) =>
    setChecked((prev) => {
      const n = new Set(prev);
      if (n.has(p)) n.delete(p); else n.add(p);
      return n;
    });

  // ---- rows, grouped
  const rows = useMemo(() => entries.map((e) => ({ e, r: results[e.id] })), [entries, results]);
  const groupOf = useCallback((e: CompareEntry, r: Result | undefined): string => {
    const it = r?.status === "done" ? r.item : null;
    switch (groupBy) {
      case "gt": return it ? (it.gt ?? "no GT") : "…";
      case "condition": return it ? (it.condition ?? "no condition recorded") : "…";
      case "room": return it ? (it.room ?? "no room recorded") : "…";
      case "day": return baseName(e.path).slice(0, 8);
      case "capture": return baseName(e.path);
      default: return "all";
    }
  }, [groupBy]);
  const groups = useMemo(() => {
    const m = new Map<string, typeof rows>();
    for (const row of rows) {
      const g = groupOf(row.e, row.r);
      if (!m.has(g)) m.set(g, []);
      m.get(g)!.push(row);
    }
    // person, empty, then the rest; rows still computing ("…") always last
    const order = (g: string) =>
      g === "…" ? 9 : groupBy !== "gt" ? 0 : g === "person" ? 0 : g === "empty" ? 1 : 2;
    return [...m.entries()]
      .sort((a, b) => order(a[0]) - order(b[0]) || a[0].localeCompare(b[0]))
      .map(([name, rs]) => ({
        name,
        rows: [...rs].sort((a, b) => a.e.path.localeCompare(b.e.path) || (a.e.t0 ?? 0) - (b.e.t0 ?? 0)),
      }));
  }, [rows, groupOf, groupBy]);

  // Per-column scale: the shown rows' own range, or the feature's fixed axis.
  const ranges = useMemo(() => {
    const out = new Map<string, [number, number]>();
    for (const f of columns) {
      if (scale === "axis" && f.axis) { out.set(f.key, f.axis); continue; }
      const v = rows.flatMap(({ r }) =>
        r?.status === "done" && r.item.values[f.key] != null ? [r.item.values[f.key] as number] : []);
      if (v.length) out.set(f.key, [Math.min(...v), Math.max(...v)]);
    }
    return out;
  }, [columns, rows, scale]);
  const strength = (key: string, v: number | null): number | null => {
    if (v == null) return null;
    const rg = ranges.get(key);
    if (!rg) return null;
    const [lo, hi] = rg;
    return hi > lo ? Math.min(1, Math.max(0, (v - lo) / (hi - lo))) : 0.5;
  };

  const cell = (f: ClassifierFeature, v: number | null, extra?: string) => {
    const t = strength(f.key, v);
    const rg = ranges.get(f.key);
    return (
      <td
        key={f.key}
        className="px-2 py-1 text-right tabular-nums whitespace-nowrap"
        style={t == null ? undefined : { backgroundColor: ramp(t, dark), color: inkOn(t, dark) }}
        title={`${f.label}: ${v == null ? "—" : v.toFixed(f.decimals)}${extra ?? ""}${
          rg ? ` · scale ${rg[0].toFixed(f.decimals)} – ${rg[1].toFixed(f.decimals)} (${scale === "axis" && f.axis ? "feature axis" : "shown rows"})` : ""}`}
      >
        {v == null ? "—" : v.toFixed(f.decimals)}
      </td>
    );
  };

  const exportCsv = () => {
    const head = ["group", "item", "path", "t0", "t1", "gt", "gt_source", "condition", "room", "camera", ...columns.map((f) => f.key)];
    const lines = [head.join(",")];
    for (const g of groups) {
      for (const { e, r } of g.rows) {
        const it = r?.status === "done" ? r.item : null;
        const vals = columns.map((f) => (it?.values[f.key] ?? ""));
        const row = [g.name, entryLabel(e), e.path, it?.t0 ?? "", it?.t1 ?? "", it?.gt ?? "", it?.gtSource ?? "",
          it?.condition ?? "", it?.room ?? "", it?.camera ?? "", ...vals];
        lines.push(row.map((x) => `"${String(x).replace(/"/g, '""')}"`).join(","));
      }
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "classifier_compare.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const nInfo = 5; // remove, item, GT, condition, camera

  return (
    <section className="space-y-3 rounded-lg border border-border p-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-xs font-semibold">Compare captures and units</h3>
        <span className="text-[11px] text-muted-foreground">
          {entries.length} row{entries.length === 1 ? "" : "s"} · {done} computed
          {failed ? ` · ${failed} failed` : ""}
          {done + failed < entries.length ? ` · computing ${entries.length - done - failed} more` : ""}
        </span>
      </div>

      {/* ---- add rows */}
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="xs" variant="outline"
          disabled={!currentUnits.length}
          onClick={() => add(currentUnits.map((u) => ({ path: currentPath, t0: u.t0, t1: u.t1 })))}
          title="Every unit of the range shown above, one row each"
        >
          + this range&apos;s units ({currentUnits.length})
        </Button>
        <Button
          size="xs" variant="outline"
          onClick={() => add([{ path: currentPath, t0: null, t1: null }])}
          title="The whole open capture as one row (at most 10 minutes)"
        >
          + this capture
        </Button>
        <Button size="xs" variant="ghost" disabled={!entries.length} onClick={clear}>clear rows</Button>
        <Button size="xs" variant="ghost" disabled={!done} onClick={exportCsv}>export CSV</Button>
      </div>

      <details className="rounded-md border border-border/60 px-2 py-1">
        <summary className="cursor-pointer text-[11px] text-muted-foreground">
          Pick captures from any day ({shown.length} match{shown.length === 1 ? "" : "es"}{checked.size ? ` · ${checked.size} checked` : ""})
        </summary>
        <div className="mt-2 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <Input
              className="h-7 w-72 text-xs" placeholder="filter: 20261021 fan still env3 …"
              value={filter} onChange={(e) => setFilter(e.target.value)}
            />
            <select className={SELECT_CLASS} value={gtFilter}
              onChange={(e) => setGtFilter(e.target.value as typeof gtFilter)} aria-label="ground truth filter">
              <option value="any">any GT</option>
              <option value="person">GT person</option>
              <option value="empty">GT empty</option>
              <option value="none">no GT</option>
            </select>
            <Button size="xs" variant="outline"
              onClick={() => setChecked((prev) => new Set([...prev, ...shownCapped.map((c) => c.path)]))}>
              check all shown
            </Button>
            <Button size="xs" variant="ghost" disabled={!checked.size} onClick={() => setChecked(new Set())}>uncheck</Button>
            <Button size="xs" disabled={!checked.size}
              onClick={() => { add([...checked].map((p) => ({ path: p, t0: null, t1: null }))); setChecked(new Set()); }}>
              add {checked.size} checked
            </Button>
          </div>
          <div className="max-h-64 overflow-auto rounded border border-border/50">
            <table className="w-full text-[11px]">
              <tbody>
                {shownCapped.map((c) => (
                  <tr key={c.path} className="border-t border-border/40 first:border-t-0 hover:bg-muted/50">
                    <td className="px-2 py-0.5 w-6">
                      <input type="checkbox" checked={checked.has(c.path)} onChange={() => toggleChecked(c.path)}
                        aria-label={`check ${c.filename}`} />
                    </td>
                    <td className="px-2 py-0.5 whitespace-nowrap font-mono">{c.filename}</td>
                    <td className="px-2 py-0.5"><GtBadge gt={gtOfCapture(c)} /></td>
                    <td className="px-2 py-0.5 text-muted-foreground">{c.recorded ?? c.scenario ?? ""}</td>
                    <td className="px-2 py-0.5 text-right text-muted-foreground tabular-nums">
                      {c.camera_occupancy != null ? `cam ${(c.camera_occupancy * 100).toFixed(0)} %` : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {shown.length > PICKER_LIMIT && (
              <p className="px-2 py-1 text-[11px] text-muted-foreground">
                {shown.length - PICKER_LIMIT} more — narrow the filter to see them.
              </p>
            )}
          </div>
        </div>
      </details>

      {/* ---- columns */}
      <details className="rounded-md border border-border/60 px-2 py-1">
        <summary className="cursor-pointer text-[11px] text-muted-foreground">
          Features to compare ({columns.length} of {bank.length})
        </summary>
        <div className="mt-2 space-y-1.5">
          <div className="flex flex-wrap gap-2">
            <Button size="xs" variant="outline" onClick={() => setKeys(DEFAULT_KEYS)}>default set</Button>
            <Button size="xs" variant="outline"
              onClick={() => setKeys((p) => [...new Set([...p, ...bank.filter((f) => f.key.startsWith("liu_")).map((f) => f.key)])])}>
              + all Liu
            </Button>
            <Button size="xs" variant="outline"
              onClick={() => setKeys((p) => [...new Set([...p, ...bank.filter((f) => f.key.startsWith("ml_")).map((f) => f.key)])])}>
              + 11F model
            </Button>
            <Button size="xs" variant="outline" onClick={() => setKeys(bank.map((f) => f.key))}>all</Button>
            <Button size="xs" variant="ghost" onClick={() => setKeys([])}>none</Button>
          </div>
          {featureGroups.map((g) => (
            <div key={g.test} className="flex flex-wrap items-center gap-1">
              <span className="w-20 shrink-0 text-[10px] uppercase tracking-wide text-muted-foreground">test {g.test}</span>
              {g.items.map((f) => {
                const on = keys.includes(f.key);
                return (
                  <button
                    key={f.key} type="button" onClick={() => toggleKey(f.key)}
                    title={`${f.description} (${f.status})`}
                    className={`rounded border px-1.5 py-0.5 text-[11px] ${
                      on ? "border-primary bg-primary text-primary-foreground" : "border-border text-muted-foreground hover:bg-muted"}`}
                  >
                    {f.label}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </details>

      {/* ---- display */}
      <div className="flex flex-wrap items-center gap-3 text-[11px]">
        <label className="flex items-center gap-1.5">
          <span className="text-muted-foreground">group by</span>
          <select className={SELECT_CLASS} value={groupBy} onChange={(e) => setGroupBy(e.target.value as GroupBy)}>
            <option value="gt">ground truth</option>
            <option value="condition">condition</option>
            <option value="room">room</option>
            <option value="day">day</option>
            <option value="capture">capture</option>
            <option value="none">none</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5">
          <span className="text-muted-foreground">colour scale</span>
          <select className={SELECT_CLASS} value={scale} onChange={(e) => setScale(e.target.value as Scale)}>
            <option value="rows">per column, over the shown rows</option>
            <option value="axis">per column, the feature&apos;s fixed axis</option>
          </select>
        </label>
        <span className="flex items-center gap-1.5 text-muted-foreground">
          weak
          <span
            className="inline-block h-2.5 w-28 rounded-sm"
            style={{ background: `linear-gradient(to right, ${[0, 0.2, 0.4, 0.6, 0.8, 1].map((t) => ramp(t, dark)).join(", ")})` }}
          />
          strong
        </span>
      </div>

      {/* ---- the table */}
      {entries.length === 0 ? (
        <p className="text-[11px] text-muted-foreground">
          No rows yet. Add this range&apos;s units, this capture, or pick captures from any day above; each row is
          computed once and kept.
        </p>
      ) : (
        <div className="max-h-[70vh] overflow-auto rounded border border-border/50">
          <table className="w-full text-[11px]">
            <thead className="sticky top-0 z-10 bg-background">
              <tr className="text-muted-foreground">
                <th className="px-1 py-1" />
                <th className="px-2 py-1 text-left font-medium">row</th>
                <th className="px-2 py-1 text-left font-medium">GT</th>
                <th className="px-2 py-1 text-left font-medium">condition</th>
                <th className="px-2 py-1 text-right font-medium" title="Camera's occupied fraction in the row's range (or the capture's, marked *)">camera</th>
                {columns.map((f) => (
                  <th key={f.key} className="px-2 py-1 text-right font-medium whitespace-nowrap"
                    title={`${f.description} (${f.status}, test ${f.test})`}>
                    {f.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => {
                const meds = columns.map((f) =>
                  median(g.rows.flatMap(({ r }) =>
                    r?.status === "done" && r.item.values[f.key] != null ? [r.item.values[f.key] as number] : [])));
                return [
                  <tr key={`g-${g.name}`} className="border-t-2 border-border bg-muted/40 font-medium">
                    <td colSpan={nInfo} className="px-2 py-1">
                      {groupBy === "gt" ? <GtBadge gt={g.name === "person" || g.name === "empty" ? g.name : null} /> : g.name}
                      {groupBy === "gt" && g.name !== "person" && g.name !== "empty" ? ` ${g.name}` : ""}
                      <span className="text-muted-foreground"> · {g.rows.length} row{g.rows.length === 1 ? "" : "s"} · median</span>
                    </td>
                    {columns.map((f, i) => cell(f, meds[i], " (group median)"))}
                  </tr>,
                  ...g.rows.map(({ e, r }) => {
                    const it = r?.status === "done" ? r.item : null;
                    return (
                      <tr key={e.id} className="border-t border-border/40">
                        <td className="px-1 py-1">
                          <button type="button" className="text-muted-foreground hover:text-foreground"
                            onClick={() => remove(e.id)} title="remove this row" aria-label={`remove ${entryLabel(e)}`}>×</button>
                        </td>
                        <td className="px-2 py-1 whitespace-nowrap font-mono" title={e.path}>
                          {entryLabel(e)}
                          {r?.status === "pending" && <span className="text-muted-foreground"> · computing…</span>}
                          {r?.status === "error" && <span className="text-red-500"> · {r.error}</span>}
                        </td>
                        <td className="px-2 py-1"><GtBadge gt={it?.gt ?? null} title={it?.gtSource ?? undefined} /></td>
                        <td className="px-2 py-1 whitespace-nowrap text-muted-foreground" title={it?.conditionSource ?? undefined}>
                          {it?.condition ?? ""}{it?.room ? <span className="opacity-70"> · {it.room}</span> : null}
                        </td>
                        <td className="px-2 py-1 text-right tabular-nums">
                          {it?.camera == null ? "—" : `${(it.camera * 100).toFixed(0)} %${it.cameraScope === "capture" ? "*" : ""}`}
                        </td>
                        {columns.map((f) => (it ? cell(f, it.values[f.key] ?? null) : <td key={f.key} />))}
                      </tr>
                    );
                  }),
                ];
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="text-[11px] text-muted-foreground">
        Rows are units or whole captures (at most 10 minutes); columns are the bank&apos;s features, in its order.
        Each column is coloured on its own scale — the stronger the colour, the larger the value — so a column reads
        across rows, not against its neighbours. The group row is the median of its rows. GT comes from the user&apos;s GT table, else the
        operator&apos;s label in the capture&apos;s sidecar; the camera column is the camera&apos;s view, not GT.
      </p>
    </section>
  );
}
