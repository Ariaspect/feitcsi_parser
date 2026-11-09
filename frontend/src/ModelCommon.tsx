import { useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import type { AccuracyTable, ModelTruth, RoomFold } from "./api";

// Pieces the per-room models share: the ML tab's 41F calibrated model and the Transformer tab.

export const PERSON = "#2f6fed";
export const NO_PERSON = "#6b7280";

const ACC_ROWS: [string, string][] = [
  ["env1", "env1"],
  ["env2", "env2"],
  ["env3", "env3"],
  ["env3-1", "env3-1"],
  ["LOEO", "4 rooms (LOEO)"],
  ["scnC", "scenario C"],
  ["scnA", "scenario A"],
  ["scnB", "scenario B"],
];

function pct(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? "—" : v.toFixed(1);
}

/** ‹ window k of n › for captures cut into several 60 s windows. */
export function WindowNav({ windows, win, setWin }: { windows: [number, number][]; win: number; setWin: (w: number) => void }) {
  if (windows.length <= 1) return null;
  return (
    <span className="flex items-center gap-1 text-[11px] text-muted-foreground">
      <Button variant="outline" size="sm" className="h-7 px-2" disabled={win === 0} onClick={() => setWin(win - 1)}>‹</Button>
      window {win + 1} of {windows.length}
      <Button variant="outline" size="sm" className="h-7 px-2" disabled={win >= windows.length - 1} onClick={() => setWin(win + 1)}>›</Button>
    </span>
  );
}

/** PERSON / NO PERSON chip with P, and how it compares with the capture's label. */
export function VerdictChip({ p, label, truth }: { p: number; label: number; truth: ModelTruth }) {
  const gt = truth.human == null ? null : truth.human === "person" ? 1 : 0;
  return (
    <span className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
      <span
        className="rounded px-2 py-1 text-sm font-semibold text-white tabular-nums"
        style={{ background: label ? PERSON : NO_PERSON }}
        title="P(person) > 0.5 → person"
      >
        {label ? "PERSON" : "NO PERSON"} · P = {p.toFixed(3)}
      </span>
      {gt != null && <span title={gt === label ? "agrees with the label" : "differs from the label"}>{gt === label ? "✓" : "✗"}</span>}
      <span>
        label: <b className="text-foreground">{truth.human ?? "none"}</b>
        {truth.human_source ? ` · ${truth.human_source}` : ""}
        {truth.condition ? ` · “${truth.condition}”` : ""}
        {truth.camera != null ? ` · camera ${(truth.camera * 100).toFixed(0)} %` : ""}
      </span>
    </span>
  );
}

/** Which room the capture is in, which model judges it and whose empties calibrate it. */
export function RoomLine({ room, trainedOn }: { room: RoomFold | null; trainedOn: string[] | null }) {
  if (!room) {
    return (
      <div className="text-[11px] text-muted-foreground">
        Room unknown (not in the GT table or the training list, and its day has no single labelled room) — there is no
        empty-room reference to calibrate against, so no verdict.
      </div>
    );
  }
  return (
    <div className="text-[11px] text-muted-foreground">
      room <b className="text-foreground">{room.room}</b> ({room.source}) · judged by the model trained on{" "}
      <b className="text-foreground">{(trainedOn ?? []).join(" + ")}</b> — it never saw this room · calibrated with{" "}
      <b className="text-foreground">{room.calibrated_with}</b>'s empty minutes
    </div>
  );
}

/** Held-out accuracy per room and scenario, overall / KPI#1 / KPI#2. */
export function AccuracyGrid({ acc, note }: { acc: AccuracyTable; note: ReactNode }) {
  const rows = ACC_ROWS.filter(([k]) => acc[k]);
  return (
    <div className="space-y-1">
      <div className="text-xs font-medium">Held-out accuracy (%)</div>
      <div className="overflow-x-auto">
        <table className="text-[11px] tabular-nums">
          <thead>
            <tr className="text-muted-foreground">
              <th className="pr-4 text-left font-normal"></th>
              <th className="pr-3 text-right font-normal">all</th>
              <th className="pr-3 text-right font-normal">KPI#1 (no object)</th>
              <th className="text-right font-normal">KPI#2 (object)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([k, name]) => (
              <tr key={k} className={k === "LOEO" ? "font-semibold" : ""}>
                <td className="pr-4">{name}</td>
                <td className="pr-3 text-right">{pct(acc[k].all)}</td>
                <td className="pr-3 text-right">{pct(acc[k].kpi1)}</td>
                <td className="text-right">{pct(acc[k].kpi2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-muted-foreground leading-relaxed">{note}</p>
    </div>
  );
}

/** A titled section that renders (and so fetches) its body only while open. */
export function Collapsible({ title, sub, children, defaultOpen = false }: {
  title: string; sub?: ReactNode; children: ReactNode; defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="space-y-3 rounded-md border p-3">
      <button type="button" className="flex w-full flex-wrap items-baseline gap-2 text-left" onClick={() => setOpen(!open)}>
        <span className="text-sm font-semibold">{open ? "▾" : "▸"} {title}</span>
        {sub && <span className="text-[11px] text-muted-foreground">{sub}</span>}
      </button>
      {open && children}
    </section>
  );
}
