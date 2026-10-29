import { Fragment, useEffect, useMemo, useState } from "react";

import { ChartBusy } from "@/components/ChartBusy";
import { Button } from "@/components/ui/button";
import { fetchMlHeads, type MlHeadName, type MlHeads, type MlKind, type MlLoeo } from "./api";

// The human-presence and anomaly-presence models (backend/ml_heads.py): 11F + aux + Liu, 43
// columns, each label as logistic regression and as one hidden layer of 16 units. Display only --
// the 11-feature screen model below is still the deployed verdict.

const PERSON = "#2f6fed";
const ANOMALY = "#b45309";
const NO = "#6b7280";
const GUIDE = "#9aa5b1";
const ENVS = ["env1", "env2", "env3", "env3-1", "scnC"];
const HEAD_NAME: Record<MlHeadName, string> = { human: "Human presence", anomaly: "Anomaly presence" };
const HEAD_YES: Record<MlHeadName, string> = { human: "PERSON", anomaly: "ANOMALY" };
const HEAD_NO: Record<MlHeadName, string> = { human: "NO PERSON", anomaly: "NO ANOMALY" };
const HEAD_COLOR: Record<MlHeadName, string> = { human: PERSON, anomaly: ANOMALY };
const KIND_NAME: Record<MlKind, string> = { lr: "LR", mlp: "hidden layer (16)" };
const COLUMNS: [MlHeadName, MlKind][] = [["human", "lr"], ["human", "mlp"], ["anomaly", "lr"], ["anomaly", "mlp"]];
const LOEO_ROWS: [string, string, boolean][] = [
  ["lr", "11F + aux + Liu · LR", false],
  ["mlp", "11F + aux + Liu · hidden layer", false],
  ["lr_marked_reference", "11F + ★ marked · LR", true],
  ["mlp_marked_reference", "11F + ★ marked · hidden layer", true],
  ["lr_11F_reference", "11F alone · LR", true],
];
const TOP = 15;

function pct(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? "—" : v.toFixed(1);
}

function fmt(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(1) : a >= 1 ? v.toFixed(3) : v.toPrecision(3);
}

export function MLHeads({ path, win, dark }: { path: string; win: number; dark: boolean }) {
  const [data, setData] = useState<MlHeads | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setBusy(true);
    fetchMlHeads(path, win, controller.signal)
      .then((d) => { setData(d); setError(null); })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setData(null);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [path, win]);

  return (
    <section className="space-y-3 rounded-md border p-3">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="text-sm font-semibold">Human &amp; anomaly models</span>
        <span className="text-[11px] text-muted-foreground">
          11F + aux + Liu (43 columns) · logistic regression and one hidden layer, for each label · display only
        </span>
        {busy && data && <span className="text-[11px] text-muted-foreground">updating…</span>}
      </div>
      {error ? (
        <div className="text-muted-foreground text-sm">No verdict from these models for this window — {error}</div>
      ) : !data ? (
        <ChartBusy busy empty height={90} label="computing the 43 columns" />
      ) : (
        <>
          <Verdicts data={data} />
          <Inputs data={data} dark={dark} />
          <Loeo data={data} />
        </>
      )}
    </section>
  );
}

function Chip({ head, label, p }: { head: MlHeadName; label: number; p: number }) {
  return (
    <span
      className="inline-block rounded px-2 py-0.5 text-xs font-semibold text-white tabular-nums"
      style={{ background: label ? HEAD_COLOR[head] : NO }}
      title={`p > 0.5 → ${HEAD_YES[head].toLowerCase()}`}
    >
      {label ? HEAD_YES[head] : HEAD_NO[head]} · p = {p.toFixed(3)}
    </span>
  );
}

function Verdicts({ data }: { data: MlHeads }) {
  const t = data.truth;
  const gt: Record<MlHeadName, number | null> = {
    human: t.human == null ? null : t.human === "person" ? 1 : 0,
    anomaly: t.anomaly,
  };
  const gtText: Record<MlHeadName, string> = {
    human: t.human == null ? "no label" : `${t.human}${t.humanSource ? ` · ${t.humanSource}` : ""}`,
    anomaly: t.anomaly == null ? "no condition recorded" : `${t.anomaly ? "anomaly" : "none"} · condition “${t.condition}”`,
  };
  return (
    <table className="text-[11px]">
      <thead>
        <tr className="text-muted-foreground">
          <th className="pr-4 text-left font-normal"></th>
          <th className="pr-4 text-left font-normal">LR</th>
          <th className="pr-4 text-left font-normal">hidden layer (16)</th>
          <th className="text-left font-normal">label (capture-level)</th>
        </tr>
      </thead>
      <tbody>
        {(["human", "anomaly"] as MlHeadName[]).map((h) => (
          <tr key={h}>
            <td className="pr-4 py-1 font-medium">{HEAD_NAME[h]}</td>
            {(["lr", "mlp"] as MlKind[]).map((k) => {
              const m = data.heads[h][k];
              const agree = gt[h] == null ? null : m.label === gt[h];
              return (
                <td key={k} className="pr-4 py-1 whitespace-nowrap">
                  <Chip head={h} label={m.label} p={m.p} />
                  {agree != null && (
                    <span className="ml-1 text-muted-foreground" title={agree ? "agrees with the label" : "differs from the label"}>
                      {agree ? "✓" : "✗"}
                    </span>
                  )}
                </td>
              );
            })}
            <td className="py-1 text-muted-foreground">{gtText[h]}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** A bar either side of zero: the input's push on one model's logit. */
function EffectBar({ v, lim, color, dark }: { v: number; lim: number; color: string; dark: boolean }) {
  const w = 64;
  const half = w / 2;
  const len = Number.isFinite(v) ? Math.min(half, (Math.abs(v) / lim) * half) : 0;
  const x = v >= 0 ? half : half - len;
  return (
    <span className="inline-flex items-center gap-1 tabular-nums">
      <svg width={w} height={10} aria-hidden>
        <line x1={half} x2={half} y1={0} y2={10} stroke={dark ? "#4b5563" : "#c3cad3"} />
        <rect x={x} y={2} width={Math.max(len, 0.5)} height={6} rx={1} fill={v >= 0 ? color : GUIDE} />
      </svg>
      <span className="w-10 text-right">{Number.isFinite(v) ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}` : "—"}</span>
    </span>
  );
}

function Inputs({ data, dark }: { data: MlHeads; dark: boolean }) {
  const [sortBy, setSortBy] = useState<string>("human:lr");
  const [all, setAll] = useState(false);
  const n = data.features.length;
  const lim = useMemo(() => {
    const out: Record<string, number> = {};
    for (const [h, k] of COLUMNS) {
      out[`${h}:${k}`] = Math.max(0.5, ...data.heads[h][k].effect.map((v) => (Number.isFinite(v) ? Math.abs(v) : 0)));
    }
    return out;
  }, [data]);
  const order = useMemo(() => {
    const idx = Array.from({ length: n }, (_, i) => i);
    if (sortBy === "order") return idx;
    const [h, k] = sortBy.split(":") as [MlHeadName, MlKind];
    const e = data.heads[h][k].effect;
    return idx.sort((a, b) => Math.abs(e[b] || 0) - Math.abs(e[a] || 0));
  }, [data, sortBy, n]);
  const rows = all ? order : order.slice(0, TOP);
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
        <span className="text-xs font-medium text-foreground">The 43 inputs and how each model weighed them</span>
        <span>sort by</span>
        <select
          className="rounded border bg-background px-1 py-0.5 text-[11px]"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
        >
          {COLUMNS.map(([h, k]) => (
            <option key={`${h}:${k}`} value={`${h}:${k}`}>|push| on {h} {KIND_NAME[k]}</option>
          ))}
          <option value="order">feature order</option>
        </select>
        <Button variant="outline" size="sm" className="h-6 px-2 text-[11px]" onClick={() => setAll(!all)}>
          {all ? `top ${TOP}` : `show all ${n}`}
        </Button>
      </div>
      <div className="overflow-x-auto">
        <table className="text-[11px] tabular-nums">
          <thead>
            <tr className="text-muted-foreground">
              <th className="pr-3 text-left font-normal">column</th>
              <th className="pr-3 text-right font-normal">value</th>
              {COLUMNS.map(([h, k]) => (
                <th key={`${h}:${k}`} className="pr-3 text-left font-normal" style={{ color: HEAD_COLOR[h] }}>
                  {h} · {k === "lr" ? "LR" : "hidden"}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((i) => {
              const f = data.features[i];
              const lab = data.labels[f];
              return (
                <tr key={f} className="border-t border-border/40">
                  <td className="pr-3 py-0.5" title={lab?.description ?? f}>
                    <span className="text-foreground">{lab?.label ?? f}</span>{" "}
                    <span className="text-muted-foreground">{lab?.test ? `· ${lab.test}` : ""}</span>
                  </td>
                  <td className="pr-3 py-0.5 text-right">
                    {data.imputed[i] ? <span className="text-muted-foreground" title="missing: the training median is used">missing</span> : fmt(data.value[i])}
                  </td>
                  {COLUMNS.map(([h, k]) => (
                    <td key={`${h}:${k}`} className="pr-3 py-0.5">
                      <EffectBar v={data.heads[h][k].effect[i]} lim={lim[`${h}:${k}`]} color={HEAD_COLOR[h]} dark={dark} />
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-muted-foreground leading-relaxed">
        Push on the logit, towards the label (coloured, right) or away from it (grey, left). LR: weight × z, exact —
        the pushes plus the intercept are the logit. Hidden layer: how much the logit falls when that one input is set
        to its training mean; the inputs interact, so these do not add up to the logit. Hover a column for what it measures.
      </p>
    </div>
  );
}

function loeoTitle(head: MlHeadName, t: MlLoeo, env: string): string {
  const e = t.envs[env];
  if (!e) return "";
  const parts = [`${e.correct}/${e.n} correct`];
  if (e.precision != null) parts.push(`precision ${pct(e.precision)}`);
  if (e.recall != null) parts.push(`recall ${pct(e.recall)}`);
  if (e.f1 != null) parts.push(`F1 ${pct(e.f1)}`);
  if (head === "human") {
    if (e.kpi1 != null) parts.push(`KPI#1 (no mover) ${pct(e.kpi1)}`);
    if (e.kpi2 != null) parts.push(`KPI#2 (mover) ${pct(e.kpi2)}`);
    if (e.nonhuman != null) parts.push(`non-human correct ${pct(e.nonhuman)}`);
  } else {
    if (e.personOnly != null) parts.push(`person alone kept as no anomaly ${pct(e.personOnly)}`);
    if (e.empty != null) parts.push(`empty kept ${pct(e.empty)}`);
  }
  return parts.join(" · ");
}

function Loeo({ data }: { data: MlHeads }) {
  return (
    <div className="space-y-1">
      <div className="text-xs font-medium">Leave-one-environment-out accuracy (%)</div>
      <div className="overflow-x-auto">
        <table className="text-[11px] tabular-nums">
          <thead>
            <tr className="text-muted-foreground">
              <th className="pr-4 text-left font-normal">held out →</th>
              {ENVS.map((e) => (
                <th key={e} className="pr-3 text-right font-normal">
                  {e} <span className="opacity-70">({data.trainedOn[e]?.n ?? "?"})</span>
                </th>
              ))}
              <th className="pr-3 text-right font-normal">pooled</th>
              <th className="text-right font-normal">mean</th>
            </tr>
          </thead>
          <tbody>
            {(["human", "anomaly"] as MlHeadName[]).map((h) => (
              <Fragment key={h}>
                <tr>
                  <td colSpan={ENVS.length + 3} className="pt-2 font-medium" style={{ color: HEAD_COLOR[h] }}>{HEAD_NAME[h]}</td>
                </tr>
                {LOEO_ROWS.map(([key, name, ref]) => {
                  const t = data.loeo[h][key];
                  if (!t) return null;
                  return (
                    <tr key={`${h}-${key}`} className={ref ? "text-muted-foreground" : ""}>
                      <td className="pr-4">{name}{ref ? " (reference)" : ""}</td>
                      {ENVS.map((e) => (
                        <td key={e} className="pr-3 text-right" title={loeoTitle(h, t, e)}>{pct(t.envs[e]?.accuracy)}</td>
                      ))}
                      <td className="pr-3 text-right" title={`${t.pooled.correct}/${t.pooled.n}`}>{pct(t.pooled.accuracy)}</td>
                      <td className="text-right">{pct(t.meanOfEnvs)}</td>
                    </tr>
                  );
                })}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-muted-foreground leading-relaxed">
        Each dataset held out in turn, the model trained on the other four (column header: minutes in it); pooled =
        every held-out minute together. Hover a cell for precision, recall, F1 and the KPI split. The models on this
        tab are trained on all five ({Object.values(data.trainedOn).reduce((a, b) => a + b.n, 0)} minutes,
        fitted {data.fitted}), without augmentation. Human label: the binary GT. Anomaly label: the recorded condition
        names a fan, vacuum or robot, with or without a person; env3-1 has none, so there it only counts false alarms.
        “★ marked” = the classifier tab's starred columns.
      </p>
    </div>
  );
}
