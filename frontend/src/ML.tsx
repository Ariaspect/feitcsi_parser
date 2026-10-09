import { useEffect, useRef, useState, type ReactNode } from "react";

import { ChartBusy } from "@/components/ChartBusy";
import { Button } from "@/components/ui/button";
import { fetchAf8, type Af8, type Af8Feature } from "./api";
import { Chart } from "./Chart";
import { ticks } from "./series";

// The AF8 classifier (backend/af8.py) on one 60 s window: the verdict, how the
// model weighed each feature, and one panel per feature showing the series it
// was read from. The numbers are af8's own -- the board computes the same.

const PERSON = "#2f6fed";
const RED = "#d62728";
const GUIDE = "#9aa5b1";
const LAG_COLOR: Record<string, string> = { "0.25": "#d97a2f", "2": "#2f7c5c", "5": "#7b5ea7" };

const DESCRIPTION: Record<Af8Feature, string> = {
  A_slope3:
    "D(τ) is the median step |r(t) − r(t−τ)| / (|r(t)| + |r(t−τ)|) at gap τ. The slope of log D " +
    "against log τ over τ = 0.25, 2 and 5 s (coloured points, line) says how fast change grows with " +
    "the gap, whatever the signal strength. Grey points are other gaps, drawn for context only.",
  A_r025_2:
    "D(0.25 s) ÷ D(2 s): how much of the 2 s change already happens within a quarter second. " +
    "Lines: the step at each gap, one median per second. Dashed: D, the median over the window.",
  A_r5_2:
    "D(5 s) ÷ D(2 s): whether change keeps growing past 2 s. Lines and dashes as above.",
  A_p90:
    "The 2 s step averaged in 1 s blocks; the 90th percentile of the blocks (dashed) -- " +
    "roughly the level of the busiest ~6 seconds of the minute.",
  F_pkmax:
    "FarSense breathing score: the first autocorrelation peak of each 10 s window, 1 s apart " +
    "(1 = perfectly repeating). Dashed red: the highest. Blue dots: windows at or above 0.25.",
  F_pkmed: "The same window peaks; dashed red: their median.",
  F_run:
    "The longest run of consecutive windows with peak ≥ 0.25 (shaded), in seconds -- " +
    "breathing has to persist, not flicker.",
  F_rpm_sd:
    "Breathing rate of each window; blue where its peak ≥ 0.25. The feature is the spread (SD) of " +
    "the blue rates (dashed: mean ± SD). Missing when fewer than two -- the model then uses its " +
    "training median.",
  C3_revisit:
    "Revisit: D(τ) at τ = 10–30 s, 1 s apart (line), and its running maximum (dashed). C3 is the " +
    "largest fall of D below that maximum, as a fraction of it (red). A rotating fan's head comes " +
    "back to the same angle every swing, so the channel returns to an earlier state and D dips at " +
    "the swing period (~0.7). A person never retraces a path exactly: D only grows, C3 near 0.",
};

function fmt(v: number | null | undefined, digits = 4): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(digits);
}

/** Upper y bound that keeps every value and guide in view, with headroom. */
function ceiling(...groups: (number | null | undefined)[][]): number {
  let hi = 0;
  for (const g of groups) for (const v of g) if (v != null && Number.isFinite(v)) hi = Math.max(hi, v);
  return hi > 0 ? hi * 1.15 : 1;
}

export interface MLProps {
  path: string;
  dark: boolean;
}

export function ML({ path, dark }: MLProps) {
  const [win, setWin] = useState(0);
  const [data, setData] = useState<Af8 | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const holder = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(880);
  useEffect(() => {
    const node = holder.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.max(320, Math.floor(entry.contentRect.width)));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  // A new capture starts at its first window.
  useEffect(() => setWin(0), [path]);

  useEffect(() => {
    const controller = new AbortController();
    setBusy(true);
    fetchAf8(path, win, controller.signal)
      .then((d) => { setData(d); setError(null); })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setData(null);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [path, win]);

  // Two panels a row when there is room for them.
  const twoUp = width >= 900;
  const panelWidth = twoUp ? Math.floor((width - 16) / 2) : width;

  return (
    <div className="space-y-4" ref={holder}>
      {error ? (
        <div className="text-muted-foreground p-8 text-sm">No AF8 verdict for this capture — {error}</div>
      ) : !data ? (
        <ChartBusy busy empty height={150} label="computing the nine features" />
      ) : (
        <>
          <Header data={data} win={win} setWin={setWin} busy={busy} />
          <ModelBars data={data} width={width} dark={dark} />
          <div className={twoUp ? "grid grid-cols-2 gap-4" : "space-y-4"}>
            {data.model.features.map((f, i) => (
              <FeaturePanel key={f} name={f} data={data} index={i} busy={busy}>
                <FeatureChart name={f} data={data} width={panelWidth} dark={dark} />
              </FeaturePanel>
            ))}
          </div>
          <p className="text-[11px] text-muted-foreground leading-relaxed">
            The classifier is <code>backend/af8.py</code> — the code the LG board runs. The window is
            what it was trained on: a capture under 90 s is one window, [0, 61) s; a longer one is
            cut into 60 s windows. Frames: the dominant AP, 2×1, full width; every 8th of the live
            subcarriers ({data.subcarriers} here); the CSI ratio r = H1/H0 on an even {data.fsHz.toFixed(1)} Hz
            grid. The camera, where there is one, is shown for comparison and never used.
          </p>
        </>
      )}
    </div>
  );
}

function Header({ data, win, setWin, busy }: {
  data: Af8; win: number; setWin: (w: number) => void; busy: boolean;
}) {
  const person = data.label === 1;
  const [t0, t1] = data.windowS;
  const cam = data.camera?.fraction;
  return (
    <div className="flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
      <span
        className="rounded px-2 py-1 text-sm font-semibold text-white"
        style={{ background: person ? PERSON : "#6b7280" }}
        title="p(person) > 0.5 → person"
      >
        {person ? "PERSON" : "NO PERSON"} · p = {data.pPerson.toFixed(3)}
      </span>
      <span>
        camera:{" "}
        {cam == null
          ? "no label"
          : <b className="text-foreground">{`person in ${(cam * 100).toFixed(0)} % of frames`}</b>}
      </span>
      {data.windows.length > 1 && (
        <span className="flex items-center gap-1">
          <Button variant="outline" size="sm" className="h-7 px-2" disabled={win === 0}
                  onClick={() => setWin(win - 1)}>‹</Button>
          window {win + 1} of {data.windows.length}
          <Button variant="outline" size="sm" className="h-7 px-2"
                  disabled={win >= data.windows.length - 1}
                  onClick={() => setWin(win + 1)}>›</Button>
        </span>
      )}
      <span className="tabular-nums">
        {t0.toFixed(0)}–{t1.toFixed(0)} s · {data.seconds.toFixed(1)} s of grid · {data.subcarriers} subcarriers ·{" "}
        {data.fsHz.toFixed(1)} Hz
      </span>
      {busy && <span>updating…</span>}
    </div>
  );
}

/** weight × z for every feature, as bars either side of zero, then the sum. */
function ModelBars({ data, width, dark }: { data: Af8; width: number; dark: boolean }) {
  const m = data.model;
  const row = 22;
  const labelW = 210;
  const valueW = 150;
  const barW = Math.max(120, width - labelW - valueW - 16);
  const lim = Math.max(1, ...m.contribution.map((c) => Math.abs(c)));
  const x = (v: number) => labelW + barW / 2 + (v / lim) * (barW / 2);
  const text = dark ? "#c7ccd4" : "#3a4048";
  const muted = dark ? "#8b95a3" : "#6b7480";
  const height = row * (m.features.length + 2) + 8;
  return (
    <div className="space-y-1">
      <div className="text-xs font-medium">How the model decided</div>
      <svg width={width} height={height} role="img" aria-label="feature contributions">
        <line x1={x(0)} x2={x(0)} y1={0} y2={height} stroke={muted} strokeWidth={1} />
        {m.features.map((f, i) => {
          const c = m.contribution[i];
          const y = 4 + i * row;
          return (
            <g key={f}>
              <text x={0} y={y + 14} fontSize={11} fill={text}>{f}</text>
              <text x={98} y={y + 14} fontSize={11} fill={muted}>
                {m.imputed[i] ? `missing → ${fmt(m.value[i], 3)}` : fmt(m.value[i], 4)}
              </text>
              <rect
                x={Math.min(x(0), x(c))} y={y + 4} height={row - 8}
                width={Math.max(1, Math.abs(x(c) - x(0)))}
                fill={c >= 0 ? PERSON : GUIDE}
              />
              <text x={labelW + barW + 8} y={y + 14} fontSize={11} fill={text} className="tabular-nums">
                {c >= 0 ? "+" : ""}{c.toFixed(2)}  (z {m.z[i].toFixed(2)})
              </text>
            </g>
          );
        })}
        <text x={0} y={4 + m.features.length * row + 14} fontSize={11} fill={muted}>intercept</text>
        <text x={labelW + barW + 8} y={4 + m.features.length * row + 14} fontSize={11} fill={muted}>
          {m.intercept >= 0 ? "+" : ""}{m.intercept.toFixed(2)}
        </text>
        <text x={0} y={4 + (m.features.length + 1) * row + 14} fontSize={11} fill={text} fontWeight={600}>
          logit = {m.logit.toFixed(2)} → p(person) = {data.pPerson.toFixed(3)}
        </text>
        <text x={x(0) - 6} y={height - 2} fontSize={9} fill={muted} textAnchor="end">← no person</text>
        <text x={x(0) + 6} y={height - 2} fontSize={9} fill={muted}>person →</text>
      </svg>
      <p className="text-[11px] text-muted-foreground leading-relaxed">
        Each feature becomes a z-score with the training mean and scale; weight × z is its push
        towards person (blue, right) or no person (grey, left). Intercept + the pushes = logit, and
        p(person) = 1 / (1 + e<sup>−logit</sup>): person when p &gt; 0.5.
      </p>
    </div>
  );
}

function FeaturePanel({ name, data, index, busy, children }: {
  name: Af8Feature; data: Af8; index: number; busy: boolean; children: ReactNode;
}) {
  const c = data.model.contribution[index];
  const raw = data.features[name];
  return (
    <div className="space-y-1 rounded-md border p-3">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="text-sm font-semibold">{name}</span>
        <span className="text-sm tabular-nums">
          = {raw == null ? `missing (model uses ${fmt(data.model.value[index], 3)})` : fmt(raw, 4)}
        </span>
        <span className="text-[11px] tabular-nums" style={{ color: c >= 0 ? PERSON : GUIDE }}>
          {c >= 0 ? "+" : ""}{c.toFixed(2)} to the logit
        </span>
      </div>
      <p className="text-[11px] text-muted-foreground leading-relaxed">{DESCRIPTION[name]}</p>
      <ChartBusy busy={busy} height={150} label="updating">{children}</ChartBusy>
    </div>
  );
}

function FeatureChart({ name, data, width, dark }: {
  name: Af8Feature; data: Af8; width: number; dark: boolean;
}) {
  const w = Math.max(280, width - 26);
  const mo = data.motion;
  const br = data.breath;
  const D = (tau: number) => mo.D[mo.lags.indexOf(tau)];
  const timeDomain: [number, number] = [data.windowS[0], data.windowS[0] + data.seconds];
  const muted = dark ? "#4b5563" : "#c3cad3";

  if (name === "A_slope3") return <LogLogChart data={data} width={w} dark={dark} />;
  if (name === "C3_revisit") return <RevisitChart data={data} width={w} dark={dark} />;

  if (name === "A_r025_2" || name === "A_r5_2") {
    const pair = name === "A_r025_2" ? ["0.25", "2"] : ["5", "2"];
    const a = mo.step[pair[0]] ?? [];
    const b = mo.step[pair[1]] ?? [];
    return (
      <>
      <Chart
        width={w} dark={dark} height={150} times={mo.cells} domain={timeDomain}
        yDomain={[0, ceiling(a, b)]} yLabel="step (per-second median)"
        series={[
          { values: a, color: LAG_COLOR[pair[0]], width: 1.4, label: `step at ${pair[0]} s` },
          { values: b, color: LAG_COLOR[pair[1]], width: 1.4, label: `step at ${pair[1]} s` },
        ]}
        guides={[
          { value: D(Number(pair[0])), color: LAG_COLOR[pair[0]], label: `D(${pair[0]} s)` },
          { value: D(Number(pair[1])), color: LAG_COLOR[pair[1]], label: `D(${pair[1]} s)` },
        ]}
      />
      <div className="flex flex-wrap gap-3 text-[11px] text-muted-foreground">
        {pair.map((g) => (
          <span key={g} style={{ color: LAG_COLOR[g] }}>
            — step at {g} s (D = {fmt(D(Number(g)), 3)})
          </span>
        ))}
        <span>dashed: D, the window median</span>
      </div>
      </>
    );
  }

  if (name === "A_p90") {
    const v = data.features.A_p90 ?? NaN;
    return (
      <Chart
        width={w} dark={dark} height={150} times={mo.p90Time} domain={timeDomain}
        yDomain={[0, ceiling(mo.p90Values, [v])]} yLabel="2 s step, 1 s mean"
        series={[{ values: mo.p90Values, color: LAG_COLOR["2"], width: 1.6, label: "per second" }]}
        guides={[{ value: v, color: RED, label: "P90" }]}
      />
    );
  }

  const dotColors = br.good.map((g) => (g ? PERSON : muted));

  if (name === "F_rpm_sd") {
    const sd = data.features.F_rpm_sd;
    const mean = br.rpmMean;
    const guides = mean == null ? [] : [
      { value: mean, color: RED, label: "mean" },
      ...(sd == null ? [] : [
        { value: mean + sd, color: GUIDE, label: "+SD" },
        { value: mean - sd, color: GUIDE, label: "−SD" },
      ]),
    ];
    return (
      <Chart
        width={w} dark={dark} height={150} times={br.time} domain={timeDomain}
        yDomain={[10, 30]} yLabel="breathing rate (rpm)" series={[]}
        dots={{ values: br.rpm, colors: dotColors, radius: 2.5, label: "rate" }}
        guides={guides}
      />
    );
  }

  // F_pkmax, F_pkmed, F_run: the window peaks, each panel marking its own reading.
  const peaks = br.peak;
  const finite = peaks.map((v, i) => [v, i] as const).filter(([v]) => v != null && Number.isFinite(v));
  const argmax = finite.length ? finite.reduce((a, b) => ((b[0] as number) > (a[0] as number) ? b : a))[1] : -1;
  const guides =
    name === "F_pkmax" ? [{ value: data.features.F_pkmax ?? NaN, color: RED, label: "max" }]
    : name === "F_pkmed" ? [{ value: data.features.F_pkmed ?? NaN, color: RED, label: "median" }]
    : [];
  return (
    <Chart
      width={w} dark={dark} height={150} times={br.time} domain={timeDomain}
      yDomain={[-0.2, 1]} yLabel="window peak (ACF)"
      series={[{ values: peaks, color: PERSON, width: 1.2, label: "peak" }]}
      dots={{ values: peaks, colors: dotColors, radius: 2.2, label: "peak" }}
      guides={[{ value: br.threshold, color: GUIDE, label: "0.25" }, ...guides]}
      bands={name === "F_run" && br.run ? [{ t0: br.run.t0 - 0.5, t1: br.run.t1 + 0.5 }] : []}
      bandColor="#2f9c6f"
      marker={name === "F_pkmax" && argmax >= 0 ? br.time[argmax] : null}
    />
  );
}

/** D(τ) against τ on log-log axes, the three model lags highlighted and fitted. */
function LogLogChart({ data, width, dark }: { data: Af8; width: number; dark: boolean }) {
  const mo = data.motion;
  const slope = data.features.A_slope3 ?? NaN;
  const height = 150;
  const M = { top: 12, right: 12, bottom: 22, left: 48 };
  const iw = width - M.left - M.right;
  const ih = height - M.top - M.bottom;
  const pts = mo.lags.map((t, i) => [t, mo.D[i]] as const).filter(([t, d]) => t > 0 && d > 0);
  if (!pts.length) return <div className="text-[11px] text-muted-foreground">No D(τ) to draw.</div>;
  const lx = pts.map(([t]) => Math.log(t));
  const ly = pts.map(([, d]) => Math.log(d));
  const x0 = Math.min(...lx), x1 = Math.max(...lx);
  const y0 = Math.min(...ly) - 0.15, y1 = Math.max(...ly) + 0.15;
  const X = (v: number) => M.left + ((Math.log(v) - x0) / (x1 - x0 || 1)) * iw;
  const Y = (v: number) => M.top + (1 - (Math.log(v) - y0) / (y1 - y0 || 1)) * ih;
  const grid = dark ? "#2a2f37" : "#e6e9ee";
  const text = dark ? "#8b95a3" : "#6b7480";
  // Plain "nice" ticks over the D range, placed on the log axis. D often spans
  // less than a factor of two, where 1-2-5 decade ticks would leave none.
  const yTicks = ticks(Math.exp(y0), Math.exp(y1), 4).filter((v) => v > 0);
  const fitAt = (t: number) => Math.exp(slope * Math.log(t) + mo.fitIntercept);
  const [s0, s2] = [mo.shapeLags[0], mo.shapeLags[mo.shapeLags.length - 1]];
  return (
    <svg width={width} height={height} role="img" aria-label="D(tau) on log-log axes">
      {yTicks.map((v) => (
        <g key={v}>
          <line x1={M.left} x2={M.left + iw} y1={Y(v)} y2={Y(v)} stroke={grid} />
          <text x={M.left - 6} y={Y(v)} dy="0.32em" textAnchor="end" fontSize={9} fill={text}>{+v.toPrecision(3)}</text>
        </g>
      ))}
      {pts.map(([t]) => (
        <text key={t} x={X(t)} y={height - 6} textAnchor="middle" fontSize={9} fill={text}>{t}s</text>
      ))}
      <polyline
        fill="none" stroke={GUIDE} strokeWidth={1}
        points={pts.map(([t, d]) => `${X(t)},${Y(d)}`).join(" ")}
      />
      {Number.isFinite(slope) && (
        <line x1={X(s0)} y1={Y(fitAt(s0))} x2={X(s2)} y2={Y(fitAt(s2))} stroke={RED} strokeWidth={1.4} strokeDasharray="4 3" />
      )}
      {pts.map(([t, d]) => {
        const key = String(t);
        const model = mo.shapeLags.includes(t);
        return (
          <circle key={key} cx={X(t)} cy={Y(d)} r={model ? 4 : 2.5}
                  fill={model ? (LAG_COLOR[key] ?? PERSON) : GUIDE} />
        );
      })}
      <text x={M.left - 44} y={M.top - 2} fontSize={9} fill={text}>D(τ), log</text>
      <text x={M.left + iw} y={M.top + 10} textAnchor="end" fontSize={10} fill={RED}>
        slope {Number.isFinite(slope) ? slope.toFixed(3) : "—"}
      </text>
    </svg>
  );
}

/** D(τ) over 10–30 s with its running maximum; the largest drop below it is C3. */
function RevisitChart({ data, width, dark }: { data: Af8; width: number; dark: boolean }) {
  const rv = data.revisit;
  const c3 = data.features.C3_revisit;
  const height = 150;
  const M = { top: 14, right: 12, bottom: 22, left: 48 };
  const iw = width - M.left - M.right;
  const ih = height - M.top - M.bottom;
  const pts = rv.lags.map((t, i) => [t, rv.D[i], rv.runningMax[i]] as const).filter(([, d]) => Number.isFinite(d));
  if (pts.length < 2) return <div className="text-[11px] text-muted-foreground">No D(τ) over 10–30 s to draw.</div>;
  const t0 = pts[0][0], t1 = pts[pts.length - 1][0];
  const hi = Math.max(...pts.map(([, , m]) => m)) * 1.12 || 1;
  const X = (t: number) => M.left + ((t - t0) / (t1 - t0 || 1)) * iw;
  const Y = (v: number) => M.top + (1 - v / hi) * ih;
  const grid = dark ? "#2a2f37" : "#e6e9ee";
  const text = dark ? "#8b95a3" : "#6b7480";
  const at = rv.at;
  const atIdx = at == null ? -1 : pts.findIndex(([t]) => t === at);
  return (
    <svg width={width} height={height} role="img" aria-label="D(tau) from 10 to 30 s with its running maximum">
      {ticks(0, hi, 4).map((v) => (
        <g key={v}>
          <line x1={M.left} x2={M.left + iw} y1={Y(v)} y2={Y(v)} stroke={grid} />
          <text x={M.left - 6} y={Y(v)} dy="0.32em" textAnchor="end" fontSize={9} fill={text}>{+v.toPrecision(3)}</text>
        </g>
      ))}
      {pts.filter(([t]) => t % 5 === 0).map(([t]) => (
        <text key={t} x={X(t)} y={height - 6} textAnchor="middle" fontSize={9} fill={text}>{t}s</text>
      ))}
      <polyline fill="none" stroke={GUIDE} strokeWidth={1.2} strokeDasharray="4 3"
                points={pts.map(([t, , m]) => `${X(t)},${Y(m)}`).join(" ")} />
      <polyline fill="none" stroke={LAG_COLOR["2"]} strokeWidth={1.6}
                points={pts.map(([t, d]) => `${X(t)},${Y(d)}`).join(" ")} />
      {pts.map(([t, d]) => <circle key={t} cx={X(t)} cy={Y(d)} r={2} fill={LAG_COLOR["2"]} />)}
      {atIdx >= 0 && (
        <g>
          <line x1={X(pts[atIdx][0])} x2={X(pts[atIdx][0])} y1={Y(pts[atIdx][2])} y2={Y(pts[atIdx][1])}
                stroke={RED} strokeWidth={2} />
          <circle cx={X(pts[atIdx][0])} cy={Y(pts[atIdx][1])} r={3.5} fill={RED} />
        </g>
      )}
      <text x={M.left - 44} y={M.top - 4} fontSize={9} fill={text}>D(τ)</text>
      <text x={M.left + iw} y={M.top + 2} textAnchor="end" fontSize={10} fill={RED}>
        C3 {c3 == null ? "—" : c3.toFixed(3)}{at == null ? "" : ` at ${at} s`}
      </text>
    </svg>
  );
}
