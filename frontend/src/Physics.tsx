import { useEffect, useRef, useState, type ReactNode } from "react";

import { ChartBusy } from "@/components/ChartBusy";
import { Button } from "@/components/ui/button";
import { fetchTfmPhys, type PhysAcc, type PhysResults, type PhysTiming, type PhysVerdict, type TfmPhys } from "./api";
import { Chart } from "./Chart";
import { NO_PERSON, PERSON, VerdictChip, WindowNav } from "./ModelCommon";
import { bridge, finite, sparse } from "./Transformer";

// The Physics tab (backend/tfm_phys.py): the default Transformer's network with physics channels in place of the
// reflection and breathing channels that machines fool -- seq13pw and seq15pwu of the challenger benchmark's
// channel-set study (exp/challengers, CHANNELSET_REPORT.md) -- with the 17-channel base under the same calibration.

const NAMES: Record<string, string> = { seq13pw: "seq13pw (13 ch)", seq15pwu: "seq15pwu (15 ch)", seq17: "base (17 ch)" };
const ORDER = ["seq13pw", "seq15pwu", "seq17"];

const DESCRIPTION: Record<string, string> = {
  s025: "Median step |r(t) − r(t − 0.25 s)| / (|r(t)| + |r(t − 0.25 s)|) over subcarriers, one value a second: fast motion. Kept from the base.",
  s2: "The same step at a 2 s gap. Kept from the base.",
  lgrow:
    "log(s5 / s025), not calibrated (replaces s5). A displacement smaller than a wavelength — breathing, posture — keeps growing from 0.25 s to 5 s; " +
    "blade vibration, a moving vacuum (more than λ/2 within 0.25 s) and noise stay flat. Empty room and static fan ≈ 0, still person 0.05–0.26.",
  g15_res:
    "2 s step of the direct-path-gated signal (g15's) after its two strongest 1 s high-passed directions are projected out (replaces g15). " +
    "Blades or a vacuum in one spot move only those few patterns; a person spreads over many. env1 static fan 0.028 and vacuum 0.029 ≈ empty 0.024.",
  sT:
    "Step at one swing period T (replaces ret = step at T ÷ step at T/2). A fan alone comes back after one swing; a person does not. " +
    "Dividing by the half-swing step let a strongly seen swing hide a person (env3 fan + person).",
  u_g15v: "g15_res with swing_a × its swing-phase average removed too: the reflection change that is not locked to the fan's swing.",
  pk_w:
    "FarSense breathing peak after w × the swing-phase average is subtracted, w = max(swing_a, swing_ab) (replaces pk, pk_diff, pk_res). " +
    "Only as much swing as is seen comes out: with no fan (w = 0) it equals the plain pk, so a still person's breathing is kept.",
  bs_w: "Breathing prominence of the subcarrier-pattern components on the same w-subtracted signal (replaces bs_all, bs_r1, bs_r2, bs_tap).",
  rpm: "FarSense breathing rate (not calibrated). Kept from the base.",
  bs_f: "Frequency of the strongest breathing line (not calibrated). Kept from the base.",
  plf_res: "Phase-locked fraction on the fan-subspace residual: how much of the 0.25 s motion energy repeats at the same swing phase (not calibrated). Kept from the base.",
  plf_g: "Phase-locked fraction on the full signal (not calibrated). Kept from the base.",
  swing_ab:
    "Share of the breathing-band change (1 s minus 6 s moving average) locked to the swing phase, above the chance share of independent cycles, in [0, 1]; " +
    "one value per window, not calibrated. The evidence that a swing is there: rotating fan 0.51–0.99, people / empty / vacuum 0–0.08.",
  pk_u: "seq15pwu only. FarSense peak with the whole swing-phase average removed: a rotating fan alone drops to the empty level even where its swing is seen weakly.",
  bs_u: "seq15pwu only. Breathing prominence with the whole swing-phase average removed. Rotating fan alone 2.0–2.4 ≈ empty, with a person 3.5–3.9.",
};

// The 17 -> 13 / 15 change, one row per channel.
const CHANGES: { base: string; phys: string; m13: boolean; m15: boolean; why: string }[] = [
  { base: "s025, s2", phys: "s025, s2", m13: true, m15: true, why: "kept: fast and 2 s motion" },
  { base: "rpm, bs_f", phys: "rpm, bs_f", m13: true, m15: true, why: "kept: breathing rate / line frequency (not calibrated)" },
  { base: "plf_res, plf_g", phys: "plf_res, plf_g", m13: true, m15: true, why: "kept: phase-locked fraction (not calibrated)" },
  { base: "s5", phys: "lgrow", m13: true, m15: true, why: "growth with the gap, not the size: slow displacement vs vibration" },
  { base: "g15", phys: "g15_res, u_g15v", m13: true, m15: true, why: "the reflection change could not tell a machine in the reflections from a still person (the core confusion)" },
  { base: "ret", phys: "sT", m13: true, m15: true, why: "a strongly seen swing made the ratio hide a person" },
  { base: "pk, pk_diff, pk_res", phys: "pk_w", m13: true, m15: true, why: "rotating-fan harmonics read as breathing" },
  { base: "bs_all, bs_r1, bs_r2, bs_tap", phys: "bs_w", m13: true, m15: true, why: "same: rotating fan alone vs with a person AUC 0.28–0.82 on bs_all" },
  { base: "drift", phys: "—", m13: false, m15: false, why: "removed, no replacement: any mover turns the leading direction (its own effect not measured)" },
  { base: "—", phys: "swing_ab", m13: true, m15: true, why: "new: is a swing there at all (how much to subtract)" },
  { base: "—", phys: "pk_u, bs_u", m13: false, m15: true, why: "seq15pwu: the fully swing-removed breathing next to the w-removed one" },
];

const ACC_SETS: [string, string, string][] = [
  ["loeo", "4 rooms held out in turn (LOEO)", "benchmark"],
  ["scnC", "scenario C (env1 + env2 model)", "benchmark"],
  ["env2_later", "env2 later sessions (people only)", "benchmark, not used to pick"],
  ["unused", "80 labelled windows the benchmark never used", "same sessions, train=False"],
  ["rooms_all", "every labelled room window (LOEO + env2 later + the 80)", "all"],
];

const STRATA: [string, string, boolean][] = [
  ["empty_no_object", "empty room: false alarms", true],
  ["object_only", "machine only (fan, vacuum): false alarms", true],
  ["person_no_object", "person, no machine: found", false],
  ["person_plus_object", "person + machine: found", false],
];

const pct = (v: number | null | undefined): string => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(1));
const sec = (v: number | null | undefined): string => (v == null || !Number.isFinite(v) ? "—" : v < 0.1 ? `${(v * 1e3).toFixed(1)} ms` : `${v.toFixed(2)} s`);

function Verdict({ name, v, truth, threshold }: { name: string; v: PhysVerdict | undefined; truth: TfmPhys["truth"]; threshold: number }) {
  if (!v) return null;
  return (
    <div className="space-y-1">
      <div className="text-[11px] font-medium">{name}</div>
      <VerdictChip p={v.p} label={v.label} truth={truth} />
      <div className="flex flex-wrap items-center gap-1 text-[11px] text-muted-foreground tabular-nums">
        <span>P per seed:</span>
        {v.p_seeds.map((p, i) => (
          <span key={i} className="rounded px-1.5 py-0.5 text-white" style={{ background: p != null && p > threshold ? PERSON : NO_PERSON }}>
            {p == null ? "—" : p.toFixed(3)}
          </span>
        ))}
      </div>
    </div>
  );
}

function Table({ head, rows, note }: { head: ReactNode[]; rows: ReactNode[][]; note?: ReactNode }) {
  return (
    <div className="space-y-1">
      <div className="overflow-x-auto">
        <table className="text-[11px] tabular-nums">
          <thead>
            <tr className="text-muted-foreground">
              {head.map((h, i) => <th key={i} className={`pr-3 font-normal ${i === 0 ? "text-left" : "text-right"}`}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-t border-muted">
                {r.map((c, j) => <td key={j} className={`pr-3 py-0.5 ${j === 0 ? "text-left" : "text-right"}`}>{c}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {note && <p className="text-[11px] text-muted-foreground leading-relaxed">{note}</p>}
    </div>
  );
}

function AccCell({ a }: { a: PhysAcc | undefined }) {
  if (!a) return <>—</>;
  return <>{pct(a.all)} <span className="text-muted-foreground">({pct(a.kpi1)} / {pct(a.kpi2)})</span></>;
}

function Verification({ r }: { r: PhysResults }) {
  const v = r.verification;
  if (!v) return <div className="text-[11px] text-muted-foreground">Not computed yet (scripts/verify_tfm_phys.py on lg).</div>;
  const rooms = ["env1", "env2", "env3", "env3-1"];
  const chMax = (k: string, skip: string[] = []) =>
    Math.max(...Object.entries(v.channels[k]?.max_rel_diff ?? { x: 0 }).filter(([c]) => !skip.includes(c)).map(([, d]) => d));
  const lgrowDiff = v.channels.phys?.max_rel_diff?.lgrow;
  const nanMis = (k: string) => Object.values(v.channels[k]?.nan_mismatch ?? {}).reduce((a, b) => a + b, 0);
  const condRows = v.conditions
    .filter((c) => ORDER.some((m) => Number(c[`wrong_${m}`]) >= 3))
    .sort((a, b) => String(a.set + a.room + a.cond).localeCompare(String(b.set + b.room + b.cond)));
  return (
    <div className="space-y-4">
      <p className="text-[11px] text-muted-foreground leading-relaxed">
        Every labelled window recomputed from the raw capture through this tab's code — {v.computed} of {v.windows} windows
        ({v.n_errors} errors, {v.references_n} of them the rooms' calibration minutes, which are never scored), on lg in {v.wall_s.toFixed(0)} s.
        Physics channels vs the benchmark's inputs: max relative difference {chMax("phys", ["lgrow"]).toExponential(1)} (the benchmark stored its
        inputs as float32){lgrowDiff != null && <>, lgrow {lgrowDiff.toExponential(1)} (a log of a ratio near 1, here from the unrounded steps)</>},
        missing-value mismatches {nanMis("phys")}; the 17: {chMax("base").toExponential(1)}, {nanMis("base")}. Probabilities vs the benchmark's saved deployed predictions (5-seed mean):{" "}
        {ORDER.map((m) => `${NAMES[m]} max |ΔP| ${v.probabilities[m]?.max_abs_diff.toExponential(1)}, ${v.probabilities[m]?.verdict_flips} verdict flips of ${v.probabilities[m]?.n}`).join("; ")}.
        The tab's room lookup picks the benchmark's fold and reference room for {v.tab_room_lookup_ok} of {v.windows} windows.
      </p>
      <div className="space-y-1">
        <div className="text-xs font-medium">Accuracy (%) — all (KPI#1 no machine / KPI#2 machine)</div>
        <Table
          head={["set", "n", ...ORDER.map((m) => NAMES[m])]}
          rows={ACC_SETS.flatMap(([k, name, tag]) => {
            const acc = (m: string) => v.accuracy[m]?.[k];
            const head: ReactNode[] = [
              <span key="n" className="font-semibold">{name} <span className="font-normal text-muted-foreground">· {tag}</span></span>,
              acc("seq17")?.all?.n ?? "",
              ...ORDER.map((m) => <span key={m} className="font-semibold"><AccCell a={acc(m)?.all} />{k === "loeo" && acc(m)?.all?.ba != null ? <span className="text-muted-foreground"> · BA {pct(acc(m)?.all?.ba)}</span> : null}</span>),
            ];
            const per = k === "scnC" ? [] : rooms.filter((rm) => acc("seq17")?.[rm]).map((rm) => [
              <span key="r" className="pl-4">{rm}</span>, acc("seq17")?.[rm]?.n ?? "", ...ORDER.map((m) => <AccCell key={m} a={acc(m)?.[rm]} />),
            ]);
            return [head, ...per];
          })}
          note={<>All three calibrated with the room's five install empty minutes and judged by the model that never saw the room (scenario C by the env1 + env2 model with env3-1's minutes); P = mean of the 5 seeds, person if P &gt; 0.5. The 80 unused windows are captures of sessions the benchmark already has (env1 10/12–10/14, env2 11/05), so they are not independent new data. The two sets were picked from 31 on the benchmark's own windows: their benchmark numbers are optimistic until scored on new sessions.</>}
        />
      </div>
      <div className="space-y-1">
        <div className="text-xs font-medium">By kind of window (LOEO, %)</div>
        <Table
          head={["", ...ORDER.map((m) => NAMES[m])]}
          rows={STRATA.map(([k, name, fa]) => [name, ...ORDER.map((m) => {
            const a = v.strata_loeo[m]?.[k];
            return pct(a == null ? null : fa ? 100 - a : a);
          })])}
        />
      </div>
      <div className="space-y-1">
        <div className="text-xs font-medium">Conditions with ≥ 3 wrong windows in any model (wrong / windows)</div>
        <Table
          head={["set", "room", "condition", "person", ...ORDER.map((m) => NAMES[m])]}
          rows={condRows.map((c) => [String(c.set), String(c.room), String(c.cond), c.y ? "yes" : "no", ...ORDER.map((m) => `${c[`wrong_${m}`]} / ${c.n}`)])}
        />
      </div>
    </div>
  );
}

function Timing({ t, title }: { t: PhysTiming | undefined; title: string }) {
  if (!t) return null;
  return (
    <div className="space-y-1">
      <div className="text-xs font-medium">{title}</div>
      <Table
        head={["stage", "median", "max"]}
        rows={t.stages.map((s) => [s, sec(t.median[s]), sec(t.max[s])])}
        note={<>{t.machine}. {t.note} {t.rows.length} one-minute windows × {t.reps} repetitions, median per window, then median / max over the windows.{t.p_match ? ` ${t.p_match}` : ""}</>}
      />
      <details className="text-[11px] text-muted-foreground">
        <summary className="cursor-pointer">per window</summary>
        <Table
          head={["capture", "room", "w", ...t.stages]}
          rows={t.rows.map((r) => [r.stem, r.room ?? "—", r.w == null ? "—" : r.w.toFixed(2), ...t.stages.map((s) => sec(r.t[s]))])}
        />
      </details>
    </div>
  );
}

export function Physics({ path, dark }: { path: string; dark: boolean }) {
  const [win, setWin] = useState(0);
  const [data, setData] = useState<TfmPhys | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState<"calibrated" | "raw">("calibrated");

  const holder = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(880);
  useEffect(() => {
    const node = holder.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(320, Math.floor(entry.contentRect.width))));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => setWin(0), [path]);

  useEffect(() => {
    const controller = new AbortController();
    setBusy(true);
    fetchTfmPhys(path, win, controller.signal)
      .then((d) => { setData(d); setError(null); })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setData(null);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [path, win]);

  const cols = width >= 1200 ? 3 : width >= 760 ? 2 : 1;
  const panelWidth = Math.floor((width - 16 * (cols - 1)) / cols);
  const grid = view === "calibrated" && data?.calibrated ? data.calibrated : data?.raw;
  const calibrated = new Set(data?.models.seq15pwu?.calibrated_channels ?? []);
  const only15 = new Set((data?.models.seq15pwu?.channels ?? []).filter((c) => !(data?.models.seq13pw?.channels ?? []).includes(c)));
  const res = data?.results ?? {};
  const loeo = (m: string) => res.verification?.accuracy[m]?.loeo?.all;
  const scnc = (m: string) => res.verification?.accuracy[m]?.scnC?.all;

  return (
    <div className="space-y-4" ref={holder}>
      <section className="space-y-3 rounded-md border p-3">
        <div className="flex flex-wrap items-baseline gap-2">
          <span className="text-sm font-semibold">Physics channels — seq13pw / seq15pwu</span>
          <span className="text-[11px] text-muted-foreground">
            the default Transformer's network on physics channels · room calibrated with its five install empty minutes · 1 token a second, one encoder layer (d 16) · 5 seeds averaged · one model per held-out room
          </span>
          {busy && data && <span className="text-[11px] text-muted-foreground">updating…</span>}
        </div>
        <p className="text-[11px] text-muted-foreground leading-relaxed">
          The base Transformer's persistent errors were machines alone called people (fans, the vacuum: 36.7 % of machine-only windows) and people under a
          rotating fan missed. Occlusion showed the cause was the input, not the network: g15, the motion seen in the reflections, cannot tell a machine in
          the reflections from a still person, and the breathing channels read a rotating fan's harmonics as breathing. These two sets replace those channels
          with ones that measure the physical difference — whether the change grows with the gap (displacement vs vibration), whether it is confined to a
          few fixed patterns (a compact scatterer), whether it comes back every swing (a fan) — and give the swing evidence so the network learns how much to
          subtract. No hand-set threshold anywhere. seq15pwu adds the fully swing-removed breathing for weakly seen fans.
          {loeo("seq13pw") && (
            <> LOEO (KPI#1 / KPI#2): base {pct(loeo("seq17")?.kpi1)} / {pct(loeo("seq17")?.kpi2)}, seq13pw {pct(loeo("seq13pw")?.kpi1)} / {pct(loeo("seq13pw")?.kpi2)},
              seq15pwu {pct(loeo("seq15pwu")?.kpi1)} / {pct(loeo("seq15pwu")?.kpi2)}; scenario C {pct(scnc("seq17")?.all)} / {pct(scnc("seq13pw")?.all)} / {pct(scnc("seq15pwu")?.all)}.
              The price: part of the still people seen only in the reflections (scenario C, env2 later sessions), the env3 robot vacuum alone
              (1 → 6 / 7 of 20 windows called a person), and for seq15pwu more missed env2 people (standing / walking 1 → 8 of 14) — see the
              condition table below.</>
          )}
        </p>
        {error ? (
          <div className="text-muted-foreground text-sm">No verdict for this window — {error}</div>
        ) : !data ? (
          <ChartBusy busy empty height={90} label="computing the physics channels" />
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-3">
              <WindowNav windows={data.windows} win={win} setWin={setWin} />
              <span className="text-[11px] text-muted-foreground tabular-nums">
                {data.window_s[0].toFixed(0)}–{data.window_s[1].toFixed(0)} s · swing period T {data.swing_period_s == null ? "—" : `${data.swing_period_s.toFixed(1)} s`}
                {" "}· swing_a {data.swing_a == null ? "—" : data.swing_a.toFixed(2)} · swing_ab {data.swing_ab == null ? "—" : data.swing_ab.toFixed(2)}
                {" "}· removed w = {data.w == null ? "—" : data.w.toFixed(2)} of the swing-phase average
              </span>
            </div>
            {data.room ? (
              <div className="text-[11px] text-muted-foreground">
                room <b className="text-foreground">{data.room.room}</b> ({data.room.source}) · judged by the models trained on{" "}
                <b className="text-foreground">{(data.verdicts.seq13pw?.trained_on ?? []).join(" + ")}</b> — they never saw this room · calibrated with{" "}
                <b className="text-foreground">{data.room.calibrated_with}</b>'s five install empty minutes
                {data.reference_minute && <b className="text-foreground"> · this window is one of those five minutes: judged in-sample, never scored</b>}
              </div>
            ) : (
              <div className="text-[11px] text-muted-foreground">Room unknown — no empty-room reference to calibrate against, so no verdict.</div>
            )}
            {data.room && (
              <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${Math.min(3, cols + 1)}, minmax(0, 1fr))` }}>
                {ORDER.map((m) => <Verdict key={m} name={NAMES[m] + (m === "seq17" ? " · same calibration, for comparison" : "")} v={data.verdicts[m]} truth={data.truth} threshold={data.threshold} />)}
              </div>
            )}
            <div className="text-[11px] text-muted-foreground tabular-nums">
              computed on this server: {Object.entries(data.timings).map(([k, v]) => `${k} ${sec(v)}`).join(" · ")}
            </div>
          </>
        )}
      </section>

      <section className="space-y-3 rounded-md border p-3">
        <div className="text-sm font-semibold">What changed from the base's 17 channels</div>
        <Table
          head={["base (17)", "physics", "seq13pw", "seq15pwu", "why"]}
          rows={CHANGES.map((c) => [c.base, <b key="p">{c.phys}</b>, c.m13 ? "✓" : "", c.m15 ? "✓" : "", <span key="w" className="text-muted-foreground">{c.why}</span>])}
          note={<>Same network, training rows, folds, seeds and 25 epochs as the base; only the input changes (17 × 61 → 13 × 61 / 15 × 61, plus one availability row per channel). Step channels (s025, s2, g15_res, sT, u_g15v) are logged and, with pk_w, bs_w, pk_u and bs_u, calibrated against the room's empty minutes; lgrow, swing_ab, rpm, bs_f and plf_* are ratios or rates and are not. Putting the raw g15 back (seq14pwg) brought the machine false alarms back (env1 static fan 2 → 11 wrong).</>}
        />
      </section>

      {data && (
        <section className="space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
            <span className="text-sm font-semibold text-foreground">The inputs ({data.channels.length} channels × 61 s)</span>
            <Button variant={view === "calibrated" ? "default" : "outline"} size="sm" className="h-6 px-2 text-[11px]"
                    disabled={!data.calibrated} onClick={() => setView("calibrated")}>calibrated (model input)</Button>
            <Button variant={view === "raw" ? "default" : "outline"} size="sm" className="h-6 px-2 text-[11px]"
                    onClick={() => setView("raw")}>raw</Button>
            <span>calibrated: log on {data.models.seq15pwu?.log_channels.join(", ")}; then (x − the room's empty median) ÷ empty MAD (shrunk halfway to the training rooms') on {data.models.seq15pwu?.calibrated_channels.join(", ")}</span>
          </div>
          {grid && (
            <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
              {data.channels.map((c, i) => {
                const vals = grid[i];
                const f = finite(vals);
                const lo = f.length ? Math.min(...f) : 0;
                const hi = f.length ? Math.max(...f) : 1;
                const pad = hi > lo ? 0.08 * (hi - lo) : 0.5;
                const times = vals.map((_, k) => data.window_s[0] + k);
                const isCal = view === "calibrated" && calibrated.has(c);
                const guides = isCal ? [{ value: 0, color: "#9aa5b1", label: "empty median" }] : [];
                const color = only15.has(c) ? "#7c3aed" : c.startsWith("plf") || c === "swing_ab" ? "#b45309" : PERSON;
                const gappy = sparse(vals);
                return (
                  <div key={c} className="space-y-1 min-w-0">
                    <div className="text-xs" title={DESCRIPTION[c]}>
                      <b>{c}</b> {only15.has(c) && <span className="text-[10px] rounded border px-1">seq15pwu only</span>}{" "}
                      <span className="text-muted-foreground">{f.length} of {vals.length} s with a value{gappy ? " · dots = values, dashed = drawn between them (the model sees the gaps as missing)" : ""}</span>
                    </div>
                    <Chart
                      width={panelWidth} dark={dark} height={120} times={times}
                      domain={[data.window_s[0], data.window_s[0] + vals.length - 1]}
                      yDomain={[lo - pad, hi + pad]} yLabel={isCal ? "z vs empty" : c}
                      series={gappy
                        ? [{ values: bridge(vals), color, width: 1, label: c, dashed: true }]
                        : [{ values: vals, color, width: 1.3, label: c }]}
                      dots={gappy ? { values: vals, colors: vals.map(() => color), radius: 2.5, label: c } : undefined}
                      guides={guides}
                    />
                    <div className="text-[11px] text-muted-foreground leading-snug">{DESCRIPTION[c]}</div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      )}

      {data && (
        <section className="space-y-3 rounded-md border p-3">
          <div className="text-sm font-semibold">Full-dataset check</div>
          <Verification r={res} />
        </section>
      )}

      {data && (
        <section className="space-y-3 rounded-md border p-3">
          <div className="text-sm font-semibold">Time for one judgment (one-minute window)</div>
          {res.board ? <Timing t={res.board} title="On the board" /> : (
            <div className="text-[11px] text-muted-foreground">Board: not measured yet (scripts/bench_tfm_phys_board.py).</div>
          )}
          <Timing t={res.lg} title="On lg (one core), same script" />
        </section>
      )}
    </div>
  );
}
