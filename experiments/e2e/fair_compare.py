"""Like-for-like comparison of the e2e candidate with the base physics models 13F / 15F (cyphy session, 2026-11-17;
user: "현재 우리 feature 뽑는 모델 (13F, 15F)랑 공정하게 비교해봐").

  e2e   sbo.B.lfhftl.st16.att, 3 crops x 2 bin offsets x seeds 0-2 (the 1.6-s board setting), refit models of
        tta_study.py (out/tta); all4 3x2x3 alongside
  13F   tfm.seq13pw.fixed5, 15F tfm.seq15pwu.fixed5 (5-seed mean P > 0.5): bench_rows p_*, per-seed challengers
        predictions, A / B per seed from cy_short_b5 (B = B's own minutes 0-4), inner 2-room models from cy_phys_nested

Same windows, same training-room sets per fold, same reference windows (the 20 fixed install empties; A env3's, B its
own minutes 0-4), scenario A / B / C scored by the env1+env2 models for all three.
Prints and writes out/fair/fair.json:
  table     Env 1..3-1 / Total / Scenario A / B / C / Env 2 + 110: KPI#1, KPI#2, Overall
  boot      paired session bootstrap (sessions resampled within room, B = 2000): e2e minus 13F / 15F
  seeds     every 3-of-5 seed subset (min .. max) for all three: how much of a gap is the draw of seeds
  inner     the 12 (2-room model, held-out room) pairs both families have: BA and accuracy on identical rows
  cond      accuracy per condition group (object type x activity), outer rows; A / B per label
  steps     inner-pair accuracy per condition group along the e2e build steps (for the step table)
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2e as E  # noqa: E402
import evaluate as V  # noqa: E402
from tta_study import TTA, pick  # noqa: E402

LAB = Path("/home/lg_csi/ml_lab/data")
CH = LAB / "challengers/repo/experiments/challengers"
OUT = HERE / "out" / "fair"
M = V.M
RID = M.row_id.to_numpy()
ROWS = pd.read_csv(CH / "manifests/frozen/rows.csv", dtype={"row_id": str, "session": str})
PHYS = {"13F": "seq13pw", "15F": "seq15pwu"}
import os  # noqa: E402
# E2E_CFG / FAIR_OUT (11-17): rerun the comparison for another e2e candidate (e.g. the .noaug refit) into its own file
E2E = {"e2e": (os.environ.get("E2E_CFG", "sbo.B.lfhftl.st16.att"), (3, 2)), "e2e_all4": ("sbo.B.all4.st16.att", (3, 2))}
FAIR_OUT = os.environ.get("FAIR_OUT", "fair")


# --------------------------------------------------------------------------------------------- per-row logits
def e2e_ens(cfg, nc, no, seeds):
    out = {}
    for rs in E.room_sets():
        Ls = [np.load(TTA / cfg / f"{E.rs_name(rs)}__s{s}.npy") for s in seeds]
        out[rs] = np.mean([pick(L, nc, no, L.shape[2]) for L in Ls], axis=0)
    return out


def e2e_rows(ens) -> pd.DataFrame:
    Lo = {f: ens[tuple(r for r in E.ROOMS if r != f)] for f in E.ROOMS}
    Lo["scnC"] = ens[("env1", "env2")]
    D = V.assemble(Lo)
    D = D[~D.set.isin(["A", "B"]) | (D.ctx == "scnC")]           # A / B scored by the env1+env2 model
    return D[D.set != "1109"].set_index("row_id")


def phys_seed_p(name: str) -> pd.DataFrame:
    """(row_id x seed) P of a physics model: challengers predictions (loeo, scnC, env2_later), A / B from cy_short_b5."""
    P = pd.read_csv(CH / f"predictions/tfm.{name}.fixed5.csv.gz", dtype={"row_id": str, "seed": str})
    P = P[(P.form == "deployed") & (P.seed != "ens")]
    T = P.pivot_table(index="row_id", columns="seed", values="score").rename(columns=int)
    B5 = pd.read_pickle(LAB / "cy_short_b5.pkl")
    ab = {}
    for (Dd, fs, sd), o in B5:
        if Dd == 60 and fs == name:
            for s in ("A", "B"):
                for w, p in enumerate(o[s]):
                    ab.setdefault(f"{s}:{w}", {})[int(sd)] = float(p)
    return pd.concat([T, pd.DataFrame(ab).T.sort_index(axis=1)])


def phys_rows(name: str, base: pd.DataFrame) -> pd.Series:
    """Deployed 5-seed P per row: bench_rows (the 1,426 benchmark windows), A / B 5-seed mean of cy_short_b5."""
    R = pd.read_csv(HERE / "bench_rows.csv.gz", dtype={"row_id": str}).set_index("row_id")[f"p_{name}"]
    S = phys_seed_p(name)
    ab = S.loc[[i for i in S.index if i[:2] in ("A:", "B:")]].mean(axis=1)
    fresh = pd.read_csv(LAB / "cy_phys_fresh.csv")
    fa = fresh[fresh.part == "A"].set_index(fresh[fresh.part == "A"].w.map(lambda w: f"A:{w}"))[f"{name}|scnC"]
    # A: the frozen deployed weights (cy_phys_fresh); B: the same-seed refits with B's own reference (cy_short_b5)
    print(f"{name}: A |cy_short_b5 - cy_phys_fresh| max {np.abs(ab.loc[fa.index] - fa).max():.2e}, "
          f"verdict flips {int(((ab.loc[fa.index] > 0.5) != (fa > 0.5)).sum())}")
    ab = pd.concat([fa, ab[[i for i in ab.index if i.startswith("B:")]]])
    p = pd.concat([R, ab]).reindex(base.index)
    assert p.notna().all(), p[p.isna()].index[:5]
    d = (S.loc[S.index.intersection(R.index)].mean(axis=1) - R.reindex(S.index.intersection(R.index))).abs().max()
    assert d < 1e-6, f"per-seed mean differs from bench_rows by {d}"
    return p


# ------------------------------------------------------------------------------------------------- scoring
def subsets(D: pd.DataFrame) -> dict[str, pd.Series]:
    lo = (D.part == "bench") & (D.set == "loeo")
    s = {f"Env {r[3:]}": lo & (D.room == r) for r in E.ROOMS}
    s["Total"] = lo
    s["Scenario A"] = D.set == "A"
    s["Scenario B"] = D.set == "B"
    s["Scenario C"] = D.set == "scnC"
    s["Env 2 + 110"] = (D.part == "bench") & (D.room == "env2") & D.set.isin(["loeo", "env2_later", "unused"])
    return s


def kpi(ok: np.ndarray, obj: np.ndarray) -> dict:
    r = lambda m: round(float(100 * ok[m].mean()), 1) if m.any() else None  # noqa: E731
    return {"k1": r(obj == 0), "k2": r(obj == 1), "all": r(np.ones_like(ok, bool))}


def sessions(D: pd.DataFrame) -> pd.Series:
    s = ROWS.set_index("row_id").session.reindex(D.index)
    fill = pd.Series([f"{D.room[i]}|{i[:8]}" if ":" not in i else i.split(":")[0] for i in D.index], index=D.index)
    return s.fillna(fill)


def boot(D: pd.DataFrame, m: pd.Series, a: str, b: str, B: int = 2000, seed: int = 20261117) -> dict:
    q = D[m]
    rng = np.random.default_rng(seed)
    groups = [[np.flatnonzero((q.room == r).to_numpy() & (q.session == s).to_numpy()) for s in q[q.room == r].session.unique()]
              for r in q.room.unique()]
    oa, ob, obj = q[f"ok_{a}"].to_numpy(), q[f"ok_{b}"].to_numpy(), q.obj.to_numpy()
    n_sess = sum(len(g) for g in groups)
    d0 = {k: round(v - w, 1) if v is not None and w is not None else None
          for (k, v), w in zip(kpi(oa, obj).items(), kpi(ob, obj).values())}
    if n_sess < 3:
        return {"diff": d0, "ci": None, "n_sessions": n_sess}
    D_ = []
    for _ in range(B):
        ii = np.concatenate([np.concatenate([g[k] for k in rng.integers(0, len(g), len(g))]) for g in groups])
        ka, kb = kpi(oa[ii], obj[ii]), kpi(ob[ii], obj[ii])
        D_.append([(ka[k] - kb[k]) if ka[k] is not None and kb[k] is not None else np.nan for k in ("k1", "k2", "all")])
    D_ = np.array(D_, float)
    out = {"diff": d0, "n_sessions": n_sess, "ci": {}, "p_le0": {}}
    for j, k in enumerate(("k1", "k2", "all")):
        v = D_[:, j]
        if np.isfinite(v).sum() > B // 2:
            out["ci"][k] = [round(float(np.nanpercentile(v, 2.5)), 1), round(float(np.nanpercentile(v, 97.5)), 1)]
            out["p_le0"][k] = round(float(np.nanmean(v <= 0)), 3)
    return out


GROUPS = [("빈방", "none", ("empty",)), ("사람 정지", "none", ("still",)), ("사람 작은 움직임", "none", ("small",)),
          ("사람 큰 움직임", "none", ("large",)), ("회전 선풍기만", "rot_fan", ("empty",)),
          ("고정 선풍기만", "static_fan", ("empty",)), ("로봇청소기만", "robot_vacuum", ("empty",)),
          ("회전 선풍기 + 사람", "rot_fan", ("still", "small")), ("고정 선풍기 + 사람", "static_fan", ("still", "small")),
          ("로봇청소기 + 사람", "robot_vacuum", ("still", "still_or_small")),
          ("scnC 사람 (기계 없음)", "none", ("scenario_c_natural",)), ("scnC 천장 선풍기만", "ceiling_fan", ("empty",)),
          ("scnC 천장 선풍기 + 사람", "ceiling_fan", ("scenario_c_natural",))]


def group_of(index) -> pd.Series:
    R = ROWS.set_index("row_id").reindex(index)
    g = pd.Series(index=index, dtype=object)
    for name, ot, acts in GROUPS:
        g[(R.obj_type == ot).to_numpy() & R.activity.isin(acts).to_numpy()] = name
    return g


# ------------------------------------------------------------------------------------------------- inner
def phys_inner(name: str) -> dict:
    """(training rooms, held-out room) -> P on the held-out room's rows (row_id index), 5-seed mean."""
    out = {}
    for (v, f, r), te, p in pd.read_pickle(LAB / "cy_phys_nested.pkl"):
        if v == f"tfm.{name}.fixed5" and f != "scnC":
            rs = tuple(x for x in E.ROOMS if x not in (f, r))
            out[(rs, r)] = pd.Series(p, index=ROWS.row_id.to_numpy()[te])
    return out


def inner_scores(pred: dict) -> dict:
    """pred[(rs, h)] = pd.Series(logit or P - 0.5, index row_id) on the scored rows of h."""
    ba, acc, k1, k2 = [], [], [], []
    for (rs, h), s in sorted(pred.items()):
        y = M.set_index("row_id").y.reindex(s.index).to_numpy(); o = M.set_index("row_id").obj.reindex(s.index).to_numpy()
        ok = (s.to_numpy() > 0) == (y == 1)
        ba.append(50 * (ok[y == 1].mean() + ok[y == 0].mean())); acc.append(100 * ok.mean())
        k1.append(100 * ok[o == 0].mean())
        if (o == 1).any():
            k2.append(100 * ok[o == 1].mean())
    return {"BA": round(float(np.mean(ba)), 1), "acc": round(float(np.mean(acc)), 1), "k1": round(float(np.mean(k1)), 1),
            "k2": round(float(np.mean(k2)), 1), "pairs": len(ba)}


def e2e_inner(ens) -> dict:
    out = {}
    for rs, h in V.inner_pairs():
        m = V.scored(h)
        out[(rs, h)] = pd.Series(ens[rs][m], index=RID[m])
    return out


def pred_inner(cfg) -> dict:
    """Original-generation inner logits (out/pred, mean over the seeds present, test-time inputs as stored)."""
    out = {}
    for rs, h in V.inner_pairs():
        p = V.ens(cfg, rs)
        if p is None:
            return {}
        m = V.scored(h)
        out[(rs, h)] = pd.Series(p[m], index=RID[m])
    return out


def group_acc(pred: dict) -> dict:
    s = pd.concat(list(pred.values()))
    y = M.set_index("row_id").y.reindex(s.index).to_numpy()
    ok = pd.Series((s.to_numpy() > 0) == (y == 1), index=s.index)
    g = group_of(s.index)
    return {name: round(float(100 * ok[g == name].mean()), 1) for name, _, _ in GROUPS if (g == name).any()}


# --------------------------------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    ens = {k: e2e_ens(c, *s, seeds=range(3)) for k, (c, s) in E2E.items()}
    D = e2e_rows(ens["e2e"])
    D["logit_e2e_all4"] = e2e_rows(ens["e2e_all4"]).logit.reindex(D.index)
    D = D.rename(columns={"logit": "logit_e2e"})
    for k, n in PHYS.items():
        D[f"p_{k}"] = phys_rows(n, D)
    D["session"] = sessions(D)
    for k in ("e2e", "e2e_all4"):
        D[f"ok_{k}"] = (D[f"logit_{k}"] > 0) == (D.y == 1)
    for k in PHYS:
        D[f"ok_{k}"] = (D[f"p_{k}"] > 0.5) == (D.y == 1)
    S = subsets(D)
    models = ["e2e", "13F", "15F", "e2e_all4"]
    res["n_rows"] = {k: int(m.sum()) for k, m in S.items()}
    res["table"] = {k: {mo: kpi(D[m][f"ok_{mo}"].to_numpy(), D[m].obj.to_numpy()) for mo in models} for k, m in S.items()}
    res["boot"] = {k: {b: boot(D, m, "e2e", b) for b in ("13F", "15F")} for k, m in S.items()}

    # seeds: every 3-of-5 subset, Overall per table row
    sp = {}
    for k, (c, s) in E2E.items():
        if k != "e2e":
            continue
        vals = {r: [] for r in S}
        for sub in itertools.combinations(range(5), 3):
            Dk = e2e_rows(e2e_ens(c, *s, seeds=sub)).reindex(D.index)
            ok = (Dk.logit > 0) == (D.y == 1)
            for r, m in S.items():
                vals[r].append(100 * ok[m].mean())
        sp[k] = {r: [round(min(v), 1), round(max(v), 1)] for r, v in vals.items()}
    for k, n in PHYS.items():
        Sd = phys_seed_p(n)
        vals = {r: [] for r in S}
        for sub in itertools.combinations(range(5), 3):
            p = Sd[list(sub)].mean(axis=1).reindex(D.index)
            ok = (p > 0.5) == (D.y == 1)
            for r, m in S.items():
                mm = m & p.notna()
                if mm.sum() == m.sum():
                    vals[r].append(100 * ok[mm].mean())
        sp[k + " (3 seeds)"] = {r: [round(min(v), 1), round(max(v), 1)] for r, v in vals.items() if v}
    res["seeds"] = sp

    # inner: identical (2-room set, held-out room) pairs
    inn = {"e2e": e2e_inner(ens["e2e"]), "e2e_all4": e2e_inner(ens["e2e_all4"])}
    for k, n in PHYS.items():
        inn[k] = {key: s - 0.5 for key, s in phys_inner(n).items()}
    for key, s in inn["13F"].items():
        assert set(s.index) == set(inn["e2e"][key].index), ("inner rows differ", key)
    res["inner"] = {k: inner_scores(v) for k, v in inn.items()}
    # inner bootstrap: pooled accuracy over the 12 pairs, sessions resampled within held-out room
    Ii = []
    for key in sorted(inn["e2e"]):
        idx = inn["e2e"][key].index
        q = pd.DataFrame({"room": key[1], "obj": M.set_index("row_id").obj.reindex(idx).to_numpy(),
                          "session": ROWS.set_index("row_id").session.reindex(idx).to_numpy(),
                          "pair": "+".join(key[0]) + ">" + key[1]}, index=idx)
        y = M.set_index("row_id").y.reindex(idx).to_numpy()
        for k in ("e2e", "13F", "15F"):
            q[f"ok_{k}"] = (inn[k][key].reindex(idx).to_numpy() > 0) == (y == 1)
        Ii.append(q.reset_index(drop=True))
    Ii = pd.concat(Ii, ignore_index=True)
    Ii["session"] = Ii.pair + "|" + Ii.session.astype(str)
    Ii["room"] = Ii.pair
    res["inner_boot"] = {b: boot(Ii, pd.Series(True, index=Ii.index), "e2e", b) for b in ("13F", "15F")}

    # condition groups, outer rows (LOEO + scnC); A / B by label
    G = group_of(D.index)
    cond = {}
    for name, _, _ in GROUPS:
        m = (G == name) & D.set.isin(["loeo", "scnC"])
        if m.any():
            cond[name] = {mo: round(float(100 * D[m][f"ok_{mo}"].mean()), 1) for mo in models}
    for s in ("A", "B"):
        for (y, o), lab in (((0, 0), "빈방"), ((1, 0), "사람"), ((0, 1), "선풍기만"), ((1, 1), "선풍기 + 사람")):
            m = (D.set == s) & (D.y == y) & (D.obj == o)
            if m.any():
                cond[f"Scenario {s} {lab}"] = {mo: round(float(100 * D[m][f"ok_{mo}"].mean()), 1) for mo in models}
    res["cond"] = cond

    # e2e build steps on the inner pairs (condition groups + BA)
    steps = [("sbo.A.all4", "pred", None), ("sbo.B.all4", "pred", None), ("sbo.B.all4.st16", "pred", None),
             ("sbo.B.all4.st16.att", "pred", None), ("sbo.B.all4.st16.att", "tta", (3, 16, 5)),
             ("sbo.B.lfhf.st16.att", "tta", (3, 16, 5)), ("sbo.B.lfhftl.st16.att", "tta", (3, 16, 5)),
             ("sbo.B.lfhftl.st16.att", "tta", (3, 2, 3))]
    st = {}
    for c, src, s in steps:
        if src == "pred":
            pr = pred_inner(c)
            key = f"{c} (first fit)"
        else:
            pr = e2e_inner(e2e_ens(c, s[0], s[1], range(s[2])))
            key = f"{c} {s[0]}x{s[1]}x{s[2]} (refit)"
        if pr:
            st[key] = {"inner": inner_scores(pr), "groups": group_acc(pr)}
    for k in PHYS:
        st[k] = {"inner": res["inner"][k], "groups": group_acc(inn[k])}
    res["steps"] = st
    (OUT / f"{FAIR_OUT}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    D.drop(columns=[c for c in D.columns if c.startswith("ok_")]).to_csv(OUT / f"{FAIR_OUT}_rows.csv.gz")
    report(res)


def report(res):
    f = lambda d: "-" if d is None else f"{d}"  # noqa: E731
    print("== table (KPI#1 / KPI#2 / Overall), n rows:", res["n_rows"])
    for r, v in res["table"].items():
        print(f"{r:14s} " + " | ".join(f"{m} {f(x['k1'])}/{f(x['k2'])}/{f(x['all'])}" for m, x in v.items()))
    print("\n== paired session bootstrap, e2e minus 13F / 15F: diff [95% CI] (Overall; K1; K2)")
    for r, v in res["boot"].items():
        print(f"{r:14s} " + " | ".join(
            f"{b}: {x['diff']['all']} {x['ci'].get('all') if x['ci'] else 'n/a'} ; K1 {x['diff']['k1']} "
            f"{x['ci'].get('k1') if x['ci'] else ''} ; K2 {x['diff']['k2']} {x['ci'].get('k2') if x['ci'] else ''} "
            f"(sess {x['n_sessions']})" for b, x in v.items()))
    print("\n== 3-of-5 seed subsets, Overall min..max")
    for k, v in res["seeds"].items():
        print(f"{k:16s} " + " | ".join(f"{r} {a}..{b}" for r, (a, b) in v.items()))
    print("\n== inner pairs (2-room models on the held-out room, identical rows)")
    for k, v in res["inner"].items():
        print(f"{k:10s} {v}")
    for b, x in res["inner_boot"].items():
        print(f"inner boot e2e - {b}: {x['diff']} {x['ci']}")
    print("\n== condition groups (outer rows), % correct")
    for g, v in res["cond"].items():
        print(f"{g:24s} " + " | ".join(f"{m} {x}" for m, x in v.items()))
    print("\n== steps (inner pairs)")
    names = [g for g, _, _ in GROUPS]
    print("step".ljust(48) + "BA    " + " ".join(n[:6].ljust(7) for n in names))
    for k, v in res["steps"].items():
        print(k.ljust(48) + f"{v['inner']['BA']:5.1f} " + " ".join(f"{v['groups'].get(n, float('nan')):6.1f} " for n in names))


if __name__ == "__main__":
    main()
