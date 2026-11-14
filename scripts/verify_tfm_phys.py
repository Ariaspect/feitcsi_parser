"""Full-dataset check of the Physics tab's models through the tab's own code (backend/tfm_phys.py, backend/tfm_series.py).

Every labelled window -- the challenger benchmark's 1,346 (LOEO 1,097 incl. the 20 reference minutes, scenario C 149,
env2 later 100) and the 80 GT-table windows the benchmark never used (env1 10/12-10/14 and env2 11/05 captures of the
same sessions, train=False in data/gt_tables.csv) -- is recomputed from the raw capture: the physics channels and the 17,
calibration against the room's five install empty minutes, the 5-seed forward of seq13pw, seq15pwu and the base seq17.

Checks: (1) the channels equal the benchmark's inputs (inputs/newch.npz, inputs/seq17.npz); (2) the probabilities equal
the benchmark's saved deployed predictions; (3) the tab's room lookup (tfm_model.room_for) picks the benchmark's fold.
Then the accuracy tables (KPI#1 = no object, KPI#2 = object) on the benchmark sets, on the 80 unused windows and on
every labelled window, and the error count of each condition.

Usage on lg (the benchmark mirror's venv; CODE = a copy of this repo's backend/ + data/):
  cd /home/lg_csi/ml_lab/data/challengers/repo/experiments/challengers
  ../../../venv/bin/python <this file> CODE OUT_DIR [NPROC]
Writes OUT_DIR/verify_rows.csv.gz and OUT_DIR/verify_summary.json. cyphy session 2026-11-12.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# "--rescore OUT_DIR": recompute the scores of OUT_DIR/verify_summary.json from its saved rows (no capture, no backend).
RESCORE = __name__ == "__main__" and sys.argv[1:2] == ["--rescore"]
if not RESCORE:
    HERE = Path.cwd()
    CODE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
    NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    sys.path.insert(0, str(CODE))
    sys.path.insert(1, str(HERE / "bench"))
    from backend import mtk, tfm_model, tfm_phys as TP, tfm_phys_fast as TF, tfm_series  # noqa: E402
    CAP = Path("/home/lg_csi/lg_csi_captures")
    SCN = CAP / "20261023" / "20261023_132301.bin"
    MODELS = TP.MODELS


def capture(stem: str) -> Path:
    p = CAP / stem[:8] / f"{stem}.bin"
    return p if p.exists() else next(CAP.rglob(f"{stem}.bin"))


def one(job: tuple) -> dict:
    rid, stem, path, t0, t1, room = job
    try:
        idx = mtk.MTKIndex(Path(path))
        s = TF.series(path, t0, t1, idx)                # the tab's series (backend.tfm_phys_fast, since 2026-11-15)
        b = tfm_series.series(path, t0, t1, idx)
        out = {"row_id": rid, "phys": np.array([s["S"][c] for c in TP.PHYS_CHANNELS]), "base": b["X"],
               "swing_ab": s["swing_ab"], "w": s["w"], "T_s": s["T"], "error": ""}
        for m in MODELS:
            X = b["X"] if m == "seq17" else TP.matrix(s["S"], m)
            out[f"p_{m}"] = TP.judge(X, m, room)[1]["p"]
        return out
    except Exception as exc:  # noqa: BLE001 -- recorded and counted, never silently scored
        return {"row_id": rid, "error": repr(exc)}


def main() -> None:
    import run as R                                   # benchmark rows, splits, saved inputs
    from common import ALL_CHANNELS, CHANNELS
    from transforms import all_series

    rows = R.ROWS.copy()
    z = np.load(HERE / "inputs" / "newch.npz")
    S_ALL = all_series(R.S_RAW, z["X_new"], z["channels"].tolist())
    rows["set_"] = rows["set"]
    calroom = {f: R.SPLITS[f]["calibration_room_of_test"] for f in R.SPLITS}
    gt = pd.read_csv(CODE / "data" / "gt_tables.csv", dtype={"stem": str})
    extra = gt[~gt.stem.isin(rows.key)].copy()
    lab = rows[rows.room.isin(["env1", "env2"])].groupby(["room", "cond"])[["obj", "stratum", "activity"]].first()
    jobs, meta = [], []
    for r in rows.itertuples():
        if r.ds == "scnC":
            k = int(str(r.key)[1:]); path, t0, t1, stem = SCN, 60.0 * k, 60.0 * k + 60.0, SCN.stem
        else:
            path, t0, t1, stem = capture(str(r.key)), 0.0, 61.0, str(r.key)
        room = {"room": r.room, "fold": r.fold, "calibrated_with": calroom[r.fold]}
        jobs.append((r.row_id, stem, str(path), t0, t1, room))
        tab = tfm_model.room_for(stem)
        meta.append({"row_id": r.row_id, "stem": stem, "set": r.set, "room": r.room, "fold": r.fold, "cond": r.cond,
                     "y": int(r.y), "obj": int(r.obj), "stratum": r.stratum, "reference": bool(r.f5_reference),
                     "tab_fold_ok": bool(tab is not None and tab["fold"] == r.fold and tab["calibrated_with"] == calroom[r.fold])})
    for e in extra.itertuples():
        o, st, act = lab.loc[(e.env, e.cond)]
        room = {"room": e.env, "fold": e.env, "calibrated_with": calroom[e.env]}
        jobs.append((e.stem, e.stem, str(capture(e.stem)), 0.0, 61.0, room))
        tab = tfm_model.room_for(e.stem)
        meta.append({"row_id": e.stem, "stem": e.stem, "set": "unused", "room": e.env, "fold": e.env, "cond": e.cond,
                     "y": int(e.y), "obj": int(o), "stratum": st, "reference": False,
                     "tab_fold_ok": bool(tab is not None and tab["fold"] == e.env)})
    t = time.perf_counter()
    with Pool(NPROC) as pool:
        res = pool.map(one, jobs, chunksize=4)
    wall = time.perf_counter() - t
    M = pd.DataFrame(meta).set_index("row_id")
    err = [r for r in res if r["error"]]
    ok = {r["row_id"]: r for r in res if not r["error"]}
    for m in MODELS:
        M[f"p_{m}"] = [ok[i][f"p_{m}"] if i in ok else np.nan for i in M.index]
    M["swing_ab"] = [ok[i]["swing_ab"] if i in ok else np.nan for i in M.index]
    M["w"] = [ok[i]["w"] if i in ok else np.nan for i in M.index]

    # (1) channels vs the benchmark inputs
    pi = [ALL_CHANNELS.index(c) for c in TP.PHYS_CHANNELS]
    chan = {}
    for name, cols, getter in (("phys", TP.PHYS_CHANNELS, lambda i: S_ALL[R.RID[i], pi]),
                               ("base", list(CHANNELS), lambda i: R.S_RAW[R.RID[i]])):
        dif = {c: 0.0 for c in cols}; nanmis = {c: 0 for c in cols}
        for i in rows.row_id:
            if i not in ok:
                continue
            a, b = ok[i][name], getter(i)
            fa, fb = np.isfinite(a), np.isfinite(b)
            for j, c in enumerate(cols):
                nanmis[c] += int((fa[j] != fb[j]).sum())
                m = fa[j] & fb[j]
                if m.any():
                    d = np.abs(a[j][m] - b[j][m]) / np.maximum(np.abs(b[j][m]), 1e-12)
                    dif[c] = max(dif[c], float(d.max()))
        chan[name] = {"max_rel_diff": dif, "nan_mismatch": nanmis}

    # (2) probabilities vs the saved deployed predictions (5-seed mean)
    saved = {}
    for m in MODELS:
        v = TP.META["models"][m]["variant"]
        pr = pd.read_csv(HERE / "predictions" / f"{v}.csv.gz", dtype={"seed": str})
        pr = pr[(pr.form == "deployed") & pr.seed.isin([str(s) for s in range(5)])]
        saved[m] = pr.groupby("row_id").score.mean()
        M[f"saved_{m}"] = saved[m].reindex(M.index)
    rep = {}
    for m in MODELS:
        q = M[M[f"saved_{m}"].notna()]
        d = (q[f"p_{m}"] - q[f"saved_{m}"]).abs()
        rep[m] = {"n": int(len(q)), "max_abs_diff": float(d.max()),
                  "verdict_flips": int(((q[f"p_{m}"] > 0.5) != (q[f"saved_{m}"] > 0.5)).sum())}

    # (3) accuracy
    sc = scores(M.reset_index(), MODELS)
    summary = {"windows": int(len(M)), "computed": int(len(ok)), "errors": err[:20], "n_errors": len(err),
               "wall_s": round(wall, 1), "nproc": NPROC, "tab_room_lookup_ok": int(M.tab_fold_ok.sum()),
               "tab_room_lookup_bad": M.index[~M.tab_fold_ok].tolist()[:30],
               "channels": chan, "probabilities": rep, **sc}
    OUT.mkdir(parents=True, exist_ok=True)
    M.reset_index().to_csv(OUT / "verify_rows.csv.gz", index=False)
    (OUT / "verify_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({"windows": summary["windows"], "computed": summary["computed"], "n_errors": len(err), "wall_s": summary["wall_s"],
                      "tab_room_lookup_ok": summary["tab_room_lookup_ok"], "probabilities": rep,
                      "max_rel_diff_phys": max(chan["phys"]["max_rel_diff"].values()),
                      "max_rel_diff_base": max(chan["base"]["max_rel_diff"].values()),
                      "nan_mismatch": {k: sum(v["nan_mismatch"].values()) for k, v in chan.items()}}, indent=1), flush=True)


def scores(M: pd.DataFrame, models: list[str]) -> dict:
    """Accuracy tables, LOEO strata and per-condition error counts of the per-window probabilities in M."""

    def acc(q: pd.DataFrame, m: str) -> float | None:
        return None if len(q) == 0 else round(100.0 * float(((q[f"p_{m}"] > 0.5).astype(int) == q.y).mean()), 1)

    def ba(q: pd.DataFrame, m: str) -> float | None:
        ok = (q[f"p_{m}"] > 0.5).astype(int) == q.y
        if not (q.y == 1).any() or not (q.y == 0).any():
            return None
        return round(50.0 * (float(ok[q.y == 1].mean()) + float(ok[q.y == 0].mean())), 1)

    def table(q: pd.DataFrame, m: str) -> dict:
        t = {}
        for k, g in q.groupby("room"):
            t[k] = {"n": int(len(g)), "all": acc(g, m), "kpi1": acc(g[g.obj == 0], m), "kpi2": acc(g[g.obj == 1], m)}
        t["all"] = {"n": int(len(q)), "all": acc(q, m), "kpi1": acc(q[q.obj == 0], m), "kpi2": acc(q[q.obj == 1], m),
                    "ba": ba(q, m)}
        return t

    sc = M[~M.reference.astype(bool)]
    sets = {"loeo": sc[sc.set == "loeo"], "scnC": sc[sc.set == "scnC"], "env2_later": sc[sc.set == "env2_later"],
            "unused": sc[sc.set == "unused"], "rooms_all": sc[sc.set != "scnC"]}
    cond = []
    for (st, room, c), g in sc.groupby(["set", "room", "cond"]):
        cond.append({"set": st, "room": room, "cond": c, "n": int(len(g)), "y": int(g.y.iloc[0]), "obj": int(g.obj.iloc[0]),
                     **{f"wrong_{m}": int(((g[f"p_{m}"] > 0.5).astype(int) != g.y).sum()) for m in models}})
    return {"accuracy": {m: {k: table(q, m) for k, q in sets.items()} for m in models},
            "strata_loeo": {m: {s: acc(sets["loeo"][sets["loeo"].stratum == s], m)
                                for s in ("empty_no_object", "object_only", "person_no_object", "person_plus_object")} for m in models},
            "conditions": cond, "sets_n": {k: int(len(q)) for k, q in sets.items()}, "references_n": int(M.reference.astype(bool).sum())}


def rescore(out: Path) -> None:
    summ = json.loads((out / "verify_summary.json").read_text())
    summ.update(scores(pd.read_csv(out / "verify_rows.csv.gz"), list(summ["probabilities"])))
    (out / "verify_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print("rescored", out / "verify_summary.json")


if __name__ == "__main__":
    if RESCORE:
        rescore(Path(sys.argv[2]))
    else:
        main()
