"""Scores of the end-to-end study (cyphy session, 2026-11-15).

  evaluate.py dev   [CFG,...]   inner folds only: every 2-room model on each of its two held-out rooms (12 pairs),
                                 the 1-room models of the scenario C fold, and each outer fold's inner score
  evaluate.py outer [CFG,...]   outer LOEO / scnC / env2_later / unused / fresh for configs with 3-room models, plus
                                 the nested pick (per outer fold the config with the best inner score)
Ensemble = mean logit over the seeds present. Person if logit > 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2e as E  # noqa: E402

PRED = HERE / "out" / "pred"
M = E.load_meta()
N = len(M)


def ens(cfg: str, rs: tuple[str, ...], seeds=None) -> np.ndarray | None:
    if cfg.startswith("ens:"):                       # ensemble of configs: mean of their seed-ensemble logits
        parts = [ens(c, rs, seeds) for c in cfg[4:].split("+")]
        return None if any(p is None for p in parts) else np.mean(parts, axis=0)
    fs = sorted((PRED / cfg).glob(f"{E.rs_name(rs)}__s*.npy"))
    if seeds is not None:
        fs = [f for f in fs if int(f.stem.split("__s")[1]) in seeds]
    if not fs:
        return None
    return np.mean([np.load(f) for f in fs], axis=0)


def n_seeds(cfg: str, rs) -> int:
    if cfg.startswith("ens:"):
        return min(n_seeds(c, rs) for c in cfg[4:].split("+"))
    return len(list((PRED / cfg).glob(f"{E.rs_name(rs)}__s*.npy")))


def scored(room: str, sets=("loeo",)) -> np.ndarray:
    return ((M.part == "bench") & M.set.isin(sets) & (M.room == room) & ~M.reference).to_numpy()


def metrics(mask: np.ndarray, logit: np.ndarray) -> dict:
    y = M.y.to_numpy()[mask]; o = M.obj.to_numpy()[mask]; p = logit[mask] > 0
    ok = p == (y == 1)
    r = lambda m: float(100 * ok[m].mean()) if m.any() else float("nan")  # noqa: E731
    return {"n": int(mask.sum()), "acc": r(np.ones_like(ok)), "k1": r(o == 0), "k2": r(o == 1),
            "ba": 0.5 * (r(y == 1) + r(y == 0)) if (y == 1).any() and (y == 0).any() else float("nan"),
            "rec": r(y == 1), "spec": r(y == 0)}


def inner_pairs():
    """(training set, held-out room) of the LOEO inner folds: all 2-room sets x their two complement rooms."""
    out = []
    for rs in E.room_sets():
        if len(rs) == 2:
            for h in E.ROOMS:
                if h not in rs:
                    out.append((rs, h))
    return out


def inner_score(cfg: str, fold: str) -> float:
    """Outer fold's inner score: mean BA over the held-out rooms h != fold of the models trained on the other two."""
    if fold == "scnC":
        v = []
        for rs, h in ((("env1",), "env2"), (("env2",), "env1")):
            p = ens(cfg, rs)
            if p is None:
                return float("nan")
            v.append(metrics(scored(h), p)["ba"])
        return float(np.mean(v))
    v = []
    for h in E.ROOMS:
        if h == fold:
            continue
        rs = tuple(r for r in E.ROOMS if r not in (fold, h))
        p = ens(cfg, rs)
        if p is None:
            return float("nan")
        v.append(metrics(scored(h), p)["ba"])
    return float(np.mean(v))


def dev_table(cfgs: list[str]) -> pd.DataFrame:
    rows = []
    for c in cfgs:
        d = {"cfg": c}
        per_room = {h: [] for h in E.ROOMS}
        k1, k2, acc = [], [], []
        for rs, h in inner_pairs():
            p = ens(c, rs)
            if p is None:
                continue
            m = metrics(scored(h), p)
            per_room[h].append(m["ba"]); k1.append(m["k1"]); k2.append(m["k2"]); acc.append(m["acc"])
        if not acc:
            continue
        d["L2O_BA"] = np.mean([x for v in per_room.values() for x in v])
        for h, v in per_room.items():
            d[f"BA_{h}"] = np.mean(v) if v else np.nan
        d["K1"] = np.nanmean(k1); d["K2"] = np.nanmean(k2)
        tb = []
        for rs in E.room_sets():
            if len(rs) == 2 and (p := ens(c, rs)) is not None:
                tm = np.zeros(N, bool); tm[E.train_rows(M, rs)] = True
                tb.append(metrics(tm, p)["ba"])
        d["trainBA"] = np.mean(tb) if tb else np.nan
        for f in list(E.ROOMS) + ["scnC"]:
            d[f"in_{f}"] = inner_score(c, f)
        d["seeds"] = n_seeds(c, ("env1", "env2"))
        rows.append(d)
    return pd.DataFrame(rows).set_index("cfg")


# ------------------------------------------------------------------------------------------------------- outer
def outer_logit(cfg: str) -> dict[str, np.ndarray]:
    """Per scoring context the logit vector: 'F' for LOEO fold F (3-room model without F), 'scnC' (env1+env2)."""
    out = {}
    for f in E.ROOMS:
        p = ens(cfg, tuple(r for r in E.ROOMS if r != f))
        if p is not None:
            out[f] = p
    p = ens(cfg, ("env1", "env2"))
    if p is not None:
        out["scnC"] = p
    return out


def assemble(L: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per scored window with the logit of the model that scores it (benchmark rule), fresh sets twice for
    A / B (fold scnC and fold env3)."""
    rows = []
    for i, r in M.iterrows():
        if r.part == "unlab" or r.reference:
            continue
        if r.part == "bench":
            ctx = "scnC" if r.set == "scnC" else r.room
            if ctx in L:
                rows.append(dict(i=i, ctx=ctx, logit=float(L[ctx][i])))
        else:
            ctxs = ["scnC", "env3"] if r.set in ("A", "B") else ["env2"]
            for ctx in ctxs:
                if ctx in L:
                    rows.append(dict(i=i, ctx=ctx, logit=float(L[ctx][i])))
    D = pd.DataFrame(rows)
    return D.join(M[["row_id", "part", "set", "room", "cond", "y", "obj", "stratum"]], on="i")


def summary(D: pd.DataFrame, col: str = "logit") -> dict:
    out = {}
    def m(q):
        y = q.y.to_numpy(); o = q.obj.to_numpy(); ok = (q[col].to_numpy() > 0) == (y == 1)
        r = lambda s: round(float(100 * ok[s].mean()), 1) if s.any() else None  # noqa: E731
        ba = None if not ((y == 1).any() and (y == 0).any()) else round(0.5 * (100 * ok[y == 1].mean() + 100 * ok[y == 0].mean()), 1)
        return {"n": int(len(q)), "acc": r(np.ones_like(ok)), "k1": r(o == 0), "k2": r(o == 1), "ba": ba}
    lo = D[(D.part == "bench") & (D.set == "loeo")]
    out["LOEO"] = m(lo)
    for f in E.ROOMS:
        out[f] = m(lo[lo.room == f])
    out["scnC"] = m(D[(D.set == "scnC")])
    out["env2_later"] = m(D[D.set == "env2_later"])
    out["unused"] = m(D[D.set == "unused"])
    for s, ctx in (("A", "scnC"), ("A", "env3"), ("B", "scnC"), ("B", "env3"), ("1109", "env2")):
        q = D[(D.set == s) & (D.ctx == ctx)]
        if len(q):
            out[f"{s}|{ctx}"] = m(q)
    return out


def export(D: pd.DataFrame, path: Path) -> None:
    """Scored rows (row_id, ctx, set, room, cond, y, obj, stratum, logit) for compare.py and the report."""
    D.drop(columns=["i"]).to_csv(path, index=False)


def cond_table(Ds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wrong windows per (set, room, condition) for several models (same rows): LOEO, scnC, env2_later, unused, fresh
    (A / B under fold env3)."""
    out = None
    for name, D in Ds.items():
        q = D[~((D.set.isin(["A", "B"])) & (D.ctx == "scnC"))].copy()
        q["wrong"] = ((q.logit > 0) != (q.y == 1)).astype(int)
        g = q.groupby(["set", "room", "cond", "y", "obj"]).agg(n=("wrong", "size"), **{name: ("wrong", "sum")})
        out = g if out is None else out.join(g[[name]])
    return out


def rate_check(D: pd.DataFrame) -> dict:
    """Shortcut check: verdicts by native frame rate (41.7 Hz vs the others) inside each room and label, LOEO rows.
    env2's off-rate windows are mostly occupied; a model that learned the rate would call env1's off-rate empties
    person more often than its 41.7 Hz empties."""
    q = D[(D.part == "bench") & (D.set == "loeo")].join(M[["fs"]], on="i")
    q = q.assign(rate=np.where((q.fs - 41.67).abs() < 0.2, "41.7", "other"), call=(q.logit > 0).astype(int))
    out = {}
    for (room, y, r), g in q.groupby(["room", "y", "rate"]):
        out[f"{room}|y{y}|{r}"] = {"n": int(len(g)), "called_person": round(float(100 * g.call.mean()), 1)}
    return out


def nested(cfgs: list[str]) -> tuple[dict, dict]:
    pick = {}
    for f in list(E.ROOMS) + ["scnC"]:
        sc = {c: inner_score(c, f) for c in cfgs}
        sc = {c: v for c, v in sc.items() if np.isfinite(v)}
        pick[f] = max(sc, key=sc.get)
    L = {}
    for f, c in pick.items():
        Lc = outer_logit(c)
        if f in Lc:
            L[f] = Lc[f]
    # fresh A/B under fold env3 use the env3 pick, under scnC the scnC pick: assemble() does it by context
    return pick, summary(assemble(L))


def main():
    mode = sys.argv[1]
    cfgs = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else sorted(p.name for p in PRED.iterdir() if p.is_dir())
    cfgs += [a for a in sys.argv[3:] if a.startswith("ens:")]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    if mode == "dev":
        T = dev_table(cfgs)
        print(T.sort_values("L2O_BA", ascending=False).round(1).to_string())
        return
    res = {}
    for c in cfgs:
        L = outer_logit(c)
        if len(L) == 5:
            res[c] = summary(assemble(L))
    rows = []
    for c, s in res.items():
        rows.append({"cfg": c, **{f"{k}_{m}": v[m] for k, v in s.items() for m in ("acc", "k1", "k2", "ba") if k in
                                  ("LOEO", "env1", "env2", "env3", "env3-1", "scnC")}})
    if rows:
        print(pd.DataFrame(rows).set_index("cfg").to_string())
    pick, s = nested(cfgs)
    print("nested pick", pick)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
