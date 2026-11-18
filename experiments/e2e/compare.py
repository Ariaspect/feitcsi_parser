"""Paired comparison of end-to-end picks with the feature models (cyphy session, 2026-11-15).

Feature models (same windows, same folds, same fixed5 references where calibrated):
  seq17      base 15+2ch Transformer, fixed5 (bench_rows p_seq17, the Physics tab's base)
  seq13pw    physics 13-ch Transformer, fixed5 (bench_rows p_seq13pw)
  seq15pwu   physics 15-ch Transformer, fixed5 (bench_rows p_seq15pwu)
  mlp50A     50F MLP deployed, no calibration (challengers predictions/mlp.f50.none.csv.gz, form deployed, seed ens)
  svm50B     SVM-RBF 50F fixed5 (challengers predictions/svm_rbf.f50.fixed5.csv.gz, form tuned)
Bootstrap: sessions resampled within each room (B = 2000, seed 20261115), LOEO pooled accuracy and BA differences.

  compare.py OUT.json NAME=path_to_rows.csv.gz [NAME=...]   rows files from evaluate.py export (row_id, ctx, logit)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CH = next((p for p in (Path("/home/lg_csi/ml_lab/data/challengers/repo/experiments/challengers"), HERE.parent / "challengers")
           if (p / "predictions").exists()), HERE.parent / "challengers")


def feature_models() -> pd.DataFrame:
    """Binary verdicts (1 = person) of the feature models per row_id."""
    R = pd.read_csv(HERE / "bench_rows.csv.gz", dtype={"row_id": str, "stem": str})
    F = pd.DataFrame({"row_id": R.row_id})
    for c in ("seq17", "seq13pw", "seq15pwu"):
        F[c] = (R[f"p_{c}"] > 0.5).astype(float)
    for name, f, form in (("mlp50A", "mlp.f50.none.csv.gz", "deployed"), ("svm50B", "svm_rbf.f50.fixed5.csv.gz", "tuned")):
        p = CH / "predictions" / f
        if not p.exists():
            continue
        P = pd.read_csv(p, dtype={"row_id": str, "seed": str})
        P = P[(P.form == form) & (P.seed.isin(["ens", "det"])) & (P.eval_set == "loeo")].drop_duplicates("row_id")
        F[name] = F.row_id.map(P.set_index("row_id").pred.astype(float))
    return F


def sessions() -> pd.Series:
    rows = pd.read_csv(CH / "manifests" / "frozen" / "rows.csv", dtype={"row_id": str})
    return rows.set_index("row_id").session


def acc_ba(y, p, o):
    ok = (p > 0.5) == (y == 1)
    return (100 * ok.mean(), 50 * (ok[y == 1].mean() + ok[y == 0].mean()), 100 * ok[o == 0].mean(), 100 * ok[o == 1].mean())


def boot(df: pd.DataFrame, a: str, b: str, B: int = 2000, seed: int = 20261115) -> dict:
    rng = np.random.default_rng(seed)
    groups = {r: [g.index.to_numpy() for _, g in d.groupby("session")] for r, d in df.groupby("room")}
    D = []
    ya, pa, pb, oa = df.y.to_numpy(), df[a].to_numpy(), df[b].to_numpy(), df.obj.to_numpy()
    pos = {k: i for i, k in enumerate(df.index)}
    for _ in range(B):
        ii = np.concatenate([np.concatenate([g[k] for k in rng.integers(0, len(g), len(g))]) for g in groups.values()])
        jj = np.array([pos[i] for i in ii])
        D.append(np.subtract(acc_ba(ya[jj], pa[jj], oa[jj]), acc_ba(ya[jj], pb[jj], oa[jj])))
    D = np.array(D)
    A0 = acc_ba(ya, pa, oa); B0 = acc_ba(ya, pb, oa)
    out = {}
    for k, nm in enumerate(("acc", "ba", "k1", "k2")):
        v = D[:, k]
        out[nm] = {"a": round(A0[k], 1), "b": round(B0[k], 1), "diff": round(A0[k] - B0[k], 1),
                   "ci": [round(float(np.nanpercentile(v, 2.5)), 1), round(float(np.nanpercentile(v, 97.5)), 1)],
                   "p_le0": round(float(np.nanmean(v <= 0)), 3)}
    return out


def main():
    out = Path(sys.argv[1])
    F = feature_models()
    S = sessions()
    res = {}
    for arg in sys.argv[2:]:
        name, f = arg.split("=", 1)
        D = pd.read_csv(f, dtype={"row_id": str})
        lo = D[(D.set == "loeo")].set_index("row_id")
        df = lo[["room", "y", "obj", "logit"]].rename(columns={"logit": name}).join(F.set_index("row_id"))
        df[name] = (df[name] > 0).astype(float)
        df["session"] = S.reindex(df.index).to_numpy()
        df = df.dropna(subset=["session"])
        res[name] = {other: boot(df.reset_index(drop=True), name, other) for other in F.columns[1:] if df[other].notna().all()}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
