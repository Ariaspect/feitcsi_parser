"""Inner-fold (L2O) error anatomy: wrong-window % per (held-out room, condition) over the 12 pairs, e2e configs
next to the feature models of l2o_feat.py (same pairs, same rows). Dev data only.
  inner_cond.py CFG[,CFG...] [FEATVARIANT,...]"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e as E  # noqa: E402
import evaluate as V  # noqa: E402

M = V.M
cfgs = sys.argv[1].split(",")
feats = sys.argv[2].split(",") if len(sys.argv) > 2 else []
rows = []
for rs, h in V.inner_pairs():
    mask = V.scored(h)
    for c in cfgs:
        p = V.ens(c, rs)
        q = M[mask].assign(model=c, train="+".join(rs), wrong=((p[mask] > 0) != (M.y.to_numpy()[mask] == 1)).astype(int))
        rows.append(q[["row_id", "room", "cond", "y", "obj", "model", "train", "wrong"]])
D = pd.concat(rows)
if feats:
    Fd = pd.read_csv(Path(__file__).resolve().parent / "out" / "l2o_feat.csv", dtype={"row_id": str})
    Fd = Fd[Fd.variant.isin(feats)]
    Fd = Fd.assign(model=Fd.variant, wrong=(Fd.dec != Fd.y).astype(int), room=Fd.held)
    Fd = Fd.merge(M[["row_id", "cond"]], on="row_id", how="left")
    D = pd.concat([D, Fd[["row_id", "room", "cond", "y", "obj", "model", "train", "wrong"]]])
T = D.pivot_table(index=["room", "y", "obj", "cond"], columns="model", values="wrong", aggfunc="mean") * 100
n = D[D.model == cfgs[0]].groupby(["room", "y", "obj", "cond"]).size().rename("n")
T = T.join(n)
T = T[cfgs + feats + ["n"]]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(T.round(0).to_string())
