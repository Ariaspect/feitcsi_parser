"""Why the e2e models miss people in scenario C (cyphy session, 2026-11-16). Diagnosis only, no model change.

1. Signal level relative to the room's five fixed install references (what calibration B divides by), per room / set /
   label: noise floor (median |1-frame step|) and 1-s change (median |z(t+1 s) - z(t)|), median over bins.
2. Logits of the env1+env2 models (scenario C context) and of the 3-room models on the same room's LOEO rows (env3-1)
   versus scnC, for the picked configs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2e as E  # noqa: E402
import evaluate as V  # noqa: E402

M = E.load_meta()
X = np.load(E.OUT / "x_40.npy", mmap_mode="r")


def levels(i: int) -> tuple[np.ndarray, np.ndarray]:
    z = X[i].astype(np.float32)
    c = z[..., 0] + 1j * z[..., 1]
    n0 = np.median(np.abs(np.diff(c, axis=0)), axis=0) / np.sqrt(2.0)
    n1 = np.median(np.abs(c[E.RATE:] - c[:-E.RATE]), axis=0) / np.sqrt(2.0)
    return n0, n1


refs = E.ref_rows(M)
ref = {r: np.median(np.stack([levels(i)[0] for i in ii]), axis=0) for r, ii in refs.items()}

sel = M[(M.part == "bench") & M.set.isin(["loeo", "scnC"]) & ~M.reference]
rows = []
for i, r in sel.iterrows():
    n0, n1 = levels(i)
    rr = ref[r.room]
    rows.append(dict(set=r.set, room=r.room, y=r.y, obj=r.obj, cond=r.cond,
                     floor=float(np.median(n0 / rr)), move1s=float(np.median(n1 / rr)),
                     move1s_self=float(np.median(n1 / n0))))
D = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("signal / room reference noise (median over bins), median [IQR] over windows")
g = D.groupby(["set", "room", "y", "obj"])
f = lambda s: f"{s.median():.2f} [{s.quantile(.25):.2f}-{s.quantile(.75):.2f}]"  # noqa: E731
print(g.agg(n=("floor", "size"), floor=("floor", f), move1s=("move1s", f), move1s_self=("move1s_self", f)).to_string())
print(D[D.set == "scnC"].groupby(["cond"]).agg(n=("floor", "size"), floor=("floor", f), move1s=("move1s", f)).to_string())

print("\nlogits, same room: env3-1 LOEO rows vs scnC rows")
for cfg in ["scnet.S.lfhf", "scnet.B.st16.all4c.att", "sbo.B.all4.st16.att", "sbo.B.all4c"]:
    for rs in [("env1", "env2"), ("env1", "env2", "env3")]:
        p = V.ens(cfg, rs)
        for s in ["loeo", "scnC"]:
            m = ((M.part == "bench") & (M.set == s) & (M.room == "env3-1") & ~M.reference).to_numpy()
            y = M.y.to_numpy()[m]; L = p[m]
            acc = 100 * ((L > 0) == (y == 1)).mean()
            print(f"{cfg:24s} {E.rs_name(rs):16s} {s:5s} acc {acc:5.1f}  empty: med logit {np.median(L[y == 0]):6.2f} "
                  f"called person {100 * (L[y == 0] > 0).mean():5.1f} %  occupied: med {np.median(L[y == 1]):6.2f} "
                  f"found {100 * (L[y == 1] > 0).mean():5.1f} %")
