"""Scenario C check for the current e2e candidate (sbo.B.lfhftl.st16.att.noaug, 3x2x3), cyphy session 2026-11-17.
Diagnosis only. Per scnC condition: accuracy, median logit; ranking (AUC) vs threshold (best single threshold) for e2e and
13F / 15F; the env1+env2 model (protocol) vs the env1+env2+env3 model (diagnostic: one more room in training); and the
signal level relative to the room's reference noise (what the calibration divides by) for scnC vs the training rooms."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import e2e as E  # noqa: E402
import fair_compare as FC  # noqa: E402

M = FC.M
cfg = "sbo.B.lfhftl.st16.att.noaug"
ens = FC.e2e_ens(cfg, 3, 2, range(3))
R = pd.read_csv(HERE / "bench_rows.csv.gz", dtype={"row_id": str}).set_index("row_id")
m = ((M.set == "scnC") & ~M.reference).to_numpy()
y = M.y.to_numpy()[m]; cond = M.cond.to_numpy()[m]; rid = M.row_id.to_numpy()[m]


def auc(s, y):
    from itertools import product
    p, n = s[y == 1], s[y == 0]
    return 100 * np.mean([(a > b) + 0.5 * (a == b) for a, b in product(p, n)])


def best_thr(s, y):
    c = np.unique(s); acc = [(100 * ((s > t) == (y == 1)).mean(), t) for t in np.r_[c - 1e-9, c[-1] + 1]]
    return max(acc)


S = {"e2e (env1+env2)": ens[("env1", "env2")][m], "e2e (env1+env2+env3, 진단용)": ens[("env1", "env2", "env3")][m],
     "13F": np.log(R.p_seq13pw.reindex(rid).clip(1e-9, 1 - 1e-9) / (1 - R.p_seq13pw.reindex(rid).clip(1e-9, 1 - 1e-9))).to_numpy(),
     "15F": np.log(R.p_seq15pwu.reindex(rid).clip(1e-9, 1 - 1e-9) / (1 - R.p_seq15pwu.reindex(rid).clip(1e-9, 1 - 1e-9))).to_numpy()}
print("== scnC: accuracy at 0, AUC, best single threshold (accuracy, threshold)")
for k, s in S.items():
    a, t = best_thr(s, y)
    print(f"{k:30s} acc {100 * ((s > 0) == (y == 1)).mean():5.1f}  AUC {auc(s, y):5.1f}  best-threshold acc {a:5.1f} at {t:+.2f}")
print("\n== per condition: % correct (median logit)")
for c in pd.unique(cond):
    k = cond == c
    print(f"{c:24s} y={int(y[k][0])} " + " | ".join(f"{n} {100 * ((s[k] > 0) == (y[k] == 1)).mean():5.1f} ({np.median(s[k]):+.2f})" for n, s in S.items()))

# signal level / room reference noise
X = np.load(E.OUT / "x_40.npy", mmap_mode="r")
def levels(i):
    z = X[i].astype(np.float32); c = z[..., 0] + 1j * z[..., 1]
    return np.median(np.abs(np.diff(c, axis=0)), axis=0) / np.sqrt(2.0), np.median(np.abs(c[40:] - c[:-40]), axis=0) / np.sqrt(2.0)
ref = {r: np.median(np.stack([levels(i)[0] for i in ii]), axis=0) for r, ii in E.ref_rows(M).items()}
rows = []
for i, r in M[(M.part == "bench") & M.set.isin(["loeo", "scnC"]) & ~M.reference & (M.obj == 0)].iterrows():
    n0, n1 = levels(i)
    rows.append(dict(grp=("scnC" if r.set == "scnC" else r.room), y=r.y, move=float(np.median(n1 / ref[r.room]))))
D = pd.DataFrame(rows)
print("\n== 1-s change / room reference noise, no-machine windows, median [IQR]")
print(D.groupby(["grp", "y"]).move.agg(lambda s: f"{s.median():.2f} [{s.quantile(.25):.2f}-{s.quantile(.75):.2f}]").to_string())
