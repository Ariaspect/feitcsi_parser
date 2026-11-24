"""Training-only auxiliary targets for the end-to-end study: the per-second physics series of the challengers harness
(the 17 + the 21 new channels + lgrow, logged as in its all_log), aligned to out/meta_40.csv rows. Never an input;
used only by configs with aux > 0 as an extra regression target during training (cyphy session, 2026-11-15).
Run in the challengers repo on lg. Out: <e2e>/out/aux.npz (A: (N, C, 61) float32, NaN where unknown; channels)."""
import os, sys
from pathlib import Path
CH = Path("/home/lg_csi/ml_lab/data/challengers/repo/experiments/challengers")
sys.path.insert(0, str(CH / "bench")); os.chdir(CH)
import numpy as np, pandas as pd
from transforms import all_series, all_log
from common import ALL_CHANNELS
E2E = Path(__file__).resolve().parent
z = np.load("inputs/seq17.npz"); n = np.load("inputs/newch.npz")
S = all_log(all_series(z["X_raw"], n["X_new"], n["channels"].tolist()))
rid = {r: i for i, r in enumerate(z["row_id"].tolist())}
M = pd.read_csv(E2E / "out" / "meta_40.csv", dtype={"row_id": str})
A = np.full((len(M), S.shape[1], 61), np.nan, np.float32)
for i, r in enumerate(M.row_id):
    if r in rid:
        A[i] = S[rid[r]]
np.savez(E2E / "out" / "aux.npz", A=A, channels=np.array(ALL_CHANNELS))
print(A.shape, np.isfinite(A).mean(), (np.isfinite(A).any(axis=(1, 2))).sum(), ALL_CHANNELS)
