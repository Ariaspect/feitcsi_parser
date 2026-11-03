"""Add (or refresh) the nine ``bs_*`` columns of ``data/ml_heads_train.csv`` from the captures.

Each row's window is the one its other columns were computed on: a one-minute capture whole
(``[0, last frame]`` from its first frame), a scenario C minute ``C<w>`` as ``[60w, 60w + 60)`` s of
the 2.6 h capture. ``backend.breath_components.unit_features`` computes the columns.

Run where the captures are (lg):
``.venv/bin/python scripts/add_bs_columns.py --captures /home/lg_csi/lg_csi_captures``
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import breath_components as BC  # noqa: E402

TRAIN = ROOT / "data" / "ml_heads_train.csv"
SCENARIO_C = "20261023/20261023_132301.bin"


def _one(a: tuple[str, str, str]) -> dict:
    key, ds, captures = a
    from backend.tiles import get_index
    root = Path(captures)
    if ds == "scnC":
        w = int(key[1:])
        return {"key": key, **BC.unit_features(root / SCENARIO_C, 60.0 * w, 60.0 * w + 60.0)}
    p = root / key[:8] / f"{key}.bin"
    if not p.exists():
        p = next(root.rglob(f"{key}.bin"))
    t = np.asarray(get_index(p).times, dtype=float)
    return {"key": key, **BC.unit_features(p, 0.0, float(t[-1] - t[0]) + 1e-6)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    D = pd.read_csv(TRAIN)
    with Pool(a.jobs, maxtasksperchild=20) as pool:
        B = pd.DataFrame(pool.map(_one, [(k, d, a.captures) for k, d in zip(D.key, D.ds)], chunksize=4))
    D = D.drop(columns=[c for c in BC.KEYS if c in D]).merge(B, on="key", how="left")
    print("blank rows:", int(D[list(BC.KEYS)].isna().all(axis=1).sum()))
    D.to_csv(TRAIN, index=False, float_format="%.7g")
    print("wrote", TRAIN.relative_to(ROOT), D.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
