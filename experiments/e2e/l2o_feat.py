"""Feature models under the end-to-end study's inner protocol (cyphy session, 2026-11-15).

Each 2-room training set predicts each of its two held-out rooms (12 pairs, LOEO rows minus the fixed references),
with the challengers harness's own inputs, calibration (fixed5: training rooms by their own references, the held-out
room by its references with the pooled-scale w 0.5 rule) and deployed configurations, seeds 0-4, seed ensemble.
Gives the end-to-end inner scores a like-for-like feature reference. Runs in the challengers repo on lg:
  cd .../challengers/repo/experiments/challengers && PY e2e/l2o_feat.py VARIANT[,VARIANT] NPROC
Out: <e2e>/out/l2o_feat.csv (variant, train rooms, held-out room, row_id, score, decision).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
CH = Path("/home/lg_csi/ml_lab/data/challengers/repo/experiments/challengers")
sys.path.insert(0, str(CH / "bench"))
os.chdir(CH)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run as RB  # noqa: E402
from models import CHANNEL_SETS, parse, target_of  # noqa: E402
from transforms import CAL_BY_INPUT, SeqAdapter, channel_set_cal, scalar_regime, seq_regime  # noqa: E402
FS = RB.FS  # feature lists of the harness

E2E = Path(__file__).resolve().parent
ROOMS = ("env1", "env2", "env3", "env3-1")


def rows_of(rooms) -> np.ndarray:
    R = RB.ROWS
    refs = set(np.concatenate(list(RB.REFS.values())).tolist())
    m = (R.set == "loeo") & R.room.isin(rooms)
    return np.array([i for i in np.flatnonzero(m.to_numpy()) if i not in refs])


def feat(inp: str, reg: str, tr: np.ndarray, te: np.ndarray, trr: list[str], test_room: str):
    drop = None
    if inp in ("f41", "f50"):
        cols = [FS["f50"].index(c) for c in FS[inp]]
        fit = np.zeros(len(RB.ROWS), bool); fit[tr] = True
        Z, _ = scalar_regime(RB.XS[:, cols], RB.ROOM, fit, reg, RB.REFS)
        return Z[tr], Z[te], drop
    if inp in CHANNEL_SETS:
        chans, dropped, p = CHANNEL_SETS[inp]
        SL, AV = RB.all_channels()
        sel = [RB.ALL_CHANNELS.index(c) for c in chans]
        L, av, cal = SL[:, sel], AV[:, sel], channel_set_cal(chans)
    else:
        L, av, cal = RB.S_LOG, RB.AVAIL, CAL_BY_INPUT[inp]
    Ztr = seq_regime(L, RB.ROOM, tr, test_room, trr, reg, RB.REFS, cal=cal)
    Zte = seq_regime(L, RB.ROOM, te, test_room, trr, reg, RB.REFS, cal=cal)
    ad = SeqAdapter(Ztr)
    return ad(Ztr, av[tr]), ad(Zte, av[te]), drop


def job(a):
    variant, trr, h = a
    fam, inp, reg = parse(variant)
    tr, te = rows_of(trr), rows_of([h])
    Xtr, Xte, drop = feat(inp, reg, tr, te, list(trr), h)
    cid = fam.config_id(fam.deployed)
    ri = fam.run_of(cid)
    S = []
    for s in (RB.SEEDS if fam.stochastic else [None]):
        sc, _, _, _ = RB.fit_run(fam, ri, [cid], Xtr, tr, Xte, s, target_of(variant), drop)
        S.append(sc[cid])
    if fam.ensemble == "vote":
        dec = (np.mean([fam.decide(x) for x in S], 0) > 0.5).astype(int); score = np.mean(S, 0)
    else:
        score = np.mean(S, 0); dec = fam.decide(score)
    return pd.DataFrame({"variant": variant, "train": "+".join(trr), "held": h, "row_id": RB.ROWS.row_id.to_numpy()[te],
                         "y": RB.Y[te], "obj": RB.OBJ[te], "score": score, "dec": dec})


def main():
    vs = sys.argv[1].split(","); nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    pairs = [(tuple(r for r in ROOMS if r not in (a, b)), h) for i, a in enumerate(ROOMS) for b in ROOMS[i + 1:]
             for h in (a, b)]
    jobs = [(v, trr, h) for v in vs for trr, h in pairs]
    with Pool(nproc) as pl:
        D = pd.concat(pl.map(job, jobs, chunksize=1))
    out = E2E / "out" / "l2o_feat.csv"
    if out.exists():
        old = pd.read_csv(out, dtype={"row_id": str})
        D = pd.concat([old[~old.variant.isin(vs)], D])
    D.to_csv(out, index=False)
    for v, d in D.groupby("variant"):
        ba = []
        for (t, h), q in d.groupby(["train", "held"]):
            ok = q.dec == q.y
            ba.append((h, 50 * (ok[q.y == 1].mean() + ok[q.y == 0].mean())))
        b = pd.DataFrame(ba, columns=["h", "ba"])
        print(v, "L2O_BA", round(b.ba.mean(), 1), b.groupby("h").ba.mean().round(1).to_dict(), flush=True)


if __name__ == "__main__":
    main()
