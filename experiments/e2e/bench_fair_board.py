"""Board time of one 1-min judgment, e2e candidate vs 13F / 15F, timed the same way (cyphy session, 2026-11-17).

  python3 bench_fair_board.py CODE BUNDLE REPS OUT.json cap1.bin [...]
Each model's time is everything after the capture file exists: its own frame index (MTKIndex), its own input
(13F / 15F: tfm_phys_fast.series of the channels it reads; e2e: the raw ratio prep), calibration and all its forwards
(13F / 15F: 5 seeds; e2e: 3 crops x 2 bin offsets x 3 seeds). The file is read once before the repetitions (warm
page cache for every model) and the model order rotates per repetition. Median over the repetitions per capture.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402

import e2e_np as N  # noqa: E402
from backend import mtk, tfm_model, tfm_phys as TP, tfm_phys_fast as TF  # noqa: E402

B = Path(sys.argv[2]); REP = int(sys.argv[3]); OUT = Path(sys.argv[4])
meta = json.loads((B / "meta.json").read_text())
rows = {q["stem"]: q for q in meta["rows"]}
E2E = {"e2e (no tf) 3x2x3": "sbo.B.lfhftl.st16.att", "e2e all4 3x2x3": "sbo.B.all4.st16.att"}
NETS = {k: [N.FastNet(N.Net(meta["cfgs"][c], dict(np.load(B / f"{c}__s{s}.npz")))) for s in range(3)] for k, c in E2E.items()}
CH = {m: TP.META["models"][m]["channels"] for m in TP.NEW_MODELS}
NAME = {"seq13pw": "13F", "seq15pwu": "15F"}
pc = time.perf_counter


def run_phys(p, t0, t1, room, m):
    a = pc(); idx = mtk.MTKIndex(p); b = pc()
    s = TF.series(p, t0, t1, idx, channels=CH[m]); c = pc()
    _, v = TP.judge(TP.matrix(s["S"], m), m, room); d = pc()
    return d - a, {"index": b - a, "input": c - b, "calibrate + forwards": d - c}, float(v["p"])


def run_e2e(p, t0, t1, room, k):
    cd = meta["cfgs"][E2E[k]]; st = cd["stride"]
    sc = np.asarray(meta["room_scale"][room], np.float32)
    a = pc(); idx = mtk.MTKIndex(p); b = pc()
    arr = N.prep(str(p), t0, t1, idx); c = pc()
    v = N.judge_fast(arr, sc, cd, NETS[k], [0, 1, 2], [int(round(j * st / 2)) for j in range(2)]); d = pc()
    return d - a, {"index": b - a, "input": c - b, "calibrate + forwards": d - c}, float(v)


JOBS = [(NAME[m], run_phys, m) for m in TP.NEW_MODELS] + [(k, run_e2e, k) for k in E2E]
out = []
for cap in sys.argv[5:]:
    p = Path(cap); r = rows[p.stem]; t0, t1 = r["t0"], r["t1"]
    p.read_bytes()
    room_p = tfm_model.room_for(p.stem)
    reps = []
    for rep in range(REP):
        t = {}
        order = JOBS[rep % len(JOBS):] + JOBS[:rep % len(JOBS)]
        for name, fn, arg in order:
            tot, parts, v = fn(p, t0, t1, room_p if fn is run_phys else r["room"], arg)
            t[name] = tot
            for k, x in parts.items():
                t[f"{name} | {k}"] = x
            t[f"{name} | out"] = v
        reps.append(t)
    med = {k: float(np.median([x[k] for x in reps])) for k in reps[0]}
    out.append({"stem": p.stem, "room": r["room"], "t": med})
    print(f"{p.stem} {r['room']:6s} | " + " | ".join(f"{n} {med[n]:.2f} s" for n, _, _ in JOBS), flush=True)
info = {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
        "affinity": sorted(os.sched_getaffinity(0)), "reps": REP}
summ = {n: float(np.median([o["t"][n] for o in out])) for n, _, _ in JOBS}
for n, _, _ in JOBS:
    for k in ("index", "input", "calibrate + forwards"):
        summ[f"{n} | {k}"] = float(np.median([o["t"][f"{n} | {k}"] for o in out]))
OUT.write_text(json.dumps({"info": info, "median": summ, "rows": out}, indent=1))
print("median over captures:", json.dumps(summ, indent=1), flush=True)
