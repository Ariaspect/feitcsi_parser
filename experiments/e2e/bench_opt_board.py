"""Where the e2e judgment's time goes, and what value-identical rearrangements save (cyphy session, 2026-11-17).

  python3 bench_opt_board.py CODE BUNDLE CFG REPS OUT.json cap1.bin [...]
Run WITHOUT taskset: the script pins itself to core 3 (CORE env) and, for the forked variants, pins each child to its
own core (CORES env, default 3,2,1). Per capture, median of REPS:
  current      index + prep() + judge_fast (3 crops x offsets 0, 8 x seeds 0-2): the 1.63-s path
  cols         index + prep_cols() (only the 30 bins the inputs read) + judge_fast
  cols+fork2/3 the same with the seeds' nets on 2 / 3 cores (os.fork, one pipe per child)
  prep_fast    11-18: prep_cols decoding only what the result depends on (checked equal to prep_cols)
  par1..4      judge_par: fast_index + prep_fast's plan, the used bins split over 1-4 processes, then the six test-time
               inputs split over them (PAR env, default cores 3,2,1,0); |logit - current| checked
  pool2..4     the same with workers forked once before the timing (N.Pool), as a running service would keep them
Stage times of prep / prep_cols / judge, |logit(cols) - logit(current)| and the forked sums are checked.
"""
from __future__ import annotations

import json
import os
import platform
import struct
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402

import e2e_np as N  # noqa: E402
from backend import mtk  # noqa: E402

B = Path(sys.argv[2]); C = sys.argv[3]; REP = int(sys.argv[4]); OUT = Path(sys.argv[5])
CORE = int(os.environ.get("CORE", "3"))
CORES = [int(c) for c in os.environ.get("CORES", "3,2,1").split(",")]
PAR = [int(c) for c in os.environ.get("PAR", "3,2,1,0").split(",")]
if hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, {CORE})
meta = json.loads((B / "meta.json").read_text())
cd = meta["cfgs"][C]
OFFS = [0, cd["stride"] // 2]
NETS = [N.FastNet(N.Net(cd, dict(np.load(B / f"{C}__s{s}.npz")))) for s in range(3)]
NEED = N.bins_used(cd, OFFS)
rows = {q["stem"]: q for q in meta["rows"]}
pc = time.perf_counter


def judge_split(arr, sc, ncore=1, tm=None):
    """judge_fast with the seeds' nets spread over ncore processes (ncore = 1: judge_fast itself)."""
    if ncore == 1:
        return N.judge_fast(arr, sc, cd, NETS, [0, 1, 2], OFFS, timings=tm)
    a = pc()
    X = N.crops_sel(arr, sc, cd, [0, 1, 2], OFFS)
    cache = {k: N.branch_input(k, X, cd.get("sbstat", 5)) for k in cd["specbr"].split("+") if k}
    b = pc()
    groups = [list(range(len(NETS)))[k::ncore] for k in range(ncore)]
    kids = []
    for k, g in enumerate(groups[1:], 1):
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {CORES[k]})
            v = [float(NETS[i](cache).sum()) for i in g]          # float32 sums, exact in a double
            os.write(w, struct.pack(f"{len(v)}d", *v))
            os._exit(0)
        os.close(w)
        kids.append((pid, r, g))
    tot = np.zeros(len(NETS))                                    # judge_fast's accumulation, seed order kept
    for i in groups[0]:
        tot[i] += NETS[i](cache).sum()
    for pid, r, g in kids:
        buf = b""
        while len(buf) < 8 * len(g):
            buf += os.read(r, 8 * len(g) - len(buf))
        for i, v in zip(g, struct.unpack(f"{len(g)}d", buf)):
            tot[i] += np.float32(v)
        os.close(r)
        os.waitpid(pid, 0)
    if tm is not None:
        tm.update({"crops + spectra": b - a, "nets (forked)": pc() - b})
    return float(tot.sum() / (len(X) * len(NETS)))


POOLS = {nc: N.Pool(PAR[:nc], NETS) for nc in (2, 3, 4)}       # forked once, as a running service would
out = []
for cap in sys.argv[6:]:
    p = Path(cap); r = rows[p.stem]; t0, t1 = r["t0"], r["t1"]
    sc = np.asarray(meta["room_scale"][r["room"]], np.float32)
    p.read_bytes()
    reps = []
    for _ in range(REP):
        t = {}
        a = pc(); idx = mtk.MTKIndex(p); t["index"] = pc() - a
        tp = {}; a = pc(); arr = N.prep(str(p), t0, t1, idx, timings=tp); t["prep"] = pc() - a
        t.update({f"prep | {k}": v for k, v in tp.items()})
        tj = {}; a = pc(); L0 = N.judge_fast(arr, sc, cd, NETS, [0, 1, 2], OFFS, timings=tj); t["judge"] = pc() - a
        t.update({f"judge | {k}": v for k, v in tj.items()})
        t["current"] = t["index"] + t["prep"] + t["judge"]
        a = pc(); idx2 = mtk.MTKIndex(p); ti = pc() - a
        tc = {}; a = pc(); arr2 = N.prep_cols(str(p), t0, t1, NEED, idx2, timings=tc); t["prep_cols"] = pc() - a
        t.update({f"prep_cols | {k}": v for k, v in tc.items()})
        a = pc(); L1 = judge_split(arr2, sc, 1); tj1 = pc() - a
        t["cols"] = ti + t["prep_cols"] + tj1
        t["cols bins identical"] = float(np.array_equal(arr[:, NEED], arr2[:, NEED]))
        t["|dlogit| cols"] = abs(L1 - L0)
        for nc in (2, 3):
            tf = {}; a = pc(); Lf = judge_split(arr2, sc, nc, tf); tjf = pc() - a
            t[f"cols+fork{nc}"] = ti + t["prep_cols"] + tjf
            t[f"|dlogit| fork{nc}"] = abs(Lf - L1)
        tq = {}; a = pc(); arr3 = N.prep_fast(str(p), t0, t1, NEED, mtk.MTKIndex(p), timings=tq); t["prep_fast"] = pc() - a
        t.update({f"prep_fast | {k}": v for k, v in tq.items()})
        t["fast input identical"] = float(np.array_equal(arr3.view(np.uint16), arr2.view(np.uint16)))
        for nc in (1, 2, 3, 4):
            tp = {}; a = pc(); Lp = N.judge_par(str(p), t0, t1, sc, cd, NETS, (0, 1, 2), OFFS, PAR[:nc], timings=tp)
            t[f"par{nc}"] = pc() - a
            t.update({f"par{nc} | {k}": v for k, v in tp.items()})
            t[f"|dlogit| par{nc}"] = abs(Lp - L0)
        for nc, pool in POOLS.items():
            tp = {}; a = pc(); Lp = N.judge_par(str(p), t0, t1, sc, cd, NETS, (0, 1, 2), OFFS, timings=tp, pool=pool)
            t[f"pool{nc}"] = pc() - a
            t.update({f"pool{nc} | {k}": v for k, v in tp.items()})
            t[f"|dlogit| pool{nc}"] = abs(Lp - L0)
        os.sched_setaffinity(0, {CORE}) if hasattr(os, "sched_setaffinity") else None
        t["logit"] = L0
        reps.append(t)
    med = {k: float(np.median([x[k] for x in reps])) for k in reps[0]}
    out.append({"stem": p.stem, "room": r["room"], "t": med})
    print(f"{p.stem} {r['room']:6s} | current {med['current']:.2f} | cols {med['cols']:.2f} | fork2 {med['cols+fork2']:.2f} "
          f"| fork3 {med['cols+fork3']:.2f} s | prep {med['prep']:.2f} -> {med['prep_cols']:.2f} judge {med['judge']:.2f} "
          f"| same bins {med['cols bins identical']:.0f} dlogit {med['|dlogit| cols']:.1e} / {med['|dlogit| fork2']:.1e} / "
          f"{med['|dlogit| fork3']:.1e} | par1-4 {med['par1']:.2f} / {med['par2']:.2f} / {med['par3']:.2f} / {med['par4']:.2f} s "
          f"prep_fast {med['prep_fast']:.2f} same {med['fast input identical']:.0f} dlogit "
          + " ".join(f"{med[f'|dlogit| par{k}']:.1e}" for k in (1, 2, 3, 4))
          + f" | pool2-4 {med['pool2']:.2f} / {med['pool3']:.2f} / {med['pool4']:.2f} s dlogit "
          + " ".join(f"{med[f'|dlogit| pool{k}']:.1e}" for k in (2, 3, 4)), flush=True)
for pool in POOLS.values():
    pool.close()
summ = {k: float(np.median([o["t"][k] for o in out])) for k in out[0]["t"]}
info = {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(), "core": CORE,
        "cores": CORES, "ncpu": os.cpu_count(), "reps": REP}
OUT.write_text(json.dumps({"info": info, "median": summ, "rows": out}, indent=1))
print("median over captures:", json.dumps(summ, indent=1), flush=True)
