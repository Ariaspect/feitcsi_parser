"""Time of one judgment of the end-to-end models on one-minute windows, as the board would run it (numpy only).

  python3 bench_e2e_board.py CODE BUNDLE REPS OUT.json cap1.bin [cap2.bin ...]
CODE holds backend/ and e2e_np.py; BUNDLE = verify_np.py's output (random-weight nets, 5 seeds per config, room
scales, torch logits of the first rows). Per capture and repetition: frame index, prep (decode, ratio, 40 Hz, z), then
per config its calibration divisor and the 5-seed x test-time-crop forward. Run pinned to one core (taskset -c 3).
cyphy session 2026-11-16.
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
from backend import mtk  # noqa: E402

B = Path(sys.argv[2]); REP = int(sys.argv[3]); OUT = Path(sys.argv[4])
meta = json.loads((B / "meta.json").read_text())
rows = {r["stem"]: r for r in meta["rows"]}
nets = {c: [N.Net(cd, dict(np.load(B / f"{c}__s{s}.npz"))) for s in range(5)] for c, cd in meta["cfgs"].items()}
pc = time.perf_counter
res = []
for cap in sys.argv[5:]:
    p = Path(cap); r = rows[p.stem]
    reps, logit = [], {}
    for _ in range(REP):
        t: dict[str, float] = {}
        a = pc(); idx = mtk.MTKIndex(p); t["index"] = pc() - a
        a = pc(); arr = N.prep(str(p), r["t0"], r["t1"], idx, timings=t); t["prep"] = pc() - a
        for c, cd in meta["cfgs"].items():
            a = pc()
            if cd["cal"] == "self":
                sc = (N.noise_level(arr) / meta["self_const"]).astype(np.float32)
            else:
                sc = np.asarray(meta["room_scale"][r["room"]], np.float32)
            t[f"scale {c}"] = pc() - a
            a = pc(); logit[c] = N.judge(arr, sc, cd, nets[c]); t[f"forward {c}"] = pc() - a
            t[f"total {c}"] = t["index"] + t["prep"] + t[f"scale {c}"] + t[f"forward {c}"]
        reps.append(t)
    med = {k: float(np.median([x[k] for x in reps])) for k in reps[0]}
    ref = {c: float(np.mean([v[[q["stem"] for q in meta["rows"]].index(p.stem)] for v in meta["torch"][c].values()]))
           for c in meta["cfgs"] if [q["stem"] for q in meta["rows"]].index(p.stem) < len(next(iter(meta["torch"][c].values())))}
    res.append({"stem": p.stem, "room": r["room"], "t": med, "logit": logit, "torch_logit": ref})
    print(f"{p.stem} {r['room']:6s} index {med['index']:.2f} prep {med['prep']:.2f} | " + " | ".join(
        f"{c} {med[f'total {c}']:.2f} s" + (f" (|dlogit| {abs(logit[c] - ref[c]):.1e})" if c in ref else "")
        for c in meta["cfgs"]), flush=True)
info = {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(), "node": platform.node(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None, "reps": REP}
OUT.write_text(json.dumps({"info": info, "rows": res}, indent=1))
print("wrote", OUT, json.dumps(info), flush=True)
