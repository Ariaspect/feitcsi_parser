"""Time of one judgment of the Physics tab's models on one-minute windows, as the board would run it.

Per capture and repetition: the frame index (MTKIndex), then for each model on its own -- seq13pw, seq15pwu and the
17-channel base -- the series of exactly the channels it reads plus calibration and the 5-seed forward; and both physics
models from one series (seq15pwu's channels cover seq13pw's). Stage times of the seq15pwu series are kept too. The
median over the repetitions is stored per capture, with each model's P (to compare machines).

Usage (numpy only; CODE = a directory holding this repo's backend/ and data/):
  python3 bench_tfm_phys_board.py CODE WINDOWS.json REPS OUT.json cap1.bin [cap2.bin ...]
WINDOWS.json maps a capture stem to [t0, t1] (s from its first frame). Run it pinned to one core (taskset -c 3).
cyphy session 2026-11-12.
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

from backend import mtk, tfm_model, tfm_phys as TP, tfm_series  # noqa: E402

W = json.loads(Path(sys.argv[2]).read_text())
REP = int(sys.argv[3])
OUT = Path(sys.argv[4])
CH = {m: TP.META["models"][m]["channels"] for m in TP.NEW_MODELS}
pc = time.perf_counter
rows = []
for cap in sys.argv[5:]:
    p = Path(cap); stem = p.stem; t0, t1 = W[stem]
    room = tfm_model.room_for(stem)
    reps, P = [], {}
    for _ in range(REP):
        t: dict[str, float] = {}
        a = pc(); idx = mtk.MTKIndex(p); t["index"] = pc() - a
        for m in TP.NEW_MODELS:
            tm: dict[str, float] = {}
            a = pc(); s = TP.series(p, t0, t1, idx, channels=CH[m], timings=tm); t[f"series {m}"] = pc() - a
            a = pc(); _, v = TP.judge(TP.matrix(s["S"], m), m, room); t[f"calibrate + forward {m}"] = pc() - a
            P[m] = v["p"]
            t[f"total {m}"] = t["index"] + t[f"series {m}"] + t[f"calibrate + forward {m}"]
            if m == "seq15pwu":
                for k, x in tm.items():
                    t[f"  seq15pwu series: {k}"] = x
                w = s["w"]
        a = pc(); b = tfm_series.series(p, t0, t1, idx); t["series base 17"] = pc() - a
        a = pc(); _, v = TP.judge(b["X"], "seq17", room); t["calibrate + forward base"] = pc() - a
        P["seq17"] = v["p"]
        t["total base 17"] = t["index"] + t["series base 17"] + t["calibrate + forward base"]
        t["both physics models, one series"] = t["index"] + t["series seq15pwu"] + t["calibrate + forward seq13pw"] + t["calibrate + forward seq15pwu"]
        reps.append(t)
    med = {k: float(np.median([r[k] for r in reps])) for k in reps[0]}
    rows.append({"stem": stem, "room": room["room"] if room else None, "w": float(w), "p": P, "t": med})
    print(f"{stem} {room['room'] if room else '?':6s} w {w:.2f} | " + " | ".join(
        f"{m} {med[f'total {m}']:.2f} s P {P[m]:.6f}" for m in TP.NEW_MODELS) + f" | base {med['total base 17']:.2f} s P {P['seq17']:.6f}", flush=True)
info = {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(), "node": platform.node(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None, "reps": REP}
OUT.write_text(json.dumps({"info": info, "rows": rows}, indent=1))
print("wrote", OUT, json.dumps(info), flush=True)
