"""Board time of the fast numpy path per test-time setting (cyphy session, 2026-11-16).

  python3 bench_fast_board.py CODE BUNDLE CFG OUT.json cap1.bin [...]
Per capture: index + prep once, then judge_fast for each (crops, bin offsets, seeds) setting. The full setting is
checked against the reference arithmetic's torch logits where the bundle has them.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402

import e2e_np as N  # noqa: E402
from backend import mtk  # noqa: E402

B = Path(sys.argv[2]); C = sys.argv[3]; OUT = Path(sys.argv[4])
meta = json.loads((B / "meta.json").read_text())
cd = meta["cfgs"][C]
fn = [N.FastNet(N.Net(cd, dict(np.load(B / f"{C}__s{s}.npz")))) for s in range(5)]
stems = [q["stem"] for q in meta["rows"]]
st = cd["stride"]
import os  # noqa: E402
SET = [tuple(int(v) for v in k.split('x')) for k in os.environ['E2E_SET'].split(',')] if os.environ.get('E2E_SET') else \
    [(3, st, 5), (1, st, 5), (3, 4, 5), (1, 4, 5), (3, st, 1), (1, st, 1), (1, 4, 1), (1, 2, 1), (1, 1, 1)]
pc = time.perf_counter
rows = []
for cap in sys.argv[5:]:
    p = Path(cap); r = {q["stem"]: q for q in meta["rows"]}[p.stem]
    a = pc(); idx = mtk.MTKIndex(p); arr = N.prep(str(p), r["t0"], r["t1"], idx); t_prep = pc() - a
    sc = np.asarray(meta["room_scale"][r["room"]], np.float32)
    res = {}
    for nc, no, ns in SET:
        tm = {}
        a = pc()
        v = N.judge_fast(arr, sc, cd, fn[:ns], [0, 1, 2] if nc == 3 else [1], [int(round(k * st / no)) for k in range(no)], timings=tm)
        res[f"{nc}x{no}x{ns}"] = {"forwards": nc * no * ns, "judge": pc() - a, "total": t_prep + pc() - a, "logit": v, **tm}
    full = res.get(f"3x{st}x5", next(iter(res.values())))
    chk = ""
    if stems.index(p.stem) < len(next(iter(meta["torch"][C].values()))):
        tl = float(np.mean([v[stems.index(p.stem)] for v in meta["torch"][C].values()]))
        chk = f" |full - torch| {abs(full['logit'] - tl):.1e}"
    rows.append({"stem": p.stem, "prep": t_prep, "settings": res})
    print(f"{p.stem} prep {t_prep:.2f}{chk} | " + " | ".join(f"{k} ({v['forwards']}) {v['total']:.2f} s" for k, v in res.items()), flush=True)
    print("   full split: " + ", ".join(f"{k} {full[k]:.2f}" for k in ("crops", "spectra (shared)", "nets")), flush=True)
OUT.write_text(json.dumps(rows, indent=1))
