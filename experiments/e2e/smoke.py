"""Smoke test: every architecture, a short fit on env1+env2, timing, memory, parameter count, held-out BA."""
import sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e as E, configs as C, run as R

R._init()
M = R._W["M"]
ep = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for name in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["cnn1d.A", "scnet.A", "spec.A", "conv2d.A", "patchtf.A", "cnn1d.B"]):
    cfg = replace(C.get(name), epochs=ep)
    torch.cuda.reset_peak_memory_stats()
    bat = E.Batcher(R._W["X"], R._scale(cfg.cal), cfg)
    rs = ("env1", "env2")
    idx = E.train_rows(M, rs); y = M.y.to_numpy()[idx].astype(np.float32); w = E.room_balance_weights(M, idx)
    a = time.time()
    m = E.fit(cfg, bat, idx, y, w, 0, log=print)
    t_fit = time.time() - a
    a = time.time()
    p = E.predict(m, bat, cfg, np.arange(len(M)))
    t_pr = time.time() - a
    np_ = sum(q.numel() for q in m.parameters())
    ba = {}
    for h in ("env3", "env3-1"):
        mm = ((M.part == "bench") & (M.set == "loeo") & (M.room == h) & ~M.reference).to_numpy()
        yy = M.y.to_numpy()[mm]; pp = p[mm] > 0
        ba[h] = round(50 * ((pp[yy == 1] == 1).mean() + (pp[yy == 0] == 0).mean()), 1)
    tr = (p[idx] > 0) == (y > 0.5)
    print(f"{name}: params {np_/1e3:.0f}k fit {t_fit:.1f} s ({ep} ep, n {len(idx)}) predict {t_pr:.1f} s "
          f"peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB train acc {100*tr.mean():.1f} BA {ba}", flush=True)
