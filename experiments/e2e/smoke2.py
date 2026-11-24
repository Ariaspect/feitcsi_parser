"""Smoke test of the round-2 options (1 epoch each): aux, ref, q, ac, adv, ema, mae / simclr pretraining + init."""
import sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e as E, configs as C, run as R, ssl_pre as S
R._init()
M = R._W["M"]
codes = M.room.map({r: i for i, r in enumerate(E.ROOMS)}).to_numpy()
refs = {i: v for i, v in enumerate(E.ref_rows(M).values())}
rs = ("env1", "env2")
idx = E.train_rows(M, rs); y = M.y.to_numpy()[idx].astype(np.float32); w = E.room_balance_weights(M, idx)
rooms = M.room.iloc[idx].map({r: i for i, r in enumerate(rs)}).to_numpy()
for name in sys.argv[1].split(","):
    cfg = replace(C.get(name), epochs=1, ssl_epochs=1)
    bat = E.Batcher(R._W["X"], R._scale(cfg.cal), cfg, room_code=codes, refs=refs)
    a = time.time()
    init = None
    if cfg.ssl:
        sd = S.pretrain(cfg, bat, E.unlab_rows(M, rs), log=print)
        init = {k: v.cuda() for k, v in sd.items() if not k.startswith(("head.", "fc."))}
    m = E.fit(cfg, bat, idx, y, w, 0, rooms=rooms, init=init, log=print)
    p = E.predict(m, bat, cfg, np.arange(200))
    print(name, "ok", f"{time.time() - a:.1f} s", np.round(p[:3], 3), flush=True)
