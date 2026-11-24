"""Task runner for the end-to-end study (cyphy session, 2026-11-15).

  run.py dev   CFG[,CFG...] [WORKERS] [SEEDS]   room sets of 1-2 rooms (inner folds only; outer data never fitted)
  run.py outer CFG[,CFG...] [WORKERS] [SEEDS]   the four 3-room sets (outer LOEO models)
  run.py final CFG[,CFG...] [WORKERS] [SEEDS]   the four 3-room sets + env1+env2 (scenario C), seeds 0..SEEDS-1
  run.py b5    CFG[,CFG...] [WORKERS] [SEEDS]   scenario B rescored with its own reference (B minutes 0-4, leave-same-
                                                 minute-out; user 2026-11-16): the two models that score B (env1+env2
                                                 and env1+env2+env3-1) refitted, B logits only -> out/pred_b5/
Every (config, room set, seed) model predicts every window (logit, 3 crops x 4 bin offsets) into
out/pred/<cfg>/<roomset>__s<seed>.npy; existing files are skipped (resumable). Configs: configs.py.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PRED = HERE / "out" / "pred"
PRED_B5 = HERE / "out" / "pred_b5"
B5_REFS = [f"B:{w}" for w in range(5)]

_W: dict = {}


def _init():
    import torch
    import e2e as E
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    X = np.load(E.OUT / "x_40.npy", mmap_mode="r")
    _W["Xnp"] = X
    _W["X"] = torch.from_numpy(np.ascontiguousarray(X)).cuda()
    if os.environ.get("E2E_GATED") == "1":               # the gated stream (x_40g.npy) too: run with <= 2 workers
        Xg = np.load(E.OUT / "x_40g.npy", mmap_mode="r")
        _W["Xgnp"] = Xg
        _W["Xg"] = torch.from_numpy(np.ascontiguousarray(Xg)).cuda()
    _W["M"] = E.load_meta()
    _W["scales"] = {}


def _scale(mode: str, stream: str = ""):
    import torch
    import e2e as E
    k = (mode, stream)
    if k not in _W["scales"]:
        X = _W["Xgnp"] if stream == "g" else _W["Xnp"]
        _W["scales"][k] = torch.from_numpy(E.calibration_scales(X, _W["M"], mode, tag=stream)).cuda()
    return _W["scales"][k]


def batcher(cfg, M):
    import e2e as E
    codes = M.room.map({r: i for i, r in enumerate(E.ROOMS)}).to_numpy()
    refs = {i: v for i, v in enumerate(E.ref_rows(M).values())}
    if cfg.inp == "g":
        return E.Batcher(_W["Xg"], _scale(cfg.cal, "g"), cfg, room_code=codes, refs=refs)
    extra = [(_W["Xg"], _scale(cfg.cal, "g"))] if cfg.inp == "rg" else None
    return E.Batcher(_W["X"], _scale(cfg.cal), cfg, room_code=codes, refs=refs, extra=extra)


def _b5_batcher(cfg, M):
    import torch
    import e2e as E
    bat = batcher(cfg, M)
    k = ("b5", cfg.cal)
    if k not in _W["scales"]:
        S = _scale(cfg.cal).clone()
        for j, v in E.own_reference_scales(_W["Xnp"], M, cfg.cal, "B", B5_REFS).items():
            S[j] = torch.from_numpy(v).cuda()
        _W["scales"][k] = S
    bat.S = _W["scales"][k]
    return bat


def task(t):
    cfg_name, rs, seed = t[:3]
    b5 = len(t) > 3
    import torch
    import e2e as E
    import configs as C
    out = (PRED_B5 if b5 else PRED) / cfg_name / f"{E.rs_name(rs)}__s{seed}.npy"
    if out.exists():
        return (t, "skip", 0.0)
    try:
        a = time.time()
        cfg = C.get(cfg_name)
        M = _W["M"]
        bat = _b5_batcher(cfg, M) if b5 else batcher(cfg, M)
        idx = E.train_rows(M, rs, extra=cfg.xtrain)
        y = M.y.to_numpy()[idx].astype(np.float32)
        w = E.room_balance_weights(M, idx)
        rooms = M.room.iloc[idx].map({r: i for i, r in enumerate(rs)}).to_numpy()
        init = None
        if cfg.ssl:
            import ssl_pre as S
            init = S.load(cfg, rs)
        model = E.fit(cfg, bat, idx, y, w, seed, rooms=rooms, init=init)
        p = E.predict(model, bat, cfg, np.flatnonzero((M.set == "B").to_numpy()) if b5 else np.arange(len(M)))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, p)
        del model
        torch.cuda.empty_cache()
        return (t, "ok", time.time() - a)
    except Exception:  # noqa: BLE001
        return (t, traceback.format_exc(), 0.0)


def main():
    import multiprocessing as mp
    import e2e as E
    phase, cfgs = sys.argv[1], sys.argv[2].split(",")
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    seeds = range(int(sys.argv[4]) if len(sys.argv) > 4 else 5)
    if phase == "final":                                  # outer LOEO models + the scenario C model (env1+env2)
        sets = [rs for rs in E.room_sets() if len(rs) == 3] + [("env1", "env2")]
    elif phase == "b5":                                   # the models that score B: folds scnC and env3
        sets = [("env1", "env2"), ("env1", "env2", "env3-1")]
        assert all(__import__("configs").get(c).cal in ("persc", "scalar") for c in cfgs), "b5 only for fixed5 configs"
        tasks = [(c, rs, s, "b5") for c in cfgs for s in seeds for rs in sets
                 if not (PRED_B5 / c / f"{E.rs_name(rs)}__s{s}.npy").exists()]
    else:
        sets = [rs for rs in E.room_sets() if (len(rs) < 3) == (phase == "dev")]
    if phase != "b5":
        tasks = [(c, rs, s) for c in cfgs for s in seeds for rs in sets
                 if not (PRED / c / f"{E.rs_name(rs)}__s{s}.npy").exists()]
    if any(__import__("configs").get(c).ssl for c in cfgs):
        import ssl_pre as S
        S.run_all([c for c in cfgs if __import__("configs").get(c).ssl], sets, workers)
    print(f"{phase}: {len(tasks)} tasks, {workers} workers", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init) as pool:
        for i, (t, st, dt) in enumerate(pool.imap_unordered(task, tasks)):
            if st not in ("ok", "skip"):
                print("ERROR", t, st, flush=True)
            else:
                print(f"{i + 1}/{len(tasks)} {t[0]} {E.rs_name(t[1])} s{t[2]} {dt:.0f} s  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
