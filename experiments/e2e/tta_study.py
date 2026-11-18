"""Board-time vs accuracy of the deployable e2e model (cyphy session, 2026-11-16; user: "정확도 좀 떨어져도 되니까 시간을 줄여").

  tta_study.py fit  CFG[,CFG...] [WORKERS] [SEEDS]   every room set (inner 1-2 rooms + outer 3 rooms) x seed: fit, save
                                                     the weights (out/weights/<cfg>/<rs>__s<k>.pt) and the logit of every
                                                     test-time input separately, (N, 3 starts, stride offsets) ->
                                                     out/tta/<cfg>/<rs>__s<k>.npy. B rows use B's own reference (minutes
                                                     0-4, leave-same-minute-out); training is unchanged by that.
  tta_study.py eval CFG[,CFG...] OUT.json            per reduced setting (crops x bin offsets x seeds): inner L2O BA,
                                                     each outer fold's inner score, and the deployment table (Env 1..3-1,
                                                     Total, Scenario A / B / C with env1+env2 models, Env 2 + 110 extra).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TTA = HERE / "out" / "tta"
WGT = HERE / "out" / "weights"


def per_input_logits(model, bat, cfg, idx, chunk=64):
    import torch
    import e2e as E
    starts = E.test_starts(cfg.crop)
    out = np.zeros((len(idx), len(starts), cfg.stride), np.float32)
    dev = bat.X.device
    with torch.no_grad():
        for a in range(0, len(idx), chunk):
            j = torch.as_tensor(idx[a:a + chunk], device=dev)
            for si, st in enumerate(starts):
                for o in range(cfg.stride):
                    x = bat.get(j, torch.full_like(j, st), torch.full_like(j, o), cfg.crop, train=False)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out[a:a + chunk, si, o] = model(x).float().cpu().numpy()
    return out


def task(t):
    import torch
    import configs as C
    import e2e as E
    import run as R
    cfg_name, rs, seed = t
    out = TTA / cfg_name / f"{E.rs_name(rs)}__s{seed}.npy"
    if out.exists():
        return (t, "skip", 0.0)
    try:
        a = time.time()
        cfg = C.get(cfg_name)
        M = R._W["M"]
        bat = R._b5_batcher(cfg, M)
        idx = E.train_rows(M, rs, extra=cfg.xtrain)
        y = M.y.to_numpy()[idx].astype(np.float32)
        w = E.room_balance_weights(M, idx)
        model = E.fit(cfg, bat, idx, y, w, seed)
        L = per_input_logits(model, bat, cfg, np.arange(len(M)))
        out.parent.mkdir(parents=True, exist_ok=True)
        (WGT / cfg_name).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), WGT / cfg_name / f"{E.rs_name(rs)}__s{seed}.pt")
        np.save(out, L)
        del model
        torch.cuda.empty_cache()
        return (t, "ok", time.time() - a)
    except Exception:  # noqa: BLE001
        return (t, traceback.format_exc(), 0.0)


def fit_all(cfgs, workers, seeds):
    import multiprocessing as mp
    import e2e as E
    import run as R
    tasks = [(c, rs, s) for c in cfgs for s in seeds for rs in E.room_sets()
             if not (TTA / c / f"{E.rs_name(rs)}__s{s}.npy").exists()]
    print(f"fit: {len(tasks)} tasks, {workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=R._init) as pool:
        for i, (t, st, dt) in enumerate(pool.imap_unordered(task, tasks)):
            if st not in ("ok", "skip"):
                print("ERROR", t, st, flush=True)
            else:
                print(f"{i + 1}/{len(tasks)} {t[0]} {E.rs_name(t[1])} s{t[2]} {dt:.0f} s  [{time.time() - t0:.0f} s]", flush=True)


# ------------------------------------------------------------------------------------------------------ eval
def settings(stride):
    out = []
    for nc in (3, 1):
        for no in sorted({stride, 8, 4, 2, 1} & set(range(1, stride + 1)), reverse=True):
            for ns in (5, 3, 1):
                out.append((nc, no, ns))
    return out


def pick(L, nc, no, stride):
    s = [0, 1, 2] if nc == 3 else [1]                  # one crop: the middle 40 s
    o = [int(round(k * stride / no)) for k in range(no)]
    return L[:, s][:, :, o].mean(axis=(1, 2))


def evaluate(cfgs, out_path):
    import e2e as E
    import evaluate as V
    res = {}
    for c in cfgs:
        cache = {}
        for rs in E.room_sets():
            cache[rs] = [np.load(TTA / c / f"{E.rs_name(rs)}__s{s}.npy") for s in range(5)]
        stride = cache[E.room_sets()[0]][0].shape[2]
        for nc, no, ns in settings(stride):
            ens = {rs: np.mean([pick(L, nc, no, stride) for L in v[:ns]], axis=0) for rs, v in cache.items()}
            # inner: 2-room models on their held-out rooms (12 pairs) and per-fold inner scores
            bas = []
            for rs, h in V.inner_pairs():
                bas.append(V.metrics(V.scored(h), ens[rs])["ba"])
            inner = {}
            for f in E.ROOMS:
                inner[f] = float(np.mean([V.metrics(V.scored(h), ens[tuple(r for r in E.ROOMS if r not in (f, h))])["ba"]
                                          for h in E.ROOMS if h != f]))
            inner["scnC"] = float(np.mean([V.metrics(V.scored("env2"), ens[("env1",)])["ba"],
                                           V.metrics(V.scored("env1"), ens[("env2",)])["ba"]]))
            Lo = {f: ens[tuple(r for r in E.ROOMS if r != f)] for f in E.ROOMS}
            Lo["scnC"] = ens[("env1", "env2")]
            D = V.assemble(Lo)
            s = V.summary(D)
            q = D[(D.part == "bench") & (D.room == "env2") & D.set.isin(["loeo", "env2_later", "unused"])]
            ok = (q.logit > 0) == (q.y == 1)
            s["env2+110"] = {"n": int(len(q)), "acc": round(100 * ok.mean(), 1), "k1": round(100 * ok[q.obj == 0].mean(), 1),
                             "k2": round(100 * ok[q.obj == 1].mean(), 1)}
            key = f"{c}|crops{nc}|off{no}|seeds{ns}"
            res[key] = {"cfg": c, "crops": nc, "offsets": no, "seeds": ns, "forwards": nc * no * ns,
                        "inner_L2O_BA": round(float(np.mean(bas)), 1), "inner": {k: round(v, 1) for k, v in inner.items()},
                        "outer": s}
            t = s
            print(f"{key:44s} fw {nc * no * ns:4d}  inner {np.mean(bas):5.1f} | Total {t['LOEO']['acc']:5.1f} "
                  f"({t['LOEO']['k1']}/{t['LOEO']['k2']}) env1 {t['env1']['acc']} env2 {t['env2']['acc']} env3 {t['env3']['acc']} "
                  f"env3-1 {t['env3-1']['acc']} | A {t['A|scnC']['acc']} B {t['B|scnC']['acc']} C {t['scnC']['acc']} "
                  f"env2+110 {s['env2+110']['acc']}", flush=True)
    Path(out_path).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit_all(sys.argv[2].split(","), int(sys.argv[3]) if len(sys.argv) > 3 else 4,
                range(int(sys.argv[4]) if len(sys.argv) > 4 else 5))
    else:
        evaluate(sys.argv[2].split(","), sys.argv[3])
