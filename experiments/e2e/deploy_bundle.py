"""Deployment bundle of the e2e model and its check through the deployed arithmetic (cyphy session, 2026-11-18; user:
"앞으로 e2e 모델은 현재 후보로 배포해").

  deploy_bundle.py OUTDIR [N_PROC]          lg
Model: sbo.B.lfhftl.st16.att.noaug, seeds 0-2, test inputs 3 crops x bin offsets 0 / 8 (judge_fast).
Bundle (numpy, for the board):
  OUTDIR/<fold>__s<k>.npz   the trained state dicts; fold = the room the model never saw (env1, env2, env3, env3-1:
                            the three other rooms) or scnC (env1 + env2, the model scenarios A / B / C are scored by)
  OUTDIR/meta.json          cfg, folds -> rooms, GLOB (the dataset constant the training divided every room's noise
                            level by: a deployment's scale = median over its install empties of noise_level / GLOB),
                            the training rooms' scales
Check: every scored window (the fair-compare rows) through e2e_np.judge_fast in float32 on its stored x_40 row (=
prep_fast, bit-identical, verify_fast.py) with the training scale, against the evaluated logits (torch, CUDA bf16
autocast, tta_study.per_input_logits): |diff|, verdict flips, and the accuracy table recomputed from the numpy logits.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, "/home/lg_csi/feitcsi_parser-ml")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

CFG = "sbo.B.lfhftl.st16.att.noaug"
SEEDS = (0, 1, 2)
OFFS = (0, 8)
FOLDS = {"env1": ("env2", "env3", "env3-1"), "env2": ("env1", "env3", "env3-1"), "env3": ("env1", "env2", "env3-1"),
         "env3-1": ("env1", "env2", "env3"), "scnC": ("env1", "env2")}
_G: dict = {}


def _init(out, xpath):
    import e2e_np as N
    meta = json.loads((Path(out) / "meta.json").read_text())
    cd = meta["cfg"]
    _G["cd"] = cd
    _G["nets"] = {f: [N.FastNet(N.Net(cd, dict(np.load(Path(out) / f"{f}__s{s}.npz")))) for s in SEEDS] for f in FOLDS}
    _G["X"] = np.load(xpath, mmap_mode="r")
    _G["S"] = np.load(Path(out) / "train_scales.npy")


def _job(i):
    import e2e_np as N
    x = np.asarray(_G["X"][i])
    return i, {f: N.judge_fast(x, _G["S"][i], _G["cd"], _G["nets"][f], [0, 1, 2], list(OFFS)) for f in FOLDS}


def main():
    import torch
    from dataclasses import asdict
    import configs as C
    import e2e as E
    import run as R
    import fair_compare as FC
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    cfg = C.get(CFG)
    cd = {k: v for k, v in asdict(cfg).items() if k in ("arch", "stride", "specbr", "sbatt", "sbstat", "width", "depth",
                                                      "kernel", "crop", "comp", "binpool", "cal")}
    M = E.load_meta()
    X = np.load(E.OUT / "x_40.npy", mmap_mode="r")
    # the training scales: fixed5 per room, scenario B its own first five minutes (run._b5_batcher)
    S = E.calibration_scales(X, M, cfg.cal)
    for j, v in E.own_reference_scales(X, M, cfg.cal, "B", R.B5_REFS).items():
        S[j] = v
    np.save(out / "train_scales.npy", S)
    refs = E.ref_rows(M)
    lev = {r: np.median(np.stack([E.noise_level(X[i]) for i in ii]), axis=0) for r, ii in refs.items()}
    glob = float(np.median(np.concatenate([v[None] for v in lev.values()])))
    room_scale = {r: (v / glob).astype(np.float32).tolist() for r, v in lev.items()}
    for r in E.ROOMS:      # the deployment formula reproduces the training scale of every room
        i = int(np.flatnonzero((M.room == r).to_numpy() & ~M.set.isin(["B"]).to_numpy())[0])
        assert np.array_equal(np.asarray(room_scale[r], np.float32), S[i]), r
    for f, rooms in FOLDS.items():
        for s in SEEDS:
            sd = torch.load(E.OUT / "weights" / CFG / f"{E.rs_name(rooms)}__s{s}.pt", map_location="cpu")
            np.savez(out / f"{f}__s{s}.npz", **{k: v.numpy() for k, v in sd.items() if v.dtype.is_floating_point})
    meta = {"model": CFG, "cfg": cd, "seeds": list(SEEDS), "offsets": list(OFFS), "starts_idx": [0, 1, 2],
            "folds": {f: list(r) for f, r in FOLDS.items()}, "glob": glob, "room_scale": room_scale,
            "ref_minutes": 5, "note": "deployment scale = median over install empties of noise_level(prep minute) / glob"}
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"bundle {out}: {len(FOLDS)} folds x {len(SEEDS)} seeds, glob {glob:.6g}", flush=True)

    # every window, every fold, numpy float32
    rows = [i for i in range(len(M)) if str(M.err.iloc[i]) in ("nan", "", "None")]
    with get_context("fork").Pool(nproc, initializer=_init, initargs=(str(out), str(E.OUT / "x_40.npy"))) as p:
        res = dict(p.imap_unordered(_job, rows, chunksize=4))
    npl = {f: np.full(len(M), np.nan) for f in FOLDS}
    for i, v in res.items():
        for f, l in v.items():
            npl[f][i] = l
    ens_np = {FOLDS[f]: npl[f] for f in FOLDS}
    ens_t = FC.e2e_ens(CFG, 3, 2, seeds=SEEDS)
    Dn = FC.e2e_rows(ens_np)
    Dt = FC.e2e_rows({rs: ens_t[rs] for rs in ens_np})
    d = (Dn.logit - Dt.logit.reindex(Dn.index)).abs()
    flips = ((Dn.logit > 0) != (Dt.logit.reindex(Dn.index) > 0))
    print(f"scored windows {len(Dn)}: |numpy fp32 - torch bf16| median {d.median():.2e}, max {d.max():.2e}; "
          f"verdict flips {int(flips.sum())} {list(Dn.index[flips])[:10]}", flush=True)
    S_ = FC.subsets(Dn)
    tab = {}
    for name, m in S_.items():
        q = Dn[m]
        tab[name] = {"numpy": FC.kpi(((q.logit > 0) == (q.y == 1)).to_numpy(), q.obj.to_numpy()),
                     "torch": FC.kpi(((Dt.logit.reindex(q.index) > 0) == (q.y == 1)).to_numpy(), q.obj.to_numpy())}
        print(f"{name:12s} numpy {tab[name]['numpy']}  torch {tab[name]['torch']}", flush=True)
    (out / "check.json").write_text(json.dumps({"n": len(Dn), "max_abs_diff": float(d.max()),
                                                "median_abs_diff": float(d.median()), "flips": list(Dn.index[flips]),
                                                "table": tab}, indent=1))
    Dn[["logit"]].join(Dt[["logit"]].rename(columns={"logit": "logit_torch"})).to_csv(out / "logits.csv.gz")


if __name__ == "__main__":
    main()
