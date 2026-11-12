"""Weights for the Physics tab (backend/tfm_phys.py) from the challenger benchmark's deployed checkpoints.

The benchmark (exp/challengers, experiments/challengers) trained tfm.seq13pw.fixed5, tfm.seq15pwu.fixed5 and the base
tfm.seq17.fixed5 in their deployed form (d 16, 25 epochs, 5 seeds) for each held-out fold. This script rebuilds each
fold's inputs exactly as bench/run.py does (fixed5 calibration, the fold's training-row standardisation), loads the
saved checkpoints, and asserts that they reproduce the saved per-seed probabilities on every scored window of the fold
and that the numpy forward of backend/tfm_model.py matches torch. It then writes, into OUT_DIR:

* ``tfm_phys.npz``  per model: each room's five reference minutes (logged series, (5, C, 61)); per fold the
  standardisation mu / sd, the training rooms' pooled reference MAD and the 5 seeds' weights.
* ``tfm_phys.json`` channels, log / calibrated channels, folds, reference keys, checkpoint hashes and check rows (raw
  series and expected P of a few windows, for tests/test_tfm_phys.py).

Usage on lg (the benchmark mirror, its venv):
  cd /home/lg_csi/ml_lab/data/challengers/repo/experiments/challengers
  ../../../venv/bin/python <this file> OUT_DIR
cyphy session 2026-11-12.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
HERE = Path.cwd()
sys.path.insert(0, str(HERE / "bench"))
OUT = Path(sys.argv[1])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import run as R  # noqa: E402  (the benchmark's frozen rows, splits, references and series)
from common import ALL_CAL_CHANNELS, ALL_LOG_CHANNELS, CAL_CHANNELS, CHANNELS, FOLDS, LOG_CHANNELS, NEW_CHANNELS, SEEDS  # noqa: E402
from models import CHANNEL_SETS, _net  # noqa: E402
from transforms import CALI, SeqAdapter, all_series, channel_set_cal, ref_stats, seq_regime  # noqa: E402

torch.set_num_threads(1)
MODELS = {"seq13pw": "tfm.seq13pw.fixed5", "seq15pwu": "tfm.seq15pwu.fixed5", "seq17": "tfm.seq17.fixed5"}
_Z = np.load(HERE / "inputs" / "newch.npz")
assert _Z["row_id"].tolist() == R.ROWS.row_id.tolist()
S_ALL = all_series(R.S_RAW, _Z["X_new"], _Z["channels"].tolist())        # raw, every channel
from common import ALL_CHANNELS  # noqa: E402


def np_forward(P: dict, x: np.ndarray) -> float:
    z = x.T @ P["inp_w"].T + P["inp_b"] + P["pos"][: x.shape[1]]
    z = np.vstack([P["cls"], z])
    d = z.shape[1]; hh = 2; dh = d // hh
    qkv = z @ P["in_w"].T + P["in_b"]; q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
    heads = []
    for i in range(hh):
        sl = slice(i * dh, (i + 1) * dh)
        a = q[:, sl] @ k[:, sl].T / np.sqrt(dh); a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        heads.append(a @ v[:, sl])
    ln = lambda u, g, b: (u - u.mean(1, keepdims=True)) / np.sqrt(u.var(1, keepdims=True) + 1e-5) * g + b
    z = ln(z + np.hstack(heads) @ P["out_w"].T + P["out_b"], P["n1_g"], P["n1_b"])
    z = ln(z + np.maximum(z @ P["l1_w"].T + P["l1_b"], 0) @ P["l2_w"].T + P["l2_b"], P["n2_g"], P["n2_b"])
    return float(z[0] @ P["h_w"][0] + P["h_b"][0])


def params(sd: dict) -> dict:
    sd = {k: v.detach().numpy().astype(np.float64) for k, v in sd.items()}
    L = "enc.layers.0."
    return {"inp_w": sd["inp.weight"], "inp_b": sd["inp.bias"], "pos": sd["pos"], "cls": sd["cls"][0],
            "in_w": sd[L + "self_attn.in_proj_weight"], "in_b": sd[L + "self_attn.in_proj_bias"],
            "out_w": sd[L + "self_attn.out_proj.weight"], "out_b": sd[L + "self_attn.out_proj.bias"],
            "l1_w": sd[L + "linear1.weight"], "l1_b": sd[L + "linear1.bias"], "l2_w": sd[L + "linear2.weight"], "l2_b": sd[L + "linear2.bias"],
            "n1_g": sd[L + "norm1.weight"], "n1_b": sd[L + "norm1.bias"], "n2_g": sd[L + "norm2.weight"], "n2_b": sd[L + "norm2.bias"],
            "h_w": sd["h.weight"], "h_b": sd["h.bias"]}


arrays: dict[str, np.ndarray] = {}
meta: dict = {"order": list(MODELS), "models": {}, "threshold": 0.5, "seeds": len(SEEDS), "epoch": 25,
              "ref_keys": {r: R.ROWS.key.to_numpy()[ix].tolist() for r, ix in R.REFS.items()},
              "benchmark_commit": (HERE.parents[1] / "MIRROR_COMMIT").read_text().strip(), "fitted": str(date.today())}
probs: dict = {}
KEEP: dict = {}                                       # (model, fold) -> (L, av, cal, test_room, trr, adapter, seed params)
for m, variant in MODELS.items():
    if m == "seq17":
        chans, L, av, cal = list(CHANNELS), R.S_LOG, R.AVAIL, CALI
        logc, calc = list(LOG_CHANNELS), list(CAL_CHANNELS)
    else:
        chans = list(CHANNEL_SETS[m][0])
        assert not CHANNEL_SETS[m][1], "no channel dropout in the shipped sets"
        SL, AV = R.all_channels()
        sel = [ALL_CHANNELS.index(c) for c in chans]
        L, av, cal = SL[:, sel], AV[:, sel], channel_set_cal(chans)
        logc, calc = [c for c in chans if c in ALL_LOG_CHANNELS], [c for c in chans if c in ALL_CAL_CHANNELS]
        assert [chans[i] for i in cal] == calc
    for r, ix in R.REFS.items():
        arrays[f"{m}_ref_{r}"] = L[ix].astype(np.float64)
    pr = pd.read_csv(HERE / "predictions" / f"{variant}.csv.gz", dtype={"seed": str})
    pr = pr[pr.form == "deployed"]
    folds, hashes = {}, {}
    for fold in FOLDS:
        sp = R.SPLITS[fold]
        trr, test_room = sp["train_rooms"], sp["calibration_room_of_test"]
        tr, te = R.split_rows(fold, "outer")
        Ztr = seq_regime(L, R.ROOM, tr, test_room, trr, "fixed5", R.REFS, cal=cal)
        Zte = seq_regime(L, R.ROOM, te, test_room, trr, "fixed5", R.REFS, cal=cal)
        ad = SeqAdapter(Ztr)
        Xte = ad(Zte, av[te])
        pool = np.exp(np.median(np.log(np.stack([ref_stats(L[R.REFS[r]])[1] for r in trr])), 0))
        arrays[f"{m}_{fold}_mu"], arrays[f"{m}_{fold}_sd"], arrays[f"{m}_{fold}_pool"] = ad.mu, ad.sd, pool
        ids = R.ROWS.row_id.to_numpy()[te]
        KEEP[(m, fold)] = (L, av, cal, test_room, trr, ad, [])
        for s in SEEDS:
            ck = HERE / "checkpoints" / variant / f"deployed_{fold}_s{s}.pt"
            hashes[f"{fold}_s{s}"] = hashlib.sha256(ck.read_bytes()).hexdigest()
            sd = torch.load(ck, map_location="cpu")
            net = _net("tfm", Xte.shape[1], 61, {"d": 16})
            net.load_state_dict(sd)
            net.eval()
            with torch.no_grad():
                p = torch.sigmoid(net(torch.tensor(Xte), 0)).numpy().astype(float)
            saved = pr[(pr.fold == fold) & (pr.seed == str(s))].set_index("row_id").score.reindex(ids).to_numpy(float)
            assert np.isfinite(saved).all() and np.abs(p - saved).max() < 1e-6, (m, fold, s, np.nanmax(np.abs(p - saved)))
            P = params(sd)
            pn = 1 / (1 + np.exp(-np.array([np_forward(P, Xte[i].astype(float)) for i in range(len(Xte))])))
            assert np.abs(pn - p).max() < 1e-5, (m, fold, s, "numpy", np.abs(pn - p).max())
            for k, v in P.items():
                arrays[f"{m}_{fold}_s{s}_{k}"] = v.astype(np.float32)
            KEEP[(m, fold)][6].append(P)
            probs.setdefault(m, {}).setdefault(fold, []).append(pn)
        folds[fold] = {"trained_on": list(trr), "calibrated_with": test_room, "n_scored": int(len(te))}
        probs[m][fold] = (ids, np.mean(probs[m][fold], 0))
        print(f"{m} {fold}: {len(te)} windows, 5 seeds reproduce the saved P, numpy == torch", flush=True)
    meta["models"][m] = {"variant": variant, "channels": chans, "log_channels": logc, "calibrated_channels": calc,
                         "folds": folds, "checkpoint_sha256": hashes}

# check rows: per fold one scored window where seq13pw and the base disagree (else the first), plus an env1 reference
# minute, which the tab judges in-sample with the env1 model; expected P = this script's numpy pipeline.
def p_of(m: str, fold: str, i: int) -> float:
    L, av, cal, test_room, trr, ad, Ps = KEEP[(m, fold)]
    Z = seq_regime(L, R.ROOM, np.array([i]), test_room, trr, "fixed5", R.REFS, cal=cal)
    x = ad(Z, av[[i]])[0].astype(float)
    return float(np.mean([1 / (1 + np.exp(-np_forward(P, x))) for P in Ps]))


phys = list(dict.fromkeys(c for m in ("seq13pw", "seq15pwu") for c in meta["models"][m]["channels"]))
pi = [ALL_CHANNELS.index(c) for c in phys]
checks = []
for fold, ref in [(f, None) for f in FOLDS] + [("env1", meta["ref_keys"]["env1"][0])]:
    if ref is None:
        ids, pn = probs["seq13pw"][fold]
        _, pb = probs["seq17"][fold]
        d = np.flatnonzero((pn > 0.5) != (pb > 0.5))
        rid = ids[d[0] if len(d) else 0]
    else:
        rid = R.ROWS.row_id.to_numpy()[R.ROWS.key.to_numpy() == ref][0]
    i = R.RID[rid]
    row = {"row_id": str(rid), "stem": str(R.ROWS.key.iloc[i]) if fold != "scnC" else "20261023_132301", "fold": fold,
           "window": int(str(rid).split(":C")[1]) if fold == "scnC" else 0, "reference": ref is not None,
           "phys": [[None if not np.isfinite(v) else float(v) for v in c] for c in S_ALL[i, pi]],
           "base": [[None if not np.isfinite(v) else float(v) for v in c] for c in R.S_RAW[i]],
           "p": {m: p_of(m, fold, i) for m in MODELS}}
    if ref is None:
        for m in MODELS:
            ids, p = probs[m][fold]
            assert abs(row["p"][m] - float(p[list(ids).index(rid)])) < 1e-9
    checks.append(row)
meta["phys_channels"] = phys
meta["check"] = checks
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / "tfm_phys.npz", **arrays)
(OUT / "tfm_phys.json").write_text(json.dumps(meta, indent=1))
pd.DataFrame([{"model": m, "fold": f, "row_id": i, "p": v} for m, fd in probs.items() for f, (ids, p) in fd.items()
              for i, v in zip(ids, p)]).to_csv(OUT / "tfm_phys_export_p.csv", index=False)
print("wrote", OUT / "tfm_phys.npz", OUT / "tfm_phys.json", flush=True)
