"""Weights for the Transformer tab (transformer + plf) and the ML tab's 41F calibrated model, one model per held-out room.

Runs on lg with the ml_lab caches (``/home/lg_csi/ml_lab``: scripts/cy_tfm_plf.py patches the 17-channel series into
cy_tfm_fine / cy_tfm_pre; scripts/cy_calib4.py builds the 41F shift calibration). Writes, next to this repo's backend:

* ``backend/tfm_plf.npz`` + ``backend/tfm_plf.json`` -- transformer alone on the C1+C3 15 + plf_res + plf_g series,
  1 s tokens, d 16, one encoder layer, CLS, random start, read at epoch 25, 5 seeds; z calibration (log on the fast
  channels, minus the room's empty median, divided by its empty MAD shrunk halfway to the training rooms'; an empty
  reference minute is calibrated without itself); one fold per held-out room (trained on the other three) and scnC
  (trained on env1 + env2). Each room's empty reference series ship too, for the leave-one-out case.
* ``backend/calib41.json`` -- 41F hidden layer (16 tanh, alpha 3, L-BFGS, balanced, seed 0) on the shift-calibrated
  41 columns (log on the all-positive columns, minus the room's empty median, leave-one-out for an empty minute),
  the same folds.

Both check themselves: the transformer's weights reproduce the evaluated probabilities (cy_tfm_plf_plf_res_plf_g.pkl)
and the numpy forward the torch one; the 41F folds reproduce cy_41hl_strengths (LOEO 88.8, scnC 85.2).
Usage on lg: python scripts/fit_tfm_calib41.py backend   (from the repo root; the last step reloads the written files
through backend.tfm_model / backend.calib41 and stores five verified check rows for the tests). cyphy session 2026-11-10.
"""
import sys
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/home/lg_csi/ml_lab/scripts")
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
from datetime import date
from pathlib import Path

OUT = sys.argv[1]
sys.argv = ["cy_tfm_plf.py"]
import numpy as np, pandas as pd
import torch, torch.nn as nn
import cy_tfm_plf as M                     # 17-channel series patched into F (cy_tfm_fine) and T (cy_tfm_pre)
F, T = M.F, M.T
D, y, ds, mc = F.D, F.y, F.ds, F.mc
c = T.CAL1
ROOMS = ["env1", "env2", "env3", "env3-1"]
CH = list(F.CH) + list(M.CHS)
EP = 25


def run_net(xs, yy, seed, d=16, h=2, crop=40):
    """cy_tfm_pre.run, arm "rand", stopped at epoch 25 -- the same random draws, so the same weights."""
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    cin, n = xs.shape[1], xs.shape[2]; nv = cin // 2

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp = nn.Linear(cin, d); s.pos = nn.Parameter(0.02 * torch.randn(n, d)); s.cls = nn.Parameter(0.02 * torch.randn(1, 1, d))
            s.msk = nn.Parameter(0.02 * torch.randn(d))
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, 2 * d, 0.1, batch_first=True), 1, enable_nested_tensor=False)
            s.drop = nn.Dropout(0.5); s.h = nn.Linear(d, 1); s.rec = nn.Linear(d, nv)

        def forward(s, x, s0=0):
            z = s.inp(x.permute(0, 2, 1)) + s.pos[s0:s0 + x.shape[2]]
            z = s.enc(torch.cat([s.cls.expand(len(x), 1, d), z], 1))
            return s.h(s.drop(z[:, 0])).squeeze(-1)

    net = Net()
    opt = torch.optim.AdamW([p for nm, p in net.named_parameters() if not nm.startswith("rec.")], lr=2e-3, weight_decay=0.1)
    w = torch.tensor(T.C.H.balanced(yy), dtype=torch.float32); yt = torch.tensor(yy, dtype=torch.float32)
    X = torch.tensor(xs); bce = nn.BCEWithLogitsLoss(reduction="none"); N = len(yy)
    for ep in range(1, EP + 1):
        net.train()
        for b in np.array_split(rng.permutation(N), max(1, N // 64)):
            s0 = int(rng.integers(0, n - crop + 1)); x = X[b, :, s0:s0 + crop]
            opt.zero_grad(); loss = (bce(net(x, s0), yt[b]) * w[b]).mean(); loss.backward(); opt.step()
    net.eval()
    return net


def np_forward(P, x):
    """numpy forward of one (34, 61) input -- the same algebra as backend/tfm_model.py."""
    z = x.T @ P["inp_w"].T + P["inp_b"] + P["pos"][: x.shape[1]]
    z = np.vstack([P["cls"], z])
    d = z.shape[1]; hh = 2; dh = d // hh
    qkv = z @ P["in_w"].T + P["in_b"]; q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
    heads = []
    for i in range(hh):
        sl = slice(i * dh, (i + 1) * dh)
        a = q[:, sl] @ k[:, sl].T / np.sqrt(dh); a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        heads.append(a @ v[:, sl])
    att = np.hstack(heads) @ P["out_w"].T + P["out_b"]
    ln = lambda u, g, b: (u - u.mean(1, keepdims=True)) / np.sqrt(u.var(1, keepdims=True) + 1e-5) * g + b
    z = ln(z + att, P["n1_g"], P["n1_b"])
    ff = np.maximum(z @ P["l1_w"].T + P["l1_b"], 0) @ P["l2_w"].T + P["l2_b"]
    z = ln(z + ff, P["n2_g"], P["n2_b"])
    return float(z[0] @ P["h_w"][0] + P["h_b"][0])


def params(net):
    sd = {k: v.detach().numpy().astype(np.float64) for k, v in net.state_dict().items()}
    L = "enc.layers.0."
    return {"inp_w": sd["inp.weight"], "inp_b": sd["inp.bias"], "pos": sd["pos"], "cls": sd["cls"][0],
            "in_w": sd[L + "self_attn.in_proj_weight"], "in_b": sd[L + "self_attn.in_proj_bias"],
            "out_w": sd[L + "self_attn.out_proj.weight"], "out_b": sd[L + "self_attn.out_proj.bias"],
            "l1_w": sd[L + "linear1.weight"], "l1_b": sd[L + "linear1.bias"], "l2_w": sd[L + "linear2.weight"], "l2_b": sd[L + "linear2.bias"],
            "n1_g": sd[L + "norm1.weight"], "n1_b": sd[L + "norm1.bias"], "n2_g": sd[L + "norm2.weight"], "n2_b": sd[L + "norm2.bias"],
            "h_w": sd["h.weight"], "h_b": sd["h.bias"]}


def day_rooms():
    lab = D[~mc].assign(day=lambda d: d.key.str[:8])
    rooms = lab.groupby("day").ds.agg(lambda s: sorted(set(s)))
    return {d: r[0] for d, r in rooms.items() if len(r) == 1}


def acc_table(ok, okA=None, okB=None, objA=None):
    obj = D.obj.to_numpy(int); a = lambda m: round(100 * float(ok[m].mean()), 1) if m.any() else None
    t = {}
    for e in ROOMS:
        m = ds == e; t[e] = {"all": a(m), "kpi1": a(m & (obj == 0)), "kpi2": a(m & (obj == 1))}
    t["LOEO"] = {"all": a(~mc), "kpi1": a(~mc & (obj == 0)), "kpi2": a(~mc & (obj == 1))}
    t["scnC"] = {"all": a(mc), "kpi1": a(mc & (obj == 0)), "kpi2": a(mc & (obj == 1))}
    if okA is not None:
        t["scnA"] = {"all": round(100 * float(okA.mean()), 1), "kpi1": round(100 * float(okA[objA == 0].mean()), 1), "kpi2": round(100 * float(okA[objA == 1].mean()), 1)}
        t["scnB"] = {"all": round(100 * float(okB.mean()), 1), "kpi1": round(100 * float(okB.mean()), 1), "kpi2": None}
    return t


# ---------------------------------------------------------------- transformer + plf
OLD = {(arm, oi, s): (te, pr) for (arm, oi, s), te, pr in pd.read_pickle(f"{F.LAB}/cy_tfm_plf_plf_res_plf_g.pkl") if arm == "rand"}
arrays, folds = {}, {}
for r in ROOMS:
    ref = np.flatnonzero((ds == r) & F.EMP)
    arrays[f"emp_{r}"] = c.S[ref].astype(np.float32)
emp_keys = {r: D.key.to_numpy()[np.flatnonzero((ds == r) & F.EMP)].tolist() for r in ROOMS}
Pfold = {}
for oi, (name, trr) in enumerate(T.OUTER):
    tr = np.flatnonzero(np.isin(ds, trr)); Ztr = c.calib(tr, trr, False)
    mu = np.nanmean(Ztr, axis=(0, 2)); sd = np.nanstd(Ztr, axis=(0, 2)); sd = np.where(sd > 1e-9, sd, 1.0)
    f = T.P4.transform(Ztr)
    te, _ = OLD[("rand", oi, 0)]
    xt = f(c.calib(te, trr, True))
    arrays[f"{name}_mu"], arrays[f"{name}_sd"], arrays[f"{name}_pool"] = mu, sd, c.pool(trr)
    ps = []
    for s in range(5):
        net = run_net(f(Ztr), y[tr], s)
        with torch.no_grad():
            p = torch.sigmoid(net(torch.tensor(xt), 0)).numpy()
        old = OLD[("rand", oi, s)][1][EP][0]
        assert np.abs(p - old).max() < 1e-5, (name, s, np.abs(p - old).max())
        P = params(net)
        pn = 1 / (1 + np.exp(-np.array([np_forward(P, xt[i]) for i in range(min(20, len(xt)))])))
        assert np.abs(pn - p[: len(pn)]).max() < 1e-5, (name, s, "numpy", np.abs(pn - p[: len(pn)]).max())
        for k, v in P.items():
            arrays[f"{name}_s{s}_{k}"] = v.astype(np.float32)
        ps.append(p)
    folds[name] = {"trained_on": list(trr), "calibrated_with": F.Z.REFROOM.get(name, name)}
    Pfold[name] = (te, ps)
    print("tfm fold", name, "reproduced, numpy == torch", flush=True)
okL = np.zeros(len(D));
for name, (te, ps) in Pfold.items():
    okL[te] = np.mean([(p > 0.5) == y[te] for p in ps], 0)
PA = [OLD[("rand", 4, s)][1][EP][1] for s in range(5)]; PB = [OLD[("rand", 4, s)][1][EP][2] for s in range(5)]
objA = pd.read_csv(f"{F.LAB}/lgg_scnA_41.csv").obj.to_numpy(int)
okA = np.mean([(p > 0.5) == F.yA for p in PA], 0); okB = np.mean([(p > 0.5) == F.yB for p in PB], 0)
np.savez_compressed(f"{OUT}/tfm_plf.npz", **arrays)
meta = {"channels": CH, "log_channels": [CH[i] for i in F.LOGC], "calibrated_channels": [CH[i] for i in F.CAL],
        "rooms": ROOMS, "empty_keys": emp_keys, "folds": folds, "seeds": 5, "epoch": EP, "threshold": 0.5,
        "scnC_stem": "20261023_132301", "day_rooms": day_rooms(),
        "accuracy": acc_table(okL, okA, okB, objA), "fitted": str(date.today()),
        "note": "LOEO: each room judged by the model trained on the other three; scnC by the env1 + env2 model; "
                "accuracy = mean over the 5 seeds of per-seed accuracy (ml_lab cy_tfm_plf_table.out)."}
json.dump(meta, open(f"{OUT}/tfm_plf.json", "w"), indent=1)
print("tfm accuracy", json.dumps(meta["accuracy"]), flush=True)

# ---------------------------------------------------------------- 41F hidden layer, shift calibration
import cy_calib4 as K
F41 = [f for f in K.F41]
DK = K.D; yk = DK.y.to_numpy(int); dsk = DK.ds.to_numpy(); mck = dsk == "scnC"
Z = K.VAR["empty calib"][F41].to_numpy(float)
c41 = {"features": F41, "log_features": [f for f in F41 if f in K.pos], "rooms": ROOMS, "threshold": 0.5,
       "empty": {}, "folds": {}, "fitted": str(date.today())}
emp = K.empty.to_numpy()
for r in ROOMS:
    ref = np.flatnonzero((dsk == r) & emp)
    c41["empty"][r] = {"keys": DK.key.to_numpy()[ref].tolist(), "values": [[None if not np.isfinite(v) else float(v) for v in row] for row in K.L[F41].to_numpy(float)[ref]]}
ok41 = np.zeros(len(DK), bool)
for name, trr in [(e, [r for r in ROOMS if r != e]) for e in ROOMS] + [("scnC", ["env1", "env2"])]:
    tr = np.isin(dsk, trr); te = (dsk == name)
    m = K.H.fit("mlp", Z[tr], yk[tr]); ok41[te] = m.predict(Z[te]) == yk[te]
    imp, sc, mlp = m.steps[0][1], m.steps[1][1], m.steps[2][1]
    c41["folds"][name] = {"trained_on": trr, "calibrated_with": K.REF[name], "median": imp.statistics_.tolist(), "mean": sc.mean_.tolist(),
                          "scale": sc.scale_.tolist(), "w1": mlp.coefs_[0].tolist(), "b1": mlp.intercepts_[0].tolist(),
                          "w2": mlp.coefs_[1][:, 0].tolist(), "b2": float(mlp.intercepts_[1][0])}
R4 = np.isin(dsk, ROOMS); objk = DK.obj.to_numpy(int)
a = lambda mm: round(100 * float(ok41[mm].mean()), 1) if mm.any() else None
c41["accuracy"] = {**{e: {"all": a(dsk == e), "kpi1": a((dsk == e) & (objk == 0)), "kpi2": a((dsk == e) & (objk == 1))} for e in ROOMS},
                   "LOEO": {"all": a(R4), "kpi1": a(R4 & (objk == 0)), "kpi2": a(R4 & (objk == 1))},
                   "scnC": {"all": a(mck), "kpi1": a(mck & (objk == 0)), "kpi2": a(mck & (objk == 1))}}
print("41F accuracy", json.dumps(c41["accuracy"]), flush=True)
assert abs(c41["accuracy"]["LOEO"]["all"] - 88.8) < 0.15 and abs(c41["accuracy"]["scnC"]["all"] - 85.2) < 0.15
json.dump(c41, open(f"{OUT}/calib41.json", "w"))
print("DONE")

# ---------------------------------------------------------------- check rows for the tests (tests/test_tfm_calib41.py)
# Five training minutes: their raw series / 41 values and the evaluated probability, through this repo's backend
# (OUT must be this repo's backend/ so the modules load the files just written).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import tfm_model as TM, calib41 as C41
import cy_znorm as Zn
V = {r["key"]: r["X"] for r in pd.read_pickle(f"{F.LAB}/cy_cvar_C1_bs31+C2b_bank31+C3_hop2.pkl") if "err" not in r}
PL = pd.read_pickle(f"{F.LAB}/cy_plf.pkl")
Pk = {}
for (arm, oi, s), (te, pr) in OLD.items():
    for i, p in zip(te, pr[EP][0]):
        Pk.setdefault(Zn.D.key.iloc[i], []).append(float(p))
CHECK = ["20261013_031054", "20261012_201527", "20261021_142736", "20261001_184202", "20261023_111510"]
chk = []
for k in CHECK:
    X = np.concatenate([np.asarray(V[k], float), np.asarray(PL["train"][k]["X"], float)])
    room = TM.room_for(k); pr = TM.predict(TM.calibrate(X, k, room), room["fold"])
    assert abs(pr["p"] - np.mean(Pk[k])) < 1e-6
    chk.append({"stem": k, "X": [[None if not np.isfinite(v) else float(v) for v in row] for row in X], "p": pr["p"]})
meta["check"] = chk; json.dump(meta, open(f"{OUT}/tfm_plf.json", "w"), indent=1)
chk = []
for k in CHECK:
    i = int(np.flatnonzero(DK.key.to_numpy() == k)[0]); e = dsk[i]; trr = [r for r in ROOMS if r != e]
    pe = float(K.H.fit("mlp", Z[np.isin(dsk, trr)], yk[np.isin(dsk, trr)]).predict_proba(Z[[i]])[0, 1])
    vals = {f: float(DK[f].iloc[i]) for f in F41}
    assert abs(C41.predict(vals, k, TM.room_for(k))["p"] - pe) < 1e-9
    chk.append({"stem": k, "values": [None if not np.isfinite(vals[f]) else vals[f] for f in F41], "p": pe})
c41["check"] = chk; json.dump(c41, open(f"{OUT}/calib41.json", "w"))
print("check rows written")
