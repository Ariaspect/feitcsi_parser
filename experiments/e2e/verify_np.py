"""Checks e2e_np (numpy, board) against e2e.py / extract.py (torch, scipy) on lg and writes the board bundle
(cyphy session, 2026-11-16).

  verify_np.py OUTDIR ROW_ID,...
1. resample_poly vs scipy on random complex data for the (up, down) pairs of the data's rates.
2. prep() on the given rows vs their stored x_40 rows.
3. Per picked config: a torch model with random weights and random BatchNorm statistics (the trained weights were not
   kept; cost does not depend on the values), E.predict on CPU in fp32 vs e2e_np.judge on the same rows.
Bundle: OUTDIR/<cfg>__s<k>.npz (5 random-weight seeds per config), OUTDIR/meta.json (cfgs, room scales B, self
constant, the rows' stems / windows and torch logits for the board to compare).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, "/home/lg_csi/feitcsi_parser-ml")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

import configs as C  # noqa: E402
import e2e as E  # noqa: E402
import e2e_np as N  # noqa: E402

torch.set_num_threads(4)
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = sys.argv[2].split(",")
CFGS = ["sbo.B.all4.st16.att", "sbo.B.all4c", "scnet.B.st16.all4c.att", "scnet.S.lfhf"]
M = E.load_meta()
X = np.load(E.OUT / "x_40.npy", mmap_mode="r")
ii = [int(np.flatnonzero(M.row_id.to_numpy() == r)[0]) for r in ROWS]

# 1. resample_poly
rng = np.random.default_rng(0)
pairs = sorted({(f.numerator, f.denominator) for f in (Fraction(40 / fs).limit_denominator(200) for fs in M.fs.dropna().unique())})
worst = 0.0
for up, down in pairs:
    x = rng.standard_normal((2500, 7)) + 1j * rng.standard_normal((2500, 7)) + np.linspace(0, 3, 2500)[:, None]
    a = resample_poly(x, up, down, axis=0, padtype="line"); b = N.resample_poly(x, up, down)
    worst = max(worst, float(np.abs(a - b).max() / np.abs(a).max()))
print(f"resample_poly: {len(pairs)} (up, down) pairs {pairs[:6]}..., max rel diff {worst:.2e}")

# 2. prep
for i in ii:
    r = M.iloc[i]
    arr = N.prep(r.path, r.t0, r.t1)
    d = np.abs(arr.astype(np.float32) - X[i].astype(np.float32))
    print(f"prep {r.row_id} {r.room} fs {r.fs:.2f}: fp16 values differing {100 * (d > 0).mean():.4f} %, max diff {d.max():.2e}")

# 3. networks (first three rows)
ii3 = ii[:3]
scales = {}
for mode in ("persc", "self"):
    S = E.calibration_scales(X, M, mode)
    scales[mode] = S
room_scale = {r: scales["persc"][int(np.flatnonzero((M.room == r).to_numpy())[0])].tolist() for r in E.ROOMS}
L = np.load(E.OUT / "self_noise.npy")
self_const = float(np.median(L))
meta = {"cfgs": {}, "room_scale": room_scale, "self_const": self_const,
        "rows": [dict(row_id=M.iloc[i].row_id, stem=Path(M.iloc[i].path).stem, t0=float(M.iloc[i].t0),
                      t1=float(M.iloc[i].t1), room=M.iloc[i].room) for i in ii], "torch": {}}
for c in CFGS:
    cfg = C.get(c)
    cd = {k: v for k, v in asdict(cfg).items() if k in ("arch", "stride", "specbr", "sbatt", "sbstat", "width", "depth",
                                                      "kernel", "crop", "comp", "binpool", "cal")}
    meta["cfgs"][c] = cd
    nb = (E.N_BINS - cfg.stride) // cfg.stride + 1
    mode = "self" if cfg.cal == "self" else "persc"
    Xg = torch.as_tensor(np.stack([X[i] for i in ii3]))
    Sg = torch.as_tensor(scales[mode][ii3])
    bat = E.Batcher(Xg, Sg, cfg)
    meta["torch"][c] = {}
    for s in range(5):
        torch.manual_seed(100 + s)
        m = E.build(cfg, nb, 2)
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    mod.running_mean.normal_(0, 0.2); mod.running_var.uniform_(0.5, 2.0)
                    mod.weight.uniform_(0.5, 1.5); mod.bias.normal_(0, 0.1)
        m.eval()
        sd = {k: v.detach().numpy() for k, v in m.state_dict().items() if v.dtype.is_floating_point}
        np.savez(OUT / f"{c}__s{s}.npz", **sd)
        a = time.time()
        ref = E.predict(m, bat, cfg, np.arange(len(ii3)))
        tt = time.time() - a
        net = N.Net(cd, sd)
        a = time.time()
        got = []
        for k, i in enumerate(ii3):
            sc = scales[mode][i] if mode == "persc" else (N.noise_level(X[i]) / self_const).astype(np.float32)
            got.append(N.judge(np.asarray(X[i]), sc, cd, [net]))
        tn = time.time() - a
        got = np.array(got)
        meta["torch"][c][s] = ref.tolist()
        print(f"{c:24s} seed {s}: torch {np.round(ref, 4)} numpy {np.round(got, 4)} max |diff| {np.abs(ref - got).max():.2e}"
              f"  (lg cpu torch {tt:.1f} s, numpy {tn:.1f} s for {len(ii3)} rows)", flush=True)
    if mode == "self":
        d = max(float(np.abs(scales["self"][i] - N.noise_level(X[i]) / self_const).max()) for i in ii3)
        print(f"  self scale: board formula vs calibration_scales max diff {d:.2e}")
(OUT / "meta.json").write_text(json.dumps(meta))
print("bundle", OUT)
