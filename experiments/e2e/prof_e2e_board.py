"""Where the board's time goes in one e2e judgment (cyphy session, 2026-11-16).

  python3 prof_e2e_board.py CODE BUNDLE CFG cap.bin
Times (one window, all seeds): crops, per branch spectra (FFT + pooling), conv1d / conv2d (by layer shape), gelu, head.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402

import e2e_np as N  # noqa: E402

B = Path(sys.argv[2]); C = sys.argv[3]; cap = Path(sys.argv[4])
meta = json.loads((B / "meta.json").read_text())
cd = meta["cfgs"][C]
nets = [N.Net(cd, dict(np.load(B / f"{C}__s{s}.npz"))) for s in range(5)]
r = {q["stem"]: q for q in meta["rows"]}[cap.stem]
T = defaultdict(float)
pc = time.perf_counter


def wrap(mod, name, key):
    f = getattr(mod, name)

    def g(*a, **k):
        t = pc(); out = f(*a, **k)
        T[key(a, out) if callable(key) else key] += pc() - t
        return out
    setattr(mod, name, g)


wrap(N, "conv1d", lambda a, o: f"conv1d {a[1].shape[1]}->{a[1].shape[0]} k{a[1].shape[2]} L{o.shape[2]}")
wrap(N, "conv2d", lambda a, o: f"conv2d {a[1].shape[1]}->{a[1].shape[0]} k{a[1].shape[2]} {o.shape[2]}x{o.shape[3]}")
wrap(N, "gelu", "gelu")
orig = N.Branch.spectra


def spectra(self, x):
    t = pc(); out = orig(self, x); T[f"spectra {self.kind}"] += pc() - t
    return out


N.Branch.spectra = spectra
a = pc(); arr = N.prep(str(cap), r["t0"], r["t1"]); T["prep"] = pc() - a
sc = np.asarray(meta["room_scale"][r["room"]], np.float32)
a = pc(); X = N.crops(arr, sc, cd); T["crops"] = pc() - a
a = pc()
for net in nets:
    for i in range(0, len(X), 8):
        net(X[i:i + 8])
T["forward total"] = pc() - a
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f"{v:8.2f} s  {k}")
