"""CPU check of aux / bottleneck on fake data and fake aux targets (writes nothing to out/ except a temp aux file)."""
import sys, shutil
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e as E, configs as C
torch.set_num_threads(4)
N = 12
X = torch.randn(N, E.T_FULL, E.N_BINS, 2).half() * 0.05
S = torch.ones(N, E.N_BINS)
y = np.r_[np.zeros(6), np.ones(6)].astype(np.float32)
z = np.load(E.OUT / "aux.npz"); A = z["A"][:N]
tmp = Path("/tmp/cy_e2e_out"); tmp.mkdir(exist_ok=True)
np.savez(tmp / "aux.npz", A=A, channels=z["channels"]); E.OUT = tmp
for name in sys.argv[1].split(","):
    cfg = replace(C.get(name), epochs=1, batch=4)
    bat = E.Batcher(X, S, cfg)
    m = E.fit(cfg, bat, np.arange(N), y, np.ones(N), 0)
    print(name, np.round(E.predict(m, bat, cfg, np.arange(4), chunk=4), 3), flush=True)
