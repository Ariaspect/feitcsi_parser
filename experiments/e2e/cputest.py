"""CPU check of the multi-stream path on fake data (no GPU): every architecture, k = 1 and 2 streams, ref head, aux off."""
import sys
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
for name in sys.argv[1].split(",") if len(sys.argv) > 1 else ["scnet.B", "cnn1d.B", "conv2d.B", "spec.B", "patchtf.B", "scnet.B.ref"]:
    for k in (1, 2):
        cfg = replace(C.get(name), epochs=1, batch=4)
        extra = [(X.clone(), S)] if k == 2 else None
        bat = E.Batcher(X, S, cfg, room_code=np.zeros(N, int), refs={0: np.arange(3)}, extra=extra)
        x = bat.get(torch.arange(4), torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long), 1600, True)
        if k == 2:
            assert torch.allclose(x[:, :, 0:2], x[:, :, 2:4]), "streams differ for identical inputs"
        m = E.fit(cfg, bat, np.arange(N), y, np.ones(N), 0)
        p = E.predict(m, bat, cfg, np.arange(4), chunk=4)
        print(name, k, tuple(x.shape), np.round(p, 3), flush=True)
