"""Step timing of a config (training step and the batch gather separately), optional torch.profiler table."""
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e as E, configs as C, run as R
R._init()
M = R._W["M"]
torch.backends.cudnn.benchmark = True
for name in sys.argv[1].split(","):
    cfg = C.get(name)
    bat = E.Batcher(R._W["X"], R._scale(cfg.cal), cfg)
    model = E.build(cfg, bat.nb).cuda(); opt = torch.optim.AdamW(model.parameters(), 1e-3)
    idx = torch.as_tensor(E.train_rows(M, ("env1", "env2")), device="cuda")
    def gather():
        j = idx[torch.randint(0, len(idx), (cfg.batch,), device="cuda")]
        st = torch.randint(0, E.T_FULL - cfg.crop + 1, (cfg.batch,), device="cuda")
        of = torch.randint(0, cfg.stride, (cfg.batch,), device="cuda")
        return bat.get(j, st, of, cfg.crop, train=True)
    def step(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(x); loss = F.binary_cross_entropy_with_logits(lo.float(), torch.ones_like(lo.float()))
        opt.zero_grad(); loss.backward(); opt.step()
    for _ in range(5): step(gather())
    torch.cuda.synchronize(); a = time.time()
    for _ in range(20): x = gather()
    torch.cuda.synchronize(); tg = (time.time() - a) / 20
    x = gather(); torch.cuda.synchronize(); a = time.time()
    for _ in range(20): step(x)
    torch.cuda.synchronize(); ts = (time.time() - a) / 20
    print(f"{name}: gather {1e3*tg:.1f} ms, step {1e3*ts:.1f} ms, x {tuple(x.shape)}", flush=True)
    if len(sys.argv) > 2:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as p:
            for _ in range(5): step(gather())
            torch.cuda.synchronize()
        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=15))
