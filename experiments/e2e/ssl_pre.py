"""Self-supervised pretraining for the end-to-end study (cyphy session, 2026-11-15).

Per room set, on its pool only (E.unlab_rows: its labelled training windows without labels, its references, its
unlabelled minutes -- never a held-out room), two objectives:
  mae     1-s blocks of the input masked (40 %), the encoder's features predict, at their own time resolution, the
          masked blocks' local mean (re, im) and log power per bin (spectrogram cells for SpecNet, tokens for PatchTF)
  simclr  two views of the same window (different 40-s crops, bin offsets, phase / conjugation / reversal), NT-Xent
          with the other windows of the batch as negatives (no other capture's data enters a view)
The encoder weights initialise the supervised fit (configs with ssl set); heads are fresh.
Saved: out/ssl/<key>/<roomset>.pt.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2e as E  # noqa: E402

SSL = HERE / "out" / "ssl"


def key(cfg: E.Cfg) -> str:
    k = f"{cfg.arch}.w{cfg.width}.d{cfg.depth}.k{cfg.kernel}.{cfg.norm}.st{cfg.stride}.c{cfg.comp}.{cfg.cal}.{cfg.ssl}.e{cfg.ssl_epochs}"
    if cfg.arch == "patchtf":
        k += f".L{cfg.layers}.p{cfg.patch}"
    if cfg.arch == "spec":
        k += f".K{cfg.mix_k}.f{cfg.n_fft}.h{cfg.hop}"
    return k


def path(cfg: E.Cfg, rs) -> Path:
    return SSL / key(cfg) / f"{E.rs_name(rs)}.pt"


def load(cfg: E.Cfg, rs) -> dict:
    sd = torch.load(path(cfg, rs), map_location="cuda")
    return {k: v for k, v in sd.items() if not k.startswith(("head.", "fc."))}


def _block_mask(b: int, T: int, block: int, frac: float, dev, g) -> torch.Tensor:
    n = T // block
    m = torch.rand(b, n, device=dev, generator=g) < frac
    return m.repeat_interleave(block, 1)[:, :T].float()            # (b, T) 1 = masked


def _targets(x: torch.Tensor) -> torch.Tensor:
    """(b, nb, 2, T) -> (b, nb, 3, T): re, im, log power."""
    pw = torch.log1p(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)
    return torch.cat([x, pw[:, :, None]], 2)


class MAEHead(nn.Module):
    def __init__(self, model: nn.Module, nb: int, cfg: E.Cfg):
        super().__init__()
        self.kind = cfg.arch
        if cfg.arch == "cnn1d":
            self.dec = nn.Conv1d(model.c, 3 * nb, 1)
        elif cfg.arch == "scnet":
            self.dec = nn.Conv1d(cfg.width, 3, 1)
        elif cfg.arch == "conv2d":
            self.dec = nn.Conv2d(model.c, 3, 1)
        elif cfg.arch == "spec":
            self.dec = nn.Conv2d(model.c, cfg.mix_k, 1)
        elif cfg.arch == "patchtf":
            self.dec = nn.Linear(cfg.width, 3 * nb)
            self.mask_tok = nn.Parameter(torch.zeros(1, 1, cfg.width))

    def loss(self, model, x, cfg, g):
        b, nb, _, T = x.shape
        dev = x.device
        m = _block_mask(b, T, cfg.patch if cfg.arch == "patchtf" else E.RATE, 0.4, dev, g)
        if self.kind == "spec":
            S = model.spec(x)                                            # (b, K, F, Fr)
            Fr = S.shape[-1]
            mf = F.interpolate(m[:, None], size=Fr, mode="nearest")[:, 0]   # (b, Fr)
            Sin = S * (1 - mf[:, None, None, :])
            h = model.cnn(Sin)
            pred = self.dec(h)
            tgt = F.adaptive_avg_pool2d(S, pred.shape[-2:])
            mm = F.adaptive_avg_pool1d(mf[:, None], pred.shape[-1])[:, 0] > 0.5
            err = ((pred - tgt) ** 2).mean((1, 2))
            return (err * mm).sum() / mm.sum().clamp(min=1)
        tg = _targets(x)                                                 # (b, nb, 3, T)
        xin = x * (1 - m[:, None, None, :])
        if self.kind == "cnn1d":
            h = model.features(xin)                                      # (b, c, T')
            pred = self.dec(h)                                           # (b, 3nb, T')
            tgt = F.adaptive_avg_pool1d(tg.flatten(1, 2), pred.shape[-1])
            mm = F.adaptive_avg_pool1d(m[:, None], pred.shape[-1])[:, 0] > 0.5
            err = ((pred - tgt) ** 2).mean(1)
        elif self.kind == "scnet":
            h = model.enc(xin.reshape(b * nb, 2, T))                     # (b nb, w, T')
            pred = self.dec(h)                                           # (b nb, 3, T')
            tgt = F.adaptive_avg_pool1d(tg.reshape(b * nb, 3, T), pred.shape[-1])
            mm = (F.adaptive_avg_pool1d(m[:, None], pred.shape[-1])[:, 0] > 0.5).repeat_interleave(nb, 0)
            err = ((pred - tgt) ** 2).mean(1)
        elif self.kind == "conv2d":
            h = model.features(xin)                                      # (b, c, F', T')
            pred = self.dec(h)                                           # (b, 3, F', T')
            tgt = F.adaptive_avg_pool2d(tg.permute(0, 2, 1, 3), pred.shape[-2:])
            mm = F.adaptive_avg_pool1d(m[:, None], pred.shape[-1])[:, 0] > 0.5
            err = ((pred - tgt) ** 2).mean((1, 2))
        else:                                                            # patchtf, BERT-style token masking
            P = cfg.patch
            n = T // P
            tok = model.tokens(x)                                        # (b, n, d)
            mt = m[:, ::P][:, :n] > 0.5
            tok = torch.where(mt[..., None], self.mask_tok.expand_as(tok), tok)
            hh = torch.cat([model.cls.expand(b, 1, tok.shape[-1]), tok], 1) + model.pos[:, : n + 1]
            hh = model.ln(model.tf(hh))[:, 1:]
            pred = self.dec(hh)                                          # (b, n, 3nb)
            tgt = tg[..., : n * P].reshape(b, nb, 3, n, P).mean(-1).permute(0, 3, 1, 2).reshape(b, n, 3 * nb)
            err = ((pred - tgt) ** 2).mean(-1)
            mm = mt
        return (err * mm).sum() / mm.sum().clamp(min=1)


def _views(bat: E.Batcher, cfg: E.Cfg, I: torch.Tensor, g):
    b = I.numel(); dev = I.device
    s1 = torch.randint(0, E.T_FULL - cfg.crop + 1, (b,), device=dev, generator=g)
    s2 = torch.randint(0, E.T_FULL - cfg.crop + 1, (b,), device=dev, generator=g)
    o1 = torch.randint(0, cfg.stride, (b,), device=dev, generator=g)
    o2 = (o1 + 1 + torch.randint(0, max(1, cfg.stride - 1), (b,), device=dev, generator=g)) % cfg.stride
    return bat.get(I, s1, o1, cfg.crop, train=True), bat.get(I, s2, o2, cfg.crop, train=True)


def pretrain(cfg: E.Cfg, bat: E.Batcher, idx: np.ndarray, seed: int = 0, log=None) -> dict:
    dev = bat.X.device
    torch.manual_seed(seed)
    model = E.build(cfg, bat.nb, bat.cin).to(dev)
    if cfg.ssl == "mae":
        aux = MAEHead(model, bat.nb, cfg).to(dev)
    else:
        dim = E._pooled_dim(model)
        aux = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 64)).to(dev)
    params = list(model.parameters()) + list(aux.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    bs = 64 if cfg.ssl == "simclr" else cfg.batch
    steps_ep = max(1, len(idx) // bs)
    total = steps_ep * cfg.ssl_epochs
    sch = E._sched(opt, total, warm=2 * steps_ep)
    I = torch.as_tensor(idx, device=dev)
    g = torch.Generator(device=dev); g.manual_seed(seed + 1000)
    model.train()
    for ep in range(cfg.ssl_epochs):
        perm = torch.randperm(len(idx), device=dev, generator=g)
        for s in range(steps_ep):
            j = I[perm[s * bs:(s + 1) * bs]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if cfg.ssl == "mae":
                    st = torch.randint(0, E.T_FULL - cfg.crop + 1, (j.numel(),), device=dev, generator=g)
                    of = torch.randint(0, cfg.stride, (j.numel(),), device=dev, generator=g)
                    x = bat.get(j, st, of, cfg.crop, train=True)
                    loss = aux.loss(model, x, cfg, g)
                else:
                    a, b2 = _views(bat, cfg, j, g)
                    za = F.normalize(aux(E.pooled(model, a)).float(), dim=1)
                    zb = F.normalize(aux(E.pooled(model, b2)).float(), dim=1)
                    z = torch.cat([za, zb]); n = za.shape[0]
                    sim = z @ z.t() / 0.2
                    sim.fill_diagonal_(-1e9)
                    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(dev)
                    loss = F.cross_entropy(sim, tgt)
            opt.zero_grad(set_to_none=True)
            loss.float().backward()
            nn.utils.clip_grad_norm_(params, 2.0)
            opt.step(); sch.step()
        if log and (ep % 10 == 9 or ep == cfg.ssl_epochs - 1):
            log(f"ssl ep {ep + 1} loss {loss.item():.4f}")
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _task(t):
    import run as R
    cname, rs = t
    import configs as C
    cfg = C.get(cname)
    out = path(cfg, rs)
    if out.exists():
        return (t, "skip", 0.0)
    a = time.time()
    try:
        M = R._W["M"]
        bat = R.batcher(cfg, M)
        sd = pretrain(cfg, bat, E.unlab_rows(M, rs))
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, out)
        torch.cuda.empty_cache()
        return (t, "ok", time.time() - a)
    except Exception:  # noqa: BLE001
        import traceback
        return (t, traceback.format_exc(), 0.0)


def run_all(cfgs: list[str], sets, workers: int) -> None:
    import multiprocessing as mp
    import configs as C
    import run as R
    seen, tasks = set(), []
    for c in cfgs:
        for rs in sets:
            k = (key(C.get(c)), rs)
            if k not in seen and not path(C.get(c), rs).exists():
                seen.add(k); tasks.append((c, rs))
    print(f"ssl: {len(tasks)} pretrainings", flush=True)
    if not tasks:
        return
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=R._init) as pool:
        for i, (t, st, dt) in enumerate(pool.imap_unordered(_task, tasks)):
            print(f"ssl {i + 1}/{len(tasks)} {t[0]} {E.rs_name(t[1])} {st if st != 'ok' else ''} {dt:.0f} s [{time.time() - t0:.0f} s]", flush=True)
