"""End-to-end human-presence models on the raw antenna ratio (cyphy session, 2026-11-15).

Input per window: z = r / mean_t(r) - 1 on all 245 live bins, 60 s at 40 Hz (extract.py). The networks see z after
a fixed complex log compression (|z| -> c log1p(|z| / c), phase kept) and nothing else: no steps, no FarSense, no
breathing lines, no gating. Optional calibration divides z by the room's empty-room noise level measured on its five
fixed install references (fixed5), before the compression.

Training: rooms balanced (each (room, y) cell the same total weight), AdamW + cosine, random 40-s crops, a random
every-4th-bin subset (61 bins), random time reversal / conjugation / global phase rotation -- all from the window
itself (augment-same-capture rule). Prediction: mean logit over 3 crops x 4 bin offsets.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RATE = 40
T_FULL = 2400
ROOMS = ("env1", "env2", "env3", "env3-1")
N_BINS = 245


# ------------------------------------------------------------------------------------------------------------ data
def load_meta() -> pd.DataFrame:
    M = pd.read_csv(OUT / "meta_40.csv", dtype={"row_id": str, "key": str})
    M["reference"] = M.reference.astype(str).eq("True")
    M["lowrate"] = M.fs < 30.0
    return M


def room_sets() -> list[tuple[str, ...]]:
    """Every training-room set the nested scheme needs: 3 rooms (outer LOEO), 2 rooms (inner LOEO; env1+env2 is also
    the scenario C model), 1 room (inner folds of the scenario C fold)."""
    out = [tuple(r for r in ROOMS if r != h) for h in ROOMS]
    out += [tuple(r for r in ROOMS if r not in (a, b)) for i, a in enumerate(ROOMS) for b in ROOMS[i + 1:]]
    out += [("env1",), ("env2",)]
    return out


def rs_name(rs: tuple[str, ...]) -> str:
    return "+".join(rs)


def train_rows(M: pd.DataFrame, rs: tuple[str, ...], extra: bool = False) -> np.ndarray:
    """Labelled training windows of the room set: the LOEO rows minus the fixed references; with extra also the
    env2_later and unused windows of those rooms (scored only when their room is held out, so never both)."""
    sets = ["loeo", "env2_later", "unused"] if extra else ["loeo"]
    m = (M.part == "bench") & M.set.isin(sets) & M.room.isin(rs) & ~M.reference & ~M.lowrate
    return np.flatnonzero(m.to_numpy())


def unlab_rows(M: pd.DataFrame, rs: tuple[str, ...]) -> np.ndarray:
    """Self-supervised pool of a room set: its labelled training windows (labels unused), its references, and the
    unlabelled minutes of its rooms."""
    m = (((M.part == "bench") & (M.set == "loeo") & M.room.isin(rs)) | ((M.part == "unlab") & M.room.isin(rs))) & ~M.lowrate
    return np.flatnonzero(m.to_numpy())


def ref_rows(M: pd.DataFrame) -> dict[str, np.ndarray]:
    return {r: np.flatnonzero(((M.part == "bench") & M.reference & (M.room == r)).to_numpy()) for r in ROOMS}


def room_balance_weights(M: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    d = M.iloc[idx]
    cell = d.room.astype(str) + "|" + d.y.astype(str)
    cnt = cell.map(cell.value_counts())
    w = 1.0 / cnt.to_numpy(float)
    return w / w.mean()


# ------------------------------------------------------------------------------------------------------- config
@dataclass
class Cfg:
    name: str
    arch: str = "cnn1d"
    width: int = 64
    depth: int = 5
    kernel: int = 7
    drop: float = 0.2
    norm: str = "bn"
    epochs: int = 40
    lr: float = 1e-3
    wd: float = 0.05
    batch: int = 32
    crop: int = 1600                 # samples (40 s)
    stride: int = 4                  # bin subset stride
    comp: float = 0.02               # log-compression knee
    cal: str = "none"                # none | self (own noise, regime A) | scalar | persc (fixed5 noise normalisation)
    aug_rev: bool = True
    aug_conj: bool = True
    aug_rot: bool = True
    aug_scale: float = 0.0           # log-uniform +-aug_scale amplitude jitter (0 = off)
    rcrop: bool = True               # random training crop position (False: always the middle of the window)
    roff: bool = True                # random training bin offset (False: always offset 0)
    label_smooth: float = 0.0
    mix_k: int = 8                   # specnet virtual channels
    n_fft: int = 64
    hop: int = 16
    heads: int = 4
    layers: int = 2
    patch: int = 40
    ssl: str = ""                    # "" | mae | ...   (self-supervised pretraining on the room set's pool)
    ssl_epochs: int = 60
    adv: float = 0.0                 # room-adversarial weight (gradient reversal), 0 = off
    ema: float = 0.0
    refmode: str = ""                # "" | diff: head sees [e(x), e(x) - mean e(the room's 5 fixed references)]
    inp: str = "r"                   # r = ratio | rg = ratio + direct-path-removed ratio (x_40g) | g = gated only
    aux: float = 0.0                 # > 0: training-only regression of the per-second physics series (out/aux.npz)
    aux_ch: str = "all"              # all | phys13 (the seq13pw channels) | seq17
    bneck: str = ""                  # physics bottleneck: "" | joint | detach -- the verdict comes only from the
                                     # predicted per-second physics series (needs aux > 0)
    xtrain: bool = False             # also train on the env2_later / unused windows of the training rooms
    specbr: str = ""                 # spectral branches added to the pooled embedding: lf, hf, tf, tl joined by + (SCNet, CNN1D, sbo)
    sbstat: int = 5                  # spectral-branch statistics over bins: 5 = mean/max/quartiles, 7 = + std/min
    sbw: int = 1                     # spectral-branch CNN width multiplier
    sbatt: bool = False              # spectral branches: + a learned attention-weighted mean over bins (order-free)
    binpool: str = "msx"             # SCNet pooling over bins: msx = mean/std/max, q = mean/std/max/quartiles/median
    acpool: int = 0                  # > 0: also pool the normalised temporal autocorrelation of the final feature
                                     #      maps at lags 1..acpool (feature steps), a generic periodicity descriptor
    extra: dict = field(default_factory=dict)

    def key(self) -> str:
        return self.name


# ------------------------------------------------------------------------------------------------ calibration
def noise_level(z: np.ndarray) -> np.ndarray:
    """Per-bin noise level of one window: median |first difference| / sqrt(2) (robust to slow drift and motion
    bursts)."""
    c = z[..., 0].astype(np.float32) + 1j * z[..., 1].astype(np.float32)
    return np.median(np.abs(np.diff(c, axis=0)), axis=0) / math.sqrt(2.0)


def calibration_scales(X: np.ndarray, M: pd.DataFrame, mode: str, tag: str = "") -> np.ndarray:
    """(N, 245) divisor per window from its room's five fixed install references; ones when mode == none."""
    out = np.ones((len(M), N_BINS), np.float32)
    if mode == "none":
        return out
    if mode == "self":                                         # regime A: each window by its own per-bin noise level
        f = OUT / f"self_noise{tag}.npy"
        if f.exists():
            L = np.load(f)
        else:
            L = np.stack([noise_level(X[i]) for i in range(len(M))]).astype(np.float32)
            np.save(f, L)
        return (L / np.median(L)).astype(np.float32)
    refs = ref_rows(M)
    lev = {}
    for r, ii in refs.items():
        L = np.stack([noise_level(X[i]) for i in ii])          # (5, 245)
        lev[r] = np.median(L, axis=0)
    glob = np.median(np.concatenate([v[None] for v in lev.values()]))  # a constant: keeps the scale O(1) after division
    for r, v in lev.items():
        s = v / glob if mode == "persc" else np.full(N_BINS, np.median(v) / glob)
        out[(M.room == r).to_numpy()] = s.astype(np.float32)
    return out


def own_reference_scales(X: np.ndarray, M: pd.DataFrame, mode: str, set_name: str, ref_ids: list[str]) -> dict:
    """Divisors for the windows of one recording from its OWN empty minutes (scenario B, user 2026-11-16: B is not env3
    and has no separate empty captures; its reference = the empty minutes inside the recording). Each reference minute
    is scored against the other ones (leave-same-minute-out). Same global constant as calibration_scales."""
    refs = ref_rows(M)
    lev = [np.median(np.stack([noise_level(X[i]) for i in ii]), axis=0) for ii in refs.values()]
    glob = np.median(np.concatenate([v[None] for v in lev]))
    rid = M.row_id.to_numpy()
    ref_idx = [int(np.flatnonzero(rid == r)[0]) for r in ref_ids]
    L = {i: noise_level(X[i]) for i in ref_idx}
    out = {}
    for j in np.flatnonzero((M.set == set_name).to_numpy()):
        v = np.median(np.stack([L[i] for i in ref_idx if i != j]), axis=0)
        out[int(j)] = (v / glob if mode == "persc" else np.full(N_BINS, np.median(v) / glob)).astype(np.float32)
    return out


# ----------------------------------------------------------------------------------------------------- batches
class Batcher:
    """Crops, bin subsets and augmentations on the GPU from a GPU-resident fp16 tensor (N, T, 245, 2)."""

    def __init__(self, Xg: torch.Tensor, scale: torch.Tensor, cfg: Cfg, room_code: np.ndarray | None = None,
                 refs: dict[int, np.ndarray] | None = None, extra: list | None = None):
        self.X, self.S, self.cfg = Xg, scale, cfg
        self.extra = extra or []                                  # further (tensor, scale) streams, same rows / grid
        self.cin = 2 * (1 + len(self.extra))
        self.room_code = None if room_code is None else torch.as_tensor(room_code, device=Xg.device)
        self.refs = refs
        self.nb = (N_BINS - cfg.stride) // cfg.stride + 1        # same count for every offset 0..stride-1

    def bins(self, offset: torch.Tensor) -> torch.Tensor:
        st = self.cfg.stride
        return offset[:, None] + st * torch.arange(self.nb, device=offset.device)[None, :]

    def get(self, idx: torch.Tensor, start: torch.Tensor, offset: torch.Tensor, crop: int, train: bool) -> torch.Tensor:
        dev = self.X.device
        b = idx.numel()
        t = start[:, None] + torch.arange(crop, device=dev)[None, :]                  # (b, crop)
        bi = self.bins(offset)                                                         # (b, nb)
        RE, IM = [], []
        for X, S in [(self.X, self.S)] + list(self.extra):
            x = X[idx[:, None, None], t[:, :, None], bi[:, None, :]].float()           # (b, crop, nb, 2)
            s = S[idx[:, None], bi]                                                    # (b, nb)
            RE.append(x[..., 0] / s[:, None, :]); IM.append(x[..., 1] / s[:, None, :])
        re, im = torch.stack(RE, -1), torch.stack(IM, -1)                              # (b, crop, nb, k)
        if train:
            cfg = self.cfg
            if cfg.aug_rot:
                ph = torch.rand(b, 1, 1, 1, device=dev) * (2 * math.pi)
                c, sn = torch.cos(ph), torch.sin(ph)
                re, im = re * c - im * sn, re * sn + im * c
            if cfg.aug_conj:
                f = (torch.rand(b, 1, 1, 1, device=dev) < 0.5).float() * 2 - 1
                im = im * f
            if cfg.aug_rev:
                r = torch.rand(b, device=dev) < 0.5
                re = torch.where(r[:, None, None, None], re.flip(1), re)
                im = torch.where(r[:, None, None, None], im.flip(1), im)
            if cfg.aug_scale > 0:
                g = torch.exp((torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * cfg.aug_scale)
                re, im = re * g, im * g
        mag = torch.sqrt(re * re + im * im + 1e-12)
        k = self.cfg.comp
        f = k * torch.log1p(mag / k) / mag / k                                         # compressed, in knee units
        re, im = re * f, im * f
        x = torch.stack([re, im], dim=-1).flatten(-2)                                  # (b, crop, nb, 2k): re0 im0 re1 im1
        return x.permute(0, 2, 3, 1).contiguous()                                      # (b, nb, 2k, crop)


def _as_bins(x: torch.Tensor) -> torch.Tensor:
    """(b, nb, 2k, T) -> (b, nb k, 2, T): extra streams as extra bins (models that mix bins themselves)."""
    b, nb, c, T = x.shape
    return x.reshape(b, nb, c // 2, 2, T).reshape(b, nb * (c // 2), 2, T)


# ------------------------------------------------------------------------------------------------------ models
def _norm1d(kind: str, c: int) -> nn.Module:
    return nn.BatchNorm1d(c) if kind == "bn" else nn.GroupNorm(min(8, c), c)


def _norm2d(kind: str, c: int) -> nn.Module:
    return nn.BatchNorm2d(c) if kind == "bn" else nn.GroupNorm(min(8, c), c)


class Res1d(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int, norm: str, dil: int = 1):
        super().__init__()
        p = dil * (k - 1) // 2
        self.c1 = nn.Conv1d(cin, cout, k, stride, p, dilation=dil, bias=False)
        self.n1 = _norm1d(norm, cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, p, dilation=dil, bias=False)
        self.n2 = _norm1d(norm, cout)
        self.sk = None if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), _norm1d(norm, cout))

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(h + (x if self.sk is None else self.sk(x)))


class Res2d(nn.Module):
    def __init__(self, cin: int, cout: int, k: tuple, stride: tuple, norm: str):
        super().__init__()
        p = (k[0] // 2, k[1] // 2)
        self.c1 = nn.Conv2d(cin, cout, k, stride, p, bias=False)
        self.n1 = _norm2d(norm, cout)
        self.c2 = nn.Conv2d(cout, cout, k, 1, p, bias=False)
        self.n2 = _norm2d(norm, cout)
        self.sk = None if (cin == cout and stride == (1, 1)) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), _norm2d(norm, cout))

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(h + (x if self.sk is None else self.sk(x)))


def _widths(w: int, depth: int) -> list[int]:
    return [int(w * min(2.0, 1.0 + 0.25 * i)) for i in range(depth)]


class Head(nn.Module):
    def __init__(self, c: int, drop: float, ac: int = 0, extra: int = 0):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.ac = ac
        self.acp = nn.Linear(ac, 4) if ac else None          # lags -> 4 numbers per channel (shared over channels)
        self.fc = nn.Linear(2 * c + (4 * c if ac else 0) + extra, 1)

    def embed(self, h):                                   # (b, c, ...) -> (b, 2c): mean and max over the rest
        h = h.flatten(2)
        e = [h.mean(-1), h.amax(-1)]
        if self.ac:
            x = h - h.mean(-1, keepdim=True)
            den = (x * x).sum(-1) + 1e-6
            r = torch.stack([(x[..., k:] * x[..., :-k]).sum(-1) / den for k in range(1, self.ac + 1)], -1)  # (b, c, ac)
            e.append(F.gelu(self.acp(r)).flatten(1))
        return torch.cat(e, 1)

    def forward(self, h):
        return self.fc(self.drop(self.embed(h))).squeeze(1)


class SpecBranch(nn.Module):
    """Per-bin power spectra from a fixed FFT layer, pooled across bins (mean, max and quartiles of the log power at
    each frequency: order-free, no learned bin mixing), then a small 1-D CNN over frequency.
      lf  decimated to 4 Hz, one Hann FFT over the whole crop: -2..2 Hz at 1/crop resolution (breathing, swing)
      hf  Welch over 3.2-s Hann segments (half overlap) at 40 Hz: -20..20 Hz at 0.31 Hz (blade lines vs Doppler)
    The spectra are two-sided (the ratio is complex: the Doppler sign is kept). First stream only."""

    def __init__(self, kind: str, cfg: Cfg):
        super().__init__()
        self.kind = kind
        norm, m = cfg.norm, cfg.sbw
        self.pool = BinPool(cfg, linear=True)
        self.cnn = nn.Sequential(nn.Conv1d(self.pool.k, 32 * m, 7, 1, 3, bias=False), _norm1d(norm, 32 * m), nn.GELU(),
                                 Res1d(32 * m, 48 * m, 5, 2, norm), Res1d(48 * m, 64 * m, 5, 2, norm))
        self.out = 128 * m

    def spectra(self, x):
        z = torch.complex(x[:, :, 0].float(), x[:, :, 1].float())          # (b, nb, T)
        z = z - z.mean(-1, keepdim=True)
        if self.kind == "lf":
            b, nb, T = z.shape
            z = z[..., : T // 10 * 10].reshape(b, nb, T // 10, 10).mean(-1)
            w = torch.hann_window(z.shape[-1], device=z.device)
            P = (torch.fft.fft(z * w).abs() ** 2) / w.square().sum()
        else:
            seg = z.unfold(-1, 128, 64)                                       # (b, nb, nseg, 128)
            w = torch.hann_window(128, device=z.device)
            P = ((torch.fft.fft(seg * w).abs() ** 2) / w.square().sum()).mean(2)
        L = torch.log(torch.fft.fftshift(P, dim=-1) + 1e-4)                 # (b, nb, F)
        return self.pool(L)                                                 # (b, k, F)

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            S = self.spectra(x)
        h = self.cnn(S)
        return torch.cat([h.mean(-1), h.amax(-1)], 1)


class BinPool(nn.Module):
    """(b, nb, F[, frames]) -> (b, k, F[, frames]): order-free statistics over the bins. 5 = mean, max, quartiles;
    7 = + std, min; sbatt adds a softmax-attention mean whose per-bin score comes from the bin's own (time-averaged)
    log spectrum, resampled to 16 points (independent of the frequency grid). Quartiles: linear interpolation
    (= torch.quantile, the 1-D branches since round 3) or nearest rank (the 2-D branches since round 5)."""

    def __init__(self, cfg: Cfg, linear: bool):
        super().__init__()
        self.linear = linear
        self.n = cfg.sbstat
        self.att = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 1)) if cfg.sbatt else None
        self.k = self.n + (1 if cfg.sbatt else 0)

    def forward(self, L):
        s = L.sort(1).values
        nb = s.shape[1]
        q = []
        for f in (0.25, 0.5, 0.75):
            pos = f * (nb - 1)
            if self.linear:
                lo = int(pos); w = pos - lo
                q.append(s[:, lo] * (1 - w) + s[:, min(lo + 1, nb - 1)] * w)
            else:
                q.append(s[:, int(round(pos))])
        out = [L.mean(1), s[:, -1]] + q
        if self.n == 7:
            out += [L.std(1), s[:, 0]]
        if self.att is not None:
            spec = L.mean(-1) if L.dim() == 4 else L                         # (b, nb, F)
            b, nb_, Fq = spec.shape
            z = F.adaptive_avg_pool1d(spec.reshape(b * nb_, 1, Fq), 16).reshape(b, nb_, 16)
            a = torch.softmax(self.att(z).squeeze(-1), 1)                    # (b, nb)
            out.append((L * a.reshape(b, nb_, *([1] * (L.dim() - 2)))).sum(1))
        return torch.stack(out, 1)


class SpecBranch2D(nn.Module):
    """Time-resolved version of SpecBranch: per-bin two-sided STFT log power pooled across bins (mean, max,
    quartiles), then a small 2-D CNN over (frequency, time), mean and max pooled. Keeps what the Welch average loses:
    when in the window the motion happens and how the Doppler content moves (swing period, walk then still).
      tf  40 Hz, 1.6-s Hann segments, hop 0.8 s: -20..20 Hz at 0.625 Hz x 49 frames (40-s crop)
      tl  decimated to 4 Hz, 8-s Hann segments, hop 1 s: -2..2 Hz at 0.125 Hz x 33 frames (breathing, swing)
      ts  40 Hz, 0.8-s segments, hop 0.4 s: -20..20 Hz at 1.25 Hz x 99 frames (short bursts)
      tm  decimated to 10 Hz, 6.4-s segments, hop 1 s: -5..5 Hz at 0.156 Hz x 35 frames
    First stream only."""

    GRID = {"tf": (1, 64, 32), "tl": (10, 32, 4), "ts": (1, 32, 16), "tm": (4, 64, 10)}   # decimation, segment, hop

    def __init__(self, kind: str, cfg: Cfg):
        super().__init__()
        self.kind = kind
        norm, m = cfg.norm, cfg.sbw
        self.dec, self.seg, self.hop = self.GRID[kind]
        self.pool = BinPool(cfg, linear=False)
        self.cnn = nn.Sequential(nn.Conv2d(self.pool.k, 32 * m, 3, 1, 1, bias=False), _norm2d(norm, 32 * m), nn.GELU(),
                                 Res2d(32 * m, 48 * m, (3, 3), (2, 2), norm), Res2d(48 * m, 64 * m, (3, 3), (2, 2), norm))
        self.out = 128 * m

    def spectra(self, x):
        z = torch.complex(x[:, :, 0].float(), x[:, :, 1].float())          # (b, nb, T)
        z = z - z.mean(-1, keepdim=True)
        if self.dec > 1:
            b, nb, T = z.shape
            d = self.dec
            z = z[..., : T // d * d].reshape(b, nb, T // d, d).mean(-1)
        seg = z.unfold(-1, self.seg, self.hop)                              # (b, nb, frames, seg)
        w = torch.hann_window(self.seg, device=z.device)
        P = (torch.fft.fft(seg * w).abs() ** 2) / w.square().sum()
        L = torch.log(torch.fft.fftshift(P, dim=-1) + 1e-4).transpose(2, 3)  # (b, nb, F, frames)
        return self.pool(L)                                                 # (b, k, F, frames)

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            S = self.spectra(x)
        h = self.cnn(S).flatten(2)
        return torch.cat([h.mean(-1), h.amax(-1)], 1)


class CohBranch(nn.Module):
    """Cross-bin coherence spectrum (order-free, fixed transform, no learned part before the CNN). Per frequency f
    the bins' cross-spectral matrix S(f) = sum over Welch segments of x_s x_s^H (x_s = the segment's FFT at f across
    all bins); its eigenvalues (computed from the segment-space Gram matrix, same non-zero spectrum) say how much of
    the energy at f sits in ONE spatial pattern that persists over the segments (a breathing person, a blade line)
    versus a pattern that keeps changing (walking) or none (noise). Per f: log trace, lambda1/tr, lambda2/tr,
    lambda3/tr, (lambda1 - lambda2)/tr, then a small 1-D CNN over frequency like SpecBranch.
      clf  decimated to 4 Hz, 16-s Hann segments, hop 4 s (7 segments / 40 s): -2..2 Hz at 0.0625 Hz
      cmf  decimated to 10 Hz, 6.4-s segments, hop 3.2 s (11 segments): -5..5 Hz at 0.156 Hz
      chf  40 Hz, 3.2-s segments, hop 1.6 s (24 segments): -20..20 Hz at 0.31 Hz
    First stream only."""

    GRID = {"clf": (10, 64, 16), "cmf": (4, 64, 32), "chf": (1, 128, 64)}

    def __init__(self, kind: str, cfg: Cfg):
        super().__init__()
        self.kind = kind
        norm, m = cfg.norm, cfg.sbw
        self.dec, self.seg, self.hop = self.GRID[kind]
        self.cnn = nn.Sequential(nn.Conv1d(5, 32 * m, 7, 1, 3, bias=False), _norm1d(norm, 32 * m), nn.GELU(),
                                 Res1d(32 * m, 48 * m, 5, 2, norm), Res1d(48 * m, 64 * m, 5, 2, norm))
        self.out = 128 * m

    @torch.no_grad()
    def spectra(self, x):
        z = torch.complex(x[:, :, 0].float(), x[:, :, 1].float())          # (b, nb, T)
        z = z - z.mean(-1, keepdim=True)
        b, nb, T = z.shape
        if self.dec > 1:
            d = self.dec
            z = z[..., : T // d * d].reshape(b, nb, T // d, d).mean(-1)
        seg = z.unfold(-1, self.seg, self.hop)                              # (b, nb, S, L)
        w = torch.hann_window(self.seg, device=z.device)
        X = torch.fft.fftshift(torch.fft.fft(seg * w), dim=-1) / w.square().sum().sqrt()
        Xp = X.permute(0, 3, 2, 1)                                          # (b, F, S, nb)
        G = Xp @ Xp.conj().transpose(-1, -2)                                # (b, F, S, S)
        ev = torch.linalg.eigvalsh(G).flip(-1).clamp(min=0)                 # descending
        S = ev.shape[-1]
        tr = ev.sum(-1).clamp(min=1e-12)
        r = ev[..., :3] / tr[..., None]
        if S < 3:
            r = F.pad(r, (0, 3 - S))
        return torch.stack([torch.log(tr / (nb * S) + 1e-4), r[..., 0], r[..., 1], r[..., 2], r[..., 0] - r[..., 1]], 1)

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            C = self.spectra(x)
        h = self.cnn(C)
        return torch.cat([h.mean(-1), h.amax(-1)], 1)


def _branch(k: str, cfg: Cfg) -> nn.Module:
    if k in SpecBranch2D.GRID:
        return SpecBranch2D(k, cfg)
    if k in CohBranch.GRID:
        return CohBranch(k, cfg)
    return SpecBranch(k, cfg)


def _branches(cfg: Cfg) -> nn.ModuleList:
    return nn.ModuleList([_branch(k, cfg) for k in cfg.specbr.split("+") if k])


class _Pooled:
    """pool(x) = the head's embedding of features(x) plus the spectral branches; forward = head on it."""

    def pool(self, x):
        return self.pool_h(self.features(x), x)

    def pool_h(self, h, x):
        e = self.head.embed(h)
        if len(self.sb):
            e = torch.cat([e] + [b(x) for b in self.sb], 1)
        return e

    def logit_h(self, h, x):
        return self.head.fc(self.head.drop(self.pool_h(h, x))).squeeze(1)

    def forward(self, x):
        return self.head.fc(self.head.drop(self.pool(x))).squeeze(1)


class CNN1D(_Pooled, nn.Module):
    """Bins x (re, im) as channels; a 1x1 mixing layer (learned beam), then strided residual blocks over time."""

    def __init__(self, cfg: Cfg, nb: int, cin: int = 2):
        super().__init__()
        w = _widths(cfg.width, cfg.depth)
        self.mix = nn.Sequential(nn.Conv1d(cin * nb, cfg.width, 1, bias=False), _norm1d(cfg.norm, cfg.width), nn.GELU())
        blocks, c = [], cfg.width
        for wi in w:
            blocks.append(Res1d(c, wi, cfg.kernel, 2, cfg.norm)); c = wi
        self.body = nn.Sequential(*blocks)
        self.sb = _branches(cfg)
        self.head = Head(c, cfg.drop, cfg.acpool, sum(b.out for b in self.sb))
        self.c = c

    def features(self, x):                                # x (b, nb, 2, T)
        return self.body(self.mix(x.flatten(1, 2)))


class SCNet(_Pooled, nn.Module):
    """One temporal encoder shared by every bin (a bin is a sample of the same scene), pooled across bins by mean,
    std and max (order-free), then a temporal stage on the pooled sequence."""

    def __init__(self, cfg: Cfg, nb: int, cin: int = 2):
        super().__init__()
        w = cfg.width
        self.cin = cin
        enc, c = [nn.Conv1d(cin, w // 2, 9, 2, 4, bias=False), _norm1d(cfg.norm, w // 2), nn.GELU()], w // 2
        for i in range(cfg.depth - 1):
            enc.append(Res1d(c, w, cfg.kernel, 2, cfg.norm)); c = w
        self.enc = nn.Sequential(*enc)
        self.binpool = cfg.binpool
        k = 3 if cfg.binpool == "msx" else 6
        self.post = nn.Sequential(nn.Conv1d(k * c, 2 * w, 1, bias=False), _norm1d(cfg.norm, 2 * w), nn.GELU(),
                                  Res1d(2 * w, 2 * w, 5, 1, cfg.norm), Res1d(2 * w, 2 * w, 5, 1, cfg.norm, dil=2))
        self.sb = _branches(cfg)
        self.head = Head(2 * w, cfg.drop, cfg.acpool, sum(b.out for b in self.sb))
        self.c = 2 * w

    def features(self, x):
        b, nb, c, T = x.shape
        h = self.enc(x.reshape(b * nb, c, T))
        h = h.reshape(b, nb, h.shape[1], h.shape[2])
        parts = [h.mean(1), h.std(1), h.amax(1)]
        if self.binpool == "q":
            q = torch.quantile(h.float(), torch.tensor([0.25, 0.5, 0.75], device=h.device), dim=1)
            parts += [q[0].to(h.dtype), q[1].to(h.dtype), q[2].to(h.dtype)]
        return self.post(torch.cat(parts, 1))


class SpecNet(nn.Module):
    """Learned complex mixing of the bins into K virtual channels, a fixed STFT (two-sided: Doppler sign kept),
    log power, then a 2-D CNN over (Doppler, time)."""

    def __init__(self, cfg: Cfg, nb: int, cin: int = 2):
        super().__init__()
        K = cfg.mix_k
        nb = nb * (cin // 2)
        self.K, self.n_fft, self.hop = K, cfg.n_fft, cfg.hop
        self.mix = nn.Parameter(torch.randn(2, nb, K) / math.sqrt(nb))
        self.register_buffer("win", torch.hann_window(cfg.n_fft), persistent=False)
        w = cfg.width // 2
        layers, c = [], K
        for i, wi in enumerate(_widths(w, cfg.depth)):
            layers.append(Res2d(c, wi, (3, 3), (2, 2) if i < 4 else (1, 1), cfg.norm)); c = wi
        self.cnn = nn.Sequential(nn.BatchNorm2d(K) if cfg.norm == "bn" else nn.GroupNorm(1, K), *layers)
        self.head = Head(c, cfg.drop, 0)
        self.c = c

    def spec(self, x):
        x = _as_bins(x)
        b, nb, _, T = x.shape
        re, im = x[:, :, 0], x[:, :, 1]                                 # (b, nb, T)
        A, B = self.mix[0], self.mix[1]                                 # complex weights A + jB
        yr = torch.einsum("bnt,nk->bkt", re, A) - torch.einsum("bnt,nk->bkt", im, B)
        yi = torch.einsum("bnt,nk->bkt", re, B) + torch.einsum("bnt,nk->bkt", im, A)
        y = torch.complex(yr.float(), yi.float()).reshape(b * self.K, T)
        S = torch.stft(y, self.n_fft, self.hop, window=self.win, return_complex=True, center=False)
        P = torch.fft.fftshift(S.abs() ** 2, dim=1)
        return torch.log(P + 1e-3).reshape(b, self.K, P.shape[1], P.shape[2])

    def features(self, x):
        return self.cnn(self.spec(x))

    def forward(self, x):
        return self.head(self.features(x))


class Conv2DNet(nn.Module):
    """(re, im) as two image channels over (bin, time): kernels local in frequency, shared across bins."""

    def __init__(self, cfg: Cfg, nb: int, cin: int = 2):
        super().__init__()
        w = cfg.width // 2
        layers = [nn.Conv2d(cin, w, (3, 9), (1, 2), (1, 4), bias=False), _norm2d(cfg.norm, w), nn.GELU()]
        c = w
        for i, wi in enumerate(_widths(w, cfg.depth)):
            layers.append(Res2d(c, wi, (3, 7), (2 if i % 2 else 1, 2), cfg.norm)); c = wi
        self.cnn = nn.Sequential(*layers)
        self.head = Head(c, cfg.drop, 0)
        self.c = c

    def features(self, x):
        return self.cnn(x.permute(0, 2, 1, 3))            # (b, 2, nb, T)

    def forward(self, x):
        return self.head(self.features(x))


class PatchTF(nn.Module):
    """1-s patches: a patch embedding shared by every bin, pooled over bins (mean, max), then a Transformer over the
    patch sequence with a CLS token."""

    def __init__(self, cfg: Cfg, nb: int, cin: int = 2):
        super().__init__()
        d, P = cfg.width, cfg.patch
        self.P = P
        self.emb = nn.Sequential(nn.Linear(2 * P, d), nn.GELU(), nn.Linear(d, d))
        self.mixp = nn.Linear(2 * d, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, T_FULL // P + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 2 * d, cfg.drop, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tf = nn.TransformerEncoder(layer, cfg.layers)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.drop)
        self.fc = nn.Linear(d, 1)
        self.c = d

    def tokens(self, x):
        x = _as_bins(x)
        b, nb, _, T = x.shape
        n = T // self.P
        p = x[..., : n * self.P].reshape(b, nb, 2, n, self.P).permute(0, 1, 3, 2, 4).reshape(b, nb, n, 2 * self.P)
        e = self.emb(p)                                   # (b, nb, n, d)
        return self.mixp(torch.cat([e.mean(1), e.amax(1)], -1))   # (b, n, d)

    def features(self, x):
        t = self.tokens(x)
        b, n, d = t.shape
        h = torch.cat([self.cls.expand(b, 1, d), t], 1) + self.pos[:, : n + 1]
        return self.ln(self.tf(h))[:, 0]

    def forward(self, x):
        return self.fc(self.drop(self.features(x))).squeeze(1)


class SBOnly(nn.Module):
    """Spectral branches alone (no learned temporal encoder on the raw series): what the fixed-FFT views carry by
    themselves. Needs specbr; no aux / bottleneck / SSL."""

    def __init__(self, cfg: Cfg, nb: int, cin: int = 2):
        super().__init__()
        assert cfg.specbr, "sbo needs spectral branches"
        self.sb = _branches(cfg)
        self.head = Head(0, cfg.drop, 0, sum(b.out for b in self.sb))
        self.c = 0

    def pool(self, x):
        return torch.cat([b(x) for b in self.sb], 1)

    def forward(self, x):
        return self.head.fc(self.head.drop(self.pool(x))).squeeze(1)


ARCH = {"cnn1d": CNN1D, "scnet": SCNet, "spec": SpecNet, "conv2d": Conv2DNet, "patchtf": PatchTF, "sbo": SBOnly}


def build(cfg: Cfg, nb: int, cin: int = 2) -> nn.Module:
    return ARCH[cfg.arch](cfg, nb, cin)


# ------------------------------------------------------------------------------------------------- adversarial
class _GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.a = a
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.a * g, None


def pooled(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, PatchTF):
        return model.features(x)
    if hasattr(model, "pool"):
        return model.pool(x)
    return model.head.embed(model.features(x))


def logits_from_pooled(model: nn.Module, e: torch.Tensor) -> torch.Tensor:
    if isinstance(model, PatchTF):
        return model.fc(model.drop(e)).squeeze(1)
    return model.head.fc(model.head.drop(e)).squeeze(1)


class RefHead(nn.Module):
    def __init__(self, dim: int, drop: float):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * dim, 1)

    def forward(self, e, er):
        return self.fc(self.drop(torch.cat([e, e - er], 1))).squeeze(1)


def _pooled_dim(model: nn.Module) -> int:
    return model.c if isinstance(model, PatchTF) else model.head.fc.in_features


def ref_logits(model, rhead, bat: "Batcher", x: torch.Tensor, codes: torch.Tensor, ref_x: dict[int, torch.Tensor]):
    """One encoder pass over the batch and the references of the rooms in it; the head compares each window with
    the mean embedding of its room's references."""
    keys = sorted(ref_x)
    xr = torch.cat([ref_x[k] for k in keys])
    e_all = pooled(model, torch.cat([x, xr]))
    e, er = e_all[: x.shape[0]], e_all[x.shape[0]:]
    n = [ref_x[k].shape[0] for k in keys]
    means = torch.stack([c.mean(0) for c in torch.split(er, n)])            # (R, D)
    pos = torch.full((int(max(keys)) + 1,), -1, device=x.device, dtype=torch.long)
    pos[torch.as_tensor(keys, device=x.device)] = torch.arange(len(keys), device=x.device)
    return rhead(e, means[pos[codes]])


def _ref_batch(bat: "Batcher", codes: list[int], cfg: Cfg, g, train: bool, start=None, off=None) -> dict:
    out = {}
    for c in codes:
        ii = torch.as_tensor(bat.refs[c], device=bat.X.device)
        k = ii.numel()
        if train:
            st = torch.randint(0, T_FULL - cfg.crop + 1, (k,), device=ii.device, generator=g)
            of = torch.randint(0, cfg.stride, (k,), device=ii.device, generator=g)
        else:
            st, of = torch.full_like(ii, start), torch.full_like(ii, off)
        out[c] = bat.get(ii, st, of, cfg.crop, train=train)
    return out


# ---------------------------------------------------------------------------------------------------- training
def _sched(opt, total: int, warm: int):
    def f(step):
        if step < warm:
            return (step + 1) / warm
        q = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, q)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


AUX_SETS = {
    "seq17": ["s025", "s2", "s5", "pk", "rpm", "pk_diff", "pk_res", "drift", "ret", "g15", "bs_all", "bs_r1", "bs_r2",
              "bs_tap", "bs_f", "plf_res", "plf_g"],
    "phys13": ["s025", "s2", "lgrow", "g15_res", "sT", "u_g15v", "pk_w", "bs_w", "rpm", "bs_f", "plf_res", "plf_g",
               "swing_ab"],
}


def aux_targets(cfg: Cfg, idx: np.ndarray, dev) -> tuple[torch.Tensor, int]:
    """(N, C, 61) standardised targets (statistics of the training rows only), NaN where unknown."""
    z = np.load(OUT / "aux.npz")
    A, ch = z["A"], z["channels"].tolist()
    if cfg.aux_ch != "all":
        A = A[:, [ch.index(c) for c in AUX_SETS[cfg.aux_ch]]]
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(A[idx], axis=(0, 2)); sd = np.nanstd(A[idx], axis=(0, 2))
    keep = np.isfinite(mu) & (sd > 1e-9)
    A = (A[:, keep] - mu[keep, None]) / sd[keep, None]
    return torch.as_tensor(A, dtype=torch.float32, device=dev), int(keep.sum())


def _tmap(h: torch.Tensor) -> torch.Tensor:
    return h if h.dim() == 3 else h.mean(2)


class BneckClf(nn.Module):
    """Small classifier over the predicted per-second physics series (b, C, seconds)."""

    def __init__(self, c: int, drop: float):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c, 32, 5, 1, 2, bias=False), nn.BatchNorm1d(32), nn.GELU(),
                                 nn.Conv1d(32, 32, 5, 1, 2, bias=False), nn.BatchNorm1d(32), nn.GELU())
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(64, 1)

    def forward(self, s):
        h = self.net(s)
        return self.fc(self.drop(torch.cat([h.mean(-1), h.amax(-1)], 1))).squeeze(1)


def bneck_logit(model, x, sec: int):
    pa = F.interpolate(model.auxh(_tmap(model.features(x))).float(), size=sec, mode="linear", align_corners=False)
    return model.bclf(pa)


def fit(cfg: Cfg, bat: Batcher, idx: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int,
        rooms: np.ndarray | None = None, init: dict | None = None, log=None) -> nn.Module:
    dev = bat.X.device
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(cfg, bat.nb, bat.cin).to(dev)
    if init is not None:
        missing = model.load_state_dict(init, strict=False)
        if log:
            log(f"init: missing {len(missing.missing_keys)} unexpected {len(missing.unexpected_keys)}")
    adv = None
    if cfg.adv > 0 and rooms is not None:
        n_r = int(rooms.max()) + 1
        adv = nn.Sequential(nn.Linear(_pooled_dim(model), 64), nn.GELU(),
                            nn.Linear(64, n_r)).to(dev)
    rhead = RefHead(_pooled_dim(model), cfg.drop).to(dev) if cfg.refmode else None
    auxh, AUX = None, None
    if cfg.aux > 0:
        AUX, n_aux = aux_targets(cfg, idx, dev)
        auxh = nn.Conv1d(model.c, n_aux, 1).to(dev)
    bclf = BneckClf(n_aux, cfg.drop).to(dev) if cfg.bneck else None
    params = list(model.parameters()) + ([] if adv is None else list(adv.parameters())) + \
        ([] if rhead is None else list(rhead.parameters())) + ([] if auxh is None else list(auxh.parameters())) + \
        ([] if bclf is None else list(bclf.parameters()))
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    n = len(idx)
    steps_ep = max(1, n // cfg.batch)
    total = steps_ep * cfg.epochs
    sch = _sched(opt, total, warm=2 * steps_ep)
    ema = None
    if cfg.ema > 0:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    I = torch.as_tensor(idx, device=dev)
    Y = torch.as_tensor(y, dtype=torch.float32, device=dev)
    W = torch.as_tensor(w, dtype=torch.float32, device=dev)
    Rm = None if rooms is None else torch.as_tensor(rooms, device=dev)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    step = 0
    model.train()
    for ep in range(cfg.epochs):
        perm = torch.randperm(n, device=dev, generator=g)
        for s in range(steps_ep):
            j = perm[s * cfg.batch:(s + 1) * cfg.batch]
            b = j.numel()
            start = torch.randint(0, T_FULL - cfg.crop + 1, (b,), device=dev, generator=g) if cfg.rcrop else \
                torch.full((b,), (T_FULL - cfg.crop) // 2, device=dev, dtype=torch.long)
            off = torch.randint(0, cfg.stride, (b,), device=dev, generator=g) if cfg.roff else \
                torch.zeros(b, device=dev, dtype=torch.long)
            x = bat.get(I[j], start, off, cfg.crop, train=True)
            yy = Y[j]
            if cfg.label_smooth > 0:
                yy = yy * (1 - cfg.label_smooth) + 0.5 * cfg.label_smooth
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if rhead is not None:
                    codes = bat.room_code[I[j]]
                    rx = _ref_batch(bat, sorted(set(codes.tolist())), cfg, g, train=True)
                    lo = ref_logits(model, rhead, bat, x, codes, rx)
                    loss = (F.binary_cross_entropy_with_logits(lo.float(), yy, reduction="none") * W[j]).mean()
                elif auxh is not None:
                    h = model.features(x)
                    sec = cfg.crop // RATE
                    pa = F.interpolate(auxh(_tmap(h)).float(), size=sec, mode="linear", align_corners=False)
                    if bclf is not None:
                        lo = bclf(pa.detach() if cfg.bneck == "detach" else pa)
                    else:
                        lo = model.logit_h(h, x) if hasattr(model, "logit_h") else model.head(h)
                    loss = (F.binary_cross_entropy_with_logits(lo.float(), yy, reduction="none") * W[j]).mean()
                    s0 = (start // RATE).clamp(max=61 - sec)
                    tt = s0[:, None] + torch.arange(sec, device=dev)[None, :]
                    tg = AUX[I[j]][torch.arange(b, device=dev)[:, None, None],
                                   torch.arange(AUX.shape[1], device=dev)[None, :, None], tt[:, None, :]]
                    fm = torch.isfinite(tg)
                    if fm.any():
                        loss = loss + cfg.aux * ((pa - torch.nan_to_num(tg)) ** 2 * fm).sum() / fm.sum()
                elif adv is None:
                    lo = model(x)
                    loss = (F.binary_cross_entropy_with_logits(lo.float(), yy, reduction="none") * W[j]).mean()
                else:
                    e = pooled(model, x)
                    lo = logits_from_pooled(model, e)
                    loss = (F.binary_cross_entropy_with_logits(lo.float(), yy, reduction="none") * W[j]).mean()
                    a = cfg.adv * min(1.0, step / max(1, total // 3))
                    ra = adv(_GradRev.apply(e, a))
                    loss = loss + F.cross_entropy(ra.float(), Rm[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 2.0)
            opt.step(); sch.step(); step += 1
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(cfg.ema).add_(v.detach(), alpha=1 - cfg.ema)
                        else:
                            ema[k].copy_(v)
        if log and (ep % 10 == 9 or ep == cfg.epochs - 1):
            log(f"ep {ep + 1} loss {loss.item():.4f}")
    if ema is not None:
        model.load_state_dict(ema)
    model.eval()
    if rhead is not None:
        rhead.eval()
        model.rhead = rhead
    if bclf is not None:
        auxh.eval(); bclf.eval()
        model.auxh, model.bclf, model.bsec = auxh, bclf, cfg.crop // RATE
    return model


def test_starts(crop: int, n: int = 3) -> list[int]:
    """Evenly spaced crop starts covering the whole window (crop 1600: 0, 400, 800 s/40)."""
    if crop >= T_FULL:
        return [0]
    return sorted({int(round(v)) for v in np.linspace(0, T_FULL - crop, n)})


@torch.no_grad()
def predict(model: nn.Module, bat: Batcher, cfg: Cfg, idx: np.ndarray, chunk: int = 64) -> np.ndarray:
    dev = bat.X.device
    out = np.zeros(len(idx), np.float32)
    starts = test_starts(cfg.crop)
    for a in range(0, len(idx), chunk):
        j = torch.as_tensor(idx[a:a + chunk], device=dev)
        acc = torch.zeros(j.numel(), device=dev)
        k = 0
        for st in starts:
            for o in range(cfg.stride):
                x = bat.get(j, torch.full_like(j, st), torch.full_like(j, o), cfg.crop, train=False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if getattr(model, "rhead", None) is not None:
                        codes = bat.room_code[j]
                        rx = _ref_batch(bat, sorted(set(codes.tolist())), cfg, None, train=False, start=st, off=o)
                        acc += ref_logits(model, model.rhead, bat, x, codes, rx).float()
                    elif getattr(model, "bclf", None) is not None:
                        acc += bneck_logit(model, x, model.bsec).float()
                    else:
                        acc += model(x).float()
                k += 1
        out[a:a + chunk] = (acc / k).cpu().numpy()
    return out


@torch.no_grad()
def embed(model: nn.Module, bat: Batcher, cfg: Cfg, idx: np.ndarray, chunk: int = 64) -> np.ndarray:
    dev = bat.X.device
    outs = []
    for a in range(0, len(idx), chunk):
        j = torch.as_tensor(idx[a:a + chunk], device=dev)
        acc, k = None, 0
        for st in test_starts(cfg.crop):
            for o in range(cfg.stride):
                x = bat.get(j, torch.full_like(j, st), torch.full_like(j, o), cfg.crop, train=False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    e = pooled(model, x).float()
                acc = e if acc is None else acc + e
                k += 1
        outs.append((acc / k).cpu().numpy())
    return np.concatenate(outs)
