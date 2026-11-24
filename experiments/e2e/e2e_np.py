"""numpy-only inference of the end-to-end models, for the board (cyphy session, 2026-11-16).

The board has numpy 1.26 and no torch / scipy. Same arithmetic as e2e.py + extract.py in eval mode:
  prep(path, t0, t1)        decode -> antenna ratio on every live bin -> af8.resample -> resample_poly to 40 Hz
                            (scipy's algorithm, kaiser(5) firwin, padtype line) -> z = r / mean - 1, float16 like x_40
  Net(cfg, params)          SCNet / SBOnly forward (BatchNorm folded), spectral branches lf / hf / tf / tl / clf / cmf /
                            chf, BinPool with attention
  judge(arr, scale, cfg, nets)  mean logit over the seeds and the test-time crops (3 starts x every bin offset)
cfg is a plain dict (arch, stride, specbr, sbatt, sbstat, width, depth, kernel, crop, comp, binpool, cal).
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np

RATE = 40
T_FULL = 2400
N_BINS = 245
F32 = np.float32


# ------------------------------------------------------------------------------------------------ preprocessing
def ratio(path: str, t0: float, t1: float, idx=None):
    from backend import af8, framediff, mtk, tfm_phys_fast as TF
    idx = idx if idx is not None else mtk.MTKIndex(Path(path))
    t = np.asarray(idx.times, dtype=float)
    tr = t - t[0]
    sel = framediff._uniform_selection(idx, None, None)
    ids = np.flatnonzero(np.asarray(sel["mask"], bool) & (tr >= t0) & (tr < t1))
    D = TF.decode2(Path(path), idx, ids)
    fin = np.isfinite(D)
    live = fin.all(axis=1).mean(axis=0) > af8.LIVE_FRACTION
    rows = np.flatnonzero(fin[:, :, live].all(axis=(1, 2)) & (D[:, 0, :] != 0)[:, live].all(axis=1))
    r = D[rows, 1][:, live] / D[rows, 0][:, live]
    g, fab, fs = af8.resample(r, t[ids][rows])
    return g, fs


def firwin_kaiser(numtaps: int, cutoff: float, beta: float = 5.0) -> np.ndarray:
    """scipy.signal.firwin(numtaps, cutoff, window=('kaiser', beta)): low pass, scaled to unit gain at 0."""
    m = np.arange(numtaps) - (numtaps - 1) / 2.0
    h = cutoff * np.sinc(cutoff * m) * np.kaiser(numtaps, beta)
    return h / h.sum()


def _out_len(len_h: int, n_in: int, up: int, down: int) -> int:
    return ((n_in - 1) * up + len_h - 1) // down + 1


def resample_poly(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """scipy.signal.resample_poly(x, up, down, axis=0, padtype='line'), polyphase, one tap at a time."""
    g = math.gcd(up, down); up //= g; down //= g
    if up == down == 1:
        return x.copy()
    n_in = x.shape[0]
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)
    half = 10 * max(up, down)
    h = firwin_kaiser(2 * half + 1, 1.0 / max(up, down)) * up
    pre = down - half % down
    post = 0
    rm = (half + pre) // down
    while _out_len(len(h) + pre + post, n_in, up, down) < n_out + rm:
        post += 1
    h = np.concatenate([np.zeros(pre), h, np.zeros(post)])
    # upfirdn: y[n] = sum_m xe[m] h[n down - m up], xe = x extended by its first-to-last linear trend ('line')
    n = np.arange(rm, rm + n_out)
    pos = n * down
    m_hi = pos // up                                       # newest input sample touching output n
    ph = pos - m_hi * up                                   # its tap index
    K = -(-len(h) // up)
    lo = int(m_hi.min()) - K + 1; hi = int(m_hi.max())
    slope = (x[-1] - x[0]) / (n_in - 1)
    left = max(0, -lo); right = max(0, hi - (n_in - 1))
    parts = []
    if left:
        parts.append(x[0][None] + slope[None] * np.arange(-left, 0)[:, None])
    parts.append(x)
    if right:
        parts.append(x[-1][None] + slope[None] * np.arange(1, right + 1)[:, None])
    xe = np.concatenate(parts, 0)
    y = np.zeros((n_out,) + x.shape[1:], np.result_type(x, h))
    for j in range(K):
        tap = ph + j * up
        ok = tap < len(h)
        if not ok.any():
            break
        y += np.where(ok, h[np.minimum(tap, len(h) - 1)], 0.0)[:, None] * xe[m_hi - j + left]
    return y


def to_rate(g: np.ndarray, fs: float, rate: float = RATE) -> np.ndarray:
    fr = Fraction(rate / fs).limit_denominator(200)
    mu = g.mean(axis=0)
    return resample_poly(g - mu, fr.numerator, fr.denominator) + mu


def prep(path: str, t0: float, t1: float, idx=None, timings: dict | None = None) -> np.ndarray:
    import time
    pc = time.perf_counter
    a = pc(); g, fs = ratio(path, t0, t1, idx); b = pc()
    y = to_rate(g, fs)
    T = int(round(60.0 * RATE))
    if len(y) < T:
        y = np.concatenate([y, np.repeat(y[-1:], T - len(y), axis=0)])
    y = y[:T]
    z = y / y.mean(axis=0) - 1.0
    if z.shape[1] != N_BINS:
        src = np.linspace(0, 1, z.shape[1]); dst = np.linspace(0, 1, N_BINS)
        z = np.stack([np.interp(dst, src, r.real) + 1j * np.interp(dst, src, r.imag) for r in z])
    arr = np.stack([z.real, z.imag], axis=-1).astype(np.float16)
    if timings is not None:
        timings["decode + ratio + grid"] = b - a
        timings["resample_poly + z"] = pc() - b
    return arr


def noise_level(arr: np.ndarray) -> np.ndarray:
    c = arr[..., 0].astype(F32) + 1j * arr[..., 1].astype(F32)
    return np.median(np.abs(np.diff(c, axis=0)), axis=0) / math.sqrt(2.0)


# --------------------------------------------------------------------------------------------------- layers
_P = (0.3275911, 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def gelu(x: np.ndarray) -> np.ndarray:
    """0.5 x (1 + erf(x / sqrt 2)); erf by Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7)."""
    u = x * F32(1 / math.sqrt(2.0))
    a = np.abs(u)
    t = 1.0 / (1.0 + F32(_P[0]) * a)
    poly = t * (F32(_P[1]) + t * (F32(_P[2]) + t * (F32(_P[3]) + t * (F32(_P[4]) + t * F32(_P[5])))))
    erf = 1.0 - poly * np.exp(-a * a)
    return (0.5 * x * (1.0 + np.sign(u) * erf)).astype(F32)


def _fold(p: dict, conv: str, bn: str):
    W = p[conv + ".weight"].astype(F32)
    s = p[bn + ".weight"] / np.sqrt(p[bn + ".running_var"] + 1e-5)
    b = p[bn + ".bias"] - p[bn + ".running_mean"] * s
    return (W * s.reshape(-1, *([1] * (W.ndim - 1)))).astype(F32), b.astype(F32)


def conv1d(x, W, b, stride=1, pad=0, dil=1):
    """x (N, Cin, L), W (Cout, Cin, k): im2col + one matmul per chunk of sequences."""
    N, C, L = x.shape
    O, _, k = W.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    Lo = (x.shape[2] - dil * (k - 1) - 1) // stride + 1
    Wm = W.reshape(O, C * k)
    out = np.empty((N, O, Lo), F32)
    step = max(1, int(4e6 // (C * k * Lo)))
    s0, s1, s2 = x.strides
    for a in range(0, N, step):
        xa = np.ascontiguousarray(x[a:a + step])
        n = xa.shape[0]
        cols = np.lib.stride_tricks.as_strided(xa, (n, C, k, Lo), (xa.strides[0], xa.strides[1], dil * xa.strides[2], stride * xa.strides[2]))
        out[a:a + n] = np.matmul(Wm, cols.reshape(n, C * k, Lo))
    if b is not None:
        out += b[None, :, None]
    return out


def conv2d(x, W, b, stride=(1, 1), pad=(0, 0)):
    """x (N, Cin, H, Wd), W (Cout, Cin, kh, kw)."""
    N, C, H, Wd = x.shape
    O, _, kh, kw = W.shape
    if pad != (0, 0):
        x = np.pad(x, ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])))
    Ho = (x.shape[2] - kh) // stride[0] + 1
    Wo = (x.shape[3] - kw) // stride[1] + 1
    Wm = W.reshape(O, C * kh * kw)
    out = np.empty((N, O, Ho * Wo), F32)
    step = max(1, int(4e6 // (C * kh * kw * Ho * Wo)))
    for a in range(0, N, step):
        xa = np.ascontiguousarray(x[a:a + step])
        n = xa.shape[0]
        sN, sC, sH, sW = xa.strides
        cols = np.lib.stride_tricks.as_strided(xa, (n, C, kh, kw, Ho, Wo), (sN, sC, sH, sW, stride[0] * sH, stride[1] * sW))
        out[a:a + n] = np.matmul(Wm, cols.reshape(n, C * kh * kw, Ho * Wo))
    out = out.reshape(N, O, Ho, Wo)
    if b is not None:
        out += b[None, :, None, None]
    return out


class Res:
    def __init__(self, p, pre, dim, k, stride, dil=1):
        self.dim, self.k, self.stride, self.dil = dim, k, stride, dil
        self.W1, self.b1 = _fold(p, pre + ".c1", pre + ".n1")
        self.W2, self.b2 = _fold(p, pre + ".c2", pre + ".n2")
        self.sk = _fold(p, pre + ".sk.0", pre + ".sk.1") if pre + ".sk.0.weight" in p else None

    def __call__(self, x):
        if self.dim == 1:
            p = self.dil * (self.k - 1) // 2
            h = gelu(conv1d(x, self.W1, self.b1, self.stride, p, self.dil))
            h = conv1d(h, self.W2, self.b2, 1, p, self.dil)
            s = x if self.sk is None else conv1d(x, self.sk[0], self.sk[1], self.stride, 0)
        else:
            p = (self.k // 2, self.k // 2)
            h = gelu(conv2d(x, self.W1, self.b1, (self.stride,) * 2, p))
            h = conv2d(h, self.W2, self.b2, (1, 1), p)
            s = x if self.sk is None else conv2d(x, self.sk[0], self.sk[1], (self.stride,) * 2, (0, 0))
        return gelu(h + s)


def hann(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(F32)      # torch.hann_window, periodic


def _decimate(z, d):
    if d == 1:
        return z
    b, nb, T = z.shape
    return z[..., : T // d * d].reshape(b, nb, T // d, d).mean(-1)


def _unfold(z, size, step):
    n = (z.shape[-1] - size) // step + 1
    s = z.strides
    return np.lib.stride_tricks.as_strided(z, z.shape[:-1] + (n, size), s[:-1] + (step * s[-1], s[-1]))


def _adaptive_avg(x, n):
    L = x.shape[-1]
    return np.stack([x[..., (i * L) // n: -(-((i + 1) * L) // n)].mean(-1) for i in range(n)], -1)


class BinPool:
    def __init__(self, p, pre, cfg, linear):
        self.linear, self.n = linear, cfg.get("sbstat", 5)
        self.att = None
        if cfg.get("sbatt"):
            self.att = [p[pre + ".att.0.weight"], p[pre + ".att.0.bias"], p[pre + ".att.2.weight"], p[pre + ".att.2.bias"]]

    def __call__(self, L):
        s = np.sort(L, axis=1)
        nb = s.shape[1]
        q = []
        for f in (0.25, 0.5, 0.75):
            pos = f * (nb - 1)
            if self.linear:
                lo = int(pos); w = F32(pos - lo)
                q.append(s[:, lo] * (1 - w) + s[:, min(lo + 1, nb - 1)] * w)
            else:
                q.append(s[:, int(round(pos))])
        out = [L.mean(1), s[:, -1]] + q
        if self.n == 7:
            out += [L.std(1, ddof=1), s[:, 0]]
        if self.att is not None:
            spec = L.mean(-1) if L.ndim == 4 else L
            z = _adaptive_avg(spec, 16)
            W1, b1, W2, b2 = self.att
            a = (gelu(z @ W1.T + b1) @ W2.T + b2)[..., 0]
            a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
            out.append((L * a.reshape(*a.shape, *([1] * (L.ndim - 2)))).sum(1))
        return np.stack(out, 1).astype(F32)


class Branch:
    G2 = {"tf": (1, 64, 32), "tl": (10, 32, 4), "ts": (1, 32, 16), "tm": (4, 64, 10)}
    GC = {"clf": (10, 64, 16), "cmf": (4, 64, 32), "chf": (1, 128, 64)}

    def __init__(self, p, pre, kind, cfg):
        self.kind = kind
        self.two = kind in self.G2
        self.coh = kind in self.GC
        self.pool = None if self.coh else BinPool(p, pre + ".pool", cfg, linear=not self.two)
        self.c0 = _fold(p, pre + ".cnn.0", pre + ".cnn.1")
        self.r1 = Res(p, pre + ".cnn.3", 2 if self.two else 1, 3 if self.two else 5, 2)
        self.r2 = Res(p, pre + ".cnn.4", 2 if self.two else 1, 3 if self.two else 5, 2)

    def spectra(self, x):
        z = (x[:, :, 0] + 1j * x[:, :, 1]).astype(np.complex64)
        z = z - z.mean(-1, keepdims=True)
        if self.coh:
            d, seg, hop = self.GC[self.kind]
            z = _decimate(z, d)
            w = hann(seg)
            X = np.fft.fftshift(np.fft.fft(_unfold(z, seg, hop) * w), axes=-1) / np.sqrt((w * w).sum())
            Xp = X.transpose(0, 3, 2, 1)                                     # (b, F, S, nb)
            G = Xp @ np.conj(Xp).swapaxes(-1, -2)
            ev = np.clip(np.linalg.eigvalsh(G)[..., ::-1], 0, None)
            S = ev.shape[-1]; nb = z.shape[1]
            tr = np.maximum(ev.sum(-1), 1e-12)
            r = ev[..., :3] / tr[..., None]
            if S < 3:
                r = np.pad(r, [(0, 0)] * (r.ndim - 1) + [(0, 3 - S)])
            return np.stack([np.log(tr / (nb * S) + 1e-4), r[..., 0], r[..., 1], r[..., 2], r[..., 0] - r[..., 1]], 1).astype(F32)
        if self.two:
            d, seg, hop = self.G2[self.kind]
            z = _decimate(z, d)
            w = hann(seg)
            P = np.abs(np.fft.fft(_unfold(z, seg, hop) * w)) ** 2 / (w * w).sum()
            L = np.log(np.fft.fftshift(P, axes=-1) + 1e-4).swapaxes(2, 3)
            return self.pool(L.astype(F32))
        if self.kind == "lf":
            z = _decimate(z, 10)
            w = hann(z.shape[-1])
            P = np.abs(np.fft.fft(z * w)) ** 2 / (w * w).sum()
        else:
            w = hann(128)
            P = (np.abs(np.fft.fft(_unfold(z, 128, 64) * w)) ** 2 / (w * w).sum()).mean(2)
        L = np.log(np.fft.fftshift(P, axes=-1) + 1e-4)
        return self.pool(L.astype(F32))

    def __call__(self, x):
        S = self.spectra(x)
        if self.two:
            h = gelu(conv2d(S, *self.c0, (1, 1), (1, 1)))
        else:
            h = gelu(conv1d(S, *self.c0, 1, 3))
        h = self.r2(self.r1(h))
        h = h.reshape(h.shape[0], h.shape[1], -1)
        return np.concatenate([h.mean(-1), h.max(-1)], 1)


class Net:
    def __init__(self, cfg: dict, p: dict):
        self.cfg = cfg
        self.sb = [Branch(p, f"sb.{i}", k, cfg) for i, k in enumerate(k for k in cfg["specbr"].split("+") if k)]
        self.fcW, self.fcb = p["head.fc.weight"].astype(F32), p["head.fc.bias"].astype(F32)
        self.scnet = cfg["arch"] == "scnet"
        if self.scnet:
            self.e0 = _fold(p, "enc.0", "enc.1")
            self.enc = [Res(p, f"enc.{3 + i}", 1, cfg.get("kernel", 7), 2) for i in range(cfg.get("depth", 5) - 1)]
            self.p0 = _fold(p, "post.0", "post.1")
            self.post = [Res(p, "post.3", 1, 5, 1), Res(p, "post.4", 1, 5, 1, dil=2)]

    def __call__(self, x):
        e = []
        if self.scnet:
            b, nb, c, T = x.shape
            h = gelu(conv1d(x.reshape(b * nb, c, T), *self.e0, 2, 4))
            for r in self.enc:
                h = r(h)
            h = h.reshape(b, nb, h.shape[1], h.shape[2])
            h = np.concatenate([h.mean(1), h.std(1, ddof=1), h.max(1)], 1)
            h = gelu(conv1d(h, *self.p0))
            for r in self.post:
                h = r(h)
            e += [h.mean(-1), h.max(-1)]
        e += [br(x) for br in self.sb]
        return (np.concatenate(e, 1) @ self.fcW.T + self.fcb)[:, 0]


def crops(arr: np.ndarray, scale: np.ndarray, cfg: dict) -> np.ndarray:
    """The test-time inputs of one window, (3 starts x stride offsets, nb, 2, crop), as Batcher.get(train=False)."""
    st, crop = cfg["stride"], cfg.get("crop", 1600)
    nb = (N_BINS - st) // st + 1
    starts = sorted({int(round(v)) for v in np.linspace(0, T_FULL - crop, 3)}) if crop < T_FULL else [0]
    out = []
    for s0 in starts:
        for o in range(st):
            bi = o + st * np.arange(nb)
            x = arr[s0:s0 + crop][:, bi].astype(F32) / scale[bi][None, :, None]     # (crop, nb, 2)
            re, im = x[..., 0], x[..., 1]
            mag = np.sqrt(re * re + im * im + F32(1e-12))
            k = F32(cfg.get("comp", 0.02))
            f = k * np.log1p(mag / k) / mag / k
            out.append(np.stack([re * f, im * f], -1).transpose(1, 2, 0))
    return np.ascontiguousarray(np.stack(out)).astype(F32)


def judge(arr: np.ndarray, scale: np.ndarray, cfg: dict, nets: list, chunk: int = 8) -> float:
    X = crops(arr, scale, cfg)
    tot = []
    for net in nets:
        tot.append(np.concatenate([net(X[a:a + chunk]) for a in range(0, len(X), chunk)]).mean())
    return float(np.mean(tot))


# ------------------------------------------------------------------------------------- fast path (board, 11-16)
# Same model, rearranged for numpy without BLAS: channels-last activations, im2col + einsum (fastest GEMM on the
# board, float32 sums in another order), GELU in place, and the fixed spectra / bin statistics of a test-time input
# computed once and shared by the seeds (only the attention weights and the CNN differ per seed).
def gelu_fast(x: np.ndarray) -> np.ndarray:
    a = np.abs(x); a *= F32(1 / math.sqrt(2.0))
    t = a * F32(_P[0]); t += F32(1.0); np.reciprocal(t, out=t)
    p = t * F32(_P[5]); p += F32(_P[4]); p *= t; p += F32(_P[3]); p *= t; p += F32(_P[2]); p *= t; p += F32(_P[1]); p *= t
    np.square(a, out=a); np.negative(a, out=a); np.exp(a, out=a); p *= a; p *= F32(0.5)
    return x * np.where(x >= 0, F32(1.0) - p, p)


def _cl_weight(W: np.ndarray) -> np.ndarray:
    """(O, C, k[, k2]) -> (O, k[, k2] * C) matching channels-last im2col."""
    return np.ascontiguousarray(np.moveaxis(W, 1, -1).reshape(W.shape[0], -1))


def conv_cl(x, Wc, b, k, stride=1, pad=0, dil=1):
    """Channels-last conv. 1-D: x (n, L, C), k int. 2-D: x (n, H, W, C), k (kh, kw). Wc from _cl_weight."""
    if x.ndim == 3:
        n, L, C = x.shape
        if pad:
            x = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
        Lo = (x.shape[1] - dil * (k - 1) - 1) // stride + 1
        s = x.strides
        cols = np.lib.stride_tricks.as_strided(x, (n, Lo, k, C), (s[0], stride * s[1], dil * s[1], s[2]))
        out = np.einsum("mk,ok->mo", np.ascontiguousarray(cols).reshape(n * Lo, k * C), Wc).reshape(n, Lo, -1)
    else:
        n, H, W, C = x.shape
        kh, kw = k
        if pad:
            x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        Ho = (x.shape[1] - kh) // stride + 1; Wo = (x.shape[2] - kw) // stride + 1
        s = x.strides
        cols = np.lib.stride_tricks.as_strided(x, (n, Ho, Wo, kh, kw, C), (s[0], stride * s[1], stride * s[2], s[1], s[2], s[3]))
        out = np.einsum("mk,ok->mo", np.ascontiguousarray(cols).reshape(n * Ho * Wo, kh * kw * C), Wc).reshape(n, Ho, Wo, -1)
    if b is not None:
        out += b
    return out


class ResCL:
    def __init__(self, r: Res):
        self.dim, self.k, self.stride, self.dil = r.dim, r.k, r.stride, r.dil
        self.W1, self.b1 = _cl_weight(r.W1), r.b1
        self.W2, self.b2 = _cl_weight(r.W2), r.b2
        self.sk = None if r.sk is None else (_cl_weight(r.sk[0]), r.sk[1])

    def __call__(self, x):
        if self.dim == 1:
            k, p = self.k, self.dil * (self.k - 1) // 2
            h = gelu_fast(conv_cl(x, self.W1, self.b1, k, self.stride, p, self.dil))
            h = conv_cl(h, self.W2, self.b2, k, 1, p, self.dil)
            s = x if self.sk is None else conv_cl(x, self.sk[0], self.sk[1], 1, self.stride)
        else:
            k, p = (self.k, self.k), self.k // 2
            h = gelu_fast(conv_cl(x, self.W1, self.b1, k, self.stride, p))
            h = conv_cl(h, self.W2, self.b2, k, 1, p)
            s = x if self.sk is None else conv_cl(x, self.sk[0], self.sk[1], (1, 1), self.stride)
        h += s
        return gelu_fast(h)


def branch_input(kind: str, x: np.ndarray, sbstat: int = 5) -> dict:
    """Seed-independent part of a branch for a batch of test-time inputs: per-bin log spectra L, the fixed bin
    statistics (stacked on axis 1) and the 16-point attention input."""
    br = Branch.__new__(Branch)
    br.kind, br.two, br.coh = kind, kind in Branch.G2, kind in Branch.GC
    cap = {}

    class _Pool:
        def __call__(self, L):
            cap["L"] = L
            return L
    br.pool = _Pool()
    br.spectra(x)
    L = cap["L"]
    two = br.two
    s = np.sort(L, axis=1)
    nb = s.shape[1]
    q = []
    for f in (0.25, 0.5, 0.75):
        pos = f * (nb - 1)
        if not two:
            lo = int(pos); w = F32(pos - lo)
            q.append(s[:, lo] * (1 - w) + s[:, min(lo + 1, nb - 1)] * w)
        else:
            q.append(s[:, int(round(pos))])
    st = [L.mean(1), s[:, -1]] + q
    if sbstat == 7:
        st += [L.std(1, ddof=1), s[:, 0]]
    spec = L.mean(-1) if L.ndim == 4 else L
    return {"L": L, "stats": np.stack(st, 1).astype(F32), "z16": _adaptive_avg(spec, 16), "two": two}


class FastNet:
    """Per-seed part of an SBOnly net (sbo configs) on cached branch inputs."""

    def __init__(self, net: Net):
        assert not net.scnet, "fast path: sbo only"
        self.br = []
        for b in net.sb:
            c0W, c0b = b.c0
            self.br.append(dict(kind=b.kind, two=b.two, att=b.pool.att, c0=(_cl_weight(c0W), c0b),
                                r1=ResCL(b.r1), r2=ResCL(b.r2)))
        self.fcW, self.fcb = net.fcW, net.fcb

    def __call__(self, cache: dict) -> np.ndarray:
        e = []
        for b in self.br:
            c = cache[b["kind"]]
            S = c["stats"]
            if b["att"] is not None:
                W1, b1, W2, b2 = b["att"]
                a = (gelu_fast(c["z16"] @ W1.T + b1) @ W2.T + b2)[..., 0]
                a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
                L = c["L"]
                S = np.concatenate([S, (L * a.reshape(*a.shape, *([1] * (L.ndim - 2)))).sum(1)[:, None].astype(F32)], 1)
            if b["two"]:
                h = gelu_fast(conv_cl(np.ascontiguousarray(S.transpose(0, 2, 3, 1)), *b["c0"], (3, 3), 1, 1))
                h = b["r2"](b["r1"](h)).reshape(h.shape[0], -1, b["r2"].W2.shape[0])
            else:
                h = gelu_fast(conv_cl(np.ascontiguousarray(S.transpose(0, 2, 1)), *b["c0"], 7, 1, 3))
                h = b["r2"](b["r1"](h))
            e += [h.mean(1), h.max(1)]
        return (np.concatenate(e, 1) @ self.fcW.T + self.fcb)[:, 0]


def crops_sel(arr, scale, cfg, starts_idx=None, offsets=None):
    """crops() restricted to some of the 3 starts (indices) and some bin offsets."""
    st, crop = cfg["stride"], cfg.get("crop", 1600)
    nb = (N_BINS - st) // st + 1
    starts = sorted({int(round(v)) for v in np.linspace(0, T_FULL - crop, 3)})
    starts = [starts[i] for i in (starts_idx if starts_idx is not None else range(len(starts)))]
    offs = offsets if offsets is not None else range(st)
    out = []
    for s0 in starts:
        for o in offs:
            bi = o + st * np.arange(nb)
            x = arr[s0:s0 + crop][:, bi].astype(F32) / scale[bi][None, :, None]
            re, im = x[..., 0], x[..., 1]
            mag = np.sqrt(re * re + im * im + F32(1e-12))
            k = F32(cfg.get("comp", 0.02))
            f = k * np.log1p(mag / k) / mag / k
            out.append(np.stack([re * f, im * f], -1).transpose(1, 2, 0))
    return np.ascontiguousarray(np.stack(out)).astype(F32)


def judge_fast(arr, scale, cfg, fnets, starts_idx=None, offsets=None, chunk=8, timings=None):
    """Mean logit over the chosen test-time inputs and the given seeds (FastNet list)."""
    import time
    pc = time.perf_counter
    a = pc()
    X = crops_sel(arr, scale, cfg, starts_idx, offsets)
    kinds = [k for k in cfg["specbr"].split("+") if k]
    tot = np.zeros(len(fnets))
    t_sp = t_nn = 0.0
    t_cr = pc() - a
    for i in range(0, len(X), chunk):
        a = pc()
        cache = {k: branch_input(k, X[i:i + chunk], cfg.get("sbstat", 5)) for k in kinds}
        b = pc(); t_sp += b - a
        for s, f in enumerate(fnets):
            tot[s] += f(cache).sum()
        t_nn += pc() - b
    if timings is not None:
        timings.update({"crops": t_cr, "spectra (shared)": t_sp, "nets": t_nn})
    return float(tot.sum() / (len(X) * len(fnets)))


# ------------------------------------------------------------------------------------- only the used bins (11-17)
def bins_used(cfg: dict, offsets) -> np.ndarray:
    """Grid bins (of the 245) that the test-time inputs with these bin offsets read."""
    st = cfg["stride"]
    nb = (N_BINS - st) // st + 1
    return np.unique(np.concatenate([o + st * np.arange(nb) for o in offsets]))


def prep_cols(path: str, t0: float, t1: float, need, idx=None, timings: dict | None = None) -> np.ndarray:
    """prep() computed for the grid bins `need` only; the other columns of the (2400, 245, 2) float16 result are zero.
    The frame selection, decode and live-bin / frame masks are prep()'s (they look at every bin); everything after
    them is per bin (ratio, af8.resample, resample_poly, mean, z, float16), so the needed columns equal prep()'s bit
    for bit. When fewer or more than 245 bins are live, the bin interpolation onto the 245 grid reads only the two
    live bins around each needed grid bin, which are the only columns computed."""
    import time
    from backend import af8, framediff, mtk, tfm_phys_fast as TF
    pc = time.perf_counter
    a = pc()
    idx = idx if idx is not None else mtk.MTKIndex(Path(path))
    t = np.asarray(idx.times, dtype=float)
    tr = t - t[0]
    sel = framediff._uniform_selection(idx, None, None)
    ids = np.flatnonzero(np.asarray(sel["mask"], bool) & (tr >= t0) & (tr < t1))
    b = pc()
    D = TF.decode2(Path(path), idx, ids)
    c = pc()
    fin = np.isfinite(D)
    live = fin.all(axis=1).mean(axis=0) > af8.LIVE_FRACTION
    rows = np.flatnonzero(fin[:, :, live].all(axis=(1, 2)) & (D[:, 0, :] != 0)[:, live].all(axis=1))
    lv = np.flatnonzero(live); nl = lv.size
    need = np.asarray(need)
    if nl == N_BINS:
        src_cols = need
    else:
        src = np.linspace(0, 1, nl); dst = np.linspace(0, 1, N_BINS)[need]
        j = np.clip(np.searchsorted(src, dst, side="right") - 1, 0, nl - 2)
        src_cols = np.unique(np.concatenate([j, j + 1]))
    cols = lv[src_cols]
    r = D[rows, 1][:, cols] / D[rows, 0][:, cols]
    d = pc()
    g, fab, fs = af8.resample(r, t[ids][rows])
    e = pc()
    y = to_rate(g, fs)
    T = int(round(60.0 * RATE))
    if len(y) < T:
        y = np.concatenate([y, np.repeat(y[-1:], T - len(y), axis=0)])
    y = y[:T]
    z = y / y.mean(axis=0) - 1.0
    if nl != N_BINS:
        full = np.zeros((T, nl), z.dtype); full[:, src_cols] = z
        src = np.linspace(0, 1, nl); dst = np.linspace(0, 1, N_BINS)
        z = np.stack([np.interp(dst, src, q.real) + 1j * np.interp(dst, src, q.imag) for q in full])[:, need]
    out = np.zeros((T, N_BINS, 2), np.float16)
    out[:, need] = np.stack([z.real, z.imag], axis=-1).astype(np.float16)
    if timings is not None:
        timings.update({"select": b - a, "decode": c - b, "masks + ratio": d - c, "grid": e - d, "resample_poly + z": pc() - e})
    return out


# ---------------------------------------------------------------------------- decode only what is read (11-18)
def _u16_views(path):
    """The capture as little-endian 16-bit words starting at even and at odd bytes (memmap views, no copy)."""
    mm = np.memmap(path, dtype="<u1", mode="r")
    n = mm.size
    return (np.ndarray((n // 2,), "<u2", buffer=mm, offset=0), np.ndarray(((n - 1) // 2,), "<u2", buffer=mm, offset=1))


def _plane(views, off, words):
    """decode_raw2's 14-bit values (int16) of the words `words` (fftshifted bin -> word) at byte offsets `off`."""
    ev, od = views
    raw = np.empty((off.size, words.size), np.uint16)
    e = (off & 1) == 0
    if e.any():
        raw[e] = ev[(off[e] >> 1)[:, None] + words[None, :]]
    if not e.all():
        raw[~e] = od[(off[~e] >> 1)[:, None] + words[None, :]]
    return (raw << 2).view(np.int16) >> 2


def _null_plan(structural: np.ndarray, width: int):
    """fill_nulls2's plan for one slot: NaN columns and (k, left, right, mu) of each interpolated column."""
    from backend import mtk
    nan = np.zeros(width, bool); interp = []
    for start, stop in mtk._null_runs(structural):
        left, right = start - 1, stop
        if stop - start > mtk.MAX_NULL_RUN or left < 0 or right >= width:
            nan[start:stop] = True
            continue
        span = stop - start + 1
        interp += [(k, left, right, (k - left) / span) for k in range(start, stop)]
    return nan, {k: (l, r, mu) for k, l, r, mu in interp}


def _values(views, r_off, i_off, perm, cols, plan):
    """decode2's complex values (after the null fill) of the given frames' slot at the width columns `cols`."""
    _, interp = plan
    def raw(c):
        w = perm[np.asarray(c, int)]
        z = np.empty((r_off.size, len(c)), np.complex128)
        z.real = _plane(views, r_off, w); z.imag = _plane(views, i_off, w)
        return z
    plain = [c for c in cols if c not in interp]
    out = np.empty((r_off.size, len(cols)), np.complex128)
    if plain:
        out[:, [j for j, c in enumerate(cols) if c not in interp]] = raw(plain)
    for j, c in enumerate(cols):
        if c in interp:
            l, r, mu = interp[c]
            zz = raw([l, r])
            out[:, j] = zz[:, 0] * (1 - mu) + zz[:, 1] * mu
    return out


def _fast_plan(path: str, t0: float, t1: float, idx=None):
    """prep_fast()'s first half: frame selection, integer decode, structural nulls, live bins and the frame mask.
    None where prep_cols() must be used instead (narrow frames, a missing slot, live != 245)."""
    from backend import af8, framediff, mtk
    idx = idx if idx is not None else mtk.MTKIndex(Path(path))
    t = np.asarray(idx.times, dtype=float)
    tr = t - t[0]
    sel = framediff._uniform_selection(idx, None, None)
    ids = np.flatnonzero(np.asarray(sel["mask"], bool) & (tr >= t0) & (tr < t1))
    width = idx.num_subcarriers
    if not ids.size or not (np.asarray(idx._bins)[ids] == width).all():
        return None
    offs = [(idx._real_off[ids, s], idx._imag_off[ids, s]) for s in range(2)]
    present = [(ro >= 0) & (io >= 0) for ro, io in offs]
    if not (present[0].any() and present[1].any()):
        return None
    views = _u16_views(path)
    perm = np.fft.fftshift(np.arange(width))
    p0 = np.flatnonzero(present[0]); p1 = np.flatnonzero(present[1])
    nz0 = (_plane(views, offs[0][0][p0], perm) != 0) | (_plane(views, offs[0][1][p0], perm) != 0)   # (n0, width)
    st0 = ~nz0.any(axis=0)
    head = p1[:64]
    cand = np.flatnonzero(((_plane(views, offs[1][0][head], perm) == 0) & (_plane(views, offs[1][1][head], perm) == 0)).all(axis=0))
    st1 = np.zeros(width, bool)
    if cand.size:
        st1[cand] = ((_plane(views, offs[1][0][p1], perm[cand]) == 0) & (_plane(views, offs[1][1][p1], perm[cand]) == 0)).all(axis=0)
    plan = [_null_plan(st0, width), _null_plan(st1, width)]
    both = present[0] & present[1]
    live = ~plan[0][0] & ~plan[1][0] & (np.mean(both) > af8.LIVE_FRACTION)
    lv = np.flatnonzero(live)
    if lv.size != N_BINS:
        return None
    # slot 0 non-zero at every live bin (interpolated nulls by their filled value)
    ok0 = np.zeros(ids.size, bool)
    plain_live = np.array([k for k in lv if k not in plan[0][1]], int)
    ok0[p0] = nz0[:, plain_live].all(axis=1)
    il = [k for k in lv if k in plan[0][1]]
    if il:
        ok0[p0] &= (_values(views, offs[0][0][p0], offs[0][1][p0], perm, il, plan[0]) != 0).all(axis=1)
    rows = np.flatnonzero(both & ok0)
    return dict(views=views, perm=perm, plan=plan, lv=lv, tt=t[ids][rows],
                offs=[(ro[rows], io[rows]) for ro, io in offs])


def _fast_cols(P: dict, need, timings: dict | None = None) -> np.ndarray:
    """prep_fast()'s second half for the grid bins `need`: (2400, len(need), 2) float16. Per bin, so any split of
    the bins gives the same values."""
    import time
    from backend import af8
    pc = time.perf_counter
    a = pc()
    cols = [int(k) for k in P["lv"][np.asarray(need)]]
    (r0, i0), (r1, i1) = P["offs"]
    D0 = _values(P["views"], r0, i0, P["perm"], cols, P["plan"][0])
    D1 = _values(P["views"], r1, i1, P["perm"], cols, P["plan"][1])
    r = D1 / D0
    b = pc()
    g, fab, fs = af8.resample(r, P["tt"])
    c = pc()
    y = to_rate(g, fs)
    T = int(round(60.0 * RATE))
    if len(y) < T:
        y = np.concatenate([y, np.repeat(y[-1:], T - len(y), axis=0)])
    y = y[:T]
    z = y / y.mean(axis=0) - 1.0
    out = np.stack([z.real, z.imag], axis=-1).astype(np.float16)
    if timings is not None:
        timings.update({"values + ratio": b - a, "grid": c - b, "resample_poly + z": pc() - c})
    return out


def prep_fast(path: str, t0: float, t1: float, need, idx=None, timings: dict | None = None) -> np.ndarray:
    """prep_cols() bit for bit, decoding only what its result depends on. decode2 + its masks need, per slot, the
    frames present (from the index), the structural-null columns (zero in every present frame) and, for the frame
    mask, whether slot 0 is non-zero at every live bin of a frame; the values themselves only at the used columns.
    So slot 0 is read at every bin as integers (zero test, no complex tensor), slot 1 at every bin only for its
    first 64 frames (a column non-zero there is not structural; the remaining candidates are read in full), and the
    complex values only at the used columns (plus the two neighbours of a used column that is an interpolated
    null). Falls back to prep_cols() for anything else (narrow frames, a missing slot, live != 245)."""
    import time
    pc = time.perf_counter
    a = pc()
    P = _fast_plan(path, t0, t1, idx)
    if P is None:
        return prep_cols(path, t0, t1, need, idx, timings)
    b = pc()
    need = np.asarray(need)
    tc = {}
    out = np.zeros((int(round(60.0 * RATE)), N_BINS, 2), np.float16)
    out[:, need] = _fast_cols(P, need, tc)
    if timings is not None:
        timings.update({"select + decode ints + masks": b - a, **tc})
    return out


def fast_index(path):
    """mtk.MTKIndex(path) with its scan reading the file through a plain ndarray view of the memmap instead of the
    memmap itself (same bytes; skips np.memmap's per-slice __getitem__ / __array_finalize__, which is most of the
    index time: one 6-byte MAC slice per frame). Every published field is the same (verify_fast.py checks)."""
    from backend import mtk

    class _Index(mtk.MTKIndex):
        def _scan_full(self) -> None:
            self._csd = mtk._CSD_UNSET
            size = self.path.stat().st_size
            if size < mtk.PREFIX_BYTES:
                self._init_empty()
                return
            mm = np.memmap(self.path, dtype=np.uint8, mode="r")
            try:
                if self._build_fast(mm.view(np.ndarray), 0, min(size, mm.size), existing=None) is not None:
                    return
            finally:
                del mm
            with self.path.open("rb") as fh:
                buf = fh.read()
            self._build(memoryview(buf), min(size, len(buf)), base=0, existing=None)
    return _Index(Path(path))


# ------------------------------------------------------------------------------- the judgment on several cores (11-18)
def _fork_map(tasks, fn, cores):
    """[fn(t) for t in tasks]: tasks[0] in this process, the others in forked children pinned to cores[1:]
    (results pickled back through one pipe each). One task: no fork."""
    import os
    import pickle
    import struct
    kids = []
    for k, task in enumerate(tasks[1:], 1):
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                os.close(r)
                if cores and hasattr(os, "sched_setaffinity"):
                    os.sched_setaffinity(0, {cores[k % len(cores)]})
                data = pickle.dumps(fn(task), protocol=pickle.HIGHEST_PROTOCOL)
                with os.fdopen(w, "wb") as f:
                    f.write(struct.pack("<Q", len(data)) + data)
                code = 0
            finally:
                os._exit(code)
        os.close(w)
        kids.append((pid, r))
    out = [fn(tasks[0])]
    for pid, r in kids:
        with os.fdopen(r, "rb") as f:
            head = f.read(8)
            data = f.read(struct.unpack("<Q", head)[0]) if len(head) == 8 else b""
        _, status = os.waitpid(pid, 0)
        if len(head) != 8 or status:
            raise RuntimeError(f"worker {pid} failed (status {status})")
        out.append(pickle.loads(data))
    return out


def _task_cols(nets, path, P, need):
    """Worker task: _fast_cols() of some used bins (the plan travels without its memmap views)."""
    return _fast_cols(dict(P, views=_u16_views(path)), need)


def _task_inputs(nets, cfg, scale, cols, vals, inputs):
    """Worker task: every seed's logits (seeds, len(inputs)) for some test-time inputs (start index, bin offset),
    from the used bins `cols` of the prepared window (`vals`, the other bins zero as in prep_fast)."""
    arr = np.zeros((int(round(60.0 * RATE)), N_BINS, 2), np.float16)
    arr[:, cols] = vals
    X = np.concatenate([crops_sel(arr, scale, cfg, [s0], [o]) for s0, o in inputs])
    cache = {k: branch_input(k, X, cfg.get("sbstat", 5)) for k in cfg["specbr"].split("+") if k}
    return np.stack([f(cache) for f in nets])


_TASKS = {"cols": _task_cols, "inputs": _task_inputs}
_POOL_FDS: set = set()          # this process's ends of every live Pool's pipes


class Pool:
    """Workers forked once (after the nets are loaded), pinned to cores[1:], for judge_par: a running service
    keeps them between judgments, so a judgment pays a pipe round trip instead of a fork. cores[0] is the caller."""

    def __init__(self, cores, nets):
        import os
        import pickle
        import struct
        self.cores, self.nets, self.kids = list(cores), nets, []
        for c in self.cores[1:]:
            ar, aw = os.pipe(); br, bw = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(aw); os.close(br)
                for fd in _POOL_FDS:              # other workers' pipe ends: a copy here would keep them from EOF
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                if hasattr(os, "sched_setaffinity"):
                    os.sched_setaffinity(0, {c})
                fin, fout = os.fdopen(ar, "rb"), os.fdopen(bw, "wb")
                while True:
                    head = fin.read(8)
                    if len(head) < 8:
                        os._exit(0)
                    kind, args = pickle.loads(fin.read(struct.unpack("<Q", head)[0]))
                    if kind is None:
                        os._exit(0)
                    try:
                        res = (True, _TASKS[kind](nets, *args))
                    except Exception as e:  # noqa: BLE001 -- reported to the caller
                        res = (False, repr(e))
                    data = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
                    fout.write(struct.pack("<Q", len(data)) + data); fout.flush()
            os.close(ar); os.close(bw)
            _POOL_FDS.update((aw, br))
            self.kids.append((pid, os.fdopen(aw, "wb"), os.fdopen(br, "rb")))

    def map(self, kind, args_list):
        import pickle
        import struct
        assert len(args_list) <= len(self.cores)
        for (_, w, _), args in zip(self.kids, args_list[1:]):
            data = pickle.dumps((kind, args), protocol=pickle.HIGHEST_PROTOCOL)
            w.write(struct.pack("<Q", len(data)) + data); w.flush()
        out = [_TASKS[kind](self.nets, *args_list[0])]
        for (pid, _, r), _ in zip(self.kids, args_list[1:]):
            head = r.read(8)
            if len(head) < 8:
                raise RuntimeError(f"worker {pid} died")
            ok, res = pickle.loads(r.read(struct.unpack("<Q", head)[0]))
            if not ok:
                raise RuntimeError(f"worker {pid}: {res}")
            out.append(res)
        return out

    def close(self):
        import os
        import pickle
        import struct
        for pid, w, r in self.kids:
            data = pickle.dumps((None, None))
            try:
                w.write(struct.pack("<Q", len(data)) + data); w.flush()
            except OSError:
                pass
            _POOL_FDS.difference_update((w.fileno(), r.fileno()))
            w.close(); r.close(); os.waitpid(pid, 0)
        self.kids = []


def judge_par(path, t0, t1, scale, cfg, fnets, starts_idx=(0, 1, 2), offsets=(0, 8), cores=(3,), idx=None,
              timings=None, pool: Pool | None = None):
    """index + prep_fast + judge_fast with the work spread over len(cores) processes (forked per call, or the
    workers of `pool`), same logit: the used bins are split between the processes (prep is per bin), then the
    test-time inputs (crops, spectra and every seed's net are per input on a numpy without BLAS, as on the board).
    The per-input logits are put back in judge_fast's order and summed as it does. cores[0] is this process. On a
    numpy with BLAS the head matmul can depend on the batch size (OpenBLAS), so the split logits are only
    guaranteed equal on the board's numpy."""
    import os
    import time
    pc = time.perf_counter
    cores = pool.cores if pool is not None else list(cores)
    nc = len(cores)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cores[0]})

    def run(kind, args_list):
        if pool is not None:
            return pool.map(kind, args_list)
        return _fork_map(args_list, lambda a: _TASKS[kind](fnets, *a), cores)
    a = pc()
    idx = idx if idx is not None else fast_index(path)
    b = pc()
    P = _fast_plan(path, t0, t1, idx)
    need = bins_used(cfg, offsets)
    c = pc()
    if P is None:
        vals = prep_cols(path, t0, t1, need, idx)[:, need]
    else:
        Pl = {k: v for k, v in P.items() if k != "views"}
        groups = [g for g in np.array_split(np.arange(len(need)), nc) if g.size]
        vals = np.concatenate(run("cols", [(str(path), Pl, need[g]) for g in groups]), axis=1)
    d = pc()
    inputs = [(s0, o) for s0 in starts_idx for o in offsets]
    ig = [g for g in np.array_split(np.arange(len(inputs)), nc) if g.size]
    L = np.concatenate(run("inputs", [(cfg, scale, need, vals, [inputs[i] for i in g]) for g in ig]), axis=1)
    tot = np.zeros(len(fnets))
    for s in range(len(fnets)):
        tot[s] += L[s].sum()
    if timings is not None:
        timings.update({"index": b - a, "decode ints + masks": c - b, "bins (split)": d - c, "inputs (split)": pc() - d})
    return float(tot.sum() / (len(inputs) * len(fnets)))
