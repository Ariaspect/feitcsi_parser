"""The Physics tab's series computed faster, for the board (user 2026-11-13: cut the ~5 s a minute without losing
accuracy). Same channels, same formulas, same windows as ``tfm_phys.series``; only how the arithmetic is arranged
changes, so values agree to floating-point rounding (checked over every labelled window by
scripts/verify_tfm_phys_fast.py). numpy only. The Physics tab computes its series with this.

Where the time went on the board (one core, seq15pwu, a minute with a fan seen: 5.2 s, now 2.2 s; seq13pw 4.4 -> 1.9 s;
the board's numpy has no BLAS and no SIMD, so every pass over the data counts) and what changed:

* decode (0.50 -> 0.23 s): ``decode2`` writes the 14-bit values straight into the complex output (the fftshift
  folded into the byte gather, nulls filled in place: bit-identical); the (frames, 2, live) tensor is not copied
  twice; the ratio is divided on the 31 read subcarriers only.
* steps + swing period (1.08 -> 0.42 s): ``af8._step`` took |g| three times per lag and ``np.nanmedian`` over
  subcarriers went through numpy's masked-array path; here |g| is taken once and the median is ``np.median`` when
  nothing is NaN (the same order statistic, bit-identical). The ~85-lag refine screens with squared magnitudes and
  recomputes exactly every lag near the best, so the swing period is the exact one.
* FarSense (1.36 -> 0.75 s, three runs): the in-band DFT of every 10 s window was a complex-by-real product redone
  on windows that overlap 80 %; here one real product per stretch between window edges (absolute phase), a window's
  DFT the sum of its stretches shifted to the window start minus the window mean's own DFT. The selected columns'
  autocorrelations are summed as one inverse FFT of their weighted power spectra. BNR per angle, best projection
  and first peak as ``farsense``.
* breathing lines (1.19 -> 0.27 s): the 245-subcarrier grid was resampled and swing-averaged to read every 8th
  column -- the grid g already holds (bit-identical); each 20 s window's 62 x 62 covariance is summed from the
  stretches between window edges; the eight components' prominences in one go, the power at the reference band's
  bins of the n = fs / 0.01-point DFT by chirp-z (n = 4167 = 3^2 x 463 at 41.7 Hz is a slow FFT length).
* reflection (0.62 -> 0.26 s): the gated chain went back through a full FFT to read 31 bins; here those bins are
  chain 1 minus the zeroed taps' own DFT (the FFT of the IFFT is the input); the 1 s moving average of the fan
  subspace is a running sum.
* swing share (0.22 -> 0.06 s): the 6 s and 1 s moving averages are running sums instead of convolutions.

Checked over all 1,426 labelled windows (scripts/verify_tfm_phys_fast.py): swing period identical, channels within
2e-10 relative, P within 1.3e-11, no verdict changes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from backend import af8, af8_env1 as E, breath_components as BC, farsense, framediff, mtk, tfm_series as TS
from backend.presence import fractional_motion
from backend.tfm_phys import PHYS_CHANNELS, sync_average

N = TS.N
FS_FFT = 8192            # FarSense's padded DFT size, its angle count and keep fraction (tfm_series._fs_windows)
FS_THETA = 200
FS_KEEP = 0.65
FS_BAND = (10.0, 30.0)


# ----------------------------------------------------------------------------------------------- small pieces
def _med(d: np.ndarray) -> np.ndarray:
    """``np.nanmedian(d, axis=1)``; ``np.median`` (the same order statistic, no masked arrays) when nothing is NaN."""
    return np.nanmedian(d, axis=1) if np.isnan(d).any() else np.median(d, axis=1)


def _step(g: np.ndarray, A: np.ndarray, L: int) -> np.ndarray:
    """``af8._step(g, L)`` with A = |g| given."""
    return _med(np.abs(g[L:] - g[:-L]) / (A[L:] + A[:-L]))


def _ma(x: np.ndarray, k: int) -> np.ndarray:
    """``np.convolve(c, np.ones(k) / k, "same")`` on every column, from a running sum (needs k <= len(x))."""
    n = len(x)
    off = (k - 1) // 2
    S = np.concatenate([np.zeros((1,) + x.shape[1:], x.dtype), np.cumsum(x, axis=0)])
    i = np.arange(n)
    return (S[np.minimum(i + off + 1, n)] - S[np.maximum(i + off - k + 1, 0)]) / k


def _vsq(g: np.ndarray, A: np.ndarray, L: int) -> float:
    """The refine's median step at lag L, from squared magnitudes: sqrt(median over subcarriers of
    |dg|^2 / (|g_t| + |g_t-L|)^2), then the median over time -- the same order statistics as af8._step's (sqrt is
    monotone), within a few ulp. Only for an odd subcarrier count with nothing NaN (else the exact value)."""
    dr = g.real[L:] - g.real[:-L]
    di = g.imag[L:] - g.imag[:-L]
    den = A[L:] + A[:-L]
    with np.errstate(invalid="ignore", divide="ignore"):
        q = (dr * dr + di * di) / (den * den)
    if np.isnan(q).any():                              # 0 / 0 somewhere: nanmedian's order statistic differs
        return float("nan")
    m = q.shape[1] // 2
    return float(np.median(np.sqrt(np.partition(q, m, axis=1)[:, m])))


def swing_period(g: np.ndarray, A: np.ndarray, fs: float) -> int:
    """``af8.swing_period``: C3's lag on the ~10 Hz curve, refined to the sample with the smallest median step.
    The refine screens every lag with ``_vsq`` and recomputes exactly (``_step``) every lag within 1e-9 of the best
    screened value, so the chosen lag is the exact one (first smallest, as af8's loop)."""
    sub = max(1, int(round(fs / af8.REVISIT_HZ)))
    h, Ah = g[::sub], A[::sub]
    f2 = fs / sub
    taus, D = [], []
    for tau in af8.REVISIT_LAGS:
        L = int(round(tau * f2))
        if L >= len(h):
            break
        taus.append(tau)
        D.append(float(np.nanmedian(_step(h, Ah, L))))
    taus, D = np.array(taus), np.array(D)
    peak = np.maximum.accumulate(D)
    t0 = float(taus[int(np.nanargmax((peak - D) / peak))])
    lo = max(1, int((t0 - af8.PERIOD_REFINE) * fs))
    hi = min(len(g) - 1, int((t0 + af8.PERIOD_REFINE) * fs))
    lags = list(range(lo, hi + 1))
    exact = g.shape[1] % 2 == 0 or np.isnan(g).any() or np.isnan(A).any()
    if not exact:
        v = np.array([_vsq(g, A, L) for L in lags])
        exact = not np.isfinite(v).all()
    if exact:
        v = np.array([float(np.nanmedian(_step(g, A, L))) for L in lags])
        return lags[int(np.argmin(v))]
    near = np.flatnonzero(v <= v.min() * (1 + 1e-9))
    if near.size == 1:
        return lags[int(near[0])]
    ve = [float(np.nanmedian(_step(g, A, lags[i]))) for i in near]
    return lags[int(near[int(np.argmin(ve))])]


def swing_alpha(x: np.ndarray, T: int, fs: float, band: bool = False) -> float:
    """``tfm_phys.swing_alpha`` with running-sum moving averages. Only the interior (k // 2 from each end) is read,
    where a column's constant cancels, so the columns are centred first to keep the running sums small."""
    k = max(1, int(round(6.0 * fs)))
    xz = np.nan_to_num(x)
    if len(xz) < k:
        from backend import tfm_phys
        return tfm_phys.swing_alpha(x, T, fs, band)
    xz = xz - xz.mean(0)
    lp = (_ma(xz, max(1, int(round(1.0 * fs)))) - _ma(xz, k) if band else _ma(xz, k))[k // 2: len(xz) - k // 2]
    lp = lp - lp.mean(0)
    n = len(lp)
    if n < 2 * T:
        return 0.0
    ph = (np.arange(n) + k // 2) % T
    cnt = np.bincount(ph, minlength=T).astype(float)
    avg = np.stack([np.bincount(ph, lp[:, j].real, T) + 1j * np.bincount(ph, lp[:, j].imag, T) for j in range(lp.shape[1])], 1) / cnt[:, None]
    tot = float((np.abs(lp) ** 2).sum())
    if tot <= 0:
        return 0.0
    r = float((np.abs(avg[ph]) ** 2).sum()) / tot
    c = float(np.mean(1.0 / cnt[ph]))
    return float(np.clip((r - c) / (1 - c), 0.0, 1.0))


def _plf(x: np.ndarray, T: int, fs: float, A: np.ndarray | None = None) -> np.ndarray:
    """``tfm_series._plf`` with the step's |x| taken once."""
    L = max(1, int(round(0.25 * fs)))
    e = _step(x, np.abs(x) if A is None else A, L)
    k = max(1, int(round(0.5 * fs)))
    e = np.convolve(np.nan_to_num(e), np.ones(k) / k, "same")
    n = len(e); num = np.zeros(n); den = np.zeros(n)
    for j in (-2, -1, 1, 2):
        s = j * T
        i = np.arange(max(0, -s), min(n, n - s))
        if len(i) == 0:
            continue
        a, b = e[i], e[i + s]; w = a + b
        num[i] += w * (1 - np.abs(a - b) / (w + 1e-12)); den[i] += w
    sec = ((np.arange(n) + L) / fs).astype(int); ok = (den > 0) & (sec < N)
    ps = np.bincount(sec[ok], num[ok], N); pd_ = np.bincount(sec[ok], den[ok], N)
    out = np.full(N, np.nan)
    out[pd_ > 0] = ps[pd_ > 0] / pd_[pd_ > 0]
    return out


def fast_subspace(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``tfm_phys._fast_subspace`` with the 1 s moving average from a running sum (the grid is mean-removed first,
    as there)."""
    xz = np.nan_to_num(x)
    mu = xz.mean(0)
    z = xz - mu
    k = max(1, int(round(E.HIGHPASS * fs)))
    if k > len(z):
        from backend.tfm_phys import _fast_subspace
        return _fast_subspace(x, fs)
    hp = z - _ma(z, k)
    V = np.linalg.eigh(hp.conj().T @ hp / len(hp))[1][:, -E.FAN_DIRECTIONS:]
    return z, V, mu


def _edges(starts: np.ndarray, win: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut points between window edges, and for each window the first and one-past-last stretch it covers."""
    b = np.unique(np.r_[0, starts, starts + win, n])
    b = b[(b >= 0) & (b <= n)]
    return b, np.searchsorted(b, starts), np.searchsorted(b, starts + win)


# ----------------------------------------------------------------------------------------------- decode
def decode2(path: Path, index: mtk.MTKIndex, ids: np.ndarray) -> np.ndarray:
    """``mtk.decode_complex(path, index, ids)`` (both slots, nulls filled) for full-width frames, without the
    NaN-filled staging arrays and copies: the fftshift is folded into the byte gather, the 14-bit values are
    sign-extended by shifts and written straight into the complex output, the structural nulls are filled in place
    with the same expression. Integer arithmetic, so the result is bit-identical."""
    width = index.num_subcarriers
    if not (np.asarray(index._bins)[ids] == width).all():
        return mtk.decode_complex(path, index, ids)
    perm = np.fft.fftshift(np.arange(width))
    cols = (2 * perm[:, None] + np.arange(2)[None, :]).ravel()
    mm = np.memmap(path, dtype="<u1", mode="r")
    out = np.empty((len(ids), 2, width), np.complex128)
    for slot in range(2):
        r_off = index._real_off[ids, slot]
        i_off = index._imag_off[ids, slot]
        present = (r_off >= 0) & (i_off >= 0)
        rows = np.flatnonzero(present)
        out[~present, slot] = np.nan
        if not rows.size:
            continue
        z = np.empty((rows.size, width), np.complex128)
        for off, part in ((r_off, z.real), (i_off, z.imag)):
            raw = np.asarray(mm[off[rows, None] + cols[None, :]]).view("<u2")
            part[...] = (raw << 2).view(np.int16) >> 2
        structural = np.all(z == 0, axis=0)
        for start, stop in mtk._null_runs(structural):
            left, right = start - 1, stop
            if stop - start > mtk.MAX_NULL_RUN or left < 0 or right >= width:
                z[:, start:stop] = np.nan
                continue
            span = stop - start + 1
            for k in range(start, stop):
                mu = (k - left) / span
                z[:, k] = z[:, left] * (1 - mu) + z[:, right] * mu
        out[rows, slot] = z
    del mm
    return out


def load_grid(path: Path, index: mtk.MTKIndex, t0: float, t1: float) -> dict[str, Any]:
    """What ``af8_env1.load_chains`` + the ratio give the series, without copying the (frames, 2, live) tensor:
    chain 1 on every bin with the non-live ones zeroed (the gating's input), both chains on the 31 read bins, the
    kept frames' times, the live mask and bin count. Same frames (chain 0 non-zero tested as != 0)."""
    t = np.asarray(index.times, dtype=float)
    tr = t - t[0]
    sel = framediff._uniform_selection(index, None, None)
    ids = np.flatnonzero(np.asarray(sel["mask"], bool) & (tr >= t0) & (tr < t1))
    if ids.size < 2:
        raise ValueError(f"fewer than 2 frames in [{t0:g}, {t1:g}) s once the set was made uniform")
    D = decode2(path, index, ids)
    fin = np.isfinite(D)
    live = fin.all(axis=1).mean(axis=0) > af8.LIVE_FRACTION
    rows = np.flatnonzero(fin[:, :, live].all(axis=(1, 2)) & (D[:, 0, :] != 0)[:, live].all(axis=1))
    p8 = np.flatnonzero(live)[::af8.STRIDE]
    full1 = D[rows, 1, :]
    full1[:, ~live] = 0
    return {"full1": full1, "h0": D[rows[:, None], 0, p8[None, :]], "h1": full1[:, p8], "tt": t[ids][rows],
            "live": live, "n_bins": D.shape[2]}


# ----------------------------------------------------------------------------------------------- reflection
_GATE_TAB: dict[tuple[int, bytes], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def gated8(full1: np.ndarray, live: np.ndarray) -> np.ndarray:
    """``af8_env1.direct_path_gated(H, live, n_bins)[:, ::af8.STRIDE]`` (full1 = chain 1 on every bin, non-live 0):
    the FFT of the IFFT is the input, so the gated response's 31 read bins are chain 1 minus the zeroed taps' DFT,
    e^{-2 pi i p pk / L} x (taps @ e^{-2 pi i p o / L}), instead of an FFT of the whole response."""
    n_bins = full1.shape[1]
    L = n_bins * E.GATE_PAD
    h = np.fft.ifft(full1, n=L, axis=1)
    pk = (h.real * h.real + h.imag * h.imag).argmax(axis=1)
    p8 = np.flatnonzero(live)[::af8.STRIDE]
    key = (L, p8.tobytes())
    if key not in _GATE_TAB:
        R = int(np.ceil(E.GATE_TAPS * E.GATE_PAD))
        off = np.arange(-R, R + 1)
        off = off[np.abs(off) < E.GATE_TAPS * E.GATE_PAD]
        Eo = np.exp(-2j * np.pi * ((off[:, None] * p8[None, :]) % L) / L)
        tab = np.exp(-2j * np.pi * ((np.arange(L)[:, None] * p8[None, :]) % L) / L)
        _GATE_TAB.clear()
        _GATE_TAB[key] = (off, Eo, tab)
    off, Eo, tab = _GATE_TAB[key]
    taps = h[np.arange(len(h))[:, None], (pk[:, None] + off[None, :]) % L]
    return full1[:, p8] - tab[pk] * (taps @ Eo)


# ----------------------------------------------------------------------------------------------- FarSense
_DFT_TAB: dict[bytes, np.ndarray] = {}


def _phase_tab(k: np.ndarray, n: int) -> np.ndarray:
    """[cos; sin] of 2 pi (k m mod F) / F for m < n: the band rows of the F-point DFT at each sample's absolute index."""
    key = k.tobytes()
    tab = _DFT_TAB.get(key)
    if tab is None or tab.shape[1] < n:
        m = np.arange(max(n, 4096))
        ph = 2.0 * np.pi * ((k[:, None] * m[None, :]) % FS_FFT) / FS_FFT
        tab = np.concatenate([np.cos(ph), np.sin(ph)])
        _DFT_TAB.clear()
        _DFT_TAB[key] = tab
    return tab


def _first_peak(r: np.ndarray, fs: float) -> dict[str, float]:
    """``farsense.first_peak(r, fs, FS_BAND, positive_only=True)`` with the lag scan vectorised."""
    n = r.size
    lag_lo = max(1, int(round(fs * 60.0 / FS_BAND[1])))
    lag_hi = min(n - 2, int(round(fs * 60.0 / FS_BAND[0])))
    out = {"lag": float("nan"), "lag_refined": float("nan"), "height": float("nan")}
    if lag_hi < lag_lo:
        return out
    k = np.arange(lag_lo, lag_hi + 1)
    hit = np.flatnonzero((r[k] > r[k - 1]) & (r[k] >= r[k + 1]) & (r[k] > 0))
    if not hit.size:
        return out
    k = int(k[hit[0]])
    env = 1.0 - np.arange(k - 1, k + 2) / n
    a, b, c = r[k - 1: k + 2] / env
    denom = a - 2.0 * b + c
    refined = k + (0.5 * (a - c) / denom if denom < 0 else 0.0)
    out.update(lag=float(k), lag_refined=float(refined), height=float(r[k]))
    return out


def combine_autocorrelations(pattern: np.ndarray, bnr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``farsense.combine_autocorrelations(pattern, bnr, keep_fraction=FS_KEEP)``: the BNR-weighted sum of the
    selected columns' normalised autocorrelations, as one inverse transform of the weighted power spectra (linear),
    each column's lag-0 value read from its own spectrum."""
    eps = float(np.nanmax(bnr)) if bnr.size else 0.0
    selected = np.isfinite(bnr) & (bnr > 0) & (bnr > FS_KEEP * eps)
    win = pattern.shape[0]
    if not selected.any():
        return np.zeros(win), selected
    x = np.nan_to_num(pattern[:, selected])
    n2 = 1
    while n2 < 2 * win:
        n2 <<= 1
    sp = np.fft.rfft(x, n=n2, axis=0)
    pw = (sp * np.conj(sp)).real
    ac0 = (pw[0] + 2.0 * pw[1:-1].sum(axis=0) + pw[-1]) / n2       # irfft at lag 0
    ac0 = np.where(ac0 <= 0, 1.0, ac0)
    return np.fft.irfft(pw @ (bnr[selected] / ac0), n=n2)[:win], selected


def farsense_pk(z: np.ndarray, fab: np.ndarray, fs: float, rpm: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """``tfm_series._fs_windows(z, fab, fs, 0)``: (pk, rpm) per second; rpm only when asked."""
    n, n_sc = z.shape
    win = min(max(farsense.MIN_WINDOW_SAMPLES, int(round(10.0 * fs))), n)
    hop = max(1, int(round(TS.FS_HOP * fs)))
    lag_hi = int(round(fs * 60.0 / FS_BAND[0]))
    if (win < farsense.MIN_WINDOW_SAMPLES or lag_hi + 1 >= win or win > FS_FFT or not np.isfinite(z).all()
            or FS_BAND[1] / 60.0 >= fs / 2.0):
        a, b = TS._fs_windows(z, fab, fs, 0)
        return a, (b if rpm else None)
    smooth = farsense.smooth_ratio(z, fs, seconds=1.0, order=3)
    starts = np.arange(0, n - win + 1, hop)
    kb = farsense.band_bins(fs, FS_BAND, FS_FFT)
    nk = kb.size
    X = np.c_[smooth.real, smooth.imag]
    X = X - X.mean(0)                                         # small numbers into the stretch sums
    tab = _phase_tab(kb, n)
    cut, i0, i1 = _edges(starts, win, n)
    seg = [tab[:, cut[j]:cut[j + 1]] @ X[cut[j]:cut[j + 1]] for j in range(len(cut) - 1)]
    m = np.arange(win)
    pw = 2.0 * np.pi * ((kb[:, None] * m[None, :]) % FS_FFT) / FS_FFT
    Dw = np.cos(pw).sum(1) - 1j * np.sin(pw).sum(1)           # the window mean's DFT per unit
    theta = farsense.projection_angles(FS_THETA)
    c, s = np.cos(theta), np.sin(theta)
    cc, ss, cs = c * c, s * s, c * s
    half = (FS_THETA + 1) // 2
    basis = np.stack([cc[:half], ss[:half], cs[:half]]).astype(np.float32)
    frac = fractional_motion(z) if rpm else None
    nw = starts.size
    unknown = np.zeros(nw, bool)
    acf = np.full(nw, np.nan)
    rp = np.full(nw, np.nan)
    for i, st in enumerate(starts):
        st = int(st)
        if fab[st:st + win].mean() > farsense.MAX_GAP_FRACTION:
            unknown[i] = True
            continue
        P = seg[i0[i]].copy()
        for j in range(i0[i] + 1, i1[i]):
            P += seg[j]
        ps = 2.0 * np.pi * ((kb * st) % FS_FFT) / FS_FFT
        xw = X[st:st + win]
        mu = xw.mean(0)
        F = (P[:nk] - 1j * P[nk:]) * (np.cos(ps) + 1j * np.sin(ps))[:, None] - Dw[:, None] * mu[None, :]
        a, b = F[:, :n_sc], F[:, n_sc:]
        xc = xw - mu
        x, y = xc[:, :n_sc], xc[:, n_sc:]
        aa = np.abs(a) ** 2
        bb = np.abs(b) ** 2
        ab = (a * np.conj(b)).real
        coef = np.stack([aa, bb, 2.0 * ab], axis=-1).reshape(-1, 3).astype(np.float32)
        energy = (coef @ basis).reshape(nk, n_sc, half)
        peak_half = energy.max(axis=0).astype(float)
        peak = np.concatenate([peak_half, peak_half[:, : FS_THETA - half]], axis=1)
        exx = (x * x).sum(axis=0)[:, None]
        eyy = (y * y).sum(axis=0)[:, None]
        exy = (x * y).sum(axis=0)[:, None]
        total = exx * cc[None, :] + eyy * ss[None, :] + 2.0 * exy * cs[None, :]
        one_sided = FS_FFT * total / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            bnr_all = np.where(one_sided > 0, peak / one_sided, 0.0)
        bnr_all[~np.isfinite(bnr_all)] = 0.0
        best = np.argmax(bnr_all, axis=1)
        bnr = bnr_all[np.arange(n_sc), best]
        pattern = x * c[best][None, :] + y * s[best][None, :]
        r_msc, selected = combine_autocorrelations(pattern, bnr)
        pk = _first_peak(r_msc, fs)
        wsum = float(bnr[selected].sum())
        pn = pk["height"] / wsum if wsum > 0 else float("nan")
        acf[i] = pn
        if rpm:
            sf = frac[st: st + win - 1]
            level = float(np.nanmedian(sf)) if sf.size and np.isfinite(sf).any() else float("nan")
            if np.isfinite(level) and level <= 1e9 and np.isfinite(pk["lag_refined"]) and pn >= -1.0:
                rp[i] = 60.0 * fs / pk["lag_refined"]
    un = np.repeat(unknown, 2)[: 2 * nw - 1]
    pkv = np.where(un, np.nan, np.repeat(acf, 2)[: 2 * nw - 1]).astype(float)
    cidx = (np.arange(len(pkv)) + 5).astype(int)
    ok = cidx < N
    A, B = np.full(N, np.nan), np.full(N, np.nan)
    A[cidx[ok]] = pkv[ok]
    if not rpm:
        return A, None
    B[cidx[ok]] = np.repeat(rp, 2)[: 2 * nw - 1][ok]
    return A, B


# ----------------------------------------------------------------------------------------------- breathing lines
_CZ: dict[tuple, tuple] = {}


def _czt_setup(W: int, n: int, fs: float) -> tuple:
    """The chirp-z pieces for the n-point DFT of a W-sample series at the reference band's bins only."""
    key = (W, n, fs)
    if key not in _CZ:
        f = np.fft.rfftfreq(n, 1.0 / fs)
        kk = np.flatnonzero((f >= BC.REFERENCE_HZ[0]) & (f <= BC.REFERENCE_HZ[1]))
        k0, K = int(kk[0]), int(kk[-1] - kk[0] + 1)
        M = 1
        while M < 2 * W + K - 2:
            M <<= 1
        t = np.arange(W)
        pre = np.exp(-1j * np.pi * ((t * t) % (2 * n)) / n)                  # e^{-i pi t^2 / n}
        j = np.arange(k0 - (W - 1), k0 + K)                                  # k - t over the needed range
        v = np.zeros(M, complex)
        v[: j.size] = np.exp(1j * np.pi * ((j * j) % (2 * n)) / n)
        k = np.arange(k0, k0 + K)
        post = np.exp(-1j * np.pi * ((k * k) % (2 * n)) / n)
        fk = f[k0:k0 + K]
        band = np.flatnonzero((fk >= BC.BAND_HZ[0]) & (fk <= BC.BAND_HZ[1]))
        if len(_CZ) >= 8:
            _CZ.clear()
        _CZ[key] = (M, pre, np.fft.fft(v), post, fk, band, W - 1)
    return _CZ[key]


def _prominences(Y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """``breath_components.prominence`` of every column of Y at once: (ln prominence, Hz) per column. One detrend
    for all columns; the power at the reference band's bins of the n-point DFT by chirp-z (a convolution through a
    power-of-two FFT) instead of an n-point FFT (n = fs / 0.01 is 4167 = 3^2 x 463 at 41.7 Hz, a slow length)."""
    W, K = Y.shape
    t = np.arange(W)
    Y = (Y - np.polyval(np.polyfit(t, Y, 1), t[:, None])) * np.hanning(W)[:, None]
    n = max(W, int(np.ceil(fs / BC.RESOLUTION_HZ)))
    M, pre, V, post, fk, band, sh = _czt_setup(W, n, fs)
    c = np.fft.ifft(np.fft.fft(Y * pre[:, None], M, axis=0) * V[:, None], axis=0)
    ps = np.abs(post[:, None] * c[sh: sh + fk.size]) ** 2
    i = band[np.argmax(ps[band], axis=0)]
    v = np.empty(K)
    for j in range(K):
        ref = np.abs(fk - fk[i[j]]) > BC.GUARD_HZ
        v[j] = np.log(ps[i[j], j] / (np.median(ps[ref, j]) + 1e-30))
    return v, fk[i]


def bs_lines(z: np.ndarray, fs: float, gram: str = "stretch") -> tuple[np.ndarray, np.ndarray]:
    """``tfm_phys._bs_lines`` on the 31-column grid z (= the full grid's every 8th column): (bs_all, bs_f).
    gram="stretch" sums each window's covariance from the stretches between window edges; "direct" as before."""
    W = int(round(BC.WINDOW_SECONDS * fs)); st = int(round(TS.BS_HOP * fs)); n = len(z)
    a, f = np.full(N, np.nan), np.full(N, np.nan)
    if n < W or z.shape[1] < BC.N_COMPONENTS:
        return a, f
    X = np.c_[z.real, z.imag]
    starts = np.arange(0, n - W + 1, st)
    if gram == "stretch":
        X = X - X.mean(0)
        cut, i0, i1 = _edges(starts, W, n)
        G = [X[cut[j]:cut[j + 1]].T @ X[cut[j]:cut[j + 1]] for j in range(len(cut) - 1)]
    for i, s0 in enumerate(starts):
        c = int(round((s0 + W / 2) / fs))
        if c >= N:
            continue
        x = X[s0:s0 + W]
        mu = x.mean(axis=0)
        Xc = x - mu
        if gram == "stretch":
            C = G[i0[i]].copy()
            for j in range(i0[i] + 1, i1[i]):
                C += G[j]
            C -= W * np.outer(mu, mu)
        else:
            C = Xc.T @ Xc
        _, V = np.linalg.eigh(C)
        v, fr = _prominences(Xc @ V[:, ::-1][:, :BC.N_COMPONENTS], fs)
        a[c] = v.max()
        f[c] = fr[int(np.argmax(v))]
    return a, f


# ----------------------------------------------------------------------------------------------- the series
def series(path: str | Path, t0: float, t1: float, index: mtk.MTKIndex | None = None,
           channels: list[str] | None = None, timings: dict[str, float] | None = None,
           gram: str = "stretch") -> dict[str, Any]:
    """``tfm_phys.series``, faster. Same return value and stage names."""
    want = set(PHYS_CHANNELS if channels is None else channels)
    tm = timings if timings is not None else {}
    clock = [time.perf_counter()]

    def lap(name: str) -> None:
        now = time.perf_counter(); tm[name] = tm.get(name, 0.0) + now - clock[0]; clock[0] = now

    path = Path(path)
    idx = index if index is not None else mtk.MTKIndex(path)
    G = load_grid(path, idx, t0, t1)
    tt = G["tt"]
    g, fab, fs = af8.resample(G["h1"] / G["h0"], tt)
    if not np.isfinite(g).all():                       # never seen (kept frames are finite); keep the old path
        from backend import tfm_phys
        return tfm_phys.series(path, t0, t1, idx, channels, timings)
    lap("decode")
    A = np.abs(g)
    S: dict[str, np.ndarray] = {}
    for nm, lag in (("s025", 0.25), ("s2", 2.0), ("s5", 5.0)):
        L = max(1, int(round(lag * fs))); S[nm] = TS.per_sec(_step(g, A, L), L, fs)
    with np.errstate(divide="ignore", invalid="ignore"):
        S["lgrow"] = np.where((S["s5"] > 0) & (S["s025"] > 0), np.log(S["s5"] / S["s025"]), np.nan)
    T = swing_period(g, A, fs)
    S["sT"] = TS.per_sec(_step(g, A, T), T, fs)
    lap("steps + swing period")
    z, V, mu = fast_subspace(g, fs)
    res = z - (z @ V) @ V.conj().T + mu
    S["plf_res"] = _plf(res, T, fs)
    S["plf_g"] = _plf(g, T, fs, A)
    lap("plf")
    a = swing_alpha(g, T, fs)
    ab = swing_alpha(g, T, fs, band=True)
    w = max(a, ab)
    S["swing_ab"] = np.full(N, ab)
    lap("swing share")
    Ag = sync_average(g, T) if (w != 0 and want & {"pk_w", "bs_w"}) or want & {"pk_u", "bs_u"} else None
    gw = None if w == 0 else g - w * Ag if want & {"pk_w", "bs_w"} else None
    gu = g - Ag + np.nanmean(g, 0) if want & {"pk_u", "bs_u"} else None
    pk, S["rpm"] = farsense_pk(g, fab, fs, rpm=True)
    if "pk_w" in want:
        S["pk_w"] = pk if w == 0 else farsense_pk(gw, fab, fs, rpm=False)[0]
    if "pk_u" in want:
        S["pk_u"] = farsense_pk(gu, fab, fs, rpm=False)[0]
    lap("FarSense")
    if want & {"g15_res", "u_g15v"}:
        gg, _, _ = af8.resample(gated8(G["full1"], G["live"]) / G["h0"], tt)
        L2 = int(round(af8.P90_LAG * fs))
        zg, Vg, mug = fast_subspace(gg, fs)
        resg = zg - (zg @ Vg) @ Vg.conj().T + mug
        Ar = np.abs(resg)
        S["g15_res"] = TS.per_sec(_step(resg, Ar, L2), L2, fs)
        u = resg - a * sync_average(resg, T)
        S["u_g15v"] = TS.per_sec(_med(np.abs(u[L2:] - u[:-L2]) / (Ar[L2:] + Ar[:-L2])), L2, fs)
    lap("reflection (g15_res, u_g15v)")
    bs_all, S["bs_f"] = bs_lines(g, fs, gram)
    if "bs_w" in want:
        S["bs_w"] = bs_all if w == 0 else bs_lines(gw, fs, gram)[0]
    if "bs_u" in want:
        S["bs_u"] = bs_lines(gu, fs, gram)[0]
    lap("breathing lines (bs)")
    return {"S": S, "T": T / fs, "swing_a": a, "swing_ab": ab, "w": w}
