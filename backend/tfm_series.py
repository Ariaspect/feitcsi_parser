"""Per-second input series of the Transformer tab's model: 17 channels x 61 s on one 60 s window.

The fifteen channels of the ML experiments' sequence models (ml_lab ``cy_ts_extract2.py``) with the two
value-changing lightweight options the board needs (C1: the ``bs_*`` PCA on every 8th subcarrier; C3: every
FarSense sweep every 2 s, each window held for 2 s, ``bs_*`` every 4 s), plus the two phase-locked-fraction
channels (user 2026-11-10):

* ``s025 s2 s5``  median step at 0.25 / 2 / 5 s, one value per second
* ``pk rpm``      FarSense breathing peak / rate on 10 s windows
* ``pk_diff``     FarSense peak on d = g(t) - g(t - T), T the fan's swing period (``af8.swing_period``)
* ``pk_res``      FarSense peak on the fan-subspace residual (the two strongest high-passed directions removed)
* ``drift``       1 - |<v, v_next>| of the leading eigenvector of consecutive 2 s blocks
* ``ret``         step at T over step at T / 2 (a rotating fan returns: small)
* ``g15``         2 s step of the direct-path-gated signal
* ``bs_*``        breathing prominence of the subcarrier-pattern components on 20 s windows (``breath_components``)
* ``plf_res plf_g``  phase-locked fraction: how much of the 0.25 s motion energy repeats at the same phase of
  the swing (n +- T, n +- 2T), energy-weighted, on the residual and on the full grid -- a fan's bursts come back
  at the same head angle, a person's motion does not

Values are identical to the training arrays (ml_lab ``cy_cvar_C1_bs31+C2b_bank31+C3_hop2.pkl`` and
``cy_plf.pkl``; checked on lg). NaN where a channel has no value in a second. numpy only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend import af8, af8_env1 as E, breath_components as BC, farsense, mtk

N = 61
CH: tuple[str, ...] = ("s025", "s2", "s5", "pk", "rpm", "pk_diff", "pk_res", "drift", "ret", "g15",
                       "bs_all", "bs_r1", "bs_r2", "bs_tap", "bs_f", "plf_res", "plf_g")
FS_HOP = 2.0          # C3: FarSense window every 2 s, held 2 s
BS_HOP = 4.0          # C3: bs window every 4 s
BS_STRIDE = 8         # C1: bs PCA on every 8th subcarrier


def per_sec(v: np.ndarray, t0: float, fs: float) -> np.ndarray:
    sec = ((np.arange(len(v)) + t0) / fs).astype(int)
    ok = np.isfinite(v) & (sec < N) & (sec >= 0)
    s = np.bincount(sec[ok], v[ok], N)
    c = np.bincount(sec[ok], minlength=N)
    out = np.full(N, np.nan)
    out[c > 0] = s[c > 0] / c[c > 0]
    return out


def _step_n(num: np.ndarray, den: np.ndarray, L: int) -> np.ndarray:
    return np.nanmedian(np.abs(num[L:] - num[:-L]) / (np.abs(den[L:]) + np.abs(den[:-L])), axis=1)


def _fs_windows(g: np.ndarray, fab: np.ndarray, fs: float, t_off: float) -> tuple[np.ndarray, np.ndarray]:
    prep = farsense.prepare(g, fs, fabricated=fab, window_seconds=10.0, hop_seconds=FS_HOP, band_rpm=(10.0, 30.0),
                            savgol_seconds=1.0, savgol_order=3, highpass_hz=0.0)
    z = farsense._run(prep, dict(n_theta=200, fft_size=8192, keep_fraction=0.65, motion_frac_hi=1e9,
                                 max_gap_fraction=0.5, min_peak=-1.0, positive_only=True), None)
    n = len(z["time_s"])
    unknown = np.repeat(np.asarray(z["unknown"]), 2)[: 2 * n - 1]            # each window held for 2 s
    acf = np.repeat(np.asarray(z["acf_peak_norm"]), 2)[: 2 * n - 1]
    rpm = np.repeat(np.asarray(z["rpm"], float), 2)[: 2 * n - 1]
    pk = np.where(unknown, np.nan, acf).astype(float)
    c = (t_off + np.arange(len(pk)) + 5).astype(int)
    a, b = np.full(N, np.nan), np.full(N, np.nan)
    ok = c < N
    a[c[ok]] = pk[ok]
    b[c[ok]] = rpm[ok]
    return a, b


def _bs_series(gf: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    gf = gf[:, np.isfinite(gf).all(axis=0)]
    W = int(round(BC.WINDOW_SECONDS * fs)); st = int(round(BS_HOP * fs)); n = len(gf)
    out = {k: np.full(N, np.nan) for k in ("bs_all", "bs_r1", "bs_r2", "bs_tap", "bs_f")}
    if n < W or gf.shape[1] < BC.N_COMPONENTS:
        return out
    cir = np.fft.ifft(gf / (np.abs(gf).mean(axis=0) + 1e-12), axis=1)[:, :BC.N_TAPS]
    g8 = gf[:, ::BS_STRIDE]
    for s0 in range(0, n - W + 1, st):
        x = g8[s0:s0 + W]
        Xc = np.c_[x.real - x.real.mean(axis=0), x.imag - x.imag.mean(axis=0)]
        _, V = np.linalg.eigh(Xc.T @ Xc)
        comps = Xc @ V[:, ::-1][:, :BC.N_COMPONENTS]
        pr = [BC.prominence(comps[:, j], fs) for j in range(BC.N_COMPONENTS)]
        v = np.array([p[0] for p in pr]); c = int(round((s0 + W / 2) / fs))
        if c >= N:
            continue
        out["bs_all"][c], out["bs_r1"][c], out["bs_r2"][c] = v.max(), v[1:].max(), v[2:].max()
        out["bs_f"][c] = pr[int(np.argmax(v))][1]
        out["bs_tap"][c] = max(BC.prominence(BC._leading(cir[s0:s0 + W, t]), fs)[0] for t in range(cir.shape[1]))
    return out


def _plf(x: np.ndarray, T: int, fs: float) -> np.ndarray:
    L = max(1, int(round(0.25 * fs)))
    e = af8._step(x, L)
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


def series(path: str | Path, t0: float, t1: float, index: mtk.MTKIndex | None = None) -> dict[str, np.ndarray | float]:
    """The 17 x 61 series of [t0, t1) s from the capture's first frame, with the swing period T (s)."""
    path = Path(path)
    idx = index if index is not None else mtk.MTKIndex(path)
    H, tt, live, nb = E.load_chains(path, idx, t0, t1)
    r = H[:, 1] / H[:, 0]
    g, fab, fs = af8.resample(r[:, ::af8.STRIDE], tt)
    S: dict[str, np.ndarray] = {}
    for nm, lag in (("s025", 0.25), ("s2", 2.0), ("s5", 5.0)):
        L = max(1, int(round(lag * fs))); S[nm] = per_sec(af8._step(g, L), L, fs)
    S["pk"], S["rpm"] = _fs_windows(g, fab, fs, 0)
    T = af8.swing_period(g, fs)
    S["pk_diff"], _ = _fs_windows(g[T:] - g[:-T], fab[T:] | fab[:-T], fs, T / fs)
    gz = np.nan_to_num(g); mu = gz.mean(0); z = gz - mu; k = max(1, int(round(E.HIGHPASS * fs)))
    hp = z - np.apply_along_axis(lambda c: np.convolve(c, np.ones(k) / k, "same"), 0, z)
    V = np.linalg.eigh(hp.conj().T @ hp / len(hp))[1][:, -E.FAN_DIRECTIONS:]
    res = z - (z @ V) @ V.conj().T + mu
    S["pk_res"], _ = _fs_windows(res, fab, fs, 0)
    nbk = int(round(E.DRIFT_BLOCK * fs)); Am = np.c_[np.ones(nbk), np.arange(nbk)]; vecs = []
    for s in range(0, len(gz) - nbk, nbk):
        zz = gz[s:s + nbk]; zz = zz - Am @ np.linalg.lstsq(Am, zz, rcond=None)[0]
        vecs.append(np.linalg.eigh(zz.conj().T @ zz / nbk)[1][:, -1])
    dr = np.full(N, np.nan)
    for i, (x, y) in enumerate(zip(vecs[:-1], vecs[1:])):
        c = int(round((i + 1) * nbk / fs))
        if c < N:
            dr[c] = 1 - abs(np.vdot(x, y))
    S["drift"] = dr
    sT = per_sec(af8._step(g, T), T, fs); sH = per_sec(af8._step(g, max(1, T // 2)), max(1, T // 2), fs)
    S["ret"] = sT / sH
    rg = E.direct_path_gated(H, live, nb) / H[:, 0]; gg, _, _ = af8.resample(rg[:, ::af8.STRIDE], tt)
    L = int(round(af8.P90_LAG * fs)); S["g15"] = per_sec(_step_n(gg, gg, L), L, fs)
    gf, _, fsf = af8.resample(r, tt)
    S.update(_bs_series(gf, fsf))
    S["plf_res"] = _plf(res, T, fs)
    S["plf_g"] = _plf(g, T, fs)
    return {"X": np.array([S[c] for c in CH], dtype=float), "T": T / fs}
