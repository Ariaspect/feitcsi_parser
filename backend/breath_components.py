"""Object-agnostic breathing (``bs_*``): does ANY component of the ratio carry a clean breathing line?

A fan, a ceiling fan or a robot is a strong, coherent mover: it takes the leading directions of the
change across subcarriers. A person's breathing sits on another spatial pattern, as a narrow line at
9-36 rpm. So the window is split into components and the question is asked of each, without a model
of any particular mover:

* the ratio ``r(t, k) = H1 / H0`` on all live subcarriers (``af8_env1.load_chains``), on its own
  native-rate uniform grid (``af8.resample``; no re-gridding between rates);
* per 20 s window, every 5 s: the top 8 principal components of ``[Re, Im]`` across subcarriers
  (mean removed), each a time series;
* per series: linear detrend, Hann window, periodogram zero-padded to 0.01 Hz; the prominence is
  ``ln(peak power in 0.15-0.6 Hz / median power over 0.05-2 Hz outside +-0.05 Hz of the peak)``;
* ``all`` = the best of the 8, ``r1`` = the best without the leading component (usually the mover),
  ``r2`` = without the leading two; ``tap`` = the same on the 16 leading delay taps (IFFT over
  subcarriers, each tap's leading [Re, Im] direction); ``rate_iqr`` = P75 - P25 of the best
  component's peak frequency over the windows (a real breather holds its rate).

Per unit: P50 and P90 over the windows of each, and the rate spread -- nine columns. No hand-set
thresholds. A person alone: all > r1 > r2 (the person is the leading component); a mover plus a
person: the line survives in r1 and r2. Inputs of the ML tab's human model only (``ml_heads``):
41 + these 9 = 50 (user, 2026-11-03); the anomaly model keeps its 41 -- these cost it held out.
Leading components from ``eigh`` of the 490 x 490 Gram matrix: the same series as an SVD up to sign,
2.5x faster. numpy only, like the board.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WINDOW_SECONDS = 20.0
STEP_SECONDS = 5.0
N_COMPONENTS = 8
N_TAPS = 16
BAND_HZ = (0.15, 0.6)            # 9-36 rpm
REFERENCE_HZ = (0.05, 2.0)
GUARD_HZ = 0.05
RESOLUTION_HZ = 0.01

KEYS = ("bs_all_p50", "bs_all_p90", "bs_r1_p50", "bs_r1_p90", "bs_r2_p50", "bs_r2_p90",
        "bs_tap_p50", "bs_tap_p90", "bs_rate_iqr")

LABELS = {
    "bs_all_p50": ("bs line, any comp. P50", "Median over 20 s windows of the clearest breathing line (ln prominence, 9-36 rpm) in any of the top 8 subcarrier-pattern components."),
    "bs_all_p90": ("bs line, any comp. P90", "90th percentile over 20 s windows of the clearest breathing line in any of the top 8 components."),
    "bs_r1_p50": ("bs line, w/o lead P50", "As 'any comp.' but without the leading component -- usually the strongest mover, a fan or robot: breathing left beside it."),
    "bs_r1_p90": ("bs line, w/o lead P90", "90th percentile of the breathing line without the leading component."),
    "bs_r2_p50": ("bs line, w/o 2 lead P50", "As 'any comp.' but without the leading two components."),
    "bs_r2_p90": ("bs line, w/o 2 lead P90", "90th percentile of the breathing line without the leading two components."),
    "bs_tap_p50": ("bs line, delay taps P50", "The clearest breathing line over the 16 leading delay taps (IFFT over subcarriers). Weak alone (AUC 0.6-0.7)."),
    "bs_tap_p90": ("bs line, delay taps P90", "90th percentile of the delay-tap breathing line."),
    "bs_rate_iqr": ("bs rate spread, Hz", "P75 - P25 over windows of the best component's line frequency: low when one breathing rate holds."),
}
assert tuple(LABELS) == KEYS


def prominence(y: np.ndarray, fs: float) -> tuple[float, float]:
    """``(ln prominence, frequency in Hz)`` of the strongest line in the breathing band of one series."""
    y = np.asarray(y, dtype=float)
    t = np.arange(len(y))
    y = (y - np.polyval(np.polyfit(t, y, 1), t)) * np.hanning(len(y))
    n = max(len(y), int(np.ceil(fs / RESOLUTION_HZ)))
    ps = np.abs(np.fft.rfft(y, n)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
    i = np.flatnonzero(band)[np.argmax(ps[band])]
    ref = (f >= REFERENCE_HZ[0]) & (f <= REFERENCE_HZ[1]) & (np.abs(f - f[i]) > GUARD_HZ)
    return float(np.log(ps[i] / (np.median(ps[ref]) + 1e-30))), float(f[i])


def _leading(z: np.ndarray) -> np.ndarray:
    """The series along the leading [Re, Im] direction of one complex column."""
    X = np.c_[z.real - z.real.mean(), z.imag - z.imag.mean()]
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    return u[:, 0] * s[0]


def from_grid(g: np.ndarray, fs: float) -> dict[str, float]:
    """The nine columns from a uniform complex grid (samples x subcarriers) at ``fs`` Hz; NaN when under one window."""
    g = np.asarray(g)
    g = g[:, np.isfinite(g).all(axis=0)]
    n = g.shape[0]
    W = int(round(WINDOW_SECONDS * fs))
    step = int(round(STEP_SECONDS * fs))
    nan = {k: float("nan") for k in KEYS}
    if n < W or g.shape[1] < N_COMPONENTS or W < 8:
        return nan
    cir = np.fft.ifft(g / (np.abs(g).mean(axis=0) + 1e-12), axis=1)[:, :N_TAPS]
    rec: dict[str, list[float]] = {"all": [], "r1": [], "r2": [], "tap": [], "f": []}
    for s0 in range(0, n - W + 1, step):
        x = g[s0:s0 + W]
        X = np.c_[x.real - x.real.mean(axis=0), x.imag - x.imag.mean(axis=0)]
        _, V = np.linalg.eigh(X.T @ X)
        comps = X @ V[:, ::-1][:, :N_COMPONENTS]
        pr = [prominence(comps[:, j], fs) for j in range(N_COMPONENTS)]
        v = np.array([p[0] for p in pr])
        rec["all"].append(float(v.max()))
        rec["r1"].append(float(v[1:].max()))
        rec["r2"].append(float(v[2:].max()))
        rec["f"].append(pr[int(np.argmax(v))][1])
        rec["tap"].append(max(prominence(_leading(cir[s0:s0 + W, t]), fs)[0] for t in range(cir.shape[1])))
    out: dict[str, float] = {}
    for k in ("all", "r1", "r2", "tap"):
        a = np.asarray(rec[k])
        out[f"bs_{k}_p50"], out[f"bs_{k}_p90"] = float(np.percentile(a, 50)), float(np.percentile(a, 90))
    q75, q25 = np.percentile(np.asarray(rec["f"]), [75, 25])
    out["bs_rate_iqr"] = float(q75 - q25)
    return out


def unit_features(path: Path, t0_rel: float, t1_rel: float) -> dict[str, float]:
    """The nine columns on [t0_rel, t1_rel) seconds from the capture's first frame; NaN where it cannot be read."""
    from backend import af8, af8_env1, mtk

    try:
        path = Path(path)
        H, tt, _, _ = af8_env1.load_chains(path, mtk.MTKIndex(path), float(t0_rel), float(t1_rel))
        g, _, fs = af8.resample(H[:, 1] / H[:, 0], tt)
        return from_grid(g, fs)
    except Exception:        # too short, undecodable, too few frames: blank, not an error for the window
        return {k: float("nan") for k in KEYS}
