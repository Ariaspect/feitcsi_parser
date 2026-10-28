"""Features from K. J. Ray Liu's group (Origin Research / University of Maryland),
computed per unit for the classifier tab -- display only.

None of these is an input of any score, rule or model here: they are shown
beside the bank so they can be read against the camera and the other columns
before anyone decides to train on them. Each is computed on the same complex
ratio ``r = H1/H0`` the bank reads (``classifier.ratio_level``), at the
capture's own frame times, with lags counted in frames of the median interval.

1. **Short-lag periodicity** -- Wi-MoID's ACF peak and valley means (Zhu, Hu,
   Wang, Wu, Zeng, Liu, IEEE IoT Journal 2024; dissertation, UMD 2023, s3.3.1.2),
   plus the peak spacing of the deployment paper (Zhu et al., arXiv 2506.04322).
   Per 6 s block, the autocorrelation of the detrended complex ratio,
   maximal-ratio combined over subcarriers, at lags up to 1.5 s: a machine
   repeats itself there, breathing (2-6 s lags) is left to FarSense.
2. **Steadiness** -- Wi-MoID's motion-statistic variance. Two readings: the
   spread of the bank's own psi-hat over the unit's 2 s windows, and the spread
   of the per-second 2 s-lag step relative to its median.
3. **Speed from the autocorrelation** -- WiSpeed (Zhang, Chen, Wang, Liu, IEEE
   IoT Journal 2018) as Wi-MoID uses it: under rich scattering the combined
   ACF of the channel's moving part is J0(k v tau), so its first minimum sits
   at k v tau = 3.832. Per 1 s block, weighted by the size of the block's own
   lag-1 statistic so quiet blocks hardly vote. At ~42 Hz a walking person (1.3 m/s)
   puts that minimum about one frame from zero, so only slow movers -- a robot
   vacuum at ~0.3 m/s -- resolve; the share of blocks at the first lag is
   reported as "too fast to resolve".
4. **Frequency coherence width** -- the proximity feature (Hu, Ozturk, Wang,
   Wu, Zhang, Liu, ICASSP 2021 / IEEE IoT Journal 2022; Hu, Zhu, Wang, Liu,
   arXiv 2404.19182): the correlation of the moving part between subcarriers
   1 and 32 bins apart. Liu reads the adjacent value as "how close the motion
   is"; on env3 a fan stayed coherent much wider than a person (0.31-0.71 at
   32 bins against 0.12-0.17, five captures each, 2026-10-28).
5. **The same in the breathing band** (0.1-0.7 Hz) -- Liu tested proximity
   only on walking people; this is the version aimed at a still person in the
   next room.
6. **The previous unit's state** -- Wi-MoID's HMM state machine and the
   deployment paper's temporal confidence carry a verdict across windows; the
   tab shows the previous unit's P(occupied) (``classifier.compute_features``
   sets it; blank on the first unit, and so on every one-minute capture).

Plus the processing change the same papers use: **maximal-ratio combining**
of the per-subcarrier autocorrelation, each weighted by its own lag-1 value
(Wi-MoID eq. 4.4; here by its magnitude, see ``mrc_acf``) -- shown as the
combined lag-1 statistic over 6 s blocks.
"""

from __future__ import annotations

import numpy as np

# The board's link (ASUS_00_5G, channel 48). Captures do not record their
# channel, so the speed column assumes it; a different carrier rescales the
# speed by a constant and leaves its spread alone.
CARRIER_HZ = 5.24e9
WAVELENGTH_M = 299_792_458.0 / CARRIER_HZ
J0_FIRST_MIN = 3.8317                 # first local minimum of the Bessel J0

PERIOD_BLOCK_SECONDS = 6.0            # Wi-MoID's instance length
PERIOD_MAX_LAG_SECONDS = 1.5          # mechanical repetition; breathing lags (2-6 s) are FarSense's
SPEED_BLOCK_SECONDS = 1.0
SPEED_MAX_LAG_SECONDS = 0.5           # first minimum beyond this = slower than ~0.07 m/s
COHERENCE_SPACINGS = (1, 32)          # subcarrier spacings of the coherence columns, bins
BREATH_BAND_HZ = (0.1, 0.7)


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #


def _blocks(n: int, size: int) -> list[slice]:
    size = int(size)
    if size < 4:
        return []
    return [slice(i * size, (i + 1) * size) for i in range(n // size)]


def _has_gap(times: np.ndarray, med: float) -> bool:
    return times.size > 1 and med > 0 and float(np.max(np.diff(times))) > 2.0 * med


def _detrend(X: np.ndarray) -> np.ndarray:
    """Remove the least-squares line per column (complex columns are fine)."""
    T = X.shape[0]
    t = np.arange(T, dtype=float)
    A = np.stack([t, np.ones(T)], axis=1)
    return X - A @ (np.linalg.pinv(A) @ X)


def mrc_acf(block: np.ndarray, max_lag: int) -> tuple[np.ndarray, float]:
    """Combined autocorrelation of one block of the complex ratio.

    Per subcarrier, after a linear detrend: ``rho_k(tau) = Re sum x_t x*_{t+tau}
    / sum |x_t|^2`` for tau = 0..max_lag. Combined over subcarriers with
    weights ``|rho_k(1)|`` -- maximal-ratio combining with each subcarrier's
    own lag-1 value as its SNR estimate (Wi-MoID eq. 4.4). Wi-MoID samples at
    1500 Hz, where that value is always positive; at ~42 Hz a walking person
    already turns it negative (the J0 curve is past its first zero within one
    frame), so the magnitude is the weight. Returns the combined curve (NaN
    when every subcarrier's lag-1 value is zero) and the combined lag-1
    statistic.
    """
    ok = np.isfinite(block).all(axis=0)
    T = block.shape[0]
    max_lag = int(min(max_lag, T - 2))
    if ok.sum() < 2 or max_lag < 2:
        return np.full(max(max_lag, 0) + 1, np.nan), float("nan")
    X = _detrend(block[:, ok])
    n = 1 << int(np.ceil(np.log2(2 * T)))
    F = np.fft.fft(X, n=n, axis=0)
    ac = np.fft.ifft(np.abs(F) ** 2, axis=0).real[: max_lag + 1]   # Re sum x_t x*_{t+tau}
    e = ac[0]
    good = e > 0
    if not good.any():
        return np.full(max_lag + 1, np.nan), float("nan")
    rho = ac[:, good] / e[good]
    w = np.abs(rho[1])
    if w.sum() <= 0:
        return np.full(max_lag + 1, np.nan), float("nan")
    curve = rho @ w / w.sum()
    return curve, float(curve[1])


def _refine(c: np.ndarray, i: int) -> tuple[float, float]:
    """Sub-frame position and height of the extremum of ``c`` at interior index i (parabola through 3 points)."""
    den = c[i - 1] - 2.0 * c[i] + c[i + 1]
    if den == 0:
        return float(i), float(c[i])
    d = float(np.clip(0.5 * (c[i - 1] - c[i + 1]) / den, -0.5, 0.5))
    return i + d, float(c[i] - 0.25 * (c[i - 1] - c[i + 1]) * d)


def _extrema(c: np.ndarray, lo: int) -> tuple[np.ndarray, np.ndarray]:
    """Interior local maxima and minima of ``c`` at indices >= lo."""
    i = np.arange(max(lo, 1), c.size - 1)
    if i.size == 0:
        return i, i
    mx = i[(c[i] > c[i - 1]) & (c[i] >= c[i + 1])]
    mn = i[(c[i] < c[i - 1]) & (c[i] <= c[i + 1])]
    return mx, mn


def weighted_quantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    v = np.asarray(v, float); w = np.asarray(w, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any():
        return float("nan")
    v, w = v[m], w[m]
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, c, v))


def _spread(v: np.ndarray, w: np.ndarray | None = None) -> float:
    """(P75 - P25) / P50 -- a scale-free spread; NaN when the median is not positive."""
    v = np.asarray(v, float)
    if w is None:
        v = v[np.isfinite(v)]
        if v.size < 2:
            return float("nan")
        q25, q50, q75 = np.percentile(v, [25, 50, 75])
    else:
        q25, q50, q75 = (weighted_quantile(v, w, q) for q in (0.25, 0.5, 0.75))
    return float((q75 - q25) / q50) if np.isfinite(q50) and q50 > 0 else float("nan")


# --------------------------------------------------------------------------- #
#  The features                                                               #
# --------------------------------------------------------------------------- #


def periodicity(r: np.ndarray, times: np.ndarray) -> dict[str, float]:
    """Features 1 and the MRC statistic, over one unit.

    Per 6 s block: the combined ACF up to 1.5 s; the mean of its local maxima
    (peaks) and of its local minima (valleys) past lag 1, and the lag of its
    highest local maximum -- heights and lags refined between frames, so a
    fundamental that falls between two frames is not beaten by a harmonic that
    lands on one. Over the unit: medians of the two means, the median
    period and its spread across blocks, and the median combined lag-1 value.
    A block across a frame gap is skipped.
    """
    r = np.asarray(r); times = np.asarray(times, float)
    nan = {"liu_acf_peak": float("nan"), "liu_acf_valley": float("nan"),
           "liu_acf_period": float("nan"), "liu_acf_period_spread": float("nan"),
           "liu_ms_mrc": float("nan")}
    dt = np.diff(times)
    if dt.size < 10:
        return nan
    med = float(np.median(dt))
    fs = 1.0 / med
    L = int(round(PERIOD_MAX_LAG_SECONDS * fs))
    peaks, valleys, periods, ms = [], [], [], []
    for s in _blocks(r.shape[0], round(PERIOD_BLOCK_SECONDS * fs)):
        if _has_gap(times[s], med):
            continue
        c, m1 = mrc_acf(r[s], L)
        if not np.isfinite(m1):
            continue
        ms.append(m1)
        mx, mn = _extrema(c, 2)
        if mx.size:
            ref = [_refine(c, int(i)) for i in mx]
            peaks.append(float(np.mean([h for _, h in ref])))
            periods.append(max(ref, key=lambda x: x[1])[0] * med)
        else:
            peaks.append(float("nan")); periods.append(float("nan"))
        valleys.append(float(np.mean([_refine(c, int(i))[1] for i in mn])) if mn.size else float("nan"))
    if not ms:
        return nan
    with np.errstate(all="ignore"):
        return {
            "liu_acf_peak": float(np.nanmedian(peaks)) if np.isfinite(peaks).any() else float("nan"),
            "liu_acf_valley": float(np.nanmedian(valleys)) if np.isfinite(valleys).any() else float("nan"),
            "liu_acf_period": float(np.nanmedian(periods)) if np.isfinite(periods).any() else float("nan"),
            "liu_acf_period_spread": _spread(np.asarray(periods)),
            "liu_ms_mrc": float(np.median(ms)),
        }


def speed(r: np.ndarray, times: np.ndarray, wavelength_m: float = WAVELENGTH_M) -> dict[str, float]:
    """Feature 3, over one unit.

    Per 1 s block: the combined ACF up to 0.5 s and its first local minimum,
    refined between frames by a parabola; speed = 3.832 lambda / (2 pi tau).
    A minimum at the first lag is "too fast to resolve"; none inside 0.5 s
    reads as the slowest speed the block can show. Over the unit, every block
    is weighted by the magnitude of its combined lag-1 value, so blocks
    without motion (near 0) hardly vote and fast ones (negative) still do: weighted median speed, its spread, and the weighted
    share of too-fast blocks.
    """
    r = np.asarray(r); times = np.asarray(times, float)
    nan = {"liu_speed": float("nan"), "liu_speed_spread": float("nan"), "liu_speed_fast": float("nan")}
    dt = np.diff(times)
    if dt.size < 10:
        return nan
    med = float(np.median(dt))
    fs = 1.0 / med
    L = max(3, int(round(SPEED_MAX_LAG_SECONDS * fs)))
    v, w, fast = [], [], []
    for s in _blocks(r.shape[0], max(L + 3, round(SPEED_BLOCK_SECONDS * fs))):
        if _has_gap(times[s], med):
            continue
        c, m1 = mrc_acf(r[s], L)
        if not np.isfinite(m1):
            continue
        _, mn = _extrema(c, 1)
        if mn.size:
            i = int(mn[0])
            tau = _refine(c, i)[0] * med
            fast.append(1.0 if i == 1 else 0.0)
        else:
            tau = c.size * med
            fast.append(0.0)
        v.append(J0_FIRST_MIN * wavelength_m / (2.0 * np.pi * tau))
        w.append(abs(m1))
    v, w, fast = np.asarray(v), np.asarray(w), np.asarray(fast)
    if not v.size or w.sum() <= 0:
        return nan
    return {
        "liu_speed": weighted_quantile(v, w, 0.5),
        "liu_speed_spread": _spread(v, w),
        "liu_speed_fast": float(np.sum(w * fast) / w.sum()),
    }


def coherence(r: np.ndarray, times: np.ndarray, band_hz: tuple[float, float] | None = None,
              spacings: tuple[int, ...] = COHERENCE_SPACINGS) -> dict[int, float]:
    """Features 4 and 5, over one unit: the mean correlation of the moving part
    between subcarriers ``n`` columns apart.

    The moving part is ``Re(r / mean_t(r) - 1)`` per subcarrier -- the relative
    amplitude change, Liu's CSI-power reading. With ``band_hz`` it is first
    band-passed (FFT mask, frames taken as evenly spaced at the median
    interval). Each column is standardised; the value for spacing n is the
    mean over column pairs (k, k+n) of their correlation. Missing frames sit
    at the column mean.
    """
    r = np.asarray(r); times = np.asarray(times, float)
    out = {n: float("nan") for n in spacings}
    if r.shape[0] < 16 or r.shape[1] < 2:
        return out
    with np.errstate(all="ignore"):
        mean = np.nanmean(r, axis=0)
        d = (r / mean - 1.0).real
    d = d[:, np.isfinite(mean) & (np.abs(mean) > 0)]
    if d.shape[1] < 2:
        return out
    with np.errstate(all="ignore"):
        d = d - np.nanmean(d, axis=0)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    if band_hz is not None:
        dt = np.diff(times)
        fs = 1.0 / float(np.median(dt)) if dt.size and np.median(dt) > 0 else 0.0
        if fs <= 0:
            return out
        F = np.fft.rfft(d, axis=0)
        fr = np.fft.rfftfreq(d.shape[0], 1.0 / fs)
        F[(fr < band_hz[0]) | (fr > band_hz[1])] = 0.0
        d = np.fft.irfft(F, n=d.shape[0], axis=0)
    sd = d.std(axis=0)
    d = d[:, sd > 0] / sd[sd > 0]
    for n in spacings:
        if d.shape[1] > n:
            out[n] = float(np.mean((d[:, :-n] * d[:, n:]).mean(axis=0)))
    return out


def spread_of(values: np.ndarray) -> float:
    """Feature 2's step reading: (P75 - P25) / P50 of a per-second series."""
    return _spread(np.asarray(values, float))


def std_of(values: np.ndarray) -> float:
    """Feature 2's psi-hat reading: the standard deviation over the unit's windows."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(v.std()) if v.size >= 2 else float("nan")


def unit_features(r: np.ndarray, times: np.ndarray, psi_windows: np.ndarray,
                  step_seconds: np.ndarray) -> dict[str, float]:
    """Features 1-5 and the MRC statistic for one unit. Feature 6 needs the
    previous unit and is set by the caller."""
    out: dict[str, float] = {}
    out.update(periodicity(r, times))
    out["liu_ms_spread"] = std_of(psi_windows)
    out["liu_step_spread"] = spread_of(step_seconds)
    out.update(speed(r, times))
    c = coherence(r, times)
    cb = coherence(r, times, BREATH_BAND_HZ)
    out["liu_coh1"], out["liu_coh32"] = c[1], c[32]
    out["liu_coh1_breath"], out["liu_coh32_breath"] = cb[1], cb[32]
    return out


# The tab's columns, appended to classifier.FEATURES. No reference line: none
# of these has a measured operating point, and none feeds a score.
_SRC_WIMOID = "Wi-MoID (Zhu et al., IEEE IoT J 2024)"
FEATURES: list[dict] = [
    {"key": "liu_acf_peak", "label": "Liu1 ACF peaks", "test": "Liu 1", "status": "display only",
     "reference": None, "axis": [-0.2, 1.0], "decimals": 3,
     "description": f"{_SRC_WIMOID}: per 6 s block, the mean of the local maxima past lag 1 of the "
                    "subcarrier-combined (maximal-ratio) autocorrelation of the detrended complex ratio, lags "
                    "up to 1.5 s; median over the unit. High when something repeats itself within 1.5 s.",
     "role": "display only -- not an input of any score",
     "separates": "expected: a fan's or a machine's short repetition from a person's irregular motion (untested here)",
     "not_separates": "a ceiling fan faster than half the frame rate aliases; frame jitter blurs it; breathing repeats at 2-6 s, outside this range"},
    {"key": "liu_acf_valley", "label": "Liu1 ACF valleys", "test": "Liu 1", "status": "display only",
     "reference": None, "axis": [-1.0, 0.2], "decimals": 3,
     "description": f"{_SRC_WIMOID}: the same, mean of the local minima. Deep valleys with high peaks = a strong oscillation.",
     "role": "display only -- not an input of any score",
     "separates": "expected: as the peaks", "not_separates": "as the peaks"},
    {"key": "liu_acf_period", "label": "Liu1 period (s)", "test": "Liu 1", "status": "display only",
     "reference": None, "axis": [0.0, 1.5], "decimals": 2,
     "description": "Lag of the highest local maximum of that autocorrelation (the deployment paper's peak "
                    "spacing, arXiv 2506.04322), median over the unit's 6 s blocks.",
     "role": "display only -- context",
     "separates": "where the repetition sits", "not_separates": "--"},
    {"key": "liu_acf_period_spread", "label": "Liu1 period spread", "test": "Liu 1", "status": "display only",
     "reference": None, "axis": [0.0, 2.0], "decimals": 2,
     "description": "(P75 - P25) / P50 of that lag across the unit's blocks: low when the same period comes back "
                    "every block, as a machine's does.",
     "role": "display only -- not an input of any score",
     "separates": "expected: a fixed-speed machine from a body", "not_separates": "noise blocks pick random lags and read high, like a body"},
    {"key": "liu_ms_mrc", "label": "Liu MRC lag-1", "test": "Liu MRC", "status": "display only",
     "reference": None, "axis": [-0.2, 1.0], "decimals": 3,
     "description": "The lag-1 autocorrelation with each subcarrier weighted by the size of its own lag-1 value (Wi-MoID's "
                    "maximal-ratio combining, eq. 4.4) instead of the plain mean of psi-hat; 6 s blocks, median "
                    "over the unit. Weights by SNR without calibration.",
     "role": "display only -- the processing change, shown as a statistic",
     "separates": "expected: the same as psi-hat with the noisy subcarriers down-weighted",
     "not_separates": "under a fan the strongest subcarriers are the fan's, so it leans towards the fan"},
    {"key": "liu_ms_spread", "label": "Liu2 ψ̂ spread", "test": "Liu 2", "status": "display only",
     "reference": None, "axis": [0.0, 0.5], "decimals": 3,
     "description": f"{_SRC_WIMOID}'s motion-statistic variance: standard deviation of psi-hat (amplitude) over the "
                    "unit's 2 s windows. The bank already has its median and P90; this is its steadiness.",
     "role": "display only -- not an input of any score",
     "separates": "expected: a steady mover (fan, robot) from a bursty person",
     "not_separates": "a still person's breathing is steady too"},
    {"key": "liu_step_spread", "label": "Liu2 step spread", "test": "Liu 2", "status": "display only",
     "reference": None, "axis": [0.0, 3.0], "decimals": 2,
     "description": "(P75 - P25) / P50 of the per-second 2 s-lag step over the unit: the same steadiness, on the "
                    "motion size the range rule reads.",
     "role": "display only -- not an input of any score",
     "separates": "expected: a steady mover from a bursty person",
     "not_separates": "a still person from a steady mover; an empty room reads its noise's spread"},
    {"key": "liu_speed", "label": "Liu3 speed (m/s)", "test": "Liu 3", "status": "display only",
     "reference": None, "axis": [0.0, 2.0], "decimals": 2,
     "description": "WiSpeed / Wi-MoID: per 1 s block, speed = 3.832 lambda / (2 pi tau) at the first minimum tau of "
                    "the combined autocorrelation (the J0(k v tau) model, lambda for 5.24 GHz); median over the unit "
                    "with each block weighted by the size of its own lag-1 statistic.",
     "role": "display only -- not an input of any score",
     "separates": "env3, 3 captures each: robot 0.16-0.18, small and big movement 0.14-0.24, fans 0.09 (the slow swing)",
     "not_separates": "an empty or still unit (~1.05: its blocks are noise, whose minimum lands at the first lag -- read with "
                      "Liu MRC lag-1); walking, whose minimum is about one frame from zero at ~42 Hz; the J0 scale is a model assumption"},
    {"key": "liu_speed_spread", "label": "Liu3 speed spread", "test": "Liu 3", "status": "display only",
     "reference": None, "axis": [0.0, 2.0], "decimals": 2,
     "description": "Weighted (P75 - P25) / P50 of the block speeds: a wheeled robot holds its speed, a body does not.",
     "role": "display only -- not an input of any score",
     "separates": "env3: fans 0.15-0.17 (steady), robot 0.45-0.50, big movement 0.75, small movement 1.48",
     "not_separates": "fast movers, whose speed does not resolve; empty and still units read noise (~0.03)"},
    {"key": "liu_speed_fast", "label": "Liu3 too fast", "test": "Liu 3", "status": "display only",
     "reference": None, "axis": [0.0, 1.0], "decimals": 2,
     "description": "Weighted share of blocks whose first minimum is at the first lag -- moving faster than the frame "
                    "rate can resolve (~1.4 m/s at 42 Hz).",
     "role": "display only -- context",
     "separates": "where the speed column cannot be read",
     "not_separates": "fast motion from no motion: an empty or still unit reads ~1 too (noise has its minimum at the first lag); "
                      "read it with Liu MRC lag-1, which is near 0 for noise"},
    {"key": "liu_coh1", "label": "Liu4 coherence 1 bin", "test": "Liu 4", "status": "display only",
     "reference": None, "axis": [-0.2, 1.0], "decimals": 3,
     "description": "Liu's proximity feature (Hu et al., ICASSP 2021 / IoT J 2022): mean correlation of the moving part "
                    "Re(r / mean - 1) between adjacent subcarriers over the unit. Liu: higher when the motion is closer "
                    "to the device.",
     "role": "display only -- not an input of any score",
     "separates": "expected: near motion from far motion; motion from noise (empty env3 0.67, motion 0.89-1.00)",
     "not_separates": "tested by Liu on walking people only; depends on the room's multipath"},
    {"key": "liu_coh32", "label": "Liu4 coherence 32 bins", "test": "Liu 4", "status": "display only",
     "reference": None, "axis": [-0.2, 1.0], "decimals": 3,
     "description": "The same 32 subcarriers apart: how wide in frequency the change stays coherent. env3, five "
                    "captures each: rotating fan 0.31, fan with a still person 0.71, still person 0.17, small move "
                    "-0.02, empty 0.04.",
     "role": "display only -- not an input of any score",
     "separates": "expected: a fan's wide coherence from a person's narrow one",
     "not_separates": "measured on five captures in one room; the eigen-share feature, its coarse cousin, was dead weight cross-environment"},
    {"key": "liu_coh1_breath", "label": "Liu5 coherence 1 bin, breath", "test": "Liu 5", "status": "display only",
     "reference": None, "axis": [-0.2, 1.0], "decimals": 3,
     "description": "Feature Liu 4 on the 0.1-0.7 Hz band only (env3 still person 0.95, empty 0.70): the proximity "
                    "reading applied to breathing, aimed at a person in the next room.",
     "role": "display only -- not an input of any score",
     "separates": "expected: a breathing person near the link from one through a wall (untested)",
     "not_separates": "Liu never tested proximity on breathing; a fan's slow swing is in this band too"},
    {"key": "liu_coh32_breath", "label": "Liu5 coherence 32 bins, breath", "test": "Liu 5", "status": "display only",
     "reference": None, "axis": [-0.2, 1.0], "decimals": 3,
     "description": "The same, 32 subcarriers apart (env3 still person 0.12, empty 0.03).",
     "role": "display only -- not an input of any score",
     "separates": "expected: as the 1-bin column, on width", "not_separates": "as the 1-bin column"},
    {"key": "liu_prev_p", "label": "Liu6 previous P(occ)", "test": "Liu 6", "status": "display only",
     "reference": None, "axis": [0.0, 1.0], "decimals": 2,
     "description": "The previous unit's P(occupied) in this range -- the state Wi-MoID's HMM and the deployment "
                    "paper's temporal confidence carry from one window to the next. Blank on the first unit, so blank "
                    "on every one-minute capture; read it on a long continuous capture.",
     "role": "display only -- not an input of any score",
     "separates": "expected: a person who sat down after walking in, in continuous sessions",
     "not_separates": "anything on a single one-minute capture"},
]
