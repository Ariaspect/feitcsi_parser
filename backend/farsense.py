"""A blind copy of FarSense's respiration pipeline, over the MTK CSI ratio.

FarSense -- Zeng, Wu, Xiong, Yi, Gao, Zhang, *Pushing the Range Limit of
WiFi-based Respiration Sensing with CSI Ratio of Two Antennas*, IMWUT 3(3):121,
2019, https://arxiv.org/abs/1907.03994. No code was released; this follows the
paper's Sec. 5.2 and Sec. 6 step for step, and every place the paper leaves a
number unstated is a named constant below with the choice written next to it.

The pipeline, in the paper's order:

1. **CSI ratio** of two antennas per subcarrier (Sec. 6.2). Here that is the
   ratio ``backend.mtk`` already builds -- the AP's two transmit chains seen
   by one receive chain, not two receive chains as in the paper. The
   cancellation is the same one: both streams sit in the same packet on the
   same receiver oscillator, so the random per-packet phase divides out.
2. **Motion gate** (Sec. 6.2): windows with large motion are excluded. The
   paper feeds the ratio to the speed-spectrum estimator of Li et al. 2018;
   this uses the fractional-motion level ``backend.presence`` already
   thresholds, at the same default. A stand-in, and the one part that is not
   a copy.
3. **Savitzky-Golay smoothing** of each subcarrier's ratio (Sec. 6.2). The
   paper gives neither window nor order; see ``SAVGOL_SECONDS``.
4. **Projection with maximal periodicity** (Sec. 5.2, 6.3): for each
   subcarrier, project the complex ratio onto ``cos(theta), sin(theta)`` for
   100 values of ``theta``, and keep the one whose breathing-to-noise ratio
   is largest. BNR is the energy of the largest FFT bin inside 10-37 bpm
   over the energy of the whole spectrum, on a 12 s window zero-padded to
   8192 points.
5. **Rate by autocorrelation** (Sec. 6.4): autocorrelate each subcarrier's
   pattern, sum the autocorrelations weighted by BNR over the subcarriers
   whose BNR exceeds 0.7 of the best, and read the lag of the first peak.

Two departures from the letter of the paper, both forced by the data rather
than chosen:

* The window mean is removed before the BNR is measured. The paper does not
  say so, but with the mean left in the DC bin holds nearly all the energy
  and the criterion would pick the axis *perpendicular to the static vector*
  rather than the axis along the arc -- the opposite of Fig. 13.
* The autocorrelation peak is refined by a parabola through its neighbours.
  The paper reads the integer lag at 100 Hz, where one lag is 0.04 rpm at
  15 rpm. This link runs at ~20 Hz, where one lag is 0.2 rpm at 15 rpm and
  1.5 rpm at 30, coarser than the paper's own 0.5 rpm detection criterion.
  The integer lag is reported alongside.

Two additions, both off by default so the default IS the paper: a zero-phase
high-pass before smoothing (``highpass_hz``) and a floor on the normalised
autocorrelation peak below which no rate is reported (``min_peak``).

Everything here works on a uniform complex grid ``(n_samples, n_subcarriers)``
at ``fs`` Hz, as ``backend.tiles._presence_grid`` produces it.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from backend.presence import autocorr_columns, fractional_motion, live_subcarriers

# Sec. 5.2.2: "the window length of projection is set to 12 seconds".
WINDOW_SECONDS = 15.0        # the paper uses 12; set 2026-09-22
# Not stated; the GUI updates continuously. One second matches the other
# panels' hop so the strips line up.
HOP_SECONDS = 1.0
# Sec. 5.2.2: "the human respiration range (10 bpm to 37 bpm)".
RATE_BAND_RPM = (10.0, 30.0)  # the paper uses (10, 37); set 2026-09-22
# Sec. 6.3: theta from 0 to 2*pi at a step of pi/50 -- 100 candidates. Half of
# them are sign flips of the other half and score the same BNR; the sweep is
# kept whole so the numbers match the paper's, and the first maximum wins.
N_THETA = 200                 # the paper uses 100; set 2026-09-22
# Sec. 5.2.2: "we increase the number of samples to 8192 by means of
# zero-padding".
FFT_SIZE = 8192
# Sec. 6.4.2: "include those sub-carriers whose BNR is larger than 0.7 * eps".
BNR_KEEP_FRACTION = 0.6       # the paper uses 0.7; set 2026-09-22
# Sec. 6.2 names the Savitzky-Golay filter and nothing else. Half a second of
# cubic fit keeps a 0.6 Hz breath (the top of the band) essentially intact
# while taking out per-packet scatter; at the paper's 100 Hz that is a
# 51-sample window, a common choice in their group's later code.
SAVGOL_SECONDS = 1.0          # was 0.5; set 2026-09-22
MIN_PEAK = 0.2                # normalised ACF peak below which no rate is reported; the paper has no such gate
SAVGOL_ORDER = 3
# The gate's threshold, shared with backend.presence.DEFAULT_MOTION_FRAC_HI so
# "large motion" means the same thing on both tabs.
MOTION_FRAC_HI = 0.25
# A window more than this fraction interpolated across a dropout reports
# nothing, as everywhere else in the backend.
MAX_GAP_FRACTION = 0.5
MIN_WINDOW_SAMPLES = 16


def smooth_ratio(
    ratio: np.ndarray,
    fs: float,
    *,
    seconds: float = SAVGOL_SECONDS,
    order: int = SAVGOL_ORDER,
) -> np.ndarray:
    """Savitzky-Golay along time, real and imaginary planes separately.

    The two planes are filtered independently because the filter is linear
    with real coefficients -- that *is* filtering the complex signal, not an
    approximation of it. A column that is NaN throughout stays NaN; the grid
    has no partial NaNs, ``resample_uniform`` interpolates each column over
    its own finite samples.
    """
    ratio = np.asarray(ratio).astype(complex, copy=False)
    if ratio.ndim != 2:
        raise ValueError(f"ratio must be 2-D (n_samples, n_sc), got {ratio.shape}")
    if seconds <= 0:
        return ratio
    win = int(round(seconds * fs))
    if win % 2 == 0:
        win += 1
    floor = order + 2 if (order + 2) % 2 else order + 3
    win = max(win, floor)
    if win > ratio.shape[0]:
        return ratio
    real = savgol_filter(ratio.real, win, order, axis=0, mode="interp")
    imag = savgol_filter(ratio.imag, win, order, axis=0, mode="interp")
    return real + 1j * imag


def highpass(ratio: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth high-pass along time; off at 0.

    Not in the paper. Offered because this link drifts -- gain steps, thermal
    wander, an occupant settling -- and on a 12 s window a drift's energy
    leaks into the 10 rpm edge of the band, where the criterion mistakes it
    for the breath and the autocorrelation reads its slope instead of a
    period. Zero-phase so the lag the rate is read from is not shifted.
    """
    ratio = np.asarray(ratio).astype(complex, copy=False)
    if cutoff_hz <= 0:
        return ratio
    nyq = fs / 2.0
    if cutoff_hz >= 0.95 * nyq:
        raise ValueError(f"a {cutoff_hz:g} Hz high-pass is at Nyquist for {fs:.2f} Hz")
    b, a = butter(2, cutoff_hz / nyq, btype="high")
    if ratio.shape[0] <= 3 * max(len(a), len(b)):
        return ratio
    finite = np.isfinite(ratio).all(axis=0)
    out = ratio.copy()
    out[:, finite] = (
        filtfilt(b, a, ratio.real[:, finite], axis=0)
        + 1j * filtfilt(b, a, ratio.imag[:, finite], axis=0)
    )
    return out


def projection_angles(n_theta: int = N_THETA) -> np.ndarray:
    """``theta = k * 2*pi / n_theta``: the paper's step of pi/50 for 100."""
    if n_theta < 2:
        raise ValueError(f"need at least 2 projection angles, got {n_theta}")
    return np.arange(n_theta, dtype=float) * (2.0 * np.pi / n_theta)


def band_bins(
    fs: float, band_rpm: tuple[float, float], fft_size: int = FFT_SIZE
) -> np.ndarray:
    """Indices of the ``fft_size``-point DFT bins inside the rate band."""
    freqs = np.arange(fft_size // 2 + 1, dtype=float) * fs / fft_size
    lo, hi = band_rpm[0] / 60.0, band_rpm[1] / 60.0
    return np.flatnonzero((freqs >= lo) & (freqs <= hi))


def extract_patterns(
    seg: np.ndarray,
    fs: float,
    *,
    band_rpm: tuple[float, float] = RATE_BAND_RPM,
    n_theta: int = N_THETA,
    fft_size: int = FFT_SIZE,
) -> dict[str, np.ndarray]:
    """Sec. 5.2 and 6.3 for one window: the best projection per subcarrier.

    ``seg`` is ``(win, n_sc)`` complex, already smoothed. Returns ``pattern``
    ``(win, n_sc)`` -- each subcarrier's respiration pattern, mean removed --
    with the ``bnr`` and ``theta`` that produced it, and ``bnr_all``
    ``(n_sc, n_theta)`` for every candidate.

    The projection is linear, so the FFT of every candidate is a combination
    of two FFTs: ``F(a cos t + b sin t) = cos t F(a) + sin t F(b)``. And the
    only bins the criterion reads are the in-band ones, so those are computed
    directly as a DFT at exactly the frequencies the 8192-point FFT would
    have there, and the denominator comes from Parseval in the time domain.
    Same numbers as padding every candidate and transforming it, without
    building a ``(8192, n_sc, 100)`` array per window.

    BNR is peak-bin energy over the ONE-SIDED spectrum's energy. That is the
    convention that reproduces the paper's Fig. 16 (weights of 0.13 for a
    1200-sample window: a pure tone scores ``N / fft_size`` = 0.146 there,
    and half that two-sided).
    """
    seg = np.asarray(seg)
    if seg.ndim != 2:
        raise ValueError(f"seg must be 2-D (win, n_sc), got {seg.shape}")
    win, n_sc = seg.shape
    if win > fft_size:
        raise ValueError(f"a {win}-sample window does not fit the {fft_size}-point FFT")
    k = band_bins(fs, band_rpm, fft_size)
    if k.size == 0:
        raise ValueError(
            f"no {fft_size}-point bin at {fs:.2f} Hz falls inside "
            f"{band_rpm[0]:g}-{band_rpm[1]:g} rpm"
        )

    alive = np.isfinite(seg).any(axis=0)
    seg = seg.astype(complex, copy=True)
    seg[:, alive] -= np.nanmean(seg[:, alive], axis=0, keepdims=True)
    x = np.nan_to_num(seg.real)
    y = np.nan_to_num(seg.imag)

    n = np.arange(win)
    dft = np.exp(-2j * np.pi * np.outer(k, n) / fft_size)   # (n_f, win)
    a = dft @ x                                              # (n_f, n_sc)
    b = dft @ y
    aa = np.abs(a) ** 2
    bb = np.abs(b) ** 2
    ab = (a * np.conj(b)).real

    theta = projection_angles(n_theta)
    c, s = np.cos(theta), np.sin(theta)
    cc, ss, cs = c * c, s * s, c * s
    # theta and theta + pi project to the same candidate negated, and a sign
    # does not change an energy -- so only the first half of the sweep is
    # scored and the second half copies it. Scored in float32 through one
    # matmul: a (n_f * n_sc, 3) @ (3, n_theta / 2) product is the whole
    # (energy per bin, per subcarrier, per angle) table.
    half = (n_theta + 1) // 2
    coef = np.stack([aa, bb, 2.0 * ab], axis=-1).reshape(-1, 3).astype(np.float32)
    basis = np.stack([cc[:half], ss[:half], cs[:half]]).astype(np.float32)
    energy = (coef @ basis).reshape(k.size, n_sc, half)
    peak_half = energy.max(axis=0).astype(float)             # (n_sc, half)
    peak = np.concatenate([peak_half, peak_half[:, : n_theta - half]], axis=1)

    # Parseval: the padded spectrum's energy is fft_size times the window's.
    exx = (x * x).sum(axis=0)[:, None]
    eyy = (y * y).sum(axis=0)[:, None]
    exy = (x * y).sum(axis=0)[:, None]
    total = exx * cc[None, :] + eyy * ss[None, :] + 2.0 * exy * cs[None, :]
    one_sided = fft_size * total / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        bnr_all = np.where(one_sided > 0, peak / one_sided, 0.0)
    bnr_all[~np.isfinite(bnr_all)] = 0.0

    best = np.argmax(bnr_all, axis=1)
    bnr = bnr_all[np.arange(n_sc), best]
    theta_best = theta[best]
    pattern = x * c[best][None, :] + y * s[best][None, :]
    # A dead subcarrier projects to zeros and scored zero; keep it visibly so.
    pattern[:, ~alive] = np.nan
    bnr[~alive] = 0.0
    return {"pattern": pattern, "bnr": bnr, "theta": theta_best, "bnr_all": bnr_all}


def combine_autocorrelations(
    pattern: np.ndarray, bnr: np.ndarray, *, keep_fraction: float = BNR_KEEP_FRACTION
) -> tuple[np.ndarray, np.ndarray]:
    """Sec. 6.4: BNR-weighted sum of the autocorrelations of the good subcarriers.

    Returns ``(r_msc, selected)``; ``r_msc`` is over lags ``0..win-1`` and is
    NOT renormalised, so its scale is the sum of the selected BNRs -- as in
    Eq. 11. ``selected`` is the mask of subcarriers that made the 0.7 cut.
    """
    pattern = np.asarray(pattern, dtype=float)
    bnr = np.asarray(bnr, dtype=float)
    if pattern.ndim != 2 or bnr.shape != (pattern.shape[1],):
        raise ValueError("pattern must be (win, n_sc) with one BNR per subcarrier")
    eps = float(np.nanmax(bnr)) if bnr.size else 0.0
    selected = np.isfinite(bnr) & (bnr > 0) & (bnr > keep_fraction * eps)
    if not selected.any():
        return np.zeros(pattern.shape[0]), selected
    r = autocorr_columns(np.nan_to_num(pattern[:, selected]))
    return (r * bnr[selected][None, :]).sum(axis=1), selected


def first_peak(
    r: np.ndarray,
    fs: float,
    band_rpm: tuple[float, float] = RATE_BAND_RPM,
    *,
    positive_only: bool = False,
) -> dict[str, float]:
    """Lag of the first local maximum of ``r`` inside the rate band.

    The paper reads "the first peak" of the combined autocorrelation; the
    search is confined to the lags the band allows, which is what keeps a
    wobble at lag 2 from being that peak. Returns NaNs when no local maximum
    exists in band -- a monotone autocorrelation has no period to report.

    ``positive_only`` skips local maxima below zero. A negative "peak" is a
    wiggle on the rising slope out of the half-period trough, not a period,
    and on a seated occupant with finger movement it is what the literal
    rule returns for a minute at a time (20260917_202029: reported 32-33 rpm
    at the shortest lag while the spectrum held a steady 15-18 rpm line;
    the first positive maximum reads +0.44 there and empties stay at 0.1).
    The paper's rule is the default; the hybrid detector uses this one.

    ``lag_refined`` fits a parabola through the peak and its neighbours,
    after dividing the three by the biased estimator's ``1 - k/N`` envelope.
    Eq. 10 is the biased estimator, whose taper leans every peak towards lag
    zero: at 15 rpm on a 12 s window that is a lag short by ~1% and a rate
    read 0.2 rpm high, at any sample rate. Correcting three points costs
    none of the noise amplification the unbiased estimator has at long lags.
    """
    r = np.asarray(r, dtype=float)
    n = r.size
    lag_lo = max(1, int(round(fs * 60.0 / band_rpm[1])))
    lag_hi = min(n - 2, int(round(fs * 60.0 / band_rpm[0])))
    out = {"lag": float("nan"), "lag_refined": float("nan"), "height": float("nan")}
    if lag_hi < lag_lo:
        return out
    for k in range(lag_lo, lag_hi + 1):
        if r[k] > r[k - 1] and r[k] >= r[k + 1] and (r[k] > 0 or not positive_only):
            env = 1.0 - np.arange(k - 1, k + 2) / n
            a, b, c = r[k - 1 : k + 2] / env
            denom = a - 2.0 * b + c
            refined = k + (0.5 * (a - c) / denom if denom < 0 else 0.0)
            out.update(lag=float(k), lag_refined=float(refined), height=float(r[k]))
            return out
    return out


def stitch_patterns(
    patterns: list[np.ndarray | None], starts: np.ndarray, win: int, n: int
) -> np.ndarray:
    """One continuous trace from overlapping per-window patterns.

    Each window owns the hop-wide slice at its centre (the first and last
    also own the ends). Every window's pattern is scaled to unit deviation
    and its sign is chosen to agree with the previous window over the
    samples the two share -- the projection fixes an axis but not its
    polarity, so consecutive windows can carry the same chest inverted.
    This is what the paper's scrolling GUI shows: the latest window's
    pattern, continuously.
    """
    out = np.full(n, np.nan)
    prev: np.ndarray | None = None
    prev_start = 0
    m = len(starts)
    for i, start in enumerate(starts):
        p = patterns[i]
        if p is None:
            prev = None
            continue
        p = np.asarray(p, dtype=float)
        scale = float(np.nanstd(p))
        p = (p - np.nanmean(p)) / scale if scale > 0 else p - np.nanmean(p)
        if prev is not None:
            lo = start
            hi = min(prev_start + win, start + win)
            if hi > lo:
                overlap = float(np.nansum(p[: hi - lo] * prev[lo - prev_start : hi - prev_start]))
                if overlap < 0:
                    p = -p
        lo = 0 if i == 0 else start + (win - (start - starts[i - 1])) // 2
        hi = n if i == m - 1 else starts[i + 1] + (win - (starts[i + 1] - start)) // 2
        lo = max(lo, start)
        hi = min(hi, start + win)
        out[lo:hi] = p[lo - start : hi - start]
        prev, prev_start = p, start
    return out


# --------------------------------------------------------------------------- #
#  The pipeline over a grid                                                    #
# --------------------------------------------------------------------------- #


def prepare(
    ratio: np.ndarray,
    fs: float,
    *,
    fabricated: np.ndarray | None = None,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
    band_rpm: tuple[float, float] = RATE_BAND_RPM,
    savgol_seconds: float = SAVGOL_SECONDS,
    savgol_order: int = SAVGOL_ORDER,
    highpass_hz: float = 0.0,
) -> dict[str, Any]:
    """Everything the per-window step needs, computed once for the whole grid.

    Validates the geometry, drops dead subcarriers, filters and smooths the
    series, measures the motion level, and lays out the windows. Kept apart
    from the window loop so that one window can be re-extracted in full for
    a detail view without redoing the sweep over every other.
    """
    ratio = np.asarray(ratio)
    if ratio.ndim != 2:
        raise ValueError(f"ratio must be 2-D (n_samples, n_sc), got {ratio.shape}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")
    if not 0 < band_rpm[0] < band_rpm[1]:
        raise ValueError(f"rate band must satisfy 0 < lo < hi, got {band_rpm}")
    if band_rpm[1] / 60.0 >= fs / 2.0:
        raise ValueError(
            f"{band_rpm[1]:g} rpm is at or above Nyquist for {fs:.2f} Hz"
        )

    n_samples = ratio.shape[0]
    fab = (
        np.zeros(n_samples, dtype=bool)
        if fabricated is None
        else np.asarray(fabricated, dtype=bool)
    )
    if fab.shape != (n_samples,):
        raise ValueError(f"fabricated must be 1-D of length {n_samples}, got {fab.shape}")

    live = live_subcarriers(ratio)
    if not live.any():
        raise ValueError("no subcarrier in this range carries a CSI ratio")
    live_index = np.flatnonzero(live)
    raw = ratio[:, live]

    win = min(max(MIN_WINDOW_SAMPLES, int(round(window_seconds * fs))), n_samples)
    if win < MIN_WINDOW_SAMPLES:
        raise ValueError(
            f"range holds {n_samples} samples at {fs:.2f} Hz, fewer than the "
            f"{MIN_WINDOW_SAMPLES}-sample minimum window"
        )
    hop = max(1, int(round(hop_seconds * fs)))
    lag_lo = int(round(fs * 60.0 / band_rpm[1]))
    lag_hi = int(round(fs * 60.0 / band_rpm[0]))
    if lag_hi + 1 >= win:
        raise ValueError(
            f"a {win / fs:.1f} s window cannot hold one period at {band_rpm[0]:g} rpm "
            f"-- lengthen the window or raise the band floor"
        )

    smooth = smooth_ratio(
        highpass(raw, fs, highpass_hz), fs, seconds=savgol_seconds, order=savgol_order
    )
    # Gate on the raw ratio, as the presence tab does; smoothing would only
    # soften the very steps the gate is looking for.
    frac = fractional_motion(raw)
    starts = np.arange(0, n_samples - win + 1, hop)
    return {
        "fs": float(fs),
        "n_samples": int(n_samples),
        "fabricated": fab,
        "live_index": live_index,
        "smooth": smooth,
        "frac": frac,
        "win": int(win),
        "hop": int(hop),
        "starts": starts,
        "lag_lo": int(lag_lo),
        "lag_hi": int(lag_hi),
        "band_rpm": (float(band_rpm[0]), float(band_rpm[1])),
    }


def window_step(
    prep: dict[str, Any],
    i: int,
    *,
    n_theta: int = N_THETA,
    fft_size: int = FFT_SIZE,
    keep_fraction: float = BNR_KEEP_FRACTION,
    motion_frac_hi: float = MOTION_FRAC_HI,
    max_gap_fraction: float = MAX_GAP_FRACTION,
    min_peak: float = 0.0,
    positive_only: bool = False,
    detail: bool = False,
) -> dict[str, Any]:
    """Sec. 6.3 and 6.4 for window ``i`` of a prepared grid."""
    fs = prep["fs"]
    win = prep["win"]
    start = int(prep["starts"][i])
    stop = start + win
    out: dict[str, Any] = {
        "unknown": bool(prep["fabricated"][start:stop].mean() > max_gap_fraction),
        "motion_level": float("nan"),
        "stationary": False,
        "rpm": float("nan"),
        "lag": float("nan"),
        "acf_peak": float("nan"),
        "acf_peak_norm": float("nan"),
        "bnr_max": float("nan"),
        "n_selected": 0,
        "best_sc": -1,
        "best_theta": float("nan"),
        "bnr": None,
        "best_pattern": None,
        "detail": None,
    }
    seg_frac = prep["frac"][start : stop - 1]
    level = (
        float(np.nanmedian(seg_frac))
        if seg_frac.size and np.isfinite(seg_frac).any()
        else float("nan")
    )
    out["motion_level"] = level
    out["stationary"] = bool(np.isfinite(level) and level <= motion_frac_hi)
    if out["unknown"]:
        return out

    seg = prep["smooth"][start:stop]
    ext = extract_patterns(
        seg, fs, band_rpm=prep["band_rpm"], n_theta=n_theta, fft_size=fft_size
    )
    bnr = ext["bnr"]
    r_msc, selected = combine_autocorrelations(ext["pattern"], bnr, keep_fraction=keep_fraction)
    pk = first_peak(r_msc, fs, prep["band_rpm"], positive_only=positive_only)
    b = int(np.argmax(np.nan_to_num(bnr)))
    # Eq. 11 sums BNR-weighted autocorrelations without renormalising, so the
    # peak's height scales with how many subcarriers made the cut. Divided by
    # the weights it is the weighted mean autocorrelation at the peak lag, in
    # -1..1 -- the number a reader can judge.
    weight_sum = float(bnr[selected].sum())
    peak_norm = pk["height"] / weight_sum if weight_sum > 0 else float("nan")

    out.update(
        lag=pk["lag"],
        acf_peak=pk["height"],
        acf_peak_norm=peak_norm,
        bnr_max=float(np.nanmax(bnr)) if np.isfinite(bnr).any() else float("nan"),
        n_selected=int(selected.sum()),
        best_sc=int(prep["live_index"][b]),
        best_theta=float(ext["theta"][b]),
        bnr=bnr,
        best_pattern=ext["pattern"][:, b],
    )
    if out["stationary"] and np.isfinite(pk["lag_refined"]) and peak_norm >= min_peak:
        out["rpm"] = 60.0 * fs / pk["lag_refined"]

    if detail:
        iq = seg[:, b] - np.nanmean(seg[:, b])
        out["detail"] = {
            "index": int(i),
            "start_s": float(start / fs),
            "t_s": (start + np.arange(win)) / fs,
            "best_sc": int(prep["live_index"][b]),
            "best_theta": float(ext["theta"][b]),
            "iq": np.stack([iq.real, iq.imag], axis=1),
            "pattern": ext["pattern"][:, b],
            "acf": r_msc,
            "lag_lo": int(prep["lag_lo"]),
            "lag_hi": int(prep["lag_hi"]),
            "lag": pk["lag"],
            "sc_index": prep["live_index"],
            "bnr": bnr,
            "theta": ext["theta"],
            "selected": selected,
        }
    return out


def farsense_windows(
    ratio: np.ndarray,
    fs: float,
    *,
    fabricated: np.ndarray | None = None,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
    band_rpm: tuple[float, float] = RATE_BAND_RPM,
    n_theta: int = N_THETA,
    fft_size: int = FFT_SIZE,
    keep_fraction: float = BNR_KEEP_FRACTION,
    savgol_seconds: float = SAVGOL_SECONDS,
    savgol_order: int = SAVGOL_ORDER,
    highpass_hz: float = 0.0,
    motion_frac_hi: float = MOTION_FRAC_HI,
    max_gap_fraction: float = MAX_GAP_FRACTION,
    min_peak: float = 0.0,
    detail_index: int | None = None,
) -> dict[str, Any]:
    """The whole pipeline over a uniform grid, one result per window.

    ``rpm`` is reported for every stationary window that has an in-band
    autocorrelation peak -- the paper displays the rate whenever the target is
    stationary and offers no confidence beyond that. ``acf_peak`` (the height
    of the peak, in units of summed BNR), ``acf_peak_norm`` (the same in
    -1..1) and ``bnr_max`` are reported so a reader can judge; ``min_peak``
    blanks rates whose normalised peak is below it, and defaults to the
    paper's behaviour of blanking nothing.

    ``detail_index`` asks for one window in full: its best subcarrier's I/Q
    trajectory, every subcarrier's pattern and BNR, the combined
    autocorrelation and where the peak was read.
    """
    prep = prepare(
        ratio, fs,
        fabricated=fabricated, window_seconds=window_seconds, hop_seconds=hop_seconds,
        band_rpm=band_rpm, savgol_seconds=savgol_seconds, savgol_order=savgol_order,
        highpass_hz=highpass_hz,
    )
    step = dict(
        n_theta=n_theta, fft_size=fft_size, keep_fraction=keep_fraction,
        motion_frac_hi=motion_frac_hi, max_gap_fraction=max_gap_fraction,
        min_peak=min_peak,
    )
    result = _run(prep, step, detail_index)
    result["params"] = {
        "window_seconds": float(window_seconds),
        "hop_seconds": float(hop_seconds),
        "band_rpm": (float(band_rpm[0]), float(band_rpm[1])),
        "n_theta": int(n_theta),
        "fft_size": int(fft_size),
        "keep_fraction": float(keep_fraction),
        "savgol_seconds": float(savgol_seconds),
        "savgol_order": int(savgol_order),
        "highpass_hz": float(highpass_hz),
        "motion_frac_hi": float(motion_frac_hi),
        "max_gap_fraction": float(max_gap_fraction),
        "min_peak": float(min_peak),
    }
    return result


def _run(prep: dict[str, Any], step: dict[str, Any], detail_index: int | None) -> dict[str, Any]:
    """Every window of a prepared grid, assembled into series."""
    fs = prep["fs"]
    starts = prep["starts"]
    win = prep["win"]
    n_win = starts.size
    n_live = prep["live_index"].size

    unknown = np.zeros(n_win, dtype=bool)
    stationary = np.zeros(n_win, dtype=bool)
    motion_level = np.full(n_win, np.nan)
    rpm = np.full(n_win, np.nan)
    lag = np.full(n_win, np.nan)
    acf_peak = np.full(n_win, np.nan)
    acf_peak_norm = np.full(n_win, np.nan)
    bnr_max = np.full(n_win, np.nan)
    n_selected = np.zeros(n_win, dtype=int)
    best_sc = np.full(n_win, -1, dtype=int)
    best_theta = np.full(n_win, np.nan)
    bnr_map = np.full((n_live, n_win), np.nan)
    best_patterns: list[np.ndarray | None] = [None] * n_win
    detail = None

    for i in range(n_win):
        w = window_step(prep, i, detail=(detail_index == i), **step)
        unknown[i] = w["unknown"]
        stationary[i] = w["stationary"]
        motion_level[i] = w["motion_level"]
        rpm[i] = w["rpm"]
        lag[i] = w["lag"]
        acf_peak[i] = w["acf_peak"]
        acf_peak_norm[i] = w["acf_peak_norm"]
        bnr_max[i] = w["bnr_max"]
        n_selected[i] = w["n_selected"]
        best_sc[i] = w["best_sc"]
        best_theta[i] = w["best_theta"]
        if w["bnr"] is not None:
            bnr_map[:, i] = w["bnr"]
        best_patterns[i] = w["best_pattern"]
        if w["detail"] is not None:
            detail = w["detail"]

    stitched = stitch_patterns(best_patterns, starts, win, prep["n_samples"])
    stitched[prep["fabricated"]] = np.nan

    return {
        "time_s": (starts + win / 2.0) / fs,
        "stationary": stationary,
        "motion_level": motion_level,
        "unknown": unknown,
        "rpm": rpm,
        "lag": lag,
        "acf_peak": acf_peak,
        "acf_peak_norm": acf_peak_norm,
        "bnr_max": bnr_max,
        "n_selected": n_selected,
        "best_sc": best_sc,
        "best_theta": best_theta,
        "sc_index": prep["live_index"],
        "bnr_map": bnr_map,
        "pattern_t": np.arange(prep["n_samples"]) / fs,
        "pattern": stitched,
        "detail": detail,
        "win": int(win),
        "hop": int(prep["hop"]),
        "window_seconds": float(win / fs),
        "fs_hz": float(fs),
        "lag_lo": int(prep["lag_lo"]),
        "lag_hi": int(prep["lag_hi"]),
    }


# --------------------------------------------------------------------------- #
#  Over a capture                                                              #
# --------------------------------------------------------------------------- #

# The sweep over a five-minute capture costs a few seconds; a click on the
# tab asking for one window in full must not cost that again. The prepared
# grid and the series are kept for the last few (capture, range, parameter)
# combinations, and a detail request re-extracts one window from them.
_CACHE_SIZE = 6
_cache: OrderedDict[tuple, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = OrderedDict()
_cache_lock = Lock()


def reset_cache() -> None:
    with _cache_lock:
        _cache.clear()


def compute_farsense(
    path: Path,
    t0: float,
    t1: float,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
    detail_t: float | None = None,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
    band_rpm: tuple[float, float] = RATE_BAND_RPM,
    n_theta: int = N_THETA,
    fft_size: int = FFT_SIZE,
    keep_fraction: float = BNR_KEEP_FRACTION,
    savgol_seconds: float = SAVGOL_SECONDS,
    savgol_order: int = SAVGOL_ORDER,
    highpass_hz: float = 0.0,
    motion_frac_hi: float = MOTION_FRAC_HI,
    max_gap_fraction: float = MAX_GAP_FRACTION,
    min_peak: float = 0.0,
) -> dict[str, Any]:
    """Decode a capture range onto the uniform ratio grid and run the pipeline.

    Same decode as the presence and Doppler panels -- ``tiles._presence_grid``
    -- so a dropout, a MIMO filter or the interpolation choice mean the same
    thing on this tab as on every other. Window centres and the pattern's
    time axis come back on the capture's own clock.

    ``detail_t`` names a time in seconds; the window whose centre is nearest
    is returned in full under ``detail``.
    """
    from backend.tiles import _presence_grid

    path = Path(path)
    st = path.stat()
    key = (
        str(path.resolve()), st.st_size, st.st_mtime_ns, float(t0), float(t1),
        mimo, source_mac, bool(interpolate),
        float(window_seconds), float(hop_seconds), tuple(float(v) for v in band_rpm),
        int(n_theta), int(fft_size), float(keep_fraction), float(savgol_seconds),
        int(savgol_order), float(highpass_hz), float(motion_frac_hi),
        float(max_gap_fraction), float(min_peak),
    )
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)

    if hit is None:
        grid, fabricated, fs, grid_times, times, times_all, n_no_ratio = _presence_grid(
            path, t0, t1, mimo=mimo, source_mac=source_mac, interpolate=interpolate,
        )
        prep = prepare(
            grid, fs,
            fabricated=fabricated, window_seconds=window_seconds, hop_seconds=hop_seconds,
            band_rpm=band_rpm, savgol_seconds=savgol_seconds, savgol_order=savgol_order,
            highpass_hz=highpass_hz,
        )
        step = dict(
            n_theta=n_theta, fft_size=fft_size, keep_fraction=keep_fraction,
            motion_frac_hi=motion_frac_hi, max_gap_fraction=max_gap_fraction,
            min_peak=min_peak,
        )
        series = _run(prep, step, None)
        origin = float(grid_times[0])
        series["time_s"] = origin + series["time_s"]
        series["pattern_t"] = origin + series["pattern_t"]
        series["frames_used"] = int(times.size)
        series["frames_without_ratio"] = int(n_no_ratio)
        series["t_min"] = float(times_all[0]) if times_all.size else 0.0
        series["t_max"] = float(times_all[-1]) if times_all.size else 0.0
        series["origin"] = origin
        series["params"] = {
            "window_seconds": float(window_seconds),
            "hop_seconds": float(hop_seconds),
            "band_rpm": (float(band_rpm[0]), float(band_rpm[1])),
            "n_theta": int(n_theta),
            "fft_size": int(fft_size),
            "keep_fraction": float(keep_fraction),
            "savgol_seconds": float(savgol_seconds),
            "savgol_order": int(savgol_order),
            "highpass_hz": float(highpass_hz),
            "motion_frac_hi": float(motion_frac_hi),
            "max_gap_fraction": float(max_gap_fraction),
            "min_peak": float(min_peak),
        }
        hit = (prep, step, series)
        with _cache_lock:
            _cache[key] = hit
            while len(_cache) > _CACHE_SIZE:
                _cache.popitem(last=False)

    prep, step, series = hit
    result = dict(series)
    result["detail"] = None
    if detail_t is not None and series["time_s"].size:
        idx = int(np.argmin(np.abs(series["time_s"] - float(detail_t))))
        w = window_step(prep, idx, detail=True, **step)
        d = w["detail"]
        if d is not None:
            origin = series["origin"]
            d = dict(d)
            d["t_s"] = origin + d["t_s"]
            d["start_s"] = origin + d["start_s"]
            result["detail"] = d
    result.pop("origin", None)
    return result
