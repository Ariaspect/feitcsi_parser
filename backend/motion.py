"""Gross-motion path: how fast the channel is decorrelating.

The question this path answers is "is anything happening", and the answer is
deliberately *not* an amplitude. A difference energy in the link's own units
has to be re-thresholded for every radio and every room -- measured on this
hardware the previous absolute threshold sat at 0.25 while the loudest window
of a real walk-through reached 0.185, so gross motion never registered at all.

What is used instead is the identity behind that difference. For a stationary
series, ``E[|x_t - x_{t-1}|^2] = 2*sigma^2*(1 - rho_1)``, so dividing the
difference energy by twice the variance leaves ``1 - rho_1`` and subtracting
from one leaves the lag-1 correlation itself. That number is dimensionless,
lives in ``[-1, 1]``, and means the same thing everywhere: how much of this
window's channel is a smooth trajectory rather than noise.

Which way it points is worth stating, because it is the opposite of a
difference energy. After static removal an empty room's residual is close to
white, so ``rho_1`` sits near zero. A body moving through adds a large,
band-limited, temporally smooth component, and ``rho_1`` rises toward one. The
score is high when something is happening.

This path cannot see respiration and is not meant to. A first difference has
frequency response ``|2 sin(pi f Ts)|``, which at 0.2 Hz and this capture's
18.12 Hz is 0.035 -- -29 dB. Anything that tried to read a chest here would be
reading the noise that survived instead.
"""

from __future__ import annotations

import numpy as np

# Spec range 1-2 s. Long enough that the variance in the denominator is
# estimated from more than a handful of samples, short enough that a
# walk-through is not averaged into the stillness on either side of it.
MOTION_WINDOW_SECONDS = 1.5
MOTION_HOP_SECONDS = 0.25

# Small movements that are not travel -- a shifted posture, a turned head.
# The spec asks for 1-10 Hz; the upper edge is clamped to Nyquist, which on
# this capture is 9.06 Hz. Clamping rather than silently measuring an empty
# band above it, because a band that is half missing reads as a quieter room.
FIDGET_BAND_HZ = (1.0, 10.0)

MIN_WINDOW_SAMPLES = 4

# A window is blanked once more than this fraction of it was interpolated
# across a dropout. Same constant and same reasoning as backend.doppler and
# backend.presence, so every panel agrees about what counts as a hole.
DEFAULT_MAX_GAP_FRACTION = 0.5


def _window_bounds(
    n_samples: int, fs: float, window_seconds: float, hop_seconds: float
) -> tuple[np.ndarray, int, int]:
    win = int(round(window_seconds * fs))
    if win < MIN_WINDOW_SAMPLES:
        raise ValueError(
            f"a {window_seconds:g} s window is {win} samples at {fs:.2f} Hz, under the "
            f"{MIN_WINDOW_SAMPLES}-sample minimum"
        )
    win = min(win, n_samples)
    hop = max(1, int(round(hop_seconds * fs)))
    starts = np.arange(0, n_samples - win + 1, hop)
    if starts.size == 0:
        raise ValueError("the range is shorter than one window")
    return starts, win, hop


def _blank_mask(
    fabricated: np.ndarray | None,
    n_samples: int,
    starts: np.ndarray,
    win: int,
    max_gap_fraction: float,
) -> np.ndarray:
    """Which windows are mostly invention rather than measurement.

    Interpolating across a dropout produces a perfectly smooth stretch, and
    smooth is exactly what this path calls motion: a bridged hole has a lag-1
    correlation of 1, the strongest possible reading. Measured on
    captures/lg/20260825_185637.bin, windows containing fabricated samples
    scored 0.69-0.79 against 0.57 for clean ones, and 9.8% of the supposedly
    empty calibration stretch was fabricated -- so the hole inflated both the
    score and the floor it was being judged against.
    """
    if fabricated is None:
        return np.zeros(starts.size, dtype=bool)
    fabricated = np.asarray(fabricated, dtype=bool)
    if fabricated.shape != (n_samples,):
        raise ValueError(
            f"fabricated must be 1-D of length {n_samples}, got {fabricated.shape}"
        )
    fractions = np.array([fabricated[s : s + win].mean() for s in starts])
    return fractions > max_gap_fraction


def lag1_correlation(
    sig: np.ndarray,
    fs: float,
    *,
    window_seconds: float = MOTION_WINDOW_SECONDS,
    hop_seconds: float = MOTION_HOP_SECONDS,
    fabricated: np.ndarray | None = None,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-subcarrier lag-1 correlation per window. Returns ``(centres, rho)``.

    The difference is complex. ``|r_t| - |r_{t-1}|`` is identically zero for a
    body at fixed range whose reflection walks the ratio around a circle in
    the complex plane -- the magnitude never changes and the movement is
    invisible. The complex difference is the only form that sees it.

    ``rho`` is ``(n_subcarriers, n_windows)`` and is kept whole: the subcarrier
    axis is the Fourier partner of delay, so which subcarriers decorrelated
    says something about how far away it happened. Folding it is the last
    thing that happens, in ``motion_score``, and never before.
    """
    sig = np.asarray(sig)
    if sig.ndim != 2:
        raise ValueError(f"sig must be 2-D (n_samples, n_sc), got {sig.shape}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")

    n, n_sc = sig.shape
    starts, win, _ = _window_bounds(n, fs, window_seconds, hop_seconds)

    diff_energy = np.abs(np.diff(sig, axis=0)) ** 2

    rho = np.empty((n_sc, starts.size), dtype=float)
    for i, start in enumerate(starts):
        seg = sig[start : start + win]
        # Variance about this window's own mean. The static path is already
        # gone, but a window straddling a step still has one.
        var = np.mean(np.abs(seg - seg.mean(axis=0, keepdims=True)) ** 2, axis=0)
        mean_diff = diff_energy[start : start + win - 1].mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho[:, i] = 1.0 - mean_diff / (2.0 * var)

    # A silent subcarrier has no trajectory to be correlated along; zero is
    # "no evidence of motion", which is the honest reading of no signal.
    rho[~np.isfinite(rho)] = 0.0
    rho[:, _blank_mask(fabricated, n, starts, win, max_gap_fraction)] = np.nan
    centres = (starts + win / 2.0) / fs
    return centres, rho


def motion_score(rho: np.ndarray) -> np.ndarray:
    """Fold the subcarrier axis into one score per window, by median.

    Median rather than mean so that a handful of subcarriers sitting on a
    fading null -- where the ratio is dominated by whatever noise was in the
    denominator -- cannot carry the verdict for the whole array.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2:
        raise ValueError(f"rho must be 2-D (n_sc, n_windows), got {rho.shape}")
    blank = ~np.isfinite(rho).any(axis=0)
    score = np.full(rho.shape[1], np.nan)
    if (~blank).any():
        score[~blank] = np.nanmedian(rho[:, ~blank], axis=0)
    return score


def eigenvalue_ratio(
    sig: np.ndarray,
    fs: float,
    *,
    window_seconds: float = MOTION_WINDOW_SECONDS,
    hop_seconds: float = MOTION_HOP_SECONDS,
    fabricated: np.ndarray | None = None,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """``lambda_1 / sum(lambda)`` of the subcarrier covariance, per window.

    Gross motion drives every subcarrier through one shared waveform seen
    through different geometry, so the covariance collapses toward rank one
    and this ratio approaches 1. Respiration moves them along directions that
    differ per subcarrier and noise is isotropic; both spread the spectrum out.
    That makes this a motion-versus-breathing feature in its own right, rather
    than another way of asking how loud the window was.

    Computed from the Gram matrix of the window rather than the covariance of
    the array. They share every nonzero eigenvalue, and the window is a few
    dozen samples against a few hundred subcarriers -- a 27x27 eigenproblem
    per window instead of 209x209.
    """
    sig = np.asarray(sig)
    if sig.ndim != 2:
        raise ValueError(f"sig must be 2-D (n_samples, n_sc), got {sig.shape}")

    starts, win, _ = _window_bounds(sig.shape[0], fs, window_seconds, hop_seconds)

    ratio = np.empty(starts.size, dtype=float)
    for i, start in enumerate(starts):
        seg = sig[start : start + win]
        seg = seg - seg.mean(axis=0, keepdims=True)
        gram = seg @ seg.conj().T
        eigs = np.linalg.eigvalsh(gram).real
        eigs = np.clip(eigs, 0.0, None)
        total = eigs.sum()
        ratio[i] = eigs[-1] / total if total > 0 else 0.0

    ratio[_blank_mask(fabricated, sig.shape[0], starts, win, max_gap_fraction)] = np.nan
    return (starts + win / 2.0) / fs, ratio


def fidget_energy(
    sig: np.ndarray,
    fs: float,
    *,
    window_seconds: float = MOTION_WINDOW_SECONDS,
    hop_seconds: float = MOTION_HOP_SECONDS,
    band_hz: tuple[float, float] = FIDGET_BAND_HZ,
    fabricated: np.ndarray | None = None,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Mean in-band power per subcarrier over the fidget band.

    This is what separates "in the room and still" from "in the room and
    shifting about": respiration cannot reach 1 Hz and travel does not stop
    there, but a posture change is a short broadband event that lands here.

    Absolute power, not a fraction of the window's total. The fraction form
    is degenerate at this sample rate -- 1 Hz to a 9.06 Hz Nyquist is 89% of
    the available spectrum, so white noise scores 0.92 by construction and
    there is nothing left for a real event to rise above. The absolute form
    needs no such normalisation *here* because the caller has already done
    it: after ``preprocess.normalize_subcarriers`` the signal is in units of
    each subcarrier's own noise sigma, so this number reads directly as
    "how many times the noise floor", and its own floor is calibrated from a
    known-empty stretch the same way the motion score's is.

    Returns the band actually used, which is not always the one requested --
    the upper edge is clamped to Nyquist. Silently measuring an empty band
    above it would read as a quieter room rather than as a narrower band.
    """
    sig = np.asarray(sig)
    if sig.ndim != 2:
        raise ValueError(f"sig must be 2-D (n_samples, n_sc), got {sig.shape}")

    nyquist = fs / 2.0
    lo = max(float(band_hz[0]), 0.0)
    hi = min(float(band_hz[1]), nyquist)
    if hi <= lo:
        raise ValueError(
            f"the fidget band {band_hz} does not fit under this capture's "
            f"{nyquist:.2f} Hz Nyquist"
        )

    starts, win, _ = _window_bounds(sig.shape[0], fs, window_seconds, hop_seconds)
    freqs = np.abs(np.fft.fftfreq(win, d=1.0 / fs))
    in_band = (freqs >= lo) & (freqs <= hi)
    if not in_band.any():
        raise ValueError(
            f"a {win / fs:.2f} s window resolves {fs / win:.2f} Hz, too coarse to hold "
            f"a bin inside {lo:.2f}-{hi:.2f} Hz"
        )
    taper = np.hanning(win)[:, None]
    # Hann halves the coherent gain and costs power; dividing by its own mean
    # square keeps the reading comparable to an untapered one.
    correction = win * float(np.mean(np.hanning(win) ** 2)) * sig.shape[1]

    energy = np.empty(starts.size, dtype=float)
    for i, start in enumerate(starts):
        seg = sig[start : start + win]
        seg = seg - seg.mean(axis=0, keepdims=True)
        power = np.abs(np.fft.fft(seg * taper, axis=0)) ** 2
        energy[i] = power[in_band].sum() / correction

    energy[_blank_mask(fabricated, sig.shape[0], starts, win, max_gap_fraction)] = np.nan
    return (starts + win / 2.0) / fs, energy, (lo, hi)
