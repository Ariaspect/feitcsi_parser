"""Static-occupant path: a chest, its rate, and how much to believe it.

This path answers a different question from ``backend.motion`` and shares
almost none of its machinery. Motion asks whether the channel is moving at
all, on a 1.5 s window, and gets its answer from how fast the channel
decorrelates. Breathing asks whether a few millimetres of chest displacement
are modulating the channel periodically, on a 45 s window, and gets its answer
from where the energy sits in 0.1-0.6 Hz.

Trying to read both from one number does not work, and the reason is
arithmetic rather than taste. A first difference has response
``|2 sin(pi f Ts)|``, which at 0.25 Hz and this capture's 18.12 Hz is 0.043 --
-27 dB. Any respiration reaching a difference-based score arrives buried under
the noise that survived alongside it.

Four things here are easy to get wrong and silent when wrong:

* **The projection.** The chest moves the ratio along one fixed complex
  direction per subcarrier, and that direction is not the magnitude axis. Where
  it is perpendicular to it, ``|r|`` sees only the second-order term and reads
  an empty room.
* **The signs.** Neighbouring subcarriers can carry the same chest with
  opposite sign. Summed unaligned they erase each other, and the erasure
  produces a clean-looking flat trace rather than an error.
* **The weights.** Weighting by amplitude ranks subcarriers by how noisy they
  are. The weight has to be periodicity.
* **The resolution.** A 45 s window resolves 1.33 rpm, which is coarser than
  the difference between a resting adult and a nervous one. The peak has to be
  interpolated, not rounded to a bin.

There is no downsampling step. The spec asks for 20-50 Hz and this link
delivers 18.12; there is nothing to decimate and the anti-aliasing filter
would only remove the fidget band that ``backend.motion`` reads.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.presence import bandpass

# 6-36 rpm. Wider than the 9-30 the earlier detector used, because a resting
# adult at 11-12 rpm was landing on the edge of that band.
RATE_BAND_RPM = (6.0, 36.0)
BREATHING_BAND_HZ = (RATE_BAND_RPM[0] / 60.0, RATE_BAND_RPM[1] / 60.0)

# 45 s: long enough that the slowest rate in band fits four times over, and
# that the transform resolves 1.33 rpm before interpolation.
BREATHING_WINDOW_SECONDS = 45.0
BREATHING_HOP_SECONDS = 2.0

# Subcarriers are combined in groups rather than all at once, so that
# agreement between groups can be measured. Folding the axis entirely would
# make the frequency diversity unavailable exactly where it is diagnostic.
N_GROUPS = 6

# The transform is zero-padded before the peak is located, so the parabolic
# fit has bins fine enough to sit on.
ZERO_PAD_FACTOR = 16

# In-band peak-to-average power below this is not a tone.
MIN_PAPR = 4.0
# Confidence below this reports no rate at all rather than a number.
MIN_CONFIDENCE = 0.4
# How fast confidence recovers once a motion gate lifts, per window.
GATE_RECOVERY = 0.25


def project_to_1d(sig: np.ndarray) -> np.ndarray:
    """Project each subcarrier's complex cloud onto its own principal axis.

    A chest at a fixed distance moves the ratio along one direction in the
    complex plane -- the perturbation is ``d * s(t)`` for a fixed complex ``d``
    and a real waveform ``s`` -- but ``d`` is set by that subcarrier's
    multipath geometry and differs across the array. Taking ``|r|`` picks the
    magnitude axis for all of them, and wherever ``d`` is perpendicular to it
    the chest survives only as a second-order term.

    Solved in closed form rather than by an eigendecomposition per subcarrier:
    the principal angle of a 2-D cloud is
    ``0.5 * atan2(2*cov_ri, var_r - var_i)``, which vectorises across the whole
    array at once.
    """
    sig = np.asarray(sig)
    if sig.ndim != 2:
        raise ValueError(f"sig must be 2-D (n_samples, n_sc), got {sig.shape}")

    x = sig.real - sig.real.mean(axis=0, keepdims=True)
    y = sig.imag - sig.imag.mean(axis=0, keepdims=True)
    var_x = np.mean(x * x, axis=0)
    var_y = np.mean(y * y, axis=0)
    cov = np.mean(x * y, axis=0)

    theta = 0.5 * np.arctan2(2.0 * cov, var_x - var_y)
    return x * np.cos(theta) + y * np.sin(theta)


def band_weights(
    sig: np.ndarray, fs: float, band_hz: tuple[float, float] = BREATHING_BAND_HZ
) -> np.ndarray:
    """In-band power fraction per subcarrier: the periodicity SNR.

    A fraction of the subcarrier's own total, so a loud broadband subcarrier
    cannot outrank a quiet tonal one. Weighting by variance instead ranks the
    array by which subcarriers sat nearest a fading null, which is the
    opposite of what is wanted.

    Must be measured on the *un*-bandpassed signal. After the bandpass there
    is nothing out of band left to compare against, so every subcarrier scores
    near 1 and the weighting silently stops doing anything -- which is exactly
    what it did here until the w[k] panel came out uniformly saturated.
    """
    sig = np.asarray(sig)
    if sig.ndim != 2:
        raise ValueError(f"sig must be 2-D (n_samples, n_sc), got {sig.shape}")

    x = sig - sig.mean(axis=0, keepdims=True)
    n = x.shape[0]
    taper = np.hanning(n)[:, None]
    power = np.abs(np.fft.fft(x * taper, axis=0)) ** 2
    freqs = np.abs(np.fft.fftfreq(n, d=1.0 / fs))

    in_band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    total = power.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = power[in_band].sum(axis=0) / total
    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)


def align_signs(s: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Flip subcarriers that carry the chest inverted, against the best one.

    The projection above fixes an axis but not its polarity: ``theta`` and
    ``theta + pi`` describe the same line. Summed unaligned, two subcarriers
    carrying the same chest with opposite sign cancel exactly, and what
    reaches the rate estimator is a flat trace that looks like an empty room
    rather than like an error.

    The reference is the highest-weight subcarrier, which is the one most
    likely to actually carry a chest to align the rest against.
    """
    s = np.asarray(s, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"s must be 2-D (n_samples, n_sc), got {s.shape}")
    if weights.shape != (s.shape[1],):
        raise ValueError(f"weights must be length {s.shape[1]}, got {weights.shape}")

    ref = int(np.argmax(weights))
    centred = s - s.mean(axis=0, keepdims=True)
    corr = centred.T @ centred[:, ref]
    flip = np.where(corr < 0, -1.0, 1.0)
    return s * flip


def group_combine(
    s: np.ndarray, weights: np.ndarray, n_groups: int = N_GROUPS
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted sums over contiguous subcarrier groups, plus the whole array.

    Contiguous because the subcarrier axis is the Fourier partner of delay:
    neighbouring subcarriers see the target through similar path lengths, so a
    group is a coarse range gate. Groups that agree on a rate raise confidence
    in it; groups that disagree are how a second person in the room becomes
    visible at all, which is information folding the axis would destroy.
    """
    s = np.asarray(s, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"s must be 2-D (n_samples, n_sc), got {s.shape}")
    if n_groups < 1 or n_groups > s.shape[1]:
        raise ValueError(
            f"cannot form {n_groups} groups from {s.shape[1]} subcarriers"
        )

    edges = np.linspace(0, s.shape[1], n_groups + 1).astype(int)
    groups = np.empty((s.shape[0], n_groups), dtype=float)
    for g in range(n_groups):
        sl = slice(edges[g], edges[g + 1])
        w = weights[sl]
        total = w.sum()
        groups[:, g] = (s[:, sl] * w).sum(axis=1) / total if total > 0 else 0.0

    total = weights.sum()
    combined = (
        (s * weights).sum(axis=1) / total if total > 0 else s.mean(axis=1)
    )
    return groups, combined


def _parabolic_peak(power: np.ndarray, index: int) -> float:
    """Sub-bin peak location by fitting a parabola through three samples."""
    if index <= 0 or index >= power.size - 1:
        return float(index)
    a, b, c = power[index - 1], power[index], power[index + 1]
    denom = a - 2.0 * b + c
    if denom == 0:
        return float(index)
    return float(index + 0.5 * (a - c) / denom)


def estimate_rate(
    x: np.ndarray,
    fs: float,
    *,
    band_rpm: tuple[float, float] = RATE_BAND_RPM,
    zero_pad: int = ZERO_PAD_FACTOR,
) -> dict[str, float]:
    """Rate of one combined trace, by two methods that must agree.

    The transform locates the peak and a parabolic fit refines it between
    bins: a 45 s window resolves 1.33 rpm raw, which cannot tell 14 rpm from
    15. The autocorrelation's first in-band peak answers the same question
    from the time domain.

    They are reported separately along with their agreement rather than
    averaged. Averaging two estimators that disagree manufactures a confident
    number for a window that had none, which on this panel is the one output
    that must never be invented.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got {x.shape}")
    n = x.size
    lo_hz, hi_hz = band_rpm[0] / 60.0, band_rpm[1] / 60.0
    if hi_hz >= fs / 2.0:
        raise ValueError(f"{band_rpm[1]:g} rpm is at or above Nyquist for {fs:.2f} Hz")

    x = x - x.mean()
    taper = np.hanning(n)
    padded = int(n * max(1, zero_pad))
    power = np.abs(np.fft.rfft(x * taper, n=padded)) ** 2
    freqs = np.fft.rfftfreq(padded, d=1.0 / fs)
    in_band = np.flatnonzero((freqs >= lo_hz) & (freqs <= hi_hz))
    if in_band.size == 0:
        raise ValueError("the window resolves no bin inside the rate band")

    peak = int(in_band[np.argmax(power[in_band])])
    refined = _parabolic_peak(power, peak)
    df = freqs[1] - freqs[0]
    rpm_fft = float(refined * df * 60.0)

    band_power = power[in_band]
    papr = float(band_power.max() / band_power.mean()) if band_power.mean() > 0 else 0.0

    # Autocorrelation, over lags the band allows and the window can support.
    spectrum = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=2 * n)[:n]
    if acf[0] > 0:
        acf = acf / acf[0]
    lag_lo = max(1, int(round(fs / hi_hz)))
    lag_hi = min(n // 2, int(round(fs / lo_hz)))
    if lag_hi <= lag_lo:
        rpm_acf = float("nan")
    else:
        segment = acf[lag_lo : lag_hi + 1]
        lag = lag_lo + int(np.argmax(segment))
        rpm_acf = float(60.0 * fs / lag) if lag > 0 else float("nan")

    if np.isfinite(rpm_acf) and rpm_fft > 0:
        agreement = float(
            1.0 - min(abs(rpm_fft - rpm_acf) / max(rpm_fft, rpm_acf), 1.0)
        )
    else:
        agreement = 0.0

    return {
        "rpm": rpm_fft,
        "rpm_fft": rpm_fft,
        "rpm_acf": rpm_acf,
        "papr": papr,
        "agreement": agreement,
    }


def breathing_windows(
    sig: np.ndarray,
    fs: float,
    *,
    window_seconds: float = BREATHING_WINDOW_SECONDS,
    hop_seconds: float = BREATHING_HOP_SECONDS,
    band_rpm: tuple[float, float] = RATE_BAND_RPM,
    n_groups: int = N_GROUPS,
    motion_gate: np.ndarray | None = None,
    fabricated: np.ndarray | None = None,
    max_gap_fraction: float = 0.5,
    min_confidence: float = MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Sliding-window respiration over a detrended, normalised complex ratio.

    *sig* must have been through ``preprocess.remove_static`` at the breathing
    window and ``preprocess.normalize_subcarriers``; a static term reaches the
    autocorrelation as a DC peak and swamps everything.

    *motion_gate* is one boolean per output window, True where
    ``backend.motion`` says the channel was being driven by something larger
    than a chest. Those windows hold the previous estimate rather than
    recomputing it, and confidence restarts low and recovers over a few
    windows: an estimate taken during a walk-through is contaminated, and the
    first window after one is built mostly from samples that overlap it.
    """
    sig = np.asarray(sig)
    if sig.ndim != 2:
        raise ValueError(f"sig must be 2-D (n_samples, n_sc), got {sig.shape}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")

    n_samples, n_sc = sig.shape
    win = min(max(8, int(round(window_seconds * fs))), n_samples)
    hop = max(1, int(round(hop_seconds * fs)))
    starts = np.arange(0, n_samples - win + 1, hop)
    if starts.size == 0:
        raise ValueError("the range is shorter than one breathing window")

    band_hz = (band_rpm[0] / 60.0, band_rpm[1] / 60.0)
    # Filtered once over the whole series: the filter is linear, so the result
    # inside a window is the same either way, minus the edge ringing a
    # per-window filter would add at every boundary.
    filtered = bandpass(sig, fs, band_hz)

    fab = None if fabricated is None else np.asarray(fabricated, dtype=bool)
    gate = None if motion_gate is None else np.asarray(motion_gate, dtype=bool)
    if gate is not None and gate.shape != (starts.size,):
        raise ValueError(
            f"motion_gate must be one flag per window ({starts.size}), got {gate.shape}"
        )

    n_win = starts.size
    rpm = np.full(n_win, np.nan)
    confidence = np.zeros(n_win)
    papr = np.full(n_win, np.nan)
    agreement = np.full(n_win, np.nan)
    spread = np.full(n_win, np.nan)
    gated = np.zeros(n_win, dtype=bool)
    weights_over_time = np.zeros((n_sc, n_win))
    group_rpm = np.full((n_groups, n_win), np.nan)

    held_rpm = np.nan
    held_conf = 0.0
    for i, start in enumerate(starts):
        stop = start + win
        if fab is not None and fab[start:stop].mean() > max_gap_fraction:
            gated[i] = True
            rpm[i] = held_rpm
            confidence[i] = 0.0
            continue
        if gate is not None and gate[i]:
            gated[i] = True
            rpm[i] = held_rpm
            held_conf = 0.0
            confidence[i] = 0.0
            continue

        seg = filtered[start:stop]
        # Weights from the raw window, projections from the filtered one: the
        # weight is a question about where this subcarrier's energy sits, and
        # the filter has already answered it everywhere.
        w = band_weights(sig[start:stop], fs, band_hz)
        weights_over_time[:, i] = w
        projected = align_signs(project_to_1d(seg), w)
        groups, combined = group_combine(projected, w, n_groups)

        est = estimate_rate(combined, fs, band_rpm=band_rpm)
        per_group = np.array(
            [estimate_rate(groups[:, g], fs, band_rpm=band_rpm)["rpm"] for g in range(n_groups)]
        )
        group_rpm[:, i] = per_group
        spread[i] = float(np.std(per_group))
        papr[i] = est["papr"]
        agreement[i] = est["agreement"]

        # Three independent ways of being wrong, so three factors. A tone that
        # is not a tone, two estimators that disagree, and groups that cannot
        # settle on one rate all have to be survived.
        tone = min(est["papr"] / (2.0 * MIN_PAPR), 1.0)
        spread_term = 1.0 / (1.0 + spread[i] / max(est["rpm"], 1e-6) * 4.0)
        raw = tone * est["agreement"] * spread_term
        # Recovery after a gate: the first window after one still overlaps it.
        held_conf = min(raw, held_conf + GATE_RECOVERY) if held_conf < raw else raw
        confidence[i] = held_conf
        rpm[i] = est["rpm"] if held_conf >= min_confidence else np.nan
        if np.isfinite(rpm[i]):
            held_rpm = rpm[i]

    return {
        "time_s": (starts + win / 2.0) / fs,
        "rpm": rpm,
        "confidence": confidence,
        "papr": papr,
        "agreement": agreement,
        "group_rpm": group_rpm,
        "group_spread_rpm": spread,
        "gated": gated,
        "weights": weights_over_time,
        "win": int(win),
        "hop": int(hop),
        "window_seconds": float(win / fs),
        "rpm_resolution": float(60.0 * fs / win),
        "band_rpm": (float(band_rpm[0]), float(band_rpm[1])),
    }
