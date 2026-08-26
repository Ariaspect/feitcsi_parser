"""Common front half of the motion and breathing detectors.

Both paths start from the same three operations and then diverge, so the
operations live here rather than being written twice with two sets of
constants. What they share: a uniform time axis, a subcarrier mask, and the
removal of the static path. What they do *not* share is the window that last
operation uses -- 2-5 s for motion, 10-20 s for breathing -- which is the
whole reason ``remove_static`` takes it as an argument rather than owning it.

Getting that window backwards is silent. A 3 s window is shorter than a 0.25
Hz breath cycle and removes the breath along with the wall; a 15 s window is
several cycles long and passes it. Neither raises anything.

One thing deliberately *not* shared: the static path itself. This module
strips it, because both detectors here are looking at how the channel moves.
``backend.presence`` keeps it, because a motionless occupant does not move the
channel at all -- they displace it, and the displacement is the only evidence
there is. The two readings of the same component are not in conflict; they are
answers to different questions, and both run on the same capture.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
#  Physical and capture constants                                              #
#                                                                              #
#  Nothing below this block hardcodes a wavelength, a rate or a band edge.     #
#  These are the only values to edit when the radio or the ping changes.       #
# --------------------------------------------------------------------------- #

SPEED_OF_LIGHT_M_S = 299_792_458.0

# The RF centre frequency. Not recoverable from the capture: the FeitCSI and
# MediaTek headers carry the bandwidth and nothing about where in the band it
# sits, so this is the operator's answer and every millimetre-scale claim
# downstream rests on it being right.
CARRIER_HZ = 5.21e9
WAVELENGTH_M = SPEED_OF_LIGHT_M_S / CARRIER_HZ          # 5.754 cm at 5210 MHz

# What the capture script asks for, against which the delivered rate is
# checked. The rate actually used is always the derived one -- this is here to
# make a divergence visible, not to override the measurement.
NOMINAL_FS_HZ = 20.0                                     # 50 ms ping
FS_MISMATCH_TOLERANCE = 0.05

# Sliding-window lengths for static removal, per path. See the module note:
# these are not interchangeable and the breathing one must span several breath
# cycles.
MOTION_DETREND_SECONDS = 3.0                             # spec range 2-5 s
BREATHING_DETREND_SECONDS = 15.0                         # spec range 10-20 s

# Fraction of subcarriers dropped for a weak denominator, lowest |H0| first.
WEAK_SUBCARRIER_FRACTION = 0.15                          # spec range 0.10-0.20


def derive_sample_rate(times: np.ndarray) -> dict[str, Any]:
    """Measure the frame rate, and say so when it is not the one requested.

    The mean rather than the median sets the grid. They disagree whenever the
    interval is multi-modal, and this link's is: measured on
    captures/lg/20260825_185637.bin the interval sits at 50 ms for 23.4% of
    frames and at 56 ms for 24.1%, so the median lands on whichever mode is
    momentarily taller -- 56 ms, an 18.86 Hz grid for a link that delivered
    18.50 Hz. Resampling onto that stretches the whole time axis and every
    frequency read off it.

    The derived rate is always what gets used. The nominal one exists so that
    a link quietly dropping a fifth of its pings shows up as a warning instead
    of as an 8% error in every reported rate.
    """
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        raise ValueError("need at least 2 frames to infer a sample rate")

    dt = np.diff(times)
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("frames carry no positive time deltas")

    mean_hz = 1.0 / float(np.mean(dt))
    median_hz = 1.0 / float(np.median(dt))

    warnings: list[str] = []
    drift = abs(mean_hz - NOMINAL_FS_HZ) / NOMINAL_FS_HZ
    if drift > FS_MISMATCH_TOLERANCE:
        warnings.append(
            f"the capture delivered {mean_hz:.2f} Hz against a nominal "
            f"{NOMINAL_FS_HZ:g} Hz ({100 * drift:.1f}% low) -- the derived rate is "
            "what every frequency axis here uses, but the link is not keeping up "
            "with its ping interval"
        )
    if abs(mean_hz - median_hz) / mean_hz > FS_MISMATCH_TOLERANCE:
        warnings.append(
            f"inter-frame intervals are multi-modal: mean {1e3 / mean_hz:.1f} ms "
            f"against median {1e3 / median_hz:.1f} ms"
        )

    return {
        "fs_hz": mean_hz,
        "mean_hz": mean_hz,
        "median_hz": median_hz,
        "nominal_hz": NOMINAL_FS_HZ,
        "nyquist_hz": mean_hz / 2.0,
        "warnings": warnings,
    }


def subcarrier_mask(
    ratio: np.ndarray,
    h0_amplitude_db: np.ndarray,
    *,
    weak_fraction: float = WEAK_SUBCARRIER_FRACTION,
) -> dict[str, Any]:
    """Which subcarriers may be averaged over, and why the others may not.

    Two exclusions, for two different reasons.

    *Structurally dead* bins -- the DC bin, the guard band, a pilot the decoder
    dropped -- never carried a ratio at all. They are not a dropout in time and
    cannot be filled: reaching an average as a zero they dilute it, reaching it
    as a NaN they destroy it.

    *Weak* bins carried one, but through a faded ``H0``. The ratio divides by
    that, so where the denominator sits in a null the quotient is dominated by
    whatever noise happened to be there. This is the likely origin of the fixed
    horizontal bands in the ratio amplitude heatmap: a band that holds the same
    subcarrier for ten minutes is a property of the link, not of the room, and
    it should not survive into a per-subcarrier statistic.

    Returns the keep mask and both exclusion lists by index, because a mask
    that cannot be audited is a mask that hides its own mistakes.
    """
    ratio = np.asarray(ratio)
    h0_amplitude_db = np.asarray(h0_amplitude_db, dtype=float)
    if ratio.ndim != 2:
        raise ValueError(f"ratio must be 2-D (n_samples, n_sc), got {ratio.shape}")
    if h0_amplitude_db.shape[1] != ratio.shape[1]:
        raise ValueError(
            f"H0 has {h0_amplitude_db.shape[1]} subcarriers and the ratio has "
            f"{ratio.shape[1]}"
        )
    if not 0.0 <= weak_fraction < 1.0:
        raise ValueError(f"weak_fraction must be in [0, 1), got {weak_fraction}")

    alive = np.isfinite(ratio).any(axis=0)
    dropped_dead = np.flatnonzero(~alive)
    if not alive.any():
        raise ValueError("no subcarrier in this range carries a CSI ratio")

    keep = alive.copy()
    dropped_weak = np.zeros(0, dtype=int)
    n_weak = int(np.floor(weak_fraction * int(alive.sum())))
    if n_weak > 0:
        strength = np.full(ratio.shape[1], np.inf)
        live_idx = np.flatnonzero(alive)
        with np.errstate(invalid="ignore"):
            for k in live_idx:
                col = h0_amplitude_db[:, k]
                finite = col[np.isfinite(col)]
                strength[k] = np.median(finite) if finite.size else -np.inf
        # Ties broken by index so the mask is reproducible run to run.
        order = sorted(live_idx.tolist(), key=lambda k: (strength[k], k))
        dropped_weak = np.array(sorted(order[:n_weak]), dtype=int)
        keep[dropped_weak] = False

    if not keep.any():
        raise ValueError("no subcarrier survived masking")

    return {
        "keep": keep,
        "dropped_dead": dropped_dead,
        "dropped_weak": dropped_weak,
        "n_kept": int(keep.sum()),
    }


def _box_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Centred running mean of *k* samples along axis 0, normalised at edges.

    Edge normalisation rather than zero padding. A padded edge pulls the mean
    toward zero, and subtracting a mean that is wrong by a known amount writes
    a step into the first and last half-window of every capture -- which the
    motion path would then report as someone walking in and out.

    The length is forced odd so the window is exactly symmetric about the
    sample it is centred on. An even window sits half a sample off centre,
    which is a half-sample group delay -- subtracting it from a linear drift
    leaves a constant offset behind (slope * dt / 2) instead of nothing, and
    the filter stops being zero-phase for the same reason ``filtfilt`` is used
    downstream rather than a one-pass filter.

    Cumulative sums rather than a convolution: this runs over 11k samples and
    a few hundred subcarriers, and the direct form is minutes rather than
    milliseconds.
    """
    n = x.shape[0]
    k = max(1, min(int(k), n))
    if k % 2 == 0:
        k = max(1, k - 1)
    lo = np.maximum(np.arange(n) - (k // 2), 0)
    hi = np.minimum(lo + k, n)
    lo = np.maximum(hi - k, 0)

    csum = np.concatenate([np.zeros((1, x.shape[1]), dtype=x.dtype), np.cumsum(x, axis=0)])
    counts = (hi - lo)[:, None]
    return (csum[hi] - csum[lo]) / counts


def remove_static(ratio: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    """Subtract a per-subcarrier complex sliding mean.

    Complex, not the magnitude. The static path and the reflection off a
    person add as vectors, and a body that changes only the *phase* of the sum
    -- a Fresnel arc, which is what small movement at a fixed distance looks
    like -- leaves the magnitude almost untouched. Detrending the magnitude
    would remove nothing and pass nothing.

    *seconds* is the caller's, and it is not a tuning knob but a choice of
    what to keep. See the module note.
    """
    ratio = np.asarray(ratio)
    if ratio.ndim != 2:
        raise ValueError(f"ratio must be 2-D (n_samples, n_sc), got {ratio.shape}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")

    x = ratio.astype(complex, copy=False)
    return x - _box_mean(x, int(round(seconds * fs)))


def normalize_subcarriers(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide each subcarrier by its own noise scale.

    Principle: no time-axis quantity is thresholded in absolute units. The
    scale a subcarrier arrives in is set by how deep its ``H0`` sat in a
    fading null, which is a property of the link, not of the room -- and the
    ratio's noise gain runs as ``1/|H0|`` continuously, so no rank cut in
    ``subcarrier_mask`` can level it. Measured on
    captures/lg/20260825_185637.bin the shoulders of the two nulls survive
    masking 9.5 dB hotter than the rest, while their spectral shape matches
    the quiet subcarriers almost exactly (respiration-band share 20.3%
    against 17.8%, above 1 Hz 30.5% against 36.1%). A pure scale is exactly
    what dividing removes, and dropping them instead would throw away the
    delay information the subcarrier axis carries.

    The scale is the *median* magnitude, converted to the sigma of a circular
    complex Gaussian through the Rayleigh median ``sigma * sqrt(2 ln 2)``.
    Median rather than RMS because a subcarrier that was quiet for 590 s and
    loud for 10 is a subcarrier that saw someone walk past: an RMS scale would
    take the burst as the subcarrier's own level and divide the walk-through
    back out.

    There is no noise-only band to measure instead. At this carrier a Doppler
    of 8 Hz is 0.23 m/s of radial velocity, so real motion reaches the top of
    a 9 Hz Nyquist and the spectrum has no quiet end to calibrate against.

    Returns the normalised signal and the per-subcarrier scale, which is
    itself an output worth keeping -- it maps where the link's nulls are.
    """
    sig = np.asarray(sig)
    if sig.ndim != 2:
        raise ValueError(f"sig must be 2-D (n_samples, n_sc), got {sig.shape}")

    rayleigh_median = np.sqrt(2.0 * np.log(2.0))
    with np.errstate(invalid="ignore"):
        scale = np.nanmedian(np.abs(sig), axis=0) / rayleigh_median

    # A subcarrier with no scale is silent, not infinitely sensitive. Dividing
    # by its own zero would hand every later average an infinity.
    positive = scale[np.isfinite(scale) & (scale > 0)]
    floor = float(np.median(positive)) * 1e-6 if positive.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, floor)

    return sig / scale, scale
