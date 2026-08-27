"""CSI-ratio motion level and static-presence (breathing) detection.

Three states are worth telling apart in a room: nobody there, somebody there
and still, somebody there and moving. They separate on two axes rather than
one. Gross motion is loud and broadband -- a walking person swamps every
subcarrier -- while a *still* person is nearly silent, betrayed only by chest
motion of a few millimetres at 0.2-0.5 Hz. So "loud" and "quiet" alone cannot
tell an empty room from a sleeping one; the quiet case has to be decided on
*periodicity*, not on energy.

That is the shape of this module, on three axes rather than two. A
dimensionless motion level answers "is anything happening". The *channel
offset* from a known-empty reference answers "is the room different from when
it was empty" -- which is what a motionless body actually produces: it does
not modulate the channel, it displaces it. And a windowed autocorrelation
answers "is any of this a chest", reported as evidence beside the verdict
rather than as the verdict itself.

The autocorrelation maths is a port of the detector in the BFI_Raspberry
project (``bfi_core.breathing_score``), which ran on beamforming-feedback V
matrices; the arithmetic carries over unchanged because both pipelines end up
holding the same thing -- a complex per-subcarrier channel ratio sampled over
time. What did not carry over is the idea that it could decide occupancy on
its own. Measured on captures/lg_csi_captures/20260825/20260825_185637.bin --
a deliberate walk-in, sit-still, walk-out -- periodicity ran *higher* in the
empty room (0.102) than with the occupant sitting still (0.076), so a branch
that votes on it alone votes wrong. It reports a rate; it does not decide.

Three things differ here, and each one matters:

* **The channel is whatever ratio the caller hands over, and that choice is
  load-bearing.** Raw CSI arrives with rx0 and rx1 swapped on some frames,
  which BFI never had to deal with: a V-matrix column cannot be confused with
  its neighbour. Uncorrected, 1.2% of frame-to-frame steps in the ratio phase
  exceed pi outright (see ``backend.ratio``), and a pi jump is a broadband
  impulse landing directly in the band respiration lives in. The API feeds
  this module the *raw* metrics (``tiles.PRESENCE_METRICS``) so that nothing
  is inferred before the detector sees it -- which means the swap impulses are
  in the signal, reading as motion energy and as periodicity neither of which
  came from the room. On a capture with frequent swaps, that is the first
  thing to rule out before trusting a verdict.

* **Gaps are not bridged blindly.** ``bfi_core`` interpolates across every
  dropout unconditionally, which turns a 23-second hole into a perfectly flat
  stretch. Flat means no motion *and* no periodicity, so the detector reports
  the one answer it must never invent: "empty room". Here a window that is
  mostly fabricated is reported as ``unknown`` and never as ``empty``.

* **Absence is a claim, so it needs a reference.** ``empty`` means "matched a
  room known to be empty", and without a stretch of capture the caller has
  named as empty there is nothing to match -- so every non-moving window comes
  back ``unknown`` instead. This is not caution for its own sake: the same
  capture above returned ``empty`` for all 584 of its windows under the
  previous design, occupant and all. Every window also carries the pieces the
  verdict was built from -- channel offset, motion level and its multiple of
  the room's floor, periodicity, tonality, motion gate -- so a caller can
  always see *why*.

What this can see, and cannot: respiration at 9-30 rpm needs a sample rate
above ~1 Hz, and these captures run 5-18 Hz, so the band is comfortably in
reach. Heartbeat (0.8-2 Hz, ~0.5 mm) is not attempted -- it sits under
respiration harmonics and would need an SNR this pipeline does not have.
Neither is a sleeper whose displacement the reference already contains: if the
reference range was recorded with them in it, they *are* the empty room as far
as this module is concerned. The reference has to come from a stretch that was
genuinely empty, which is a fact about the world and not one the data can
supply.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

# The per-subcarrier signal the detector runs on. "complex" is the default and
# the one to trust: amplitude and phase have *complementary* Fresnel blind
# spots, so a chest that is invisible in one is visible in the other, and
# keeping the ratio complex avoids having to choose. The real-valued channels
# are kept because seeing which one carries the signal is diagnostic -- it says
# where the subject is sitting relative to the antennas.
CHANNELS = ("complex", "phase", "magnitude")

# Defaults carried over from bfi_core, where they were tuned against labelled
# walk-in / sit-still / walk-out captures. They are a starting point on raw
# CSI, not a calibration: nothing here has been fitted to this hardware.
# 30 s rather than the 12 s carried over from bfi_core. ``lag_hi`` below caps
# the slowest reachable rate at ``120 / window_seconds`` rpm, so a 12 s window
# reaches only 10 rpm -- above a resting adult's 11-15 rpm, and above the 9 rpm
# this module advertises by default, which is why every call at the old
# default emitted a rate-floor warning. 30 s reaches 4 rpm. The hop is
# unchanged, so the strip keeps its one-second resolution.
DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_HOP_SECONDS = 1.0
DEFAULT_RATE_BAND_RPM = (9.0, 30.0)
DEFAULT_BANDPASS_HZ = (0.1, 0.6)
DEFAULT_MOTION_FRAC_LO = 0.10
DEFAULT_MOTION_FRAC_HI = 0.25
DEFAULT_TONALITY_FLAT_LO = 0.5
DEFAULT_TONALITY_FLAT_HI = 0.95
DEFAULT_SMOOTH_WINDOWS = 3
DEFAULT_PRESENT_THRESHOLD = 0.25

# How far a window's channel state must sit from the empty-room reference
# before a still occupant is claimed, in units of how much that same empty
# room wandered on its own (``dev_p95``). Expressing it that way is what makes
# one number work across radios: the absolute deviation that means "occupied"
# is a property of the room and the hardware, the *ratio* to the room's own
# variability is not. Measured on
# captures/lg_csi_captures/20260825/20260825_185637.bin, a reference taken
# from the empty tail has dev_p95 = 0.62 dB while an occupant sitting still
# reads 2.32 dB at the 5th percentile -- 3.75x -- so 3.0 separates them with
# margin on both sides.
DEFAULT_BASELINE_DEV_K = 3.0

# Gross motion as a multiple of the room's own fractional-motion floor. The
# absolute test alone cannot do this job: on the capture above the floor is
# 0.069 and a walk-through peaks at 0.185, so the 0.25 absolute threshold
# never fired while the walk was 2.7x the floor.
DEFAULT_MOTION_RATIO_HI = 2.0

# The in-band weight compares respiration-band power against a shoulder just
# outside the band on *both* sides, spanning these multiples of the band
# edges. Weighting against everything above the band instead -- which is what
# this did -- measures 1/f drift rather than a chest, because drift is
# enormous below 0.5 Hz and absent above it. On the capture above that made an
# empty room outscore an occupied one on periodicity (0.102 against 0.076).
SHOULDER_LO_FACTOR = 1.0 / 3.0
SHOULDER_HI_FACTOR = 3.0

# A window blanked once more than this fraction of it was interpolated across
# a dropout. Same constant and same reasoning as backend.doppler, so the
# presence strip and the spectrogram agree about what counts as a hole.
DEFAULT_MAX_GAP_FRACTION = 0.5

# An autocorrelation over fewer samples than this is not worth computing.
MIN_WINDOW_SAMPLES = 8

# Verdict vocabulary. UNKNOWN is first among equals: it exists so that missing
# data can never be reported as an empty room.
STATE_UNKNOWN = "unknown"
STATE_MOVING = "moving"
STATE_PRESENT = "present"
STATE_EMPTY = "empty"


def complex_ratio(amplitude_db: np.ndarray, phase_rad: np.ndarray) -> np.ndarray:
    """Rebuild the complex CSI ratio from its dB magnitude and its phase.

    The decode path stores the ratio split into two real planes because that
    is what a heatmap draws. Every operation below is complex, so it is put
    back together here. ``10 ** (dB / 20)`` inverts ``20*log10(|r|)``; the
    power-domain ``10 ** (dB / 10)`` is the wrong inverse and would square the
    magnitude ratio between the chains.
    """
    amplitude_db = np.asarray(amplitude_db, dtype=float)
    phase_rad = np.asarray(phase_rad, dtype=float)
    if amplitude_db.shape != phase_rad.shape:
        raise ValueError(
            f"amplitude {amplitude_db.shape} and phase {phase_rad.shape} must match"
        )
    return 10 ** (amplitude_db / 20.0) * np.exp(1j * phase_rad)


def live_subcarriers(ratio: np.ndarray) -> np.ndarray:
    """Boolean mask of subcarriers that carry a ratio anywhere in the range.

    A capture always holds some that never do -- the DC bin, the guard band, a
    pilot the decoder dropped; 11 of 256 on a MediaTek capture, measured in
    ``backend.doppler``. They are not a dropout in time, they are structurally
    absent, and they have to be removed rather than filled: a dead subcarrier
    that reaches the averages as a zero dilutes every one of them, and one that
    reaches them as a NaN destroys them outright.
    """
    ratio = np.asarray(ratio)
    if ratio.ndim != 2:
        raise ValueError(f"ratio must be 2-D (n_samples, n_sc), got {ratio.shape}")
    return np.isfinite(ratio).any(axis=0)


def fractional_motion(ratio: np.ndarray) -> np.ndarray:
    """Per-sample fractional change of the ratio, averaged over subcarriers.

    ``|dr| / |r|`` rather than ``|dr|``: the ratio's absolute scale is set by
    how the two chains happen to be gained, which varies by capture and by
    hardware, and a threshold expressed in those units would have to be
    re-tuned for every file. The fractional form is dimensionless, so the same
    number means the same amount of channel disturbance everywhere.

    The average across subcarriers ignores NaN. It has to: a plain ``mean``
    returns NaN for a frame if a *single* subcarrier is missing, and since the
    structural nulls are missing in every frame, the whole trace goes NaN --
    which a downstream ``nan_to_num`` then renders as a perfectly flat zero.
    Measured on captures/20260821_170002.bin, that is exactly what happened:
    motion read 0.0000 at every percentile, the most confident possible claim
    that the room is perfectly still, from 11 dead bins out of 256.

    Returns a length ``n - 1`` array; index *i* is the change between samples
    *i* and *i + 1*. A sample pair with nothing finite anywhere yields NaN --
    genuinely unmeasurable, and left for the caller to blank rather than
    rounded down to "no motion".
    """
    ratio = np.asarray(ratio)
    if ratio.ndim != 2:
        raise ValueError(f"ratio must be 2-D (n_samples, n_sc), got {ratio.shape}")
    if ratio.shape[0] < 2:
        return np.zeros(0, dtype=float)

    mag = np.abs(ratio)
    # Midpoint magnitude, so a step is measured against the average of where
    # it started and where it landed rather than against either end.
    denom = 0.5 * (mag[1:] + mag[:-1]) + 1e-9
    step = np.abs(np.diff(ratio, axis=0)) / denom
    step[~np.isfinite(step)] = np.nan
    alive = np.isfinite(step).any(axis=1)

    frac = np.full(step.shape[0], np.nan, dtype=float)
    if alive.any():
        frac[alive] = np.nanmean(step[alive], axis=1)
    return frac


def amplitude_profile(ratio: np.ndarray) -> np.ndarray:
    """Per-subcarrier median ratio amplitude in dB, over the samples given.

    The *static* state of the channel, which is exactly what every other part
    of this module throws away: ``channel_from_ratio`` mean-removes each
    subcarrier so that a constant offset never reaches the autocorrelation. A
    body parked in a room is a constant offset, so absence and stillness are
    indistinguishable once the mean is gone, and something has to hold on to
    it. This is that something.

    Median rather than mean so a burst of movement inside the range does not
    drag the profile with it -- the profile is meant to describe where the
    channel *sits*, not how far it travelled.

    dB rather than linear because a multipath displacement is multiplicative:
    the same body moves a strong subcarrier and a weak one by similar
    fractions, and only in dB do those become the same number. Kept at full
    subcarrier width, with NaN where a bin never carried a ratio, so that
    profiles from two different ranges -- or two different captures -- line up
    bin for bin.
    """
    mag = np.abs(np.asarray(ratio))
    if mag.ndim != 2:
        raise ValueError(f"ratio must be 2-D (n_samples, n_sc), got {mag.shape}")
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20.0 * np.log10(np.where(mag > 0.0, mag, np.nan))
    alive = np.isfinite(db).any(axis=0)
    profile = np.full(db.shape[1], np.nan, dtype=float)
    if alive.any():
        profile[alive] = np.nanmedian(db[:, alive], axis=0)
    return profile


def baseline_deviation(profile: np.ndarray, reference: np.ndarray) -> float:
    """Mean absolute dB distance between a window's profile and a reference.

    Averaged over the subcarriers finite in *both*, because the two sides need
    not agree about which bins are alive: a reference taken from another
    capture, or from a stretch where one transmitter was quiet, carries a
    different set of structural nulls. Bins missing on either side are
    dropped rather than filled -- an invented bin would contribute an invented
    distance, and this number is the whole evidence for a presence claim.

    Returns NaN when nothing is comparable, which the caller must treat as
    "no verdict" rather than as "no deviation".
    """
    profile = np.asarray(profile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if profile.shape != reference.shape:
        raise ValueError(
            f"profile {profile.shape} and reference {reference.shape} must match -- "
            "a reference from a capture with different subcarrier geometry "
            "cannot be compared bin for bin"
        )
    both = np.isfinite(profile) & np.isfinite(reference)
    if not both.any():
        return float("nan")
    return float(np.mean(np.abs(profile[both] - reference[both])))


def _weight_from_power(
    power: np.ndarray, freqs: np.ndarray, band: tuple[float, float]
) -> np.ndarray:
    """Per-column in-band power against its own two-sided shoulder.

    See ``SHOULDER_LO_FACTOR``: the denominator has to contain the sub-band
    drift, or the weight ranks subcarriers by how much they wander rather than
    by how much of a chest they carry. Falls back to everything above the band
    when the window is too short to resolve a shoulder below it.
    """
    lo, hi = float(band[0]), float(band[1])
    in_mask = (freqs >= lo) & (freqs <= hi)
    shoulder = ((freqs >= lo * SHOULDER_LO_FACTOR) & (freqs < lo)) | (
        (freqs > hi) & (freqs <= hi * SHOULDER_HI_FACTOR)
    )
    if not shoulder.any():
        shoulder = freqs > hi
    return power[in_mask, :].sum(axis=0) / (power[shoulder, :].sum(axis=0) + 1e-12)


def in_band_weight(
    sig: np.ndarray, fs: float, band: tuple[float, float]
) -> np.ndarray:
    """How much respiration-band evidence each subcarrier carries.

    Used to weight subcarriers before their autocorrelations are averaged: a
    real chest moves them all coherently, so weighting by evidence reinforces
    the signature while independent noise does not survive the average.
    """
    sig = np.asarray(sig)
    if sig.ndim != 2:
        raise ValueError(f"sig must be 2-D (n_samples, n_sc), got {sig.shape}")
    n = sig.shape[0]
    sig = np.nan_to_num(sig - np.nanmean(sig, axis=0, keepdims=True))
    taper = np.hanning(n)[:, None]
    if np.iscomplexobj(sig):
        power = np.abs(np.fft.fft(sig * taper, axis=0)) ** 2
        freqs = np.abs(np.fft.fftfreq(n, d=1.0 / fs))
    else:
        power = np.abs(np.fft.rfft(sig * taper, axis=0)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return _weight_from_power(power, freqs, band)


def presence_reference(
    ratio: np.ndarray,
    fs: float,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
) -> dict[str, Any]:
    """Reduce a stretch of known-empty capture to what a verdict needs.

    Three numbers, each with one job. ``profile`` is what a window's channel
    state is compared against. ``dev_p95`` is how far the empty room's own
    windows strayed from that profile, which is the unit the presence
    threshold is expressed in -- an absolute dB threshold would have to be
    re-tuned per radio and per room, a multiple of the room's own wander does
    not. ``motion_floor`` is the fractional-motion noise floor here, which is
    emphatically not zero: 0.069 on
    captures/lg_csi_captures/20260825/20260825_185637.bin.

    The caller names the range. Deriving it instead -- taking the quietest
    stretch of the range under analysis -- was tried and does not work: a
    reference built from recent history absorbs an occupant who sits still for
    a few minutes, and then reports the room as empty precisely while it is
    not. Measured on the capture above with a 180 s lookback, the metric
    inverts outright (1.51 dB occupied against 2.23 dB empty). An empty room
    is a fact about the world, so it has to come from outside the data.
    """
    ratio = np.asarray(ratio)
    if ratio.ndim != 2:
        raise ValueError(f"ratio must be 2-D (n_samples, n_sc), got {ratio.shape}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")

    profile = amplitude_profile(ratio)
    if not np.isfinite(profile).any():
        raise ValueError("no subcarrier in the reference range carries a CSI ratio")

    n_samples = ratio.shape[0]
    win = min(max(MIN_WINDOW_SAMPLES, int(round(window_seconds * fs))), n_samples)
    hop = max(1, int(round(hop_seconds * fs)))
    frac_full = fractional_motion(ratio)

    devs, levels = [], []
    for start in range(0, n_samples - win + 1, hop):
        devs.append(baseline_deviation(amplitude_profile(ratio[start : start + win]), profile))
        seg = frac_full[start : start + win - 1]
        if seg.size and np.isfinite(seg).any():
            levels.append(float(np.nanmedian(seg)))

    dev_a = np.asarray(devs, dtype=float)
    level_a = np.asarray(levels, dtype=float)
    finite_dev = dev_a[np.isfinite(dev_a)]
    finite_lvl = level_a[np.isfinite(level_a)]
    if finite_dev.size == 0 or finite_lvl.size == 0:
        raise ValueError("the reference range holds no measurable window")

    return {
        "profile": profile,
        # A floor under both, so a pathologically quiet reference cannot make
        # every later window look like an occupant by dividing by nothing.
        "dev_p95": max(float(np.percentile(finite_dev, 95)), 1e-3),
        "motion_floor": max(float(np.median(finite_lvl)), 1e-6),
        "n_windows": int(finite_dev.size),
        "window_seconds": float(win / fs),
    }


def channel_from_ratio(ratio: np.ndarray, channel: str) -> np.ndarray:
    """Project the complex ratio onto the requested detector channel.

    Phase is unwrapped along time *after* resampling, which is the only order
    that works: interpolating a wrapped phase across a +/-pi boundary averages
    the two ends of the circle and lands halfway round it. Resampling the
    complex ratio and taking the angle afterwards has no such seam, which is
    why this function takes the complex form rather than a phase plane.

    Every channel is mean-removed per subcarrier, so a static offset -- a body
    parked in the room is one -- never reaches the autocorrelation.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
    ratio = np.asarray(ratio)

    if channel == "complex":
        sig: np.ndarray = ratio
    elif channel == "phase":
        sig = np.unwrap(np.angle(ratio), axis=0)
    else:
        sig = np.abs(ratio)

    sig = sig - np.nanmean(sig, axis=0, keepdims=True)
    return np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)


def bandpass(sig: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth bandpass along axis 0.

    Filtered once over the whole series rather than per window. The filter is
    linear so the result inside a window is the same either way, minus the
    edge ringing a per-window filter would add at every boundary.

    Zero-phase matters for more than tidiness: the autocorrelation downstream
    reads lag, and a filter with group delay would shift the peak and bias the
    reported rate. ``filtfilt`` runs the filter forwards and backwards, so the
    delay cancels exactly.

    Complex input is filtered as two real planes. That is not an approximation
    -- the coefficients are real, so it is exactly what applying the filter to
    the complex signal means.

    Returns *sig* untouched when the series is shorter than ``filtfilt``'s pad
    length, or when the band collapses against Nyquist.
    """
    sig = np.asarray(sig)
    nyq = fs / 2.0
    lo = max(1e-3, float(band[0]))
    hi = min(float(band[1]), 0.95 * nyq)
    if hi <= lo:
        return sig

    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    padlen = 3 * max(len(a), len(b))
    if sig.shape[0] <= padlen:
        return sig

    if np.iscomplexobj(sig):
        return filtfilt(b, a, sig.real, axis=0) + 1j * filtfilt(b, a, sig.imag, axis=0)
    return filtfilt(b, a, sig, axis=0)


def autocorr_columns(x: np.ndarray) -> np.ndarray:
    """Per-column autocorrelation, normalised to 1.0 at lag 0, lags 0..n-1.

    Computed through the FFT because the direct sum is O(n^2) per subcarrier
    per window and this runs over hundreds of windows.

    For complex input the real part is returned. A breathing perturbation is
    collinear -- it moves the ratio along one fixed complex direction *d* as a
    real waveform *s(t)*, so ``x = d * s(t)`` and the autocorrelation is
    ``|d|^2`` times the real autocorrelation of *s*. The periodicity signature
    is therefore carried entirely by the real part, and taking it loses
    nothing while avoiding any choice of projection.
    """
    x = np.asarray(x)
    win = x.shape[0]
    n2 = 1
    while n2 < 2 * win:
        n2 <<= 1

    if np.iscomplexobj(x):
        spectrum = np.fft.fft(x, n=n2, axis=0)
        ac = np.fft.ifft(spectrum * np.conj(spectrum), n=n2, axis=0)[:win, :].real
    else:
        spectrum = np.fft.rfft(x, n=n2, axis=0)
        ac = np.fft.irfft(spectrum * np.conj(spectrum), n=n2, axis=0)[:win, :]

    denom = ac[0:1, :].copy()
    denom[denom <= 0] = 1.0
    return ac / denom


def moving_average(y: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average of *k* samples, normalised at the edges."""
    y = np.asarray(y, dtype=float)
    if k <= 1 or y.size == 0:
        return y
    k = min(k, y.size)
    kernel = np.ones(k, dtype=float)
    return np.convolve(y, kernel, mode="same") / np.convolve(
        np.ones_like(y), kernel, mode="same"
    )


def classify(
    motion_level: np.ndarray,
    unknown: np.ndarray,
    *,
    baseline_dev: np.ndarray | None = None,
    motion_ratio: np.ndarray | None = None,
    baseline_dev_threshold: float | None = None,
    motion_frac_hi: float = DEFAULT_MOTION_FRAC_HI,
    motion_ratio_hi: float = DEFAULT_MOTION_RATIO_HI,
) -> list[str]:
    """Reduce the per-window quantities to one verdict each.

    Order is the whole content of this function. ``unknown`` is tested first
    because a window built mostly from invented samples has no verdict to
    give, and the failure that matters is reporting it as ``empty``. Motion is
    tested next: a walking person is present regardless of what their channel
    offset looks like, so gross motion decides on its own, and it decides on
    either the absolute fractional change or its multiple of the room's own
    floor -- whichever fires, because the absolute one demonstrably does not
    fire on every radio. Only then does the channel offset get to speak.

    ``empty`` is the weakest claim and is made last, and only when there is a
    reference to make it against. Without one this returns ``unknown`` for
    every window that is not moving: absence measured against nothing is not a
    measurement. The breathing score is not consulted at all -- it is
    evidence, reported alongside, and it does not vote.
    """
    motion_level = np.asarray(motion_level, dtype=float)
    unknown = np.asarray(unknown, dtype=bool)
    n = motion_level.size

    dev = (
        np.full(n, np.nan)
        if baseline_dev is None
        else np.asarray(baseline_dev, dtype=float)
    )
    ratio = (
        np.full(n, np.nan)
        if motion_ratio is None
        else np.asarray(motion_ratio, dtype=float)
    )
    referenced = baseline_dev_threshold is not None and np.isfinite(
        baseline_dev_threshold
    )

    states: list[str] = []
    for m, u, d, r in zip(motion_level, unknown, dev, ratio):
        if u or not np.isfinite(m):
            states.append(STATE_UNKNOWN)
        elif m > motion_frac_hi or (np.isfinite(r) and r > motion_ratio_hi):
            states.append(STATE_MOVING)
        elif not referenced or not np.isfinite(d):
            states.append(STATE_UNKNOWN)
        elif d > baseline_dev_threshold:
            states.append(STATE_PRESENT)
        else:
            states.append(STATE_EMPTY)
    return states


def presence_windows(
    ratio: np.ndarray,
    fs: float,
    *,
    fabricated: np.ndarray | None = None,
    channel: str = "complex",
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    rate_band_rpm: tuple[float, float] = DEFAULT_RATE_BAND_RPM,
    bandpass_hz: tuple[float, float] = DEFAULT_BANDPASS_HZ,
    motion_frac_lo: float = DEFAULT_MOTION_FRAC_LO,
    motion_frac_hi: float = DEFAULT_MOTION_FRAC_HI,
    tonality_flat_lo: float = DEFAULT_TONALITY_FLAT_LO,
    tonality_flat_hi: float = DEFAULT_TONALITY_FLAT_HI,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    smooth_windows: int = DEFAULT_SMOOTH_WINDOWS,
    present_threshold: float = DEFAULT_PRESENT_THRESHOLD,
    reference: dict[str, Any] | None = None,
    baseline_dev_k: float = DEFAULT_BASELINE_DEV_K,
    motion_ratio_hi: float = DEFAULT_MOTION_RATIO_HI,
) -> dict[str, Any]:
    """Sliding-window motion and breathing analysis of a uniformly sampled ratio.

    *ratio* is ``(n_samples, n_subcarriers)`` complex, already on a uniform
    grid at *fs* Hz -- see ``backend.doppler.resample_uniform``, whose
    ``fabricated`` mask is what should be passed here. Returns one entry per
    window in each of ``time_s``, ``score``, ``periodicity``, ``tonality``,
    ``motion_gate``, ``motion_level``, ``rate_rpm``, ``unknown`` and ``state``,
    plus scalar ``fs_hz``, ``params`` and a ``warnings`` list.

    The score is a product of three terms in ``0..1`` and every one of them is
    a veto:

    * **periodicity** -- the height of the autocorrelation peak inside the
      respiration lag band, over the bandpassed channel, averaged across
      subcarriers weighted by each subcarrier's own in-band SNR. Weighting is
      what makes multiple subcarriers worth having: a real chest moves all of
      them coherently, so the weighted sum reinforces, while independent noise
      does not survive the average. Real breathing peaks around 0.5 rather
      than near 1.0, which is why the default verdict threshold is 0.25.

    * **tonality** -- spectral flatness of the in-band spectrum, inverted.
      Breathing is a narrow tone; an empty room is broadband *within the same
      band*. This exists because the obvious gate does not work: 1/f drift
      keeps in-band power high in an empty room, so an in-band-versus-out-band
      power ratio scores absence nearly as high as presence. Flatness
      separates them where energy cannot.

    * **motion gate** -- the median fractional channel change over the window,
      ramped down from ``motion_frac_lo`` to ``motion_frac_hi``. Walking is
      also periodic and would otherwise read as an enormous chest. Median
      rather than mean so a single posture shift does not close the gate for a
      whole window. Breathing itself moves the channel far less than
      ``motion_frac_lo``, so it passes at full weight.

    Raises ``ValueError`` when the geometry cannot support the requested band
    -- too few samples, or a window too short for the slowest rate asked for.
    Both are the caller's parameters rather than a fault in the data.
    """
    ratio = np.asarray(ratio)
    if ratio.ndim != 2:
        raise ValueError(f"ratio must be 2-D (n_samples, n_sc), got {ratio.shape}")
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")

    n_samples = int(ratio.shape[0])
    rpm_lo, rpm_hi = float(rate_band_rpm[0]), float(rate_band_rpm[1])
    if not 0 < rpm_lo < rpm_hi:
        raise ValueError(f"rate band must satisfy 0 < lo < hi, got {rate_band_rpm}")
    band_lo_hz, band_hi_hz = rpm_lo / 60.0, rpm_hi / 60.0

    if fabricated is None:
        fab = np.zeros(n_samples, dtype=bool)
    else:
        fab = np.asarray(fabricated, dtype=bool)
        if fab.shape != (n_samples,):
            raise ValueError(
                f"fabricated must be 1-D of length {n_samples}, got {fab.shape}"
            )

    # Profiles are taken at full subcarrier width, before the dead bins are
    # dropped: a reference and a window must line up bin for bin, and the two
    # need not agree about which bins are dead. Everything downstream of here
    # averages across subcarriers and so works on the live subset instead.
    ratio_full = ratio
    live = live_subcarriers(ratio)
    if not live.any():
        raise ValueError("no subcarrier in this range carries a CSI ratio")
    ratio = ratio[:, live]

    nyq = fs / 2.0
    if band_hi_hz >= nyq:
        raise ValueError(
            f"{rpm_hi:g} rpm is {band_hi_hz:.2f} Hz, at or above this capture's "
            f"{nyq:.2f} Hz Nyquist ({fs:.2f} Hz frame rate) -- it would alias"
        )

    win = max(MIN_WINDOW_SAMPLES, int(round(window_seconds * fs)))
    # Clamp rather than refuse. Zooming past the window length is ordinary use
    # of a linked time axis, and the window actually used is reported back.
    win = min(win, n_samples)
    if win < MIN_WINDOW_SAMPLES:
        raise ValueError(
            f"range holds {n_samples} samples at {fs:.2f} Hz, fewer than the "
            f"{MIN_WINDOW_SAMPLES}-sample minimum window"
        )
    hop = max(1, int(round(hop_seconds * fs)))

    lag_lo = max(1, int(round(fs * 60.0 / rpm_hi)))
    # Lags beyond half the window are not trusted: the detected period must
    # fit at least twice inside the window it was measured in. Longer lags
    # overlap too few samples and throw up spurious peaks, which in a presence
    # panel means claiming a sleeping occupant in an empty room.
    lag_hi = min(win // 2, int(round(fs * 60.0 / rpm_lo)))
    if lag_hi <= lag_lo:
        raise ValueError(
            f"a {win / fs:.1f} s window cannot resolve {rpm_lo:g}-{rpm_hi:g} rpm "
            f"at {fs:.2f} Hz -- lengthen the window or narrow the rate band"
        )
    # The slowest rate this geometry can actually reach, which is not always
    # the one that was asked for.
    rpm_floor_eff = 60.0 * fs / lag_hi

    sig = channel_from_ratio(ratio, channel)
    sig_bp = bandpass(sig, fs, bandpass_hz)
    frac_full = fractional_motion(ratio)

    is_complex = np.iscomplexobj(sig)
    # A real breathing waveform along a fixed complex direction has content at
    # both +f and -f, so the complex channel is masked on |f| over the full
    # two-sided spectrum. The real channels are conjugate-symmetric and use
    # the one-sided transform.
    if is_complex:
        freqs = np.abs(np.fft.fftfreq(win, d=1.0 / fs))
    else:
        freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    band_mask = (freqs >= band_lo_hz) & (freqs <= band_hi_hz)
    if not band_mask.any():
        raise ValueError(
            f"a {win / fs:.1f} s window resolves {fs / win:.3f} Hz, too coarse to "
            f"hold any bin inside {band_lo_hz:.2f}-{band_hi_hz:.2f} Hz"
        )
    taper = np.hanning(win)[:, None]

    starts = range(0, n_samples - win + 1, hop)
    centres, periodicity_l, tonality_l = [], [], []
    motion_gate_l, motion_level_l, rate_l, unknown_l = [], [], [], []
    baseline_dev_l: list[float] = []

    ref_profile = None if reference is None else np.asarray(reference["profile"])

    for start in starts:
        stop = start + win
        seg_raw = sig[start:stop, :]
        seg_raw = seg_raw - seg_raw.mean(axis=0, keepdims=True)
        seg_bp = sig_bp[start:stop, :]
        seg_bp = seg_bp - seg_bp.mean(axis=0, keepdims=True)

        # Per-subcarrier in-band SNR, measured on the *un*-bandpassed window:
        # after the bandpass there is nothing out of band left to compare
        # against, so the weights have to come from the raw spectrum.
        if is_complex:
            power = np.abs(np.fft.fft(seg_raw * taper, axis=0)) ** 2
        else:
            power = np.abs(np.fft.rfft(seg_raw * taper, axis=0)) ** 2
        weight = _weight_from_power(power, freqs, (band_lo_hz, band_hi_hz))

        ac = autocorr_columns(seg_bp)
        w_sum = float(weight.sum())
        combined = ac.mean(axis=1) if w_sum <= 0 else (ac * weight).sum(axis=1) / w_sum

        band_ac = combined[lag_lo : lag_hi + 1]
        peak = int(np.argmax(band_ac))
        periodicity = float(np.clip(band_ac[peak], 0.0, 1.0))
        peak_lag = lag_lo + peak
        rate = 60.0 * fs / peak_lag if peak_lag > 0 else float("nan")

        # Spectral flatness (Wiener entropy) of the in-band spectrum, averaged
        # over subcarriers first so a coherent tone reinforces before it is
        # measured. 0 = pure tone, 1 = flat.
        in_band = power[band_mask, :].mean(axis=1) + 1e-20
        flatness = float(np.exp(np.mean(np.log(in_band))) / in_band.mean())
        span = max(tonality_flat_hi - tonality_flat_lo, 1e-6)
        tonality = float(np.clip((tonality_flat_hi - flatness) / span, 0.0, 1.0))

        seg_frac = frac_full[start : stop - 1]
        # Median, not mean: a single posture shift should not close the gate
        # for a whole window. NaN-aware, so structural nulls cannot decide it.
        level = (
            float(np.nanmedian(seg_frac))
            if seg_frac.size and np.isfinite(seg_frac).any()
            else float("nan")
        )
        gate_span = max(motion_frac_hi - motion_frac_lo, 1e-6)
        gate = (
            float(np.clip((motion_frac_hi - level) / gate_span, 0.0, 1.0))
            if np.isfinite(level)
            else 0.0
        )

        baseline_dev_l.append(
            float("nan")
            if ref_profile is None
            else baseline_deviation(amplitude_profile(ratio_full[start:stop]), ref_profile)
        )

        centres.append((start + win / 2.0) / fs)
        periodicity_l.append(periodicity)
        tonality_l.append(tonality)
        motion_gate_l.append(gate)
        motion_level_l.append(level)
        rate_l.append(rate)
        unknown_l.append(bool(fab[start:stop].mean() > max_gap_fraction))

    periodicity_a = np.asarray(periodicity_l, dtype=float)
    tonality_a = np.asarray(tonality_l, dtype=float)
    motion_gate_a = np.asarray(motion_gate_l, dtype=float)
    motion_level_a = np.asarray(motion_level_l, dtype=float)
    rate_a = np.asarray(rate_l, dtype=float)
    unknown_a = np.asarray(unknown_l, dtype=bool)
    baseline_dev_a = np.asarray(baseline_dev_l, dtype=float)

    score = moving_average(
        periodicity_a * tonality_a * motion_gate_a, int(smooth_windows)
    )
    score = np.clip(score, 0.0, 1.0)
    # A window built mostly from invented samples reports nothing rather than
    # a number derived from the interpolator. Smoothing runs first, so the
    # blanking is not undone by a neighbour bleeding into it.
    score[unknown_a] = np.nan
    rate_a[unknown_a] = np.nan
    motion_level_a[unknown_a] = np.nan
    baseline_dev_a[unknown_a] = np.nan

    # The breathing score is evidence now, not a verdict, and this is the job
    # ``present_threshold`` keeps: it decides whether the rate this window
    # found is worth reporting at all. A rate read off an autocorrelation peak
    # that cleared nothing is a number with no claim behind it, so it is
    # blanked rather than drawn.
    breathing_a = np.isfinite(score) & (score > present_threshold)
    rate_a[~breathing_a] = np.nan

    if reference is None:
        motion_ratio_a = np.full(motion_level_a.shape, np.nan)
        dev_threshold: float | None = None
    else:
        motion_ratio_a = motion_level_a / float(reference["motion_floor"])
        dev_threshold = float(baseline_dev_k) * float(reference["dev_p95"])

    warnings: list[str] = []
    if reference is None:
        warnings.append(
            "no empty-room reference was given, so absence cannot be measured -- "
            "every window that is not moving is reported as unknown rather than "
            "as an empty room"
        )
    if rpm_floor_eff > rpm_lo * 1.05:
        warnings.append(
            f"a {win / fs:.1f} s window reaches only {rpm_floor_eff:.0f} rpm, not the "
            f"{rpm_lo:g} rpm requested -- slower breathing needs a longer window"
        )
    if fab.any():
        warnings.append(
            f"{100.0 * fab.mean():.1f}% of this range was interpolated across "
            "capture dropouts"
        )

    return {
        "time_s": np.asarray(centres, dtype=float),
        "score": score,
        "periodicity": periodicity_a,
        "tonality": tonality_a,
        "motion_gate": motion_gate_a,
        "motion_level": motion_level_a,
        "motion_ratio": motion_ratio_a,
        "baseline_dev": baseline_dev_a,
        "breathing": breathing_a,
        "rate_rpm": rate_a,
        "unknown": unknown_a,
        "state": classify(
            motion_level_a,
            unknown_a,
            baseline_dev=baseline_dev_a,
            motion_ratio=motion_ratio_a,
            baseline_dev_threshold=dev_threshold,
            motion_frac_hi=motion_frac_hi,
            motion_ratio_hi=motion_ratio_hi,
        ),
        "fs_hz": float(fs),
        "win": int(win),
        "hop": int(hop),
        "window_seconds": float(win / fs),
        "rpm_floor_eff": float(rpm_floor_eff),
        "baseline_dev_threshold": dev_threshold,
        "reference": None
        if reference is None
        else {
            "dev_p95": float(reference["dev_p95"]),
            "motion_floor": float(reference["motion_floor"]),
            "n_windows": int(reference["n_windows"]),
        },
        "params": {
            "channel": channel,
            "window_seconds": float(window_seconds),
            "hop_seconds": float(hop_seconds),
            "rate_band_rpm": [rpm_lo, rpm_hi],
            "bandpass_hz": [float(bandpass_hz[0]), float(bandpass_hz[1])],
            "motion_frac_lo": float(motion_frac_lo),
            "motion_frac_hi": float(motion_frac_hi),
            "tonality_flat_lo": float(tonality_flat_lo),
            "tonality_flat_hi": float(tonality_flat_hi),
            "max_gap_fraction": float(max_gap_fraction),
            "smooth_windows": int(smooth_windows),
            "present_threshold": float(present_threshold),
            "baseline_dev_k": float(baseline_dev_k),
            "motion_ratio_hi": float(motion_ratio_hi),
        },
        "warnings": warnings,
    }
