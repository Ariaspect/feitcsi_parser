"""Signed two-scale Doppler on the complex CSI ratio.

``backend.doppler.stft_average`` takes real input and returns a one-sided
spectrum, so its Doppler is unsigned -- approaching and receding motion land
on the same row. That is fine for a magnitude heatmap and useless for the two
things this module exists for: telling a chest's symmetric sidebands from a
body's one-sided shift, and using the sign of a known movement to check that
the ratio pipeline is not conjugated somewhere.

The sign is meaningful here specifically because the signal is a *ratio*. Both
chains share an oscillator, so the carrier frequency offset that would
otherwise smear and bias the whole axis divides out, and what is left is
geometry.

Two configurations, not one. A window that resolves a 2 Hz body cannot resolve
a 0.25 Hz chest and vice versa -- 0.033 Hz resolution needs 30 s, and 30 s of
a walk-through is a smear. The spec's numbers assume roughly 100 Hz sampling;
this link delivers 18.12, so ``stft_config`` derives what is actually
reachable and says which requests it could not meet.

Respiration is a sideband, not a shift. Chest displacement is a few
millimetres against a 5.75 cm wavelength, so the target oscillates in place
rather than traversing the wave. Writing the ratio as
``H_s + A*exp(j*(4*pi/lambda)*(d0 + a*sin(2*pi*f_b*t)))`` and expanding in
Bessel functions puts symmetric pairs at ``+/-f_b``, ``+/-2f_b``, ... around
the static term. A detector looking for one peak at one velocity finds
nothing at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.preprocess import WAVELENGTH_M

# Spec: motion 0.25-1 s window, 1-4 Hz resolution, +/-50 Hz axis, 0.1 s hop.
# Breathing 20-40 s window, 0.025-0.05 Hz resolution, +/-1 Hz axis, 1-2 s hop.
STFT_PRESETS: dict[str, dict[str, Any]] = {
    "motion": {
        "window_seconds": 1.0,
        "hop_seconds": 0.1,
        "display_hz": (-50.0, 50.0),
        "min_window_samples": 12,
        "zero_pad_factor": 4,
    },
    "breathing": {
        "window_seconds": 30.0,
        "hop_seconds": 2.0,
        "display_hz": (-1.0, 1.0),
        "min_window_samples": 64,
        "zero_pad_factor": 2,
    },
}

# Blackman-Harris. The first sidelobe is -92 dB against a Hann window's -31 and
# a rectangular window's -13, and a respiration sideband sits 30-40 dB under
# the static peak -- with anything shallower the leak from DC covers it.
TAPERS = ("blackmanharris", "hann", "rect")

# A sideband pair must be this far above the in-band median to count.
MIN_SIDEBAND_PROMINENCE = 4.0
# Upper and lower sidebands within this ratio of each other count as symmetric.
MIN_SYMMETRY = 0.5
# A candidate within this fraction of an integer multiple of the fundamental is
# that fundamental's harmonic, not another target.
HARMONIC_TOLERANCE = 0.12

DEFAULT_MAX_GAP_FRACTION = 0.5

# Width of the running median that estimates the in-band background. Wide
# enough that a genuine sideband does not become its own baseline, narrow
# enough to follow the high-pass slope the detrend leaves behind.
_BACKGROUND_BINS = 21


def _running_median(x: np.ndarray, width: int) -> np.ndarray:
    """Centred running median, edge-clamped rather than padded."""
    n = x.size
    width = max(3, min(int(width), n))
    out = np.empty(n, dtype=float)
    half = width // 2
    for i in range(n):
        out[i] = np.median(x[max(0, i - half) : min(n, i + half + 1)])
    return out


def velocity_of(doppler_hz: float | np.ndarray) -> float | np.ndarray:
    """Radial velocity for a Doppler shift: ``v = f_d * lambda / 2``.

    The factor of two is the round trip. At this carrier the whole reachable
    axis, +/-9.06 Hz, is +/-0.26 m/s -- slower than a walk, which is why gross
    motion aliases and is read here as broadband rather than as a velocity.
    """
    return np.asarray(doppler_hz) * WAVELENGTH_M / 2.0


def stft_config(fs: float, purpose: str) -> dict[str, Any]:
    """Window, hop and axis for one scale, against what *fs* can deliver.

    Clamps rather than approximates, and reports what it could not give. A
    display axis wider than Nyquist is not a wider view of the same data, it
    is empty canvas that reads as an absence of fast motion.
    """
    if purpose not in STFT_PRESETS:
        raise ValueError(f"purpose must be one of {tuple(STFT_PRESETS)}, got {purpose!r}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")

    preset = STFT_PRESETS[purpose]
    nyquist = fs / 2.0
    warnings: list[str] = []

    win = int(round(preset["window_seconds"] * fs))
    if win < preset["min_window_samples"]:
        needed = preset["min_window_samples"] / fs
        warnings.append(
            f"a {preset['window_seconds']:g} s window is {win} samples at {fs:.2f} Hz; "
            f"lengthened to {needed:.2f} s to reach "
            f"{preset['min_window_samples']} samples"
        )
        win = preset["min_window_samples"]

    lo, hi = preset["display_hz"]
    if hi > nyquist:
        warnings.append(
            f"the {hi:g} Hz axis this scale asks for is above the {nyquist:.2f} Hz "
            f"Nyquist of a {fs:.2f} Hz capture -- clamped, and motion faster than "
            f"{abs(velocity_of(nyquist)):.2f} m/s aliases into it rather than "
            "appearing beyond it"
        )
        lo, hi = -nyquist, nyquist

    hop = max(1, int(round(preset["hop_seconds"] * fs)))
    zero_pad = int(win * preset["zero_pad_factor"])
    nfft = win + zero_pad

    return {
        "purpose": purpose,
        "win": win,
        "hop": hop,
        "zero_pad": zero_pad,
        "window_seconds": win / fs,
        "hop_seconds": hop / fs,
        "resolution_hz": fs / win,
        "bin_hz": fs / nfft,
        "display_hz": (float(lo), float(hi)),
        "nyquist_hz": nyquist,
        "warnings": warnings,
    }


def _taper(name: str, win: int) -> np.ndarray:
    if name not in TAPERS:
        raise ValueError(f"taper must be one of {TAPERS}, got {name!r}")
    if name == "rect":
        return np.ones(win)
    if name == "hann":
        return np.hanning(win)
    # Blackman-Harris, 4-term, written out rather than imported so the
    # coefficients are visible next to the sidelobe claim above.
    n = np.arange(win)
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    return (
        a[0]
        - a[1] * np.cos(2 * np.pi * n / (win - 1))
        + a[2] * np.cos(4 * np.pi * n / (win - 1))
        - a[3] * np.cos(6 * np.pi * n / (win - 1))
    )


def stft_complex(
    sig: np.ndarray,
    fs: float,
    win: int,
    hop: int,
    *,
    taper: str = "blackmanharris",
    zero_pad: int = 0,
    fabricated: np.ndarray | None = None,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided power spectrogram, averaged over subcarriers.

    Returns ``(spectrogram, freqs, times)`` with *freqs* ascending from
    ``-fs/2`` to ``+fs/2`` and *spectrogram* shaped ``(n_freqs, n_windows)``.

    Averaged in **power**, not amplitude, and after the transform rather than
    before it. Incoherent averaging is what buys SNR across the array without
    needing the per-subcarrier phase to agree -- which it does not, since each
    subcarrier sees the target through its own geometry. It also sidesteps the
    sign-alignment problem entirely: power has no sign to align.

    The per-window mean is removed first. The static path is enormous compared
    to anything moving, and its leakage through the taper is the thing that
    buries respiration sidebands.
    """
    sig = np.asarray(sig)
    if sig.ndim != 2:
        raise ValueError(f"sig must be 2-D (n_samples, n_sc), got {sig.shape}")
    if win < 2 or hop < 1:
        raise ValueError(f"need win >= 2 and hop >= 1, got {win} and {hop}")
    n_samples, n_sc = sig.shape
    if n_samples < win:
        raise ValueError(f"{n_samples} samples is shorter than the {win}-sample window")

    n_out = (n_samples - win) // hop + 1
    starts = np.arange(n_out) * hop
    offsets = np.arange(win)
    window = _taper(taper, win)[None, :]
    nfft = win + int(zero_pad)
    nfft += nfft % 2

    blank = np.zeros(n_out, dtype=bool)
    if fabricated is not None:
        fab = np.asarray(fabricated, dtype=bool)
        if fab.shape != (n_samples,):
            raise ValueError(
                f"fabricated must be 1-D of length {n_samples}, got {fab.shape}"
            )
        blank = fab[starts[:, None] + offsets].mean(axis=1) > max_gap_fraction

    acc = np.zeros((nfft, n_out), dtype=float)
    contributing = 0
    for col in range(n_sc):
        series = sig[:, col]
        if not np.isfinite(series).any():
            continue
        seg = series[starts[:, None] + offsets]
        seg = seg - seg.mean(axis=1, keepdims=True)
        acc += (np.abs(np.fft.fft(seg * window, n=nfft, axis=1)) ** 2).T
        contributing += 1

    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)
    order = np.argsort(freqs)
    freqs = freqs[order]
    if contributing == 0:
        return np.full((nfft, n_out), np.nan), freqs, (starts + win / 2.0) / fs

    spec = acc[order] / contributing
    spec[:, blank] = np.nan
    return spec, freqs, (starts + win / 2.0) / fs


def find_sidebands(
    mean_power: np.ndarray,
    freqs: np.ndarray,
    *,
    band_hz: tuple[float, float] = (0.1, 0.6),
    min_prominence: float = MIN_SIDEBAND_PROMINENCE,
    min_symmetry: float = MIN_SYMMETRY,
) -> dict[str, Any]:
    """Look for a symmetric ``+/-f`` pair inside *band_hz*, and reject harmonics.

    Symmetry is the discriminating feature. A body moving toward the receiver
    puts power on one side only; a chest oscillating in place puts matched
    power on both. Requiring the pair is what stops a slow drift on one side
    from being read as respiration.

    Harmonic rejection matters at realistic amplitudes: the modulation index
    ``beta = 4*pi*a/lambda`` is about 1 radian for 5 mm of chest travel at this
    wavelength, so the second harmonic is strong. Untreated, a 0.5 Hz harmonic
    of a 15 rpm chest reads as a second person breathing at 30 rpm.
    """
    mean_power = np.asarray(mean_power, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if mean_power.shape != freqs.shape:
        raise ValueError(
            f"power {mean_power.shape} and freqs {freqs.shape} must match"
        )

    lo, hi = float(band_hz[0]), float(band_hz[1])
    positive = np.flatnonzero((freqs >= lo) & (freqs <= hi))
    if positive.size == 0:
        raise ValueError(f"no bin inside {lo:g}-{hi:g} Hz")

    # Pair each positive candidate with its mirror, and score the pair by its
    # weaker half -- a pair is only as good as the side that is harder to see.
    pair_power = np.empty(positive.size)
    symmetry = np.empty(positive.size)
    for i, idx in enumerate(positive):
        mirror = int(np.argmin(np.abs(freqs + freqs[idx])))
        up, down = mean_power[idx], mean_power[mirror]
        pair_power[i] = min(up, down)
        symmetry[i] = min(up, down) / max(up, down) if max(up, down) > 0 else 0.0

    # Prominence against a *local* background, not the band median. The
    # detrend that produced this signal is a high-pass whose corner sits just
    # below the band, so in-band power falls monotonically with frequency and
    # the largest bin is always the lowest one. Measured on
    # captures/lg/20260825_185637.bin that put the "sideband" at exactly 6.00
    # rpm -- the band edge -- in three regimes out of four, including an empty
    # room. A local baseline removes the slope and leaves only real peaks.
    background = _running_median(pair_power, _BACKGROUND_BINS)
    prominence = np.divide(
        pair_power, background,
        out=np.zeros_like(pair_power), where=background > 0,
    )

    order = np.argsort(pair_power)[::-1]
    best = int(order[0])
    fundamental = float(freqs[positive[best]])

    harmonics: list[int] = []
    for i in order[1:]:
        f = float(freqs[positive[i]])
        if prominence[i] < min_prominence:
            continue
        multiple = f / fundamental if fundamental > 0 else 0.0
        if multiple > 1.5 and abs(multiple - round(multiple)) < HARMONIC_TOLERANCE:
            harmonics.append(int(round(multiple)))

    found = bool(
        prominence[best] >= min_prominence and symmetry[best] >= min_symmetry
    )
    return {
        "found": found,
        "hz": fundamental,
        "rpm": fundamental * 60.0,
        "prominence": float(prominence[best]),
        "symmetry": float(symmetry[best]),
        "harmonics_rejected": sorted(set(harmonics)),
    }


def doppler_sign_bias(spec: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Per-column ``(P+ - P-) / (P+ + P-)``: which way the channel is moving.

    Diagnostic before it is a feature. A known approach and a known departure
    must come out with opposite signs; if they do not, the ratio is conjugated
    somewhere upstream and every direction this pipeline reports is backwards.

    Respiration should sit near zero here, since its sidebands are symmetric
    by construction -- which makes this a way of telling a chest from travel
    as well as a way of checking the plumbing.
    """
    spec = np.asarray(spec, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if spec.shape[0] != freqs.size:
        raise ValueError(f"spec has {spec.shape[0]} rows against {freqs.size} freqs")

    up = spec[freqs > 0].sum(axis=0)
    down = spec[freqs < 0].sum(axis=0)
    total = up + down
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, (up - down) / total, np.nan)
