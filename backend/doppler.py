"""Irregularly-sampled CSI -> uniform time grid -> Doppler spectrogram.

An STFT assumes samples arrive at a constant rate. CSI frames do not: on the
captures this was built against the median interval ranges from 56 ms to
197 ms, the 1st percentile can be under half a millisecond, and one FeitCSI
capture holds a single 23-second hole. So the series is resampled onto a
uniform grid before any FFT touches it.

The grid's rate comes from the *median* inter-frame interval, never from a
requested display width. Deriving it from width is a trap worth naming: 2048
columns spanning 113 s of a 5 Hz capture implies an 18 Hz sample rate, and the
spectrogram then shows peaks above that capture's own 2.55 Hz Nyquist --
manufactured entirely by the interpolator. The same mistake decimates a faster
capture and silently truncates the top of its band. Both were observed while
prototyping this module.

Gaps are bridged by interpolation, but only up to a point. Bridging every gap
unconditionally is the obvious choice and is what a notebook doing this by hand
usually does -- a missed packet or two amid a 50 ms cadence is genuinely safe to
fill in. It stops being safe at scale: this repo's FeitCSI captures hold six
holes over 10 seconds, one of 22.9 s, and interpolating those produces a
perfectly flat stretch. Flat reads as "no motion", not as "no data", which on a
presence panel is the one lie that matters -- an empty room is exactly the
signal being looked for.

So the rule is proportional: a window is kept when the fabricated samples are a
minority of it and blanked when they dominate. Measured on
csi_20260813_030001.dat (130 gaps, 6.3% dead time), blanking on *any* gap kills
32.5% of columns at a 30 s window; blanking above 50% of a window kills 3.8%,
while the 22.9 s hole still shows as a hole.

What this can and cannot see: Doppler shift is f_d = 2v/lambda, so at 5 GHz a
1 m/s hand movement sits near 33 Hz -- far above the +/-2.5 to +/-8.9 Hz
Nyquist of every capture on hand. Within reach are respiration (0.2-0.5 Hz),
slow torso motion, and presence. Measured on captures/20260821_170002.bin, a
0.08-0.28 Hz line appears at 2.4-4.2x contrast on every slice; the 07:00
capture from the following morning is flat at 1.05-1.23x. Occupied versus
empty is legible; gesture and gait are not.
"""

from __future__ import annotations

import numpy as np

# A gap wider than this many times the 95th-percentile inter-frame interval is
# a dropout rather than jitter. Matches the convention backend.tiles already
# uses when filling tile columns, so the two panels agree about what a hole is.
DEFAULT_GAP_FACTOR = 2.0

# A window is blanked once more than this fraction of it is interpolated across
# dropouts. Below it the fabricated samples are a minority the Hann taper and
# per-window detrend absorb; above it the column would mostly be invention.
DEFAULT_MAX_GAP_FRACTION = 0.5

# Extra zero-padded samples appended to each window before the FFT. Adds no
# information -- it interpolates the frequency axis -- but a spectrogram drawn
# on a canvas reads far better with a finer grid than the raw window length
# gives, especially at the short windows that suit respiration.
DEFAULT_ZERO_PAD = 512


def uniform_grid(times: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(grid_times, fs)`` spanning *times* at the median frame rate.

    *times* must be sorted ascending, as a FrameIndex produces them.
    """
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        raise ValueError("need at least 2 frames to infer a sample rate")

    dt = np.diff(times)
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("frames carry no positive time deltas")

    step = float(np.median(dt))
    fs = 1.0 / step
    span = float(times[-1] - times[0])
    n = int(np.floor(span / step)) + 1
    grid = times[0] + np.arange(n, dtype=float) * step
    return grid, fs


def gap_limit_for(times: np.ndarray, factor: float = DEFAULT_GAP_FACTOR) -> float:
    """Longest inter-frame interval still treated as jitter, not a dropout."""
    times = np.asarray(times, dtype=float)
    dt = np.diff(times)
    dt = dt[dt > 0]
    if dt.size == 0:
        return float("inf")
    return float(np.percentile(dt, 95)) * factor


def resample_uniform(
    times: np.ndarray,
    values: np.ndarray,
    grid_times: np.ndarray,
    gap_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample ``(n_frames, n_cols)`` *values* onto *grid_times*.

    Returns ``(samples, fabricated)``. *samples* is interpolated everywhere,
    including across dropouts; *fabricated* is a 1-D boolean mask over
    *grid_times*, True where a sample falls strictly inside a gap wider than
    *gap_limit* and is therefore invented rather than measured.

    Interpolating first and reporting the mask separately, rather than writing
    NaN here, is what lets the caller decide *proportionally* -- a window that
    is 2% fabricated is fine, one that is 90% fabricated is not. Writing NaN
    at this layer forces the whole window to die for a single missed packet.

    A column that is entirely non-finite on input stays entirely non-finite,
    and one dead subcarrier never blanks its neighbours: the finite mask is
    per column.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    grid_times = np.asarray(grid_times, dtype=float)

    if values.ndim != 2:
        raise ValueError(f"values must be 2-D (n_frames, n_cols), got {values.shape}")
    if values.shape[0] != times.shape[0]:
        raise ValueError(f"values has {values.shape[0]} rows for {times.shape[0]} times")

    out = np.empty((grid_times.size, values.shape[1]), dtype=float)
    for col in range(values.shape[1]):
        series = values[:, col]
        finite = np.isfinite(series)
        if not finite.any():
            out[:, col] = np.nan
            continue
        out[:, col] = np.interp(grid_times, times[finite], series[finite])

    # Mark samples inside a real dropout. Computed once against the frame times
    # rather than per column: a gap is a property of when frames arrived, not
    # of any one subcarrier.
    fabricated = np.zeros(grid_times.size, dtype=bool)
    dt = np.diff(times)
    for start, width in zip(times[:-1], dt):
        if width > gap_limit:
            fabricated |= (grid_times > start) & (grid_times < start + width)

    return out, fabricated


def stft_average(
    samples: np.ndarray,
    fs: float,
    win: int,
    hop: int,
    *,
    fabricated: np.ndarray | None = None,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    zero_pad: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Subcarrier-averaged magnitude spectrogram of ``(n_samples, n_cols)``.

    Returns ``(spectrogram, freqs)``. *spectrogram* is
    ``(win // 2 + 1, n_windows)`` with **row 0 = highest frequency**, matching
    the tile contract's "row 0 = highest subcarrier" convention so the same
    renderer draws both. *freqs* is ascending, ``0 .. fs/2``.

    Three things the maths requires, each of which silently ruins the output
    if skipped:

    * **Real input means a one-sided spectrum.** Amplitude and unwrapped phase
      are real, so their spectra are conjugate-symmetric and the negative half
      carries no information. ``rfft`` is correct, and the consequence is that
      Doppler here is *unsigned*: approaching and receding motion cannot be
      told apart. Signed Doppler would need the complex CSI.

    * **The per-window mean must go.** Measured on a real capture, DC holds
      47.8% of total spectrogram power undetrended and 0.2% detrended. Leaving
      it in makes bin 0 swamp the panel.

    * **A NaN anywhere in a window poisons that whole column**, because each
      output bin sums over the entire window. Which is why *fabricated* is a
      mask rather than NaN in the data: a column is blanked when more than
      *max_gap_fraction* of it was interpolated across dropouts, and kept
      otherwise. Blanking on the mere presence of a gap makes one missed
      packet cost an entire window -- 32.5% of columns on a real capture whose
      actual dead time is 6.3%.

    Two different things produce NaN and they are *not* handled alike. A
    column that is non-finite for the whole capture is a structural null --
    the DC/guard band, or a pilot CSIKit dropped -- and carries no signal at
    any time; measured on an MTK capture, 11 subcarriers of 256. Those are
    excluded from the average, because ``acc += nan`` would otherwise poison
    every bin of every window and blank the entire panel. A NaN confined to
    some windows is a real dropout in time, and that one must propagate.

    Subcarriers are accumulated one at a time rather than stacked. The stacked
    form is ``(n_cols, n_windows, win)``, a quarter of a gigabyte for a full
    capture; accumulating holds one subcarrier's windows at a time.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2-D (n_samples, n_cols), got {samples.shape}")
    if win < 2:
        raise ValueError("win must be at least 2")
    if hop < 1:
        raise ValueError("hop must be at least 1")
    if samples.shape[0] < win:
        raise ValueError(
            f"series of {samples.shape[0]} samples is shorter than the "
            f"{win}-sample window"
        )

    if zero_pad < 0:
        raise ValueError("zero_pad must not be negative")

    n_samples, n_cols = samples.shape
    n_out = (n_samples - win) // hop + 1
    starts = np.arange(n_out) * hop
    offsets = np.arange(win)
    taper = np.hanning(win)
    # Even transform length, so rfft has a true Nyquist bin and the axis really
    # ends at fs/2 rather than just short of it.
    nfft = win + zero_pad
    nfft += nfft % 2

    # Columns whose window is mostly invention, decided before any transform.
    blank = np.zeros(n_out, dtype=bool)
    if fabricated is not None:
        fab = np.asarray(fabricated, dtype=bool)
        if fab.shape != (n_samples,):
            raise ValueError(
                f"fabricated must be 1-D of length {n_samples}, got {fab.shape}"
            )
        blank = fab[starts[:, None] + offsets].mean(axis=1) > max_gap_fraction

    acc = np.zeros((nfft // 2 + 1, n_out), dtype=float)
    contributing = 0
    for col in range(n_cols):
        series = samples[:, col]
        if not np.isfinite(series).any():
            continue                                       # structural null
        seg = series[starts[:, None] + offsets]            # (n_out, win)
        seg = seg - seg.mean(axis=1, keepdims=True)        # detrend per window
        acc += np.abs(np.fft.rfft(seg * taper, n=nfft, axis=1)).T
        contributing += 1

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    if contributing == 0:
        return np.full((nfft // 2 + 1, n_out), np.nan), freqs

    spec = acc / contributing
    spec[:, blank] = np.nan
    # Row 0 = highest frequency, so the renderer's top-down row order puts fast
    # motion at the top of the panel.
    return spec[::-1, :], freqs
