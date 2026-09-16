"""CSI ratio -> channel impulse response: IFFT along the subcarrier axis.

``csi_ratio_amplitude``/``csi_ratio_phase`` are a complex frequency response
— tpi1/tpi0's measured ratio, split into a dB magnitude and a phase, one
value per subcarrier. An inverse FFT of that response gives the channel impulse
response (CIR): a row per frame, delay tap along the columns instead of
subcarrier. Echoes at increasing round-trip delay show up as separated
peaks — a different read on the same data than anything the
frequency-domain panels give, and one the frequency-axis unwrapping in
``backend.phase`` cannot answer.

**This runs on the ratio, and the raw single channel was tried and dropped.**
An earlier version took the IFFT of rx0/tx0 — the honest channel impulse
response, and the only form absolute path structure could be read from. It
was removed because it answered no question this project actually has.

It could not give absolute delay: a single channel has nothing to cancel the
receiver's timing offset against, so the peak sat 14 taps off centre and
wandered with a standard deviation of 1.4 taps (5.3 m) frame to frame, which
is the receiver's clock rather than the room. And it could not give *change*
either. Scored against a capture with a known walk at 72-80 s, each tap's
deviation from its own long-run median came to 10.21 dB during the motion
against 10.09 while quiet -- a ratio of 1.01, no separation at all. Smoothing
from 1 to 190 frames took that noise from 10.2 dB down to 2.8 and never
lifted the ratio above 1.0, which is the signature of no signal to find
rather than of too much noise. The same per-packet timing jitter that moves
the peak redistributes energy across every tap on every frame, and aligning
on the peak corrects it only to the nearest whole tap -- 3.75 m at 80 MHz --
leaving the sub-tap remainder to swamp the room.

The ratio cancels that jitter exactly, because both transmit chains come out
of one packet, one receive chain and one timing recovery. On the same test it
separates 4.59 against 2.79, a ratio of 1.65.

What that costs is the absolute read, and the cost is real: an IFFT of
tpi1/tpi0 is the *difference* of two impulse responses rather than one
channel's, so a peak here is a delay at which the two chains disagree. Read
it as "something changed about 10 m further out than the direct path", never
as "a reflector sits 10 m away".

It is also weaker than what the frequency domain already gives --
``presence.fractional_motion`` separates the same two windows 2.8x against
this 1.65x. So this panel is for *where* in delay a change sits, not for
whether one happened.

Two things about the array must be respected before an IFFT means anything:

* **The array is DC-centred, not in FFT bin order.** Every metric in this
  pipeline already has index N//2 at DC — see the FeitCSI and MTK parser
  docstrings for why neither is ``fftshift``-ed on the way in.
  ``np.fft.ifft`` assumes the opposite (index 0 at DC, ascending positive
  frequencies, then wrapping to negative ones at the top), so the centred
  layout has to be undone with ``ifftshift`` immediately before the
  transform. Skipping this does not blur the result — every echo comes out
  at the wrong delay, silently.

* **A missing subcarrier is zero, not absent.** Null bins arrive as NaN —
  the DC/pilot/guard bins on an MTK capture, whatever CSIKit dropped as
  unusable on a FeitCSI one — and one NaN anywhere in an IFFT's input
  poisons every output sample, since each output tap sums over the whole
  row. Reading an unmeasured subcarrier as zero is the standard convention:
  it is what a receiver missing that tone would report, and it is exactly
  what the MTK hardware's own null-tone encoding already means (see
  ``backend.mtk``).

  On an MTK capture the null bins sit at their true positions in a uniform
  comb, so zero-filling them reconstructs the transmitted spectrum's shape
  faithfully. On a FeitCSI capture CSIKit has *deleted* the unusable
  subcarriers from the array rather than zeroing them in place, so the comb
  handed to this function is not perfectly uniform there; the impulse
  response it produces is still peaked at the true delay but carries extra
  sidelobe smearing from the gaps. Good enough to read off timing, not to
  trust to the last dB.

A frame with no primary stream decoded (e.g. rx0 absent on a group missing
that slot) has every subcarrier NaN, not just the null bins. Zero-filling
that row would compute the IFFT of silence and report it as a flat,
confident zero — indistinguishable from "measured and found nothing". It is
reported as NaN instead, matching the coverage the frame's own amplitude/
phase already had.
"""

from __future__ import annotations

import warnings

import numpy as np


def csi_to_cir(amplitude_db: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """abs(IFFT(H)) along the subcarrier axis, one row per frame.

    Inputs are what ``decode_frames`` produces for the primary channel:
    ``amplitude_db`` in 20*log10 dB, ``phase`` in radians. The output has
    the same shape, one magnitude per delay tap, in the channel's own
    linear (dimensionless) units — never dB, since a delay-domain response
    is genuinely zero between echoes and dB cannot show that.
    """
    amp = np.asarray(amplitude_db, dtype=np.float64)
    ph = np.asarray(phase, dtype=np.float64)
    live = np.isfinite(amp) & np.isfinite(ph)

    magnitude = np.where(live, 10.0 ** (amp / 20.0), 0.0)
    h = magnitude * np.exp(1j * np.where(live, ph, 0.0))
    cir = np.fft.ifft(np.fft.ifftshift(h, axes=1), axis=1)

    out = np.abs(cir).astype(np.float32)
    out[~live.any(axis=1)] = np.nan
    return out


def csi_to_cir_centred(amplitude_db: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """``csi_to_cir``, re-centred for display on this app's axis convention.

    Raw ``csi_to_cir`` puts delay 0 first (index 0) and ascends from there —
    the ordinary DSP convention, and the right one for anything that wants
    to *index* a specific delay. Every frequency-domain metric in this
    pipeline is already plotted DC-centred (see the FeitCSI and MTK parser
    docstrings), so an ``np.fft.fftshift`` here lets the CIR panel reuse
    that same centred axis without the frontend needing to know delay from
    subcarrier — the peak lands off-centre by this channel's own timing
    offset (see the module docstring) rather than at the centre itself, but
    the axis machinery is shared either way. Taps that wrap past the row's
    far edge are the DFT's circularity, not a physical acausal path, exactly
    mirroring how negative subcarrier index is not a second physical band
    on the frequency-domain panels.
    """
    return np.fft.fftshift(csi_to_cir(amplitude_db, phase), axes=1)


# Floor under the relative response, so a tap with no energy becomes a very
# quiet number rather than -inf. -120 dB is far below anything the fixed
# range draws.
CIR_DB_FLOOR = 1e-6

# The fixed colour range, in dB below each frame's own peak. Measured over
# fourteen September captures on the CROPPED tiles the panel draws, 137k
# cells: p5 -29.8, p25 -21.8, p50 -11.3, p75 -4.4, p95 0.0. This holds 95.3%
# of them and spends 58% of the ramp on the interquartile range; the 4.7% it
# clips is at the floor, where the taps are noise.
#
# The number was -50 until the crop landed, and that was a real mistake worth
# recording. -50 was measured on the UNCROPPED 256-row tile, where most rows
# are far taps sitting near the noise floor and drag the median to -40.9. The
# crop then removed exactly those rows and kept the strong ones near the peak,
# which moved the median to -11.3 -- so the range was set from a distribution
# the panel no longer draws, and everything landed in the top 40% of the ramp
# as one yellow wash. Re-measure the scale whenever the rows change.
#
# Nothing clips at the top: 0 dB is each frame's own peak, so no cell can
# exceed it by construction. About 1 in 17 cells sits exactly at 0, which is
# one peak row per column and is structural rather than saturation.
CIR_SCALE_DB: tuple[float, float] = (-30.0, 0.0)

# Metres of EXCESS path length per delay tap, at the 80 MHz these captures
# use: c / bandwidth. One tap is 3.75 m, which is the resolution limit and no
# amount of processing improves it.
CIR_TAP_METRES = 3.75

# Taps kept either side of the centre. The rest is not the room.
#
# A tap is 3.75 m of excess path. A 5x5 m room has a 7.07 m diagonal, so a
# single-bounce echo can be at most about 14 m longer than the direct path --
# under four taps. Serving all 256 drew 960 m of range for a room that can
# fill four taps of it, and put the whole of the room inside two pixels at the
# centre of a panel otherwise showing nothing.
#
# What the far taps do hold is not echoes. A ratio H1(f)/H0(f) is a division
# in frequency, which is a deconvolution in time and not compactly supported,
# so its transform spreads energy across the whole row as a matter of
# arithmetic; the 22 zero-filled null bins leak into it as well. Measured on
# 20260911_095127 the median tap 8 out (30 m of excess path, geometrically
# impossible in this room) still reads -10.7 dB. Drawing it invites reading it
# as an echo.
#
# Eight rather than four, to leave room to see that the spread is there rather
# than cropping to the point where it cannot be judged.
CIR_CROP_TAPS = 8


def cir_relative_db(amplitude_db: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Centred CIR as dB below each frame's own peak.

    Linear magnitude is what this module served until it was measured against
    a real panel: a delay response spans three orders of magnitude, so a
    linear scale saturates the direct path into one flat block and buries
    every echo under it. The earlier docstring argued dB cannot show a
    response that is genuinely zero between echoes, which is true and is what
    ``CIR_DB_FLOOR`` answers -- an empty tap lands at the bottom of the ramp
    rather than at negative infinity.

    Normalised per frame rather than globally, so the number means "how far
    below this frame's strongest path", which is comparable across frames,
    across captures, and across any gain sitting in front of the receiver.
    """
    cir = csi_to_cir_centred(amplitude_db, phase)
    # A frame with no channel at all is legitimately all-NaN here -- see the
    # module docstring -- so the empty-slice warning says only that, and the
    # NaN it returns is what the caller wants.
    warnings.filterwarnings("ignore", r"All-NaN slice encountered", RuntimeWarning)
    # Normalised against the WHOLE row's peak before cropping, so the 0 dB
    # reference is the frame's strongest path wherever it landed -- cropping
    # first would renormalise to whatever survived the crop.
    peak = np.nanmax(np.where(np.isfinite(cir), cir, np.nan), axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = cir / np.where(np.isfinite(peak) & (peak > 0), peak, np.nan)
        out = 20.0 * np.log10(np.maximum(rel, CIR_DB_FLOOR))
    out = np.where(np.isfinite(cir) & np.isfinite(out), out, np.nan).astype(np.float32)

    centre = cir.shape[1] // 2
    lo = max(0, centre - CIR_CROP_TAPS)
    hi = min(cir.shape[1], centre + CIR_CROP_TAPS + 1)
    return out[:, lo:hi]


def cir_rows() -> int:
    """Rows ``cir_relative_db`` emits, for callers that must size a grid.

    An empty chunk has no data to take its height from, so the tile pipeline
    needs this before any frame is decoded.
    """
    return 2 * CIR_CROP_TAPS + 1
