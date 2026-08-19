"""Raw CSI -> channel impulse response: IFFT along the subcarrier axis.

``amplitude``/``phase`` are a complex frequency response — rx0/tx0's
measured channel, split into a dB magnitude and a phase, one value per
subcarrier. An inverse FFT of that response gives the channel impulse
response (CIR): a row per frame, delay tap along the columns instead of
subcarrier. Echoes at increasing round-trip delay show up as separated
peaks — a different read on the same data than anything the
frequency-domain panels give, and one the frequency-axis unwrapping in
``backend.phase`` cannot answer.

This is deliberately the *raw* channel, not the rx1/rx0 ratio ``backend.ratio``
corrects: dividing two chains cancels the receiver's CFO/SFO and per-packet
timing offset, which is exactly why the ratio's own IFFT (an earlier version
of this module) came out so cleanly centred on zero delay. A single channel
has none of that cancellation, so this CIR is not zero-referenced: it shows
propagation delay plus whatever uncalibrated hardware/timing offset the
receiver adds on top, and that combined offset moves a little from frame to
frame as CFO/SFO drift. Measured on both captures on hand — 1500 frames of
a single sender, no swap-correction applicable since there is no second
chain to be swapped with — the peak sits at a roughly constant offset from
centre (13 taps out of 256 on an MTK capture, 8 out of 242 on a FeitCSI one)
with a few taps of frame-to-frame spread (std 3.1 and 1.7 taps
respectively). Read *relative* delay between echoes off this panel — where
the second bump sits relative to the main peak — not absolute time-of-flight
from the row's centre; that is what the ratio-based CIR was for; a single
channel's cancellation-free timing offset makes an absolute read unsound.

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
