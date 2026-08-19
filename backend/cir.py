"""CSI ratio -> channel impulse response: IFFT along the subcarrier axis.

The ratio metrics are a complex frequency response split into a dB
magnitude and a phase, both taken per subcarrier. An inverse FFT of that
response gives the channel impulse response (CIR): a row per frame, delay
tap along the columns instead of subcarrier. Echoes at increasing
round-trip delay show up as separated peaks — a different read on the same
ratio than anything the frequency-domain panels give, and one the
frequency-axis unwrapping in ``backend.phase`` cannot answer.

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

A frame with no ratio at all (single-stream) has every subcarrier NaN, not
just the null bins. Zero-filling that row would compute the IFFT of silence
and report it as a flat, confident zero — indistinguishable from "measured
and found nothing". It is reported as NaN instead, matching the coverage
the frame's ratio already had.
"""

from __future__ import annotations

import numpy as np


def ratio_to_cir(ratio_amplitude_db: np.ndarray, ratio_phase: np.ndarray) -> np.ndarray:
    """abs(IFFT(ratio)) along the subcarrier axis, one row per frame.

    Inputs are what ``decode_frames`` and the swap correction produce:
    ``ratio_amplitude_db`` in 20*log10 dB, ``ratio_phase`` in radians. The
    output has the same shape, one magnitude per delay tap, in the ratio's
    own linear (dimensionless) units — never dB, since a delay-domain
    response is genuinely zero between echoes and dB cannot show that.
    """
    amp = np.asarray(ratio_amplitude_db, dtype=np.float64)
    phase = np.asarray(ratio_phase, dtype=np.float64)
    live = np.isfinite(amp) & np.isfinite(phase)

    magnitude = np.where(live, 10.0 ** (amp / 20.0), 0.0)
    h = magnitude * np.exp(1j * np.where(live, phase, 0.0))
    cir = np.fft.ifft(np.fft.ifftshift(h, axes=1), axis=1)

    out = np.abs(cir).astype(np.float32)
    out[~live.any(axis=1)] = np.nan
    return out


def ratio_to_cir_centred(
    ratio_amplitude_db: np.ndarray, ratio_phase: np.ndarray
) -> np.ndarray:
    """``ratio_to_cir``, re-centred for display on this app's axis convention.

    Raw ``ratio_to_cir`` puts the zero-delay tap first (index 0) and delay
    ascends from there — the ordinary DSP convention, and the right one for
    anything that wants to *index* a specific delay. It is also the wrong
    shape for a plot: a channel whose true delay sits a fraction of a tap
    before zero wraps circularly to the *last* tap, so the single strongest
    peak in a LOS-dominated capture shows up split across the two opposite
    edges of the row instead of together. Measured on both captures on hand,
    that split carries 15-30% of the peak's neighbourhood weight — visibly
    two peaks where there is physically one.

    Every frequency-domain metric in this pipeline is already plotted DC-
    centred (see the FeitCSI and MTK parser docstrings), so an
    ``np.fft.fftshift`` here — moving the zero-delay tap from index 0 to the
    row's centre — reunites the split peak *and* lets the CIR panel reuse
    that existing centred axis without the frontend needing to know delay
    from subcarrier. Taps to one side of centre are positive delay (real
    echoes); taps to the other are small negative delay, an artifact of the
    DFT's circularity rather than a physical acausal path, exactly mirroring
    how negative subcarrier index is not a second physical band on the
    frequency-domain panels.
    """
    return np.fft.fftshift(ratio_to_cir(ratio_amplitude_db, ratio_phase), axes=1)
