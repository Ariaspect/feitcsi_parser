"""backend.cir: IFFT of the raw CSI into a delay-domain impulse response."""

from __future__ import annotations

import numpy as np

from backend.cir import csi_to_cir, csi_to_cir_centred


def _to_amp_phase(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a complex frequency response into the (amp_db, phase) pair
    ``csi_to_cir`` expects, i.e. the inverse of what it reconstructs."""
    amp_db = 20 * np.log10(np.abs(h))
    phase = np.angle(h)
    return amp_db, phase


def test_a_flat_spectrum_is_a_single_tap_at_zero_delay() -> None:
    """H[k] = 1 for all k is a perfect impulse: all energy at delay zero.

    csi_to_cir undoes the centring with ifftshift before the IFFT, so this
    also pins down that step is correct in direction — get it backwards and
    the peak would land at the last tap instead of the first.
    """
    n = 64
    h = np.ones((1, n), dtype=complex)
    amp_db, phase = _to_amp_phase(h)
    cir = csi_to_cir(amp_db, phase)
    assert cir.shape == (1, n)
    assert np.argmax(cir[0]) == 0
    np.testing.assert_allclose(cir[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(cir[0, 1:], 0.0, atol=1e-6)


def test_a_linear_phase_ramp_shifts_the_peak() -> None:
    """A pure delay in frequency domain is a single tap at that delay.

    H[k] = exp(-i*2*pi*k*d/n) for centred bin index k is a d-sample delay;
    the IFFT must place all the energy at tap d, not smear it.
    """
    n = 64
    d = 5
    k = np.fft.fftfreq(n, 1 / n)  # centred bin index, same axis ifftshift expects
    h = np.exp(-1j * 2 * np.pi * k * d / n)[None, :]
    amp_db, phase = _to_amp_phase(h)
    cir = csi_to_cir(amp_db, phase)
    assert np.argmax(cir[0]) == d
    np.testing.assert_allclose(cir[0, d], 1.0, atol=1e-6)


def test_a_null_subcarrier_is_zero_filled_not_nan() -> None:
    """One missing bin must not poison the whole row.

    A single NaN in a naive IFFT input would propagate to every output tap;
    here it is read as zero energy on that tone, so the result stays finite
    and close to the all-ones case with one bin dropped.
    """
    n = 32
    h = np.ones((1, n), dtype=complex)
    amp_db, phase = _to_amp_phase(h)
    amp_db[0, 10] = np.nan
    phase[0, 10] = np.nan
    cir = csi_to_cir(amp_db, phase)
    assert np.isfinite(cir).all()
    assert np.argmax(cir[0]) == 0


def test_a_frame_with_no_live_data_comes_back_nan() -> None:
    """A frame with no primary stream decoded (every subcarrier NaN) must
    not report a confident flat zero — that reads as "measured, no echoes"
    rather than "not measured"."""
    n = 16
    amp_db = np.full((3, n), np.nan)
    phase = np.full((3, n), np.nan)
    amp_db[1] = 0.0  # one live frame sandwiched between two dead ones
    phase[1] = 0.0
    cir = csi_to_cir(amp_db, phase)
    assert np.isnan(cir[0]).all()
    assert np.isnan(cir[2]).all()
    assert np.isfinite(cir[1]).all()


def test_centred_moves_the_peak_to_the_middle_and_reunites_a_split() -> None:
    """fftshift both relocates a dead-centre peak to the row's middle and
    reunites one that a fractional delay splits across the wrap: a half-tap
    delay puts half its energy at tap 0 and half at tap N-1 in the raw
    (uncentred) layout, and those two must land adjacent once centred."""
    n = 32
    h = np.ones((1, n), dtype=complex)
    amp_db, phase = _to_amp_phase(h)
    centred = csi_to_cir_centred(amp_db, phase)
    assert np.argmax(csi_to_cir(amp_db, phase)[0]) == 0
    assert np.argmax(centred[0]) == n // 2
    np.testing.assert_allclose(np.sort(csi_to_cir(amp_db, phase)[0]), np.sort(centred[0]))

    k = np.fft.fftfreq(n, 1 / n)
    h_split = np.exp(-1j * 2 * np.pi * k * 0.5 / n)[None, :]
    amp_db2, phase2 = _to_amp_phase(h_split)
    split_centred = csi_to_cir_centred(amp_db2, phase2)
    assert abs(int(np.argmax(split_centred[0])) - n // 2) <= 1
