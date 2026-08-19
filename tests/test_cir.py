"""backend.cir: IFFT of the CSI ratio into a delay-domain impulse response."""

from __future__ import annotations

import numpy as np

from backend.cir import ratio_to_cir, ratio_to_cir_centred


def _to_ratio(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a complex frequency response into the (amp_db, phase) pair
    ``ratio_to_cir`` expects, i.e. the inverse of what it reconstructs."""
    amp_db = 20 * np.log10(np.abs(h))
    phase = np.angle(h)
    return amp_db, phase


def test_a_flat_spectrum_is_a_single_tap_at_zero_delay() -> None:
    """H[k] = 1 for all k is a perfect impulse: all energy at delay zero.

    ratio_to_cir undoes the centring with ifftshift before the IFFT, so this
    also pins down that step is correct in direction — get it backwards and
    the peak would land at the last tap instead of the first.
    """
    n = 64
    h = np.ones((1, n), dtype=complex)
    amp_db, phase = _to_ratio(h)
    cir = ratio_to_cir(amp_db, phase)
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
    amp_db, phase = _to_ratio(h)
    cir = ratio_to_cir(amp_db, phase)
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
    amp_db, phase = _to_ratio(h)
    amp_db[0, 10] = np.nan
    phase[0, 10] = np.nan
    cir = ratio_to_cir(amp_db, phase)
    assert np.isfinite(cir).all()
    assert np.argmax(cir[0]) == 0


def test_a_frame_with_no_ratio_at_all_comes_back_nan() -> None:
    """A single-stream frame (every subcarrier NaN) must not report a
    confident flat zero — that reads as "measured, no echoes" rather than
    "not measured"."""
    n = 16
    amp_db = np.full((3, n), np.nan)
    phase = np.full((3, n), np.nan)
    amp_db[1] = 0.0  # one live frame sandwiched between two dead ones
    phase[1] = 0.0
    cir = ratio_to_cir(amp_db, phase)
    assert np.isnan(cir[0]).all()
    assert np.isnan(cir[2]).all()
    assert np.isfinite(cir[1]).all()


def test_output_is_real_valued_and_non_negative_float32() -> None:
    rng = np.random.default_rng(0)
    n = 48
    amp_db = rng.uniform(-10, 10, (5, n))
    phase = rng.uniform(-np.pi, np.pi, (5, n))
    cir = ratio_to_cir(amp_db, phase)
    assert cir.dtype == np.float32
    assert (cir >= 0).all()


def test_shape_matches_input_regardless_of_subcarrier_count() -> None:
    for n in (64, 242, 256):
        amp_db = np.zeros((2, n))
        phase = np.zeros((2, n))
        cir = ratio_to_cir(amp_db, phase)
        assert cir.shape == (2, n)


def test_centred_moves_the_zero_delay_tap_to_the_middle() -> None:
    n = 32
    h = np.ones((1, n), dtype=complex)
    amp_db, phase = _to_ratio(h)
    raw = ratio_to_cir(amp_db, phase)
    centred = ratio_to_cir_centred(amp_db, phase)
    assert np.argmax(raw[0]) == 0
    assert np.argmax(centred[0]) == n // 2
    np.testing.assert_allclose(np.sort(raw[0]), np.sort(centred[0]))


def test_centred_reunites_a_peak_split_across_the_wrap() -> None:
    """A half-tap delay splits energy between tap 0 and tap N-1 in the raw
    layout; centring must place both halves adjacent to one another."""
    n = 32
    k = np.fft.fftfreq(n, 1 / n)
    h = np.exp(-1j * 2 * np.pi * k * 0.5 / n)[None, :]
    amp_db, phase = _to_ratio(h)
    centred = ratio_to_cir_centred(amp_db, phase)
    peak = np.argmax(centred[0])
    assert abs(peak - n // 2) <= 1


def test_centred_keeps_the_no_ratio_row_nan() -> None:
    n = 16
    amp_db = np.full((1, n), np.nan)
    phase = np.full((1, n), np.nan)
    assert np.isnan(ratio_to_cir_centred(amp_db, phase)).all()

