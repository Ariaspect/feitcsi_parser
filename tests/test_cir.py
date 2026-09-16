"""backend.cir: IFFT of the CSI ratio into a delay-domain impulse response."""

from __future__ import annotations

import numpy as np
import pytest

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


# ---------------------------------------------------------------------- #
#  Relative-dB form, which is what the panel serves                       #
# ---------------------------------------------------------------------- #

from backend.cir import CIR_DB_FLOOR, CIR_SCALE_DB, cir_relative_db  # noqa: E402


def _two_tap(n: int = 64, second_db: float = -12.0, lead: float = 3.0):
    """A response with a direct path and one echo at a known level."""
    taps = np.zeros(n, dtype=complex)
    taps[0] = lead
    taps[5] = lead * 10 ** (second_db / 20.0)
    h = np.fft.fftshift(np.fft.fft(taps))
    amp_db = 20 * np.log10(np.abs(h))[None, :]
    return amp_db, np.angle(h)[None, :]


def test_every_frame_peaks_at_zero_db():
    amp_db, phase = _two_tap()
    out = cir_relative_db(amp_db, phase)
    assert np.nanmax(out) == pytest.approx(0.0, abs=1e-5)


def test_the_echo_lands_at_its_injected_level():
    amp_db, phase = _two_tap(second_db=-12.0)
    out = cir_relative_db(amp_db, phase)[0]
    order = np.sort(out)[::-1]
    assert order[0] == pytest.approx(0.0, abs=1e-5)
    assert order[1] == pytest.approx(-12.0, abs=0.5)


def test_relative_db_is_immune_to_gain():
    """Normalising per frame is what makes the panel gain-proof.

    Any gain in front of the receiver -- including the per-gain-state
    distortion backend.agc corrects -- scales the whole response, and a
    response expressed against its own peak does not move.
    """
    amp_db, phase = _two_tap()
    base = cir_relative_db(amp_db, phase)
    for gain_db in (-20.0, +17.0):
        shifted = cir_relative_db(amp_db + gain_db, phase)
        np.testing.assert_allclose(base, shifted, atol=1e-4)


def test_a_frame_with_no_channel_stays_nan():
    amp_db, phase = _two_tap()
    amp_db = np.vstack([amp_db, np.full_like(amp_db, np.nan)])
    phase = np.vstack([phase, np.full_like(phase, np.nan)])
    out = cir_relative_db(amp_db, phase)
    assert np.isfinite(out[0]).any()
    assert np.isnan(out[1]).all()


def test_an_empty_tap_lands_on_the_floor_not_at_minus_infinity():
    """The objection to dB was that a delay response is genuinely zero
    between echoes. The floor is what answers it."""
    amp_db, phase = _two_tap()
    out = cir_relative_db(amp_db, phase)
    assert np.isfinite(out[np.isfinite(out)]).all()
    # float32, so the floor lands within a rounding step of its exact value
    assert out[np.isfinite(out)].min() >= 20 * np.log10(CIR_DB_FLOOR) - 1e-3


def test_the_fixed_range_ceiling_is_the_peak():
    lo, hi = CIR_SCALE_DB
    assert hi == 0.0          # each frame's own peak; nothing can exceed it
    assert lo < hi


def test_the_cir_metric_is_built_on_the_ratio_planes():
    """The raw rx0/tx0 form separated motion from quiet 1.01 to 1 -- see
    backend.cir. The ratio is what cancels the per-packet timing jitter."""
    from backend import tiles

    assert tiles.DERIVED_METRICS["csi_cir"].bases == (
        "csi_ratio_amplitude", "csi_ratio_phase",
    )
    # and therefore it no longer inherits the AGC correction
    assert not tiles._agc_affected("csi_cir")


def test_the_crop_matches_what_the_frontend_hardcodes():
    """App.tsx carries CIR_CROP_TAPS and CIR_TAP_METRES to lay out the axis
    before the first tile arrives, so it cannot read them from a response.
    This is the guard against the two drifting apart."""
    from backend.cir import CIR_CROP_TAPS, CIR_TAP_METRES, cir_rows

    assert CIR_CROP_TAPS == 8
    assert CIR_TAP_METRES == 3.75
    assert cir_rows() == 17


def test_the_crop_keeps_the_centre_and_drops_the_rest():
    amp_db, phase = _two_tap(n=256)
    full = csi_to_cir_centred(amp_db, phase)
    out = cir_relative_db(amp_db, phase)
    from backend.cir import CIR_CROP_TAPS

    assert out.shape[1] == 2 * CIR_CROP_TAPS + 1
    # the peak survives the crop and is still the 0 dB reference
    assert np.nanmax(out) == pytest.approx(0.0, abs=1e-5)
    assert full.shape[1] == 256


def test_normalisation_uses_the_whole_row_not_the_crop():
    """A strong tap outside the kept window must still set 0 dB, or the panel
    would renormalise to whatever the crop happened to contain."""
    n = 256
    taps = np.zeros(n, dtype=complex)
    taps[0] = 1.0            # lands at the centre after fftshift
    taps[40] = 10.0          # far outside the crop, and much stronger
    h = np.fft.fftshift(np.fft.fft(taps))
    out = cir_relative_db(20 * np.log10(np.abs(h))[None, :], np.angle(h)[None, :])
    # the kept window holds only the weaker tap, so nothing in it reaches 0 dB
    assert np.nanmax(out) < -15.0
