"""Tests for backend.preprocess -- the common front half of both detectors."""

from __future__ import annotations

import numpy as np
import pytest

from backend.preprocess import (
    BREATHING_DETREND_SECONDS,
    CARRIER_HZ,
    MOTION_DETREND_SECONDS,
    NOMINAL_FS_HZ,
    WAVELENGTH_M,
    derive_sample_rate,
    normalize_subcarriers,
    remove_static,
    subcarrier_mask,
)

FS = 20.0


# --------------------------------------------------------------------------- #
#  Physical constants                                                          #
# --------------------------------------------------------------------------- #


def test_the_wavelength_follows_the_carrier_rather_than_a_literal() -> None:
    """5210 MHz is the operator's answer, not a constant of nature.

    Every millimetre-scale claim downstream divides by this, so it is derived
    from the carrier in one place and the carrier is the only thing to edit.
    """
    assert CARRIER_HZ == 5.21e9
    assert WAVELENGTH_M == pytest.approx(0.05754, abs=1e-5)


# --------------------------------------------------------------------------- #
#  Sample rate                                                                 #
# --------------------------------------------------------------------------- #


def test_the_sample_rate_is_derived_not_assumed() -> None:
    times = np.arange(0, 60, 1 / 18.5)
    rate = derive_sample_rate(times)

    assert rate["mean_hz"] == pytest.approx(18.5, rel=1e-3)
    assert rate["fs_hz"] == pytest.approx(18.5, rel=1e-3)


def test_a_derived_rate_far_from_nominal_is_a_warning_not_a_silent_swap() -> None:
    """The capture pings at 50 ms and arrives at 54.

    Measured on captures/lg/20260825_185637.bin the interval is bimodal, 50 ms
    for 23.4% of frames and 56 ms for 24.1%, mean 54.06 ms. Taking the nominal
    rate on faith puts every frequency axis 8% out; taking the derived rate
    silently hides that the link is dropping a fifth of its pings.
    """
    times = np.arange(0, 60, 1 / 18.5)
    rate = derive_sample_rate(times)

    assert rate["nominal_hz"] == NOMINAL_FS_HZ
    assert any("nominal" in w for w in rate["warnings"])

    on_nominal = derive_sample_rate(np.arange(0, 60, 1 / NOMINAL_FS_HZ))
    assert on_nominal["warnings"] == []


def test_the_mean_rate_is_preferred_to_the_median_for_the_grid() -> None:
    """A bimodal interval makes the two disagree, and the grid wants the mean.

    The median lands on whichever mode happens to be taller -- 56 ms on the
    capture above, an 18.86 Hz grid for a link that actually delivered 18.50
    Hz. Resampling onto it stretches the whole time axis.
    """
    times = np.cumsum(np.concatenate([np.full(500, 0.050), np.full(501, 0.056)]))
    rate = derive_sample_rate(times)

    assert rate["median_hz"] == pytest.approx(1 / 0.056, rel=1e-6)
    assert rate["fs_hz"] == pytest.approx(rate["mean_hz"], rel=1e-9)
    assert rate["fs_hz"] != pytest.approx(rate["median_hz"], rel=1e-3)


# --------------------------------------------------------------------------- #
#  Subcarrier masking                                                          #
# --------------------------------------------------------------------------- #


def test_structurally_dead_subcarriers_are_dropped_and_named() -> None:
    """Dropping them silently is how a mask becomes impossible to audit."""
    ratio = np.ones((100, 6), dtype=complex)
    ratio[:, 2] = np.nan
    h0_db = np.zeros((100, 6))

    mask = subcarrier_mask(ratio, h0_db, weak_fraction=0.0)

    assert not mask["keep"][2]
    assert 2 in mask["dropped_dead"]
    assert mask["dropped_weak"].size == 0


def test_the_weakest_subcarriers_by_h0_are_dropped() -> None:
    """A ratio divides by H0, so a faded H0 is where the ratio explodes.

    The bright horizontal bands in the amplitude heatmap sit at fixed
    subcarriers for the whole capture, which is the signature of a fading null
    in the denominator rather than of anything in the room.
    """
    ratio = np.ones((100, 10), dtype=complex)
    h0_db = np.tile(np.arange(10, dtype=float) * 3.0, (100, 1))

    mask = subcarrier_mask(ratio, h0_db, weak_fraction=0.2)

    # Bottom 20% by median |H0| is subcarriers 0 and 1.
    assert set(mask["dropped_weak"].tolist()) == {0, 1}
    assert not mask["keep"][:2].any()
    assert mask["keep"][2:].all()


def test_dropping_every_subcarrier_is_refused() -> None:
    ratio = np.full((10, 4), np.nan, dtype=complex)
    with pytest.raises(ValueError, match="no subcarrier"):
        subcarrier_mask(ratio, np.zeros((10, 4)))


# --------------------------------------------------------------------------- #
#  Static removal                                                              #
# --------------------------------------------------------------------------- #


def test_a_constant_path_is_removed_completely() -> None:
    """A body parked in the room is a constant, and this is the path that
    deliberately throws it away -- the other one keeps it."""
    ratio = np.full((400, 3), 2.0 + 1.0j)

    out = remove_static(ratio, FS, seconds=3.0)

    assert np.abs(out).max() < 1e-9


def test_the_breathing_window_passes_a_breath_the_motion_window_eats() -> None:
    """The whole reason the two paths use different windows.

    A box filter of length W has gain ``sinc(f*W)`` at frequency f, so what
    survives subtracting it is ``1 - sinc(f*W)``. At 12 rpm (0.2 Hz) the 15 s
    breathing window lands on ``sinc(3) = 0`` and passes the breath untouched,
    while the 3 s motion window passes half of it.

    Note what this does *not* say: the motion window is not a breathing
    filter. Half a breath still reaches the motion path, and what actually
    keeps respiration out of it is the first difference in
    ``backend.motion`` -- ``|2 sin(pi f Ts)|`` is -19.5 dB at 0.2 Hz and this
    capture's rate. Reading the window as the separator is how the two paths
    end up measuring each other.
    """
    breath_hz = 0.2
    t = np.arange(0, 300, 1 / FS)
    ratio = ((1 + 0.01 * t) + np.exp(2j * np.pi * breath_hz * t))[:, None]

    core = slice(int(60 * FS), int(240 * FS))
    n_core = int(240 * FS) - int(60 * FS)
    bin_at = int(round(breath_hz * n_core / FS))

    def amp(x: np.ndarray) -> float:
        # fft, not rfft: the ratio is complex and the breath sits at +f only.
        return float(np.abs(np.fft.fft(x[core, 0])[bin_at]))

    reference = amp(ratio)

    short = remove_static(ratio, FS, seconds=MOTION_DETREND_SECONDS)
    long = remove_static(ratio, FS, seconds=BREATHING_DETREND_SECONDS)

    def sinc_residual(seconds: float) -> float:
        return abs(1.0 - np.sinc(breath_hz * seconds))

    assert amp(long) / reference == pytest.approx(sinc_residual(15.0), abs=0.05)
    assert amp(short) / reference == pytest.approx(sinc_residual(3.0), abs=0.05)
    assert amp(long) > 1.8 * amp(short)


def test_both_windows_remove_a_linear_drift() -> None:
    t = np.arange(0, 200, 1 / FS)
    ratio = (1 + 0.01 * t)[:, None].astype(complex)

    core = slice(int(30 * FS), int(170 * FS))
    for seconds in (MOTION_DETREND_SECONDS, BREATHING_DETREND_SECONDS):
        out = remove_static(ratio, FS, seconds=seconds)
        # A linear ramp through a centred box mean leaves only the edge
        # asymmetry, which the core slice excludes.
        assert np.abs(out[core]).max() < 1e-9, seconds


def test_static_removal_keeps_the_series_length_and_the_edges_usable() -> None:
    """Edge normalisation, not zero padding: a padded edge is a fake step."""
    ratio = np.ones((200, 2), dtype=complex)

    out = remove_static(ratio, FS, seconds=3.0)

    assert out.shape == ratio.shape
    assert np.isfinite(out).all()
    assert np.abs(out[0]).max() < 1e-9
    assert np.abs(out[-1]).max() < 1e-9


def test_a_window_longer_than_the_series_is_clamped() -> None:
    ratio = np.ones((20, 2), dtype=complex)
    out = remove_static(ratio, FS, seconds=600.0)
    assert np.abs(out).max() < 1e-9


# --------------------------------------------------------------------------- #
#  Per-subcarrier noise normalisation                                          #
# --------------------------------------------------------------------------- #


def test_subcarriers_with_different_noise_scales_come_out_equal() -> None:
    """The point of normalising: a hot subcarrier stops dominating a sum.

    Measured on captures/lg/20260825_185637.bin the shoulders of the two H0
    fading nulls run 9.5 dB hotter than the rest after static removal, purely
    as a scale -- their spectral shape matches the quiet subcarriers almost
    exactly. Un-normalised, they carry 9x the weight in anything that averages
    across the subcarrier axis.
    """
    rng = np.random.default_rng(0)
    n = 4000
    noise = rng.standard_normal((n, 3)) + 1j * rng.standard_normal((n, 3))
    sig = noise * np.array([1.0, 3.0, 9.0])

    out, scale = normalize_subcarriers(sig)

    assert scale[1] / scale[0] == pytest.approx(3.0, rel=0.1)
    assert scale[2] / scale[0] == pytest.approx(9.0, rel=0.1)
    spread = np.std(np.abs(out), axis=0)
    assert spread.max() / spread.min() == pytest.approx(1.0, abs=0.1)


def test_the_scale_is_robust_to_a_burst_of_real_motion() -> None:
    """A walk-through must not rescale the subcarrier that saw it.

    A mean-square scale would take a subcarrier that was quiet for 590 s and
    loud for 10 and call the whole thing loud, then divide the walk-through
    away. The median does not move.
    """
    rng = np.random.default_rng(1)
    n = 4000
    quiet = rng.standard_normal((n, 1)) + 1j * rng.standard_normal((n, 1))
    burst = quiet.copy()
    burst[n // 2 : n // 2 + n // 20] *= 50.0

    _, quiet_scale = normalize_subcarriers(quiet)
    _, burst_scale = normalize_subcarriers(burst)

    assert burst_scale[0] / quiet_scale[0] == pytest.approx(1.0, abs=0.1)


def test_normalising_rescales_without_reshaping() -> None:
    """One scalar per subcarrier, so no spectrum changes shape."""
    rng = np.random.default_rng(2)
    t = np.arange(2000) / FS
    sig = (rng.standard_normal((2000, 2)) + 1j * rng.standard_normal((2000, 2))) * 5.0
    sig[:, 0] += 3.0 * np.exp(2j * np.pi * 0.25 * t)

    out, _ = normalize_subcarriers(sig)

    before = np.abs(np.fft.fft(sig[:, 0]))
    after = np.abs(np.fft.fft(out[:, 0]))
    assert np.corrcoef(before, after)[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_a_dead_subcarrier_cannot_divide_the_others_away() -> None:
    """A zero scale is an infinity waiting to poison every later average."""
    sig = np.zeros((100, 2), dtype=complex)
    sig[:, 1] = 1.0 + 0j

    out, scale = normalize_subcarriers(sig)

    assert np.isfinite(out).all()
    assert (scale > 0).all()
