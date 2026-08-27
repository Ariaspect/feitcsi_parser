"""Tests for backend.motion -- the gross-motion path."""

from __future__ import annotations

import numpy as np
import pytest

from backend.motion import (
    eigenvalue_ratio,
    fidget_energy,
    lag1_correlation,
    motion_score,
)

FS = 18.0


def _ar1(rho: float, n: int, n_sc: int = 4, seed: int = 0) -> np.ndarray:
    """Complex AR(1) with a known lag-1 correlation."""
    rng = np.random.default_rng(seed)
    innov = rng.standard_normal((n, n_sc)) + 1j * rng.standard_normal((n, n_sc))
    x = np.zeros((n, n_sc), dtype=complex)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho**2) * innov[i]
    return x


# --------------------------------------------------------------------------- #
#  The normalised difference                                                   #
# --------------------------------------------------------------------------- #


def test_rho_recovers_the_lag_one_correlation_it_is_defined_as() -> None:
    """``E[(x_t - x_{t-1})^2] = 2*sigma^2*(1 - rho_1)`` is the whole identity.

    Dividing the difference energy by the variance is not a convenience, it is
    what turns a quantity in the link's arbitrary units into one that means
    the same thing on every radio.
    """
    for rho in (0.0, 0.5, 0.9):
        _, measured = lag1_correlation(_ar1(rho, 20000), FS, window_seconds=60.0)
        assert np.median(measured) == pytest.approx(rho, abs=0.05), rho


def test_white_noise_scores_no_motion_at_any_amplitude() -> None:
    """An empty room's residual is noise, and noise must read zero.

    Scaling it up must not change the answer -- that is the difference
    between this and the raw difference energy it replaces.
    """
    rng = np.random.default_rng(3)
    quiet = rng.standard_normal((6000, 8)) + 1j * rng.standard_normal((6000, 8))

    _, faint = lag1_correlation(quiet * 0.01, FS)
    _, loud = lag1_correlation(quiet * 100.0, FS)

    assert abs(np.median(motion_score(faint))) < 0.1
    assert np.median(motion_score(faint)) == pytest.approx(
        np.median(motion_score(loud)), abs=0.02
    )


def test_the_difference_is_complex_so_phase_only_motion_is_seen() -> None:
    """A body at fixed range moves the ratio around a circle.

    ``|r_t| - |r_{t-1}|`` is identically zero on that trajectory: the
    magnitude never changes. Only the complex difference sees it, and this is
    the case that motivates the whole "complex throughout" rule.
    """
    t = np.arange(4000) / FS
    circle = np.exp(2j * np.pi * 0.7 * t)[:, None] * np.ones((1, 4))

    _, rho = lag1_correlation(circle, FS)

    assert np.median(motion_score(rho)) > 0.9
    assert np.std(np.abs(circle)) < 1e-12, "the magnitude really is constant"


def test_windows_are_reported_on_their_centres_and_do_not_run_off_the_end() -> None:
    centres, rho = lag1_correlation(_ar1(0.5, 1800), FS, window_seconds=2.0, hop_seconds=0.25)

    assert rho.shape[1] == centres.size
    assert rho.shape[0] == 4
    assert centres[0] == pytest.approx(1.0, abs=0.1)
    assert centres[-1] <= 1800 / FS


def test_a_window_shorter_than_two_samples_is_refused() -> None:
    with pytest.raises(ValueError, match="window"):
        lag1_correlation(_ar1(0.5, 100), FS, window_seconds=0.01)


# --------------------------------------------------------------------------- #
#  Combination and side indicators                                             #
# --------------------------------------------------------------------------- #


def test_the_score_is_a_median_so_a_few_wild_subcarriers_cannot_carry_it() -> None:
    rho = np.zeros((20, 5))
    rho[:3] = 0.95                       # three subcarriers screaming

    assert motion_score(rho).max() == pytest.approx(0.0, abs=1e-9)


def test_coherent_motion_concentrates_the_covariance_into_one_eigenvalue() -> None:
    """The feature that separates a body from a chest.

    Gross motion moves every subcarrier through one shared waveform, so the
    subcarrier covariance is nearly rank one. Breathing moves them along
    directions that differ per subcarrier, and noise is isotropic; both spread
    the eigenvalues out.
    """
    n, n_sc = 3600, 12
    rng = np.random.default_rng(5)
    noise = 0.05 * (rng.standard_normal((n, n_sc)) + 1j * rng.standard_normal((n, n_sc)))

    t = np.arange(n) / FS
    shared = np.sin(2 * np.pi * 0.8 * t)[:, None] * np.exp(
        1j * rng.uniform(0, 2 * np.pi, n_sc)
    )

    _, coherent = eigenvalue_ratio(shared + noise, FS)
    _, isotropic = eigenvalue_ratio(noise, FS)

    assert np.median(coherent) > 0.8
    assert np.median(isotropic) < 0.5
    assert np.median(coherent) > np.median(isotropic)


def test_the_fidget_band_is_clamped_to_nyquist() -> None:
    """The spec asks for 1-10 Hz and this capture's Nyquist is 9.06 Hz.

    Asking for a band that runs past Nyquist silently measures nothing above
    it, which reads as a quieter room rather than as a narrower band.
    """
    _, energy, band = fidget_energy(_ar1(0.0, 3600), FS, band_hz=(1.0, 10.0))

    assert band[1] == pytest.approx(FS / 2, rel=1e-6)
    assert np.isfinite(energy).all()


def test_fidgeting_shows_up_in_band_and_a_still_room_does_not() -> None:
    n = 3600
    t = np.arange(n) / FS
    rng = np.random.default_rng(7)
    quiet = rng.standard_normal((n, 6)) + 1j * rng.standard_normal((n, 6))
    twitch = quiet + 3.0 * np.sin(2 * np.pi * 3.0 * t)[:, None]

    _, still, _ = fidget_energy(quiet, FS)
    _, moving, _ = fidget_energy(twitch, FS)

    assert np.median(moving) > 3 * np.median(still)


def test_the_fidget_reading_is_absolute_so_a_louder_room_reads_louder() -> None:
    """The fraction form cannot do this, which is why it is not used.

    From 1 Hz to a 9 Hz Nyquist is 89% of the spectrum, so an in-band
    *fraction* reads 0.92 for white noise and has nowhere left to go. The
    absolute reading scales with the event, in units of the noise floor the
    preprocessing already divided out.
    """
    rng = np.random.default_rng(11)
    quiet = rng.standard_normal((3600, 6)) + 1j * rng.standard_normal((3600, 6))

    _, faint, _ = fidget_energy(quiet, FS)
    _, loud, _ = fidget_energy(quiet * 4.0, FS)

    assert np.median(loud) / np.median(faint) == pytest.approx(16.0, rel=0.05)


# --------------------------------------------------------------------------- #
#  Dropouts                                                                    #
# --------------------------------------------------------------------------- #


def test_a_bridged_dropout_is_blanked_rather_than_read_as_motion() -> None:
    """The lie this path is most prone to.

    Interpolating across a hole leaves a perfectly smooth stretch, and smooth
    is what a lag-1 correlation calls motion -- a bridged hole reads 1.0, the
    strongest possible score. Measured on
    captures/lg/20260825_185637.bin, fabricated windows scored 0.69-0.79
    against 0.57 for clean ones.
    """
    n = 3600
    sig = _ar1(0.0, n)
    hole = slice(1000, 1200)
    sig[hole] = np.linspace(sig[hole.start - 1], sig[hole.stop], hole.stop - hole.start)
    fabricated = np.zeros(n, dtype=bool)
    fabricated[hole] = True

    centres, rho = lag1_correlation(sig, FS, fabricated=fabricated)
    score = motion_score(rho)

    inside = (centres > 1000 / FS + 0.75) & (centres < 1200 / FS - 0.75)
    assert inside.any()
    assert np.isnan(score[inside]).all(), "a bridged hole must have no verdict"
    assert np.isfinite(score[centres < 1000 / FS - 1]).all()


def test_every_motion_metric_blanks_the_same_windows() -> None:
    """Three series drawn on one axis must agree about where the holes are."""
    n = 3600
    sig = _ar1(0.3, n)
    fabricated = np.zeros(n, dtype=bool)
    fabricated[1500:2000] = True

    _, rho = lag1_correlation(sig, FS, fabricated=fabricated)
    _, eig = eigenvalue_ratio(sig, FS, fabricated=fabricated)
    _, fid, _ = fidget_energy(sig, FS, fabricated=fabricated)

    blanked = np.isnan(motion_score(rho))
    assert blanked.any()
    assert np.array_equal(blanked, np.isnan(eig))
    assert np.array_equal(blanked, np.isnan(fid))
