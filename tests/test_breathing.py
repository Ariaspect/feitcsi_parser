"""Tests for backend.breathing -- the static-occupant path."""

from __future__ import annotations

import numpy as np
import pytest

from backend.breathing import (
    align_signs,
    band_weights,
    breathing_windows,
    estimate_rate,
    group_combine,
    project_to_1d,
)

FS = 18.116
BREATH_RPM = 15.0
BREATH_HZ = BREATH_RPM / 60.0


def _chest(
    n: float = 300.0,
    rpm: float = BREATH_RPM,
    n_sc: int = 24,
    depth: float = 0.05,
    noise: float = 0.02,
    fs: float = FS,
    seed: int = 0,
) -> np.ndarray:
    """A detrended ratio carrying one chest.

    The perturbation is collinear per subcarrier -- one physical motion seen
    through a different multipath geometry per subcarrier -- which is why the
    direction differs across k and the projection has to be found per k.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(n * fs)) / fs
    direction = np.exp(1j * rng.uniform(0, 2 * np.pi, n_sc))
    wave = depth * np.sin(2 * np.pi * rpm / 60.0 * t)[:, None]
    noise_term = noise * (
        rng.standard_normal((t.size, n_sc)) + 1j * rng.standard_normal((t.size, n_sc))
    )
    return wave * direction + noise_term


# --------------------------------------------------------------------------- #
#  Projection                                                                  #
# --------------------------------------------------------------------------- #


def test_the_projection_finds_the_axis_the_chest_actually_moves_along() -> None:
    """Amplitude alone has a blind spot the projection does not.

    A perturbation along the imaginary axis of a real-valued static term
    barely changes the magnitude -- it rotates the sum. Reading |r| there
    measures the second-order term and calls the chest absent.
    """
    t = np.arange(int(120 * FS)) / FS
    wave = 0.05 * np.sin(2 * np.pi * BREATH_HZ * t)
    blind = (1.0 + 1j * wave)[:, None]

    projected = project_to_1d(blind)

    assert np.corrcoef(projected[:, 0], wave)[0, 1] == pytest.approx(1.0, abs=1e-3)
    magnitude = np.abs(blind[:, 0]) - np.abs(blind[:, 0]).mean()
    assert np.std(magnitude) < 0.02 * np.std(wave), "the magnitude really is blind here"


def test_every_subcarrier_gets_its_own_axis() -> None:
    sig = _chest()
    projected = project_to_1d(sig)

    assert projected.shape == sig.shape
    assert np.isrealobj(projected)
    for k in range(sig.shape[1]):
        assert np.std(projected[:, k]) > 0


# --------------------------------------------------------------------------- #
#  Weights and sign alignment                                                  #
# --------------------------------------------------------------------------- #


def test_weight_follows_periodicity_not_amplitude() -> None:
    """A loud broadband subcarrier must not outrank a quiet tonal one."""
    t = np.arange(int(200 * FS)) / FS
    rng = np.random.default_rng(4)
    tonal = 0.01 * np.sin(2 * np.pi * BREATH_HZ * t)
    loud_noise = 1.0 * rng.standard_normal(t.size)

    w = band_weights(np.column_stack([tonal, loud_noise]), FS)

    assert w[0] > w[1]


def test_opposite_signed_subcarriers_are_aligned_before_they_are_summed() -> None:
    """Unaligned, they cancel -- and the cancellation is silent."""
    t = np.arange(int(200 * FS)) / FS
    wave = np.sin(2 * np.pi * BREATH_HZ * t)
    s = np.column_stack([wave, -wave, wave])
    w = np.array([1.0, 1.0, 1.0])

    assert np.std(s.sum(axis=1)) == pytest.approx(np.std(wave), rel=0.01), (
        "unaligned, two of the three erase each other"
    )

    aligned = align_signs(s, w)
    assert np.std(aligned.sum(axis=1)) == pytest.approx(3 * np.std(wave), rel=0.01)


def test_alignment_leaves_the_reference_subcarrier_alone() -> None:
    t = np.arange(int(120 * FS)) / FS
    wave = np.sin(2 * np.pi * BREATH_HZ * t)
    s = np.column_stack([-wave, wave])
    w = np.array([9.0, 1.0])          # subcarrier 0 is the reference

    aligned = align_signs(s, w)
    assert np.allclose(aligned[:, 0], s[:, 0])
    assert np.allclose(aligned[:, 1], -s[:, 1])


# --------------------------------------------------------------------------- #
#  Grouping                                                                    #
# --------------------------------------------------------------------------- #


def test_grouping_keeps_frequency_diversity_instead_of_folding_it_away() -> None:
    """Groups exist so agreement between them can be measured at all."""
    sig = _chest()
    s = align_signs(project_to_1d(sig), band_weights(sig, FS))
    groups, combined = group_combine(s, band_weights(sig, FS), n_groups=6)

    assert groups.shape == (s.shape[0], 6)
    assert combined.shape == (s.shape[0],)
    for g in range(6):
        assert np.corrcoef(groups[:, g], combined)[0, 1] > 0.5


def test_grouping_refuses_more_groups_than_subcarriers() -> None:
    with pytest.raises(ValueError, match="groups"):
        group_combine(np.zeros((100, 3)), np.ones(3), n_groups=8)


# --------------------------------------------------------------------------- #
#  Rate estimation                                                             #
# --------------------------------------------------------------------------- #


def test_the_rate_beats_the_raw_bin_spacing() -> None:
    """A 45 s window resolves 1.33 rpm raw, which is too coarse to report.

    Zero-padding interpolates the transform and a parabolic fit on the peak
    recovers the true maximum between bins.
    """
    seconds = 45.0
    t = np.arange(int(seconds * FS)) / FS
    true_rpm = 14.4                      # deliberately between bins
    wave = np.sin(2 * np.pi * true_rpm / 60.0 * t)

    result = estimate_rate(wave, FS)

    raw_bin_rpm = 60.0 / seconds
    assert abs(result["rpm_fft"] - true_rpm) < 0.3 * raw_bin_rpm
    assert result["rpm"] == pytest.approx(true_rpm, abs=0.4)


def test_the_two_estimators_agree_on_a_clean_chest_and_say_so() -> None:
    t = np.arange(int(45 * FS)) / FS
    wave = np.sin(2 * np.pi * BREATH_HZ * t)

    result = estimate_rate(wave, FS)

    assert result["rpm_fft"] == pytest.approx(BREATH_RPM, abs=0.5)
    assert result["rpm_acf"] == pytest.approx(BREATH_RPM, abs=1.0)
    assert result["agreement"] > 0.9


def test_disagreement_between_the_estimators_is_reported_not_averaged() -> None:
    """Two methods that disagree mean the peak is not a chest.

    Averaging them produces a confident number for a window that had none.
    """
    rng = np.random.default_rng(9)
    noise = rng.standard_normal(int(45 * FS))

    result = estimate_rate(noise, FS)

    assert result["agreement"] < 0.9 or result["papr"] < 4.0


# --------------------------------------------------------------------------- #
#  End to end                                                                  #
# --------------------------------------------------------------------------- #


def test_a_chest_is_found_and_its_rate_recovered() -> None:
    out = breathing_windows(_chest(n=200.0), FS)

    detected = out["rpm"][np.isfinite(out["rpm"])]
    assert detected.size > 0.5 * out["rpm"].size
    assert np.median(detected) == pytest.approx(BREATH_RPM, abs=1.0)
    assert np.nanmedian(out["confidence"]) > 0.4


def test_an_empty_room_reports_no_rate_rather_than_a_number() -> None:
    """The failure that matters: a confident rate read off noise."""
    rng = np.random.default_rng(12)
    n = int(200 * FS)
    noise = 0.02 * (rng.standard_normal((n, 24)) + 1j * rng.standard_normal((n, 24)))

    out = breathing_windows(noise, FS)

    assert np.isnan(out["rpm"]).mean() > 0.8
    assert np.nanmedian(out["confidence"]) < 0.4


def test_gross_motion_holds_the_estimate_and_flags_it() -> None:
    """A breathing estimate taken during a walk-through is contaminated."""
    sig = _chest(n=200.0)
    out_free = breathing_windows(sig, FS)

    n_windows = out_free["rpm"].size
    motion = np.zeros(n_windows, dtype=bool)
    motion[n_windows // 2 : n_windows // 2 + 5] = True
    out = breathing_windows(sig, FS, motion_gate=motion)

    assert out["gated"][n_windows // 2 : n_windows // 2 + 5].all()
    held = out["rpm"][n_windows // 2 : n_windows // 2 + 5]
    assert np.allclose(held, held[0], equal_nan=True), "the estimate is held, not recomputed"
    # Confidence restarts low after the gate lifts and recovers.
    after = out["confidence"][n_windows // 2 + 5]
    assert after < np.nanmedian(out_free["confidence"])


def test_group_spread_is_reported_so_two_targets_are_visible() -> None:
    """Groups that disagree are the signature of more than one chest."""
    a = _chest(n=200.0, rpm=12.0, n_sc=24, seed=1)
    b = _chest(n=200.0, rpm=24.0, n_sc=24, seed=2)
    two = np.concatenate([a[:, :12], b[:, :12]], axis=1)

    one = breathing_windows(a, FS)
    both = breathing_windows(two, FS)

    assert np.nanmedian(both["group_spread_rpm"]) > np.nanmedian(one["group_spread_rpm"])


def test_weights_are_useless_on_an_already_filtered_signal() -> None:
    """Why the weight is taken from the raw window and not the filtered one.

    After a 0.1-0.6 Hz bandpass every subcarrier has all of its remaining
    power in band, so the in-band fraction is ~1 for all of them and the
    weighting stops discriminating. The failure is silent -- the sum still
    works, it just stops preferring the subcarriers that carry the chest.
    """
    from backend.presence import bandpass

    sig = _chest(n=200.0, n_sc=12)
    raw = band_weights(sig, FS)
    filtered = band_weights(bandpass(sig, FS, (0.1, 0.6)), FS)

    assert filtered.min() > 0.95, "everything is in band after the filter"
    assert raw.max() - raw.min() > 5 * (filtered.max() - filtered.min())


def test_the_reported_weights_discriminate_between_subcarriers() -> None:
    """End to end: w[k] must vary, or nothing is being weighted."""
    sig = _chest(n=200.0, n_sc=24)
    out = breathing_windows(sig, FS)

    w = out["weights"][:, out["weights"].any(axis=0)]
    assert w.size > 0
    assert np.ptp(w, axis=0).mean() > 0.05, "w[k] is flat, so the weighting is inert"
