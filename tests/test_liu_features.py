"""Tests for backend.liu_features -- display-only columns on the classifier tab."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend import classifier, liu_features as L
from tests.test_hybrid2 import _capture

FS = 42.0


def _times(n: int) -> np.ndarray:
    return np.arange(n) / FS


def test_a_short_repetition_reads_as_peaks_at_its_period_and_noise_does_not() -> None:
    """A 4 Hz oscillation shared across subcarriers -- a machine repeating
    itself -- puts a high ACF peak at 0.25 s in every block; white noise has
    only small ripples and no stable period."""
    rng = np.random.default_rng(0)
    n, F = int(60 * FS), 24
    t = _times(n)
    osc = 1.0 + 0.2 * np.exp(1j * 0.8 * np.sin(2 * np.pi * 4.0 * t))[:, None] * np.exp(1j * rng.uniform(0, 6.3, F))[None, :]
    r = osc + 0.01 * (rng.standard_normal((n, F)) + 1j * rng.standard_normal((n, F)))
    f = L.periodicity(r, t)
    assert f["liu_acf_peak"] > 0.5
    assert f["liu_acf_period"] == pytest.approx(0.25, abs=0.03)
    assert f["liu_acf_period_spread"] < 0.2
    assert f["liu_ms_mrc"] > 0.5

    noise = 1.0 + 0.01 * (rng.standard_normal((n, F)) + 1j * rng.standard_normal((n, F)))
    g = L.periodicity(noise, t)
    assert abs(g["liu_acf_peak"]) < 0.15
    assert abs(g["liu_ms_mrc"]) < 0.15


def _scattered(v: float, n: int, F: int, seed: int) -> np.ndarray:
    """Isotropic scattering from a mover at speed v: per subcarrier a sum of
    many paths with uniform arrival angles, so the field's ACF is J0(k v tau)."""
    rng = np.random.default_rng(seed)
    t = _times(n)
    k = 2 * np.pi / L.WAVELENGTH_M
    M = 300
    th = rng.uniform(0, 2 * np.pi, M)
    out = np.empty((n, F), complex)
    for f in range(F):
        ph = rng.uniform(0, 2 * np.pi, M)
        out[:, f] = np.exp(1j * (k * v * np.cos(th)[None, :] * t[:, None] + ph[None, :])).sum(axis=1) / np.sqrt(M)
    return 1.0 + 0.3 * out


def test_a_slow_mover_reads_its_speed_and_a_fast_one_reads_too_fast() -> None:
    n, F = int(60 * FS), 16
    slow = L.speed(_scattered(0.3, n, F, seed=1), _times(n))
    assert slow["liu_speed"] == pytest.approx(0.3, rel=0.2)
    assert slow["liu_speed_fast"] < 0.2
    walk = L.speed(_scattered(1.34, n, F, seed=2), _times(n))   # a walking person
    assert walk["liu_speed_fast"] > 0.5


def test_coherence_is_wide_for_a_shared_change_and_flat_for_independent_noise() -> None:
    rng = np.random.default_rng(3)
    n, F = int(60 * FS), 64
    t = _times(n)
    shared = 1.0 + 0.05 * np.sin(2 * np.pi * 0.3 * t)[:, None] * np.ones(F)[None, :]
    noise = 0.002 * rng.standard_normal((n, F))
    c = L.coherence(shared + noise, t)
    assert c[1] > 0.9 and c[32] > 0.9
    c0 = L.coherence(1.0 + noise, t)
    assert abs(c0[1]) < 0.1 and abs(c0[32]) < 0.1
    # Breathing-band only: a 0.3 Hz shared swing under louder independent 5 Hz noise
    loud = 0.05 * np.sin(2 * np.pi * 5.0 * t[:, None] + rng.uniform(0, 6.3, F)[None, :])
    allband = L.coherence(shared + loud, t)
    breath = L.coherence(shared + loud, t, L.BREATH_BAND_HZ)
    assert breath[32] > 0.9 > allband[32]


def test_weighted_quantile_and_spread() -> None:
    v = np.array([1.0, 2.0, 3.0, 100.0])
    assert L.weighted_quantile(v, np.array([1, 1, 1, 0]), 0.5) == pytest.approx(2.0)
    assert np.isnan(L.weighted_quantile(v, np.zeros(4), 0.5))
    assert L.spread_of(np.full(10, 5.0)) == 0.0
    assert np.isnan(L.std_of(np.array([np.nan, 1.0])))


def test_short_or_empty_input_is_blank_not_an_error() -> None:
    t = _times(5)
    r = np.ones((5, 4), complex)
    assert all(np.isnan(v) for v in L.periodicity(r, t).values())
    assert all(np.isnan(v) for v in L.speed(r, t).values())
    assert all(np.isnan(v) for v in L.coherence(r, t).values())


def test_the_columns_are_appended_display_only_and_feed_nothing() -> None:
    keys = [f["key"] for f in classifier.FEATURES]
    liu = [f["key"] for f in L.FEATURES]
    i = keys.index(liu[0])
    assert keys[i:i + len(liu)] == liu                  # appended after the bank, in order
    for f in L.FEATURES:
        assert f["status"] == "display only" and f["reference"] is None
        assert f["role"] and f["separates"] and f["not_separates"]
    assert not set(liu) & set(classifier.LR_FEATURES)
    assert not set(liu) & set(classifier.HUMAN_FEATURES)


def test_every_unit_carries_the_columns_and_the_previous_units_score(tmp_path: Path) -> None:
    units = classifier.compute_features(_capture(tmp_path), 0.0, 200.0)["units"]
    assert len(units) == 3
    for u in units:
        for f in L.FEATURES:
            assert f["key"] in u
    assert np.isnan(units[0]["liu_prev_p"])
    for a, b in zip(units[:-1], units[1:]):
        assert b["liu_prev_p"] == a["p_occupied"] or (np.isnan(b["liu_prev_p"]) and np.isnan(a["p_occupied"]))
