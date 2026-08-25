"""Tests for backend.doppler pure functions."""

from __future__ import annotations

import numpy as np
import pytest

from backend.doppler import gap_limit_for, resample_uniform, stft_average, uniform_grid


def _tone(freq: float, fs: float, n: int, n_cols: int = 1) -> np.ndarray:
    t = np.arange(n) / fs
    return np.tile(np.sin(2 * np.pi * freq * t)[:, None], (1, n_cols))


def test_uniform_grid_uses_the_median_interval() -> None:
    """fs comes from the median gap, so one outlier cannot drag the rate."""
    times = np.array([0.0, 0.1, 0.2, 0.3, 5.0, 5.1, 5.2])
    grid, fs = uniform_grid(times)
    assert fs == pytest.approx(10.0, rel=1e-6)
    assert np.allclose(np.diff(grid), 0.1)
    assert grid[-1] <= times[-1] + 1e-9


def test_resample_blanks_samples_inside_a_large_gap() -> None:
    """A 5 s hole in a 10 Hz capture must not be interpolated into signal."""
    times = np.array([0.0, 0.1, 0.2, 5.2, 5.3, 5.4])
    values = np.arange(6, dtype=float).reshape(6, 1)
    grid, _ = uniform_grid(times)
    # Explicit limit: with only five intervals the 95th percentile is itself
    # dragged up by the outlier. gap_limit_for is exercised separately below,
    # on a distribution shaped like a real capture's.
    out = resample_uniform(times, values, grid, gap_limit=0.25)

    inside = (grid > 0.2 + 1e-9) & (grid < 5.2 - 1e-9)
    assert inside.any(), "fixture must place samples strictly inside the gap"
    assert np.all(np.isnan(out[inside, 0]))
    assert np.isfinite(out[grid <= 0.2, 0]).all()


def test_stft_finds_a_known_tone_with_row_zero_highest() -> None:
    """A 2 Hz tone lands in the 2 Hz bin, counting rows from the top."""
    fs, win, hop = 20.0, 128, 64
    spec, freqs = stft_average(_tone(2.0, fs, 2048, n_cols=4), fs, win, hop)

    assert spec.shape[0] == win // 2 + 1
    assert freqs[0] == pytest.approx(0.0)
    assert freqs[-1] == pytest.approx(fs / 2)
    peak_row = int(np.argmax(spec.mean(axis=1)))
    assert freqs[::-1][peak_row] == pytest.approx(2.0, abs=fs / win)


def test_stft_removes_the_per_window_mean() -> None:
    """A large DC offset must not dominate; detrending is not optional."""
    fs, win, hop = 20.0, 128, 64
    spec, freqs = stft_average(_tone(2.0, fs, 2048) + 1000.0, fs, win, hop)
    profile = spec.mean(axis=1)
    dc_row = int(np.argmin(np.abs(freqs[::-1] - 0.0)))
    assert profile[dc_row] < profile.max() * 0.1


def test_stft_leaves_a_gapped_window_nan() -> None:
    """A window containing a resampling hole is NaN, not zero-filled."""
    fs, win, hop = 20.0, 128, 64
    samples = _tone(2.0, fs, 512)
    samples[200:260, :] = np.nan
    spec, _ = stft_average(samples, fs, win, hop)
    assert np.isnan(spec).any()
    assert np.isfinite(spec).any(), "only the affected columns should be NaN"


def test_stft_rejects_a_window_longer_than_the_series() -> None:
    with pytest.raises(ValueError, match="shorter than"):
        stft_average(_tone(2.0, 20.0, 64), 20.0, 128, 64)


def test_gap_limit_tracks_jitter_not_dropouts() -> None:
    """On a realistic interval distribution, the limit sits above jitter
    and below a dropout."""
    rng = np.random.default_rng(0)
    dt = np.abs(rng.normal(0.056, 0.004, 2000))   # ~18 Hz, MTK-like jitter
    dt[900] = 23.0                                 # one real dropout
    times = np.concatenate([[0.0], np.cumsum(dt)])

    limit = gap_limit_for(times)
    assert limit > np.percentile(dt, 90), "must not flag ordinary jitter"
    assert limit < 23.0, "must flag a real dropout"
