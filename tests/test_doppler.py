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


def test_resample_reports_fabricated_samples_without_blanking_them() -> None:
    """A 5 s hole is interpolated, and flagged so the caller can judge it.

    Writing NaN here instead would force a whole window to die for one missed
    packet; the caller decides proportionally.
    """
    times = np.array([0.0, 0.1, 0.2, 5.2, 5.3, 5.4])
    values = np.arange(6, dtype=float).reshape(6, 1)
    grid, _ = uniform_grid(times)
    # Explicit limit: with only five intervals the 95th percentile is itself
    # dragged up by the outlier. gap_limit_for is exercised separately below,
    # on a distribution shaped like a real capture's.
    out, fabricated = resample_uniform(times, values, grid, gap_limit=0.25)

    inside = (grid > 0.2 + 1e-9) & (grid < 5.2 - 1e-9)
    assert inside.any(), "fixture must place samples strictly inside the gap"
    assert np.array_equal(fabricated, inside)
    assert np.isfinite(out).all(), "values are interpolated, not blanked"


def test_stft_blanks_only_windows_that_are_mostly_fabricated() -> None:
    """The proportional rule: a minority of invented samples is tolerated.

    Blanking on the mere presence of a gap costs 32.5% of columns on a real
    capture whose actual dead time is 6.3%.
    """
    fs, win, hop = 20.0, 128, 64
    samples = _tone(2.0, fs, 1024, n_cols=2)

    light = np.zeros(1024, dtype=bool)
    light[:8] = True                    # 6% of the first window
    spec, _ = stft_average(samples, fs, win, hop, fabricated=light,
                           max_gap_fraction=0.5)
    assert np.isfinite(spec).all(), "a 6% gap must not blank a column"

    heavy = np.zeros(1024, dtype=bool)
    heavy[:100] = True                  # 78% of the first window
    spec, _ = stft_average(samples, fs, win, hop, fabricated=heavy,
                           max_gap_fraction=0.5)
    assert np.all(np.isnan(spec[:, 0])), "a mostly-invented column must blank"
    assert np.isfinite(spec[:, -1]).all(), "later columns are unaffected"


def test_stft_zero_pad_refines_the_frequency_grid() -> None:
    """Zero-padding adds rows and keeps the axis ending at fs/2."""
    fs, win, hop = 20.0, 128, 64
    plain, f_plain = stft_average(_tone(2.0, fs, 1024), fs, win, hop)
    padded, f_pad = stft_average(_tone(2.0, fs, 1024), fs, win, hop, zero_pad=256)

    assert padded.shape[0] > plain.shape[0]
    assert padded.shape[1] == plain.shape[1], "columns must not change"
    assert f_pad[-1] == pytest.approx(fs / 2)
    assert f_pad[-1] == pytest.approx(f_plain[-1])


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


def test_stft_ignores_structurally_dead_subcarriers() -> None:
    """A subcarrier that is NaN for the whole capture must not blank the panel.

    Guard, DC, and dropped pilot bins are non-finite for every frame -- 11 of
    256 on a real MTK capture. Accumulating one into the average poisons every
    bin of every window, which showed up as an entirely NaN spectrogram.
    """
    fs, win, hop = 20.0, 128, 64
    samples = _tone(2.0, fs, 1024, n_cols=4)
    samples[:, 1] = np.nan                       # structural null
    spec, freqs = stft_average(samples, fs, win, hop)

    assert np.isfinite(spec).all(), "a dead subcarrier must not propagate"
    peak_row = int(np.argmax(spec.mean(axis=1)))
    assert freqs[::-1][peak_row] == pytest.approx(2.0, abs=fs / win)

    # And the average is over the three live columns, not all four.
    live_only, _ = stft_average(samples[:, [0, 2, 3]], fs, win, hop)
    assert np.allclose(spec, live_only)


# ----------------------------------------------------------------------- #
#  compute_doppler orchestration                                          #
# ----------------------------------------------------------------------- #

from pathlib import Path  # noqa: E402

CAPTURES = Path(__file__).resolve().parent.parent / "captures"


def _capture_or_skip(name: str = "capture.dat") -> Path:
    p = CAPTURES / name
    if not p.is_file():
        pytest.skip(f"{name} not present")
    return p


def test_compute_doppler_shape_metadata_and_nyquist() -> None:
    """Grid shape follows the window, and f_max is the capture's own Nyquist."""
    from backend.tiles import compute_doppler, get_index

    p = _capture_or_skip()
    idx = get_index(p)
    t = np.asarray(idx.times, dtype=float)
    t0, t1 = float(t[0]), float(t[-1])
    expected_fs = 1.0 / float(np.median(np.diff(t)[np.diff(t) > 0]))

    spec, meta = compute_doppler(p, t0, t1, "amplitude", win_seconds=10.0)

    assert spec.dtype == np.float32
    assert spec.shape[0] > meta["win"] // 2      # zero-padded frequency grid
    assert spec.shape[1] >= 1
    assert meta["fs"] == pytest.approx(expected_fs, rel=1e-6)
    assert meta["f_max"] == pytest.approx(meta["fs"] / 2)
    assert meta["hop"] == meta["win"] // 2          # overlap 0.5
    assert meta["col_t0"] <= meta["col_t1"] <= t1 + 1e-6
    assert np.isfinite(spec).any(), "structural nulls must not blank the panel"


def test_compute_doppler_rejects_a_tile_only_metric() -> None:
    from backend.tiles import compute_doppler

    with pytest.raises(ValueError, match="metric"):
        compute_doppler(_capture_or_skip(), 0.0, 1e9, "csi_cir")


def test_compute_doppler_clamps_a_window_longer_than_the_range() -> None:
    """Zooming past the window length must not blank the panel.

    It previously raised, which surfaced as a 400 the frontend swallowed --
    the panel silently kept stale pixels.
    """
    from backend.tiles import compute_doppler, get_index

    p = _capture_or_skip()
    t0 = float(get_index(p).times[0])
    spec, meta = compute_doppler(p, t0, t0 + 20.0, "amplitude", win_seconds=600.0)

    assert spec.shape[1] >= 1
    assert meta["win_seconds"] < 600.0, "the window must report what it used"
    assert meta["win_seconds"] <= 20.0


def test_compute_doppler_still_refuses_a_range_too_short_to_mean_anything() -> None:
    """Clamping has a floor: a couple of samples is not a spectrum."""
    from backend.tiles import compute_doppler, get_index

    p = _capture_or_skip()
    t0 = float(get_index(p).times[0])
    with pytest.raises(ValueError, match="too few"):
        compute_doppler(p, t0, t0 + 1.0, "amplitude", win_seconds=30.0)


def test_compute_doppler_sample_rate_is_stable_across_zoom() -> None:
    """The frequency axis must not rescale as the user zooms in time.

    Taking fs from the visible slice made capture.dat report 1144 Hz on a 1 s
    view against its true 5.1 Hz, because its 1st-percentile interval is
    0.42 ms and a short slice can be all burst.
    """
    from backend.tiles import compute_doppler, get_index

    p = _capture_or_skip()
    t = np.asarray(get_index(p).times, dtype=float)
    t0 = float(t[0])

    rates = []
    for span in (float(t[-1] - t[0]), 100.0, 20.0):
        _, meta = compute_doppler(p, t0, t0 + span, "amplitude", win_seconds=10.0)
        rates.append(meta["fs"])

    assert max(rates) == pytest.approx(min(rates), rel=1e-9)
