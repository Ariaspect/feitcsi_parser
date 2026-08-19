"""Tests for backend.phase — subcarrier unwrapping and linear detrending."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.batch import decode_frames
from backend.index import FrameIndex
from backend.phase import detrend_subcarrier, unwrap_subcarrier

CAPTURE = Path(__file__).resolve().parent.parent / "captures" / "capture.dat"


# ----------------------------------------------------------------------- #
#  Synthetic signals — exact expectations                                 #
# ----------------------------------------------------------------------- #


def test_unwrap_recovers_known_ramp():
    """A ramp steeper than 2*pi comes back continuous, not sawtoothed."""
    k = np.arange(64)
    true_phase = 0.4 * k  # spans ~25 rad, wraps four times
    wrapped = np.angle(np.exp(1j * true_phase))[None, :]

    out = unwrap_subcarrier(wrapped)

    # np.unwrap anchors on the first sample, which the wrap already folded
    # into (-pi, pi]; the recovered curve differs from the truth only by that
    # constant, so compare shapes rather than absolute values.
    assert np.allclose(out[0] - out[0, 0], true_phase - true_phase[0], atol=1e-5)
    assert np.all(np.abs(np.diff(out[0])) < np.pi)


def test_unwrap_is_a_noop_on_a_flat_row():
    flat = np.full((3, 16), 0.7, dtype=np.float32)
    assert np.allclose(unwrap_subcarrier(flat), flat, atol=1e-6)


def test_detrend_removes_offset_and_slope():
    """Offset + slope + a bump must come back as just the bump."""
    k = np.arange(128)
    bump = np.zeros(128)
    bump[40:60] = 1.0
    bump -= bump.mean()

    true_phase = 2.1 + 0.37 * k + bump
    wrapped = np.angle(np.exp(1j * true_phase))[None, :]

    out = detrend_subcarrier(wrapped)[0]

    # The bump itself carries a little slope through the fit, so compare
    # against the same bump put through an identical least-squares removal.
    coeffs = np.polyfit(k, bump, 1)
    expected = bump - np.polyval(coeffs, k)
    assert np.allclose(out, expected, atol=1e-4)


def test_detrend_is_invariant_to_a_per_frame_offset():
    """The whole point: a random per-packet phase offset must not survive."""
    rng = np.random.default_rng(0)
    k = np.arange(96)
    base = 0.2 * k + np.sin(k / 9.0)
    frames = np.tile(base, (20, 1))
    offsets = rng.uniform(-np.pi, np.pi, size=(20, 1))

    wrapped_plain = np.angle(np.exp(1j * frames))
    wrapped_shifted = np.angle(np.exp(1j * (frames + offsets)))

    a = detrend_subcarrier(wrapped_plain)
    b = detrend_subcarrier(wrapped_shifted)
    assert np.allclose(a, b, atol=1e-4)
    # And every frame collapses onto the same curve.
    assert np.nanmax(np.nanstd(b, axis=0)) < 1e-4


def test_detrend_is_invariant_to_a_per_frame_slope():
    """Sampling time offset shows up as a slope; it must not survive either."""
    k = np.arange(96)
    shape = np.sin(k / 7.0)
    slopes = np.array([0.0, 0.05, -0.12, 0.31])[:, None]
    frames = shape[None, :] + slopes * k[None, :]
    wrapped = np.angle(np.exp(1j * frames))

    out = detrend_subcarrier(wrapped)
    assert np.nanmax(np.nanstd(out, axis=0)) < 1e-4


# ----------------------------------------------------------------------- #
#  Non-finite handling                                                    #
# ----------------------------------------------------------------------- #


def test_all_nan_row_stays_nan():
    """1-rx frames have no CSI ratio at all; the row must not be invented."""
    rows = np.full((2, 32), np.nan, dtype=np.float32)
    assert np.all(np.isnan(unwrap_subcarrier(rows)))
    assert np.all(np.isnan(detrend_subcarrier(rows)))


def test_isolated_nan_does_not_poison_the_rest_of_the_row():
    """np.unwrap accumulates, so one hole must not NaN every later subcarrier."""
    k = np.arange(64)
    wrapped = np.angle(np.exp(1j * (0.4 * k)))[None, :].astype(np.float64).copy()
    clean = unwrap_subcarrier(wrapped)

    holed = wrapped.copy()
    holed[0, 20] = np.nan
    out = unwrap_subcarrier(holed)

    assert np.isnan(out[0, 20])
    assert np.isfinite(out[0, 21:]).all()
    # The bridge carries the unwrap across the gap, so the tail is unchanged.
    assert np.allclose(out[0, 21:], clean[0, 21:], atol=1e-4)


def test_nan_positions_are_preserved_by_detrend():
    rows = np.zeros((2, 32), dtype=np.float32)
    rows[0, 5] = np.nan
    out = detrend_subcarrier(rows)
    assert np.isnan(out[0, 5])
    assert np.isfinite(out[1]).all()


def test_row_with_one_finite_sample_has_no_determined_line():
    rows = np.full((1, 16), np.nan, dtype=np.float32)
    rows[0, 3] = 0.5
    assert np.all(np.isnan(detrend_subcarrier(rows)))


def test_empty_input_keeps_its_shape():
    empty = np.empty((0, 242), dtype=np.float32)
    assert unwrap_subcarrier(empty).shape == (0, 242)
    assert detrend_subcarrier(empty).shape == (0, 242)


def test_output_is_float32():
    rows = np.zeros((2, 8), dtype=np.float64)
    assert unwrap_subcarrier(rows).dtype == np.float32
    assert detrend_subcarrier(rows).dtype == np.float32


def test_input_is_not_modified():
    rows = np.angle(np.exp(1j * 0.4 * np.arange(32)))[None, :].copy()
    before = rows.copy()
    unwrap_subcarrier(rows)
    detrend_subcarrier(rows)
    assert np.array_equal(rows, before)


# ----------------------------------------------------------------------- #
#  Against the real capture                                               #
# ----------------------------------------------------------------------- #


@pytest.mark.skipif(not CAPTURE.is_file(), reason="captures/capture.dat not present")
def test_real_capture_unwrap_has_no_residual_jumps():
    idx = FrameIndex(CAPTURE)
    _, phase, _, ratio_phase = decode_frames(CAPTURE, idx, np.arange(min(200, idx.count)))

    for arr in (phase, ratio_phase):
        out = unwrap_subcarrier(arr)
        finite = np.isfinite(out)
        if not finite.any():
            continue
        steps = np.abs(np.diff(out, axis=1))
        # Every remaining step is below pi — that is what unwrapping means.
        assert np.nanmax(steps[np.isfinite(steps)]) < np.pi + 1e-4


@pytest.mark.skipif(not CAPTURE.is_file(), reason="captures/capture.dat not present")
def test_detrending_beats_unwrapping_for_cross_frame_stability():
    """The claim the panel makes: detrend stabilises frames, unwrap alone does not."""
    idx = FrameIndex(CAPTURE)
    _, phase, _, _ = decode_frames(CAPTURE, idx, np.arange(min(500, idx.count)))

    spread = lambda a: float(np.nanmean(np.nanstd(a, axis=0)))
    wrapped_spread = spread(phase)
    unwrapped_spread = spread(unwrap_subcarrier(phase))
    detrended_spread = spread(detrend_subcarrier(phase))

    assert detrended_spread < wrapped_spread
    # Unwrapping alone frees the per-packet offset and slope to accumulate, so
    # it makes cross-frame spread worse, not better. This is why the detrend
    # toggle exists rather than unwrapping being applied silently.
    assert unwrapped_spread > wrapped_spread


# ----------------------------------------------------------------------- #
#  Time-axis unwrapping                                                   #
# ----------------------------------------------------------------------- #


def test_time_unwrap_recovers_a_continuous_ramp():
    """A channel turning steadily must come back as a straight line."""
    from backend.phase import unwrap_time

    t = np.arange(60) * 0.1
    true_phase = np.linspace(0, 24, 60)  # spans ~4 turns, wraps repeatedly
    wrapped = np.angle(np.exp(1j * true_phase))[:, None] * np.ones((1, 8))

    out = unwrap_time(wrapped, t)
    # Anchored at the segment start, so compare against the ramp less its own
    # first value.
    assert np.allclose(out[:, 0], true_phase - true_phase[0], atol=1e-4)


def test_time_unwrap_restarts_after_a_gap():
    """Nothing may carry across a dropout."""
    from backend.phase import unwrap_time

    t = np.arange(60) * 0.1
    t[30:] += 100.0  # a 100 s hole
    true_phase = np.linspace(0, 24, 60)
    wrapped = np.angle(np.exp(1j * true_phase))[:, None] * np.ones((1, 4))

    out = unwrap_time(wrapped, t)
    assert np.allclose(out[0], 0.0, atol=1e-6), "first segment anchored at 0"
    assert np.allclose(out[30], 0.0, atol=1e-6), "second segment re-anchored at 0"
    assert out[29, 0] > 10.0, "first segment still accumulated before the gap"


def test_time_unwrap_is_per_subcarrier():
    """Each subcarrier accumulates independently."""
    from backend.phase import unwrap_time

    t = np.arange(40) * 0.1
    ramps = np.outer(np.linspace(0, 12, 40), np.array([1.0, -1.0, 0.5]))
    wrapped = np.angle(np.exp(1j * ramps))

    out = unwrap_time(wrapped, t)
    assert out[-1, 0] > 10 and out[-1, 1] < -10 and 4 < out[-1, 2] < 8


def test_time_unwrap_leaves_the_wrapped_range():
    """Accumulated phase is not an angle any more."""
    from backend.phase import unwrap_time

    t = np.arange(50) * 0.1
    wrapped = np.angle(np.exp(1j * np.linspace(0, 30, 50)))[:, None]
    out = unwrap_time(wrapped, t)
    assert out.max() > np.pi


def test_time_unwrap_preserves_nan_and_does_not_poison():
    """One missing frame must not NaN every frame after it."""
    from backend.phase import unwrap_time

    t = np.arange(40) * 0.1
    wrapped = np.angle(np.exp(1j * np.linspace(0, 12, 40)))[:, None] * np.ones((1, 3))
    wrapped[15, 1] = np.nan

    out = unwrap_time(wrapped, t)
    assert np.isnan(out[15, 1])
    assert np.isfinite(out[16:, 1]).all()


def test_time_unwrap_handles_empty_and_single_frame():
    from backend.phase import unwrap_time

    assert unwrap_time(np.empty((0, 8)), np.empty(0)).shape == (0, 8)
    out = unwrap_time(np.zeros((1, 8)), np.zeros(1))
    assert out.shape == (1, 8) and np.allclose(out, 0.0)


@pytest.mark.skipif(not CAPTURE.is_file(), reason="captures/capture.dat not present")
def test_time_unwrap_on_corrected_ratio_stays_continuous():
    """On real frames the unwrapped trace must have no 2*pi cliffs."""
    from backend.phase import unwrap_time
    from backend.ratio import correct_ratio_phase

    idx = FrameIndex(CAPTURE)
    macs = np.array(idx.source_macs)
    mac = max(set(macs.tolist()), key=lambda m: (macs == m).sum())
    sel = np.flatnonzero((macs == mac) & (idx.num_rx_arr >= 2))
    if len(sel) < 100:
        pytest.skip("not enough 2-rx frames from one transmitter")

    _, _, _, rp = decode_frames(CAPTURE, idx, sel)
    out = unwrap_time(correct_ratio_phase(rp), idx.times[sel])

    steps = np.abs(np.diff(out, axis=0))
    finite = steps[np.isfinite(steps)]
    # Within a segment nothing may jump by a full turn.
    assert np.percentile(finite, 99.9) < 2 * np.pi
