"""Tests for backend.ratio — detection and correction of swapped rx streams."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.batch import decode_frames
from backend.index import FrameIndex
from backend.ratio import (
    CONFIDENCE_MIN,
    correct_ratio_amplitude,
    correct_ratio_phase,
    detect_swaps,
)

CAPTURE = Path(__file__).resolve().parent.parent / "captures" / "capture.dat"


def _synthetic(n: int = 40, num_sc: int = 64, seed: int = 0):
    """A smoothly drifting ratio-phase sequence, wrapped like real output."""
    rng = np.random.default_rng(seed)
    k = np.arange(num_sc)
    base = 0.05 * k + np.sin(k / 11.0)
    drift = np.cumsum(rng.normal(0, 0.01, size=n))[:, None]
    return np.angle(np.exp(1j * (base[None, :] + drift)))


# ----------------------------------------------------------------------- #
#  Detection on synthetic sequences with known swaps                      #
# ----------------------------------------------------------------------- #


def test_isolated_swap_is_detected():
    phase = _synthetic()
    phase[17] = -phase[17]

    flip = detect_swaps(phase)
    # Parity: frame 17 sits opposite to everything around it either way the
    # convention falls, so it must differ from both neighbours.
    assert flip[17] != flip[16]
    assert flip[17] != flip[18]


def test_run_of_swaps_is_detected_as_one_block():
    phase = _synthetic(n=60)
    phase[20:31] = -phase[20:31]

    flip = detect_swaps(phase)
    assert len(set(flip[20:31].tolist())) == 1, "the run must share one orientation"
    assert flip[20] != flip[19]
    assert flip[30] != flip[31]


def test_correction_restores_the_original_sequence():
    """Corrupt a clean sequence, correct it, and get the clean one back."""
    clean = _synthetic(n=50)
    corrupted = clean.copy()
    corrupted[[7, 8, 9, 23, 40]] = -corrupted[[7, 8, 9, 23, 40]]

    fixed = correct_ratio_phase(corrupted)

    # Orientation is only defined up to a global sign, so accept either.
    same = np.allclose(fixed, clean, atol=1e-5)
    negated = np.allclose(fixed, -clean, atol=1e-5)
    assert same or negated


def test_clean_sequence_is_left_alone():
    """No swaps present: every frame keeps one orientation."""
    phase = _synthetic(n=50)
    flip = detect_swaps(phase)
    assert len(set(flip.tolist())) == 1


def test_amplitude_correction_follows_the_phase_decision():
    """Both metrics must flip the same frames, or the panels disagree."""
    phase = _synthetic(n=40)
    amp = np.tile(np.linspace(-6, 6, 64), (40, 1)).astype(np.float32)
    phase[12] = -phase[12]
    amp[12] = -amp[12]

    flip = detect_swaps(phase)
    fixed_amp = correct_ratio_amplitude(amp, phase)

    expected = amp.copy()
    expected[flip] = -expected[flip]
    assert np.allclose(fixed_amp, expected)
    # And frame 12's amplitude is now consistent with its neighbours again.
    assert np.sign(fixed_amp[12, -1]) == np.sign(fixed_amp[11, -1])


def test_majority_sets_the_convention():
    """A batch opening on a swapped frame must not come out globally inverted."""
    phase = _synthetic(n=40)
    phase[0] = -phase[0]
    flip = detect_swaps(phase)
    assert flip.mean() <= 0.5
    assert flip[0] != flip[1]


# ----------------------------------------------------------------------- #
#  Degradation when neighbours are not comparable                         #
# ----------------------------------------------------------------------- #


def test_unrelated_frames_are_left_untouched():
    """Mixed transmitters: nothing is comparable, so nothing should flip.

    This is the 'source_mac=all' case. Flipping on noise would be worse than
    not correcting, so the confidence gate must decline to act.
    """
    rng = np.random.default_rng(3)
    noise = rng.uniform(-np.pi, np.pi, size=(200, 64))
    flip = detect_swaps(noise)
    assert len(set(flip.tolist())) == 1, "unrelated frames must not toggle parity"


def test_all_nan_frames_do_not_break_the_chain():
    """A 1-rx frame has no ratio; it must not derail the frames after it."""
    phase = _synthetic(n=40)
    phase[10] = np.nan
    phase[25] = -phase[25]

    flip = detect_swaps(phase)
    assert flip[25] != flip[24]
    assert flip[25] != flip[26]


def test_short_input_is_handled():
    assert detect_swaps(np.zeros((0, 8))).shape == (0,)
    assert detect_swaps(np.zeros((1, 8))).tolist() == [False]


def test_confidence_threshold_is_in_range():
    assert 0.0 < CONFIDENCE_MIN < 1.0


def test_correction_preserves_shape_dtype_and_nan():
    phase = _synthetic(n=20).astype(np.float32)
    phase[5, 3] = np.nan
    out = correct_ratio_phase(phase)
    assert out.shape == phase.shape
    assert out.dtype == np.float32
    assert np.isnan(out[5, 3])


def test_input_is_not_modified():
    phase = _synthetic(n=20)
    before = phase.copy()
    correct_ratio_phase(phase)
    correct_ratio_amplitude(np.zeros_like(phase), phase)
    assert np.array_equal(phase, before)


# ----------------------------------------------------------------------- #
#  Against the real capture                                               #
# ----------------------------------------------------------------------- #


@pytest.mark.skipif(not CAPTURE.is_file(), reason="captures/capture.dat not present")
def test_correction_reduces_artefacts_at_short_gaps():
    """The claim the panel makes, measured on real frames.

    Restricted to closely-spaced packets, where the channel cannot physically
    have moved much, so a large step between adjacent frames is an artefact
    rather than the room changing.
    """
    idx = FrameIndex(CAPTURE)
    macs = np.array(idx.source_macs)
    mac = max(set(macs.tolist()), key=lambda m: (macs == m).sum())
    sel = np.flatnonzero((macs == mac) & (idx.num_rx_arr >= 2))
    if len(sel) < 200:
        pytest.skip("not enough 2-rx frames from one transmitter")

    _, _, _, rp = decode_frames(CAPTURE, idx, sel)
    fixed = correct_ratio_phase(rp)

    dt = np.diff(idx.times[sel])
    close = dt < np.percentile(dt, 25)

    def bad(a):
        s = np.nanmedian(np.abs(np.angle(np.exp(1j * np.diff(a, axis=0)))), axis=1)
        return float(np.mean(s[close] > 0.5))

    assert bad(fixed) <= bad(rp), "correction must not add artefacts"


# ----------------------------------------------------------------------- #
#  pi rotation — a different corruption from the reciprocal swap          #
# ----------------------------------------------------------------------- #


def test_rotation_is_detected_and_distinguished_from_a_swap():
    """A pi-rotated block must be flagged as rotated, not as swapped."""
    from backend.ratio import detect_states

    phase = _synthetic(n=60)
    phase[20:31] = np.angle(np.exp(1j * (phase[20:31] + np.pi)))

    swap, rot = detect_states(phase)
    assert not swap.any(), "a rotation is not a swap"
    assert len(set(rot[20:31].tolist())) == 1
    assert rot[20] != rot[19]
    assert rot[30] != rot[31]


def test_rotation_is_corrected():
    clean = _synthetic(n=60)
    corrupted = clean.copy()
    corrupted[20:31] = np.angle(np.exp(1j * (corrupted[20:31] + np.pi)))

    fixed = correct_ratio_phase(corrupted)
    same = np.allclose(fixed, clean, atol=1e-4)
    negated = np.allclose(fixed, np.angle(np.exp(1j * -clean)), atol=1e-4)
    assert same or negated


def test_swap_and_rotation_together():
    """Both corruptions at once on the same block must both come out."""
    from backend.ratio import detect_states

    clean = _synthetic(n=60)
    corrupted = clean.copy()
    corrupted[20:31] = np.angle(np.exp(1j * (-corrupted[20:31] + np.pi)))

    swap, rot = detect_states(corrupted)
    assert swap[25] != swap[19]
    assert rot[25] != rot[19]

    fixed = correct_ratio_phase(corrupted)
    same = np.allclose(fixed, clean, atol=1e-4)
    negated = np.allclose(fixed, np.angle(np.exp(1j * -clean)), atol=1e-4)
    assert same or negated


def test_corrected_phase_stays_in_range():
    """Adding pi must not push the metric outside its declared [-pi, pi]."""
    phase = _synthetic(n=40)
    phase[10:20] = np.angle(np.exp(1j * (phase[10:20] + np.pi)))
    out = correct_ratio_phase(phase)
    finite = out[np.isfinite(out)]
    assert finite.min() >= -np.pi - 1e-5
    assert finite.max() <= np.pi + 1e-5


def test_amplitude_ignores_rotation():
    """A pi rotation leaves the dB amplitude alone, so nothing should move."""
    phase = _synthetic(n=40)
    amp = np.tile(np.linspace(-6, 6, 64), (40, 1)).astype(np.float32)
    phase[10:20] = np.angle(np.exp(1j * (phase[10:20] + np.pi)))

    out = correct_ratio_amplitude(amp, phase)
    assert np.allclose(out, amp), "rotation must not negate the amplitude"


@pytest.mark.skipif(not CAPTURE.is_file(), reason="captures/capture.dat not present")
def test_real_capture_correction_is_stable():
    """On real frames the correction must not introduce phase discontinuity."""
    idx = FrameIndex(CAPTURE)
    macs = np.array(idx.source_macs)
    mac = max(set(macs.tolist()), key=lambda m: (macs == m).sum())
    sel = np.flatnonzero((macs == mac) & (idx.num_rx_arr >= 2))
    if len(sel) < 100:
        pytest.skip("not enough 2-rx frames from one transmitter")

    _, _, _, rp = decode_frames(CAPTURE, idx, sel)
    fixed = correct_ratio_phase(rp)

    step = lambda a: float(
        np.nanmedian(np.abs(np.angle(np.exp(1j * np.diff(a, axis=0)))))
    )
    assert step(fixed) <= step(rp) + 1e-6


def test_boundary_survives_an_unreadable_transition():
    """The failure that motivated the segment-merge pass.

    A block flips at exactly the point where the two adjacent frames cannot
    be compared (here, a dropout sits on the boundary). Chaining alone must
    miss it and leave the whole block inverted; comparing the segments either
    side of the split recovers it.
    """
    from backend.ratio import detect_states

    clean = _synthetic(n=80)
    corrupted = clean.copy()
    corrupted[40:65] = np.angle(np.exp(1j * (corrupted[40:65] + np.pi)))
    corrupted[39] = np.nan  # the transition itself is unreadable
    corrupted[40] = np.nan

    swap, rot = detect_states(corrupted)
    # Frames well inside the rotated block must be flagged differently from
    # frames well outside it.
    assert rot[50] != rot[20]
    assert rot[50] != rot[75]


def test_long_inverted_region_is_recovered():
    """A region long enough to be internally self-consistent."""
    clean = _synthetic(n=200)
    corrupted = clean.copy()
    corrupted[60:140] = -corrupted[60:140]

    fixed = correct_ratio_phase(corrupted)
    same = np.allclose(fixed, clean, atol=1e-4)
    negated = np.allclose(fixed, np.angle(np.exp(1j * -clean)), atol=1e-4)
    assert same or negated


def test_correction_leaves_no_residual_boundary():
    """After correcting, no adjacent pair may still differ by ~pi."""
    clean = _synthetic(n=150)
    corrupted = clean.copy()
    corrupted[30:50] = -corrupted[30:50]
    corrupted[70:95] = np.angle(np.exp(1j * (corrupted[70:95] + np.pi)))
    corrupted[110] = -corrupted[110]

    fixed = correct_ratio_phase(corrupted)
    m = np.mean(np.exp(1j * np.diff(fixed, axis=0)), axis=1)
    bad = (np.abs(np.angle(m)) > np.pi / 2) & (np.abs(m) > 0.7)
    assert not bad.any(), f"{int(bad.sum())} residual boundaries remain"


# ----------------------------------------------------------------------- #
#  Amplitude anchoring — which side of a boundary is the right way up      #
# ----------------------------------------------------------------------- #


def _synthetic_amp(n: int = 600, num_sc: int = 64) -> np.ndarray:
    """A ratio-amplitude profile with real shape, as the antennas produce."""
    k = np.arange(num_sc)
    profile = 6.0 * np.sin(k / 9.0) + 3.0 * np.cos(k / 21.0)
    rng = np.random.default_rng(11)
    return profile[None, :] + rng.normal(0, 0.4, size=(n, num_sc))


def test_block_inversion_is_caught_by_the_amplitude_anchor():
    """The regression: a long region left inverted between two bad toggles.

    Phase alone cannot see this — the region agrees with itself perfectly, so
    every frame-to-frame check passes. Only the amplitude profile says which
    orientation is the right way up.
    """
    from backend.ratio import detect_states

    n = 600
    phase = _synthetic(n=n, num_sc=64)
    amp = _synthetic_amp(n=n, num_sc=64)

    # Invert a long block in *both* quantities, as a real swap would.
    block = slice(200, 450)
    phase[block] = -phase[block]
    amp[block] = -amp[block]

    without = detect_states(phase)[0]
    with_amp = detect_states(phase, amp)[0]

    # Phase alone places the boundaries but may settle on either side.
    # With amplitude, the block must end up opposite to the majority.
    assert with_amp[300] != with_amp[50]
    assert with_amp[300] != with_amp[550]
    # And the corrected amplitude must agree with the rest of the file.
    fixed = amp.copy()
    fixed[with_amp] = -fixed[with_amp]
    prof = np.median(fixed, axis=0)
    prof = prof - prof.mean()
    corr = ((fixed - fixed.mean(axis=1, keepdims=True)) * prof).sum(axis=1)
    assert (corr > 0).mean() > 0.9, "most frames must align with the profile"
    del without


def test_anchor_leaves_isolated_swaps_to_the_phase_passes():
    """Short excursions must not be blurred away by the smoothed anchor."""
    from backend.ratio import detect_states

    n = 600
    phase = _synthetic(n=n, num_sc=64)
    amp = _synthetic_amp(n=n, num_sc=64)
    phase[321] = -phase[321]
    amp[321] = -amp[321]

    swap, _ = detect_states(phase, amp)
    assert swap[321] != swap[320]
    assert swap[321] != swap[322]


def test_anchor_declines_on_a_featureless_profile():
    """With balanced antennas there is no shape to correlate against."""
    from backend.ratio import detect_states

    n = 600
    phase = _synthetic(n=n, num_sc=64)
    flat = np.zeros((n, 64))  # no profile at all
    a = detect_states(phase, flat)[0]
    b = detect_states(phase)[0]
    assert np.array_equal(a, b), "a featureless profile must change nothing"


def test_anchor_is_optional_and_backwards_compatible():
    from backend.ratio import detect_states

    phase = _synthetic(n=80)
    assert detect_states(phase)[0].shape == (80,)
    assert correct_ratio_phase(phase).shape == phase.shape


@pytest.mark.skipif(not CAPTURE.is_file(), reason="captures/capture.dat not present")
def test_real_capture_orientation_matches_the_amplitude_profile():
    """Every corrected frame should align with the file's own profile.

    Skips unless the capture can actually support the check. It needs enough
    frames for the anchor to engage, and an amplitude profile with real shape
    to correlate against — on a near-flat profile the correlation sign is
    noise and asserting on it would only produce a flaky test. The committed
    fixture has 272 frames and a 0.72 dB profile, so it skips; an hourly
    capture (4.51 dB) exercises it properly. Block-scale behaviour is covered
    deterministically by the synthetic tests above.
    """
    from backend.ratio import MIN_ANCHOR_RUN

    idx = FrameIndex(CAPTURE)
    macs = np.array(idx.source_macs)
    mac = max(set(macs.tolist()), key=lambda m: (macs == m).sum())
    sel = np.flatnonzero((macs == mac) & (idx.num_rx_arr >= 2))
    if len(sel) < MIN_ANCHOR_RUN * 2:
        pytest.skip("too few frames for the amplitude anchor to engage")

    _, _, ramp, rp = decode_frames(CAPTURE, idx, sel)
    swap, _ = detect_states(rp, ramp)

    fixed = np.where(np.isfinite(ramp), ramp, np.nan)
    fixed = fixed.copy()
    fixed[swap] = -fixed[swap]
    prof = np.nanmedian(fixed, axis=0)
    prof = np.nan_to_num(prof - np.nanmean(prof))
    if prof.std() < 2.0:
        pytest.skip("amplitude profile too flat for the orientation check")
    corr = np.nansum(
        np.nan_to_num(fixed - np.nanmean(fixed, axis=1, keepdims=True)) * prof, axis=1
    )
    w = min(201, max(3, len(corr) // 4))
    smoothed = np.convolve(corr, np.ones(w) / w, mode="same")
    assert (smoothed < 0).mean() < 0.05, "no large stretch may sit inverted"


def test_rotation_block_is_caught_by_the_phase_anchor():
    """A long region left pi-rotated between two miscounted toggles.

    The amplitude anchor cannot see this: a pi rotation multiplies the ratio
    by -1, which leaves the dB amplitude untouched. Only the phase's own mean
    direction says the region sits a pi away from the rest of the capture.
    """
    from backend.ratio import detect_states

    n = 900
    phase = _synthetic(n=n, num_sc=64)
    amp = _synthetic_amp(n=n, num_sc=64)

    block = slice(300, 650)
    phase[block] = np.angle(np.exp(1j * (phase[block] + np.pi)))
    # amplitude deliberately untouched — that is the point

    swap, rot = detect_states(phase, amp)
    assert rot[450] != rot[100]
    assert rot[450] != rot[800]

    fixed = correct_ratio_phase(phase, amp)
    v = np.mean(np.exp(1j * fixed), axis=1)
    ref = v.sum() / abs(v.sum())
    agree = np.real(v * np.conj(ref))
    assert (agree < 0).mean() < 0.05, "no stretch may sit a pi from the mean"


def test_frequent_rotations_do_not_desynchronise():
    """Rotations are common in real data (~24% of transitions); parity must hold."""
    from backend.ratio import detect_states

    rng = np.random.default_rng(5)
    n = 1200
    phase = _synthetic(n=n, num_sc=64)
    amp = _synthetic_amp(n=n, num_sc=64)

    events = rng.random(n) < 0.24
    state = np.cumsum(events) % 2 == 1
    corrupted = phase.copy()
    corrupted[state] = np.angle(np.exp(1j * (corrupted[state] + np.pi)))

    fixed = correct_ratio_phase(corrupted, amp)
    v = np.mean(np.exp(1j * fixed), axis=1)
    ref = v.sum() / abs(v.sum())
    agree = np.real(v * np.conj(ref))
    assert (agree < 0).mean() < 0.05


def test_rotation_anchor_leaves_short_events_alone():
    from backend.ratio import detect_states

    n = 900
    phase = _synthetic(n=n, num_sc=64)
    amp = _synthetic_amp(n=n, num_sc=64)
    phase[500] = np.angle(np.exp(1j * (phase[500] + np.pi)))

    _, rot = detect_states(phase, amp)
    assert rot[500] != rot[499]
    assert rot[500] != rot[501]
