"""Tests for backend.truth -- camera labels with the untrusted seconds removed."""

from __future__ import annotations

import numpy as np
import pytest

from backend import truth as tr

# One frame per second, like the webcam: empty 0-59, occupied 60-179, empty 180-299.
T = np.arange(300, dtype=float)
P = np.zeros(300, dtype=bool)
P[60:180] = True


def test_transitions_sit_midway_between_the_frames_that_flip() -> None:
    assert tr.transition_times(T, P).tolist() == [59.5, 179.5]
    assert tr.transition_times(T[:1], P[:1]).size == 0
    assert tr.transition_times(T, np.zeros(300, bool)).size == 0


def test_the_margin_removes_empty_frames_on_both_sides_and_no_occupied_ones() -> None:
    amb = tr.ambiguous_frames(T, P, margin_s=5.0)
    # Approach: 55..59 are within 5 s of the 59.5 flip and empty.
    assert np.flatnonzero(amb).tolist() == [55, 56, 57, 58, 59, 180, 181, 182, 183, 184]
    assert not amb[P].any()


def test_margin_zero_trusts_every_frame() -> None:
    assert not tr.ambiguous_frames(T, P, margin_s=0.0).any()


def test_a_lone_dropped_detection_is_excluded_not_scored_as_absence() -> None:
    p = P.copy()
    p[120] = False                      # one missed frame mid-sit
    amb = tr.ambiguous_frames(T, p, margin_s=5.0)
    assert amb[120]
    assert not amb[115:120].any() and not amb[121:126].any()
    truth, excluded = tr.cell_truth(np.arange(300) + 0.0, T, p, 0.5, margin_s=5.0)
    assert np.isnan(truth[120]) and excluded[120]
    assert truth[119] == 1.0 and truth[121] == 1.0


def test_cells_next_to_a_transition_are_excluded_and_counted() -> None:
    centres = np.arange(300, dtype=float)
    truth, excluded = tr.cell_truth(centres, T, P, 0.5, margin_s=5.0)
    assert np.nansum(truth == 0.0) == 180 - 10     # 5 s before the entry, 5 s after the exit
    assert np.nansum(truth == 1.0) == 120
    assert excluded.sum() == 10
    unpadded, none = tr.cell_truth(centres, T, P, 0.5, margin_s=0.0)
    assert np.nansum(unpadded == 0.0) == 180 and none.sum() == 0


def test_a_cell_the_camera_did_not_cover_is_not_counted_as_excluded() -> None:
    truth, excluded = tr.cell_truth(np.array([400.0]), T, P, 0.5)
    assert np.isnan(truth[0]) and not excluded[0]


def test_a_wide_cell_is_judged_by_its_whole_span_and_mixed_spans_are_excluded() -> None:
    # A 15 s cell centred 3 s before the entry spans frames 50..64: the margin
    # removes 55..59, leaving 5 empty and 5 occupied -- neither empty nor more
    # than half occupied, so it is excluded.
    truth, excluded = tr.cell_truth(np.array([57.0]), T, P, 7.5, margin_s=5.0)
    assert np.isnan(truth[0]) and excluded[0]
    # Centred 10 s into the sit (frames 62..77): occupied throughout.
    truth, excluded = tr.cell_truth(np.array([70.0]), T, P, 7.5, margin_s=5.0)
    assert truth[0] == 1.0


def test_confusion_counts_only_scored_cells_and_reports_the_rest() -> None:
    truth = np.array([1.0, 1.0, 0.0, 0.0, np.nan, np.nan])
    excluded = np.array([False, False, False, False, True, False])
    predicted = np.array([True, False, True, False, True, True])
    c = tr.confusion(truth, predicted, excluded)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)
    assert c["total"] == 4 and c["excluded"] == 1
    assert c["accuracy"] == 0.5 and c["recall"] == 0.5 and c["specificity"] == 0.5


def test_rates_without_a_denominator_are_none_not_zero() -> None:
    c = tr.confusion(np.array([1.0, 1.0]), np.array([True, False]))
    assert c["specificity"] is None and c["recall"] == 0.5


def test_shape_mismatches_are_refused() -> None:
    with pytest.raises(ValueError):
        tr.transition_times(T[:10], P[:9])
    with pytest.raises(ValueError):
        tr.confusion(np.array([1.0]), np.array([True, False]))
