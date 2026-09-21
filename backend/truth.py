"""Camera labels as scoring truth, minus the seconds the camera cannot vouch for.

The webcam labels one frame per second: a person is "present" when a box sits
inside the chair-area ROI. That is the instantaneous content of the frame, and
it is a narrower claim than the one a WiFi detector is being scored on. WiFi
sees the whole room, and through its wall; the camera sees its field of view.
So on the way in a person is in the CSI for several seconds before they are
in the label, and on the way out for several seconds after -- and on every
labelled run those seconds have been scored as the detector's false
positives while the detector was right and the label was wrong. Measured:
the walk between the door and the chair costs about 5 s at this room's
geometry (the same margin ``Presence.tsx`` keeps calibration stretches away
from transitions); the peer session found 42 of 65 false positives within
20 s of a transition and 86% of the correctable ones within 5 s.

The ambiguity is one-sided. Next to a transition the camera's *occupied*
frames are sure -- the person is in the ROI, which is within the KPI's 3 m --
while its *empty* frames may or may not hold someone crossing the room. So the
margin removes empty frames near a transition and leaves occupied ones alone,
which is exactly the rule the calibration picker already applies.

Removed frames are excluded from scoring in both directions and counted. A
rule that quietly discards data is indistinguishable from one that fixes it,
so ``excluded`` travels with every confusion matrix built here. And ``margin_s
= 0`` reproduces the unpadded scoring every earlier number was produced with.
"""

from __future__ import annotations

import numpy as np

# Seconds of the empty label next to a transition that are not trusted.
DEFAULT_MARGIN_S = 5.0
# A cell is occupied when more than this fraction of its trusted frames are;
# empty when none are; anything between is not scored.
OCCUPIED_FRACTION = 0.5


def transition_times(times: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Instants where the label flips, midway between the two frames."""
    times = np.asarray(times, dtype=float)
    present = np.asarray(present, dtype=bool)
    if times.shape != present.shape:
        raise ValueError(f"times {times.shape} and present {present.shape} must match")
    if times.size < 2:
        return np.zeros(0, dtype=float)
    flip = present[1:] != present[:-1]
    return 0.5 * (times[1:][flip] + times[:-1][flip])


def ambiguous_frames(
    times: np.ndarray, present: np.ndarray, margin_s: float = DEFAULT_MARGIN_S
) -> np.ndarray:
    """Frames the camera called empty within ``margin_s`` of a transition.

    A lone dropped detection in the middle of a sit -- one empty frame between
    two occupied ones -- is a pair of transitions with one empty frame inside
    the margin of both, so it is excluded rather than scored as a second of
    absence. The occupied frames around it stay.
    """
    times = np.asarray(times, dtype=float)
    present = np.asarray(present, dtype=bool)
    if margin_s <= 0 or times.size < 2:
        return np.zeros(times.shape, dtype=bool)
    flips = transition_times(times, present)
    if flips.size == 0:
        return np.zeros(times.shape, dtype=bool)
    near = np.min(np.abs(times[:, None] - flips[None, :]), axis=1) <= margin_s
    return near & ~present


def cell_truth(
    centres: np.ndarray,
    times: np.ndarray,
    present: np.ndarray,
    half_span: float,
    margin_s: float = DEFAULT_MARGIN_S,
) -> tuple[np.ndarray, np.ndarray]:
    """One truth value per verdict cell: 1 occupied, 0 empty, NaN unscored.

    A cell takes the trusted camera frames within ``half_span`` of its centre.
    No frames at all means the camera did not cover it (NaN, not counted as
    excluded); frames that were all removed by the margin, or a mix that is
    neither empty nor more than half occupied, means it is excluded (NaN, and
    counted). Returns ``(truth, excluded)``.
    """
    centres = np.asarray(centres, dtype=float)
    times = np.asarray(times, dtype=float)
    present = np.asarray(present, dtype=bool)
    ambiguous = ambiguous_frames(times, present, margin_s)

    truth = np.full(centres.shape, np.nan)
    excluded = np.zeros(centres.shape, dtype=bool)
    for i, c in enumerate(centres):
        m = (times >= c - half_span) & (times <= c + half_span)
        if not m.any():
            continue
        kept = m & ~ambiguous
        if not kept.any():
            excluded[i] = True
            continue
        frac = float(present[kept].mean())
        if frac == 0.0:
            truth[i] = 0.0
        elif frac > OCCUPIED_FRACTION:
            truth[i] = 1.0
        else:
            excluded[i] = True
    return truth, excluded


def confusion(truth: np.ndarray, predicted: np.ndarray, excluded: np.ndarray | None = None) -> dict:
    """Counts and rates over the cells that carry a truth value.

    Rates are ``None`` where they have no denominator: a capture with no
    empty cell has no specificity, and reporting 0% would be a different
    claim from "not measured".
    """
    truth = np.asarray(truth, dtype=float)
    predicted = np.asarray(predicted, dtype=bool)
    if truth.shape != predicted.shape:
        raise ValueError(f"truth {truth.shape} and predicted {predicted.shape} must match")
    scored = np.isfinite(truth)
    occ = truth > 0.5
    tp = int(np.sum(scored & occ & predicted))
    fn = int(np.sum(scored & occ & ~predicted))
    fp = int(np.sum(scored & ~occ & predicted))
    tn = int(np.sum(scored & ~occ & ~predicted))
    total = tp + fp + fn + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": total,
        "excluded": int(np.sum(excluded)) if excluded is not None else 0,
        "accuracy": (tp + tn) / total if total else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "specificity": tn / (fp + tn) if (fp + tn) else None,
        "precision": tp / (tp + fp) if (tp + fp) else None,
    }
