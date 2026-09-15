"""Per-gain-state correction of the receiver's own amplitude distortion.

The NIC's automatic gain control holds the digital CSI *level* constant --
measured across 36 September captures, frames in a non-dominant gain state sit
a median of 0.05 dB from the dominant one -- but each gain state has its own
frequency response, so the *shape* tilts. That tilt is not the room. It shows
in an amplitude heatmap as isolated single-frame vertical stripes, which is
the form the artifact takes precisely because the level is already flat: a
median-amplitude line plot never sees it.

Three measurements say it is the radio rather than the channel:

* **It is monotonic in the gain step.** On 20260911_095127 the mean absolute
  shape error against the dominant state runs 0.40 / 1.17 / 2.02 / 2.32 /
  4.60 dB for RSSI steps of 1, 1, 2, 3 and 4 dB. Over 33 captures the
  correlation between ``|RSSI step|`` and shape error has a median of +0.928.

* **The signature repeats across captures.** Normalised per dB of step, the
  distortion profile correlates at +0.993 (median over 15 pairs) across six
  captures from four different days in four different room states. A channel
  or transmitter effect cannot do that; a property of the receive path can.

* **It is not noise.** Frames within one gain state agree with each other
  just as tightly as frames in the dominant state do -- 0.587 against 0.657,
  0.261 against 0.249, 0.069 against 0.075 dB on three captures. The
  off-state frames are differently shaped, not noisier.

Across the same 36 captures, 7.8-72% of 2x1 frames (median 36.9%) sit in a
non-dominant gain state, and the worst state's shape error has a median of
3.88 dB. The CSI *ratio* divides the common gain out and is 17x less affected
(median 0.23 dB), which is why the presence detector never had to care and
the amplitude panel always did.

**What this corrects and what it leaves alone.** Only ``amplitude``. The ratio
needs no correction and measuring one there would be measuring noise; the
effect of a gain step on *phase* has not been measured here, so phase is left
untouched rather than corrected on a guess.

**Why the correction is per FRAME and not per RSSI value.** RSSI is a proxy
for the gain state and it is imperfect at a boundary. On 20260911_095127 the
RSSI -48 frames are bimodal: 39 of them sit with the deep states and 85 with
the shallow ones, so any single per-state offset is wrong for one group or the
other, and those 39 were the loudest stripes left after a per-state pass. What
resolves it is that 90.9% of the variance of the measured distortions lies
along ONE direction, so each frame's own coefficient along that direction can
be measured individually and subtracted. Measured on the stripe metric (a
frame's mean absolute distance from its neighbours' profile), frames more than
3 dB out go 173 -> 0 and those over 1 dB go 738 -> 95 against 541 for a
per-state pass.

That firing decision needs a floor, and the floor is measured rather than
chosen: the same coefficient is computed on DOMINANT-state frames, where it
should be zero, and its 99th percentile becomes the threshold. Below it a
frame is left alone, because below it the artifact cannot be told from the
room. What that null costs when it does fire is small -- 0.046-0.181 dB across
three captures, against the 4-9 dB being removed.

**When it declines, and why that is the right answer.** On some captures the
floor sits above the whole signal and almost nothing is corrected --
20260911_123609 is the clearest, with a 1 dB gain step and a floor of 100.6.
That is not a failure to tune: its dominant state is itself split (the null
jumps from 1.4 at the median to 38.4 at the 75th percentile, so a quarter of
the frames that should read zero do not), which says its RSSI does not resolve
its gain states at all. Dropping the floor to median+5*MAD there would fire on
59% of off-state frames and on 39% of DOMINANT ones -- it would start
rewriting good frames. The conservative refusal is the honest outcome, and the
amplitude comes back exactly as decoded.

**The table is measured per capture, not hard-coded.** The signature repeats
well enough that a fixed table is tempting, and 20260910_203337 is why it is
not built that way: its signature correlates 0.935-0.947 with the others
rather than 0.99+. A per-capture table costs one pass over sampled frames and
cannot be wrong about the radio it was measured on.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np

__all__ = ["GainTable", "build_gain_table", "apply_gain_table"]

# Half-width, in frames, of the neighbourhood an off-state frame is judged
# against. The comparison is deliberately LOCAL: a global "off-state mean
# minus dominant mean" would also contain every way the room differed while
# the gain happened to be stepped, and subtracting that would remove signal.
# At ~19 Hz this is +/-2.6 s, over which the channel is effectively static
# while the gain state is not -- off-state frames arrive overwhelmingly as
# isolated singles (median run length 1 frame, measured on 20260911_095127).
DEFAULT_HALF_WIDTH = 50

# Dominant-state frames required inside that neighbourhood before a frame
# contributes. Fewer than this and the local profile is itself an estimate
# with more noise than the offset it is trying to measure.
MIN_LOCAL_FRAMES = 8

# Contributions required before a gain state gets an offset at all. A state
# seen a handful of times is left uncorrected rather than corrected badly --
# the frames stay exactly as decoded, which is the same thing the whole table
# being absent would do to them.
MIN_STATE_FRAMES = 20

# Percentile of the dominant state's own coefficient that a frame must clear
# before the per-frame correction fires. The dominant state is by definition
# undistorted, so whatever coefficient it reaches is the room and the noise
# rather than the gain -- which makes this a measured floor rather than a
# chosen one. At the 99th percentile roughly one dominant-looking frame in a
# hundred would be touched, which is the false-fire rate being bought.
NULL_PERCENTILE = 99.0

# Deltas needed before a shared direction is worth extracting at all. Below
# this the first singular vector is fitting noise, and the per-frame path is
# skipped in favour of the per-state offsets.
MIN_SHAPE_FRAMES = 30


class GainTable(NamedTuple):
    """Per-gain-state amplitude shape offsets, in dB, for one capture.

    ``offsets`` maps a gain state (the frame's reported RSSI) to a
    per-subcarrier dB offset to SUBTRACT from frames in that state. The
    dominant state is absent by construction -- it is the reference every
    other state is expressed against, so its offset is zero.

    Every offset has had its own median across subcarriers removed, so the
    correction is shape-only by construction. That is deliberate: the artifact
    being corrected is shape-only (median level shift 0.05 dB across 36
    captures), so a correction that carried a level term would be correcting
    something that was never measured. A frame's own median can still drift by
    a few hundredths of a dB, since a median is not linear -- measured at
    +0.04 dB over 20260911_095127 -- but no level offset is ever applied.

    ``shape`` is the single direction the distortions lie along, unit norm and
    zero median, with 0 in bins no frame could measure. ``null_k`` is the
    coefficient along it that the dominant state itself reaches 99% of the
    time -- the floor a frame must clear before it is corrected at all.

    ``offsets`` remains the fallback for views whose rows are not consecutive
    frames, where a frame has no neighbours to be measured against.
    """

    offsets: dict[int, np.ndarray]
    dominant: int
    n_frames: int
    shape: np.ndarray | None = None
    null_k: float = 0.0

    @property
    def n_states(self) -> int:
        """How many gain states carry a correction, the dominant one aside."""
        return len(self.offsets)


@contextmanager
def _quiet_all_nan():
    """Suppress NumPy's all-NaN-slice warning, which here is expected.

    A structurally dead subcarrier is NaN in every frame -- 11 of 256 on a
    MediaTek capture -- so a median down the frame axis legitimately meets a
    column with nothing in it. The result is NaN, which every caller below
    already handles; the warning says only that the data has guard bands.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN slice encountered",
                                RuntimeWarning)
        yield


def _level_removed(amplitude: np.ndarray) -> np.ndarray:
    """Per-frame profile with its own level taken out.

    Median rather than mean across subcarriers, so a few extreme bins -- a
    guard-band edge, a bin the decoder dropped -- cannot set the level the
    shape is then measured against.
    """
    amp = np.asarray(amplitude, dtype=float)
    with np.errstate(invalid="ignore"), _quiet_all_nan():
        return amp - np.nanmedian(amp, axis=1, keepdims=True)


def build_gain_table(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    half_width: int = DEFAULT_HALF_WIDTH,
    min_local: int = MIN_LOCAL_FRAMES,
    min_state: int = MIN_STATE_FRAMES,
) -> GainTable | None:
    """Measure one capture's per-gain-state amplitude offsets.

    *segments* is a sequence of ``(amplitude, rssi)`` pairs, each a stretch of
    CONSECUTIVE frames: ``amplitude`` is ``(n_frames, n_subcarriers)`` in dB
    and ``rssi`` is ``(n_frames,)``. Several stretches may be passed and they
    are pooled, which is how a whole capture gets covered without decoding it
    end to end. They must be consecutive *within* a stretch, because every
    comparison here is against a frame's own neighbours; concatenating two
    distant stretches into one would invent neighbours minutes apart, and the
    room between them is exactly what this must not absorb.

    Returns ``None`` when nothing can be measured -- no second gain state, or
    none of them seen often enough. ``None`` means "leave every frame as
    decoded", which is what a caller should do rather than invent a table.
    """
    prof_segs: list[tuple[np.ndarray, np.ndarray]] = []
    for amp, rssi in segments:
        a = np.asarray(amp, dtype=float)
        r = np.asarray(rssi)
        if a.ndim != 2:
            raise ValueError(f"amplitude must be 2-D (n_frames, n_sc), got {a.shape}")
        if r.shape != (a.shape[0],):
            raise ValueError(
                f"rssi must be 1-D of length {a.shape[0]}, got {r.shape}"
            )
        if a.shape[0]:
            prof_segs.append((_level_removed(a), r.astype(np.int64)))
    if not prof_segs:
        return None
    if len({p.shape[1] for p, _ in prof_segs}) != 1:
        raise ValueError("every segment must have the same subcarrier count")

    all_rssi = np.concatenate([r for _, r in prof_segs])
    values, counts = np.unique(all_rssi, return_counts=True)
    if values.size < 2:
        return None
    dominant = int(values[int(counts.argmax())])

    deltas: dict[int, list[np.ndarray]] = defaultdict(list)
    for prof, rssi in prof_segs:
        is_dom = rssi == dominant
        if not is_dom.any():
            continue
        dom_idx = np.flatnonzero(is_dom)
        for i in np.flatnonzero(~is_dom):
            # Dominant-state frames within +/-half_width of this one. searchsorted
            # over the dominant indices rather than a mask over the window, so
            # the cost does not grow with the window.
            lo = int(np.searchsorted(dom_idx, i - half_width, side="left"))
            hi = int(np.searchsorted(dom_idx, i + half_width, side="right"))
            if hi - lo < min_local:
                continue
            with np.errstate(invalid="ignore"), _quiet_all_nan():
                local = np.nanmedian(prof[dom_idx[lo:hi]], axis=0)
            deltas[int(rssi[i])].append(prof[i] - local)

    offsets: dict[int, np.ndarray] = {}
    for state, rows in deltas.items():
        if len(rows) < min_state:
            continue
        with np.errstate(invalid="ignore"), _quiet_all_nan():
            off = np.nanmedian(np.vstack(rows), axis=0)
            # Re-centre so the correction is shape-only and cannot move a
            # frame's level. See GainTable.
            off = off - np.nanmedian(off)
        if np.isfinite(off).any():
            offsets[state] = np.nan_to_num(off, nan=0.0, posinf=0.0, neginf=0.0)

    shape, null_k = _shared_direction(prof_segs, deltas, dominant,
                                      half_width=half_width, min_local=min_local)
    if not offsets and shape is None:
        return None
    return GainTable(
        offsets=offsets,
        dominant=dominant,
        n_frames=int(sum(len(v) for v in deltas.values())),
        shape=shape,
        null_k=null_k,
    )


def _shared_direction(
    prof_segs: Sequence[tuple[np.ndarray, np.ndarray]],
    deltas: dict[int, list[np.ndarray]],
    dominant: int,
    *,
    half_width: int,
    min_local: int,
) -> tuple[np.ndarray | None, float]:
    """The one direction the distortions lie along, and the floor to clear.

    The direction is the leading right singular vector of the pooled deltas --
    90.9% of their variance on 20260911_095127, which is what makes a single
    scalar per frame a sufficient description. It is re-centred to zero median
    so a correction along it stays shape-only, exactly as the per-state
    offsets are.

    The floor is that same coefficient measured on DOMINANT-state frames,
    where the true value is zero. It therefore measures the room's own
    movement along this direction over the comparison window, which is the
    thing a real correction has to stand out from.
    """
    rows = [r for v in deltas.values() for r in v]
    if len(rows) < MIN_SHAPE_FRAMES:
        return None, 0.0
    D = np.vstack(rows)
    usable = np.isfinite(D).all(axis=0)
    if usable.sum() < 8:
        return None, 0.0

    M = D[:, usable]
    _, _, vt = np.linalg.svd(M - M.mean(axis=0), full_matrices=False)
    direction = vt[0]
    direction = direction - np.median(direction)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0:
        return None, 0.0
    direction /= norm

    shape = np.zeros(D.shape[1], dtype=float)
    shape[usable] = direction

    # The null: the same coefficient on frames that should read zero.
    nulls: list[float] = []
    for prof, rssi in prof_segs:
        y = _project(prof, shape)
        dom_idx = np.flatnonzero(rssi == dominant)
        if dom_idx.size < min_local + 1:
            continue
        for i in dom_idx:
            lo = int(np.searchsorted(dom_idx, i - half_width, side="left"))
            hi = int(np.searchsorted(dom_idx, i + half_width, side="right"))
            nb = dom_idx[lo:hi]
            nb = nb[nb != i]          # never a frame's own reference
            if nb.size < min_local:
                continue
            nulls.append(abs(float(y[i] - np.median(y[nb]))))
    if not nulls:
        return shape, 0.0
    return shape, float(np.percentile(nulls, NULL_PERCENTILE))


def _project(profile: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Per-frame coefficient along *shape*, NaN bins contributing nothing.

    Done in the scalar domain rather than by taking a per-subcarrier median
    first: the two agree to a correlation of 0.999949 (median disagreement
    0.28 against a typical coefficient of 38.6) and this one is 13x cheaper,
    which is what makes a per-frame correction affordable inside a decode.
    """
    good = shape != 0.0
    if not good.any():
        return np.zeros(profile.shape[0], dtype=float)
    block = np.where(np.isfinite(profile[:, good]), profile[:, good], 0.0)
    return block @ shape[good]


def apply_gain_table(
    amplitude: np.ndarray,
    rssi: np.ndarray,
    table: GainTable | None,
    *,
    contiguous: bool = False,
    half_width: int = DEFAULT_HALF_WIDTH,
    min_local: int = MIN_LOCAL_FRAMES,
) -> np.ndarray:
    """Remove the receiver's gain distortion from *amplitude*.

    *contiguous* declares that consecutive rows really are consecutive frames,
    which is what lets each frame be measured against its own neighbours and
    corrected individually -- the mode that resolves the states RSSI cannot
    separate. Without it only the per-state offsets apply, because a frame
    whose neighbours are seconds apart has nothing local to be judged against.
    That is the same distinction ``backend.ratio`` draws with ``native``.

    Frames in the dominant state, in a state the table never measured, or in
    any state at all when *table* is ``None``, come back untouched -- the same
    array values they were decoded with. So does a frame whose coefficient
    does not clear ``null_k``: below that floor the artifact cannot be told
    from the room, and inventing a correction there would be inventing
    signal. NaN bins stay NaN throughout.
    """
    amp = np.asarray(amplitude)
    if table is None or amp.size == 0:
        return amp
    r = np.asarray(rssi)
    if r.shape[:1] != amp.shape[:1]:
        raise ValueError(
            f"rssi has {r.shape[:1]} entries for {amp.shape[:1]} frames"
        )

    out = amp.astype(np.float32, copy=True)
    off_state = r != table.dominant
    handled = np.zeros(amp.shape[0], dtype=bool)

    if contiguous and table.shape is not None and np.any(off_state):
        shape32 = table.shape.astype(np.float32)
        with _quiet_all_nan(), np.errstate(invalid="ignore"):
            prof = np.asarray(amp, dtype=float) - np.nanmedian(
                np.asarray(amp, dtype=float), axis=1, keepdims=True
            )
        y = _project(prof, table.shape)
        dom_idx = np.flatnonzero(~off_state)
        if dom_idx.size >= min_local:
            for i in np.flatnonzero(off_state):
                lo = int(np.searchsorted(dom_idx, i - half_width, side="left"))
                hi = int(np.searchsorted(dom_idx, i + half_width, side="right"))
                nb = dom_idx[lo:hi]
                if nb.size < min_local:
                    continue
                k = float(y[i] - np.median(y[nb]))
                # Measured, not assumed: below the dominant state's own 99th
                # percentile this frame's deviation is indistinguishable from
                # the room, so the per-frame measurement is inconclusive and
                # the state's pooled offset is the better estimate. Only a
                # frame that clears the floor is taken out of its state's
                # hands -- marking it handled either way would strip the
                # fallback from every frame the floor rejected, which on a
                # capture where almost nothing clears it leaves the artifact
                # whole.
                if abs(k) > table.null_k:
                    out[i] -= np.float32(k) * shape32
                    handled[i] = True

    # Anything the per-frame pass could not reach falls back to its state's
    # offset, which is better than leaving the artifact whole.
    for state, offset in table.offsets.items():
        m = (r == state) & ~handled
        if m.any():
            out[m] -= offset.astype(np.float32)
    return out
