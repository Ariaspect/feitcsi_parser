"""Detection and correction of swapped rx streams in the CSI ratio.

Some frames arrive with the two rx streams exchanged relative to their
neighbours, so ``rx1/rx0`` comes out as ``rx0/rx1``. The signature is exact
and unambiguous: the complex ratio is replaced by its reciprocal, which
negates the phase *and* negates the dB amplitude. Measured on a 6000-frame
window of one transmitter, an affected frame deviates from the mean of its
two neighbours by 1.664 rad where a normal frame deviates by 0.116; negating
it brings that to 0.103, at the normal baseline. The dB amplitude tells the
same story (5.034 -> 0.625).

Nothing in the 272-byte header announces it — all 272 bytes were scanned and
none separates affected frames from normal ones — and no per-frame property
identifies it either (the sign of a frame's median ratio amplitude gets it
right barely more often than chance). Detection therefore has to be
*relative*: a frame is judged against the orientation established by the
frames around it.

Two consequences follow from that, and both are load-bearing:

* **Neighbours must be comparable.** Run this on frames from a single
  transmitter. On an unfiltered capture, consecutive frames come from
  different senders (~18% same-sender in the captures at hand), the
  comparison is meaningless, and the confidence gate below declines to act
  rather than flipping at random.
* **Orientation needs an absolute reference.** Which of the two states is
  "correct" is not observable from one frame, and every quantity the
  detection can derive from a batch is derived *from that batch* — so a
  window can only ever produce an answer self-consistent within itself, and
  which of the two self-consistent answers you get depends on which frames
  are in the window. Panning or zooming a view then inverts whole panels.
  ``Reference`` breaks that: the band profile and the mean phase direction
  are measured once over the whole capture and passed in, so every window
  lands on the same absolute orientation. Build one with ``build_reference``
  and pass it to everything here. Without it the old batch-relative
  behaviour remains, majority vote and all, for callers holding a bare
  array with no capture to refer to.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

__all__ = [
    "Reference",
    "build_reference",
    "with_context",
    "detect_states",
    "detect_swaps",
    "detect_rotations",
    "correct_ratio_phase",
    "correct_ratio_amplitude",
]

# A hypothesis must reach this alignment score before a frame is flipped.
# Score is |mean(exp(i*delta))| over subcarriers: 1.0 is perfect agreement,
# 0.0 is unrelated. Normal neighbours score ~0.98 and a genuine swap scores
# ~0.98 against the negated hypothesis, so the two populations sit far above
# this line while unrelated frames (mixed transmitters) fall far below it.
# Frames that clear neither hypothesis are left exactly as decoded.
CONFIDENCE_MIN = 0.7

# Half-width, in frames, of the neighbourhood used to build the local
# reference each frame is judged against in the refinement pass. The window
# is symmetric, so a channel drifting steadily through it cancels out of the
# mean; that is what lets the reference track a live channel instead of
# fighting it. Too wide and genuine motion smears the reference, too narrow
# and it carries no more authority than a single neighbour.
TEMPLATE_HALF_WIDTH = 8

# Rounds of {refine, merge, re-chain} over the whole batch. The reference
# depends on the corrections and the corrections depend on the reference, so
# it is iterated. One round already reaches zero residual boundaries on every
# capture measured; the extra rounds are kept as headroom for data that needs
# them and cost nothing when the state stops changing, since the loop exits
# on a fixed point. Each round is ~0.5 s on a full 8192-frame tile, so this
# ceiling is also what bounds worst-case latency.
REFINE_PASSES = 2

# Frames averaged when testing a stretch against the amplitude profile.
# A single frame's profile correlation is barely better than a coin toss, but
# the state being tested is piecewise-constant over thousands of frames, so
# averaging recovers it decisively.
ANCHOR_WINDOW = 201

# Only runs at least this long are re-oriented by the *batch-relative*
# amplitude anchor. Short excursions are the phase passes' business — they
# resolve isolated frames far more sharply than a smoothed correlation can,
# and letting the anchor touch them would blur genuine single-frame swaps
# across its whole window. Does not apply once a Reference is supplied: an
# absolute reference only ever corrects a *sign*, so the run-length gate has
# nothing left to protect against.
MIN_ANCHOR_RUN = 200

# The anchor is only trustworthy when the profile has a real shape to match.
# Perfectly balanced antennas would leave nothing to correlate against.
MIN_PROFILE_STD = 0.5  # dB

# Frames averaged when anchoring a *stride-sampled* view: one, i.e. none.
# Smoothing is what makes a near-chance per-frame correlation readable, and it
# works because neighbouring frames share a state. Under a stride they do not
# — the rows are seconds apart and independent — so averaging them mixes in
# evidence about other frames and swamps the frame's own. Measured against the
# native-rate answer, judging each sampled frame alone errs on 5% of frames at
# every stride from 2 to 64; averaging 5 errs on 14-23%, and leaving the
# frames uncorrected errs on 28%.
SAMPLED_ANCHOR_WINDOW = 1

# The reference's phase direction is read off a bimodal distribution — the
# right way up and, a pi away, the rotated frames. The vector sum finds it
# only while the correct mode is the majority. Measured on an hour of frames
# the resultant reaches 0.52 of the total arc length; below this it is not a
# majority worth trusting and no reference is issued.
MIN_PHASE_MARGIN = 0.15

# Frames of context decoded either side of a window before correcting it.
# _chain and _refine judge a frame against its neighbours, so the first and
# last few frames of any window are decided on half the evidence. Correcting
# with a margin and trimming it off keeps a window's interior identical to
# what the whole capture would have given it.
CONTEXT_FRAMES = 128


class Reference(NamedTuple):
    """The capture's own orientation, measured once and reused by every view.

    ``amp_profile`` is the mean-removed median dB ratio amplitude across the
    band. Its shape is set by the antennas rather than the moving channel, so
    it holds over a whole capture, and a swap negates it — which makes the
    sign of a frame's correlation against it an *absolute* verdict rather
    than a comparison with whatever else happens to be on screen.

    ``phase_dir`` is the unit vector the capture's frames point along on
    average. The amplitude cannot see a pi rotation (it leaves dB untouched),
    so the rotation parity needs its own absolute reference; this is it.
    """

    amp_profile: np.ndarray
    phase_dir: complex


def build_reference(
    ratio_amplitude: np.ndarray, ratio_phase: np.ndarray
) -> Reference | None:
    """Measure the capture's orientation from a spread sample of its frames.

    Feed this frames drawn evenly across the whole capture *of a single
    transmitter* — they need not be contiguous, since neither quantity here
    involves a neighbour. Both are majority statistics, which is what makes
    the chicken-and-egg go away: the corruption is the minority (4.1% of
    frames swapped, 14.2% a pi out), so raw frames already point the right
    way in bulk and no prior correction is needed to measure the reference.

    Returns ``None`` when the capture cannot support an absolute verdict — a
    band too flat to correlate against, or a phase direction with no clear
    majority. Callers should fall back to the batch-relative path rather than
    trusting a reference that is really a coin toss.
    """
    amp = np.asarray(ratio_amplitude, dtype=np.float64)
    amp = np.where(np.isfinite(amp), amp, np.nan)
    if amp.size == 0:
        return None
    with np.errstate(invalid="ignore"):
        profile = np.nanmedian(amp, axis=0)
        profile = profile - np.nanmean(profile)
        if not np.isfinite(profile).any() or np.nanstd(profile) < MIN_PROFILE_STD:
            return None
    profile = np.nan_to_num(profile)

    phase = np.asarray(ratio_phase, dtype=np.float64)
    z = np.exp(1j * phase)
    z[~np.isfinite(phase)] = 0.0
    v = z.mean(axis=1)
    resultant = v.sum()
    arc = np.abs(v).sum()
    if arc == 0 or not np.isfinite(resultant) or np.abs(resultant) / arc < MIN_PHASE_MARGIN:
        return None
    return Reference(profile, complex(resultant / np.abs(resultant)))


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average, normalised at the edges.

    ``np.convolve(..., mode="same")`` pads with zeros, which drags the first
    and last w/2 outputs toward zero — right where a tile window's own edge
    sits. Dividing by the number of contributing samples keeps the ends
    readable, and it is the sign of this that decides a flip.
    """
    w = int(max(1, min(w, len(x))))
    if w == 1:
        return np.asarray(x, dtype=np.float64)
    kernel = np.ones(w)
    norm = np.convolve(np.ones(len(x)), kernel, mode="same")
    return np.convolve(np.asarray(x, dtype=np.float64), kernel, mode="same") / norm


def _profile_corr(amplitude: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Per-frame correlation of a dB amplitude block against *profile*.

    ``-inf`` from ``db(0)`` has to be masked out, not merely tolerated: left
    in, it overflows the dot product and poisons the frame's verdict.
    """
    a = np.asarray(amplitude, dtype=np.float64)
    a = np.where(np.isfinite(a), a, np.nan)
    with np.errstate(invalid="ignore"):
        centred = a - np.nanmean(a, axis=1, keepdims=True)
    return np.nan_to_num(centred) @ profile


def _fit(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Fit ``a ≈ b + constant`` on the unit circle.

    Returns ``(quality, offset)``: quality is 1.0 when a single constant
    explains the whole band and 0.0 when the two are unrelated; offset is
    that constant, in (-pi, pi].

    Quality is deliberately blind to the offset — that is what lets the same
    score rank the swapped and unswapped hypotheses fairly. The offset is
    then read separately to decide whether the frame is also pi-rotated.
    Scoring with the offset folded in would rate a perfectly-explained
    pi-rotated frame as "unrelated" and hide the rotation entirely.
    """
    d = a - b
    finite = np.isfinite(d)
    if not finite.any():
        return 0.0, 0.0
    m = np.mean(np.exp(1j * d[finite]))
    return float(np.abs(m)), float(np.angle(m))


def _alignment(a: np.ndarray, b: np.ndarray) -> float:
    """Backwards-compatible alias for the quality half of ``_fit``."""
    return _fit(a, b)[0]


def detect_states(
    ratio_phase: np.ndarray,
    ratio_amplitude: np.ndarray | None = None,
    *,
    reference: Reference | None = None,
    native: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover each frame's ``(swapped, rotated)`` state relative to frame 0.

    The observed ratio can differ from its neighbours' convention in two
    independent ways, and they are *different corruptions* with different
    signatures:

    * **swapped** — the rx streams were exchanged, so the ratio is its
      reciprocal: phase negated, dB amplitude negated.
    * **rotated** — the ratio is multiplied by -1: phase shifted by pi, dB
      amplitude untouched.

    Together they form four states (``r``, ``-r``, ``1/r``, ``-1/r``). Each
    transition is fitted both ways round — ``phi_i`` against ``phi_prev`` and
    against ``-phi_prev`` — and the better-fitting orientation wins. The
    fitted *offset* of the winner then says whether a pi rotation came with
    it. Both decisions accumulate as parities, so runs need no special
    handling.

    Passing *ratio_amplitude* enables a final anchoring pass that fixes
    block-scale swap errors the phase alone cannot see. Strongly recommended:
    without it, two mistaken toggles can leave a thousand-second region
    inverted and internally consistent, which is invisible to every
    phase-based check. See ``_anchor_to_amplitude``.

    *reference* is what makes the answer a property of the capture instead of
    a property of the window. With one, both anchors judge against it and the
    result for a given frame is the same whichever view asked; without one,
    the anchors fall back to statistics of this batch and a closing majority
    vote fixes the convention, which is stable only as long as the batch is.

    *native* declares that consecutive rows really are consecutive frames.
    ``_chain`` and ``_refine`` compare a frame to its neighbours, so on a
    stride-sampled view — where "neighbours" are seconds apart and the fit
    quality collapses — they are skipped and the anchors, which judge each
    frame on its own against *reference*, carry the whole decision. That
    costs the isolated single-frame swaps, which at those zooms occupy a
    fraction of one display column and cannot be seen anyway. Passing
    ``native=False`` without a *reference* leaves the frames untouched, there
    being nothing left that could decide anything.

    Returns ``(swap, rot)``, both bool arrays of length ``n_frames``.
    """
    n = len(ratio_phase)
    swap = np.zeros(n, dtype=bool)
    rot = np.zeros(n, dtype=bool)
    if n < 2:
        return swap, rot
    if not native and reference is None:
        return swap, rot

    # The two passes fail in complementary ways, so they are alternated.
    #
    # The chain finds *boundaries*: it walks adjacent frames and toggles on a
    # step, which is the only way to notice that a whole region sits inverted.
    # But it propagates — one missed toggle inverts everything downstream.
    #
    # The refinement judges each frame against a local consensus, so its
    # mistakes stay local. But a large inverted region is internally
    # consistent — every frame in it agrees with its inverted neighbours — so
    # the refinement alone cannot see one.
    #
    # Chaining over the *already corrected* phase re-detects whatever survived
    # the last round, and states compose by XOR (a second swap undoes the
    # first; a second rotation likewise). Iterating converges quickly and is
    # idempotent once no residual boundary remains.
    if native:
        swap, rot = _chain(ratio_phase)
        for _ in range(REFINE_PASSES):
            prev_swap, prev_rot = swap, rot
            swap, rot = _refine(ratio_phase, swap, rot)
            swap, rot = _merge_segments(ratio_phase, swap, rot)
            d_swap, d_rot = _chain(_apply(ratio_phase, swap, rot))
            swap = swap ^ d_swap
            rot = rot ^ d_rot
            if np.array_equal(swap, prev_swap) and np.array_equal(rot, prev_rot):
                break

    # Last: settle which side of each boundary is actually the right way up.
    # Everything above only compares frames to other frames, so it can place a
    # boundary perfectly and still leave the whole region between two of them
    # inverted.
    smooth = ANCHOR_WINDOW if native else SAMPLED_ANCHOR_WINDOW
    swap = _anchor_to_amplitude(
        ratio_amplitude, swap, reference=reference, smooth=smooth
    )

    # The rotation parity needs its own anchor: the amplitude is blind to a
    # pi rotation, so nothing above can tell a correctly-oriented region from
    # one sitting a pi away from the rest of the capture.
    rot = _anchor_rotation(
        ratio_phase, swap, rot, reference=reference, smooth=smooth
    )

    if reference is None:
        # No absolute verdict available, so the convention is whatever most
        # of this batch agrees on. Note this overrides the anchors above —
        # it is a last resort, and it is exactly what a Reference replaces.
        if swap.mean() > 0.5:
            swap = ~swap
        if rot.mean() > 0.5:
            rot = ~rot

    return swap, rot


def _apply(ratio_phase: np.ndarray, swap: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Return *ratio_phase* with the given states undone, re-wrapped."""
    out = np.array(ratio_phase, dtype=np.float64, copy=True)
    out[swap] = -out[swap]
    out[rot] += np.pi
    out = np.angle(np.exp(1j * out))
    out[~np.isfinite(ratio_phase)] = np.nan
    return out


def _chain(ratio_phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parity chain over adjacent frames. Finds region boundaries."""
    n = len(ratio_phase)
    swap = np.zeros(n, dtype=bool)
    rot = np.zeros(n, dtype=bool)
    if n < 2:
        return swap, rot

    # Vectorised: every transition is scored at once and the running state is
    # a parity cumsum. A frame-by-frame Python loop here cost ~2 s on a full
    # 8192-frame tile, which is felt on every pan.
    z = np.exp(1j * np.asarray(ratio_phase, dtype=np.float64))
    valid = np.isfinite(ratio_phase)
    z = np.where(valid, z, 0.0)
    row_ok = valid.any(axis=1)

    # Index of the most recent usable row strictly before each row, so an
    # all-NaN frame is stepped over rather than compared against.
    seen = np.where(row_ok, np.arange(n), -1)
    prev_idx = np.empty(n, dtype=np.int64)
    prev_idx[0] = -1
    prev_idx[1:] = np.maximum.accumulate(seen)[:-1]
    have_prev = prev_idx >= 0
    base = z[np.clip(prev_idx, 0, n - 1)]

    shared = (valid & valid[np.clip(prev_idx, 0, n - 1)]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_keep = (z * np.conj(base)).sum(axis=1) / shared
        m_swap = (np.conj(z) * np.conj(base)).sum(axis=1) / shared

    q_keep = np.abs(m_keep)
    q_swap = np.abs(m_swap)
    ok = have_prev & row_ok & np.isfinite(q_keep) & np.isfinite(q_swap)

    use_swap = ok & (q_swap > q_keep) & (q_swap >= CONFIDENCE_MIN)
    use_keep = ok & ~use_swap & (q_keep >= CONFIDENCE_MIN)
    # Neither credible: no toggle, so both parities carry forward unchanged.

    t_swap = use_swap
    t_rot = (use_swap & (np.abs(np.angle(m_swap)) > np.pi / 2)) | (
        use_keep & (np.abs(np.angle(m_keep)) > np.pi / 2)
    )

    swap = np.cumsum(t_swap) % 2 == 1
    rot = np.cumsum(t_rot) % 2 == 1
    return swap, rot



def _anchor_to_amplitude(
    ratio_amplitude: np.ndarray | None,
    swap: np.ndarray,
    reference: Reference | None = None,
    smooth: int = ANCHOR_WINDOW,
) -> np.ndarray:
    """Re-orient block-scale swap errors against the global amplitude profile.

    The phase passes decide orientation only by comparing frames to other
    frames. That is enough to place a boundary, but it cannot tell which
    *side* of a boundary is the right way up — a whole region flipped between
    two mistaken toggles is internally consistent, so nothing in the phase
    ever objects. The result is a correction that removes the real isolated
    swaps and then inverts a thousand-second block on top.

    The dB ratio amplitude breaks that tie, because a swap negates it too and
    its shape across the band is fixed by the antennas rather than by the
    moving channel. Measured on an hour of frames, every 2000-frame chunk
    correlates +0.955 to +0.999 with the file's median profile: the bulk
    orientation is never actually in doubt, so a stretch that anti-correlates
    is wrong rather than interesting.

    Only long runs are touched. A single frame's correlation is near chance,
    and the smoothing needed to make it readable spans far more frames than a
    genuine isolated swap covers — so short excursions are left to the phase
    passes, which resolve them sharply.

    With a *reference* none of that hedging is needed. The profile no longer
    comes from the frames being judged, so there is no risk of confirming a
    window's own inversion, no need to iterate it to a fixed point, and no
    need for the run-length gate: an absolute reference can only ever flip a
    sign that is wrong. It also drops the minimum-length bail, which is what
    left every view under 400 frames with no absolute orientation at all.
    """
    if ratio_amplitude is None:
        return swap
    n = len(swap)

    if reference is not None:
        oriented = np.asarray(ratio_amplitude, dtype=np.float64).copy()
        oriented[swap] = -oriented[swap]
        corr = _profile_corr(oriented, reference.amp_profile)
        return swap ^ (_moving_average(corr, smooth) < 0)

    if n < MIN_ANCHOR_RUN * 2:
        return swap

    amp = np.where(np.isfinite(ratio_amplitude), ratio_amplitude, np.nan).astype(np.float64)
    out = swap.copy()

    for _ in range(2):
        oriented = amp.copy()
        oriented[out] = -oriented[out]

        with np.errstate(invalid="ignore"):
            profile = np.nanmedian(oriented, axis=0)
        profile = profile - np.nanmean(profile)
        if not np.isfinite(profile).any() or np.nanstd(profile) < MIN_PROFILE_STD:
            return swap  # nothing to correlate against; leave the decision alone

        profile = np.nan_to_num(profile)
        centred = oriented - np.nanmean(oriented, axis=1, keepdims=True)
        corr = np.nansum(np.nan_to_num(centred) * profile, axis=1)

        w = min(smooth, max(3, n // 4))
        kernel = np.ones(w) / w
        smoothed = np.convolve(corr, kernel, mode="same")

        flipped = _long_runs(smoothed < 0, MIN_ANCHOR_RUN)
        if not flipped.any():
            break
        out = out ^ flipped

    return out



def _anchor_rotation(
    ratio_phase: np.ndarray,
    swap: np.ndarray,
    rot: np.ndarray,
    reference: Reference | None = None,
    smooth: int = ANCHOR_WINDOW,
) -> np.ndarray:
    """Settle the rotation parity against the capture's own mean direction.

    Rotations are not rare: on an hourly capture ~24% of transitions carry a
    pi offset, so the parity toggles thousands of times. Every toggle is a
    chance to be wrong by one, and being wrong by one flips every frame after
    it until the next mistake — the same propagation failure the swap has,
    except the amplitude cannot help here. A pi rotation multiplies the ratio
    by -1, which leaves the dB amplitude exactly where it was.

    What does anchor it is the phase's own mean direction. Each frame's
    circular mean over subcarriers points somewhere, and across a whole
    capture that direction is stable — it is set by the fixed offset between
    the two antennas, not by the moving channel. Measured over an hour it
    holds at +1.3 rad from end to end, while a wrongly-rotated stretch sits
    at -1.8: a full pi away, and trivially separable by sign.

    As with the amplitude anchor, only long runs are re-oriented. Genuine
    single-frame events belong to the phase passes, which see them sharply.

    And as there, a *reference* removes the hedging: the direction is the
    capture's, measured once, so a single pass against it settles the parity
    for any window at any length.
    """
    n = len(rot)

    if reference is not None:
        corrected = _apply(ratio_phase, swap, rot)
        z = np.exp(1j * corrected)
        z[~np.isfinite(corrected)] = 0.0
        agree = np.real(z.mean(axis=1) * np.conj(reference.phase_dir))
        return rot ^ (_moving_average(agree, smooth) < 0)

    if n < MIN_ANCHOR_RUN * 2:
        return rot

    out = rot.copy()
    for _ in range(2):
        corrected = _apply(ratio_phase, swap, out)
        z = np.exp(1j * corrected)
        z[~np.isfinite(corrected)] = 0.0
        # One unit vector per frame: where that frame's band points overall.
        v = z.mean(axis=1)

        ref = v.sum()
        if not np.isfinite(ref) or np.abs(ref) == 0:
            return rot
        ref /= np.abs(ref)

        # Positive when the frame agrees with the capture's mean direction,
        # negative when it sits a pi away from it.
        agree = np.real(v * np.conj(ref))

        w = min(smooth, max(3, n // 4))
        smoothed = np.convolve(agree, np.ones(w) / w, mode="same")

        flipped = _long_runs(smoothed < 0, MIN_ANCHOR_RUN)
        if not flipped.any():
            break
        out = out ^ flipped

    return out


def _long_runs(mask: np.ndarray, minimum: int) -> np.ndarray:
    """Keep only runs of True at least *minimum* long."""
    out = np.zeros_like(mask)
    if not mask.any():
        return out
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    for s, e in zip(starts, ends):
        if e - s >= minimum:
            out[s:e] = True
    return out


def _merge_segments(
    ratio_phase: np.ndarray, swap: np.ndarray, rot: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reconcile whole segments across the boundaries between them.

    This is the pass that catches what the other two structurally cannot.
    The chain compares single adjacent frames, so where one transition is
    unreadable — a long gap, a sampled view where neighbours sit 400 ms
    apart — it declines, and a missed toggle leaves everything downstream
    inverted. The refinement then cannot repair that, because a whole
    inverted region agrees with itself, and its symmetric window straddles
    the boundary and goes incoherent exactly where it is needed most.

    Here each candidate split point is judged by comparing the mean of the
    ``TEMPLATE_HALF_WIDTH`` corrected frames *before* it against the mean of
    those *after* it. Averaging many frames on each side lifts the signal
    far above what any single pair carries, so a boundary that no adjacent
    comparison could resolve becomes obvious. Detected boundaries toggle
    everything downstream, which composes by XOR just like the chain.

    Non-maximum suppression keeps one detection per boundary: a genuine step
    also shows up, weaker, at the offsets either side of it.
    """
    n = len(ratio_phase)
    w = min(TEMPLATE_HALF_WIDTH, n // 2)
    if n < 4 or w < 2:
        return swap, rot

    z = np.exp(1j * np.asarray(ratio_phase, dtype=np.float64))
    z = np.where(np.isfinite(ratio_phase), z, 0.0)

    zc = np.where(swap[:, None], np.conj(z), z)
    zc = np.where(rot[:, None], -zc, zc)

    csum = np.cumsum(np.vstack([np.zeros((1, z.shape[1])), zc]), axis=0)
    idx = np.arange(n)
    lo = np.clip(idx - w, 0, n)
    hi = np.clip(idx + w, 0, n)
    before = csum[idx] - csum[lo]          # frames [i-w, i)
    after = csum[hi] - csum[idx]           # frames [i, i+w)

    norm = (np.abs(before) * np.abs(after)).sum(axis=1)
    m_keep = (after * np.conj(before)).sum(axis=1)
    m_swap = (np.conj(after) * np.conj(before)).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        q_keep = np.abs(m_keep) / norm
        q_swap = np.abs(m_swap) / norm

    take_swap = q_swap > q_keep
    best = np.where(take_swap, m_swap, m_keep)
    quality = np.where(take_swap, q_swap, q_keep)

    d_swap = take_swap & np.isfinite(quality) & (quality >= CONFIDENCE_MIN)
    d_rot = (
        (np.abs(np.angle(best)) > np.pi / 2)
        & np.isfinite(quality)
        & (quality >= CONFIDENCE_MIN)
    )
    candidate = d_swap | d_rot
    # Only interior split points have a full segment on both sides.
    candidate[:w] = False
    candidate[n - w:] = False

    if not candidate.any():
        return swap, rot

    # Non-maximum suppression: strongest evidence wins within +/- w.
    evidence = np.where(candidate, quality, -np.inf)
    order = np.argsort(evidence)[::-1]
    taken = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    for i in order:
        if not np.isfinite(evidence[i]):
            break
        if blocked[i]:
            continue
        taken[i] = True
        blocked[max(0, i - w) : min(n, i + w + 1)] = True

    swap = swap ^ (np.cumsum(taken & d_swap) % 2 == 1)
    rot = rot ^ (np.cumsum(taken & d_rot) % 2 == 1)
    return swap, rot


def _refine(
    ratio_phase: np.ndarray, swap: np.ndarray, rot: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Re-decide every frame against a local reference instead of its neighbour.

    The chaining pass above is a parity chain, so a single missed toggle
    inverts everything downstream until another miss happens to undo it. That
    is exactly the failure mode seen on stride-sampled views: sampled frames
    sit further apart, alignment confidence drops below the gate, a rotation
    goes unnoticed, and a whole region stays inverted.

    Here each frame is instead judged against the circular mean of its
    corrected neighbours — a consensus that one bad frame cannot move. A
    decision that comes out wrong stays local, because nothing downstream
    depends on it.

    Scoring uses the *real part* of the correlation with the reference, not
    its magnitude. Magnitude is offset-blind (see ``_fit``) and so cannot see
    a pi rotation at all; the real part is maximal at zero offset and most
    negative at pi, which is precisely the discrimination needed.
    """
    n = len(ratio_phase)
    if n < 3:
        return swap, rot

    z = np.exp(1j * np.asarray(ratio_phase, dtype=np.float64))
    valid = np.isfinite(ratio_phase)
    z = np.where(valid, z, 0.0)

    w = min(TEMPLATE_HALF_WIDTH, (n - 1) // 2)
    if w < 1:
        return swap, rot

    for _ in range(REFINE_PASSES):
        # Apply the current hypothesis: a swap conjugates, a rotation negates.
        zc = np.where(swap[:, None], np.conj(z), z)
        zc = np.where(rot[:, None], -zc, zc)

        # Local reference: the window sum minus the frame's own contribution,
        # so a frame is never judged against a reference containing itself.
        csum = np.cumsum(np.vstack([np.zeros((1, z.shape[1])), zc]), axis=0)
        lo = np.clip(np.arange(n) - w, 0, n)
        hi = np.clip(np.arange(n) + w + 1, 0, n)
        ref = csum[hi] - csum[lo] - zc

        weight = np.abs(ref).sum(axis=1)
        m_keep = (z * np.conj(ref)).sum(axis=1)
        m_swap = (np.conj(z) * np.conj(ref)).sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            q_keep = np.abs(m_keep) / weight
            q_swap = np.abs(m_swap) / weight

        take_swap = q_swap > q_keep
        best = np.where(take_swap, m_swap, m_keep)

        # Gate on whether the *reference* is trustworthy, not on how well this
        # particular frame matches it. A noisy frame still belongs to whichever
        # state its coherent neighbours are in, and leaving it alone looks
        # identical to guessing wrong — so declining helps nobody. What must
        # be guarded against is a meaningless reference (mixed transmitters),
        # where the neighbours do not agree with each other either.
        #
        # ``ref`` sums unit vectors, so |ref| approaches the contributor count
        # when the neighbourhood agrees and collapses toward zero when it does
        # not.
        usable = valid.any(axis=1).astype(np.float64)
        counts = np.convolve(usable, np.ones(2 * w + 1), mode="same") - usable
        with np.errstate(invalid="ignore", divide="ignore"):
            coherence = weight / np.maximum(counts, 1.0) / z.shape[1]

        decided = (
            np.isfinite(coherence)
            & (coherence >= CONFIDENCE_MIN)
            & valid.any(axis=1)
        )
        new_swap = np.where(decided, take_swap, swap)
        new_rot = np.where(decided, np.abs(np.angle(best)) > np.pi / 2, rot)

        if np.array_equal(new_swap, swap) and np.array_equal(new_rot, rot):
            swap, rot = new_swap, new_rot
            break
        swap, rot = new_swap, new_rot

    return swap, rot


def detect_rotations(ratio_phase: np.ndarray) -> np.ndarray:
    """Frames whose ratio is pi-rotated relative to the batch's convention."""
    return detect_states(ratio_phase)[1]


def detect_swaps(ratio_phase: np.ndarray) -> np.ndarray:
    """Flag frames whose CSI ratio is inverted relative to the run around them.

    *ratio_phase* is ``(n_frames, num_subcarriers)`` wrapped ratio phase, in
    capture order, ideally already filtered to one transmitter. Returns a
    bool array of length ``n_frames``.

    Each *transition* between adjacent frames is classified as same-or-
    flipped, and a frame's flag is the running parity of those decisions.
    Runs of consecutive swapped frames therefore need no special handling:
    the parity simply stays toggled until something toggles it back.

    Because parity accumulates, roughly half the frames in a long batch end
    up flagged even though swaps are individually rare — the flag means
    "opposite orientation to frame 0", not "anomalous".

    A transition too noisy to call (a dropout, or neighbours from different
    senders) carries the current orientation forward rather than guessing.
    The result is majority-normalised so the convention is set by the bulk
    of the batch rather than by whichever orientation frame 0 happened to
    have.
    """
    return detect_states(ratio_phase)[0]


def correct_ratio_phase(
    ratio_phase: np.ndarray,
    ratio_amplitude: np.ndarray | None = None,
    *,
    reference: Reference | None = None,
    native: bool = True,
) -> np.ndarray:
    """Undo both corruptions of the ratio phase. Returns float32.

    Swapped frames are negated and pi-rotated frames are shifted back, then
    the result is re-wrapped to (-pi, pi].

    The rotation term needs no sign of its own: recovering a swapped frame
    means negating it, which would turn ``+pi`` into ``-pi`` — the same angle.
    So ``sign * observed + pi * rot`` is correct for either orientation.

    Pass *ratio_amplitude* whenever it is available. The phase carries no
    information about which side of a boundary is the right way up, so
    without it a mistaken pair of toggles can invert a long region and leave
    it looking perfectly self-consistent.

    Pass *reference* too, or the answer is only as stable as the batch — see
    ``detect_states``.
    """
    swap, rot = detect_states(
        ratio_phase, ratio_amplitude, reference=reference, native=native
    )
    out = np.array(ratio_phase, dtype=np.float64, copy=True)
    out[swap] = -out[swap]
    out[rot] += np.pi
    # Re-wrap: the rotation pushes values past pi, and a phase metric that
    # leaves the declared [-pi, pi] range would break the fixed colour scale.
    out = np.angle(np.exp(1j * out))
    out[~np.isfinite(ratio_phase)] = np.nan
    return out.astype(np.float32)


def correct_ratio_amplitude(
    ratio_amplitude: np.ndarray,
    ratio_phase: np.ndarray,
    *,
    reference: Reference | None = None,
    native: bool = True,
) -> np.ndarray:
    """Negate the dB ratio amplitude of swapped frames. Returns float32.

    Detection runs on the same inputs as the corrected phase panel, so the
    two always agree about which frames were flipped. Detecting independently
    per metric would let them disagree on a marginal frame and show a
    correction in one plot but not the other.
    """
    out = np.array(ratio_amplitude, dtype=np.float32, copy=True)
    flip, _ = detect_states(
        ratio_phase, ratio_amplitude, reference=reference, native=native
    )
    out[flip] = -out[flip]
    return out


def with_context(
    correct: "Callable[..., np.ndarray]",
    blocks: "Sequence[np.ndarray]",
    *,
    lead: int,
    length: int,
    **kwargs,
) -> np.ndarray:
    """Correct a window that carries context, and return only its interior.

    *blocks* are the metric arrays for ``lead`` frames of leading context,
    then the ``length`` frames actually wanted, then whatever trailing
    context was available. Correcting the whole thing and slicing the middle
    out gives the interior frames the same neighbours they would have had in
    a full-capture pass, which is what keeps a tile's edge columns from being
    decided on half the evidence.
    """
    corrected = correct(*blocks, **kwargs)
    return corrected[lead : lead + length]
