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
* **Orientation is only ever relative.** Which of the two states is "correct"
  is not observable from one frame, so a majority vote fixes the convention
  per batch: since swaps are the minority, the orientation shared by most
  frames wins. This keeps separately-decoded batches consistent with each
  other without needing a global anchor.
"""

from __future__ import annotations

import numpy as np

__all__ = [
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

# Frames averaged when testing a stretch against the global amplitude profile.
# A single frame's profile correlation is barely better than a coin toss, but
# the state being tested is piecewise-constant over thousands of frames, so
# averaging recovers it decisively.
ANCHOR_WINDOW = 201

# Only runs at least this long are re-oriented by the amplitude anchor. Short
# excursions are the phase passes' business — they resolve isolated frames far
# more sharply than a smoothed correlation can, and letting the anchor touch
# them would blur genuine single-frame swaps across its whole window.
MIN_ANCHOR_RUN = 200

# The anchor is only trustworthy when the profile has a real shape to match.
# Perfectly balanced antennas would leave nothing to correlate against.
MIN_PROFILE_STD = 0.5  # dB


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
    ratio_phase: np.ndarray, ratio_amplitude: np.ndarray | None = None
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

    Returns ``(swap, rot)``, both bool arrays of length ``n_frames``, each
    majority-normalised so the convention follows the bulk of the batch.
    """
    n = len(ratio_phase)
    swap = np.zeros(n, dtype=bool)
    rot = np.zeros(n, dtype=bool)
    if n < 2:
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
    if ratio_amplitude is not None:
        swap = _anchor_to_amplitude(ratio_amplitude, swap)

    # The rotation parity needs its own anchor: the amplitude is blind to a
    # pi rotation, so nothing above can tell a correctly-oriented region from
    # one sitting a pi away from the rest of the capture.
    rot = _anchor_rotation(ratio_phase, swap, rot)

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
    ratio_amplitude: np.ndarray, swap: np.ndarray
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
    """
    n = len(swap)
    if n < MIN_ANCHOR_RUN * 2 or ratio_amplitude is None:
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

        w = min(ANCHOR_WINDOW, max(3, n // 4))
        kernel = np.ones(w) / w
        smoothed = np.convolve(corr, kernel, mode="same")

        flipped = _long_runs(smoothed < 0, MIN_ANCHOR_RUN)
        if not flipped.any():
            break
        out = out ^ flipped

    return out



def _anchor_rotation(
    ratio_phase: np.ndarray, swap: np.ndarray, rot: np.ndarray
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
    """
    n = len(rot)
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

        w = min(ANCHOR_WINDOW, max(3, n // 4))
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
    ratio_phase: np.ndarray, ratio_amplitude: np.ndarray | None = None
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
    """
    swap, rot = detect_states(ratio_phase, ratio_amplitude)
    out = np.array(ratio_phase, dtype=np.float64, copy=True)
    out[swap] = -out[swap]
    out[rot] += np.pi
    # Re-wrap: the rotation pushes values past pi, and a phase metric that
    # leaves the declared [-pi, pi] range would break the fixed colour scale.
    out = np.angle(np.exp(1j * out))
    out[~np.isfinite(ratio_phase)] = np.nan
    return out.astype(np.float32)


def correct_ratio_amplitude(
    ratio_amplitude: np.ndarray, ratio_phase: np.ndarray
) -> np.ndarray:
    """Negate the dB ratio amplitude of swapped frames. Returns float32.

    Detection runs on the same inputs as the corrected phase panel, so the
    two always agree about which frames were flipped. Detecting independently
    per metric would let them disagree on a marginal frame and show a
    correction in one plot but not the other.
    """
    out = np.array(ratio_amplitude, dtype=np.float32, copy=True)
    flip, _ = detect_states(ratio_phase, ratio_amplitude)
    out[flip] = -out[flip]
    return out
