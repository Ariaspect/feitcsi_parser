"""Phase post-processing: subcarrier-axis unwrapping and linear detrending.

The decoded metrics from ``backend.batch`` come out of ``np.angle``, so every
value is wrapped into (-pi, pi]. Two derived views undo parts of that:

* **unwrap** removes the 2*pi sawtooth *within* a frame, along the subcarrier
  axis. It does nothing for frame-to-frame stability — the per-packet offset
  and slope survive it untouched.
* **detrend** additionally fits and subtracts a line over subcarrier index,
  removing the random per-packet phase offset (CFO/PLL) and the linear slope
  from sampling time offset. This is the standard sanitization from the
  SpotFi/PhaseFi lineage, and it is what makes raw phase comparable across
  packets.

Detrending is lossy on purpose: any genuinely linear-in-frequency component
goes with the fit, which means absolute time-of-flight is gone from the
result. That is an acceptable trade for motion sensing and a fatal one for
ranging, so it is exposed as a separate metric rather than applied silently.

Neither transform is appropriate for the CSI ratio's *detrend* step: rx1/rx0
shares an oscillator and clock between the two chains, so the division has
already cancelled the common offset and most of the slope. Only unwrapping is
wired up for the ratio (see ``backend.tiles.DERIVED_METRICS``).

Both functions operate per frame on a full ``(n_frames, num_subcarriers)``
block. They must run before tile column aggregation, never after: aggregation
drops frames, and a phase sequence with holes in it cannot be unwrapped.
"""

from __future__ import annotations

import numpy as np

__all__ = ["unwrap_subcarrier", "detrend_subcarrier", "unwrap_time"]

# A gap longer than this multiple of the median inter-frame interval ends a
# segment. Unwrapping accumulates, so a single misjudged step shifts every
# later frame by 2*pi permanently; segmenting keeps such an error inside one
# stretch instead of letting it run to the end of the capture. On the hourly
# captures (median interval ~100 ms) this puts the threshold near 1 s, which
# only the genuine dropouts exceed.
TIME_GAP_FACTOR = 10.0


def _bridge_holes(rows: np.ndarray, bad: np.ndarray) -> None:
    """Fill non-finite entries in place by linear interpolation along axis 1.

    ``np.unwrap`` accumulates differences, so a single NaN poisons every
    subcarrier after it in the row. Bridging the hole first keeps the damage
    local; the caller restores NaN afterwards, so the bridged values are only
    ever used to carry the unwrap across the gap. A row that is entirely
    non-finite (the ratio metrics on a 1-rx frame) is left alone.
    """
    idx = np.arange(rows.shape[1])
    for r in np.flatnonzero(bad.any(axis=1)):
        good = ~bad[r]
        if not good.any():
            continue
        rows[r] = np.interp(idx, idx[good], rows[r][good])


def unwrap_subcarrier(phase: np.ndarray) -> np.ndarray:
    """Unwrap wrapped phase along the subcarrier axis, per frame.

    *phase* is ``(n_frames, num_subcarriers)`` in radians, wrapped to
    (-pi, pi]. Returns float32 of the same shape with 2*pi jumps removed
    within each row. Non-finite entries are preserved as NaN.
    """
    if phase.size == 0:
        return phase.astype(np.float32, copy=True)

    out = np.asarray(phase, dtype=np.float64).copy()
    bad = ~np.isfinite(out)
    if bad.any():
        _bridge_holes(out, bad)

    out = np.unwrap(out, axis=1)
    out[bad] = np.nan
    return out.astype(np.float32)


def detrend_subcarrier(phase: np.ndarray) -> np.ndarray:
    """Unwrap, then subtract a per-frame least-squares line over subcarriers.

    *phase* is ``(n_frames, num_subcarriers)`` wrapped phase in radians.
    Returns float32 of the same shape, each row unwrapped and with its own
    best-fit line (slope and intercept over subcarrier index) removed.

    The fit uses only finite samples, so a bridged hole does not drag the
    slope. Rows with fewer than two finite samples, or with every finite
    sample at one subcarrier index, have no determined line and come back
    as NaN rather than as an arbitrary one.

    A least-squares fit is used rather than the two-endpoint slope common in
    the literature: the endpoints are single noisy samples and the band edges
    are the noisiest part of the spectrum, so anchoring the whole correction
    to them injects the very tilt this is meant to remove.
    """
    if phase.size == 0:
        return phase.astype(np.float32, copy=True)

    unwrapped = unwrap_subcarrier(phase).astype(np.float64)
    num_sc = unwrapped.shape[1]

    finite = np.isfinite(unwrapped)
    vals = np.where(finite, unwrapped, 0.0)
    k = np.arange(num_sc, dtype=np.float64)
    kk = np.broadcast_to(k, unwrapped.shape) * finite

    n = finite.sum(axis=1).astype(np.float64)
    sum_k = kk.sum(axis=1)
    sum_y = vals.sum(axis=1)
    sum_kk = (kk * kk).sum(axis=1)
    sum_ky = (kk * vals).sum(axis=1)

    denom = n * sum_kk - sum_k * sum_k
    ok = (n >= 2) & (denom != 0)

    slope = np.zeros_like(denom)
    intercept = np.zeros_like(denom)
    np.divide(n * sum_ky - sum_k * sum_y, denom, out=slope, where=ok)
    np.divide(sum_y - slope * sum_k, n, out=intercept, where=ok)

    out = unwrapped - (slope[:, None] * k[None, :] + intercept[:, None])
    out[~ok] = np.nan
    out[~finite] = np.nan
    return out.astype(np.float32)


def unwrap_time(phase: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Unwrap along the *frame* axis, restarting at each capture gap.

    *phase* is ``(n_frames, num_subcarriers)`` wrapped phase in capture order;
    *times* holds the matching frame timestamps in seconds. Returns float32 of
    the same shape.

    This is the view that shows motion: each subcarrier's trace becomes the
    continuous phase accumulated over time rather than a sawtooth folded into
    (-pi, pi]. It is only meaningful on input whose per-packet offset has
    already been cancelled — the CSI ratio, and only after swap/rotation
    correction. On raw phase, where consecutive frames differ by ~1.6 rad at
    random, the result is noise dressed up as a trajectory.

    Two properties follow from unwrapping being cumulative:

    * **Segments restart.** A gap wider than ``TIME_GAP_FACTOR`` times the
      median interval breaks the sequence. Nothing carries across a dropout,
      because the phase may have turned any number of times while nothing was
      being received, and guessing would be fabrication.
    * **Each segment is anchored at its own start**, with its first frame
      subtracted, so the value plotted is phase change *since the segment
      began*. Segments are therefore not comparable with one another in
      absolute terms — only the motion within one means anything. Leaving the
      arbitrary wrapped value of each segment's first frame in place would
      add a meaningless step between segments.
    """
    n = len(phase)
    if n == 0:
        return np.asarray(phase, dtype=np.float32).copy()

    out = np.asarray(phase, dtype=np.float64).copy()
    bad = ~np.isfinite(out)
    times = np.asarray(times, dtype=np.float64)

    dt = np.diff(times)
    positive = dt[dt > 0]
    if positive.size:
        limit = TIME_GAP_FACTOR * float(np.median(positive))
        breaks = np.flatnonzero(dt > limit) + 1
    else:
        breaks = np.empty(0, dtype=np.int64)

    bounds = np.concatenate([[0], breaks, [n]]).astype(np.int64)

    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s < 1:
            continue
        seg = out[s:e]
        seg_bad = bad[s:e]

        if seg_bad.any():
            # Bridge holes down each subcarrier's column so one missing frame
            # cannot poison every frame after it, exactly as the subcarrier
            # unwrap does across the band.
            idx = np.arange(e - s)
            for c in np.flatnonzero(seg_bad.any(axis=0)):
                good = ~seg_bad[:, c]
                if not good.any():
                    continue
                seg[:, c] = np.interp(idx, idx[good], seg[good, c])

        seg = np.unwrap(seg, axis=0)
        seg -= seg[0]
        out[s:e] = seg

    out[bad] = np.nan
    return out.astype(np.float32)
