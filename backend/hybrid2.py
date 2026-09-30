"""Hybrid 2: the bounded ratio-complex frame step for motion, FarSense for breath.

Same shape as ``backend.hybrid`` -- motion opens presence, breathing keeps it
open, a hold carries it across the gaps -- and one substitution: the motion
channel is the **complex frame step** of ``backend.framediff`` rather than
``presence.fractional_motion``.

**What actually changes, and what does not.** The two motion quantities are the
same formula: ``fractional_motion`` divides by the midpoint magnitude and the
frame step divides by the sum, so the frame step is exactly half, verified per
subcarrier at 0.500000 over 14 captures. So this is not a new detector built on
a new signal. What it changes is everything around the number:

* **The fold is a median, not a mean** -- robust to a handful of subcarriers
  sitting on a fading null.
* **The frame set is uniform** -- one transmitter, one MIMO mode, one
  bandwidth (``framediff.STRICT_MIMO``). Hybrid 1 differences whatever the
  caller filtered to, and 0.6-0.9 % of its pairs cross a bandwidth and read
  9-18x the same-width level.
* **Native frame times, never resampled.** A pair spanning a dropout is
  blanked rather than interpolated through.
* **The phase is in it.** ``fractional_motion`` is already complex, so this
  matters less than it sounds; what the complex step adds is the readback --
  a step is an angle, ``framediff.unit_to_radians``.

Because the metric is half of hybrid 1's, every *relative* default carries over
untouched and only the absolute floor halves: ``MOTION_ABS`` 0.10 becomes 0.05.
That is arithmetic, not tuning.

**The breathing channel is hybrid 1's, unchanged** -- ``hybrid.evidence_series``
runs the FarSense sweep and this module keeps its output verbatim, so a
difference between the two tabs is never the breathing half.

**What this is for.** The frame step detects motion, big or small, and nothing
else: measured within a capture against its own empty seconds, walking clears
it 4.56x (98 % of its seconds past the empty P90) while a still occupant reads
0.94x and 13 % -- correctly, since a still body is not moving. The breathing
channel is what covers that case, which is the whole reason this is a hybrid
and not a threshold.

**What it cannot do**, and the reason is not in this file: the motion channel's
empty-room floor spans ~10x between link states (``docs/frame_step.md``), and
on the noisiest links an *empty* room reads what a sitting person reads on a
clean one -- 0.95-1.13x at every lag from 24 ms to 4 s. So the threshold is
relative to the range's own floor, exactly as hybrid 1's is, with the same
known failure: a range occupied throughout has no quiet stretch and its own
percentile *is* the occupant. Measured on the six 09-21 sitting captures, the
own-floor rule recalls 0 % and a session floor recalls 100 %. Nothing here
fixes that; the floor is reported so a reader can see which case they are in.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from backend import framediff, hybrid

# The motion signal. Fixed rather than a parameter: the point of this tab is
# that one channel, and the ratio is the only form of it whose floor is not
# also the receiver's gain control (20.5x cross-capture spread on raw
# amplitude against 10.3x here).
SIGNAL = "ratio_complex"

# Half of hybrid.MOTION_ABS, because the metric is half of hybrid's. Not a
# retune: the same physical step, written on an axis bounded by 1 instead of 2.
MOTION_ABS = hybrid.MOTION_ABS / 2.0

# Everything else is hybrid 1's, by reference rather than by copy, so the two
# tabs cannot drift apart on a shared default.
MOTION_REL = hybrid.MOTION_REL
HOLD_SECONDS = hybrid.HOLD_SECONDS
BURST_SECONDS = hybrid.BURST_SECONDS
FLOOR_PERCENTILE = hybrid.FLOOR_PERCENTILE
BREATH_MIN_PEAK = hybrid.BREATH_MIN_PEAK
BREATH_PERSIST_SECONDS = hybrid.BREATH_PERSIST_SECONDS

# The range verdict: one present/empty call for a whole range, the rule the
# user set for the 1-minute captures (docs/hybrid2.md, 2026-09-30). Present
# when the 90th percentile of the per-second step at a 2 s lag clears
# RANGE_MOTION_P90, else when the FarSense peak holds RANGE_BREATH_PEAK for
# RANGE_BREATH_RUN consecutive windows, else empty. Absolute thresholds -- no
# floor -- chosen on 235 one-minute captures (200 empty, 35 occupied): 98.3 %
# in-sample, 95.3 % / 97.2 % under 5- and 2-fold blocked cross-validation;
# the same triple came out of 3 of 5 folds.
RANGE_LAG_SECONDS = 2.0
RANGE_MOTION_P90 = 0.035
RANGE_BREATH_PEAK = 0.25
RANGE_BREATH_RUN = 5

_CACHE_SIZE = 4
_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
_cache_lock = Lock()


def reset_cache() -> None:
    with _cache_lock:
        _cache.clear()


def motion_per_second(
    path,
    t0: float,
    t1: float,
    seconds: np.ndarray,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
    gate_gain: bool = False,
    lag_seconds: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The frame step reduced to one level per second, on *seconds*.

    *lag_seconds* differences frames that far apart instead of adjacent ones.
    0 keeps the adjacent pair. Measured within a capture against its own empty
    seconds, a longer lag is what makes *small* motion visible -- recall over
    six small-movement captures runs 19 % at one frame and 88 % at 2 s, while a
    still occupant stays near chance (13 % to 21 %). It does nothing for the
    cross-capture floor, which saturates at one frame already.
    """
    steps = framediff.capture_steps(
        path, t0, t1, mimo=mimo, source_mac=source_mac,
        interpolate=interpolate, signal=SIGNAL, gate_gain=gate_gain,
        lag_seconds=lag_seconds,
    )
    level = hybrid.per_second(
        np.asarray(steps["magnitude"], dtype=float),
        np.asarray(steps["time_s"], dtype=float),
        seconds,
    )
    return level, steps


def capture_evidence(
    path,
    t0: float,
    t1: float,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
    gate_gain: bool = False,
    lag_seconds: float = 0.0,
    **evidence: Any,
) -> dict[str, Any]:
    """Hybrid 1's evidence with the motion channel replaced, cached.

    The frame step goes into ``motion_ratio`` -- the slot ``hybrid.verdict``
    thresholds -- so the verdict stage is reused verbatim rather than forked.
    Hybrid 1's own ``|dr|/|r|`` is kept alongside as ``motion_reference`` so a
    reader can see both channels on one axis without opening the other tab.
    """
    path = Path(path)
    st = path.stat()
    key = (
        str(path.resolve()), st.st_size, st.st_mtime_ns, float(t0), float(t1),
        mimo, source_mac, bool(interpolate), bool(gate_gain), float(lag_seconds),
        tuple(sorted((k, tuple(v) if isinstance(v, (tuple, list)) else v)
                     for k, v in evidence.items())),
    )
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit

    ev = hybrid.capture_evidence(
        path, t0, t1, mimo=mimo, source_mac=source_mac,
        interpolate=interpolate, **evidence,
    )
    # hybrid.capture_evidence returns seconds on the capture's clock, centred
    # at +0.5; per_second wants the left edges.
    seconds = np.asarray(ev["time_s"], dtype=float) - 0.5
    level, steps = motion_per_second(
        path, t0, t1, seconds, mimo=mimo, source_mac=source_mac,
        interpolate=interpolate, gate_gain=gate_gain, lag_seconds=lag_seconds,
    )
    level[np.asarray(ev["unknown"], dtype=bool)] = np.nan

    out = dict(ev)
    out["motion_reference"] = np.asarray(ev["motion_ratio"], dtype=float)
    out["motion_ratio"] = level
    out["motion_amp"] = np.full(level.shape, np.nan)   # hybrid 1's amplitude
    out["signal"] = SIGNAL                             # channel is off here
    out["selection_note"] = steps["selection_note"]
    out["frames_used"] = int(steps["frames_used"])
    out["frames_dropped"] = int(steps["frames_dropped"])
    out["n_gain_crossed"] = int(steps["n_gain_crossed"])
    out["gain_gated"] = bool(steps["gain_gated"])
    out["lag_seconds"] = float(lag_seconds)
    out["lag_frames"] = int(steps.get("lag_frames", 1))

    with _cache_lock:
        _cache[key] = out
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return out


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive true cells."""
    best = cur = 0
    for x in np.asarray(mask, dtype=bool):
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def range_verdict(
    level_lag2: np.ndarray,
    breath_peak: np.ndarray,
    *,
    motion_p90: float = RANGE_MOTION_P90,
    breath_peak_min: float = RANGE_BREATH_PEAK,
    breath_run: int = RANGE_BREATH_RUN,
) -> dict[str, Any]:
    """One call for the whole range from fixed thresholds (see RANGE_*)."""
    lv = np.asarray(level_lag2, dtype=float)
    lv = lv[np.isfinite(lv)]
    p90 = float(np.percentile(lv, 90)) if lv.size else float("nan")
    pk = np.asarray(breath_peak, dtype=float)
    run = longest_run(np.isfinite(pk) & (pk >= breath_peak_min))
    by_motion = bool(np.isfinite(p90) and p90 > motion_p90)
    by_breath = bool(run >= breath_run)
    return {
        "present": by_motion or by_breath,
        "by": "motion" if by_motion else ("breathing" if by_breath else None),
        "motion_p90": p90,
        "breath_run": int(run),
        "seconds": int(lv.size),
        "thresholds": {
            "lag_seconds": RANGE_LAG_SECONDS, "motion_p90": float(motion_p90),
            "breath_peak": float(breath_peak_min), "breath_run": int(breath_run),
        },
    }


def compute_hybrid2(
    path,
    t0: float,
    t1: float,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
    gate_gain: bool = False,
    lag_seconds: float = 0.0,
    **params: Any,
) -> dict[str, Any]:
    """Decode a range and run the composite detector; times on the capture's clock."""
    evidence, decision = hybrid.split_params(params)
    decision.setdefault("motion_abs", MOTION_ABS)
    # This tab has one motion channel; hybrid 1's amplitude one is not in play.
    decision["use_amplitude"] = False

    ev = capture_evidence(
        path, t0, t1, mimo=mimo, source_mac=source_mac, interpolate=interpolate,
        gate_gain=gate_gain, lag_seconds=lag_seconds, **evidence,
    )
    origin = float(ev["time_s"][0]) - 0.5
    local = dict(ev)
    local["time_s"] = np.asarray(ev["time_s"], dtype=float) - origin
    out = hybrid.verdict(local, **decision)
    out["time_s"] = np.asarray(out["time_s"], dtype=float) + origin
    for key in ("motion_reference", "signal", "selection_note", "frames_used",
                "frames_dropped", "n_gain_crossed", "gain_gated",
                "lag_seconds", "lag_frames", "breath_note"):
        if key in ev:
            out[key] = ev[key]

    # The range verdict always reads the step at RANGE_LAG_SECONDS, whatever
    # lag the per-second series above was drawn with.
    seconds = np.asarray(ev["time_s"], dtype=float) - 0.5
    if float(lag_seconds) == RANGE_LAG_SECONDS:
        level2 = np.asarray(ev["motion_ratio"], dtype=float)
    else:
        level2, _ = motion_per_second(
            path, t0, t1, seconds, mimo=mimo, source_mac=source_mac,
            interpolate=interpolate, gate_gain=gate_gain, lag_seconds=RANGE_LAG_SECONDS,
        )
        level2 = np.asarray(level2, dtype=float)
        level2[np.asarray(ev["unknown"], dtype=bool)] = np.nan
    out["range_verdict"] = range_verdict(level2, ev["breath_peak"])
    return out
