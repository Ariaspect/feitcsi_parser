"""The frame-to-frame amplitude step, read on a bounded -1..1 axis.

The oldest presence signal in this project is the one the LG board ships: take
the raw per-subcarrier amplitude, difference it against the frame before, and
fire when the step is large. ``backend.hybrid.amplitude_diff`` still computes
it as the median ``|dA|`` in dB, and the Phase 1 tab scores the board's own
version of it at a 26 dB trigger. What neither offers is a number a reader can
place without knowing the link: a dB step is unbounded above, so "is 8 dB a
lot" has no answer that survives a change of radio.

This module keeps the difference and changes only the axis it is read on. The
step between two frames is expressed as the *symmetric relative* difference of
the linear amplitudes,

    d = (a_t - a_{t-1}) / (a_t + a_{t-1}),

which is bounded in (-1, 1) by construction -- +1 is a subcarrier appearing
from nothing, -1 one vanishing into it, 0 no change -- and needs no reference,
no floor and no per-room constant.

**It is the same measurement, not a different one.** Writing the ratio of the
two amplitudes as ``a_t / a_{t-1} = 10^(dB/20)`` and substituting,

    d = (e^u - 1) / (e^u + 1) = tanh(u / 2),   u = dB * ln10 / 20

so ``d = tanh(dB * ln10 / 40)`` exactly. The bounded axis is a monotone
squash of the dB step and nothing else: no information is added and none is
lost, the ordering of steps is preserved, and a step can always be read back in
dB (``unit_to_db``). Phase 1's 26 dB, for the record, lands at d = 0.9045.

The consequence worth stating is that ``median`` commutes with a monotone map,
so the median fold over subcarriers *here* is the tanh of the median fold in
dB -- this panel's magnitude trace and ``hybrid.amplitude_diff`` are the same
series on two scales, not two series that happen to agree.

Two folds over the subcarrier axis are returned, and the difference matters:

* ``signed`` keeps the sign, as asked: positive is the array brightening,
  negative fading. Its weakness is structural -- a body arriving brightens
  some subcarriers and fades others, so a *signed* median can sit near zero
  through strong motion.
* ``magnitude`` is the median of ``|d|``, which cannot cancel, and is the one
  that equals the hybrid's amplitude channel.

No threshold is drawn or reported. The board's 26 dB line was carried here at
first, and measuring it removed the reason to: the loudest median step over 21
captures is 0.675 (22.5 dB), so the line sits above everything this fold can
produce, and the per-subcarrier share that does cross it has a median of 0.0000
-- an axis marking nothing and a counter reading zero. The dB equivalent of a
step is still reported (``median_db``, ``max_db``) because that needs no
threshold; ``docs/frame_step.md`` keeps the finding about the board.

**The receiver's gain control is not negligible here** (measured 2026-09-28
over 21 captures, ``docs/frame_step.md``). The reported gain state changes
across 10-52 % of consecutive frame pairs; at such a pair the median step is
1.0-9.5x larger; and **84-100 % of the loudest 1 % of steps are gain
crossings**, against that 10-52 % base rate. The tail of this signal belongs
to the radio, not the room.

Two things that do *not* fix it, both measured rather than argued:

* **The AGC correction** (``backend.agc``, per-gain-state shape offsets) leaves
  the median within 2 % on most captures and makes the 99th percentile *worse*
  by 12-125 %. It is a per-frame, shape-only correction, so differencing two
  differently-corrected frames adds a step of its own -- the same jitter
  ``hybrid`` records for its amplitude channel. Amplitude here stays **raw**,
  matching the board and ``hybrid.amplitude_diff``.
* **Inferring it from the fold.** The obvious idea -- a gain step moves the
  whole array one way, a body leaves subcarriers disagreeing -- does not
  survive measurement. ``|signed| / |d|`` at a gain crossing runs 0.62-1.00
  against 0.45-0.97 at a same-state step, which separates one capture of four
  and none of the rest.

What does work is not an inference at all: the gain state is *reported* per
frame, so ``gate_gain`` blanks the pairs that cross one, the same way a pair
spanning a dropout is blanked. It costs 12-52 % of the steps and halves the
tail (99th percentile down 0.3-82 %, median -50 %), leaves the empty/occupied
separation where it was, and does **not** shrink the 15.5x spread of the empty
level across captures -- that spread is the link, not the AGC. Off by default:
nothing here changes what the panel showed before it was measured.

Measured levels, and what this does and does not separate, are in
``docs/frame_step.md``: walking clears its own capture's empty seconds by
3.6-4.2x, phone and still sitters not at all, and the empty level itself spans
15.5x across captures -- so a bounded axis is not a calibrated one.

Steps across a dropout are dropped, not shown. Two frames a second apart have
no frame-to-frame difference worth the name, and the interpolator's smoothness
would read here as calm. The limit is ``doppler.gap_limit_for`` -- the same one
the spectrogram and the presence grid use, so every panel agrees about what
counts as a hole.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

# d = tanh(dB * ln10 / 40). The whole identity between this axis and a dB step.
DB_TO_UNIT = math.log(10.0) / 40.0

# Enough columns for a wide panel with room to spare, and small enough that the
# response stays a few hundred kB rather than a few megabytes: 600 s at 42 Hz is
# 25 000 steps, which no plot has pixels for.
DEFAULT_MAX_POINTS = 2000

# Subcarriers a step needs before its fold is reported. One live subcarrier is
# a median of one.
MIN_LIVE_SUBCARRIERS = 4

# The frame set this metric is defined on, when the caller names none.
#
# A step is only a step between two frames of the SAME SHAPE. Measured over
# three captures, 0.6-0.9 % of consecutive pairs change bandwidth, and those
# pairs read a median |d| of 0.220-0.385 against 0.012-0.044 for the 80 MHz
# ones -- 9-18x, on 57 live subcarriers instead of 245, because a narrow frame
# is centred and NaN-padded into the wide row so only the middle bins overlap.
# On 20260916_143259 EVERY such pair exceeds the entire same-width series'
# maximum. That is a bigger per-step artifact than the AGC, and unlike the AGC
# it is pure bookkeeping: those pairs are two different bandwidths, not two
# moments in one room.
#
# ``(num_rx, num_tx)`` as ``index.filter_mask`` reads them, which on a MediaTek
# capture means "both tpi slots present" -- the UI writes it "2x1". Measured on
# four captures, every (2, 1) frame is full width, while 43-51 frames per
# capture are full width at (1, 1), so the mode is the load-bearing filter and
# the width check below is the belt to its braces.
STRICT_MIMO = (2, 1)


def db_to_unit(step_db: np.ndarray | float) -> np.ndarray | float:
    """Map a dB step onto the bounded axis. ``tanh(dB * ln10 / 40)``."""
    return np.tanh(np.asarray(step_db, dtype=float) * DB_TO_UNIT)


def unit_to_db(d: np.ndarray | float) -> np.ndarray | float:
    """The inverse, for labelling an axis in the units the step was measured in."""
    d = np.asarray(d, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.arctanh(np.clip(d, -1.0, 1.0)) / DB_TO_UNIT


def relative_step(amp_db: np.ndarray) -> np.ndarray:
    """Per-subcarrier signed step in ``(-1, 1)``, one row per frame pair.

    *amp_db* is ``(n_frames, n_subcarriers)`` of raw amplitude in dB, as
    ``tiles._decode_for_doppler`` returns it. Row *i* is the step from frame
    *i* to frame *i + 1*, so the result is one row shorter.
    """
    amp_db = np.asarray(amp_db, dtype=float)
    if amp_db.ndim != 2:
        raise ValueError(f"amp_db must be 2-D (n_frames, n_sc), got {amp_db.shape}")
    if amp_db.shape[0] < 2:
        return np.zeros((0, amp_db.shape[1]))
    with np.errstate(invalid="ignore"):
        return np.tanh(np.diff(amp_db, axis=0) * DB_TO_UNIT)


def fold(steps: np.ndarray) -> dict[str, np.ndarray]:
    """Reduce ``(n_steps, n_sc)`` to one row of numbers per step.

    ``signed`` and ``magnitude`` are the two median folds and ``live`` is how
    many subcarriers carried them. A step with too few live subcarriers reports
    NaN rather than a median of one or two: a dead array is not a quiet room.
    """
    steps = np.asarray(steps, dtype=float)
    if steps.ndim != 2:
        raise ValueError(f"steps must be 2-D (n_steps, n_sc), got {steps.shape}")
    n = steps.shape[0]
    out = {
        "signed": np.full(n, np.nan),
        "magnitude": np.full(n, np.nan),
        "live": np.zeros(n, dtype=int),
    }
    if n == 0:
        return out

    alive = np.isfinite(steps)
    live = alive.sum(axis=1)
    out["live"] = live
    usable = live >= MIN_LIVE_SUBCARRIERS
    if not usable.any():
        return out

    rows = steps[usable]
    out["signed"][usable] = np.nanmedian(rows, axis=1)
    out["magnitude"][usable] = np.nanmedian(np.abs(rows), axis=1)
    return out


def frame_steps(
    amp_db: np.ndarray,
    times: np.ndarray,
    *,
    gap_limit: float | None = None,
    gain_state: np.ndarray | None = None,
    gate_gain: bool = False,
) -> dict[str, Any]:
    """The per-step series over one decoded range, holes removed.

    *times* are the frame times, same length as *amp_db*. Each step is
    timestamped at the *later* of its two frames, which is the convention
    ``hybrid.capture_evidence`` already uses for its amplitude channel.

    *gain_state* is the reported per-frame gain state (in practice the index's
    ``rssi_1``). Given, the crossings are always *counted*; with *gate_gain*
    they are blanked too. Counted even when not gated because a reader owed the
    number is the point -- see the module note on how much of the tail they are.
    """
    from backend.doppler import gap_limit_for

    amp_db = np.asarray(amp_db, dtype=float)
    times = np.asarray(times, dtype=float)
    if times.shape[0] != amp_db.shape[0]:
        raise ValueError(
            f"{times.shape[0]} times against {amp_db.shape[0]} frames of amplitude"
        )

    steps = relative_step(amp_db)
    folded = fold(steps)
    dt = np.diff(times) if times.size >= 2 else np.zeros(0)
    limit = gap_limit_for(times) if gap_limit is None else float(gap_limit)

    # A step that spans a dropout is not a frame-to-frame step. Blanked rather
    # than dropped, so the series keeps its place on the time axis.
    blank = dt > limit
    n_bridged = int(blank.sum())

    crossed = np.zeros(dt.shape, dtype=bool)
    if gain_state is not None:
        gain_state = np.asarray(gain_state)
        if gain_state.shape[0] != amp_db.shape[0]:
            raise ValueError(
                f"{gain_state.shape[0]} gain states against {amp_db.shape[0]} frames"
            )
        if gain_state.size >= 2:
            crossed = np.diff(gain_state) != 0
        if gate_gain:
            blank = blank | crossed

    for key in ("signed", "magnitude"):
        folded[key][blank] = np.nan

    return {
        "time_s": times[1:] if times.size >= 2 else np.zeros(0),
        "dt": dt,
        "gap_limit": limit,
        "n_bridged": n_bridged,
        "n_gain_crossed": int(crossed.sum()),
        "gain_gated": bool(gate_gain and gain_state is not None),
        **folded,
    }


def decimate(series: dict[str, Any], max_points: int = DEFAULT_MAX_POINTS) -> dict[str, Any]:
    """Reduce the per-step series to at most *max_points* columns.

    Envelopes, not averages. The point of a per-frame signal is the single
    frame that moved, and a mean over 12 neighbours is exactly what hides it,
    so each column carries the median *and* the two extremes it spans. When
    the range holds fewer steps than columns asked for, every column is one
    step and the envelope collapses onto the value itself.
    """
    t = np.asarray(series["time_s"], dtype=float)
    n = t.size
    if n == 0:
        empty = np.zeros(0)
        return {
            "time_s": empty, "signed": empty, "signed_lo": empty, "signed_hi": empty,
            "magnitude": empty, "magnitude_hi": empty,
            "count": np.zeros(0, dtype=int), "bin_seconds": 0.0, "decimated": False,
        }

    max_points = max(1, int(max_points))
    if n <= max_points:
        return {
            "time_s": t,
            "signed": np.asarray(series["signed"], dtype=float),
            "signed_lo": np.asarray(series["signed"], dtype=float),
            "signed_hi": np.asarray(series["signed"], dtype=float),
            "magnitude": np.asarray(series["magnitude"], dtype=float),
            "magnitude_hi": np.asarray(series["magnitude"], dtype=float),
            "count": np.ones(n, dtype=int),
            "bin_seconds": float(np.median(np.diff(t))) if n >= 2 else 0.0,
            "decimated": False,
        }

    # Equal-time columns rather than equal-count ones: the panel's x axis is
    # time, and a dropout would otherwise stretch one column across it.
    edges = np.linspace(t[0], t[-1], max_points + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    idx = np.searchsorted(t, edges)

    signed = np.asarray(series["signed"], dtype=float)
    magnitude = np.asarray(series["magnitude"], dtype=float)

    keys = ("time_s", "signed", "signed_lo", "signed_hi", "magnitude",
            "magnitude_hi")
    out: dict[str, Any] = {k: np.full(max_points, np.nan) for k in keys}
    out["count"] = np.zeros(max_points, dtype=int)

    for i in range(max_points):
        lo, hi = idx[i], idx[i + 1]
        if hi <= lo:
            continue
        out["time_s"][i] = 0.5 * (edges[i] + edges[i + 1])
        out["count"][i] = hi - lo
        sl = signed[lo:hi]
        good = sl[np.isfinite(sl)]
        if good.size == 0:
            continue
        out["signed"][i] = np.median(good)
        out["signed_lo"][i] = good.min()
        out["signed_hi"][i] = good.max()
        # Each array masked on its own finiteness rather than on the signed
        # fold's. They are blanked together today; nothing here depends on it.
        mg = magnitude[lo:hi]
        mg = mg[np.isfinite(mg)]
        if mg.size:
            out["magnitude"][i] = np.median(mg)
            out["magnitude_hi"][i] = mg.max()

    out["bin_seconds"] = float(edges[1] - edges[0])
    out["decimated"] = True
    return out


# --------------------------------------------------------------------------- #
#  Over a capture                                                             #
# --------------------------------------------------------------------------- #

# The decode dominates and the folds are free, so a change of decimation or of
# the reference threshold on the tab costs nothing.
_CACHE_SIZE = 4
_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
_cache_lock = Lock()


def reset_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _uniform_selection(
    index, mimo: tuple[int, int] | None, source_mac: str | None
) -> dict[str, Any]:
    """One transmitter, one MIMO mode, one bandwidth -- and say which.

    The MAC defaults to the capture's dominant peer, the same choice
    ``tiles.get_gain_table`` makes and for the same reason: a frame from
    another sender differs from its neighbours by the channel to that sender.
    The mode defaults to ``STRICT_MIMO`` when the capture has it and to its
    most common mode when it does not, so a capture recorded some other way
    still yields a uniform set rather than an empty one.
    """
    mac = source_mac or (index.dominant_peer() if hasattr(index, "dominant_peer") else None)
    note_parts = []

    mode = mimo
    if mode is None:
        base = index.filter_mask(source_mac=mac)
        if index.filter_mask(mimo=STRICT_MIMO, source_mac=mac).any():
            mode = STRICT_MIMO
        else:
            rx = np.asarray(getattr(index, "num_rx_arr", np.zeros(0)))
            tx = np.asarray(getattr(index, "num_tx_arr", np.zeros(0)))
            if rx.size and tx.size and base.any():
                pairs, counts = np.unique(
                    np.stack([rx[base], tx[base]], axis=1), axis=0, return_counts=True
                )
                mode = tuple(int(v) for v in pairs[counts.argmax()])
                note_parts.append(f"no {STRICT_MIMO[0]}x{STRICT_MIMO[1]} frames")
    mask = index.filter_mask(mimo=mode, source_mac=mac)

    # Width, as a guard rather than as the filter: every STRICT_MIMO frame
    # measured so far is full width, but a link that dropped to 20 MHz while
    # keeping both slots would sail through the mode test.
    bins = getattr(index, "_bins", None)
    narrow = np.zeros(mask.shape, dtype=bool)
    if bins is not None:
        bins = np.asarray(bins)
        if bins.shape == mask.shape:
            narrow = mask & (bins != index.num_subcarriers)
            mask = mask & ~narrow
            if narrow.any():
                note_parts.append(f"{int(narrow.sum())} narrow frames dropped")

    mode_label = "any" if mode is None else f"{mode[0]}x{mode[1]}"
    note = f"{mac or 'any MAC'}, {mode_label}, full width"
    if note_parts:
        note += " (" + "; ".join(note_parts) + ")"
    return {"mask": mask, "narrow": narrow, "source_mac": mac,
            "mimo": None if mode is None else [int(mode[0]), int(mode[1])],
            "note": note}


def capture_steps(
    path,
    t0: float,
    t1: float,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
    gate_gain: bool = False,
) -> dict[str, Any]:
    """Decode a range and reduce it to the per-step series, cached.

    **One transmitter, one MIMO mode, one bandwidth.** Unlike
    ``hybrid.capture_evidence``, which takes whatever the caller filtered to,
    this enforces a uniform frame set, because a difference between two frames
    of different shape is bookkeeping rather than motion -- see
    ``STRICT_MIMO``. Left to themselves the filters resolve to the capture's
    dominant peer and ``STRICT_MIMO``; an explicit *mimo* or *source_mac* is
    honoured instead, and full width is required either way. What was actually
    used comes back in the result, so the panel never has to assume.

    The gain state comes from the index's ``rssi_1``, which is what
    ``backend.agc`` uses as its own state proxy. Taken for the *selected*
    frames, so a filtered view gates on the pairs it actually shows.
    """
    from backend.tiles import _decode_for_doppler, get_index

    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns, float(t0), float(t1),
           mimo, source_mac, bool(interpolate), bool(gate_gain))
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit

    index = get_index(path)
    times_all = np.asarray(index.times, dtype=float)
    sel = _uniform_selection(index, mimo, source_mac)
    in_range = (times_all >= t0) & (times_all <= t1)
    ids = np.flatnonzero(sel["mask"] & in_range)
    if ids.size < 2:
        raise ValueError(
            f"fewer than 2 frames in range once the set was made uniform "
            f"({sel['note']}), so there is no step to take"
        )

    amp_db = _decode_for_doppler(path, index, ids, "amplitude", None, interpolate)
    rssi = np.asarray(getattr(index, "rssi_1", None))
    gain_state = rssi[ids] if rssi.ndim == 1 and rssi.size > int(ids[-1]) else None
    out = frame_steps(amp_db, times_all[ids],
                      gain_state=gain_state, gate_gain=gate_gain)
    out["source_mac"] = sel["source_mac"]
    out["mimo"] = sel["mimo"]
    out["selection_note"] = sel["note"]
    # Frames the uniformity rule removed from this range, so the reader can see
    # the price of it rather than a silently shorter series.
    dropped = int((in_range & ~sel["mask"]).sum())
    out["frames_dropped"] = dropped
    out["frames_dropped_narrow"] = int((in_range & sel["narrow"]).sum())
    out["frames_used"] = int(ids.size)
    out["n_subcarriers"] = int(amp_db.shape[1])
    out["t_min"] = float(times_all[0]) if times_all.size else 0.0
    out["t_max"] = float(times_all[-1]) if times_all.size else 0.0

    with _cache_lock:
        _cache[key] = out
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return out


def compute_frame_diff(
    path,
    t0: float,
    t1: float,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
    gate_gain: bool = False,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    """The per-step series, decimated for a plot, with its summary."""
    steps = capture_steps(
        path, t0, t1, mimo=mimo, source_mac=source_mac,
        interpolate=interpolate, gate_gain=gate_gain,
    )
    binned = decimate(steps, max_points)

    mag = np.asarray(steps["magnitude"], dtype=float)
    finite = mag[np.isfinite(mag)]
    # How many subcarriers actually carried a step, which is not the array
    # width: the guard band and the dead bins never do.
    live = np.asarray(steps["live"], dtype=float)
    live = live[live > 0]
    return {
        **binned,
        "summary": {
            "steps": int(mag.size),
            "steps_measured": int(finite.size),
            "live_median": int(np.median(live)) if live.size else 0,
            "n_bridged": int(steps["n_bridged"]),
            "n_gain_crossed": int(steps["n_gain_crossed"]),
            "gain_gated": bool(steps["gain_gated"]),
            "gap_limit": float(steps["gap_limit"]),
            "median": float(np.median(finite)) if finite.size else None,
            "p99": float(np.percentile(finite, 99)) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "median_db": float(unit_to_db(np.median(finite))) if finite.size else None,
            "max_db": float(unit_to_db(finite.max())) if finite.size else None,
        },
        "frames_used": steps["frames_used"],
        "frames_dropped": steps["frames_dropped"],
        "frames_dropped_narrow": steps["frames_dropped_narrow"],
        "source_mac": steps["source_mac"],
        "mimo": steps["mimo"],
        "selection_note": steps["selection_note"],
        "n_subcarriers": steps["n_subcarriers"],
        "t_min": steps["t_min"],
        "t_max": steps["t_max"],
    }
