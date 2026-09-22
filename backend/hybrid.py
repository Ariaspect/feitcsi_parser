"""Calibration-free presence: motion opens it, breathing keeps it open.

A human in a room cannot be quiet in both senses at once. Over any half
minute they either move -- irregularly -- or sit still, and a still human
breathes. A pushed chair does neither once it is down; a fan moves without
breathing. So neither channel decides on its own:

* **Motion** is a *burst*: the per-second motion level above a threshold for
  at least ``BURST_SECONDS`` in a row. It opens presence and refreshes it.
* **Breathing** is the FarSense evidence (``backend.farsense``, 30 s window,
  0.1 Hz high-pass): the normalised autocorrelation peak above
  ``BREATH_MIN_PEAK`` through ``BREATH_PERSIST_SECONDS`` of consecutive windows
  that mostly agree on the rate (``BREATH_RATE_TOL`` of the run's median,
  ``BREATH_MIN_FRACTION`` of them). It opens presence too -- a person
  already seated when the range starts has no entry burst to be found -- and
  it is what keeps presence open while they sit.
* Presence **holds** for ``HOLD_SECONDS`` after the last evidence and then
  drops. Nothing holds it forever: a room is empty once it has been still and
  unbreathing for that long, which is the whole difference from a baseline
  detector that a moved chair resets for the rest of the capture.

No empty-room reference, no labelled stretch, no number learned per room.
What the thresholds are relative to is the range's *own* quiet level: the
motion floor is a low percentile of the per-second level over the range, and
a burst has to clear a multiple of it. That is not calibration in the sense
that costs a protocol -- it needs no one to say when the room was empty --
but it is what lets one rule survive links whose frame-to-frame noise differs
seven-fold (0.028 on the 42 Hz captures of 2026-09-21 against 0.13-0.20 on
the September 20 Hz ones, same code, same room state). An absolute minimum
sits under the relative threshold so a range that is nothing but motion
cannot raise its own floor out of reach.

Two motion channels are available. The **ratio** channel is ``|dr|/|r|``
(``presence.fractional_motion``): dimensionless, immune to AGC, blind to
shadowing that hits both AP chains alike. The **amplitude** channel is the
per-frame median ``|dA|`` in dB over subcarriers, on the *raw* amplitude:
the AGC correction was measured to add jitter here (it fires on a person's
own frame-to-frame variation), and a per-second median already ignores the
single-frame impulse a gain step is. Amplitude sees shadowing; it is off by
default until the corpus says it earns its false positives.

The work is split in two. ``evidence_series`` is the expensive half -- the
per-second motion levels and the FarSense sweep -- and depends only on the
signal-shaping parameters; ``verdict`` is the cheap half that turns those
series into bursts, breathing runs and the held state, and depends on the
decision parameters. A tab that moves the hold or the peak threshold, or a
sweep over them, re-runs only the second.

Everything is scored on a 1 s grid against the camera through
``backend.truth``, margin and all.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any

import numpy as np

from backend import farsense
from backend.presence import fractional_motion, live_subcarriers

HOLD_SECONDS = 20.0
BURST_SECONDS = 2.0
# Motion threshold = max(MOTION_REL * floor, MOTION_ABS); floor is the
# FLOOR_PERCENTILE of the per-second level over the range.
MOTION_REL = 2.0
MOTION_ABS = 0.10          # |dr|/|r|
AMP_REL = 2.0
AMP_ABS = 0.5              # dB
FLOOR_PERCENTILE = 20.0
BREATH_MIN_PEAK = 0.2      # was 0.15; set 2026-09-22
# Half a 30 s window at the 1 s hop: the run's ends see mostly different data.
BREATH_PERSIST_SECONDS = 10.0  # was 15; set 2026-09-22
BREATH_RATE_TOL = 3.0      # rpm, from the run's median
# A run may carry a few windows that dip below the peak or stray in rate.
BREATH_MIN_FRACTION = 0.8
BREATH_WINDOW_SECONDS = 10.0   # was 30, then 15; set 2026-09-22
BREATH_HIGHPASS_HZ = 0.0       # was 0.1; set 2026-09-22
# A second is "unknown" when more than this fraction of its samples were
# interpolated across a dropout; it carries no evidence and no verdict.
MAX_GAP_FRACTION = 0.5

STATE_UNKNOWN = "unknown"
STATE_MOVING = "moving"
STATE_BREATHING = "breathing"
STATE_HELD = "held"
STATE_BRIDGED = "bridged"
STATE_EMPTY = "empty"


def per_second(values: np.ndarray, times: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    """Median of ``values`` inside each 1 s cell of ``seconds``; NaN where none.

    Median rather than mean so one impulse -- a gain step, a swapped frame --
    cannot make a second look like motion. ``times`` and ``values`` must be
    aligned; ``seconds`` are the cells' left edges.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    seconds = np.asarray(seconds, dtype=float)
    if values.shape != times.shape:
        raise ValueError(f"values {values.shape} and times {times.shape} must match")
    out = np.full(seconds.shape, np.nan)
    if not times.size:
        return out
    cell = np.searchsorted(seconds, times, side="right") - 1
    ok = (cell >= 0) & (cell < seconds.size) & np.isfinite(values)
    for c in np.unique(cell[ok]):
        out[c] = float(np.median(values[ok & (cell == c)]))
    return out


def floor_level(level: np.ndarray, percentile: float = FLOOR_PERCENTILE) -> float:
    """The range's own quiet level: a low percentile of the finite seconds."""
    finite = np.asarray(level, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def bursts(level: np.ndarray, threshold: float, min_run: int) -> np.ndarray:
    """Seconds inside a run of at least ``min_run`` consecutive seconds above ``threshold``."""
    above = np.isfinite(level) & (np.asarray(level, dtype=float) > threshold)
    out = np.zeros(above.shape, dtype=bool)
    i, n = 0, above.size
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        if j - i >= min_run:
            out[i:j] = True
        i = j
    return out


def hold_after(mask: np.ndarray, n: int) -> np.ndarray:
    """Cells within ``n`` cells after a true cell (the true cells included)."""
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    for k in range(1, min(int(n), mask.size - 1) + 1):
        out[k:] |= mask[:-k]
    return out


def hold_before(mask: np.ndarray, n: int) -> np.ndarray:
    """Cells within ``n`` cells before a true cell (the true cells included)."""
    return hold_after(np.asarray(mask, dtype=bool)[::-1], n)[::-1]


def gaps_between_bursts_with(burst: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Cells between two burst runs when any ``inside`` cell lies between them.

    The heuristic: a person who moved, then moved again, and showed any
    breathing in between never left -- fill the whole stretch.
    """
    burst = np.asarray(burst, dtype=bool)
    inside = np.asarray(inside, dtype=bool)
    out = np.zeros(burst.shape, dtype=bool)
    idx = np.flatnonzero(burst)
    if idx.size < 2:
        return out
    # ends of burst runs and starts of the next ones
    breaks = np.flatnonzero(np.diff(idx) > 1)
    for k in breaks:
        a, b = idx[k], idx[k + 1]
        if inside[a + 1 : b].any():
            out[a + 1 : b] = True
    return out


def bridged_gaps(evidence: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Gaps between two evidence cells whose every cell a hold covers.

    The heuristic: a hold running forward from one piece of evidence and a
    hold running back from the next that meet make the whole gap present.
    """
    evidence = np.asarray(evidence, dtype=bool)
    covered = np.asarray(covered, dtype=bool)
    out = np.zeros(evidence.shape, dtype=bool)
    idx = np.flatnonzero(evidence)
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1 and covered[a + 1 : b].all():
            out[a + 1 : b] = True
    return out


def consistent_breathing(
    peak: np.ndarray,
    rpm: np.ndarray,
    *,
    min_peak: float = BREATH_MIN_PEAK,
    n_consistent: int = 15,
    rate_tol: float = BREATH_RATE_TOL,
    min_fraction: float = BREATH_MIN_FRACTION,
) -> np.ndarray:
    """Windows inside a run of ``n_consistent`` that mostly clear ``min_peak`` and mostly agree on the rate.

    Agreement is what separates a chest from a noise peak: an empty room's
    first-peak rates scatter across the band from one window to the next, a
    person's stay within a few rpm. But windows a hop apart share almost all
    their samples, so two neighbours agreeing is no test at all -- the run
    has to be long enough that its ends see mostly different data, which is
    why the default is half a window (15 windows at a 1 s hop under a 30 s
    window).

    "Mostly", not "all": a run is accepted when at least ``min_fraction`` of
    its windows clear the peak and, of those, at least ``min_fraction`` sit
    within ``rate_tol`` of the run's median rate. Measured on
    20260916_202702, 17 straight windows of real breathing (peaks 0.16-0.40,
    rates 16.7-20.0) were rejected by an all-or-nothing rule twice over: one
    window dipped to 0.142, and the max-min spread was 3.3 rpm against a
    3 rpm tolerance. Every window of an accepted run is marked, so the
    evidence is as long as the run.
    """
    peak = np.asarray(peak, dtype=float)
    rpm = np.asarray(rpm, dtype=float)
    n = peak.size
    strong = np.isfinite(peak) & (peak >= min_peak) & np.isfinite(rpm)
    out = np.zeros(n, dtype=bool)
    if n_consistent <= 1:
        return strong
    need = max(1, int(np.ceil(min_fraction * n_consistent)))
    for i in range(n - n_consistent + 1):
        block = slice(i, i + n_consistent)
        q = strong[block]
        if q.sum() < need:
            continue
        rates = rpm[block][q]
        agree = np.abs(rates - np.median(rates)) <= rate_tol
        if agree.sum() >= need:
            out[block] = True
    return out


# --------------------------------------------------------------------------- #
#  Stage 1: the evidence series                                                #
# --------------------------------------------------------------------------- #


def evidence_series(
    grid: np.ndarray,
    fs: float,
    *,
    fabricated: np.ndarray | None = None,
    amp_diff: np.ndarray | None = None,
    amp_times: np.ndarray | None = None,
    breath_window_seconds: float = BREATH_WINDOW_SECONDS,
    breath_highpass_hz: float = BREATH_HIGHPASS_HZ,
    max_gap_fraction: float = MAX_GAP_FRACTION,
    band_rpm: tuple[float, float] = farsense.RATE_BAND_RPM,
    n_theta: int = farsense.N_THETA,
    fft_size: int = farsense.FFT_SIZE,
    keep_fraction: float = farsense.BNR_KEEP_FRACTION,
    savgol_seconds: float = farsense.SAVGOL_SECONDS,
    savgol_order: int = farsense.SAVGOL_ORDER,
    motion_frac_hi: float | None = None,
    positive_only: bool = True,
) -> dict[str, Any]:
    """Per-second motion levels and per-window breathing peaks, on one clock.

    The FarSense knobs (``band_rpm`` … ``positive_only``) are passed straight
    to ``farsense.prepare``/``window_step``. ``motion_frac_hi`` is the paper's
    stationary gate; ``None`` (the default) leaves it off so every window
    reports a peak and the hybrid's own burst rule decides what motion means.

    ``grid`` is ``(n_samples, n_sc)`` at ``fs`` Hz from ``tiles._presence_grid``;
    times are relative to its first sample. ``amp_diff``/``amp_times`` are the
    per-frame amplitude differences and their times for the amplitude
    channel, on the same clock.

    Breathing evidence is assigned to the *centre* of the window it was found
    in, which reads the future by half a window: fine for a replay scored
    against a camera, and the number to subtract for a live delay.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2-D (n_samples, n_sc), got {grid.shape}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be positive and finite, got {fs}")
    n = grid.shape[0]
    fab = (
        np.zeros(n, dtype=bool) if fabricated is None else np.asarray(fabricated, dtype=bool)
    )
    if fab.shape != (n,):
        raise ValueError(f"fabricated must be 1-D of length {n}, got {fab.shape}")

    duration = n / fs
    seconds = np.arange(0.0, max(duration, 1.0))
    n_sec = seconds.size
    sample_t = np.arange(n) / fs

    live = live_subcarriers(grid)
    if not live.any():
        raise ValueError("no subcarrier in this range carries a CSI ratio")
    ratio = grid[:, live]

    frac = fractional_motion(ratio)
    motion_ratio = per_second(frac, sample_t[1:], seconds)
    unknown = per_second(fab.astype(float), sample_t, seconds)
    unknown = ~np.isfinite(unknown) | (unknown > max_gap_fraction)
    motion_ratio[unknown] = np.nan

    motion_amp = np.full(n_sec, np.nan)
    if amp_diff is not None and amp_times is not None and np.asarray(amp_diff).size:
        motion_amp = per_second(np.asarray(amp_diff), np.asarray(amp_times), seconds)
        motion_amp[unknown] = np.nan

    breath_peak = np.full(n_sec, np.nan)
    breath_rpm = np.full(n_sec, np.nan)
    breath_note = None
    try:
        prep = farsense.prepare(
            grid, fs, fabricated=fab,
            window_seconds=breath_window_seconds, hop_seconds=1.0,
            band_rpm=(float(band_rpm[0]), float(band_rpm[1])),
            savgol_seconds=savgol_seconds, savgol_order=savgol_order,
            highpass_hz=breath_highpass_hz,
        )
        # With the stationary gate off, every window gets its peak and rate
        # and this module's relative burst decides what motion means.
        # ``positive_only``: a negative "first peak" is a wiggle on the slope
        # out of the trough, not a period -- see ``farsense.first_peak``.
        step = dict(
            n_theta=int(n_theta), fft_size=int(fft_size),
            keep_fraction=float(keep_fraction),
            motion_frac_hi=1e9 if motion_frac_hi is None else float(motion_frac_hi),
            max_gap_fraction=max_gap_fraction, min_peak=-1.0, positive_only=bool(positive_only),
        )
        fz = farsense._run(prep, step, None)
        cell = np.floor(fz["time_s"]).astype(int)
        inside = (cell >= 0) & (cell < n_sec) & ~fz["unknown"]
        breath_peak[cell[inside]] = fz["acf_peak_norm"][inside]
        breath_rpm[cell[inside]] = fz["rpm"][inside]
    except ValueError as exc:
        # A range shorter than one breathing window has no breathing channel;
        # the motion channel still runs, and the reason is reported.
        breath_note = str(exc)

    return {
        "time_s": seconds + 0.5,
        "unknown": unknown,
        "motion_ratio": motion_ratio,
        "motion_amp": motion_amp,
        "breath_peak": breath_peak,
        "breath_rpm": breath_rpm,
        "breath_note": breath_note,
        "fs_hz": float(fs),
        "evidence_params": {
            "breath_window_seconds": float(breath_window_seconds),
            "breath_highpass_hz": float(breath_highpass_hz),
            "max_gap_fraction": float(max_gap_fraction),
            "rpm_lo": float(band_rpm[0]),
            "rpm_hi": float(band_rpm[1]),
            "n_theta": int(n_theta),
            "fft_size": int(fft_size),
            "keep_fraction": float(keep_fraction),
            "savgol_seconds": float(savgol_seconds),
            "savgol_order": int(savgol_order),
            "motion_frac_hi": None if motion_frac_hi is None else float(motion_frac_hi),
            "positive_only": bool(positive_only),
        },
    }


# --------------------------------------------------------------------------- #
#  Stage 2: the verdict                                                        #
# --------------------------------------------------------------------------- #


def verdict(
    ev: dict[str, Any],
    *,
    use_amplitude: bool = False,
    hold_seconds: float = HOLD_SECONDS,
    burst_seconds: float = BURST_SECONDS,
    motion_rel: float = MOTION_REL,
    motion_abs: float = MOTION_ABS,
    amp_rel: float = AMP_REL,
    amp_abs: float = AMP_ABS,
    floor_percentile: float = FLOOR_PERCENTILE,
    breath_min_peak: float = BREATH_MIN_PEAK,
    breath_persist_seconds: float = BREATH_PERSIST_SECONDS,
    breath_rate_tol: float = BREATH_RATE_TOL,
    breath_min_fraction: float = BREATH_MIN_FRACTION,
    motion_floor: float | None = None,
    lead_hold: bool = True,
    bridge_bursts: str = "off",
) -> dict[str, Any]:
    """Bursts, breathing runs and the held state from an evidence series.

    ``bridge_bursts``: ``"off"``; ``"run"`` fills the whole stretch between
    two bursts when a breathing *run* (the persistence rule) lies between
    them; ``"any"`` does so when any single window's peak clears
    ``breath_min_peak`` there.

    Presence runs ``hold_seconds`` past any evidence. With ``lead_hold`` the
    breathing evidence also holds presence for ``hold_seconds`` *before* it
    (the person was already there while the first window filled), and a gap
    whose holds meet -- a burst's trailing hold reaching a breathing run's
    leading one, say -- is present throughout (state ``bridged``).

    The floor is the range's own quiet level (``floor_percentile`` of its
    per-second motion). ``motion_floor`` replaces it with an explicit value.
    A range occupied throughout has no quiet stretch -- its own percentile
    *is* the occupant -- and a floor pooled from neighbouring captures was
    tried for that (docs/hybrid.md §2) and removed: it is a reference taken
    from other captures, and on a link whose idle level had risen it set the
    threshold under an empty room.
    """
    seconds = np.asarray(ev["time_s"], dtype=float) - 0.5
    n_sec = seconds.size
    unknown = np.asarray(ev["unknown"], dtype=bool)
    motion_ratio = np.asarray(ev["motion_ratio"], dtype=float)
    motion_amp = np.asarray(ev["motion_amp"], dtype=float)
    min_run = max(1, int(round(burst_seconds)))

    ratio_floor = (
        float(motion_floor)
        if motion_floor is not None and np.isfinite(motion_floor)
        else floor_level(motion_ratio, floor_percentile)
    )
    ratio_threshold = (
        max(motion_rel * ratio_floor, motion_abs) if np.isfinite(ratio_floor) else motion_abs
    )
    burst = bursts(motion_ratio, ratio_threshold, min_run)

    amp_floor = float("nan")
    amp_threshold = float("nan")
    if np.isfinite(motion_amp).any():
        amp_floor = floor_level(motion_amp, floor_percentile)
        amp_threshold = max(amp_rel * amp_floor, amp_abs) if np.isfinite(amp_floor) else amp_abs
        if use_amplitude:
            burst = burst | bursts(motion_amp, amp_threshold, min_run)

    breathing = consistent_breathing(
        ev["breath_peak"], ev["breath_rpm"],
        min_peak=breath_min_peak,
        n_consistent=max(1, int(round(breath_persist_seconds))),
        rate_tol=breath_rate_tol,
        min_fraction=breath_min_fraction,
    )
    breathing &= ~burst & ~unknown

    if bridge_bursts not in ("off", "run", "any"):
        raise ValueError(f"bridge_bursts must be 'off', 'run' or 'any', got {bridge_bursts!r}")
    n_hold = max(0, int(round(hold_seconds)))
    evidence = burst | breathing
    trailing = hold_after(evidence, n_hold)
    leading = hold_before(breathing, n_hold) if lead_hold else np.zeros(n_sec, dtype=bool)
    covered = (trailing | leading) & ~evidence
    bridged = bridged_gaps(evidence, covered)
    if bridge_bursts != "off":
        peak_arr = np.asarray(ev["breath_peak"], dtype=float)
        inside = breathing if bridge_bursts == "run" else (
            np.isfinite(peak_arr) & (peak_arr >= breath_min_peak) & ~unknown & ~burst
        )
        filled = gaps_between_bursts_with(burst, inside) & ~evidence
        bridged = bridged | filled
        covered = covered | filled
    present = (evidence | covered) & ~unknown

    state = np.full(n_sec, STATE_EMPTY, dtype=object)
    state[covered] = STATE_HELD
    state[bridged] = STATE_BRIDGED
    state[breathing] = STATE_BREATHING
    state[burst] = STATE_MOVING
    state[unknown] = STATE_UNKNOWN

    out = dict(ev)
    out.update(
        present=present,
        state=[str(s) for s in state],
        burst=burst,
        breathing=breathing,
        ratio_floor=ratio_floor,
        ratio_threshold=float(ratio_threshold),
        amp_floor=amp_floor,
        amp_threshold=float(amp_threshold),
        params={
            **ev["evidence_params"],
            "use_amplitude": bool(use_amplitude),
            "hold_seconds": float(hold_seconds),
            "burst_seconds": float(burst_seconds),
            "motion_rel": float(motion_rel),
            "motion_abs": float(motion_abs),
            "amp_rel": float(amp_rel),
            "amp_abs": float(amp_abs),
            "floor_percentile": float(floor_percentile),
            "breath_min_peak": float(breath_min_peak),
            "breath_persist_seconds": float(breath_persist_seconds),
            "breath_rate_tol": float(breath_rate_tol),
            "breath_min_fraction": float(breath_min_fraction),
            "motion_floor": None if motion_floor is None else float(motion_floor),
            "lead_hold": bool(lead_hold),
            "bridge_bursts": str(bridge_bursts),
        },
    )
    out.pop("evidence_params", None)
    return out


EVIDENCE_PARAMS = (
    "breath_window_seconds", "breath_highpass_hz", "max_gap_fraction",
    "band_rpm", "n_theta", "fft_size", "keep_fraction",
    "savgol_seconds", "savgol_order", "motion_frac_hi", "positive_only",
)


def split_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate ``evidence_series`` keywords from ``verdict`` keywords."""
    evidence = {k: v for k, v in params.items() if k in EVIDENCE_PARAMS}
    decision = {k: v for k, v in params.items() if k not in EVIDENCE_PARAMS}
    return evidence, decision


def hybrid_seconds(
    grid: np.ndarray,
    fs: float,
    *,
    fabricated: np.ndarray | None = None,
    amp_diff: np.ndarray | None = None,
    amp_times: np.ndarray | None = None,
    **params: Any,
) -> dict[str, Any]:
    """Both stages over a uniform complex ratio grid, one verdict per second."""
    evidence, decision = split_params(params)
    ev = evidence_series(grid, fs, fabricated=fabricated, amp_diff=amp_diff, amp_times=amp_times, **evidence)
    return verdict(ev, **decision)


# --------------------------------------------------------------------------- #
#  Over a capture                                                              #
# --------------------------------------------------------------------------- #


def amplitude_diff(path, index, frame_ids: np.ndarray, *, interpolate: bool = True) -> np.ndarray:
    """Per-frame median ``|dA|`` in dB over subcarriers, raw amplitude, no AGC table.

    Length ``len(frame_ids) - 1``; entry *i* is the step from frame *i* to
    *i + 1*. Raw on purpose -- see the module note.
    """
    from backend.tiles import _decode_for_doppler

    amp = _decode_for_doppler(path, index, np.asarray(frame_ids), "amplitude", None, interpolate)
    if amp.shape[0] < 2:
        return np.zeros(0)
    with np.errstate(invalid="ignore"):
        step = np.abs(np.diff(amp.astype(float), axis=0))
    out = np.full(step.shape[0], np.nan)
    alive = np.isfinite(step).any(axis=1)
    if alive.any():
        out[alive] = np.nanmedian(step[alive], axis=1)
    return out


# The evidence for a (capture, range, shaping parameters) is kept for the
# last few requests, so a change of hold or threshold on the tab -- or a sweep
# over them -- costs the cheap stage only.
_CACHE_SIZE = 6
_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
_cache_lock = Lock()


def reset_cache() -> None:
    with _cache_lock:
        _cache.clear()


def capture_evidence(
    path,
    t0: float,
    t1: float,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
    **evidence: Any,
) -> dict[str, Any]:
    """Decode a capture range and build its evidence series, cached; times on the capture's clock.

    ``evidence`` are ``evidence_series`` keywords (see ``EVIDENCE_PARAMS``).
    """
    from pathlib import Path

    from backend.tiles import _presence_grid, get_index

    unknown = set(evidence) - set(EVIDENCE_PARAMS)
    if unknown:
        raise TypeError(f"not evidence parameters: {sorted(unknown)}")
    path = Path(path)
    st = path.stat()
    key = (
        str(path.resolve()), st.st_size, st.st_mtime_ns, float(t0), float(t1),
        mimo, source_mac, bool(interpolate),
        tuple(sorted((k, tuple(v) if isinstance(v, (tuple, list)) else v) for k, v in evidence.items())),
    )
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit

    grid, fabricated, fs, grid_times, times, times_all, n_no_ratio = _presence_grid(
        path, t0, t1, mimo=mimo, source_mac=source_mac, interpolate=interpolate,
    )
    origin = float(grid_times[0])
    index = get_index(path)
    mask = index.filter_mask(mimo=mimo, source_mac=source_mac)
    ids = np.flatnonzero(mask & (times_all >= t0) & (times_all <= t1))
    amp_diff = amplitude_diff(path, index, ids, interpolate=interpolate) if ids.size >= 2 else None
    amp_times = (times_all[ids][1:] - origin) if amp_diff is not None else None

    ev = evidence_series(
        grid, fs, fabricated=fabricated, amp_diff=amp_diff, amp_times=amp_times, **evidence,
    )
    ev["time_s"] = origin + ev["time_s"]
    ev["frames_used"] = int(times.size)
    ev["frames_without_ratio"] = int(n_no_ratio)
    ev["t_min"] = float(times_all[0]) if times_all.size else 0.0
    ev["t_max"] = float(times_all[-1]) if times_all.size else 0.0

    with _cache_lock:
        _cache[key] = ev
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return ev


def compute_hybrid(
    path,
    t0: float,
    t1: float,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
    **params: Any,
) -> dict[str, Any]:
    """Decode a capture range and run the detector; times on the capture's clock.

    ``params`` are ``evidence_series`` keywords and ``verdict`` keywords mixed.
    """
    evidence, decision = split_params(params)
    ev = capture_evidence(path, t0, t1, mimo=mimo, source_mac=source_mac, interpolate=interpolate, **evidence)
    # ``verdict`` works on times relative to the first second; the cached
    # series is on the capture's clock, so shift in and back out.
    origin = float(ev["time_s"][0]) - 0.5
    local = dict(ev)
    local["time_s"] = np.asarray(ev["time_s"], dtype=float) - origin
    out = verdict(local, **decision)
    out["time_s"] = np.asarray(out["time_s"], dtype=float) + origin
    return out
