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

Everything is scored on a 1 s grid against the camera through
``backend.truth``, margin and all.
"""

from __future__ import annotations

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
BREATH_MIN_PEAK = 0.15
# Half a 30 s window at the 1 s hop: the run's ends see mostly different data.
BREATH_PERSIST_SECONDS = 15.0
BREATH_RATE_TOL = 3.0      # rpm, from the run's median
# A run may carry a few windows that dip below the peak or stray in rate.
BREATH_MIN_FRACTION = 0.8
BREATH_WINDOW_SECONDS = 30.0
BREATH_HIGHPASS_HZ = 0.1
# A second is "unknown" when more than this fraction of its samples were
# interpolated across a dropout; it carries no evidence and no verdict.
MAX_GAP_FRACTION = 0.5

STATE_UNKNOWN = "unknown"
STATE_MOVING = "moving"
STATE_BREATHING = "breathing"
STATE_HELD = "held"
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


def hybrid_seconds(
    grid: np.ndarray,
    fs: float,
    *,
    fabricated: np.ndarray | None = None,
    amp_diff: np.ndarray | None = None,
    amp_times: np.ndarray | None = None,
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
    breath_window_seconds: float = BREATH_WINDOW_SECONDS,
    breath_highpass_hz: float = BREATH_HIGHPASS_HZ,
    max_gap_fraction: float = MAX_GAP_FRACTION,
) -> dict[str, Any]:
    """The detector over a uniform complex ratio grid, one verdict per second.

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

    # -- motion, ratio channel ------------------------------------------
    frac = fractional_motion(ratio)
    motion_ratio = per_second(frac, sample_t[1:], seconds)
    unknown = per_second(fab.astype(float), sample_t, seconds)
    unknown = ~np.isfinite(unknown) | (unknown > max_gap_fraction)
    motion_ratio[unknown] = np.nan
    ratio_floor = floor_level(motion_ratio, floor_percentile)
    ratio_threshold = max(motion_rel * ratio_floor, motion_abs) if np.isfinite(ratio_floor) else motion_abs
    burst = bursts(motion_ratio, ratio_threshold, max(1, int(round(burst_seconds))))

    # -- motion, amplitude channel --------------------------------------
    motion_amp = np.full(n_sec, np.nan)
    amp_floor = float("nan")
    amp_threshold = float("nan")
    if amp_diff is not None and amp_times is not None and np.asarray(amp_diff).size:
        motion_amp = per_second(np.asarray(amp_diff), np.asarray(amp_times), seconds)
        motion_amp[unknown] = np.nan
        amp_floor = floor_level(motion_amp, floor_percentile)
        amp_threshold = max(amp_rel * amp_floor, amp_abs) if np.isfinite(amp_floor) else amp_abs
        if use_amplitude:
            burst = burst | bursts(motion_amp, amp_threshold, max(1, int(round(burst_seconds))))

    # -- breathing --------------------------------------------------------
    breath_peak = np.full(n_sec, np.nan)
    breath_rpm = np.full(n_sec, np.nan)
    breathing = np.zeros(n_sec, dtype=bool)
    breath_note = None
    try:
        prep = farsense.prepare(
            grid, fs, fabricated=fab,
            window_seconds=breath_window_seconds, hop_seconds=1.0,
            highpass_hz=breath_highpass_hz,
        )
        # The paper's absolute stationary gate is replaced by this module's
        # relative burst; every window gets its peak and rate.
        step = dict(
            n_theta=farsense.N_THETA, fft_size=farsense.FFT_SIZE,
            keep_fraction=farsense.BNR_KEEP_FRACTION, motion_frac_hi=1e9,
            max_gap_fraction=max_gap_fraction, min_peak=-1.0,
        )
        fz = farsense._run(prep, step, None)
        centres = fz["time_s"]
        ok = ~fz["unknown"]
        w_evidence = consistent_breathing(
            np.where(ok, fz["acf_peak_norm"], np.nan), fz["rpm"],
            min_peak=breath_min_peak, n_consistent=max(1, int(round(breath_persist_seconds))),
            rate_tol=breath_rate_tol, min_fraction=breath_min_fraction,
        )
        cell = np.floor(centres).astype(int)
        inside = (cell >= 0) & (cell < n_sec)
        breath_peak[cell[inside]] = fz["acf_peak_norm"][inside]
        breath_rpm[cell[inside]] = fz["rpm"][inside]
        breathing[cell[inside]] = w_evidence[inside]
    except ValueError as exc:
        # A range shorter than one breathing window has no breathing channel;
        # the motion channel still runs, and the reason is reported.
        breath_note = str(exc)
    breathing &= ~burst & ~unknown

    # -- the state machine ------------------------------------------------
    present = np.zeros(n_sec, dtype=bool)
    state = np.full(n_sec, STATE_EMPTY, dtype=object)
    last_evidence = -np.inf
    for i in range(n_sec):
        if unknown[i]:
            state[i] = STATE_UNKNOWN
            continue
        if burst[i] or breathing[i]:
            last_evidence = seconds[i]
        present[i] = (seconds[i] - last_evidence) <= hold_seconds
        if burst[i]:
            state[i] = STATE_MOVING
        elif breathing[i]:
            state[i] = STATE_BREATHING
        elif present[i]:
            state[i] = STATE_HELD
        else:
            state[i] = STATE_EMPTY

    return {
        "time_s": seconds + 0.5,
        "present": present,
        "state": [str(s) for s in state],
        "unknown": unknown,
        "motion_ratio": motion_ratio,
        "motion_amp": motion_amp,
        "burst": burst,
        "breathing": breathing,
        "breath_peak": breath_peak,
        "breath_rpm": breath_rpm,
        "ratio_floor": ratio_floor,
        "ratio_threshold": float(ratio_threshold),
        "amp_floor": amp_floor,
        "amp_threshold": float(amp_threshold),
        "breath_note": breath_note,
        "fs_hz": float(fs),
        "params": {
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
            "breath_window_seconds": float(breath_window_seconds),
            "breath_highpass_hz": float(breath_highpass_hz),
            "max_gap_fraction": float(max_gap_fraction),
        },
    }


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
    """Decode a capture range and run the detector; times on the capture's clock."""
    from pathlib import Path

    from backend.tiles import _presence_grid, get_index

    path = Path(path)
    grid, fabricated, fs, grid_times, times, times_all, n_no_ratio = _presence_grid(
        path, t0, t1, mimo=mimo, source_mac=source_mac, interpolate=interpolate,
    )
    origin = float(grid_times[0])

    index = get_index(path)
    mask = index.filter_mask(mimo=mimo, source_mac=source_mac)
    ids = np.flatnonzero(mask & (times_all >= t0) & (times_all <= t1))
    amp_diff = amplitude_diff(path, index, ids, interpolate=interpolate) if ids.size >= 2 else None
    amp_times = (times_all[ids][1:] - origin) if amp_diff is not None else None

    result = hybrid_seconds(
        grid, fs, fabricated=fabricated, amp_diff=amp_diff, amp_times=amp_times, **params,
    )
    result["time_s"] = origin + result["time_s"]
    result["frames_used"] = int(times.size)
    result["frames_without_ratio"] = int(n_no_ratio)
    result["t_min"] = float(times_all[0]) if times_all.size else 0.0
    result["t_max"] = float(times_all[-1]) if times_all.size else 0.0
    return result
