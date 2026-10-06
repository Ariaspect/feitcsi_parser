"""The one-minute capture classifier: a feature bank per unit of a range.

The application verdict is **one call per one-minute capture**
(``docs/one_minute_classifier.md``). A one-minute capture is almost always a
single state, which rules out every within-capture self-calibration, so the
classifier has to be built from features whose empty-room level either does
not depend on the link state or can be normalised by something that does not
depend on occupancy. Seven tests were agreed to find those features; this
module is where the ones that survive accumulate, computed on one decode so
the tab can show them side by side and the classifier of test 4 is fitted on
the same numbers the tab draws.

A range is cut into *units* (one-minute captures are one unit; a longer range
is divided into equal units nearest the requested length). Every feature is a
scalar per unit, listed in ``FEATURES`` with its origin and, where one has
been measured, a reference operating point. The per-window and per-second
series behind the scalars are returned too, so a reader can see *why* a unit
scored what it did.

**What is in the bank so far.**

*The range rule's input* -- the lag-2 s ratio-complex frame step, per-second
median, P90 over the unit (``hybrid2.range_verdict``). Sharp on its own link
(AUC 0.996, still sitter against empty) and at chance across the 09-16/17
link-state shift, because its empty floor moves 10x with the link.

*Test 1 -- WiDetect's statistic* (Zhang et al., IMWUT 2019): per T-frame
window, per subcarrier, the lag-1 sample autocorrelation of the linearly
detrended ratio level, averaged over subcarriers (``acf_windows``). Receiver
noise, clean or ten times noisier, is white at the frame rate and lands this
at the same place -- measured on 1,043 units over 14 days the empty floor is
-0.009 on the clean 09-30 link and +0.022 / +0.008 on the noisy 09-16/17
one, inside one null width, where the step reads 0.011 against 0.105 /
0.123. What moves it is slow coherent variation in 0.1-2 Hz: a sitter's
micro-motion and breathing, a robot, and whatever moved near the link on the
camera-empty mornings of 09-30. Two things the paper promised do not hold
here: the subcarriers of the ratio move together (F_eff ~ 2, not 244), so
the null is 0.08 wide rather than 0.007, and its tails are heavy -- the
closed-form threshold is not usable and the reference below is an
empirical operating point (90 % specificity on the current link's empties,
96 % on 09-16/17, 56 % recall of still sitters). A feature, not a
replacement: it buys a floor that transfers across link states at the price
of same-link sensitivity.

*Test 2 -- the link's own jitter.* The hypothesis was that the across-
subcarrier roughness of one frame would read the receiver's noise and not the
room. It does not read the room (occupied / empty 0.9-1.3x) but it does not
read the link either: 0.012-0.023 on every day while the step floor spans
10x. The noisy link is not per-subcarrier estimation noise; it is a frame-to-
frame change that is smooth across frequency, mostly frequency-selective, and
only partly a transmit-delay switch (09-17: 44 % of frames jump > 3 ns, but
pairs in the same state still read 0.090 against 0.110; 09-16 never jumps).
What does read the link is the **step at one frame of lag**: its 20th
percentile tracks the empty lag-2 s floor 1:1 across every day (ratio
1.1-1.9) and a still sitter, a phone user or the robot leave it at 0.97x,
1.07x, 1.02x of the empties -- continuous movement inflates it, so the
operational floor is its minimum over a trailing window of units, which came
within 0.9-1.06x of the true empty floor on 10 of 13 days with no knowledge of
occupancy. A threshold relative to that floor recovers cross-link
specificity (09-16/17: 0 % -> 100 %) and loses the still sitter there
(recall 0.14-0.40), because on that link the sitter's slow signal is smaller
than the link's own jitter. ``lag1_p20`` and ``step_norm`` are in the bank;
the mechanism behind the noisy state is still unknown.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from backend import hybrid, hybrid2

# One unit per one-minute capture.
UNIT_SECONDS = 60.0

# T for the autocorrelation window. ~2 s at 42 Hz; WiDetect's T = 60 at 30 Hz
# is also 2 s. Fixed in frames, not seconds, because the null distribution is
# a function of the sample count.
ACF_WINDOW_FRAMES = 84

# The step is read at the range rule's lag, so the P90 here is the P90 there.
STEP_LAG_SECONDS = hybrid2.RANGE_LAG_SECONDS

# Reference operating points, drawn as guides. Measured, not chosen: test 1's
# threshold at 90 % specificity on the current link's clean empties (301
# units, 09-29 .. 10-02), which held 96 % on the 09-16/17 link.
ACF_AMP_P90W_REFERENCE = 0.20
ACF_PHASE_P90W_REFERENCE = 0.22
STEP_P90_REFERENCE = hybrid2.RANGE_MOTION_P90
# Test 2: the step as a multiple of the link's own one-frame jitter. Empty
# units sit at 1.1-1.9 on every link state measured; 2.0 is where 09-16/17
# specificity reaches 100 % (and still-sitter recall there falls to 0.14-0.40).
STEP_NORM_REFERENCE = 2.0
# Test 3: breathing. The range rule's run, and the 20 s-window variant.
BREATH_PEAK = hybrid2.RANGE_BREATH_PEAK
BREATH_RUN_REFERENCE = hybrid2.RANGE_BREATH_RUN
BREATH_WINDOW_LONG_SECONDS = 20.0
BREATH_RATE_FLOOR_RPM = 11.0

# The bank. ``key`` is the unit field; ``test`` says where the feature came
# from; ``status`` is "in rule" for the inputs of the current range rule and
# "candidate" for everything the programme has produced since; ``reference``
# is the guide value or None; ``axis`` is the y range a chart of it should
# use. Future tests append here and set the matching unit field in
# ``compute_features`` -- the tab renders this list, so a new feature needs
# no new column code.
FEATURES: list[dict[str, Any]] = [
    {
        "key": "step_p90", "label": "step P90", "test": "range rule", "status": "in rule",
        "reference": STEP_P90_REFERENCE, "axis": [0.0, 0.3], "decimals": 4,
        "description": "90th percentile over the unit of the per-second median of the lag-2 s "
                       "ratio-complex frame step. The motion half of the range rule.",
    },
    {
        "key": "step_p50", "label": "step P50", "test": "range rule", "status": "context",
        "reference": None, "axis": [0.0, 0.3], "decimals": 4,
        "description": "Median over the unit of the same per-second step; the unit's typical level.",
    },
    {
        "key": "acf_amp_median", "label": "ψ̂ amp · med", "test": "1", "status": "candidate",
        "reference": None, "axis": [-0.2, 1.0], "decimals": 3,
        "description": "Median over the unit's 2 s windows of the lag-1 autocorrelation of the "
                       "detrended ratio amplitude, averaged over subcarriers.",
    },
    {
        "key": "acf_amp_p90", "label": "ψ̂ amp · P90w", "test": "1", "status": "candidate",
        "reference": ACF_AMP_P90W_REFERENCE, "axis": [-0.2, 1.0], "decimals": 3,
        "description": "90th percentile over the unit's windows of the same statistic. The "
                       "aggregate that caught intermittent motion; reference 0.20 is 90 % "
                       "specificity on the current link, 96 % on 09-16/17.",
    },
    {
        "key": "acf_phase_median", "label": "ψ̂ phase · med", "test": "1", "status": "candidate",
        "reference": None, "axis": [-0.2, 1.0], "decimals": 3,
        "description": "As ψ̂ amp · med, on the unwrapped ratio phase.",
    },
    {
        "key": "acf_phase_p90", "label": "ψ̂ phase · P90w", "test": "1", "status": "candidate",
        "reference": ACF_PHASE_P90W_REFERENCE, "axis": [-0.2, 1.0], "decimals": 3,
        "description": "As ψ̂ amp · P90w, on the unwrapped ratio phase. Marginally better than the "
                       "amplitude for seated movement against the noisy link (AUC 0.90 vs 0.85).",
    },
    {
        "key": "lag1_p20", "label": "link jitter", "test": "2", "status": "candidate",
        "reference": None, "axis": [0.0, 0.15], "decimals": 4,
        "description": "20th percentile over the unit of the per-second median frame step at ONE frame "
                       "of lag: the link's own frame-to-frame jitter. Tracks the empty lag-2 s floor 1:1 "
                       "across every link state (lag-2 P90 / this = 1.1-1.9 on 13 days whose floors span "
                       "10x) and a still sitter, a phone user or the robot leave it alone (0.97x, 1.07x, "
                       "1.02x of the same day's empties); continuous movement does inflate it (seated "
                       "fidgeting 7.5x), so the operational floor is the MINIMUM of this over a trailing "
                       "window of units, which occupancy can only push up.",
    },
    {
        "key": "step_norm", "label": "step / jitter", "test": "2", "status": "candidate",
        "reference": STEP_NORM_REFERENCE, "axis": [0.0, 10.0], "decimals": 2,
        "description": "step P90 divided by this unit's own link jitter. Empty units read 1.1-1.9 "
                       "(p90 2.5) on every link state; a still sitter 3.0 median but 1.3 at p10, and on "
                       "the noisy 09-16/17 link at the floor -- a relative threshold recovers cross-link "
                       "specificity (0 % -> 100 % at 2x) and loses the still sitter there (recall 0.14-0.40). "
                       "A continuously moving person inflates the denominator too, so read it with the "
                       "jitter column.",
    },
    {
        "key": "breath_run", "label": "breath run 10 s", "test": "range rule", "status": "in rule",
        "reference": BREATH_RUN_REFERENCE, "axis": [0.0, 60.0], "decimals": 0,
        "description": "Longest run of consecutive seconds whose 10 s FarSense window has a normalised "
                       "autocorrelation peak >= 0.25 -- the breathing half of the range rule (run >= 5). "
                       "Finds 85 % of still sitters, on the noisy 09-15/16/17 link as well as the clean "
                       "one, with 0 % false alarms on the current link's empties.",
    },
    {
        "key": "breath_run20", "label": "breath run 20 s", "test": "3", "status": "candidate",
        "reference": BREATH_RUN_REFERENCE, "axis": [0.0, 60.0], "decimals": 0,
        "description": "The same run with 20 s FarSense windows and a rate floor at 11 rpm (DeMan's "
                       "longer-window finding; the floor steps over the 7.5-10.9 rpm artefact of empty "
                       "rooms). Still sitters 85 % -> 94 % (noisy link 96 %) for 0 -> 1 % false alarms "
                       "on the current link's empties.",
    },
    {
        "key": "breath_rpm", "label": "breath rpm", "test": "3", "status": "context",
        "reference": None, "axis": [0.0, 40.0], "decimals": 1,
        "description": "Median FarSense rate over the seconds in the 20 s run, rpm. Empty-room artefacts "
                       "cluster at 8-11 rpm; sitters at 14-22.",
    },
    {
        "key": "gain_crossings", "label": "gain steps", "test": "context", "status": "context",
        "reference": None, "axis": None, "decimals": 0,
        "description": "Receiver gain-state changes (rssi_1) inside the unit. The ratio divides the "
                       "common gain out; this says how hard it had to.",
    },
    {
        "key": "rssi_median", "label": "RSSI", "test": "context", "status": "context",
        "reference": None, "axis": None, "decimals": 0,
        "description": "Median reported rssi_1 over the unit, dBm. A link-state indicator, not a feature.",
    },
]

_CACHE_SIZE = 4
_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
_cache_lock = Lock()


def reset_cache() -> None:
    with _cache_lock:
        _cache.clear()


# --------------------------------------------------------------------------- #
#  The statistic                                                              #
# --------------------------------------------------------------------------- #


def acf_windows(
    X: np.ndarray, times: np.ndarray, T: int,
) -> tuple[np.ndarray, np.ndarray]:
    """WiDetect's ψ̂ per T-frame block of ``X`` (frames × subcarriers).

    Per block and subcarrier: subtract the least-squares line, take the lag-1
    sample autocorrelation ``Σ r_t r_{t+1} / Σ r_t²``; ψ̂ is the mean over the
    subcarriers that are finite throughout the block. A block that spans a
    frame gap (any interval over twice the median) is NaN rather than a
    statistic of two different stretches. Returns the block centres and ψ̂.

    Under WiDetect's model ψ̂ is N(−1/T, 1/(F·T)) with no motion, whatever the
    noise variance; on this link the F is effective, not nominal.
    """
    X = np.asarray(X, dtype=float)
    times = np.asarray(times, dtype=float)
    if X.ndim != 2 or X.shape[0] != times.shape[0]:
        raise ValueError(f"X {X.shape} and times {times.shape} must share the frame axis")
    T = int(T)
    if T < 4:
        raise ValueError(f"an autocorrelation window needs at least 4 frames, got {T}")
    nb = X.shape[0] // T
    if nb == 0:
        return np.zeros(0), np.zeros(0)
    blocks = X[: nb * T].reshape(nb, T, X.shape[1])
    tb = times[: nb * T].reshape(nb, T)

    t = np.arange(T, dtype=float)
    A = np.stack([t, np.ones(T)], axis=1)
    P = np.eye(T) - A @ np.linalg.pinv(A)              # residual projector
    ok = np.isfinite(blocks).all(axis=1)                # (nb, F)
    R = np.einsum("ij,bjk->bik", P, np.nan_to_num(blocks))
    R = np.where(ok[:, None, :], R, 0.0)
    num = (R[:, :-1, :] * R[:, 1:, :]).sum(axis=1)
    den = (R * R).sum(axis=1)
    valid = ok & (den > 0)
    phi = np.where(valid, num / np.where(den > 0, den, 1.0), np.nan)
    count = valid.sum(axis=1)
    psi = np.full(nb, np.nan)
    enough = count >= 4
    if enough.any():
        psi[enough] = np.nanmean(np.where(valid, phi, np.nan)[enough], axis=1)

    dt = np.diff(times)
    med = float(np.median(dt)) if dt.size else 0.0
    if med > 0 and T > 1:
        gaps = np.diff(tb, axis=1).max(axis=1) > 2.0 * med
        psi[gaps] = np.nan
    return tb.mean(axis=1), psi


def unit_edges(t0: float, t1: float, unit_seconds: float) -> np.ndarray:
    """Equal units across the range, as many as fit nearest ``unit_seconds``.

    A one-minute capture is one unit whatever its exact length; a five-minute
    capture is five. Equal division rather than a fixed length with a
    remainder, so every unit in a range is comparable to the others.
    """
    span = float(t1) - float(t0)
    if span <= 0 or unit_seconds <= 0:
        raise ValueError("the range and the unit length must be positive")
    n = max(1, int(round(span / float(unit_seconds))))
    return np.linspace(float(t0), float(t1), n + 1)


# --------------------------------------------------------------------------- #
#  Over a capture                                                             #
# --------------------------------------------------------------------------- #


def ratio_level(
    path,
    t0: float,
    t1: float,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
) -> dict[str, Any]:
    """The complex CSI ratio, per frame and subcarrier, on the uniform frame set.

    The same selection as the frame step's (``framediff._uniform_selection``:
    one transmitter, one MIMO mode, full width), so a feature here and the
    step on the Hybrid 2 tab are read from the same frames. Cached; the decode
    dominates and everything downstream is cheap.
    """
    from backend.framediff import _uniform_selection
    from backend.presence import complex_ratio
    from backend.tiles import _decode_for_doppler, get_index

    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns, float(t0), float(t1),
           mimo, source_mac, bool(interpolate))
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
            f"fewer than 2 frames in range once the set was made uniform ({sel['note']})"
        )
    amp_db = _decode_for_doppler(path, index, ids, "csi_ratio_amplitude", None, interpolate)
    phase = _decode_for_doppler(path, index, ids, "csi_ratio_phase", None, interpolate)
    ratio = complex_ratio(amp_db, phase)
    # Subcarriers that are null in every frame (structural, with interpolation
    # off) carry nothing; the ones that drop out now and then are handled per
    # window by acf_windows.
    ever = np.isfinite(ratio).any(axis=0)
    rssi = np.asarray(getattr(index, "rssi_1", None))
    gain = rssi[ids] if rssi.ndim == 1 and rssi.size > int(ids[-1]) else None

    out = {
        "time_s": times_all[ids],
        "ratio": ratio[:, ever],
        "n_subcarriers": int(ever.sum()),
        "gain_state": gain,
        "source_mac": sel["source_mac"],
        "mimo": sel["mimo"],
        "selection_note": sel["note"],
        "frames_used": int(ids.size),
        "frames_dropped": int((in_range & ~sel["mask"]).sum()),
    }
    with _cache_lock:
        _cache[key] = out
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return out


def _quantiles(values: np.ndarray) -> tuple[float, float, int]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan"), float("nan"), 0
    return float(np.median(v)), float(np.percentile(v, 90)), int(v.size)


def compute_features(
    path,
    t0: float,
    t1: float,
    *,
    unit_seconds: float = UNIT_SECONDS,
    acf_frames: int = ACF_WINDOW_FRAMES,
    lag_seconds: float = STEP_LAG_SECONDS,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
) -> dict[str, Any]:
    """Every feature in the bank, per unit of the range, plus the series behind them.

    Times are on the capture's clock. ``units`` carries one dict per unit
    with a field per ``FEATURES`` key (NaN where the unit had nothing to
    measure); ``acf`` the per-window ψ̂ series on amplitude and phase;
    ``step`` the per-second lag step the range rule reads.
    """
    lvl = ratio_level(path, t0, t1, mimo=mimo, source_mac=source_mac, interpolate=interpolate)
    times = lvl["time_s"]
    ratio = lvl["ratio"]
    dt = np.diff(times)
    fs = 1.0 / float(np.median(dt)) if dt.size and np.median(dt) > 0 else float("nan")

    amp = np.abs(ratio)
    ang = np.angle(ratio)
    missing = ~np.isfinite(ang)
    phase = np.unwrap(np.where(missing, 0.0, ang), axis=0)
    phase[missing] = np.nan

    centres, psi_amp = acf_windows(amp, times, acf_frames)
    _, psi_phase = acf_windows(phase, times, acf_frames)

    # The step the range rule reads, on whole seconds of the capture clock.
    seconds = np.arange(np.floor(float(t0)), float(t1), 1.0)
    level, steps = hybrid2.motion_per_second(
        path, t0, t1, seconds, mimo=mimo, source_mac=source_mac,
        interpolate=interpolate, gate_gain=False, lag_seconds=lag_seconds,
    )
    level = np.asarray(level, dtype=float)
    # Test 2: the same step at one frame of lag -- the link's own jitter, which a
    # still occupant does not move. Read on the same seconds.
    jitter, _ = hybrid2.motion_per_second(
        path, t0, t1, seconds, mimo=mimo, source_mac=source_mac,
        interpolate=interpolate, gate_gain=False, lag_seconds=0.0,
    )
    jitter = np.asarray(jitter, dtype=float)

    # Test 3: the breathing channel, through hybrid2 so the 10 s run is the
    # range rule's own number; the 20 s run adds the rate floor.
    breath: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for W in (hybrid.BREATH_WINDOW_SECONDS, BREATH_WINDOW_LONG_SECONDS):
        try:
            h = hybrid2.compute_hybrid2(
                path, t0, t1, mimo=mimo, source_mac=source_mac, interpolate=interpolate,
                lag_seconds=lag_seconds, breath_window_seconds=W,
            )
            breath[W] = (np.asarray(h["time_s"], float), np.asarray(h["breath_peak"], float),
                         np.asarray(h["breath_rpm"], float))
        except ValueError:
            breath[W] = (np.zeros(0), np.zeros(0), np.zeros(0))

    def _breath_run(W: float, a: float, b: float, floor_rpm: float | None) -> tuple[int, float]:
        ts, pk, rpm = breath[W]
        sel = (ts >= a) & (ts <= b)
        q = np.isfinite(pk[sel]) & (pk[sel] >= BREATH_PEAK)
        if floor_rpm is not None:
            q &= rpm[sel] >= floor_rpm
        run = hybrid2.longest_run(q)
        med = float(np.nanmedian(rpm[sel][q])) if q.any() else float("nan")
        return int(run), med

    gain = lvl["gain_state"]
    # Units span the frames, not the request: a range asked wider than the
    # capture would otherwise put its last units over nothing and cut the
    # capture's own minute short of its end.
    lo, hi = max(float(t0), float(times[0])), min(float(t1), float(times[-1]))
    edges = unit_edges(lo, hi, unit_seconds) if hi > lo else np.array([float(t0), float(t1)])
    units: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        wsel = (centres >= a) & ((centres <= b) if last else (centres < b))
        ssel = (seconds + 0.5 >= a) & ((seconds + 0.5 <= b) if last else (seconds + 0.5 < b))
        fsel = (times >= a) & ((times <= b) if last else (times < b))
        a_med, a_p90, n_win = _quantiles(psi_amp[wsel])
        p_med, p_p90, _ = _quantiles(psi_phase[wsel])
        s_med, s_p90, n_sec = _quantiles(level[ssel])
        j = jitter[ssel]
        j = j[np.isfinite(j)]
        j_p20 = float(np.percentile(j, 20)) if j.size else float("nan")
        g = gain[fsel] if gain is not None else None
        units.append({
            "t0": float(a), "t1": float(b),
            "n_windows": n_win, "n_seconds": n_sec, "n_frames": int(fsel.sum()),
            "step_p90": s_p90, "step_p50": s_med,
            "lag1_p20": j_p20,
            "step_norm": s_p90 / j_p20 if np.isfinite(j_p20) and j_p20 > 0 else float("nan"),
            "breath_run": _breath_run(hybrid.BREATH_WINDOW_SECONDS, a, b, None)[0],
            "breath_run20": (br20 := _breath_run(BREATH_WINDOW_LONG_SECONDS, a, b, BREATH_RATE_FLOOR_RPM))[0],
            "breath_rpm": br20[1],
            "acf_amp_median": a_med, "acf_amp_p90": a_p90,
            "acf_phase_median": p_med, "acf_phase_p90": p_p90,
            "gain_crossings": int(np.sum(g[1:] != g[:-1])) if g is not None and g.size > 1 else None,
            "rssi_median": float(np.median(g)) if g is not None and g.size else None,
        })

    return {
        "acf": {
            "time_s": centres, "amp": psi_amp, "phase": psi_phase,
            "window_frames": int(acf_frames),
            "window_seconds": float(acf_frames / fs) if np.isfinite(fs) else float("nan"),
            "null_mean": -1.0 / float(acf_frames),
        },
        "step": {
            "time_s": seconds + 0.5, "level": level,
            "lag_seconds": float(lag_seconds), "lag_frames": int(steps.get("lag_frames", 1)),
        },
        "units": units,
        "unit_seconds": float(edges[1] - edges[0]),
        "fs_hz": fs,
        "n_subcarriers": lvl["n_subcarriers"],
        "frames_used": lvl["frames_used"],
        "frames_dropped": lvl["frames_dropped"],
        "selection_note": lvl["selection_note"],
        "source_mac": lvl["source_mac"],
        "mimo": lvl["mimo"],
        "features": FEATURES,
    }
