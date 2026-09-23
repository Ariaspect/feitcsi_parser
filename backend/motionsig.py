"""Motion from amplitude alone: the two-feature signal the stage-1/2 experiment settled on.

A separate question from the hybrid detector next door. That one asks *is
someone there*, and answers it from motion **or** breathing. This one asks a
narrower thing -- **how much is the channel moving right now** -- and answers
it as one scalar per window, meant to be handed to a back-end classifier
later. Nothing here holds a verdict open, and nothing here looks for a
rhythm.

What the experiment fixed (see ``report_motion_signal.md``):

* **Signal source: RATIO**, ``|H_tx1 / H_tx0|`` -- the amplitude of the
  complex ratio along the AP's transmit chains, the same grid the presence
  detector rides on (``tiles._presence_grid``), so a change of decode cannot
  masquerade as motion. AUC 0.993 against 0.983 for raw amplitude on the
  walking scenario, and the plan's expected "RAW-PC1 is the AGC component"
  never appeared: no candidate source correlated with common gain above 0.29.
  A single PC of the ratio scored within 0.02 of the full 245 streams on
  walking, which by the plan's own tie-break would have won on being
  lighter -- but it loses by 0.084 on *small* motion, which is the case that
  matters, so the full stream set is kept.
* **Features: variance and lag-1 autocorrelation.** Variance on the
  high-passed window; lag-1 on the window before the high-pass, mean removed
  only, because the filter manufactures correlation in noise. Both reduced
  over subcarriers by the median. Six other candidates (MAD, kurtosis,
  skewness, low-band ratio, spectral entropy, and mean as a control) earned
  nothing on top.
* **Normalisation is per capture, against that capture's own quiet level.**
  This is the part that was a methodology error first and a result second:
  pooling the empty windows of every capture into one negative class measured
  the *link*, not the room. Empty-room variance spans 36x between sample
  rates, and the seated-with-phone and still-posture groups swap order
  (0.575/0.868 pooled, 0.860/0.527 self-referenced) when the pooling is
  removed. So nothing here is comparable across captures until it has been
  divided by its own capture's scale.

Two normalisations are offered, and the difference between them is the whole
deployment question:

* ``label`` centres and scales each feature on the windows the *camera* calls
  empty. It is what every number in the report was measured with, and it
  cannot be deployed -- there is no camera in a product.
* ``free`` uses the capture's own 10th and 40th percentiles instead, no
  labels. The pair has to sit low: at the 20th/80th the occupant of a capture
  40% occupied is inside the upper percentile and scales the signal away --
  walking recall collapsed from 100% to 25.8% that way, the same self-defeat
  the hybrid's motion floor has when a range is mostly motion.

The logistic weights are **fixed**, fitted once over the 29 marked captures
(``CORPUS``) under each normalisation, with the threshold at 90% specificity
on that corpus's empty windows. They are constants here, not something
re-fitted per request: a tab that re-fits on the capture it is displaying is
reporting its own training error. For the 29 captures the weights were fitted
on, the panel says so.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

from backend import truth as truthmod

# Window geometry, fixed by the experiment plan and not swept.
WINDOW_SECONDS = 2.0
HOP_SECONDS = 0.5
# DC removal before the variance. The plan's value; at 20 Hz it is already
# 1.5% of Nyquist, so it is a DC notch rather than a band choice.
HIGHPASS_HZ = 0.3
HIGHPASS_ORDER = 4
# Hampel outlier replacement, half-width in samples and the MAD multiple.
HAMPEL_HALF = 5
HAMPEL_NSIG = 3.0
# A window more than this fraction interpolated across dropouts reports
# nothing rather than reporting the interpolator's own smoothness as calm.
MAX_GAP_FRACTION = 0.5
# Percentile pair for the label-free scale. Both must sit below the occupant.
FREE_PERCENTILES = (10.0, 40.0)
# Empty windows a capture needs before its own empty statistics are trusted.
MIN_REFERENCE_WINDOWS = 10

FEATURES = ("variance", "lag1")

# Fitted on CORPUS, class-weight balanced, C = 1, threshold at the 90th
# percentile of the empty windows' score. Reproduce with
# ``scripts/fit_motionsig.py``.
COEFFICIENTS: dict[str, dict[str, float]] = {
    "label": {
        "variance": 0.135084,
        "lag1": 0.629663,
        "intercept": -0.578889,
        "threshold": 0.321564,
    },
    "free": {
        "variance": 0.003270,
        "lag1": 0.630147,
        "intercept": -1.059958,
        "threshold": 0.439962,
    },
}

MODES = tuple(COEFFICIENTS)

# The captures the weights were fitted on. A panel showing one of these is
# showing training error, which is worth saying out loud.
CORPUS = frozenset("""
20260904_193228 20260909_195450 20260910_203337 20260911_095127 20260911_100145
20260911_140353 20260911_142929 20260911_144658 20260911_150435 20260911_153305
20260914_193002 20260915_120831 20260915_132712 20260915_133849 20260915_143211
20260915_150502 20260915_195845 20260915_202020 20260916_140908 20260916_143259
20260916_202702 20260917_195828 20260917_202029 20260921_121406 20260921_125836
20260921_133234 20260922_151145 20260922_155219 20260922_161219
""".split())


# --------------------------------------------------------------------------- #
#  Preprocessing                                                              #
# --------------------------------------------------------------------------- #


def hampel(x: np.ndarray, half: int = HAMPEL_HALF, nsig: float = HAMPEL_NSIG,
           chunk: int = 32) -> np.ndarray:
    """Replace only the samples that stray from their local median. Along axis 0.

    Chunked over streams because the sliding window is ``(T, S, 2*half+1)``
    and a 600 s range at 43 Hz over 245 subcarriers would be half a gigabyte
    of it at once.
    """
    if x.shape[0] < 2 * half + 1:
        return x.copy()
    out = x.copy()
    for lo in range(0, x.shape[1], chunk):
        block = x[:, lo:lo + chunk]
        pad = np.pad(block, ((half, half), (0, 0)), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1, axis=0)
        med = np.median(win, axis=-1)
        mad = np.median(np.abs(win - med[..., None]), axis=-1) * 1.4826
        stray = np.abs(block - med) > nsig * np.maximum(mad, 1e-12)
        blk = out[:, lo:lo + chunk]
        blk[stray] = med[stray]
    return out


def _window_moments(x: np.ndarray, starts: np.ndarray, n: int) -> tuple[np.ndarray, ...]:
    """Per-window sums of ``x``, ``x^2`` and ``x_t x_{t+1}``, from cumulative sums.

    The features wanted here are moments, so no window is ever materialised:
    a 600 s range would otherwise hold ``(n_windows, n_samples, n_streams)``
    at once. Cost is one pass over the range per quantity.
    """
    c1 = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    c2 = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x * x, axis=0)])
    prod = x[:-1] * x[1:]
    cp = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(prod, axis=0)])
    w1 = c1[starts + n] - c1[starts]
    w2 = c2[starts + n] - c2[starts]
    # Pairs inside the window: t from start to start + n - 2.
    wp = cp[starts + n - 1] - cp[starts]
    return w1, w2, wp


def features(pre: np.ndarray, post: np.ndarray, starts: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Variance and lag-1 per window per stream, each ``(n_windows, n_streams)``.

    Variance comes off the high-passed signal, lag-1 off the signal before
    it -- a high-pass leaves neighbouring noise samples anticorrelated, so
    lag-1 on a filtered empty room reads as structure that is the filter's.
    """
    w1p, w2p, _ = _window_moments(post, starts, n)
    mu_post = w1p / n
    variance = np.maximum(w2p / n - mu_post * mu_post, 0.0)

    w1, w2, wp = _window_moments(pre, starts, n)
    mu = w1 / n
    # sum (x_t - mu)^2 over the window, and sum (x_t - mu)(x_{t+1} - mu) over
    # its n-1 adjacent pairs, both expanded so only the sums above are needed.
    den = w2 - n * mu * mu
    edge = pre[starts] + pre[starts + n - 1]
    num = wp - mu * (2.0 * w1 - edge) + (n - 1) * mu * mu
    lag1 = num / np.maximum(den, 1e-30)
    return {"variance": variance, "lag1": lag1}


# --------------------------------------------------------------------------- #
#  Normalisation and scoring                                                  #
# --------------------------------------------------------------------------- #


def reference(values: dict[str, np.ndarray], mode: str,
              empty: np.ndarray | None = None) -> dict[str, tuple[float, float]]:
    """Per-feature ``(centre, scale)`` for one capture under one normalisation.

    ``label``: the median and standard deviation of the windows the camera
    calls empty -- what the report measured, and not deployable. ``free``: the
    capture's own low percentiles, which need no labels; see the module note
    on why the pair has to be low.
    """
    out: dict[str, tuple[float, float]] = {}
    for f in FEATURES:
        v = np.asarray(values[f], dtype=float)
        finite = np.isfinite(v)
        if not finite.any():
            out[f] = (0.0, 1.0)
            continue
        if mode == "label":
            pool = v[finite & empty] if empty is not None else v[:0]
            if pool.size >= MIN_REFERENCE_WINDOWS:
                centre, scale = float(np.median(pool)), float(np.std(pool))
            else:
                centre, scale = float(np.median(v[finite])), float(np.std(v[finite]))
        else:
            lo, hi = np.percentile(v[finite], FREE_PERCENTILES)
            centre, scale = float(lo), float(max(hi - lo, 1e-30))
        out[f] = (centre, max(scale, 1e-30))
    return out


def score(values: dict[str, np.ndarray], ref: dict[str, tuple[float, float]],
          coef: dict[str, float]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Normalised features and the linear score built from them."""
    z = {f: (np.asarray(values[f], dtype=float) - ref[f][0]) / ref[f][1] for f in FEATURES}
    s = np.full(next(iter(z.values())).shape, np.nan)
    ok = np.ones(s.shape, dtype=bool)
    for f in FEATURES:
        ok &= np.isfinite(z[f])
    s[ok] = coef["intercept"] + sum(coef[f] * z[f][ok] for f in FEATURES)
    return z, s


# --------------------------------------------------------------------------- #
#  Over a capture                                                             #
# --------------------------------------------------------------------------- #

# Decoding and filtering dominate; the normalisations and the linear score
# are free, so one decode serves both modes and repeat requests on the same
# range (a changed margin, a re-render) cost nothing.
_CACHE_SIZE = 4
_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
_cache_lock = Lock()


def reset_cache() -> None:
    with _cache_lock:
        _cache.clear()


def capture_features(
    path,
    t0: float,
    t1: float,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
    highpass_hz: float = HIGHPASS_HZ,
    max_gap_fraction: float = MAX_GAP_FRACTION,
) -> dict[str, Any]:
    """Decode a range and reduce it to the two raw features per window, cached.

    Window centres are on the capture's clock, so they line up with the
    camera sidecar and with every other tab.
    """
    from backend.tiles import _presence_grid

    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns, float(t0), float(t1),
           mimo, source_mac, bool(interpolate),
           float(window_seconds), float(hop_seconds), float(highpass_hz),
           float(max_gap_fraction))
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit

    grid, fabricated, fs, grid_times, times, _times_all, n_no_ratio = _presence_grid(
        path, t0, t1, mimo=mimo, source_mac=source_mac, interpolate=interpolate,
    )
    n = int(round(window_seconds * fs))
    hop = max(1, int(round(hop_seconds * fs)))
    if n < 8:
        raise ValueError(f"{window_seconds} s is {n} samples at {fs:.1f} Hz; too few for a window")
    if grid.shape[0] < n:
        raise ValueError(f"range holds {grid.shape[0]} samples, one window needs {n}")

    # RATIO: the magnitude of the complex ratio. Dead subcarriers out, then
    # each stream divided by its own median so the 245 of them are one
    # population before the median over streams reduces them.
    amp = np.abs(grid).astype(np.float64)
    # `alive` first, so the median never runs over an all-NaN dead subcarrier.
    alive = np.isfinite(amp).all(axis=0)
    if alive.any():
        alive[alive] &= np.median(amp[:, alive], axis=0) > 0
    if alive.sum() < 4:
        raise ValueError(f"only {int(alive.sum())} subcarriers carry a usable ratio")
    amp = amp[:, alive]
    amp /= np.median(amp, axis=0)

    pre = hampel(amp)
    b, a = butter(HIGHPASS_ORDER, highpass_hz / (fs / 2.0), btype="high")
    post = filtfilt(b, a, pre, axis=0)

    starts = np.arange(0, pre.shape[0] - n + 1, hop)
    raw = features(pre, post, starts, n)

    # Median over subcarriers, and a window mostly interpolated says nothing.
    fab = np.asarray(fabricated, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(fab)])
    gap = (c[starts + n] - c[starts]) / n
    blank = gap > max_gap_fraction
    out: dict[str, Any] = {
        "time_s": float(grid_times[0]) + (starts + (n - 1) / 2.0) / fs,
        "fs": float(fs),
        "n_samples": int(n),
        "streams": int(alive.sum()),
        "gap_fraction": gap,
        "frames_used": int(times.size),
        "frames_without_ratio": int(n_no_ratio),
        "window_seconds": float(window_seconds),
        "hop_seconds": float(hop_seconds),
        "highpass_hz": float(highpass_hz),
        "raw": {},
    }
    for f in FEATURES:
        v = np.median(raw[f], axis=1)
        v[blank] = np.nan
        out["raw"][f] = v

    with _cache_lock:
        _cache[key] = out
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return out


def compute_motion_signal(
    path,
    t0: float,
    t1: float,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
    highpass_hz: float = HIGHPASS_HZ,
    max_gap_fraction: float = MAX_GAP_FRACTION,
    margin_s: float = truthmod.DEFAULT_MARGIN_S,
    camera: np.ndarray | None = None,
) -> dict[str, Any]:
    """The adopted signal for one range, under both normalisations, scored.

    ``camera`` is the sidecar as ``(time, present)`` rows on the capture's
    clock; without it the label normalisation has nothing to centre on and
    reports why. Scoring goes through ``backend.truth``, so the empty seconds
    next to a transition are dropped here exactly as they are everywhere else
    -- note that this is the asymmetric rule, which is *not* what the
    experiment's own ``+/-5 s on every frame`` mask did.
    """
    feat = capture_features(
        path, t0, t1, mimo=mimo, source_mac=source_mac, interpolate=interpolate,
        window_seconds=window_seconds, hop_seconds=hop_seconds,
        highpass_hz=highpass_hz, max_gap_fraction=max_gap_fraction,
    )
    centres = np.asarray(feat["time_s"], dtype=float)
    half = float(window_seconds) / 2.0

    cells = excluded = None
    empty = None
    if camera is not None and camera.size:
        cells, excluded = truthmod.cell_truth(
            centres, camera[:, 0], camera[:, 1] > 0.5, half, margin_s,
        )
        empty = np.isfinite(cells) & (cells <= 0.5)

    modes: dict[str, Any] = {}
    for mode in MODES:
        coef = COEFFICIENTS[mode]
        note = None
        if mode == "label":
            if empty is None:
                note = "no camera sidecar: nothing to centre the empty reference on"
            elif int(empty.sum()) < MIN_REFERENCE_WINDOWS:
                note = (f"only {int(empty.sum())} camera-empty window(s) in range "
                        f"(needs {MIN_REFERENCE_WINDOWS}); centred on the whole range instead")
        ref = reference(feat["raw"], mode, empty)
        z, s = score(feat["raw"], ref, coef)
        present = np.isfinite(s) & (s > coef["threshold"])
        entry: dict[str, Any] = {
            "z": z,
            "score": s,
            "present": present,
            "threshold": float(coef["threshold"]),
            "reference": {f: {"centre": ref[f][0], "scale": ref[f][1]} for f in FEATURES},
            "coefficients": {f: float(coef[f]) for f in FEATURES} | {"intercept": float(coef["intercept"])},
            "note": note,
        }
        if cells is not None:
            conf = truthmod.confusion(cells, present, excluded)
            scored = np.isfinite(cells)
            conf["base_rate"] = float(np.mean(cells[scored] > 0.5)) if scored.any() else None
            conf["margin_s"] = float(margin_s)
            entry["confusion"] = conf
        modes[mode] = entry

    return feat | {
        "modes": modes,
        "truth": None if camera is None or not camera.size else {
            "time_s": camera[:, 0].astype(float),
            "present": camera[:, 1] > 0.5,
        },
        "in_corpus": Path(path).stem in CORPUS,
        "margin_s": float(margin_s),
    }
