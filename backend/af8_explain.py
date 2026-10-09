"""What the ML tab draws: AF8's eight features and the series each is read from.

Server-side only -- the board runs ``backend.af8`` and nothing here. The
features are ``af8``'s own (``af8.motion_features`` / ``af8.breath_features``
on ``af8``'s grid), so the tab can never show a number the model did not see.
The series behind them are recomputed with the same helpers (``af8._step``,
the FarSense sweep with ``af8``'s settings) for drawing only; the tests hold
the two together.

One window, as the model was trained:

* a capture shorter than 90 s is one window, ``[0, 61)`` s (``af8.evaluate``'s
  default);
* a longer one is cut into 60 s windows ``[60w, 60(w+1))``, as the training
  inventory cut the five-minute captures.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from backend import af8, farsense, mtk

# The model reads D(tau) at af8.SHAPE_LAGS only; the other lags are drawn as
# context, so a reader sees the curve the three points sit on.
CONTEXT_LAGS = (0.05, 0.1, 0.5, 1.0, 3.0, 8.0)


def _num(v: float) -> float | None:
    """A float for JSON, ``None`` for NaN/inf."""
    v = float(v)
    return v if np.isfinite(v) else None


def _nums(values) -> list[float | None]:
    return [_num(v) for v in np.asarray(values, dtype=float)]


def windows_for(duration_s: float) -> list[tuple[float, float]]:
    """The windows a capture is judged in, in seconds from its first frame."""
    if duration_s < 90.0:
        return [(0.0, af8.WINDOW_SECONDS)]
    return [(60.0 * w, 60.0 * (w + 1)) for w in range(int(duration_s // 60))]


def _per_second_median(values: np.ndarray, times: np.ndarray,
                       cells: np.ndarray) -> np.ndarray:
    """Median of *values* in each 1 s cell starting at *cells*; NaN where empty.

    For drawing only: the step series are ~2,800 points a minute, one median
    a second keeps the line readable. The features use the full series.
    """
    out = np.full(cells.size, np.nan)
    idx = np.floor(times - cells[0]).astype(int)
    ok = (idx >= 0) & (idx < cells.size) & np.isfinite(values)
    for c in np.unique(idx[ok]):
        out[c] = float(np.median(values[ok & (idx == c)]))
    return out


def explain(path: str | Path, window: int = 0) -> dict[str, Any]:
    """Features, model breakdown and drawing series for one window of a capture.

    Times in the result are seconds from the capture's first frame. Raises
    ``ValueError`` when the window cannot be judged, as ``af8.evaluate`` does.
    """
    path = Path(path)
    if not mtk.can_read(path):
        raise ValueError(f"{path.name} is not a MediaTek capture")
    index = mtk.MTKIndex(path)
    t_all = np.asarray(index.times, dtype=float)
    if t_all.size < 2:
        raise ValueError("fewer than 2 frames in the capture")
    origin = float(t_all[0])
    wins = windows_for(float(t_all[-1] - origin))
    if not 0 <= window < len(wins):
        raise ValueError(f"window {window} out of range: this capture has {len(wins)}")
    t0, t1 = wins[window]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # --- the model's own inputs, exactly as af8.evaluate builds them -----
        r, tt = af8.load_ratio(path, index, t0, t1)
        if tt.size < 2:
            raise ValueError("no frame in the window carries a usable ratio")
        g, fab, fs = af8.resample(r[:, ::af8.STRIDE], tt)
        if len(g) < af8.MIN_SECONDS * fs:
            raise ValueError(f"only {len(g) / fs:.1f} s of usable grid, under {af8.MIN_SECONDS:g} s")
        feats = af8.motion_features(g, fs)
        feats.update(af8.breath_features(g, fab, fs))

        grid_t = (tt[0] - origin) + np.arange(len(g)) / fs     # s from the first frame

        # --- motion: the structure function and the step series --------------
        lags = sorted(set(CONTEXT_LAGS) | set(af8.SHAPE_LAGS))
        D, step_s = {}, {}
        for tau in lags:
            L = max(1, int(round(tau * fs)))
            if L >= len(g):
                continue
            s = af8._step(g, L)
            D[tau] = float(np.nanmedian(s))
            if tau in af8.SHAPE_LAGS:
                step_s[tau] = (s, grid_t[L:])
        cells = np.arange(np.floor(grid_t[0]), np.ceil(grid_t[-1]))
        slope = feats["A_slope3"]
        fit_b = float(np.mean(np.log([D[t] for t in af8.SHAPE_LAGS]))
                      - slope * np.mean(np.log(af8.SHAPE_LAGS)))

        # A_p90's own folding: 1 s blocks of k samples from the first 2 s step,
        # the MEAN of each, the tail dropped -- af8.motion_features, drawn.
        Lp = int(round(af8.P90_LAG * fs))
        sp = af8._step(g, Lp)
        k = int(round(fs))
        m = len(sp) // k
        p90_vals = np.nanmean(sp[: m * k].reshape(m, k), axis=1)
        p90_t = grid_t[Lp + np.arange(m) * k + k // 2]

        # --- breathing: the FarSense sweep with af8's settings ---------------
        prep = farsense.prepare(g, fs, fabricated=fab, window_seconds=10.0, hop_seconds=1.0,
                                band_rpm=(10.0, 30.0), savgol_seconds=1.0, savgol_order=3,
                                highpass_hz=0.0)
        fz = farsense._run(prep, dict(n_theta=200, fft_size=8192, keep_fraction=0.65,
                                      motion_frac_hi=1e9, max_gap_fraction=0.5, min_peak=-1.0,
                                      positive_only=True), None)
        pk = np.where(fz["unknown"], np.nan, fz["acf_peak_norm"]).astype(float)
        rpm = np.asarray(fz["rpm"], float)
        win_t = (tt[0] - origin) + np.asarray(fz["time_s"], float)   # window centres
        good = np.isfinite(pk) & (pk >= af8.BREATH_PEAK)
        best = cur = 0
        run_end = -1
        for i, gi in enumerate(good):
            cur = cur + 1 if gi else 0
            if cur > best:
                best, run_end = cur, i
        run = None
        if best:
            a, b = run_end - best + 1, run_end
            run = {"windows": int(best), "t0": float(win_t[a]), "t1": float(win_t[b])}

    # --- the model, step by step (af8.probability, spelled out) ----------------
    x = np.array([feats[f] for f in af8.FEATURES], dtype=float)
    imputed = ~np.isfinite(x)
    x = np.where(imputed, af8.IMPUTER_MEDIAN, x)
    z = (x - af8.SCALER_MEAN) / af8.SCALER_SCALE
    contrib = af8.COEF * z
    logit = float(af8.INTERCEPT + contrib.sum())
    p = af8.probability(feats)

    good_rpm = rpm[good]
    return {
        "file": path.name,
        "window": int(window),
        "windows": [[float(a), float(b)] for a, b in wins],
        "window_s": [float(t0), float(t1)],
        "fs_hz": float(fs),
        "subcarriers": int(g.shape[1]),
        "seconds": round(len(g) / fs, 3),
        "p_person": p,
        "label": int(p > af8.THRESHOLD),
        "threshold": af8.THRESHOLD,
        "features": {f: _num(feats[f]) for f in af8.FEATURES},
        "model": {
            "features": list(af8.FEATURES),
            "value": _nums(x),
            "imputed": [bool(v) for v in imputed],
            "mean": _nums(af8.SCALER_MEAN),
            "scale": _nums(af8.SCALER_SCALE),
            "coef": _nums(af8.COEF),
            "z": _nums(z),
            "contribution": _nums(contrib),
            "intercept": float(af8.INTERCEPT),
            "logit": logit,
        },
        "motion": {
            "lags": [float(t) for t in D],
            "D": [float(v) for v in D.values()],
            "shape_lags": [float(t) for t in af8.SHAPE_LAGS],
            "fit_intercept": fit_b,
            "cells": [float(c) + 0.5 for c in cells],
            "step": {
                f"{tau:g}": _nums(_per_second_median(s, ts, cells))
                for tau, (s, ts) in step_s.items()
            },
            "p90_time": _nums(p90_t),
            "p90_values": _nums(p90_vals),
        },
        "breath": {
            "time": _nums(win_t),
            "peak": _nums(pk),
            "rpm": _nums(rpm),
            "good": [bool(v) for v in good],
            "threshold": af8.BREATH_PEAK,
            "run": run,
            "rpm_mean": _num(np.nanmean(good_rpm)) if np.isfinite(good_rpm).any() else None,
        },
    }
