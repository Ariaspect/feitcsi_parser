"""AF8: the A+F 8-feature presence classifier. 60 s of CSI in, P(person) out.

    python3 -m backend.af8 CAPTURE.bin            # prints 1 (person) or 0
    python3 -m backend.af8 CAPTURE.bin --json     # and the features behind it

The pipeline of the AF8 implementation guide (2026-10-07), held to its
reference implementation (``af8_reference/sp_light.py``, ``sp_final3.py``)
line for line -- where the two disagree, the code that produced the model
wins, and the disagreement is noted where it falls:

1. frames: dominant AP, 2x1, full width (``framediff._uniform_selection``),
   decoded as complex ``H`` for the AP's two transmit chains;
2. subcarriers carrying both chains in more than 90 % of frames, then every
   8th of those (31 of the 245 live ones) -- thinned *after* the live test,
   so no dead bin is picked;
3. the ratio ``r = H1 / H0``, resampled onto a uniform grid as two real planes;
4. **A, motion (4):** the step ``|r_t - r_{t-L}| / (|r_t| + |r_{t-L}|)``,
   median over subcarriers. Its time median ``D(tau)`` at 0.25, 2 and 5 s gives
   the shape -- the log-log slope and two ratios, all scale-free -- and at
   2 s, folded to 1 s blocks by the *mean*, its 90th percentile;
5. **F, breathing (4):** the FarSense sweep exactly as hybrid 2 runs it, on
   the 31-subcarrier grid: max and median window peak, the longest run of
   windows with peak >= 0.25, and the spread of their rates;
6. median impute, standardise, logistic regression; person when p > 0.5.

The model is the guide's final fit on all 705 labelled windows (§5) and is
plain arithmetic here -- no sklearn -- so this runs where ``hybrid2_calc``
does: NumPy and the pure backend modules, no scipy, no CSIKit. On the LG
board that is its own 32-bit Python and NumPy 1.26.4, ~1.6 s and 85 MB a
one-minute capture (docs/board_calculator.md).

Not hybrid 2's rule and not its numbers: ``A_p90`` is a mean-folded P90 on a
resampled grid, ~4-5 % below hybrid 2's median-folded one, and 0.035 / 0.25
mean nothing to this model.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from backend import doppler, farsense, framediff, mtk

FEATURES = ("A_slope3", "A_r025_2", "A_r5_2", "A_p90",
            "F_pkmax", "F_pkmed", "F_run", "F_rpm_sd")

# The guide's §5: SimpleImputer(median) -> StandardScaler -> LogisticRegression
# (L2, C=1, class_weight="balanced"), fitted on all 705 windows, sklearn 1.9.
IMPUTER_MEDIAN = np.array([0.037603, 0.934161, 1.042473, 0.020831, 0.29405, 0.088733, 1.0, 1.455456])
SCALER_MEAN = np.array([0.08173, 0.866621, 1.090836, 0.058477, 0.319959, 0.121962, 4.153191, 1.659002])
SCALER_SCALE = np.array([0.108607, 0.160828, 0.14738, 0.080433, 0.106294, 0.095455, 8.496262, 1.141405])
COEF = np.array([0.473535, -0.30626, 1.072024, 3.11185, 1.250435, 2.520094, 1.489033, 0.251582])
INTERCEPT = -0.113493
THRESHOLD = 0.5

LIVE_FRACTION = 0.9       # a subcarrier must carry both chains in > 90 % of frames
STRIDE = 8                # every 8th live subcarrier: 31 of 245
SHAPE_LAGS = (0.25, 2.0, 5.0)
P90_LAG = 2.0
BREATH_PEAK = 0.25
MIN_SECONDS = 30.0        # less usable grid than this and the window is not judged
# A one-minute capture is one window, [0, 61) s from its first frame -- the
# guide's §6. (The reference inventory ended those windows at the *camera's*
# span, 58-59 s, which no board has. Over the 514 one-minute captures the
# two windows change no verdict -- docs/board_calculator.md.)
WINDOW_SECONDS = 61.0

# The oldest NumPy this was checked on: the board's own (see main).
MIN_NUMPY = (1, 26)


def load_ratio(path: Path, index: mtk.MTKIndex, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """``sp_light.raw``: the ratio over the live subcarriers, and its frame times.

    *t0*, *t1* are seconds from the capture's first frame, half-open.
    """
    t = np.asarray(index.times, dtype=float)
    tr = t - t[0]
    sel = framediff._uniform_selection(index, None, None)
    ids = np.flatnonzero(np.asarray(sel["mask"], bool) & (tr >= t0) & (tr < t1))
    if ids.size < 2:
        raise ValueError(f"fewer than 2 frames in [{t0:g}, {t1:g}) s once the set was made uniform")
    H = mtk.decode_complex(path, index, ids)[:, :2, :]
    live = np.isfinite(H).all(axis=1).mean(axis=0) > LIVE_FRACTION
    H = H[:, :, live]
    tt = t[ids]
    keep = np.isfinite(H).all(axis=(1, 2)) & (np.abs(H[:, 0, :]).min(axis=1) > 0)
    return H[keep, 1, :] / H[keep, 0, :], tt[keep]


def resample(r: np.ndarray, tt: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """``sp_light.resample``: the uniform complex grid, its invented-sample mask, and fs."""
    _, fs = doppler.uniform_grid(tt)
    n = int(np.floor((tt[-1] - tt[0]) * fs)) + 1
    gt = tt[0] + np.arange(n) / fs
    gl = doppler.gap_limit_for(tt)
    re, fab = doppler.resample_uniform(tt, r.real, gt, gl)
    im, _ = doppler.resample_uniform(tt, r.imag, gt, gl)
    return re + 1j * im, fab, fs


def _step(g: np.ndarray, L: int) -> np.ndarray:
    """Per-sample step at a lag of *L* samples, median over subcarriers."""
    d = np.abs(g[L:] - g[:-L]) / (np.abs(g[L:]) + np.abs(g[:-L]))
    return np.nanmedian(d, axis=1)


def motion_features(g: np.ndarray, fs: float) -> dict[str, float]:
    """A: the structure function's shape at three lags, and the 2 s P90."""
    D = {tau: float(np.nanmedian(_step(g, max(1, int(round(tau * fs)))))) for tau in SHAPE_LAGS}
    slope = float(np.polyfit(np.log(SHAPE_LAGS), np.log([D[t] for t in SHAPE_LAGS]), 1)[0])
    s = _step(g, int(round(P90_LAG * fs)))
    k = int(round(fs))
    m = len(s) // k
    per_sec = np.nanmean(s[: m * k].reshape(m, k), axis=1)   # 1 s blocks, MEAN, tail dropped
    per_sec = per_sec[np.isfinite(per_sec)]
    return {"A_slope3": slope, "A_r025_2": D[0.25] / D[2.0], "A_r5_2": D[5.0] / D[2.0],
            "A_p90": float(np.percentile(per_sec, 90))}


def breath_features(g: np.ndarray, fab: np.ndarray, fs: float) -> dict[str, float]:
    """F: the FarSense sweep with hybrid 2's settings, on this grid."""
    prep = farsense.prepare(g, fs, fabricated=fab, window_seconds=10.0, hop_seconds=1.0,
                            band_rpm=(10.0, 30.0), savgol_seconds=1.0, savgol_order=3,
                            highpass_hz=0.0)
    fz = farsense._run(prep, dict(n_theta=200, fft_size=8192, keep_fraction=0.65,
                                  motion_frac_hi=1e9, max_gap_fraction=0.5, min_peak=-1.0,
                                  positive_only=True), None)
    pk = np.where(fz["unknown"], np.nan, fz["acf_peak_norm"]).astype(float)
    rpm = np.asarray(fz["rpm"], float)
    good = np.isfinite(pk) & (pk >= BREATH_PEAK)
    best = cur = 0
    for gi in good:
        cur = cur + 1 if gi else 0
        best = max(best, cur)
    return {"F_pkmax": float(np.nanmax(pk)), "F_pkmed": float(np.nanmedian(pk)),
            "F_run": best * 1.0,
            "F_rpm_sd": float(np.nanstd(rpm[good])) if good.sum() >= 2 else float("nan")}


def probability(features: dict[str, float]) -> float:
    """The fitted pipeline as arithmetic: impute, standardise, logistic."""
    x = np.array([features[f] for f in FEATURES], dtype=float)
    x = np.where(np.isnan(x), IMPUTER_MEDIAN, x)
    z = (x - SCALER_MEAN) / SCALER_SCALE
    return float(1.0 / (1.0 + np.exp(-(INTERCEPT + COEF @ z))))


def evaluate(path: str | Path, t0: float = 0.0, t1: float = WINDOW_SECONDS) -> dict[str, Any]:
    """Judge one window. ``label`` is 1 for a person, 0 for no person.

    Raises ``ValueError`` when the window is not judged: too little usable
    data, a FarSense window that cannot form, or a feature other than
    ``F_rpm_sd`` (NaN there just means no breathing) coming out NaN.
    """
    t_start = time.perf_counter()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such capture: {path}")
    if not mtk.can_read(path):
        raise ValueError(f"{path.name} is not a MediaTek capture")
    # Stage times in ms, so a run elsewhere -- the board -- can be read against
    # the guide's §8-3 (resample / A / F / model), with the file decode apart:
    # a live stream has no file to decode.
    ms: dict[str, float] = {}
    lap = time.perf_counter()

    def mark(stage: str) -> None:
        nonlocal lap
        now = time.perf_counter()
        ms[stage] = round((now - lap) * 1000.0, 2)
        lap = now

    index = mtk.MTKIndex(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN slices, as the reference
        r, tt = load_ratio(path, index, t0, t1)
        mark("decode")
        if tt.size < 2:
            raise ValueError("no frame in the window carries a usable ratio")
        g, fab, fs = resample(r[:, ::STRIDE], tt)
        mark("resample")
        if len(g) < MIN_SECONDS * fs:
            raise ValueError(f"only {len(g) / fs:.1f} s of usable grid, under {MIN_SECONDS:g} s")
        feats = motion_features(g, fs)
        mark("A")
        feats.update(breath_features(g, fab, fs))
        mark("F")
    missing = [f for f in FEATURES if f != "F_rpm_sd" and not np.isfinite(feats[f])]
    if missing:
        raise ValueError(f"not judged: {', '.join(missing)} undefined")
    p = probability(feats)
    mark("model")
    return {
        "file": path.name,
        "label": int(p > THRESHOLD),
        "p_person": p,
        "features": {f: (None if not np.isfinite(feats[f]) else feats[f]) for f in FEATURES},
        "window_s": [float(t0), float(t1)],
        "seconds": round(len(g) / fs, 3),
        "subcarriers": int(g.shape[1]),
        "fs_hz": float(fs),
        "numpy": np.__version__,
        "elapsed_s": round(time.perf_counter() - t_start, 3),
        "stage_ms": ms,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="af8", description="AF8 presence verdict: prints 1 (person) or 0 (no person).")
    ap.add_argument("capture", type=Path, help="MediaTek .bin capture, about one minute long")
    ap.add_argument("--t0", type=float, default=0.0, help="window start, s from the first frame")
    ap.add_argument("--t1", type=float, default=WINDOW_SECONDS, help="window end (exclusive)")
    ap.add_argument("--json", action="store_true", help="print p(person) and the features as JSON")
    args = ap.parse_args(argv)

    # Checked on NumPy 2.5.2 (x86-64) and the board's 1.26.4 (32-bit ARM): all
    # 256 environment-2 windows agree to the verdict, p within 2.4e-6. Older is
    # untested, so it stops rather than answers.
    if tuple(int(v) for v in np.__version__.split(".")[:2]) < MIN_NUMPY:
        print(f"af8: needs NumPy >= {MIN_NUMPY[0]}.{MIN_NUMPY[1]}, got {np.__version__} "
              f"from {Path(np.__file__).parent}", file=sys.stderr)
        return 3
    try:
        out = evaluate(args.capture, args.t0, args.t1)
    except (OSError, ValueError) as exc:
        print(f"af8: {args.capture}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out) if args.json else out["label"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
