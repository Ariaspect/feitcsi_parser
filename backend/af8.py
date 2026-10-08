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

How the functions fit together (``evaluate`` calls them in this order)::

    capture.bin
      -> load_ratio()        frames -> complex CSI -> ratio r = H1 / H0
      -> resample()          r on an evenly spaced time grid
      -> motion_features()   A_slope3, A_r025_2, A_r5_2, A_p90
      -> breath_features()   F_pkmax, F_pkmed, F_run, F_rpm_sd
      -> probability()       8 features -> p(person)
      -> label = 1 if p > 0.5 else 0
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

# Other modules of this package that do the heavy lifting:
#   mtk       -- reads the MediaTek capture file (frame index, complex CSI)
#   framediff -- picks one consistent set of frames (one AP, 2x1, full width)
#   doppler   -- puts unevenly timed frames onto an evenly spaced time grid
#   farsense  -- the FarSense breathing detector (used for the F features)
from backend import doppler, farsense, framediff, mtk

# --------------------------------------------------------------------------- #
#  The model                                                                   #
# --------------------------------------------------------------------------- #

# The eight inputs of the model, in the order the weights below expect.
# A_* describe motion, F_* describe breathing.
FEATURES = ("A_slope3", "A_r025_2", "A_r5_2", "A_p90",
            "F_pkmax", "F_pkmed", "F_run", "F_rpm_sd")

# The guide's §5: SimpleImputer(median) -> StandardScaler -> LogisticRegression
# (L2, C=1, class_weight="balanced"), fitted on all 705 windows, sklearn 1.9.
# Each array has one number per feature, in the FEATURES order:
#   IMPUTER_MEDIAN -- value used when a feature is missing (NaN)
#   SCALER_MEAN / SCALER_SCALE -- turn each feature into a z-score
#   COEF / INTERCEPT -- the logistic regression weights
IMPUTER_MEDIAN = np.array([0.037603, 0.934161, 1.042473, 0.020831, 0.29405, 0.088733, 1.0, 1.455456])
SCALER_MEAN = np.array([0.08173, 0.866621, 1.090836, 0.058477, 0.319959, 0.121962, 4.153191, 1.659002])
SCALER_SCALE = np.array([0.108607, 0.160828, 0.14738, 0.080433, 0.106294, 0.095455, 8.496262, 1.141405])
COEF = np.array([0.473535, -0.30626, 1.072024, 3.11185, 1.250435, 2.520094, 1.489033, 0.251582])
INTERCEPT = -0.113493
THRESHOLD = 0.5           # p(person) above this -> label 1

# --------------------------------------------------------------------------- #
#  Feature settings (must match what the model was trained with)               #
# --------------------------------------------------------------------------- #

LIVE_FRACTION = 0.9       # a subcarrier must carry both chains in > 90 % of frames
STRIDE = 8                # every 8th live subcarrier: 31 of 245
SHAPE_LAGS = (0.25, 2.0, 5.0)   # seconds; the three lags of the motion shape
P90_LAG = 2.0             # seconds; the lag of the motion P90
BREATH_PEAK = 0.25        # a FarSense window counts as "breathing" at peak >= this
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

    In plain words: read the frames of the window, keep only the ones that are
    all alike (same AP, 2x1, full bandwidth), and for every frame and every
    usable subcarrier divide the AP's second transmit chain by its first.
    Dividing cancels the random phase and gain that change from packet to
    packet, so what is left moves only when the room does.

    Returns ``(ratio, times)``: ``ratio`` is ``(n_frames, n_subcarriers)``
    complex, ``times`` the matching frame times in seconds.
    """
    # Frame times, and the same times counted from the first frame.
    t = np.asarray(index.times, dtype=float)
    tr = t - t[0]

    # One consistent set of frames: the dominant AP, 2x1, full width.
    sel = framediff._uniform_selection(index, None, None)
    ids = np.flatnonzero(np.asarray(sel["mask"], bool) & (tr >= t0) & (tr < t1))
    if ids.size < 2:
        raise ValueError(f"fewer than 2 frames in [{t0:g}, {t1:g}) s once the set was made uniform")

    # Complex CSI of those frames: (frames, transmit chain, subcarrier).
    # Slot 0 and slot 1 are the AP's two transmit chains.
    H = mtk.decode_complex(path, index, ids)[:, :2, :]

    # Keep the subcarriers where both chains are present in > 90 % of frames
    # (drops the DC bin, the guard band and other bins that are always empty).
    live = np.isfinite(H).all(axis=1).mean(axis=0) > LIVE_FRACTION
    H = H[:, :, live]
    tt = t[ids]

    # Drop any frame that still has a missing value or a zero in chain 0,
    # which would make the division below undefined.
    keep = np.isfinite(H).all(axis=(1, 2)) & (np.abs(H[:, 0, :]).min(axis=1) > 0)

    # The CSI ratio r = H1 / H0, per frame and subcarrier.
    return H[keep, 1, :] / H[keep, 0, :], tt[keep]


def resample(r: np.ndarray, tt: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """``sp_light.resample``: the uniform complex grid, its invented-sample mask, and fs.

    Frames arrive at uneven times. The features below compare samples a fixed
    number of seconds apart, and FarSense needs a steady sample rate, so the
    ratio is linearly interpolated onto evenly spaced times.

    Returns ``(grid, fabricated, fs)``: the ``(n_samples, n_subcarriers)``
    complex grid; a boolean mask marking samples that fall inside a real gap
    in the data (interpolated across a dropout, not measured); and the grid's
    sample rate in Hz.
    """
    # Sample rate = 1 / (median time between frames).
    _, fs = doppler.uniform_grid(tt)

    # Evenly spaced times from the first frame to the last.
    n = int(np.floor((tt[-1] - tt[0]) * fs)) + 1
    gt = tt[0] + np.arange(n) / fs

    # A pause longer than this between two frames is a dropout, not jitter.
    gl = doppler.gap_limit_for(tt)

    # Interpolate the real and imaginary parts separately. Interpolating the
    # phase directly would break where it wraps around at +/- pi.
    re, fab = doppler.resample_uniform(tt, r.real, gt, gl)
    im, _ = doppler.resample_uniform(tt, r.imag, gt, gl)
    return re + 1j * im, fab, fs


def _step(g: np.ndarray, L: int) -> np.ndarray:
    """Per-sample step at a lag of *L* samples, median over subcarriers.

    For each sample t and subcarrier, how much the ratio changed since sample
    t - L, as a fraction between 0 (identical) and 1 (opposite)::

        |r_t - r_{t-L}| / (|r_t| + |r_{t-L}|)

    It counts a change in magnitude and a change in phase alike. The median
    across subcarriers then gives one number per sample -- a median, so a few
    subcarriers sitting in a fading dip cannot inflate it.
    """
    d = np.abs(g[L:] - g[:-L]) / (np.abs(g[L:]) + np.abs(g[:-L]))
    return np.nanmedian(d, axis=1)


def motion_features(g: np.ndarray, fs: float) -> dict[str, float]:
    """A: the structure function's shape at three lags, and the 2 s P90.

    The four motion features:

    * ``D(tau)`` is the typical (median) step at a lag of ``tau`` seconds,
      measured at 0.25, 2 and 5 s. How D grows with the lag is the *shape* of
      the motion, and it does not depend on how strong the signal is:

      - ``A_slope3`` -- slope of log D against log tau (how fast change grows
        with time);
      - ``A_r025_2`` -- D(0.25 s) / D(2 s);
      - ``A_r5_2``   -- D(5 s) / D(2 s).

    * ``A_p90`` -- the step at a 2 s lag, averaged within each second, then
      the 90th percentile over the seconds: roughly "how much the channel
      moved in its busiest ~6 seconds of the minute".
    """
    # D(tau) for each lag: lag in samples = round(tau * fs), at least 1.
    D = {tau: float(np.nanmedian(_step(g, max(1, int(round(tau * fs)))))) for tau in SHAPE_LAGS}

    # The shape: a straight-line fit of log D against log tau, and two ratios.
    slope = float(np.polyfit(np.log(SHAPE_LAGS), np.log([D[t] for t in SHAPE_LAGS]), 1)[0])

    # The P90: the 2 s step for every sample ...
    s = _step(g, int(round(P90_LAG * fs)))
    # ... averaged within each 1 s block (k samples a block; the tail is dropped) ...
    k = int(round(fs))
    m = len(s) // k
    per_sec = np.nanmean(s[: m * k].reshape(m, k), axis=1)   # 1 s blocks, MEAN, tail dropped
    per_sec = per_sec[np.isfinite(per_sec)]
    # ... and the 90th percentile of those per-second values.
    return {"A_slope3": slope, "A_r025_2": D[0.25] / D[2.0], "A_r5_2": D[5.0] / D[2.0],
            "A_p90": float(np.percentile(per_sec, 90))}


def breath_features(g: np.ndarray, fab: np.ndarray, fs: float) -> dict[str, float]:
    """F: the FarSense sweep with hybrid 2's settings, on this grid.

    FarSense (backend/farsense.py) slides a 10 s window along the minute in
    1 s steps -- about 51 windows. In each window it finds, per subcarrier,
    the direction in which the ratio swings most regularly at a breathing
    rate (10-30 breaths per minute), then autocorrelates those swings. The
    height of the first autocorrelation peak (``peak``, -1..1) says how
    regularly the signal repeats; its lag gives a breathing rate (``rpm``).

    The four breathing features:

    * ``F_pkmax``  -- the highest window peak;
    * ``F_pkmed``  -- the median window peak;
    * ``F_run``    -- the longest run of consecutive windows with peak >= 0.25
      (in seconds, since windows are 1 s apart);
    * ``F_rpm_sd`` -- the spread (standard deviation) of the breathing rates
      of those windows; NaN when fewer than two windows qualify.
    """
    # Prepare the grid once: smoothing, dead subcarriers, window layout.
    prep = farsense.prepare(g, fs, fabricated=fab, window_seconds=10.0, hop_seconds=1.0,
                            band_rpm=(10.0, 30.0), savgol_seconds=1.0, savgol_order=3,
                            highpass_hz=0.0)
    # Run every window. motion_frac_hi=1e9 switches FarSense's own motion
    # gate off, so every window reports a peak and a rate.
    fz = farsense._run(prep, dict(n_theta=200, fft_size=8192, keep_fraction=0.65,
                                  motion_frac_hi=1e9, max_gap_fraction=0.5, min_peak=-1.0,
                                  positive_only=True), None)

    # One peak and one rate per window. Windows mostly made of interpolated
    # (fabricated) samples are "unknown" and count as missing.
    pk = np.where(fz["unknown"], np.nan, fz["acf_peak_norm"]).astype(float)
    rpm = np.asarray(fz["rpm"], float)

    # The windows that look like breathing, and the longest unbroken run of them.
    good = np.isfinite(pk) & (pk >= BREATH_PEAK)
    best = cur = 0
    for gi in good:
        cur = cur + 1 if gi else 0
        best = max(best, cur)
    return {"F_pkmax": float(np.nanmax(pk)), "F_pkmed": float(np.nanmedian(pk)),
            "F_run": best * 1.0,
            "F_rpm_sd": float(np.nanstd(rpm[good])) if good.sum() >= 2 else float("nan")}


def probability(features: dict[str, float]) -> float:
    """The fitted pipeline as arithmetic: impute, standardise, logistic.

    This is the whole "machine learning" step. The model was trained earlier
    (scikit-learn, 705 labelled windows); its learned numbers are the
    constants at the top of this file, so predicting is three lines of maths:

    1. a missing feature (NaN) is replaced by its training median;
    2. each feature becomes a z-score: (value - mean) / scale;
    3. p = 1 / (1 + exp(-(intercept + sum of weight * z))).

    Returns p(person), between 0 and 1.
    """
    x = np.array([features[f] for f in FEATURES], dtype=float)
    x = np.where(np.isnan(x), IMPUTER_MEDIAN, x)        # 1. fill missing values
    z = (x - SCALER_MEAN) / SCALER_SCALE                 # 2. standardise
    return float(1.0 / (1.0 + np.exp(-(INTERCEPT + COEF @ z))))   # 3. logistic


def evaluate(path: str | Path, t0: float = 0.0, t1: float = WINDOW_SECONDS) -> dict[str, Any]:
    """Judge one window. ``label`` is 1 for a person, 0 for no person.

    Raises ``ValueError`` when the window is not judged: too little usable
    data, a FarSense window that cannot form, or a feature other than
    ``F_rpm_sd`` (NaN there just means no breathing) coming out NaN.

    Runs the whole pipeline on ``[t0, t1)`` seconds of one capture and
    returns a dict with the label, p(person), the eight features, and how
    long each stage took (``stage_ms``) -- the last for timing on the board.
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
        """Record the time since the previous mark under *stage*."""
        nonlocal lap
        now = time.perf_counter()
        ms[stage] = round((now - lap) * 1000.0, 2)
        lap = now

    # Index the capture file: where every frame starts and what it carries.
    index = mtk.MTKIndex(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN slices, as the reference

        # 1. frames -> CSI ratio
        r, tt = load_ratio(path, index, t0, t1)
        mark("decode")
        if tt.size < 2:
            raise ValueError("no frame in the window carries a usable ratio")

        # 2. every 8th live subcarrier, on an even time grid
        g, fab, fs = resample(r[:, ::STRIDE], tt)
        mark("resample")
        if len(g) < MIN_SECONDS * fs:
            raise ValueError(f"only {len(g) / fs:.1f} s of usable grid, under {MIN_SECONDS:g} s")

        # 3. the four motion features and 4. the four breathing features
        feats = motion_features(g, fs)
        mark("A")
        feats.update(breath_features(g, fab, fs))
        mark("F")

    # A missing breathing-rate spread is normal (no breathing seen); any other
    # missing feature means there was not enough data to judge.
    missing = [f for f in FEATURES if f != "F_rpm_sd" and not np.isfinite(feats[f])]
    if missing:
        raise ValueError(f"not judged: {', '.join(missing)} undefined")

    # 5. the model
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
    """Command line: ``af8 CAPTURE.bin [--t0 S] [--t1 S] [--json]``.

    Prints ``1`` or ``0`` (or the full result as JSON with ``--json``).
    Exit status: 0 judged, 2 the capture could not be judged, 3 NumPy too old.
    """
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
