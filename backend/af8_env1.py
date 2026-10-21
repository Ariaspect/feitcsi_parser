"""The 11-feature model: AF8's ten features plus X_gate_dc (the stationary-fan switch times residual breathing).

    python3 -m backend.af8_env1 CAPTURE.bin [--json]

Trained on the user's GT selection of one-minute captures from both rooms --
env1 (10/12-10/14, 270) and env2 (260) -- plus two SNR-augmented copies of
each (see the end of this docstring); not on the 1151 windows of
``backend.af8``'s model. (The module keeps its first name, env1.) The ten
features are ``af8``'s own; one is added, ``X_gate_dc`` (#11), built from two parts:

* ``gate_dc`` -- the stationary-fan switch (computed, NOT a model input since
  2026-10-21: dropping it alone left accuracy the same or higher). The leading subcarrier
  eigenvector of each 2 s block (detrended) is compared with the next one;
  ``ev_drift`` is the median of ``1 - |<v_i, v_i+1>|``. A fan that stays put
  shakes one path the same way all minute, so its direction holds
  (ev_drift ~ 0.005); noise and people wander (~0.1-0.2).
  ``gate_dc = -log10(ev_drift + 1e-4) * (1 - clip(C3, 0, 1))``: large when
  one path shakes steadily, switched off when C3 says the fan rotates.
* ``X_gate_dc`` (#11) -- the switch times ``R_F_run``: the FarSense run on
  the grid with the fan's subspace projected out. The subspace is the top two
  eigendirections of the 1 s high-pass (the blades' fast part); removing them
  from the whole signal removes the slow fake breathing the blades alias
  into, while a person's breathing on another path stays.
``A_g15`` is computed here (``gated_motion``) but is NOT a model input since
2026-10-19: it reads ``A_p90`` of the ratio after the direct path is removed.
It helped env1 (5-fold KPI#2 80.7 -> 87.2) but dividing by the small residual
amplifies receiver noise (packet timing jitter, CSI noise) when the direct path
is strong (10/19: 6 % residual, empty rooms read 0.19-0.23 and were judged
person), and a fan shaking a reflection reads like a person. Per
  frame, chain 1 over all live subcarriers goes to the delay domain (IFFT,
  zero-padded x4); everything within 1.5 taps (1 tap = 1/80 MHz ~ 3.75 m) of
  the strongest tap is zeroed and the rest transformed back; chain 0 stays as
  the reference: ``r = H1_gated / H0``.

The window, grid and every setting are ``af8``'s; the weights below are the
same pipeline (median impute, standardise, logistic, class-balanced) fitted on
the 530 selected captures with these eleven features, each capture three times:

* as captured;
* two "farther" copies. On the resampled grid ``g`` the change around each
  subcarrier's mean is scaled by ``a`` (0.6, 0.35) and fresh white noise at
  the capture's own level (from its 1-sample difference) refills the rest, so
  the noise floor stays put: a person or object farther from the link moves
  the channel less, the receiver noise does not. The copies' ten af8
  features are recomputed; ``X_gate_dc`` is the original
  capture's, because white noise is not the receiver's (its timing jitter is
  common to all subcarriers) and recomputing them broke the stationary-fan
  switch (env1 static fan, leave-one-session-out, 45 -> 10 %). Only the same
  capture's information is used.

On env1 alone (train and test env1, leave-one-session-out) the augmentation
moved KPI#2 70.0 -> 71.9; on copies at a = 0.35 the model holds 74 % against
58 % without it, and the 10/19 far small movements went 3/8 -> 5/8.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from backend import af8, framediff, mtk

FEATURES = af8.FEATURES + ("X_gate_dc",)

# Fitted on env1 270 + env2 260 (user's GT selection) x {original, a=0.6, a=0.35}
# (2026-10-21, 11 features: gate_dc dropped), FEATURES order.
IMPUTER_MEDIAN = np.array([0.0645842, 0.871246, 1.05091, 0.0242607, 0.32004, 0.114291, 1, 1.81537, 0.0449383, 0.0482942, 1.10207])
SCALER_MEAN = np.array([0.105382, 0.819076, 1.09548, 0.0423376, 0.328244, 0.128712, 3.76415, 2.04394, 0.123209, 0.249725, 7.40922])
SCALER_SCALE = np.array([0.108161, 0.177403, 0.114489, 0.0530163, 0.0924005, 0.0782143, 6.80596, 1.28134, 0.171773, 0.620005, 17.1996])
COEF = np.array([-0.170463, -3.71581, 0.645044, 0.256662, 0.226932, 1.2454, -0.106985, -0.190893, -2.99984, 0.361517, 2.93606])
INTERCEPT = 1.64562
THRESHOLD = af8.THRESHOLD
TRAINED_ON = "env1 270 + env2 260, SNR-augmented (X_gate_dc from the original), 11 features"

DRIFT_BLOCK = 2.0         # seconds; ev_drift compares the leading eigenvector of consecutive blocks
DRIFT_FLOOR = 1e-4        # keeps -log10 finite when ev_drift is 0
FAN_DIRECTIONS = 2        # eigendirections of the high-pass projected out for R_F_run
HIGHPASS = 1.0            # seconds; moving average removed to leave the blades' fast part
GATE_PAD = 4              # IFFT zero-padding: 1 tap = 4 points
GATE_TAPS = 1.5           # the direct path is zeroed within +-this many taps of the peak


def load_chains(path: Path, index: mtk.MTKIndex, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``af8.load_ratio`` stopped before the division: both chains, the live mask, the bin count, times.

    Returns ``(H, tt, live, n_bins)``: ``H`` is ``(frames, 2, live subcarriers)``
    complex with the same frames ``load_ratio`` keeps, so ``H[:, 1] / H[:, 0]``
    is its ratio exactly.
    """
    t = np.asarray(index.times, dtype=float)
    tr = t - t[0]
    sel = framediff._uniform_selection(index, None, None)
    ids = np.flatnonzero(np.asarray(sel["mask"], bool) & (tr >= t0) & (tr < t1))
    if ids.size < 2:
        raise ValueError(f"fewer than 2 frames in [{t0:g}, {t1:g}) s once the set was made uniform")
    H = mtk.decode_complex(path, index, ids)[:, :2, :]
    n_bins = H.shape[2]
    live = np.isfinite(H).all(axis=1).mean(axis=0) > af8.LIVE_FRACTION
    H = H[:, :, live]
    tt = t[ids]
    keep = np.isfinite(H).all(axis=(1, 2)) & (np.abs(H[:, 0, :]).min(axis=1) > 0)
    return H[keep], tt[keep], live, n_bins


def stationary_switch(g: np.ndarray, fab: np.ndarray, fs: float, c3: float) -> dict[str, float]:
    """``ev_drift``, ``R_F_run`` and the two features built from them, ``gate_dc`` and ``X_gate_dc``."""
    gz = np.nan_to_num(g)
    nb = int(round(DRIFT_BLOCK * fs))
    A = np.c_[np.ones(nb), np.arange(nb)]
    vecs = []
    for s in range(0, len(gz) - nb, nb):
        z = gz[s:s + nb]
        z = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]           # detrend each subcarrier
        vecs.append(np.linalg.eigh(z.conj().T @ z / nb)[1][:, -1])  # leading direction
    drift = float(np.median([1 - abs(np.vdot(x, y)) for x, y in zip(vecs[:-1], vecs[1:])]))

    mu = gz.mean(axis=0)
    z = gz - mu
    k = max(1, int(round(HIGHPASS * fs)))
    hp = z - np.apply_along_axis(lambda c: np.convolve(c, np.ones(k) / k, "same"), 0, z)
    V = np.linalg.eigh(hp.conj().T @ hp / len(hp))[1][:, -FAN_DIRECTIONS:]
    res = z - (z @ V) @ V.conj().T + mu
    run = af8.breath_features(res, fab, fs)["F_run"]

    gate = -np.log10(drift + DRIFT_FLOOR) * (1.0 - float(np.clip(c3, 0.0, 1.0)))
    return {"ev_drift": drift, "R_F_run": run, "gate_dc": gate, "X_gate_dc": gate * run}


def direct_path_gated(H: np.ndarray, live: np.ndarray, n_bins: int) -> np.ndarray:
    """Chain 1 with the direct path zeroed in the delay domain, on the live subcarriers."""
    pos = np.flatnonzero(live)
    full = np.zeros((len(H), n_bins), complex)
    full[:, pos] = H[:, 1]
    L = n_bins * GATE_PAD
    h = np.fft.ifft(full, n=L, axis=1)
    pk = (np.abs(h) ** 2).argmax(axis=1)
    d = (np.arange(L)[None, :] - pk[:, None] + L // 2) % L - L // 2
    h = np.where(np.abs(d) < GATE_TAPS * GATE_PAD, 0, h)
    return np.fft.fft(h, axis=1)[:, :n_bins][:, pos]


def gated_motion(H: np.ndarray, tt: np.ndarray, live: np.ndarray, n_bins: int) -> float:
    """``A_g15``: af8's ``A_p90`` on ``H1_gated / H0``."""
    r = direct_path_gated(H, live, n_bins) / H[:, 0]
    g, _, fs = af8.resample(r[:, ::af8.STRIDE], tt)
    return af8.motion_features(g, fs)["A_p90"]


def probability(features: dict[str, float]) -> float:
    """Impute, standardise, logistic -- ``af8.probability`` with this model's numbers."""
    x = np.array([features[f] for f in FEATURES], dtype=float)
    x = np.where(np.isnan(x), IMPUTER_MEDIAN, x)
    z = (x - SCALER_MEAN) / SCALER_SCALE
    return float(1.0 / (1.0 + np.exp(-(INTERCEPT + COEF @ z))))


def features(path: str | Path, t0: float = 0.0, t1: float = af8.WINDOW_SECONDS, gated: bool = False) -> dict[str, Any]:
    """The eleven model features of one window (and gate_dc), plus what they are built from; ``gated`` also adds A_g15 (not a model input)."""
    path = Path(path)
    index = mtk.MTKIndex(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        H, tt, live, n_bins = load_chains(path, index, t0, t1)
        r = H[:, 1] / H[:, 0]
        g, fab, fs = af8.resample(r[:, ::af8.STRIDE], tt)
        if len(g) < af8.MIN_SECONDS * fs:
            raise ValueError(f"only {len(g) / fs:.1f} s of usable grid, under {af8.MIN_SECONDS:g} s")
        f = af8.motion_features(g, fs)
        f.update(af8.breath_features(g, fab, fs))
        f.update(af8.revisit_feature(g, fs))
        f.update(af8.diff_feature(g, fab, fs, f["C3_revisit"]))
        f.update(stationary_switch(g, fab, fs, f["C3_revisit"]))
        if gated:
            f["A_g15"] = gated_motion(H, tt, live, n_bins)
    return f


def evaluate(path: str | Path, t0: float = 0.0, t1: float = af8.WINDOW_SECONDS) -> dict[str, Any]:
    """Judge one window with the env1 model."""
    start = time.perf_counter()
    f = features(path, t0, t1)
    p = probability(f)
    return {"file": Path(path).name, "label": int(p > THRESHOLD), "p_person": p,
            "features": {k: (None if not np.isfinite(f[k]) else float(f[k])) for k in FEATURES},
            "gate_dc": f["gate_dc"],
            "ev_drift": f["ev_drift"], "R_F_run": f["R_F_run"],
            "elapsed_s": round(time.perf_counter() - start, 3)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="af8_env1", description="env1-model presence verdict: 1 (person) or 0.")
    ap.add_argument("capture", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        out = evaluate(args.capture)
    except (OSError, ValueError) as exc:
        print(f"af8_env1: {args.capture}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out) if args.json else out["label"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
