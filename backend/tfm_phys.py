"""The Physics tab's models (user 2026-11-12): the default Transformer with physics channels in place of the
reflection and breathing channels that machines fool. ``seq13pw`` (13 channels) and ``seq15pwu`` (the same + pk_u,
bs_u) are the two best sets of the challenger benchmark's channel-set study (exp/challengers,
experiments/challengers/CHANNELSET_REPORT.md); the 17-channel base (``seq17``) is kept here, calibrated the same way,
for comparison.

What the study changed from the 17 (s025 s2 s5 pk rpm pk_diff pk_res drift ret g15 bs_all bs_r1 bs_r2 bs_tap bs_f
plf_res plf_g): kept s025 s2 rpm bs_f plf_res plf_g; removed s5 pk pk_diff pk_res drift ret g15 bs_all bs_r1 bs_r2
bs_tap; added, per second on the same 61 s window:

* ``lgrow``     log(s5 / s025): a sub-wavelength slow displacement (breathing, posture) grows from 0.25 s to 5 s;
  blades, a moving vacuum and noise are flat. Scale-free, not calibrated.
* ``g15_res``   2 s step of the direct-path-gated signal (g15's) after its two strongest 1 s high-passed directions
  are projected out: a compact scatterer (blades, a vacuum) moves only those few patterns.
* ``u_g15v``    the same with ``swing_a`` x its swing-phase average removed too (the swing-locked reflection change).
* ``sT``        step at one swing period T: a fan alone comes back after one swing (ret divided it by the T/2 step,
  which let a strongly seen swing hide a person).
* ``swing_ab``  the share of the breathing-band change (1 s minus 6 s moving average) locked to the swing phase,
  above the chance share of independent cycles, clipped to [0, 1]; one value per window, not calibrated.
* ``pk_w``, ``bs_w``  FarSense peak / breathing prominence after w = max(swing_a, swing_ab) x the swing-phase average
  is subtracted: only as much swing as is seen comes out, so with no fan (w = 0) they equal pk / bs_all.
* ``pk_u``, ``bs_u`` (seq15pwu only)  the same with the whole swing-phase average removed.

``swing_a`` is ``swing_ab`` on the slow band (6 s moving average). T is ``af8.swing_period``. Values are identical to
the benchmark's extraction (bench/newch_extract.py, itself checked against ``backend.tfm_series``): only the channels a
model needs are computed, and a w = 0 window reuses the plain FarSense / bs runs (subtracting 0 x the average leaves the
signal bit-identical).

Calibration (the benchmark's regime B, "fixed5"): log on the step channels, then on the calibrated channels minus the
median of the room's five frozen install empty minutes, divided by their MAD x 1.4826 shrunk halfway (geometrically)
toward the training rooms' (pool). The judged room is always the held-out one: env1 by the model trained on env2 + env3
+ env3-1, ..., the scenario C session by the env1 + env2 model with env3-1's references. Then the fold's training-row
standardisation, missing -> 0, and each channel's availability in the raw series appended (2C x 61). Network as
``backend.tfm_model`` (one token a second, d 16, one post-norm encoder layer, 2 heads, FFN 32, CLS), 5 seeds, P = their
mean, person if P > 0.5. Weights from ``scripts/fit_tfm_phys.py`` (``tfm_phys.npz`` / ``tfm_phys.json``), which checked
them against the benchmark's saved probabilities. numpy only (runs on the board).
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from backend import af8, af8_env1 as E, breath_components as BC, mtk, tfm_series as TS
from backend.tfm_model import forward as _forward, room_for

N = TS.N
_HERE = Path(__file__).resolve().parent
META: dict[str, Any] = json.loads((_HERE / "tfm_phys.json").read_text())
_W = np.load(_HERE / "tfm_phys.npz")
MODELS: list[str] = META["order"]                      # seq13pw, seq15pwu, seq17
NEW_MODELS = [m for m in MODELS if m != "seq17"]
THRESHOLD = float(META["threshold"])
SEEDS = int(META["seeds"])
_PARAMS = ("inp_w", "inp_b", "pos", "cls", "in_w", "in_b", "out_w", "out_b", "l1_w", "l1_b", "l2_w", "l2_b",
           "n1_g", "n1_b", "n2_g", "n2_b", "h_w", "h_b")
NETS = {m: {f: [{k: _W[f"{m}_{f}_s{s}_{k}"].astype(float) for k in _PARAMS} for s in range(SEEDS)]
            for f in META["models"][m]["folds"]} for m in MODELS}
FOLD = {m: {f: {k: _W[f"{m}_{f}_{k}"].astype(float) for k in ("mu", "sd", "pool")} for f in META["models"][m]["folds"]}
        for m in MODELS}
REF = {m: {r: _W[f"{m}_ref_{r}"].astype(float) for r in META["ref_keys"]} for m in MODELS}   # logged, (5, C, 61)
# Every channel the two physics models read, in display order (seq15pwu's: seq13pw's then pk_u, bs_u).
PHYS_CHANNELS: list[str] = list(dict.fromkeys(c for m in NEW_MODELS for c in META["models"][m]["channels"]))
# Display-only results: the full-dataset check (scripts/verify_tfm_phys.py) and the board timing
# (scripts/bench_tfm_phys_board.py), merged by scripts/tfm_phys_results.py; {} until they exist.
try:
    RESULTS: dict[str, Any] = json.loads((_HERE / "tfm_phys_results.json").read_text())
except OSError:
    RESULTS = {}


def _stats(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD (x 1.4826) per channel over every second of every reference minute; SD, then 1, where 0."""
    K = v.shape[1]
    med = np.array([np.nanmedian(v[:, c]) for c in range(K)])
    mad = np.array([np.nanmedian(np.abs(v[:, c] - med[c])) * 1.4826 for c in range(K)])
    sd = np.array([np.nanstd(v[:, c]) for c in range(K)])
    return med, np.where(mad > 1e-9, mad, np.where(sd > 1e-9, sd, 1.0))


_REF_STATS = {m: {r: _stats(REF[m][r]) for r in META["ref_keys"]} for m in MODELS}


# ----------------------------------------------------------------------------------------------- series
def _fast_subspace(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-removed grid, its two strongest 1 s high-passed directions and the mean (af8_env1's fan subspace)."""
    xz = np.nan_to_num(x)
    mu = xz.mean(0)
    z = xz - mu
    k = max(1, int(round(E.HIGHPASS * fs)))
    hp = z - np.apply_along_axis(lambda c: np.convolve(c, np.ones(k) / k, "same"), 0, z)
    V = np.linalg.eigh(hp.conj().T @ hp / len(hp))[1][:, -E.FAN_DIRECTIONS:]
    return z, V, mu


def sync_average(x: np.ndarray, T: int) -> np.ndarray:
    """Per sample: x's average over the window's swing cycles at the same phase (n mod T). T is 10-30 s, so every
    phase has 2-6 cycles in a 61 s window."""
    ph = np.arange(len(x)) % T
    xs = np.where(np.isfinite(x), x, 0)
    cnt = np.bincount(ph, minlength=T).astype(float)
    avg = np.stack([np.bincount(ph, xs[:, j].real, T) + 1j * np.bincount(ph, xs[:, j].imag, T) for j in range(x.shape[1])], 1) / cnt[:, None]
    return avg[ph]


def swing_alpha(x: np.ndarray, T: int, fs: float, band: bool = False) -> float:
    """How much of the slow (< ~0.17 Hz) change repeats every T, above the chance share of independent cycles: 0 with
    no swing, -> 1 when the slow change is the swing. band=True: the same in the breathing band (1 s moving average
    minus the 6 s one)."""
    k = max(1, int(round(6.0 * fs)))
    xz = np.nan_to_num(x)
    ma = lambda w: np.apply_along_axis(lambda c: np.convolve(c, np.ones(w) / w, "same"), 0, xz)
    lp = (ma(max(1, int(round(1.0 * fs)))) - ma(k) if band else ma(k))[k // 2: len(xz) - k // 2]
    lp = lp - lp.mean(0)
    n = len(lp)
    if n < 2 * T:
        return 0.0
    ph = (np.arange(n) + k // 2) % T
    cnt = np.bincount(ph, minlength=T).astype(float)
    avg = np.stack([np.bincount(ph, lp[:, j].real, T) + 1j * np.bincount(ph, lp[:, j].imag, T) for j in range(lp.shape[1])], 1) / cnt[:, None]
    tot = float((np.abs(lp) ** 2).sum())
    if tot <= 0:
        return 0.0
    r = float((np.abs(avg[ph]) ** 2).sum()) / tot
    c = float(np.mean(1.0 / cnt[ph]))
    return float(np.clip((r - c) / (1 - c), 0.0, 1.0))


def _bs_lines(gf: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """bs_all and bs_f of ``tfm_series._bs_series`` (the same windows and arithmetic) without the CIR-tap and
    leave-one-out lines the physics models do not read."""
    gf = gf[:, np.isfinite(gf).all(axis=0)]
    W = int(round(BC.WINDOW_SECONDS * fs)); st = int(round(TS.BS_HOP * fs)); n = len(gf)
    a, f = np.full(N, np.nan), np.full(N, np.nan)
    if n < W or gf.shape[1] < BC.N_COMPONENTS:
        return a, f
    g8 = gf[:, ::TS.BS_STRIDE]
    for s0 in range(0, n - W + 1, st):
        x = g8[s0:s0 + W]
        Xc = np.c_[x.real - x.real.mean(axis=0), x.imag - x.imag.mean(axis=0)]
        _, V = np.linalg.eigh(Xc.T @ Xc)
        comps = Xc @ V[:, ::-1][:, :BC.N_COMPONENTS]
        pr = [BC.prominence(comps[:, j], fs) for j in range(BC.N_COMPONENTS)]
        v = np.array([p[0] for p in pr]); c = int(round((s0 + W / 2) / fs))
        if c >= N:
            continue
        a[c] = v.max()
        f[c] = pr[int(np.argmax(v))][1]
    return a, f


def series(path: str | Path, t0: float, t1: float, index: mtk.MTKIndex | None = None,
           channels: list[str] | None = None, timings: dict[str, float] | None = None) -> dict[str, Any]:
    """{"S": channel -> (61,) raw series, "T": swing period (s), "swing_a", "swing_ab", "w"} of [t0, t1) s from the
    capture's first frame, for ``channels`` (default: every physics channel). ``timings`` collects seconds per stage.
    The reference arithmetic; the tab computes it with ``backend.tfm_phys_fast.series`` (same values to rounding,
    checked on every labelled window by scripts/verify_tfm_phys_fast.py)."""
    want = set(PHYS_CHANNELS if channels is None else channels)
    tm = timings if timings is not None else {}
    clock = [time.perf_counter()]

    def lap(name: str) -> None:
        now = time.perf_counter(); tm[name] = tm.get(name, 0.0) + now - clock[0]; clock[0] = now

    path = Path(path)
    idx = index if index is not None else mtk.MTKIndex(path)
    H, tt, live, nb = E.load_chains(path, idx, t0, t1)
    r = H[:, 1] / H[:, 0]
    g, fab, fs = af8.resample(r[:, ::af8.STRIDE], tt)
    lap("decode")
    S: dict[str, np.ndarray] = {}
    for nm, lag in (("s025", 0.25), ("s2", 2.0), ("s5", 5.0)):
        L = max(1, int(round(lag * fs))); S[nm] = TS.per_sec(af8._step(g, L), L, fs)
    with np.errstate(divide="ignore", invalid="ignore"):
        S["lgrow"] = np.where((S["s5"] > 0) & (S["s025"] > 0), np.log(S["s5"] / S["s025"]), np.nan)
    T = af8.swing_period(g, fs)
    S["sT"] = TS.per_sec(af8._step(g, T), T, fs)
    lap("steps + swing period")
    z, V, mu = _fast_subspace(g, fs)
    res = z - (z @ V) @ V.conj().T + mu
    S["plf_res"] = TS._plf(res, T, fs)
    S["plf_g"] = TS._plf(g, T, fs)
    lap("plf")
    a = swing_alpha(g, T, fs)
    ab = swing_alpha(g, T, fs, band=True)
    w = max(a, ab)
    S["swing_ab"] = np.full(N, ab)
    lap("swing share")
    pk, S["rpm"] = TS._fs_windows(g, fab, fs, 0)
    if "pk_w" in want:
        S["pk_w"] = pk if w == 0 else TS._fs_windows(g - w * sync_average(g, T), fab, fs, 0)[0]
    if "pk_u" in want:
        S["pk_u"] = TS._fs_windows(g - sync_average(g, T) + np.nanmean(g, 0), fab, fs, 0)[0]
    lap("FarSense")
    if want & {"g15_res", "u_g15v"}:
        rg = E.direct_path_gated(H, live, nb) / H[:, 0]
        gg, _, _ = af8.resample(rg[:, ::af8.STRIDE], tt)
        L2 = int(round(af8.P90_LAG * fs))
        zg, Vg, mug = _fast_subspace(gg, fs)
        resg = zg - (zg @ Vg) @ Vg.conj().T + mug
        S["g15_res"] = TS.per_sec(TS._step_n(resg, resg, L2), L2, fs)
        S["u_g15v"] = TS.per_sec(TS._step_n(resg - a * sync_average(resg, T), resg, L2), L2, fs)
    lap("reflection (g15_res, u_g15v)")
    gf, _, fsf = af8.resample(r, tt)
    bs_all, S["bs_f"] = _bs_lines(gf, fsf)
    if "bs_w" in want:
        S["bs_w"] = bs_all if w == 0 else _bs_lines(gf - w * sync_average(gf, T), fsf)[0]
    if "bs_u" in want:
        S["bs_u"] = _bs_lines(gf - sync_average(gf, T) + np.nanmean(gf, 0), fsf)[0]
    lap("breathing lines (bs)")
    return {"S": S, "T": T / fs, "swing_a": a, "swing_ab": ab, "w": w}


def matrix(S: dict[str, np.ndarray], model: str) -> np.ndarray:
    """The model's (C, 61) raw input, its channels in training order."""
    return np.array([S[c] for c in META["models"][model]["channels"]], dtype=float)


# ----------------------------------------------------------------------------------------------- model
def is_reference(stem: str, room: dict[str, Any]) -> bool:
    return stem in META["ref_keys"].get(room["calibrated_with"], [])


def calibrate(X: np.ndarray, model: str, room: dict[str, Any]) -> np.ndarray:
    """Log on the step channels, then (x - reference median) / (reference MAD shrunk halfway to the fold's pool)."""
    m = META["models"][model]
    ch = m["channels"]
    x = X.astype(float).copy()
    lg = [ch.index(c) for c in m["log_channels"]]
    cal = [ch.index(c) for c in m["calibrated_channels"]]
    with np.errstate(invalid="ignore"):
        x[lg] = np.log(np.clip(x[lg], 1e-6, None))
    med, mad = _REF_STATS[model][room["calibrated_with"]]
    pool = FOLD[model][room["fold"]]["pool"]
    sd = np.exp((1 - 0.5) * np.log(mad) + 0.5 * np.log(pool))
    x[cal] = (x[cal] - med[cal, None]) / sd[cal, None]
    return x


def predict(Z: np.ndarray, avail: np.ndarray, model: str, fold: str) -> dict[str, Any]:
    """5-seed mean P of the calibrated (C, 61) input; ``avail`` = finite in the raw series."""
    f = FOLD[model][fold]
    v = (Z - f["mu"][:, None]) / f["sd"][:, None]
    inp = np.concatenate([np.where(np.isfinite(v), v, 0.0), avail.astype(float)], 0).astype(np.float32).astype(float)
    ps = 1 / (1 + np.exp(-np.array([_forward(P, inp) for P in NETS[model][fold]])))
    p = float(ps.mean())
    return {"p": p, "label": int(p > THRESHOLD), "p_seeds": [float(q) for q in ps]}


def judge(X: np.ndarray, model: str, room: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    Z = calibrate(X, model, room)
    return Z, predict(Z, np.isfinite(X), model, room["fold"])


@lru_cache(maxsize=64)
def _cached(path: str, mtime: float, window: int) -> dict[str, Any]:
    from backend import af8_explain
    from backend.tiles import get_index

    p = Path(path)
    times = np.asarray(get_index(p).times, dtype=float)
    if times.size < 2:
        raise ValueError("fewer than 2 frames in the capture")
    wins = af8_explain.windows_for(float(times[-1] - times[0]))
    if not 0 <= window < len(wins):
        raise ValueError(f"window {window} out of range: this capture has {len(wins)}")
    t0, t1 = wins[window]
    tm: dict[str, float] = {}
    a = time.perf_counter()
    idx = mtk.MTKIndex(p)
    tm["index"] = time.perf_counter() - a
    from backend import tfm_phys_fast                 # the same values as series(), faster (board timing)
    s = tfm_phys_fast.series(p, t0, t1, idx, timings=tm)
    room = room_for(p.stem)
    out: dict[str, Any] = {"window": int(window), "windows": [[float(u), float(v)] for u, v in wins],
                           "window_s": [float(t0), float(t1)], "channels": PHYS_CHANNELS,
                           "raw": np.array([s["S"][c] for c in PHYS_CHANNELS]), "swing_period_s": float(s["T"]),
                           "swing_a": s["swing_a"], "swing_ab": s["swing_ab"], "w": s["w"], "room": room,
                           "models": {}, "calibrated": None, "reference_minute": False}
    if room is None:
        out["timings"] = tm
        return out
    out["reference_minute"] = is_reference(p.stem, room)
    cal = {}
    for m in NEW_MODELS:
        a = time.perf_counter()
        Z, v = judge(matrix(s["S"], m), m, room)
        tm[f"calibrate + forward {m}"] = time.perf_counter() - a
        cal.update(dict(zip(META["models"][m]["channels"], Z)))
        out["models"][m] = {**v, "trained_on": META["models"][m]["folds"][room["fold"]]["trained_on"]}
    out["calibrated"] = np.array([cal[c] for c in PHYS_CHANNELS])
    a = time.perf_counter()
    b = TS.series(p, t0, t1, idx)
    _, v = judge(b["X"], "seq17", room)
    tm["base: 17-channel series + judgment"] = time.perf_counter() - a
    out["models"]["seq17"] = {**v, "trained_on": META["models"]["seq17"]["folds"][room["fold"]]["trained_on"]}
    out["timings"] = tm
    return out


def evaluate(path: str | Path, window: int = 0) -> dict[str, Any]:
    """Both physics models (and the 17-channel base, same calibration) on one window of the ML tab's windows
    (``af8_explain.windows_for``). Raises ``ValueError``."""
    p = Path(path)
    return _cached(str(p), p.stat().st_mtime, int(window))
