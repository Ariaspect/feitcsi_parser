"""The Transformer tab's model: transformer alone on 17 per-second channels (15 + phase-locked fraction), one
model per held-out room.

A window is judged the way it was evaluated (user 2026-11-10): the capture's room is looked up, and the model that
never saw that room judges it -- env1 by the model trained on env2 + env3 + env3-1, and so on; the 2.6 h
scenario C session by the env1 + env2 model. The series (``backend.tfm_series``) is calibrated against that room's
empty minutes: log on the fast channels, minus the empty median, divided by the empty MAD shrunk halfway
(geometrically) toward the training rooms' -- an empty reference minute is calibrated without itself -- then
standardised as in training, NaN set to 0, and 17 availability masks appended (34 x 61). The model: one token a
second, Linear 34 -> 16, learned position, CLS, one post-norm encoder layer (2 heads, FFN 32), Linear 16 -> 1;
five seeds, P = their mean. Weights from ``scripts/fit_tfm_calib41.py`` (``tfm_plf.npz`` / ``tfm_plf.json``),
which checked that these weights reproduce the evaluated probabilities and that this numpy forward matches torch.

Rooms: the GT table's ``env`` (data/gt_tables.csv), else the training list's (data/ml_heads_train.csv), else the
room of the capture's day when every labelled capture that day is in one room. A capture of no known room has no
empty reference, so no verdict. Display only; numpy only.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from backend import tfm_series

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "data"
META: dict[str, Any] = json.loads((_HERE / "tfm_plf.json").read_text())
_W = np.load(_HERE / "tfm_plf.npz")

CH: list[str] = META["channels"]
ROOMS: list[str] = META["rooms"]
LOGC = [CH.index(c) for c in META["log_channels"]]
CAL = [CH.index(c) for c in META["calibrated_channels"]]
SEEDS = int(META["seeds"])
THRESHOLD = float(META["threshold"])
_PARAMS = ("inp_w", "inp_b", "pos", "cls", "in_w", "in_b", "out_w", "out_b", "l1_w", "l1_b", "l2_w", "l2_b",
           "n1_g", "n1_b", "n2_g", "n2_b", "h_w", "h_b")
NETS = {f: [{k: _W[f"{f}_s{s}_{k}"].astype(float) for k in _PARAMS} for s in range(SEEDS)] for f in META["folds"]}
FOLD = {f: {k: _W[f"{f}_{k}"].astype(float) for k in ("mu", "sd", "pool")} for f in META["folds"]}
EMPTY = {r: _W[f"emp_{r}"].astype(float) for r in ROOMS}            # the room's empty reference minutes, log applied


def _stats(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD (x 1.4826) per channel over every second of every minute; SD, then 1, where the MAD is 0."""
    K = v.shape[1]
    med = np.array([np.nanmedian(v[:, c]) for c in range(K)])
    mad = np.array([np.nanmedian(np.abs(v[:, c] - med[c])) * 1.4826 for c in range(K)])
    sd = np.array([np.nanstd(v[:, c]) for c in range(K)])
    return med, np.where(mad > 1e-9, mad, np.where(sd > 1e-9, sd, 1.0))


_ROOM_STATS = {r: _stats(EMPTY[r]) for r in ROOMS}


@lru_cache(maxsize=1)
def _room_tables() -> tuple[dict[str, str], dict[str, str]]:
    gt, train = {}, {}
    try:
        with (_DATA / "gt_tables.csv").open(newline="") as fh:
            gt = {r["stem"]: r["env"] for r in csv.DictReader(fh) if r.get("env") in ROOMS}
    except OSError:
        pass
    try:
        with (_DATA / "ml_heads_train.csv").open(newline="") as fh:
            train = {r["key"]: r["ds"] for r in csv.DictReader(fh) if r.get("ds") in ROOMS}
    except OSError:
        pass
    return gt, train


def room_for(stem: str) -> dict[str, Any] | None:
    """{room, fold, calibrated_with, source} for a capture stem, or None when its room is unknown."""
    if stem == META["scnC_stem"]:
        return {"room": "env3-1", "fold": "scnC", "calibrated_with": "env3-1", "source": "scenario C session"}
    gt, train = _room_tables()
    for table, src in ((gt, "GT table"), (train, "training list")):
        if stem in table:
            r = table[stem]
            return {"room": r, "fold": r, "calibrated_with": r, "source": src}
    r = META["day_rooms"].get(stem[:8])
    if r:
        return {"room": r, "fold": r, "calibrated_with": r, "source": f"day {stem[4:6]}/{stem[6:8]} (one room that day)"}
    return None


def _ln(z: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (z - z.mean(1, keepdims=True)) / np.sqrt(z.var(1, keepdims=True) + 1e-5) * g + b


def forward(P: dict[str, np.ndarray], x: np.ndarray) -> float:
    """Logit of one (34, 61) input."""
    z = x.T @ P["inp_w"].T + P["inp_b"] + P["pos"][: x.shape[1]]
    z = np.vstack([P["cls"], z])
    d = z.shape[1]; hh = 2; dh = d // hh
    qkv = z @ P["in_w"].T + P["in_b"]
    q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
    heads = []
    for i in range(hh):
        sl = slice(i * dh, (i + 1) * dh)
        a = q[:, sl] @ k[:, sl].T / np.sqrt(dh)
        a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        heads.append(a @ v[:, sl])
    z = _ln(z + np.hstack(heads) @ P["out_w"].T + P["out_b"], P["n1_g"], P["n1_b"])
    z = _ln(z + np.maximum(z @ P["l1_w"].T + P["l1_b"], 0) @ P["l2_w"].T + P["l2_b"], P["n2_g"], P["n2_b"])
    return float(z[0] @ P["h_w"][0] + P["h_b"][0])


def calibrate(X: np.ndarray, stem: str, room: dict[str, Any]) -> np.ndarray:
    """Log, then (x - empty median) / shrunk empty MAD on the calibrated channels; the plf ratios, rpm and bs_f stay."""
    x = X.copy()
    x[LOGC] = np.log(np.clip(x[LOGC], 1e-6, None))
    ref = room["calibrated_with"]
    keys = META["empty_keys"][ref]
    if stem in keys:                                                  # an empty reference minute: without itself
        med, mad = _stats(np.delete(EMPTY[ref], keys.index(stem), axis=0))
    else:
        med, mad = _ROOM_STATS[ref]
    sd = np.exp(0.5 * np.log(mad) + 0.5 * np.log(FOLD[room["fold"]]["pool"]))
    x[CAL] = (x[CAL] - med[CAL, None]) / sd[CAL, None]
    return x


def predict(Z: np.ndarray, fold: str) -> dict[str, Any]:
    f = FOLD[fold]
    inp = np.concatenate([np.nan_to_num((Z - f["mu"][:, None]) / f["sd"][:, None]), np.isfinite(Z).astype(float)], 0)
    logits = np.array([forward(P, inp) for P in NETS[fold]])
    ps = 1 / (1 + np.exp(-logits))
    p = float(ps.mean())
    return {"p": p, "label": int(p > THRESHOLD), "p_seeds": [float(v) for v in ps]}


def evaluate(path: str | Path, window: int = 0) -> dict[str, Any]:
    """The model on one window, the ML tab's windows (``af8_explain.windows_for``). Raises ``ValueError``."""
    from backend import af8_explain, mtk
    from backend.tiles import get_index

    path = Path(path)
    times = np.asarray(get_index(path).times, dtype=float)
    if times.size < 2:
        raise ValueError("fewer than 2 frames in the capture")
    wins = af8_explain.windows_for(float(times[-1] - times[0]))
    if not 0 <= window < len(wins):
        raise ValueError(f"window {window} out of range: this capture has {len(wins)}")
    t0, t1 = wins[window]
    s = tfm_series.series(path, t0, t1, mtk.MTKIndex(path))
    X = s["X"]
    room = room_for(path.stem)
    out: dict[str, Any] = {"window": int(window), "windows": [[float(a), float(b)] for a, b in wins],
                           "window_s": [float(t0), float(t1)], "channels": CH, "calibrated_channels": META["calibrated_channels"],
                           "log_channels": META["log_channels"], "raw": X, "swing_period_s": float(s["T"]), "room": room}
    if room is None:
        out.update({"calibrated": None, "verdict": None})
        return out
    Z = calibrate(X, path.stem, room)
    out.update({"calibrated": Z, "verdict": predict(Z, room["fold"]),
                "trained_on": META["folds"][room["fold"]]["trained_on"]})
    return out
