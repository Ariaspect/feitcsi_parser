"""The ML tab's top model: 41F hidden layer with the install-time empty-room calibration (shift), one model per
held-out room.

The 41 columns are the anomaly models' (``backend.ml_heads``: 11F + aux + Liu), computed by the classifier tab's
own code. Calibration (ml_lab ``cy_calib4.py``, "empty calib"): log on the columns that are positive in every
training minute, then minus the median of the room's empty minutes -- an empty reference minute without itself.
Model: median imputer, standard scaler, 16 tanh units, logistic output (L2 alpha 3, L-BFGS, balanced classes,
seed 0). As on the Transformer tab, a window is judged by the model that never saw its room (env1 by env2 + env3 +
env3-1, ...; the scenario C session by env1 + env2), calibrated with that room's empties
(``backend.tfm_model.room_for``). Weights and the rooms' empty values from ``scripts/fit_tfm_calib41.py``
(``calib41.json``), which reproduced the evaluated LOEO 88.8 % / scenario C 85.2 %. Display only; numpy only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend import tfm_model

_J = json.loads(Path(__file__).with_name("calib41.json").read_text())
FEATURES: list[str] = _J["features"]
LOG = np.array([f in set(_J["log_features"]) for f in FEATURES])
THRESHOLD = float(_J["threshold"])
ACCURACY: dict[str, Any] = _J["accuracy"]
_EMPTY = {r: (v["keys"], np.array([[np.nan if x is None else x for x in row] for row in v["values"]], dtype=float))
          for r, v in _J["empty"].items()}
_FOLDS = {f: {k: (np.asarray(v, dtype=float) if isinstance(v, list) and k not in ("trained_on",) else v) for k, v in m.items()}
          for f, m in _J["folds"].items()}


def _mlp_logit(m: dict, z: np.ndarray) -> np.ndarray:
    return np.tanh(z @ m["w1"] + m["b1"]) @ m["w2"] + m["b2"]


def predict(values: dict[str, float], stem: str, room: dict[str, Any]) -> dict[str, Any]:
    """Raw, logged and calibrated values, the model's P and per-feature effect (the logit's fall when that input is
    set to its training mean)."""
    x = np.array([values.get(f, np.nan) for f in FEATURES], dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        lx = np.where(LOG, np.log(np.where(x > 0, x, np.nan)), x)
    keys, E = _EMPTY[room["calibrated_with"]]
    ref = np.delete(E, keys.index(stem), axis=0) if stem in keys else E
    with np.errstate(all="ignore"):
        centre = np.nanmedian(ref, axis=0)
    cal = lx - centre
    m = _FOLDS[room["fold"]]
    imputed = ~np.isfinite(cal)
    ci = np.where(imputed, m["median"], cal)
    z = (ci - m["mean"]) / m["scale"]
    logit = float(_mlp_logit(m, z[None, :])[0])
    zs = np.repeat(z[None, :], len(z), axis=0)
    np.fill_diagonal(zs, 0.0)
    effect = logit - _mlp_logit(m, zs)
    p = float(1 / (1 + np.exp(-logit)))
    return {"value": x, "logged": LOG, "centre": centre, "calibrated": cal, "imputed": imputed, "z": z, "effect": effect,
            "p": p, "label": int(p > THRESHOLD), "logit": logit}


def evaluate(path: str | Path, window: int = 0) -> dict[str, Any]:
    """The model on one window, cut as the ML tab cuts it (``ml_heads.evaluate``). Raises ``ValueError``."""
    from backend import af8_explain, classifier
    from backend.tiles import get_index

    path = Path(path)
    times = np.asarray(get_index(path).times, dtype=float)
    if times.size < 2:
        raise ValueError("fewer than 2 frames in the capture")
    origin = float(times[0])
    wins = af8_explain.windows_for(float(times[-1] - origin))
    if not 0 <= window < len(wins):
        raise ValueError(f"window {window} out of range: this capture has {len(wins)}")
    if len(wins) == 1:
        res = classifier.item_values(path)
    else:
        t0, t1 = wins[window]
        res = classifier.item_values(path, origin + t0, origin + t1)
    values = {f: res["values"].get(f) for f in FEATURES}
    values = {f: (float(v) if isinstance(v, (int, float)) and v is not None else float("nan")) for f, v in values.items()}
    room = tfm_model.room_for(path.stem)
    out: dict[str, Any] = {"window": int(window), "windows": [[float(a), float(b)] for a, b in wins],
                           "window_s": [float(res["t0"]) - origin, float(res["t1"]) - origin], "features": FEATURES, "room": room}
    if room is None:
        out.update({"value": np.array([values[f] for f in FEATURES]), "verdict": None})
        return out
    pr = predict(values, path.stem, room)
    out.update(pr)
    out["verdict"] = {"p": pr["p"], "label": pr["label"]}
    out["trained_on"] = _J["folds"][room["fold"]]["trained_on"]
    return out
