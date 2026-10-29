"""The ML tab's human-presence and anomaly-presence models: 11F + aux + Liu, each as LR and as one hidden layer.

Two labels, two models each, all reading the same 43 columns of one 60 s window:

* ``human``: is a person in the window (the binary GT);
* ``anomaly``: is a non-human mover in it -- a fan, a vacuum / robot, a ceiling fan -- with or
  without a person.

The columns are the 11-feature screen model's inputs (``ml_*``, ``backend.ml_columns``), the
classifier bank's measurements (the aux bank: step, jitter, breathing, ACF, eigen, delay,
revisit) and the Liu columns (``backend.liu_features``), all computed by
``backend.classifier.item_values`` -- the classifier tab's own code, so the tab and the models
can never disagree about a number.

Models (``scripts/fit_ml_heads.py``, weights in ``ml_heads.json``): median imputer and standard
scaler, then either a logistic regression or 16 tanh units and a logistic output; both trained
with balanced classes on 1,246 labelled minutes of env1, env2, env3, env3-1 and scenario C,
without augmentation. ``ml_heads.json`` also carries the leave-one-environment-out accuracy.

Display only: nothing else reads these models; the 11-feature screen model (``af8_env1``) is
unchanged and is still the ML tab's first verdict. numpy only, like the board.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_W = json.loads(Path(__file__).with_name("ml_heads.json").read_text())

FEATURES: tuple[str, ...] = tuple(_W["features"])
HEADS = ("human", "anomaly")
KINDS = ("lr", "mlp")
THRESHOLD: float = float(_W["threshold"])
LOEO: dict[str, Any] = _W["loeo"]
TRAINED_ON: dict[str, Any] = _W["trained_on"]
FITTED: str = _W["fitted"]

# Labels for the columns the classifier bank does not list in its FEATURES (it computes them for its own logistic).
LABEL_FALLBACK = {
    "log_step_norm_own": {"label": "ln step / jitter", "test": "bank",
                          "description": "Natural log of step / jitter (step P90 over the unit's own link jitter); an input of the bank's own logistic."},
}

# A recorded condition that names a non-human mover -- the anomaly label the models were trained on.
MOVER = re.compile(r"fan|vacuum|robot", re.I)


def _arr(m: dict, k: str) -> np.ndarray:
    return np.asarray(m[k], dtype=float)


_M = {h: {k: {key: (_arr(m, key) if isinstance(m[key], list) else float(m[key])) for key in m}
              for k, m in _W["heads"][h].items()} for h in HEADS}


def _sigmoid(v: float) -> float:
    return float(1.0 / (1.0 + np.exp(-v)))


def _mlp_logit(m: dict, z: np.ndarray) -> np.ndarray:
    """Output logit for each row of z (rows x features)."""
    return np.tanh(z @ m["w1"] + m["b1"]) @ m["w2"] + m["b2"]


def predict(values: dict[str, float]) -> dict[str, Any]:
    """Every model on one window's columns.

    Returns, per head and model, ``p``, ``label`` (p > 0.5) and ``logit``, with ``z`` (the scaled
    inputs) and ``effect`` (per feature): for LR its exact push ``coef * z``; for the hidden layer
    how much the logit falls when that one input is set to its training mean (z = 0). ``imputed``
    marks the inputs that were missing and took the training median.
    """
    x = np.array([values.get(f, np.nan) for f in FEATURES], dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    imputed = np.isnan(x)
    out: dict[str, Any] = {"value": x, "imputed": imputed}
    for h in HEADS:
        out[h] = {}
        for k in KINDS:
            m = _M[h][k]
            xi = np.where(imputed, m["median"], x)
            z = (xi - m["mean"]) / m["scale"]
            if k == "lr":
                effect = m["coef"] * z
                logit = float(m["intercept"] + effect.sum())
            else:
                logit = float(_mlp_logit(m, z[None, :])[0])
                zs = np.repeat(z[None, :], len(z), axis=0)
                np.fill_diagonal(zs, 0.0)
                effect = logit - _mlp_logit(m, zs)
            p = _sigmoid(logit)
            out[h][k] = {"p": p, "label": int(p > THRESHOLD), "logit": logit, "z": z, "effect": effect}
    return out


def evaluate(path: str | Path, window: int = 0) -> dict[str, Any]:
    """The four models on one window of a capture, the ML tab's windows (``af8_explain.windows_for``).

    A capture under 90 s is one window, the whole capture -- as the training rows were computed;
    a longer one is cut into [60w, 60(w + 1)) s from its first frame. Raises ``ValueError`` when
    the window cannot be read.
    """
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
    pr = predict(values)
    return {
        "window": int(window),
        "windows": [[float(a), float(b)] for a, b in wins],
        "window_s": [float(res["t0"]) - origin, float(res["t1"]) - origin],
        "features": list(FEATURES),
        "value": pr["value"],
        "imputed": pr["imputed"],
        **{h: pr[h] for h in HEADS},
    }


def anomaly_truth(condition: str | None) -> int | None:
    """1 when a recorded condition names a non-human mover, 0 when it names none, None when there is no condition."""
    if not condition:
        return None
    return int(bool(MOVER.search(condition)))
