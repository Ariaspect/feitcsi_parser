"""Fit the ML tab's human-presence and anomaly-presence models, and their leave-one-environment-out accuracy.

Input: ``data/ml_heads_train.csv`` -- 1,246 labelled one-minute windows, one row each, with every
column the models read, computed by ``backend.classifier.item_values`` (the classifier tab's own code):

* env1 270, env2 442 (the full GT table), env3 275, env3-1 110: whole one-minute captures;
* scnC 149: minutes of the 2.6 h scenario C capture 20261023_132301 (key ``C<w>`` = [60w, 60w + 60) s
  from its first frame), on the corrected timeline; a minute that crosses a change is left out.

Labels:

* ``y`` (human presence): the binary GT (a minute with a person in any part of it is occupied);
* ``obj`` (anomaly presence): a non-human mover in the minute -- the condition names a fan,
  vacuum or robot, with or without a person (scnC: the ceiling fan's timeline).

Features: the 11-feature screen model's inputs (``ml_*``), the classifier bank's measurements
(the aux bank) and the Liu columns -- 43 in all, the same for both labels.

Models, per label: logistic regression (median imputer, standard scaler, C = 1, balanced classes)
and one hidden layer (the same scaling, 16 tanh units, L2 alpha = 3, L-BFGS, balanced sample
weights, seed 0). Both are fitted on every row and written as plain numbers to
``backend/ml_heads.json``; ``backend/ml_heads.py`` evaluates them with numpy alone.

LOEO over the four environments: env1, env2, env3 and env3-1 each held out in turn, trained on the
other three; scenario C is never in their training -- it was recorded on 10/23 with env3-1, and
training on it lifted held-out env3-1 by ~9 points (human LR 84.5 -> 93.6), so it is not an unseen
environment for env3-1. Scenario C itself is tested on a model trained on all four. Pooled and
mean are over the four. Also given, for reference only: the 11 features alone with the same LR,
and the 11 plus the classifier tab's starred columns (``MARKED``) with both models. No
augmentation anywhere.

Run: ``.venv/bin/python scripts/fit_ml_heads.py`` (about a minute).
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "ml_heads_train.csv"
OUT = ROOT / "backend" / "ml_heads.json"

META = ["key", "ds", "cond", "y", "obj"]
ENVS = ["env1", "env2", "env3", "env3-1", "scnC"]
ROOMS = ["env1", "env2", "env3", "env3-1"]          # the LOEO environments; scnC is tested, never trained on
TARGETS = {"human": "y", "anomaly": "obj"}
HIDDEN = 16
ALPHA = 3.0
# The classifier tab's starred columns outside the 11 (backend/feature_marks.py; p_occupied left out:
# it is a fitted output, not a measurement). LOEO reference only -- the models read all 43.
MARKED = ["breath_run20", "breath_peak10", "liu_ms_spread", "liu_step_spread", "liu_coh1", "liu_coh1_breath",
          "delay_spread", "step_norm", "acf_amp_p90"]


def model(kind: str):
    pre = [SimpleImputer(strategy="median"), StandardScaler()]
    if kind == "lr":
        return make_pipeline(*pre, LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"))
    if kind == "mlp":
        return make_pipeline(*pre, MLPClassifier((HIDDEN,), activation="tanh", solver="lbfgs", alpha=ALPHA,
                                                 max_iter=3000, random_state=0))
    raise ValueError(kind)


def balanced(y: np.ndarray) -> np.ndarray:
    w = np.ones(len(y))
    p = y.mean()
    w[y == 1] = 0.5 / p
    w[y == 0] = 0.5 / (1 - p)
    return w


def fit(kind: str, X: np.ndarray, y: np.ndarray):
    m = model(kind)
    if kind == "mlp":
        m.fit(X, y, mlpclassifier__sample_weight=balanced(y))
    else:
        m.fit(X, y)
    return m


def export(kind: str, m) -> dict:
    imp, sc, est = m[0], m[1], m[2]
    out = {"median": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()}
    if kind == "lr":
        out.update(coef=est.coef_[0].tolist(), intercept=float(est.intercept_[0]))
    else:
        out.update(w1=est.coefs_[0].tolist(), b1=est.intercepts_[0].tolist(),
                   w2=est.coefs_[1][:, 0].tolist(), b2=float(est.intercepts_[1][0]))
    return out


def pct(mask: np.ndarray, ok: np.ndarray) -> float | None:
    return round(100.0 * float(ok[mask].mean()), 1) if mask.any() else None


def scores(S: pd.DataFrame, target: str, pred: np.ndarray) -> dict:
    y = S[target].to_numpy(int)
    ok = pred == y
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    out = {
        "n": int(len(y)), "correct": int(ok.sum()), "accuracy": pct(np.ones(len(y), bool), ok),
        "precision": round(100 * prec, 1) if prec is not None else None,
        "recall": round(100 * rec, 1) if rec is not None else None,
        "f1": round(100 * 2 * prec * rec / (prec + rec), 1) if prec and rec else None,
    }
    obj = S.obj.to_numpy(int)
    hum = S.y.to_numpy(int)
    if target == "y":
        out.update(kpi1=pct(obj == 0, ok), kpi2=pct(obj == 1, ok), nonhuman=pct((obj == 1) & (hum == 0), ok))
    else:
        out.update(person_only=pct((obj == 0) & (hum == 1), ok), empty=pct((obj == 0) & (hum == 0), ok))
    out["by_condition"] = {c: [int(ok[(S.cond == c).to_numpy()].sum()), int((S.cond == c).sum())]
                           for c in sorted(S.cond.unique())}
    return out


def loeo(D: pd.DataFrame, cols: list[str], target: str, kind: str) -> dict:
    """Each room held out, trained on the other three (scnC unused); scnC on a model of all four."""
    res, pooled_ok, pooled_n = {}, 0, 0
    for env in ENVS:
        tr = D[D.ds.isin(ROOMS) & (D.ds != env)]
        te = D[D.ds == env]
        m = fit(kind, tr[cols].to_numpy(float), tr[target].to_numpy(int))
        pred = (m.predict_proba(te[cols].to_numpy(float))[:, 1] > 0.5).astype(int)
        res[env] = scores(te, target, pred)
        if env in ROOMS:
            pooled_ok += res[env]["correct"]
            pooled_n += res[env]["n"]
    res["pooled"] = {"n": pooled_n, "correct": pooled_ok, "accuracy": round(100 * pooled_ok / pooled_n, 1)}
    res["mean_of_envs"] = round(float(np.mean([res[e]["accuracy"] for e in ROOMS])), 1)
    return res


def main() -> int:
    D = pd.read_csv(TRAIN)
    cols = [c for c in D.columns if c not in META]
    f11 = [c for c in cols if c.startswith("ml_")]
    assert len(cols) == 43 and len(f11) == 11, (len(cols), len(f11))
    heads, table, check = {}, {}, []
    X = D[cols].to_numpy(float)
    for head, target in TARGETS.items():
        y = D[target].to_numpy(int)
        heads[head] = {}
        table[head] = {}
        for kind in ("lr", "mlp"):
            m = fit(kind, X, y)
            heads[head][kind] = export(kind, m)
            table[head][kind] = loeo(D, cols, target, kind)
            print(head, kind, "LOEO", {e: table[head][kind][e]["accuracy"] for e in ENVS},
                  "pooled", table[head][kind]["pooled"]["accuracy"], flush=True)
        table[head]["lr_11F_reference"] = loeo(D, f11, target, "lr")
        print(head, "11F LR (reference)", {e: table[head]["lr_11F_reference"][e]["accuracy"] for e in ENVS}, flush=True)
        for kind in ("lr", "mlp"):
            table[head][f"{kind}_marked_reference"] = loeo(D, f11 + MARKED, target, kind)
            print(head, f"11F + marked {kind} (reference)",
                  {e: table[head][f"{kind}_marked_reference"][e]["accuracy"] for e in ENVS},
                  "pooled", table[head][f"{kind}_marked_reference"]["pooled"]["accuracy"], flush=True)
    # a few rows with the fitted pipeline's own probabilities, so the numpy port can be checked
    for i in (0, 300, 700, 1000, 1200):
        row = {"key": D.key.iloc[i], "x": [None if not np.isfinite(v) else float(v) for v in X[i]], "p": {}}
        for head, target in TARGETS.items():
            y = D[target].to_numpy(int)
            for kind in ("lr", "mlp"):
                row["p"][f"{head}_{kind}"] = float(fit(kind, X, y).predict_proba(X[i:i + 1])[0, 1])
        check.append(row)
    out = {
        "fitted": date.today().isoformat(),
        "script": "scripts/fit_ml_heads.py",
        "train": "data/ml_heads_train.csv",
        "features": cols,
        "threshold": 0.5,
        "hidden": HIDDEN,
        "trained_on": {e: {"n": int((D.ds == e).sum()), "human": int(D[D.ds == e].y.sum()),
                           "anomaly": int(D[D.ds == e].obj.sum())} for e in ENVS},
        "heads": heads,
        "loeo": table,
        "check": check,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
