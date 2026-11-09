"""Tests for backend.tfm_model / backend.tfm_series (the Transformer tab) and backend.calib41 (the ML tab's 41F calibrated model)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend import calib41, tfm_model, tfm_series

ROOT = Path(__file__).resolve().parents[1]
TFM = json.loads((ROOT / "backend" / "tfm_plf.json").read_text())
C41 = json.loads((ROOT / "backend" / "calib41.json").read_text())


def test_channels_and_folds() -> None:
    assert list(tfm_series.CH) == tfm_model.CH and len(tfm_model.CH) == 17
    assert set(tfm_model.META["calibrated_channels"]).isdisjoint({"rpm", "bs_f", "plf_res", "plf_g"})
    for f, m in TFM["folds"].items():
        assert len(tfm_model.NETS[f]) == 5
        assert f not in m["trained_on"]
    assert TFM["folds"]["scnC"]["trained_on"] == ["env1", "env2"]
    assert calib41.FEATURES == C41["features"] and len(calib41.FEATURES) == 41


def test_rooms() -> None:
    assert tfm_model.room_for("20261023_132301")["fold"] == "scnC"
    assert tfm_model.room_for("20261013_031054")["room"] == "env1"
    assert tfm_model.room_for("20260101_000000") is None


def test_transformer_reproduces_the_evaluated_probabilities() -> None:
    for row in TFM["check"]:
        X = np.array([[np.nan if v is None else v for v in r] for r in row["X"]], dtype=float)
        room = tfm_model.room_for(row["stem"])
        assert tfm_model.predict(tfm_model.calibrate(X, row["stem"], room), room["fold"])["p"] == pytest.approx(row["p"], abs=1e-9)


def test_an_empty_reference_minute_is_calibrated_without_itself() -> None:
    row = TFM["check"][0]                      # an env1 empty minute, one of env1's references
    assert row["stem"] in TFM["empty_keys"]["env1"]
    X = np.array([[np.nan if v is None else v for v in r] for r in row["X"]], dtype=float)
    room = tfm_model.room_for(row["stem"])
    loo = tfm_model.calibrate(X, row["stem"], room)
    full = tfm_model.calibrate(X, "not-a-reference", room)
    assert not np.allclose(np.nan_to_num(loo), np.nan_to_num(full))
    assert np.allclose(np.nan_to_num(loo[[tfm_model.CH.index("plf_res")]]), np.nan_to_num(X[[tfm_model.CH.index("plf_res")]]))


def test_calib41_reproduces_sklearn_and_pushes_are_one_at_a_time() -> None:
    for row in C41["check"]:
        values = {f: (np.nan if v is None else v) for f, v in zip(calib41.FEATURES, row["values"])}
        out = calib41.predict(values, row["stem"], tfm_model.room_for(row["stem"]))
        assert out["p"] == pytest.approx(row["p"], abs=1e-9)
        assert out["label"] == int(out["p"] > calib41.THRESHOLD) and len(out["effect"]) == 41


def test_calib41_missing_input_is_imputed() -> None:
    row = C41["check"][1]
    values = {f: (np.nan if v is None else v) for f, v in zip(calib41.FEATURES, row["values"])}
    values[calib41.FEATURES[0]] = np.nan
    out = calib41.predict(values, row["stem"], tfm_model.room_for(row["stem"]))
    assert out["imputed"][0] and 0.0 <= out["p"] <= 1.0
