"""Tests for backend.ml_heads -- the ML tab's human-presence and anomaly-presence models (11F + aux + Liu)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend import classifier, ml_heads

ROOT = Path(__file__).resolve().parents[1]
REAL = Path("/home/cyphy/feitcsi_parser/captures/20261021_172407.bin")
WEIGHTS = json.loads((ROOT / "backend" / "ml_heads.json").read_text())


def test_both_labels_read_the_same_43_columns_the_bank_computes() -> None:
    train = pd.read_csv(ROOT / "data" / "ml_heads_train.csv", nrows=1)
    cols = [c for c in train.columns if c not in ("key", "ds", "cond", "y", "obj")]
    assert list(ml_heads.FEATURES) == cols and len(cols) == 43
    assert sum(c.startswith("ml_") for c in cols) == 11 and sum(c.startswith("liu_") for c in cols) == 14
    bank = {f["key"] for f in classifier.FEATURES} | set(ml_heads.LABEL_FALLBACK)
    assert set(cols) <= bank
    for h in ml_heads.HEADS:
        assert set(WEIGHTS["heads"][h]) == set(ml_heads.KINDS)


def test_the_numpy_port_gives_the_fitted_pipelines_probabilities() -> None:
    for row in WEIGHTS["check"]:
        values = {f: (np.nan if v is None else v) for f, v in zip(ml_heads.FEATURES, row["x"])}
        out = ml_heads.predict(values)
        for h in ml_heads.HEADS:
            for k in ml_heads.KINDS:
                assert out[h][k]["p"] == pytest.approx(row["p"][f"{h}_{k}"], abs=1e-9)
                assert out[h][k]["label"] == int(out[h][k]["p"] > ml_heads.THRESHOLD)


def test_lr_pushes_add_up_and_the_hidden_layers_are_one_at_a_time() -> None:
    values = dict(zip(ml_heads.FEATURES, WEIGHTS["check"][0]["x"]))
    values = {f: (np.nan if v is None else v) for f, v in values.items()}
    out = ml_heads.predict(values)
    for h in ml_heads.HEADS:
        lr = out[h]["lr"]
        assert lr["logit"] == pytest.approx(WEIGHTS["heads"][h]["lr"]["intercept"] + lr["effect"].sum())
        mlp = out[h]["mlp"]
        i = int(np.argmax(np.abs(mlp["effect"])))
        moved = dict(values)
        moved[ml_heads.FEATURES[i]] = WEIGHTS["heads"][h]["mlp"]["mean"][i]      # z = 0
        assert ml_heads.predict(moved)[h]["mlp"]["logit"] == pytest.approx(mlp["logit"] - mlp["effect"][i])


def test_a_missing_input_takes_the_training_median_and_is_flagged() -> None:
    values = {f: np.nan for f in ml_heads.FEATURES}
    out = ml_heads.predict(values)
    assert out["imputed"].all()
    for h in ml_heads.HEADS:
        for k in ml_heads.KINDS:
            m = WEIGHTS["heads"][h][k]
            z = (np.asarray(m["median"]) - np.asarray(m["mean"])) / np.asarray(m["scale"])
            assert np.allclose(out[h][k]["z"], z) and 0.0 <= out[h][k]["p"] <= 1.0


def test_the_anomaly_label_is_a_named_mover() -> None:
    assert ml_heads.anomaly_truth("rot fan + still") == 1
    assert ml_heads.anomaly_truth("Robot (empty)") == 1
    assert ml_heads.anomaly_truth("vacuum") == 1
    assert ml_heads.anomaly_truth("2 people + ceiling fan") == 1
    assert ml_heads.anomaly_truth("Static occupancy") == 0
    assert ml_heads.anomaly_truth(None) is None


def test_loeo_holds_every_room_out_and_tests_scenario_c_for_both_labels_and_models() -> None:
    envs = {"env1", "env2", "env3", "env3-1", "scnC"}
    for h in ml_heads.HEADS:
        for k in ("lr", "mlp", "lr_11F_reference", "lr_marked_reference", "mlp_marked_reference"):
            t = ml_heads.LOEO[h][k]
            rooms = envs - {"scnC"}
            assert envs <= set(t) and t["pooled"]["n"] == sum(t[e]["n"] for e in rooms) == 1246 - 149
            assert 0 <= t["pooled"]["accuracy"] <= 100


@pytest.mark.skipif(not REAL.is_file(), reason="needs the env3 capture 20261021_172407 on this machine")
def test_the_endpoint_reads_a_capture_with_both_labels() -> None:
    from fastapi.testclient import TestClient

    from backend.app import app

    classifier.reset_cache()
    r = TestClient(app).get("/api/ml/heads", params={"path": str(REAL)})
    assert r.status_code == 200
    j = r.json()
    assert j["features"] == list(ml_heads.FEATURES) and len(j["value"]) == 43
    for h in ("human", "anomaly"):
        for k in ("lr", "mlp"):
            m = j["heads"][h][k]
            assert 0.0 <= m["p"] <= 1.0 and m["label"] in (0, 1) and len(m["effect"]) == 43
    assert j["truth"]["human"] in ("person", "empty", None)
    assert set(j["labels"]) == set(ml_heads.FEATURES)
    # the window is the one the training rows were computed on: the whole one-minute capture
    v = classifier.item_values(REAL)["values"]
    assert j["value"][ml_heads.FEATURES.index("liu_coh1")] == pytest.approx(v["liu_coh1"])
