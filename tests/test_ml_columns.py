"""Tests for backend.ml_columns -- the ML tab's 11 inputs as classifier-tab columns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend import af8_env1, classifier, ml_columns

REAL = Path("/home/cyphy/feitcsi_parser/captures/20261021_172407.bin")


def test_the_columns_come_last_display_only_and_name_the_models_inputs() -> None:
    keys = [f["key"] for f in classifier.FEATURES]
    ml = [f["key"] for f in ml_columns.FEATURES]
    assert keys[-len(ml):] == ml
    assert ml == [f"ml_{k}" for k in af8_env1.FEATURES] + ["ml_p_person"]
    for f in ml_columns.FEATURES:
        assert f["status"] == "display only" and f["reference"] is None
    assert not set(ml) & set(classifier.LR_FEATURES) and not set(ml) & set(classifier.HUMAN_FEATURES)


def test_a_window_the_model_cannot_judge_is_blank_not_an_error(bank_capture: Path, bank_units: dict) -> None:
    blank = ml_columns.unit_features(bank_capture, 0.0, 5.0)   # under the model's 30 s minimum
    assert set(blank) == set(ml_columns.KEYS) and all(np.isnan(v) for v in blank.values())
    for u in bank_units["units"]:
        assert set(ml_columns.KEYS) <= set(u)
        assert np.isnan(u["ml_p_person"]) or 0.0 <= u["ml_p_person"] <= 1.0


@pytest.mark.skipif(not REAL.is_file(), reason="needs the env3 capture 20261021_172407 on this machine")
def test_a_whole_capture_reads_what_the_ml_tab_reads() -> None:
    classifier.reset_cache()
    v = classifier.item_values(REAL)["values"]
    ref = af8_env1.evaluate(REAL)
    assert v["ml_p_person"] == pytest.approx(ref["p_person"], abs=1e-9)
    for k in af8_env1.FEATURES:
        r = ref["features"][k]
        assert (r is None and np.isnan(v[f"ml_{k}"])) or v[f"ml_{k}"] == pytest.approx(r, rel=1e-9, abs=1e-12)
