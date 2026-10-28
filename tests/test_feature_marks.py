"""Tests for backend.feature_marks -- which columns are marked, and why."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend import classifier, feature_marks as M
from backend.app import app


def test_every_note_names_a_bank_column_with_evidence() -> None:
    keys = {f["key"] for f in classifier.FEATURES}
    for k, note in M.NOTES.items():
        assert k in keys, k
        assert k in M.AUC, k
        assert note["explain"] and note["strength"] and note["weakness"]


def test_stars_are_read_off_the_evidence() -> None:
    assert M._level([0.95, 0.91, 1.0]) == (2, "higher")
    assert M._level([0.05, 0.0, 0.08]) == (2, "lower")
    assert M._level([0.8, 0.9, 0.76]) == (1, "higher")
    assert M._level([0.8, 0.4, 0.9]) is None                 # rooms disagree: no mark
    assert M._level([0.7, 0.72, 0.74]) is None               # agrees but too close to chance
    assert M._level([0.95, None, 0.92]) == (2, "higher")     # a room without that pair is skipped


def test_marks_are_set_on_the_bank_and_only_on_noted_columns() -> None:
    for f in classifier.FEATURES:
        if f["key"] in M.NOTES:
            assert f["mark"] is not None and f["mark"]["stars"] >= 1, f["key"]
            for t in f["mark"]["tags"]:
                assert t["pair"] in M.PAIR_LABEL and t["direction"] in ("higher", "lower")
        else:
            assert f["mark"] is None
    best = next(f for f in classifier.FEATURES if f["key"] == "ml_p_person")["mark"]
    assert best["stars"] == 2
    assert any(t["pair"] == "rfan|+still" for t in best["tags"])


def test_the_feature_list_endpoint_carries_the_marks() -> None:
    body = TestClient(app).get("/api/classifier/features").json()
    marked = {f["key"] for f in body["features"] if f.get("mark")}
    assert marked == set(M.NOTES)
