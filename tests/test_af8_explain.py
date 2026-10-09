"""Tests for backend.af8_explain and /api/af8 -- what the ML tab draws."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import af8, af8_explain
from backend.app import app
from tests.test_hybrid2 import _capture


def test_a_short_capture_is_one_window_and_a_long_one_is_cut_in_minutes() -> None:
    assert af8_explain.windows_for(59.9) == [(0.0, af8.WINDOW_SECONDS)]
    assert af8_explain.windows_for(200.0) == [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0)]


@pytest.mark.parametrize("n, window, span", [
    (1200, 0, (0.0, af8.WINDOW_SECONDS)),   # 60 s at 20 Hz: the one-minute case
    (4000, 1, (60.0, 120.0)),               # 200 s: the second 60 s window
])
def test_the_features_are_af8s_own(tmp_path: Path, n: int, window: int, span) -> None:
    """The tab must never show a number the model did not see."""
    p = _capture(tmp_path, n=n)
    out = af8_explain.explain(p, window)
    ref = af8.evaluate(p, *span)
    assert out["window_s"] == list(span)
    assert out["features"] == ref["features"]
    assert out["p_person"] == ref["p_person"]
    assert out["label"] == ref["label"]


def test_the_series_drawn_give_back_the_features(tmp_path: Path) -> None:
    """Each panel's series, read the way af8 reads it, is that panel's feature."""
    out = af8_explain.explain(_capture(tmp_path, n=1200))
    f, mo, br = out["features"], out["motion"], out["breath"]

    # A_p90: the 90th percentile of the drawn per-second blocks.
    blocks = np.array([v for v in mo["p90_values"] if v is not None])
    assert np.percentile(blocks, 90) == pytest.approx(f["A_p90"], rel=1e-12)

    # The shape: the drawn D at the three model lags, and the fitted line through them.
    D = dict(zip(mo["lags"], mo["D"]))
    assert D[0.25] / D[2.0] == pytest.approx(f["A_r025_2"], rel=1e-12)
    assert D[5.0] / D[2.0] == pytest.approx(f["A_r5_2"], rel=1e-12)
    fit = f["A_slope3"] * np.log(mo["shape_lags"]) + mo["fit_intercept"]
    assert np.mean(fit) == pytest.approx(np.mean(np.log([D[t] for t in mo["shape_lags"]])), rel=1e-9)

    # The breathing panels: max, median, longest run of peaks >= 0.25.
    peaks = np.array([np.nan if v is None else v for v in br["peak"]])
    assert np.nanmax(peaks) == pytest.approx(f["F_pkmax"], rel=1e-12)
    assert np.nanmedian(peaks) == pytest.approx(f["F_pkmed"], rel=1e-12)
    assert (br["run"]["windows"] if br["run"] else 0) == f["F_run"]

    # C3: the largest drop of the drawn D below its drawn running maximum.
    rv = out["revisit"]
    drop = np.array([np.nan if v is None else v for v in rv["drop"]])
    assert np.nanmax(drop) == pytest.approx(f["C3_revisit"], rel=1e-12)
    assert rv["running_max"] == list(np.maximum.accumulate(rv["D"]))


def test_the_model_breakdown_adds_up(tmp_path: Path) -> None:
    out = af8_explain.explain(_capture(tmp_path, n=1200))
    m = out["model"]
    assert m["logit"] == pytest.approx(m["intercept"] + sum(m["contribution"]), rel=1e-12)
    assert 1 / (1 + np.exp(-m["logit"])) == pytest.approx(out["p_person"], rel=1e-12)


def test_the_endpoint(tmp_path: Path) -> None:
    p = _capture(tmp_path, n=4000)
    client = TestClient(app)
    res = client.get("/api/af8", params={"path": str(p), "window": 1})
    assert res.status_code == 200, res.text
    assert "NaN" not in res.text
    body = res.json()
    assert body["window"] == 1 and len(body["windows"]) == 3
    assert set(body["features"]) == set(af8.FEATURES)
    assert body["camera"] is not None                 # the fixture carries a _cv.json
    assert client.get("/api/af8", params={"path": str(p), "window": 3}).status_code == 400
