"""Tests for backend.hybrid2 -- the frame step for motion, FarSense for breath."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import framediff, hybrid, hybrid2
from backend.app import app
from tests.test_mtk import band, group_records, write

FS = 20.0


def _capture(tmp_path: Path, n: int = 4000) -> Path:
    """200 s at 20 Hz: still, then a moving occupant from 80 s to 140 s.

    The modulation rides slot 0 AND slot 1 so both the raw amplitude and the
    ratio move -- this tab reads the ratio, the frame-step tests read the
    amplitude, and one fixture should serve a reader of either.
    """
    tpi0 = band(seed=1)
    tpi1 = band(seed=2)
    rng = np.random.default_rng(7)
    blob = []
    for g in range(n):
        t = 0.05 * g
        mod = 1.0 + 0.01 * rng.standard_normal()
        if 80.0 <= t < 140.0:
            mod *= 1.0 + rng.uniform(-0.5, 0.5)
        blob.append(group_records(
            g, {(0, 0): tpi0 * mod, (1, 0): tpi1 * (2.0 - mod)}, ts=1000 + 50 * g,
        ))
    p = write(tmp_path, b"".join(blob), "hybrid2.bin")
    base = 1_700_000_000.0
    frames = [
        {"epoch": base + s, "n": 1 if 80 <= s < 140 else 0,
         "max_conf": 0.9 if 80 <= s < 140 else 0.0}
        for s in range(200)
    ]
    p.with_name("hybrid2_cv.json").write_text(json.dumps({"frames": frames}))
    return p


# --------------------------------------------------------------------------- #
#  What it inherits, and what it replaces                                      #
# --------------------------------------------------------------------------- #


def test_the_absolute_floor_is_exactly_half_of_hybrid_ones() -> None:
    """Arithmetic, not tuning: the metric is |dr|/|r| over two, so a threshold
    written in its units is the same threshold halved."""
    assert hybrid2.MOTION_ABS == hybrid.MOTION_ABS / 2.0
    assert hybrid2.MOTION_REL == hybrid.MOTION_REL


def test_the_motion_channel_is_the_frame_step_not_the_old_one(tmp_path: Path) -> None:
    """Both are carried, and they are not the same series -- the second is kept
    only so a reader can see the channel this tab replaced."""
    hybrid2.reset_cache()
    p = _capture(tmp_path)
    ev = hybrid2.capture_evidence(p, 0.0, 200.0)
    mine = np.asarray(ev["motion_ratio"], dtype=float)
    theirs = np.asarray(ev["motion_reference"], dtype=float)
    both = np.isfinite(mine) & np.isfinite(theirs)
    assert both.sum() > 100
    assert not np.allclose(mine[both], theirs[both])
    # ...but they measure the same thing, so they move together
    assert np.corrcoef(mine[both], theirs[both])[0, 1] > 0.9
    assert ev["signal"] == "ratio_complex"


def test_the_breathing_half_is_hybrid_ones_verbatim(tmp_path: Path) -> None:
    """The one invariant worth asserting across the two tabs: if these ever
    differ, a disagreement between them could be the breath rather than the
    motion, and neither panel would say so.
    """
    hybrid2.reset_cache()
    hybrid.reset_cache()
    p = _capture(tmp_path)
    a = hybrid.capture_evidence(p, 0.0, 200.0)
    b = hybrid2.capture_evidence(p, 0.0, 200.0)
    for key in ("breath_peak", "breath_rpm"):
        x, y = np.asarray(a[key], dtype=float), np.asarray(b[key], dtype=float)
        assert x.shape == y.shape
        finite = np.isfinite(x) & np.isfinite(y)
        assert np.array_equal(np.isfinite(x), np.isfinite(y))
        assert np.allclose(x[finite], y[finite])


def test_hybrid_ones_amplitude_channel_is_not_in_play(tmp_path: Path) -> None:
    p = _capture(tmp_path)
    out = hybrid2.compute_hybrid2(p, 0.0, 200.0)
    assert np.isnan(np.asarray(out["motion_amp"], dtype=float)).all()
    assert out["params"]["use_amplitude"] is False


# --------------------------------------------------------------------------- #
#  The verdict                                                                 #
# --------------------------------------------------------------------------- #


def _score(p: Path, **kw) -> dict:
    from backend import truth as truthmod
    from backend.app import _camera_truth

    out = hybrid2.compute_hybrid2(p, 0.0, 200.0, **kw)
    cam = _camera_truth(p)
    cells, excluded = truthmod.cell_truth(
        out["time_s"], cam[:, 0], cam[:, 1] > 0.5, 0.5, 5.0)
    return truthmod.confusion(cells, out["present"], excluded) | {"out": out}


def test_it_finds_every_second_of_the_visit(tmp_path: Path) -> None:
    hybrid2.reset_cache()
    c = _score(_capture(tmp_path))
    assert c["recall"] == 1.0
    assert c["fn"] == 0


def test_the_motion_channel_alone_is_clean_on_this_fixture(tmp_path: Path) -> None:
    """Specificity is scored with the breathing channel OFF, deliberately.

    This fixture's quiet stretch is white noise, and FarSense finds a chest in
    it -- 19 seconds of it -- which `lead_hold` then extends twenty seconds
    backwards, so presence starts 54 s before the occupant does. That is the
    empty-room-fakes-breathing artefact the real corpus shows too, not a fault
    in the wiring, so the motion channel is scored on its own here and the
    breathing channel is scored by the tests that own it.
    """
    hybrid2.reset_cache()
    c = _score(_capture(tmp_path), breath_min_peak=1.0, lead_hold=False)
    assert c["recall"] > 0.9
    assert c["specificity"] > 0.8


def test_the_breathing_channel_is_what_widens_presence(tmp_path: Path) -> None:
    """The counterpart: turning it back on can only ADD presence, never remove
    it, which is what 'motion OR breathing' means."""
    hybrid2.reset_cache()
    p = _capture(tmp_path)
    motion_only = np.asarray(
        _score(p, breath_min_peak=1.0, lead_hold=False)["out"]["present"], dtype=bool)
    both = np.asarray(_score(p)["out"]["present"], dtype=bool)
    assert both.sum() > motion_only.sum()
    assert np.all(both | ~motion_only)


def test_the_floor_is_the_range_s_own_unless_told_otherwise(tmp_path: Path) -> None:
    """The known failure is worth a test rather than only a docstring: an
    explicit floor overrides the percentile, which is the only lever a
    fully-occupied range has."""
    p = _capture(tmp_path)
    own = hybrid2.compute_hybrid2(p, 0.0, 200.0)
    forced = hybrid2.compute_hybrid2(p, 0.0, 200.0, motion_floor=1e-6)
    assert own["ratio_floor"] > forced["ratio_floor"]
    assert forced["ratio_threshold"] == pytest.approx(hybrid2.MOTION_ABS)
    assert int(np.sum(forced["present"])) >= int(np.sum(own["present"]))


# --------------------------------------------------------------------------- #
#  The lag                                                                     #
# --------------------------------------------------------------------------- #


def test_the_lag_differences_frames_further_apart(tmp_path: Path) -> None:
    hybrid2.reset_cache()
    p = _capture(tmp_path)
    near = hybrid2.compute_hybrid2(p, 0.0, 200.0, lag_seconds=0.0)
    far = hybrid2.compute_hybrid2(p, 0.0, 200.0, lag_seconds=2.0)
    assert near["lag_frames"] == 1
    assert far["lag_frames"] == pytest.approx(round(2.0 * FS), abs=2)
    assert len(near["time_s"]) == len(far["time_s"])      # same per-second grid


def test_a_lag_longer_than_the_range_is_refused(tmp_path: Path) -> None:
    p = _capture(tmp_path, n=200)
    with pytest.raises(ValueError):
        hybrid2.compute_hybrid2(p, 0.0, 2.0, lag_seconds=10.0)


# --------------------------------------------------------------------------- #
#  The endpoint                                                                #
# --------------------------------------------------------------------------- #


def test_the_endpoint_serves_a_verdict_and_scores_it(tmp_path: Path) -> None:
    p = _capture(tmp_path)
    res = TestClient(app).get("/api/hybrid2", params={
        "path": str(p), "t0": 0.0, "t1": 200.0})
    assert res.status_code == 200, res.text
    b = res.json()
    n = len(b["time_s"])
    for key in ("present", "state", "unknown", "motion", "motion_reference",
                "burst", "breathing", "breath_peak", "breath_rpm"):
        assert len(b[key]) == n, key
    assert b["signal"] == "ratio_complex"
    assert b["confusion"]["recall"] > 0.8
    assert b["floor_scope"] == "own"
    assert "full width" in b["selection_note"]
    assert "NaN" not in res.text


def test_the_endpoint_reports_the_lag_it_used(tmp_path: Path) -> None:
    p = _capture(tmp_path)
    client = TestClient(app)
    a = client.get("/api/hybrid2", params={"path": str(p), "t0": 0.0, "t1": 200.0}).json()
    b = client.get("/api/hybrid2", params={
        "path": str(p), "t0": 0.0, "t1": 200.0, "lag_s": 2.0}).json()
    assert a["lag_frames"] == 1 and a["lag_seconds"] == 0.0
    assert b["lag_frames"] > 1 and b["lag_seconds"] == 2.0


def test_the_endpoint_refuses_an_inverted_rate_band(tmp_path: Path) -> None:
    p = _capture(tmp_path, n=400)
    res = TestClient(app).get("/api/hybrid2", params={
        "path": str(p), "t0": 0.0, "t1": 200.0, "rpm_lo": 30, "rpm_hi": 10})
    assert res.status_code == 400


def test_range_verdict_motion_then_breath_then_empty() -> None:
    from backend import hybrid2
    quiet = np.full(60, 0.011)
    nopeak = np.full(60, 0.05)
    assert hybrid2.range_verdict(quiet, nopeak)["present"] is False
    moving = quiet.copy(); moving[10:20] = 0.09          # 10 of 60 s above -> P90 clears 0.035
    v = hybrid2.range_verdict(moving, nopeak); assert v["present"] and v["by"] == "motion"
    peaks = nopeak.copy(); peaks[30:35] = 0.3            # five consecutive windows at 0.3
    v = hybrid2.range_verdict(quiet, peaks); assert v["present"] and v["by"] == "breathing" and v["breath_run"] == 5
    peaks[32] = 0.1                                      # broken run -> 2 + 2
    assert hybrid2.range_verdict(quiet, peaks)["present"] is False
    assert hybrid2.range_verdict(np.full(60, np.nan), nopeak)["present"] is False


def test_range_series_is_the_rule_on_a_trailing_window() -> None:
    from backend import hybrid2
    n = 200
    lv = np.full(n, 0.011); pk = np.full(n, 0.05); unk = np.zeros(n, dtype=bool)
    lv[100:110] = 0.09                      # 10 s of motion
    pk[150:155] = 0.3                       # five consecutive breathing windows
    out = hybrid2.range_series(lv, pk, unk, window_seconds=60)
    st = np.array(out["state"]); pres = out["present"]
    assert (st[:29] == "unknown").all() and st[29] != "unknown"      # half the window known
    assert not pres[30:100].any()
    assert (st[109:160] == "present:motion").all()                  # motion holds while the 10 s sit inside the trailing 60
    assert st[170] == "present:breathing"                            # motion left the window, the breathing run is still inside
    assert st[199] == "present:breathing"
    assert out["window_seconds"] == 60.0
