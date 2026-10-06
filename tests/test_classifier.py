"""Tests for backend.classifier -- the one-minute feature bank."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import classifier, hybrid2
from backend.app import app
from tests.test_hybrid2 import _capture
from tests.test_mtk import band, group_records, write

T = 84


# --------------------------------------------------------------------------- #
#  The statistic                                                              #
# --------------------------------------------------------------------------- #


def test_white_noise_sits_at_the_null_whatever_its_variance() -> None:
    """WiDetect's claim, and the reason the feature exists: ψ̂ of white noise
    is N(−1/T, 1/(F·T)) and does not move when the noise gets 10x louder."""
    rng = np.random.default_rng(3)
    n, F = T * 200, 64
    times = np.arange(n) / 42.0
    quiet = rng.standard_normal((n, F))
    loud = 10.0 * rng.standard_normal((n, F))
    _, psi_q = classifier.acf_windows(quiet, times, T)
    _, psi_l = classifier.acf_windows(loud, times, T)
    null_mean, null_sd = -1.0 / T, 1.0 / np.sqrt(F * T)
    # Detrending costs a little bias on top of −1/T; allow a few null widths.
    assert abs(psi_q.mean() - null_mean) < 4 * null_sd / np.sqrt(psi_q.size) + 0.01
    assert abs(psi_l.mean() - psi_q.mean()) < 0.01
    assert abs(psi_q.std() - null_sd) < 0.5 * null_sd


def test_slow_coherent_variation_reads_high() -> None:
    """A 0.3 Hz swing shared by every subcarrier -- a breathing sitter's
    signature -- is what the statistic is for."""
    rng = np.random.default_rng(5)
    n, F = T * 30, 32
    times = np.arange(n) / 42.0
    swing = np.sin(2 * np.pi * 0.3 * times)[:, None]
    X = swing + 0.1 * rng.standard_normal((n, F))
    _, psi = classifier.acf_windows(X, times, T)
    assert np.nanmedian(psi) > 0.8


def test_a_window_across_a_gap_is_blank_and_a_dead_subcarrier_is_skipped() -> None:
    rng = np.random.default_rng(1)
    n, F = T * 6, 16
    times = np.arange(n) / 42.0
    times[3 * T + 10:] += 2.0                     # a two-second dropout inside block 3
    X = rng.standard_normal((n, F))
    X[:, 5] = np.nan                               # one subcarrier never decodes
    X[T + 2, 7] = np.nan                           # one dropout in block 1
    centres, psi = classifier.acf_windows(X, times, T)
    assert centres.size == 6
    assert np.isnan(psi[3]) and np.isfinite(psi[[0, 1, 2, 4, 5]]).all()


def test_too_few_frames_or_too_short_a_window_is_refused() -> None:
    times = np.arange(10) / 42.0
    centres, psi = classifier.acf_windows(np.zeros((10, 4)), times, T)
    assert centres.size == 0 and psi.size == 0
    with pytest.raises(ValueError):
        classifier.acf_windows(np.zeros((10, 4)), times, 3)


def test_units_divide_the_range_equally_nearest_the_requested_length() -> None:
    assert classifier.unit_edges(0.0, 59.6, 60).tolist() == [0.0, 59.6]
    assert classifier.unit_edges(0.0, 300.0, 60).tolist() == [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    e = classifier.unit_edges(10.0, 105.0, 60)
    assert e.size == 3 and e[0] == 10.0 and e[-1] == 105.0
    with pytest.raises(ValueError):
        classifier.unit_edges(5.0, 5.0, 60)


# --------------------------------------------------------------------------- #
#  Over a capture                                                             #
# --------------------------------------------------------------------------- #


def test_every_feature_in_the_bank_is_a_field_of_every_unit(tmp_path: Path) -> None:
    p = _capture(tmp_path)
    out = classifier.compute_features(p, 0.0, 200.0)
    assert len(out["units"]) == 3                 # 200 s nearest 60 s units
    for u in out["units"]:
        for f in classifier.FEATURES:
            assert f["key"] in u
    assert out["acf"]["window_frames"] == classifier.ACF_WINDOW_FRAMES
    assert out["acf"]["null_mean"] == pytest.approx(-1.0 / classifier.ACF_WINDOW_FRAMES)
    assert out["step"]["lag_seconds"] == hybrid2.RANGE_LAG_SECONDS


def _slow_capture(tmp_path: Path, n: int = 4000) -> Path:
    """200 s at 20 Hz: still, then a 0.3 Hz swing shared by every subcarrier
    from 80 s to 140 s -- a breathing sitter rather than hybrid2's fixture,
    whose "motion" is a fresh random factor every frame. White frame-to-frame
    modulation is noise to the autocorrelation by design; slow coherent
    modulation is what it measures."""
    tpi0 = band(seed=1)
    tpi1 = band(seed=2)
    rng = np.random.default_rng(11)
    blob = []
    for g in range(n):
        t = 0.05 * g
        mod = 1.0 + 0.005 * rng.standard_normal()
        if 80.0 <= t < 140.0:
            mod *= 1.0 + 0.3 * np.sin(2 * np.pi * 0.3 * t)
        blob.append(group_records(
            g, {(0, 0): tpi0 * mod, (1, 0): tpi1 * (2.0 - mod)}, ts=1000 + 50 * g,
        ))
    return write(tmp_path, b"".join(blob), "slow.bin")


def test_the_moving_unit_scores_above_the_still_ones(tmp_path: Path) -> None:
    """Both fixtures move from 80 s to 140 s; with three 66.7 s units that is
    the middle one. The step sees hybrid2's white per-frame modulation, the
    autocorrelation sees the slow swing -- each channel on the motion it is
    built for."""
    units = classifier.compute_features(_capture(tmp_path), 0.0, 200.0)["units"]
    assert units[1]["step_p90"] > max(units[0]["step_p90"], units[2]["step_p90"])

    slow = classifier.compute_features(_slow_capture(tmp_path), 0.0, 200.0)["units"]
    assert slow[1]["acf_amp_p90"] > max(slow[0]["acf_amp_p90"], slow[2]["acf_amp_p90"])
    assert slow[1]["acf_amp_median"] > 0.5


def test_white_per_frame_modulation_is_noise_to_the_autocorrelation(tmp_path: Path) -> None:
    """The flip side, and the reason ψ̂ is a feature beside the step rather
    than in place of it: a fresh random factor every frame moves the step a
    lot and ψ̂ not upward at all."""
    units = classifier.compute_features(_capture(tmp_path), 0.0, 200.0)["units"]
    assert units[1]["acf_amp_median"] <= max(units[0]["acf_amp_median"], units[2]["acf_amp_median"]) + 0.05


def test_the_link_jitter_is_the_one_frame_step_and_the_norm_is_the_ratio(tmp_path: Path) -> None:
    """Test 2's columns: lag1_p20 is the 20th percentile of the adjacent-frame
    step, step_norm the lag step over it. On the white-modulation fixture the
    moving unit's one-frame step is inflated too, so its norm is NOT larger
    than the still units' -- which is the caveat the bank records."""
    p = _capture(tmp_path)
    units = classifier.compute_features(p, 0.0, 200.0)["units"]
    for u in units:
        assert np.isfinite(u["lag1_p20"]) and u["lag1_p20"] > 0
        assert u["step_norm"] == pytest.approx(u["step_p90"] / u["lag1_p20"])
    assert units[1]["lag1_p20"] > max(units[0]["lag1_p20"], units[2]["lag1_p20"])


def test_the_breath_run_is_the_range_rules_run(tmp_path: Path) -> None:
    """Test 3's columns ride on hybrid2's own FarSense sweep: over a whole-range
    unit the 10 s run is range_verdict's breath_run to the second, and the
    20 s run with the rate floor is an integer that never exceeds the seconds."""
    p = _capture(tmp_path)
    out = classifier.compute_features(p, 0.0, 200.0, unit_seconds=200.0)
    u = out["units"][0]
    r = hybrid2.compute_hybrid2(p, 0.0, 200.0, lag_seconds=hybrid2.RANGE_LAG_SECONDS)
    assert u["breath_run"] == r["range_verdict"]["breath_run"]
    assert isinstance(u["breath_run20"], int) and 0 <= u["breath_run20"] <= u["n_seconds"]
    assert u["breath_rpm"] != u["breath_rpm"] or u["breath_rpm"] >= classifier.BREATH_RATE_FLOOR_RPM


def test_the_learned_score_and_the_rule_follow_the_slow_swing(tmp_path: Path) -> None:
    """Test 4's columns. The slow fixture's middle unit breathes at 18 rpm, so
    both the logistic score and rule B should call it and not the still ones;
    the score is a probability and the rule a 0/1."""
    units = classifier.compute_features(_slow_capture(tmp_path), 0.0, 200.0)["units"]
    for u in units:
        assert u["p_occupied"] != u["p_occupied"] or 0.0 <= u["p_occupied"] <= 1.0
        assert u["rule_b"] in (0, 1)
    assert units[1]["p_occupied"] > max(units[0]["p_occupied"], units[2]["p_occupied"])
    assert units[1]["rule_b"] == 1
    # the rule's floor is the smallest jitter seen so far, so it can only fall along the range
    floors = [u["rule_b_floor"] for u in units]
    assert floors == sorted(floors, reverse=True)


def test_coherence_windows_read_a_steady_pattern_as_no_drift() -> None:
    """Test 5's columns. One fixed pattern across subcarriers, scaled by a
    slow swing, is one-dimensional (λ₁ share near 1) and keeps its eigenvector
    (drift near 0); independent noise per subcarrier is neither."""
    rng = np.random.default_rng(2)
    n, F = 84 * 10, 32
    times = np.arange(n) / 42.0
    pattern = rng.standard_normal(F) + 1j * rng.standard_normal(F)
    swing = np.sin(2 * np.pi * 0.3 * times)[:, None]
    steady = 1.0 + 0.2 * swing * pattern[None, :] + 0.01 * (rng.standard_normal((n, F)) + 1j * rng.standard_normal((n, F)))
    lam, drift = classifier.coherence_windows(steady, times, 84)
    assert np.nanmedian(lam) > 0.9 and np.nanmedian(drift) < 0.05
    noise = 1.0 + 0.1 * (rng.standard_normal((n, F)) + 1j * rng.standard_normal((n, F)))
    lam_n, drift_n = classifier.coherence_windows(noise, times, 84)
    assert np.nanmedian(lam_n) < 0.3 and np.nanmedian(drift_n) > 0.5
    assert np.isnan(drift[0]) and np.isfinite(drift[1:]).all()


def test_lr_probability_is_nan_on_a_missing_input() -> None:
    u = {k: 0.0 for k in classifier.LR_FEATURES}
    assert 0.0 <= classifier.lr_probability(u) <= 1.0
    u["breath_peak10"] = float("nan")
    assert classifier.lr_probability(u) != classifier.lr_probability(u)   # NaN


def test_units_span_the_frames_not_the_request(tmp_path: Path) -> None:
    """A range asked wider than the capture still yields units over the
    capture's own extent -- three for 200 s -- rather than units over nothing."""
    p = _capture(tmp_path)
    out = classifier.compute_features(p, 0.0, 1000.0)
    assert len(out["units"]) == 3
    assert out["units"][-1]["t1"] == pytest.approx(199.95, abs=0.1)
    assert all(u["n_windows"] > 0 for u in out["units"])


def test_the_step_is_the_range_rules_step(tmp_path: Path) -> None:
    """Same frames, same lag: a unit's step P90 here is what range_verdict
    would compute over the same seconds, so the tab and the rule agree."""
    p = _capture(tmp_path)
    out = classifier.compute_features(p, 0.0, 200.0, unit_seconds=200.0)
    r = hybrid2.compute_hybrid2(p, 0.0, 200.0, lag_seconds=hybrid2.RANGE_LAG_SECONDS)
    assert out["units"][0]["step_p90"] == pytest.approx(r["range_verdict"]["motion_p90"], rel=0.05)


# --------------------------------------------------------------------------- #
#  The endpoint                                                               #
# --------------------------------------------------------------------------- #


def test_endpoint_returns_units_with_camera_occupancy_and_the_bank(tmp_path: Path) -> None:
    p = _capture(tmp_path)
    res = TestClient(app).get("/api/classifier", params={"path": str(p), "t0": 0.0, "t1": 200.0})
    assert res.status_code == 200, res.text
    body = res.json()
    assert [f["key"] for f in body["features"]] == [f["key"] for f in classifier.FEATURES]
    assert len(body["units"]) == 3
    occ = [u["camera_occupancy"] for u in body["units"]]
    assert occ[1] > 0.5 and occ[0] < 0.5 and occ[2] < 0.5
    assert len(body["acf"]["time_s"]) == len(body["acf"]["amp"]) == len(body["acf"]["phase"])
    assert body["truth"] is not None and body["truth_excluded"] is None
    # The fixture's last frame is at 199.95 s; units span the frames, so a
    # third of that, not of the 200 s asked for.
    assert body["unit_seconds"] == pytest.approx(199.95 / 3, abs=0.05)


def test_endpoint_honours_the_unit_length_and_refuses_an_empty_range(tmp_path: Path) -> None:
    p = _capture(tmp_path)
    client = TestClient(app)
    body = client.get("/api/classifier", params={"path": str(p), "t0": 0.0, "t1": 200.0, "unit_s": 100}).json()
    assert len(body["units"]) == 2
    res = client.get("/api/classifier", params={"path": str(p), "t0": 150.0, "t1": 150.01})
    assert res.status_code == 400
    assert "frames" in res.json()["detail"]
