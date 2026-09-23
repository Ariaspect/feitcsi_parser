"""Tests for backend.motionsig -- the two-feature amplitude motion signal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import motionsig
from backend.app import app
from tests.test_mtk import band, group_records, write

FS = 20.0
N_SC = 32


# --------------------------------------------------------------------------- #
#  The moment identities                                                       #
# --------------------------------------------------------------------------- #


def _naive(x: np.ndarray, starts: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Variance and lag-1 the obvious way, for the fast path to agree with."""
    var = np.empty((starts.size, x.shape[1]))
    lag = np.empty((starts.size, x.shape[1]))
    for i, s in enumerate(starts):
        w = x[s:s + n]
        d = w - w.mean(axis=0)
        var[i] = (d ** 2).mean(axis=0)
        lag[i] = (d[1:] * d[:-1]).sum(axis=0) / np.maximum((d ** 2).sum(axis=0), 1e-30)
    return var, lag


def test_the_cumulative_sums_reproduce_the_windowed_moments() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((500, 7)) + 4.0          # offset: the identities must not need a centred signal
    starts = np.arange(0, 500 - 40 + 1, 10)
    want_var, want_lag = _naive(x, starts, 40)
    got = motionsig.features(x, x, starts, 40)
    assert np.allclose(got["variance"], want_var, atol=1e-9)
    assert np.allclose(got["lag1"], want_lag, atol=1e-9)


def test_lag1_separates_a_random_walk_from_white_noise_at_the_same_variance() -> None:
    """The reason lag-1 is in the feature set: it is scale free.

    A random walk and white noise scaled to the same variance are the same
    number under `variance` and opposite ends of the range under lag-1.
    """
    rng = np.random.default_rng(11)
    n = 4000
    walk = np.cumsum(rng.standard_normal((n, 1)), axis=0)
    white = rng.standard_normal((n, 1))
    walk *= white.std() / walk.std()
    starts = np.arange(0, n - 200 + 1, 50)
    f_walk = motionsig.features(walk, walk, starts, 200)
    f_white = motionsig.features(white, white, starts, 200)
    assert np.median(f_walk["lag1"]) > 0.9
    assert abs(float(np.median(f_white["lag1"]))) < 0.1


def test_the_hampel_replaces_only_the_strays() -> None:
    x = np.zeros((101, 2))
    x[:, 0] = np.linspace(0.0, 1.0, 101)
    x[:, 1] = np.linspace(0.0, 1.0, 101)
    x[50, 1] = 90.0
    out = motionsig.hampel(x)
    assert np.allclose(out[:, 0], x[:, 0])          # a clean ramp is untouched
    assert out[50, 1] < 1.0                         # the spike is pulled to the local median
    assert np.allclose(np.delete(out[:, 1], 50), np.delete(x[:, 1], 50))


def test_hampel_chunking_does_not_change_the_answer() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((300, 70))
    x[100, ::7] += 40.0
    assert np.allclose(motionsig.hampel(x, chunk=70), motionsig.hampel(x, chunk=8))


# --------------------------------------------------------------------------- #
#  Normalisation                                                               #
# --------------------------------------------------------------------------- #


def test_label_reference_centres_on_the_empty_windows_not_the_whole_range() -> None:
    v = np.concatenate([np.full(40, 1.0), np.full(60, 9.0)])
    empty = np.zeros(100, dtype=bool)
    empty[:40] = True
    ref = motionsig.reference({"variance": v, "lag1": v}, "label", empty)
    assert ref["variance"][0] == pytest.approx(1.0)      # the occupied 60% does not move the centre


def test_label_reference_falls_back_when_the_range_has_too_few_empty_windows() -> None:
    v = np.arange(100.0)
    empty = np.zeros(100, dtype=bool)
    empty[:3] = True
    ref = motionsig.reference({"variance": v, "lag1": v}, "label", empty)
    assert ref["variance"][0] == pytest.approx(float(np.median(v)))


def test_the_label_free_percentiles_stay_below_an_occupant_taking_half_the_range() -> None:
    """Why the pair is 10/40 and not 20/80.

    A capture 40% occupied has its occupant inside its own 80th percentile;
    scaling by (p20, p80) divides the signal away. The shipped pair has to
    leave the occupied windows far above 1.
    """
    v = np.concatenate([np.full(60, 1.0), np.full(40, 20.0)])
    ref = motionsig.reference({"variance": v, "lag1": v}, "free")
    z = (v - ref["variance"][0]) / ref["variance"][1]
    assert z[:60].max() < 1.0
    assert z[60:].min() > 5.0

    lo, hi = np.percentile(v, [20, 80])
    assert (v[60:] - lo).min() / max(hi - lo, 1e-30) < z[60:].min()   # the wide pair is worse


def test_the_free_reference_needs_no_labels_at_all() -> None:
    v = np.arange(100.0)
    assert motionsig.reference({"variance": v, "lag1": v}, "free") == \
        motionsig.reference({"variance": v, "lag1": v}, "free", np.ones(100, dtype=bool))


def test_the_score_is_the_fixed_linear_combination() -> None:
    values = {"variance": np.array([2.0]), "lag1": np.array([3.0])}
    ref = {"variance": (0.0, 1.0), "lag1": (0.0, 1.0)}
    coef = motionsig.COEFFICIENTS["label"]
    _, s = motionsig.score(values, ref, coef)
    assert s[0] == pytest.approx(coef["intercept"] + 2.0 * coef["variance"] + 3.0 * coef["lag1"])


def test_a_feature_that_is_all_nan_yields_no_score_rather_than_a_number() -> None:
    values = {"variance": np.array([1.0, 2.0]), "lag1": np.array([np.nan, 1.0])}
    ref = motionsig.reference(values, "free")
    _, s = motionsig.score(values, ref, motionsig.COEFFICIENTS["free"])
    assert np.isnan(s[0]) and np.isfinite(s[1])


def test_the_shipped_weights_lean_on_lag1() -> None:
    """A regression guard on the result, not the code.

    Under both normalisations the fit put most of the weight on lag-1 --
    4.7x the variance weight label-based, 190x label-free, because the
    percentile scale leaves variance spanning orders of magnitude while
    lag-1 is already bounded. Variance survives as a tie-break. If a refit
    ever inverts that, the report's reading of the linear model is stale.
    """
    for mode in motionsig.MODES:
        c = motionsig.COEFFICIENTS[mode]
        assert c["lag1"] > 0 and c["variance"] > 0, mode
        assert c["lag1"] > 4 * c["variance"], mode
        assert c["intercept"] < 0, mode          # quiet is the default verdict
        assert c["threshold"] > 0, mode


# --------------------------------------------------------------------------- #
#  Over a capture                                                              #
# --------------------------------------------------------------------------- #


def _capture_with_a_visit(tmp_path: Path, n: int = 4000) -> Path:
    """200 s at 20 Hz: still, then a moving occupant from 80 s to 140 s.

    The camera saw them over the same span. Unlike the hybrid's fixture there
    is no breathing stretch -- this detector has no breathing channel, and a
    still occupant is a miss by construction rather than a bug.
    """
    tpi0 = band(seed=1)
    tpi1 = band(seed=2)
    rng = np.random.default_rng(7)
    blob = []
    phase = 0.0
    for g in range(n):
        t = 0.05 * g
        mod = np.ones(tpi0.size, dtype=complex)
        if 80.0 <= t < 140.0:
            phase += rng.uniform(-2.0, 2.0)
            mod *= np.exp(1j * phase) * (1.0 + rng.uniform(-0.6, 0.6))
        mod *= 1.0 + 0.01 * (rng.standard_normal(tpi0.size) + 1j * rng.standard_normal(tpi0.size))
        blob.append(group_records(g, {(0, 0): tpi0, (1, 0): tpi1 * mod}, ts=1000 + 50 * g))
    p = write(tmp_path, b"".join(blob), "visit.bin")
    base = 1_700_000_000.0
    frames = [
        {"epoch": base + s, "n": 1 if 80 <= s < 140 else 0, "max_conf": 0.9 if 80 <= s < 140 else 0.0}
        for s in range(200)
    ]
    p.with_name("visit_cv.json").write_text(json.dumps({"frames": frames}))
    return p


def test_the_windows_are_centred_and_cover_the_range(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    out = motionsig.capture_features(p, 0.0, 200.0)
    t = out["time_s"]
    assert out["fs"] == pytest.approx(FS, rel=0.05)
    assert t[0] == pytest.approx(motionsig.WINDOW_SECONDS / 2, abs=0.2)
    assert np.allclose(np.diff(t), motionsig.HOP_SECONDS, atol=1e-6)
    assert t[-1] < 200.0


def test_motion_raises_both_features_over_the_still_stretches(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    out = motionsig.capture_features(p, 0.0, 200.0)
    t = np.asarray(out["time_s"])
    moving = (t >= 85) & (t < 135)
    still = (t < 75) | (t >= 145)
    for f in motionsig.FEATURES:
        v = np.asarray(out["raw"][f])
        assert np.nanmedian(v[moving]) > np.nanmedian(v[still])


def test_the_detector_finds_the_visit_under_both_normalisations(tmp_path: Path) -> None:
    from backend.app import _camera_truth

    p = _capture_with_a_visit(tmp_path)
    out = motionsig.compute_motion_signal(p, 0.0, 200.0, camera=_camera_truth(p))
    assert out["in_corpus"] is False
    for mode in motionsig.MODES:
        c = out["modes"][mode]["confusion"]
        assert c["recall"] > 0.8, mode
        assert c["specificity"] > 0.8, mode


def test_both_normalisations_share_one_decode(tmp_path: Path) -> None:
    """The raw features are computed once; only the scale differs."""
    p = _capture_with_a_visit(tmp_path)
    motionsig.reset_cache()
    out = motionsig.compute_motion_signal(p, 0.0, 200.0)
    a = out["modes"]["label"]["z"]["lag1"]
    b = out["modes"]["free"]["z"]["lag1"]
    finite = np.isfinite(a) & np.isfinite(b)
    # Two affine views of one series: perfectly correlated, different scale.
    assert np.corrcoef(a[finite], b[finite])[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert not np.allclose(a[finite], b[finite])


def test_without_a_camera_the_label_mode_says_why(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    out = motionsig.compute_motion_signal(p, 0.0, 200.0, camera=None)
    assert "no camera sidecar" in out["modes"]["label"]["note"]
    assert out["modes"]["free"]["note"] is None
    assert "confusion" not in out["modes"]["free"]


def test_a_gap_free_capture_reports_no_interpolation(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    out = motionsig.capture_features(p, 0.0, 200.0, max_gap_fraction=1e-9)
    assert np.allclose(out["gap_fraction"], 0.0)
    assert np.isfinite(np.asarray(out["raw"]["variance"])).all()


def _capture_with_a_hole(tmp_path: Path, hole: tuple[float, float] = (90.0, 110.0)) -> Path:
    """The same room, with the frames over `hole` simply never sent."""
    tpi0 = band(seed=1)
    tpi1 = band(seed=2)
    rng = np.random.default_rng(7)
    blob = []
    kept = 0
    for g in range(4000):
        t = 0.05 * g
        if hole[0] <= t < hole[1]:
            continue
        mod = 1.0 + 0.01 * (rng.standard_normal(tpi0.size) + 1j * rng.standard_normal(tpi0.size))
        blob.append(group_records(kept, {(0, 0): tpi0, (1, 0): tpi1 * mod}, ts=1000 + 50 * g))
        kept += 1
    return write(tmp_path, b"".join(blob), "hole.bin")


def test_a_mostly_interpolated_window_reports_nothing(tmp_path: Path) -> None:
    """A dropout is not a calm room.

    Across a hole the resampler hands back its own smooth interpolation,
    whose variance and lag-1 are the interpolator's, not the channel's. A
    window mostly made of it has to say nothing rather than say "still".
    """
    p = _capture_with_a_hole(tmp_path)
    out = motionsig.capture_features(p, 0.0, 200.0)
    t = np.asarray(out["time_s"])
    v = np.asarray(out["raw"]["variance"])
    inside = (t > 95) & (t < 105)
    outside = (t < 85) | (t > 115)
    assert inside.any() and np.isnan(v[inside]).all()
    assert np.isfinite(v[outside]).all()


# --------------------------------------------------------------------------- #
#  The endpoint                                                                #
# --------------------------------------------------------------------------- #


def test_endpoint_returns_both_modes_scored(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    r = TestClient(app).get("/api/motion-signal", params={"path": str(p), "t0": 0.0, "t1": 200.0})
    assert r.status_code == 200, r.text
    body = r.json()
    n = len(body["time_s"])
    assert body["streams"] > 4 and body["in_corpus"] is False
    for mode in ("label", "free"):
        m = body["modes"][mode]
        for key in ("variance", "lag1", "score", "present"):
            assert len(m[key]) == n, (mode, key)
        assert m["confusion"]["margin_s"] == 5.0
        assert m["confusion"]["recall"] > 0.8
    assert body["truth"]["present"][100] is True and body["truth"]["present"][10] is False
    assert "NaN" not in r.text


def test_endpoint_rejects_a_window_longer_than_the_range(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    r = TestClient(app).get(
        "/api/motion-signal", params={"path": str(p), "t0": 0.0, "t1": 4.0, "window_s": 30},
    )
    assert r.status_code == 400
    assert "window" in r.json()["detail"]


def test_endpoint_without_a_sidecar_returns_no_confusion(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    p.with_name("visit_cv.json").unlink()
    body = TestClient(app).get(
        "/api/motion-signal", params={"path": str(p), "t0": 0.0, "t1": 200.0},
    ).json()
    assert body["truth"] is None
    assert body["modes"]["free"]["confusion"] is None
    assert any(body["modes"]["free"]["present"])


def test_a_sidecar_flagged_exclude_from_eval_yields_no_truth(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    cv = p.with_name("visit_cv.json")
    data = json.loads(cv.read_text())
    data["exclude_from_eval"] = {"reason": "people walking past outside the room", "set_by": "user"}
    cv.write_text(json.dumps(data))
    body = TestClient(app).get(
        "/api/motion-signal", params={"path": str(p), "t0": 0.0, "t1": 200.0},
    ).json()
    assert body["truth"] is None and body["modes"]["label"]["confusion"] is None
    assert body["truth_excluded"] == "people walking past outside the room"
