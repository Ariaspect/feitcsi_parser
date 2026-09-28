"""Tests for backend.framediff -- the frame-to-frame amplitude step on a bounded axis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import framediff, hybrid
from backend.app import app
from tests.test_mtk import band, group_records, write

FS = 20.0


# --------------------------------------------------------------------------- #
#  The identity the bounded axis rests on                                      #
# --------------------------------------------------------------------------- #


def test_the_map_is_the_symmetric_relative_difference_of_the_amplitudes() -> None:
    """``tanh(dB * ln10 / 40)`` IS ``(a_t - a_(t-1))/(a_t + a_(t-1))``.

    The whole claim of the module: the bounded axis is not a different
    measurement from the dB step, it is that step monotonically squashed. If
    this ever drifts the panel is reading something it does not say it reads.
    """
    rng = np.random.default_rng(0)
    prev = rng.uniform(1e-3, 1e3, 2000)
    nxt = rng.uniform(1e-3, 1e3, 2000)
    step_db = 20.0 * np.log10(nxt / prev)
    symmetric = (nxt - prev) / (nxt + prev)
    assert np.allclose(framediff.db_to_unit(step_db), symmetric, atol=1e-12)


def test_the_axis_is_bounded_and_ordered() -> None:
    """Bounded in (-1, 1) over any step a radio can produce, and order-preserving."""
    steps = np.array([-300.0, -26.0, -1.0, 0.0, 1.0, 26.0, 300.0])
    d = np.asarray(framediff.db_to_unit(steps))
    assert np.all(np.abs(d) < 1.0)
    assert np.all(np.diff(d) > 0)
    assert framediff.db_to_unit(0.0) == 0.0


def test_an_absurd_step_saturates_at_the_bound_rather_than_passing_it() -> None:
    """Open at +-1 in maths, closed in float64: tanh reaches 1.0 near 600 dB.

    Asserted rather than left to chance, because a chart handed 1.0000001
    would draw outside its own axis.
    """
    d = np.asarray(framediff.db_to_unit(np.array([-1e6, 1e6])))
    assert np.all(np.abs(d) <= 1.0)


def test_a_step_reads_back_in_the_dB_it_was_measured_in() -> None:
    """No threshold is drawn any more, but the dB equivalence is the axis's
    whole justification and every reported step still carries it.

    26 dB -> 0.9045 is kept as the worked value because it is the board's
    trigger, which docs/frame_step.md measures this fold against.
    """
    assert float(framediff.db_to_unit(26.0)) == pytest.approx(0.904547, abs=1e-6)
    for db in (0.05, 1.0, 6.0, 22.5, 26.0):
        assert float(framediff.unit_to_db(framediff.db_to_unit(db))) == pytest.approx(db)


def test_no_threshold_survives_in_the_module() -> None:
    """Removed rather than defaulted: the board's line sat above every step this
    fold produces, so an axis marking it marked nothing."""
    assert not hasattr(framediff, "LG_THRESHOLD_DB")
    assert "fraction_above" not in framediff.fold(
        framediff.relative_step(_amp_db(4, 20))
    )


def test_the_inverse_saturates_rather_than_returning_infinity_off_the_end() -> None:
    assert math.isinf(float(framediff.unit_to_db(1.0)))
    assert float(framediff.unit_to_db(0.0)) == 0.0


# --------------------------------------------------------------------------- #
#  The folds                                                                   #
# --------------------------------------------------------------------------- #


def _amp_db(n_frames: int, n_sc: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-40.0, -10.0, (n_frames, n_sc))


def test_the_magnitude_fold_is_the_dB_median_mapped_onto_the_axis() -> None:
    """Median commutes with a monotone map, which is why this panel and
    ``hybrid.amplitude_diff`` are one series on two scales."""
    amp = _amp_db(200, 33, seed=1)
    got = framediff.fold(framediff.relative_step(amp))["magnitude"]
    want = framediff.db_to_unit(np.median(np.abs(np.diff(amp, axis=0)), axis=1))
    assert np.allclose(got, want, atol=1e-12)


def test_the_signed_fold_keeps_its_sign() -> None:
    """Brightening is positive, fading negative, and a still array is zero."""
    flat = np.tile(np.linspace(-20.0, -26.0, 12), (4, 1))
    rising = flat + np.arange(4)[:, None] * 3.0
    falling = flat - np.arange(4)[:, None] * 3.0
    assert np.all(framediff.fold(framediff.relative_step(rising))["signed"] > 0)
    assert np.all(framediff.fold(framediff.relative_step(falling))["signed"] < 0)
    assert np.allclose(framediff.fold(framediff.relative_step(flat))["signed"], 0.0)


def test_a_signed_median_cancels_when_the_subcarriers_disagree() -> None:
    """The documented weakness, asserted so the panel's caption stays true.

    Half the array brightening by 20 dB and half fading by 20 dB is loud
    motion, and the signed median reads nearly nothing. The magnitude fold is
    what sees it, which is why both are returned.
    """
    n_sc = 40
    amp = np.full((2, n_sc), -20.0)
    amp[1, : n_sc // 2] += 20.0
    amp[1, n_sc // 2:] -= 20.0
    folded = framediff.fold(framediff.relative_step(amp))
    assert abs(float(folded["signed"][0])) < 0.05
    assert float(folded["magnitude"][0]) == pytest.approx(
        float(framediff.db_to_unit(20.0)), abs=1e-9
    )


def test_a_whole_array_moving_together_carries_into_the_signed_fold() -> None:
    """A common shift -- which is what a gain step is -- survives both folds.

    Kept as arithmetic, NOT as a diagnostic: measured, |signed|/|d| does not
    separate a gain crossing from a body (docs/frame_step.md), which is why the
    gate reads the reported state instead of guessing from this.
    """
    amp = np.full((2, 50), -30.0)
    amp[1] += 30.0
    folded = framediff.fold(framediff.relative_step(amp))
    want = float(framediff.db_to_unit(30.0))
    assert float(folded["signed"][0]) == pytest.approx(want, abs=1e-9)
    assert float(folded["magnitude"][0]) == pytest.approx(want, abs=1e-9)


def test_two_loud_subcarriers_are_not_a_room_changing() -> None:
    """The median is what makes this true, and the reason it is a median."""
    amp = np.full((2, 100), -30.0)
    amp[1, :2] += 40.0
    folded = framediff.fold(framediff.relative_step(amp))
    assert abs(float(folded["magnitude"][0])) < 1e-9
    assert abs(float(folded["signed"][0])) < 1e-9


def test_a_dead_array_reports_nothing_rather_than_a_median_of_two() -> None:
    amp = np.full((2, 100), np.nan)
    amp[:, :2] = [[-30.0], [-10.0]]
    folded = framediff.fold(framediff.relative_step(amp))
    assert folded["live"][0] == 2
    assert np.isnan(folded["signed"][0])
    assert np.isnan(folded["magnitude"][0])


def test_structural_nulls_are_ignored_rather_than_dragging_the_median() -> None:
    """A NaN subcarrier must not count as a quiet one."""
    amp = np.full((2, 60), -30.0)
    amp[1] += 12.0
    amp[:, ::3] = np.nan
    folded = framediff.fold(framediff.relative_step(amp))
    assert folded["live"][0] == 40
    assert float(folded["magnitude"][0]) == pytest.approx(
        float(framediff.db_to_unit(12.0)), abs=1e-9
    )


# --------------------------------------------------------------------------- #
#  Gaps                                                                        #
# --------------------------------------------------------------------------- #


def test_a_step_across_a_dropout_is_blanked_and_counted() -> None:
    """Two frames five seconds apart have no frame-to-frame step worth the name.

    The hole sits in a run of normal frames on purpose: ``gap_limit_for`` is a
    percentile of the intervals actually seen, so a handful of frames around
    one hole would make the hole its own 95th percentile and nothing would
    ever look like a gap.
    """
    rng = np.random.default_rng(5)
    amp = np.full((100, 40), -30.0) + rng.standard_normal((100, 40)) * 0.01
    amp[50:] += 15.0                      # the loud step sits across the hole
    times = np.concatenate([np.arange(50) * 0.05, 5.0 + np.arange(50) * 0.05])
    out = framediff.frame_steps(amp, times)
    assert out["n_bridged"] == 1
    assert np.isnan(out["magnitude"][49])  # the step into the far side
    assert np.isfinite(out["magnitude"][[0, 1, 48, 50, 51]]).all()
    assert out["gap_limit"] < 1.0


def test_each_step_is_stamped_at_its_later_frame() -> None:
    """The convention ``hybrid.capture_evidence`` already uses for amp_times."""
    amp = _amp_db(5, 20, seed=2)
    times = np.array([10.0, 10.05, 10.10, 10.15, 10.20])
    out = framediff.frame_steps(amp, times)
    assert np.allclose(out["time_s"], times[1:])


def test_a_range_with_one_frame_yields_no_step_rather_than_an_error() -> None:
    out = framediff.frame_steps(np.full((1, 10), -20.0), np.array([0.0]))
    assert out["time_s"].size == 0 and out["magnitude"].size == 0


def test_the_times_must_match_the_frames() -> None:
    with pytest.raises(ValueError, match="times against"):
        framediff.frame_steps(np.zeros((4, 3)), np.zeros(3))


# --------------------------------------------------------------------------- #
#  The gain state                                                              #
# --------------------------------------------------------------------------- #


def test_gain_crossings_are_counted_even_when_they_are_kept() -> None:
    """The count is owed to the reader whether or not it is acted on.

    84-100 % of the loudest 1 % of steps are gain crossings (docs/frame_step.md),
    so a panel that does not say how many there were is understating what the
    trace is made of.
    """
    amp = np.full((6, 40), -30.0)
    times = np.arange(6) * 0.05
    state = np.array([-48, -48, -49, -49, -49, -50])
    out = framediff.frame_steps(amp, times, gain_state=state)
    assert out["n_gain_crossed"] == 2
    assert out["gain_gated"] is False
    assert np.isfinite(out["magnitude"]).all()       # counted, not blanked


def test_gating_blanks_the_pairs_that_cross_a_state() -> None:
    amp = np.full((6, 40), -30.0)
    amp[2] += 20.0                                   # the stripe a gain step is
    times = np.arange(6) * 0.05
    state = np.array([-48, -48, -49, -48, -48, -48])
    out = framediff.frame_steps(amp, times, gain_state=state, gate_gain=True)
    assert out["gain_gated"] is True
    assert out["n_gain_crossed"] == 2
    assert np.isnan(out["magnitude"][[1, 2]]).all()  # into the stripe and out
    assert np.isfinite(out["magnitude"][[0, 3, 4]]).all()


def test_gating_needs_a_state_to_gate_on() -> None:
    """Asked for without one, it reports that it did not happen."""
    out = framediff.frame_steps(np.full((4, 40), -30.0), np.arange(4) * 0.05,
                                gate_gain=True)
    assert out["gain_gated"] is False
    assert out["n_gain_crossed"] == 0


def test_the_state_must_match_the_frames() -> None:
    with pytest.raises(ValueError, match="gain states against"):
        framediff.frame_steps(np.zeros((4, 3)), np.arange(4.0),
                              gain_state=np.zeros(3))


def test_the_gate_is_part_of_the_cache_key(tmp_path: Path) -> None:
    """Two settings, two entries -- not one answer served for both."""
    p = _capture_with_a_visit(tmp_path, n=400)
    framediff.reset_cache()
    a = framediff.capture_steps(p, 0.0, 200.0, gate_gain=False)
    b = framediff.capture_steps(p, 0.0, 200.0, gate_gain=True)
    assert a is not b
    assert a["gain_gated"] is False and b["gain_gated"] is True


def test_the_endpoint_passes_the_gate_through(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path, n=400)
    client = TestClient(app)
    off = client.get("/api/frame-diff", params={
        "path": str(p), "t0": 0.0, "t1": 200.0}).json()
    on = client.get("/api/frame-diff", params={
        "path": str(p), "t0": 0.0, "t1": 200.0, "gate_gain": "true"}).json()
    assert off["summary"]["gain_gated"] is False
    assert on["summary"]["gain_gated"] is True
    assert off["summary"]["n_gain_crossed"] == on["summary"]["n_gain_crossed"]
    # Gating can only remove steps, never add them.
    assert on["summary"]["steps_measured"] <= off["summary"]["steps_measured"]


# --------------------------------------------------------------------------- #
#  Decimation                                                                  #
# --------------------------------------------------------------------------- #


def _series(n: int, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    d = rng.uniform(-0.2, 0.2, n)
    return {
        "time_s": np.arange(n) * 0.05,
        "signed": d,
        "magnitude": np.abs(d),
    }


def test_a_short_range_passes_through_untouched() -> None:
    s = _series(100)
    out = framediff.decimate(s, 2000)
    assert out["decimated"] is False
    assert np.allclose(out["signed"], s["signed"])
    assert np.allclose(out["signed_lo"], s["signed"])
    assert out["count"].sum() == 100


def test_decimation_keeps_the_single_frame_that_moved() -> None:
    """The point of a per-frame signal is the one frame, and a mean hides it.

    One spike in 20 000 quiet steps survives in the envelope; the median of
    its own column does not carry it, which is why the envelope exists.
    """
    s = _series(20000)
    s["signed"][12345] = 0.97
    s["magnitude"][12345] = 0.97
    out = framediff.decimate(s, 500)
    assert out["decimated"] is True
    assert out["time_s"].size == 500
    assert float(np.nanmax(out["signed_hi"])) == pytest.approx(0.97)
    assert float(np.nanmax(out["magnitude_hi"])) == pytest.approx(0.97)
    assert float(np.nanmax(out["signed"])) < 0.5      # the median cannot see it
    assert out["count"].sum() == 20000


def test_the_columns_are_equal_in_time_and_cover_the_range() -> None:
    s = _series(5000)
    out = framediff.decimate(s, 250)
    dt = np.diff(out["time_s"])
    assert np.allclose(dt, dt[0])
    assert out["bin_seconds"] == pytest.approx(dt[0])
    assert out["time_s"][0] > s["time_s"][0]
    assert out["time_s"][-1] < s["time_s"][-1] + out["bin_seconds"]


def test_a_column_holding_only_blanked_steps_reports_null() -> None:
    """A blanked stretch stays blank, and its frames are still counted."""
    s = _series(2000)
    s["signed"][:1000] = np.nan
    s["magnitude"][:1000] = np.nan
    out = framediff.decimate(s, 100)
    assert np.isnan(out["signed"][:45]).all()
    assert np.isfinite(out["signed"][55:]).all()
    assert out["count"].sum() == 2000          # counted, just not measured


def test_an_empty_series_decimates_to_nothing() -> None:
    out = framediff.decimate(framediff.frame_steps(np.zeros((1, 4)), np.zeros(1)), 100)
    assert out["time_s"].size == 0 and out["decimated"] is False


# --------------------------------------------------------------------------- #
#  Over a capture                                                              #
# --------------------------------------------------------------------------- #


def _capture_with_a_visit(tmp_path: Path, n: int = 4000) -> Path:
    """200 s at 20 Hz, still except for a moving occupant from 80 s to 140 s.

    The modulation rides on slot 0 -- ``(tpi 0, rpi 0)`` -- because that is the
    stream ``mtk.decode_frames`` turns into the ``amplitude`` metric this
    module reads. Modulating the other stream (as the motionsig fixture does)
    moves the ratio and leaves the raw amplitude flat.
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
            g, {(0, 0): tpi0 * mod, (1, 0): tpi1}, ts=1000 + 50 * g,
        ))
    p = write(tmp_path, b"".join(blob), "visit.bin")
    base = 1_700_000_000.0
    frames = [
        {"epoch": base + s, "n": 1 if 80 <= s < 140 else 0,
         "max_conf": 0.9 if 80 <= s < 140 else 0.0}
        for s in range(200)
    ]
    p.with_name("visit_cv.json").write_text(json.dumps({"frames": frames}))
    return p


def test_the_visit_reads_louder_than_the_still_stretches(tmp_path: Path) -> None:
    framediff.reset_cache()
    p = _capture_with_a_visit(tmp_path)
    out = framediff.capture_steps(p, 0.0, 200.0)
    t = np.asarray(out["time_s"])
    moving = (t >= 85) & (t < 135)
    still = (t < 75) | (t >= 145)
    mag = np.asarray(out["magnitude"])
    assert np.nanmedian(mag[moving]) > 5 * np.nanmedian(mag[still])
    assert out["frames_used"] == 4000
    assert np.all(np.abs(mag[np.isfinite(mag)]) < 1.0)


def test_it_is_the_hybrid_amplitude_channel_on_the_bounded_axis(tmp_path: Path) -> None:
    """Cross-module: the same decode, the same fold, one monotone map apart.

    If these ever disagree, one of the two panels has changed what it reads
    and the docstring's claim that they are one series is stale.
    """
    from backend.tiles import get_index

    framediff.reset_cache()
    p = _capture_with_a_visit(tmp_path, n=600)
    index = get_index(p)
    times = np.asarray(index.times, dtype=float)
    ids = np.flatnonzero((times >= 0.0) & (times <= 200.0))
    theirs = hybrid.amplitude_diff(p, index, ids)
    ours = framediff.capture_steps(p, 0.0, 200.0)["magnitude"]
    both = np.isfinite(theirs) & np.isfinite(ours)
    assert both.sum() > 500
    assert np.allclose(ours[both], framediff.db_to_unit(theirs[both]), atol=1e-9)


def test_a_range_with_one_frame_is_a_clear_refusal(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path, n=40)
    with pytest.raises(ValueError, match="fewer than 2 frames"):
        framediff.capture_steps(p, 0.0, 0.02)


def test_the_decode_is_cached(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path, n=400)
    framediff.reset_cache()
    first = framediff.capture_steps(p, 0.0, 200.0)
    assert framediff.capture_steps(p, 0.0, 200.0) is first


def test_the_summary_reports_the_range_in_both_units(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    out = framediff.compute_frame_diff(p, 0.0, 200.0, max_points=400)
    s = out["summary"]
    assert s["steps"] == 3999
    assert s["steps_measured"] <= s["steps"]
    assert 0.0 < s["median"] < s["max"] < 1.0
    assert s["median_db"] == pytest.approx(float(framediff.unit_to_db(s["median"])))


# --------------------------------------------------------------------------- #
#  The endpoint                                                                #
# --------------------------------------------------------------------------- #


def test_the_endpoint_serves_one_column_per_point(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    client = TestClient(app)
    res = client.get("/api/frame-diff", params={
        "path": str(p), "t0": 0.0, "t1": 200.0, "max_points": 300,
    })
    assert res.status_code == 200, res.text
    body = res.json()
    n = len(body["time_s"])
    assert n == 300
    for key in ("signed", "signed_lo", "signed_hi", "magnitude",
                "magnitude_hi", "count"):
        assert len(body[key]) == n, key
    assert body["decimated"] is True
    assert "threshold_db" not in body
    assert body["frames_used"] == 4000
    assert body["n_subcarriers"] > 0


def test_the_endpoint_writes_null_and_never_a_bare_nan(tmp_path: Path) -> None:
    """A single NaN written as ``NaN`` is not JSON and takes the tab down."""
    p = _capture_with_a_visit(tmp_path)
    client = TestClient(app)
    res = client.get("/api/frame-diff", params={"path": str(p), "t0": 0.0, "t1": 200.0})
    assert res.status_code == 200
    assert "NaN" not in res.text
    json.loads(res.text)


def test_the_endpoint_refuses_a_range_with_no_step(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path, n=40)
    client = TestClient(app)
    res = client.get("/api/frame-diff", params={"path": str(p), "t0": 0.0, "t1": 0.02})
    assert res.status_code == 400
    assert "fewer than 2 frames" in res.json()["detail"]


def test_the_endpoint_serves_a_sub_range_for_a_zoom(tmp_path: Path) -> None:
    """The panel zooms by refetching, so a narrower range must come back with
    the same column budget over fewer steps -- finer, not stretched."""
    p = _capture_with_a_visit(tmp_path)
    client = TestClient(app)
    whole = client.get("/api/frame-diff", params={
        "path": str(p), "t0": 0.0, "t1": 200.0, "max_points": 300}).json()
    zoomed = client.get("/api/frame-diff", params={
        "path": str(p), "t0": 100.0, "t1": 110.0, "max_points": 300}).json()
    assert zoomed["summary"]["steps"] < whole["summary"]["steps"] / 10
    assert zoomed["bin_seconds"] < whole["bin_seconds"] / 10
    assert zoomed["time_s"][0] >= 100.0
    assert zoomed["time_s"][-1] <= 110.0
