"""Tests for backend.hybrid -- the calibration-free motion/breathing detector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import hybrid
from backend.app import app
from tests.test_mtk import band, group_records, write

FS = 19.61
N_SC = 32


def _room(
    seconds: float,
    *,
    chest: tuple[float, float] | None = None,
    walk: tuple[float, float] | None = None,
    noise: float = 0.03,
    rpm: float = 15.0,
    seed: int = 0,
) -> np.ndarray:
    """A static ratio of 1 with optional breathing and walking stretches.

    ``chest`` is (start, stop) seconds of a 3% breath along a random axis per
    subcarrier; ``walk`` is (start, stop) seconds of a random walk in phase
    and magnitude, far above the noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    sig = np.ones((t.size, N_SC), dtype=complex)
    if chest is not None:
        a, b = chest
        m = (t >= a) & (t < b)
        axis = np.exp(1j * rng.uniform(0, 2 * np.pi, N_SC))
        sig[m] += 0.03 * np.sin(2 * np.pi * rpm / 60.0 * t[m])[:, None] * axis[None, :]
    sig += noise * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))
    if walk is not None:
        a, b = walk
        m = (t >= a) & (t < b)
        k = int(m.sum())
        sig[m] *= np.exp(1j * np.cumsum(rng.uniform(-2.0, 2.0, (k, 1)), axis=0))
        sig[m] *= 1.0 + rng.uniform(-0.6, 0.6, (k, 1))
    return sig


def _between(out: dict, a: float, b: float) -> np.ndarray:
    return (out["time_s"] >= a) & (out["time_s"] < b)


# --------------------------------------------------------------------------- #
#  Pieces                                                                      #
# --------------------------------------------------------------------------- #


def test_per_second_is_a_median_so_one_impulse_does_not_make_a_second() -> None:
    times = np.arange(0, 3, 0.05)
    values = np.full(times.size, 0.02)
    values[10] = 5.0                              # one gain step in second 0
    level = hybrid.per_second(values, times, np.arange(3.0))
    assert level[0] == pytest.approx(0.02)
    assert np.isnan(hybrid.per_second(values, times, np.arange(3.0, 6.0))).all()


def test_a_burst_needs_the_minimum_run() -> None:
    level = np.array([0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5])
    assert hybrid.bursts(level, 0.3, 2).tolist() == [False, False, False, True, True, False, True, True, True]
    assert hybrid.bursts(level, 0.3, 1).sum() == 6


def test_breathing_evidence_needs_persistence_and_mostly_agreeing_rates() -> None:
    peak = np.full(30, 0.3)
    rpm = np.full(30, 15.0)
    # One straying window in thirty is what real breathing looks like; it
    # must not break the run (measured: it did, on 20260916_202702).
    rpm[10] = 25.0
    peak[20] = 0.10
    ev = hybrid.consistent_breathing(peak, rpm, min_peak=0.15, n_consistent=15, rate_tol=3.0)
    assert ev.all()
    # Four stray windows in a run of fifteen is more than a fifth: every run
    # holding all four is rejected, and the first run that drops one passes.
    rpm2 = np.full(30, 15.0)
    rpm2[[3, 6, 9, 12]] = 25.0
    ev2 = hybrid.consistent_breathing(np.full(30, 0.3), rpm2, min_peak=0.15, n_consistent=15, rate_tol=3.0)
    assert not ev2[:4].any() and ev2[4:].all()
    # Rates that wander steadily across the band never share a median.
    drift = np.linspace(10.0, 37.0, 30)
    assert not hybrid.consistent_breathing(np.full(30, 0.3), drift, min_peak=0.15, n_consistent=15, rate_tol=3.0).any()
    short = hybrid.consistent_breathing(peak[:14], rpm[:14], min_peak=0.15, n_consistent=15)
    assert not short.any()


# --------------------------------------------------------------------------- #
#  The detector                                                                #
# --------------------------------------------------------------------------- #


def test_an_empty_room_stays_empty() -> None:
    out = hybrid.hybrid_seconds(_room(200.0), FS)
    assert not out["present"].any()
    assert not out["burst"].any()
    assert not out["breathing"].any()


def test_a_still_breathing_person_is_found_without_any_reference() -> None:
    out = hybrid.hybrid_seconds(_room(200.0, chest=(0.0, 200.0)), FS)
    # The first window's centre is 15 s in and the run needs 15 more.
    assert out["present"][_between(out, 40.0, 180.0)].mean() > 0.95
    assert out["breathing"].any() and not out["burst"].any()


def test_motion_opens_presence_and_the_hold_releases_it() -> None:
    out = hybrid.hybrid_seconds(_room(200.0, walk=(60.0, 70.0)), FS, hold_seconds=20.0)
    assert out["burst"][_between(out, 61.0, 69.0)].all()
    assert out["present"][_between(out, 60.0, 89.0)].all()      # walk + 20 s hold
    assert not out["present"][_between(out, 0.0, 59.0)].any()
    assert not out["present"][_between(out, 95.0, 200.0)].any()


def test_a_pushed_chair_is_a_burst_followed_by_nothing() -> None:
    """Nobody stays: presence lasts the hold and no longer, unlike a baseline
    detector that the moved chair would reset for the rest of the capture."""
    out = hybrid.hybrid_seconds(_room(300.0, walk=(100.0, 103.0)), FS, hold_seconds=20.0)
    present = out["present"]
    assert present[_between(out, 100.0, 122.0)].all()
    assert not present[_between(out, 125.0, 300.0)].any()


def _first_true(out: dict, key: str) -> float:
    idx = np.flatnonzero(out[key])
    return float(out["time_s"][idx[0]]) if idx.size else float("nan")


def _last_true(out: dict, key: str) -> float:
    idx = np.flatnonzero(out[key])
    return float(out["time_s"][idx[-1]]) if idx.size else float("nan")


def test_breathing_holds_presence_before_it_as_well_as_after() -> None:
    """The person was in the chair while the first window filled: with the
    leading hold, presence opens ``hold_seconds`` earlier than the evidence."""
    sig = _room(200.0, chest=(80.0, 200.0))
    with_lead = hybrid.hybrid_seconds(sig, FS, hold_seconds=20.0, lead_hold=True)
    without = hybrid.hybrid_seconds(sig, FS, hold_seconds=20.0, lead_hold=False)
    onset = _first_true(without, "breathing")
    assert 55.0 < onset < 95.0
    assert _first_true(without, "present") == onset
    assert abs(_first_true(with_lead, "present") - (onset - 20.0)) <= 1.0
    assert not with_lead["present"][_between(with_lead, 0.0, onset - 22.0)].any()


def test_holds_that_meet_make_the_gap_between_present() -> None:
    """Two breathing stretches with a pause the two holds can span between
    them: the seconds between are present (bridged) rather than empty."""
    sig = _room(200.0, chest=(0.0, 60.0))
    sig += _room(200.0, chest=(110.0, 200.0), noise=0.0) - 1.0
    out = hybrid.hybrid_seconds(sig, FS, hold_seconds=20.0, lead_hold=True)
    plain = hybrid.hybrid_seconds(sig, FS, hold_seconds=20.0, lead_hold=False)
    breathing = np.asarray(plain["breathing"])
    ends = np.flatnonzero(breathing[:-1] & ~breathing[1:])
    starts = np.flatnonzero(~breathing[:-1] & breathing[1:]) + 1
    assert ends.size >= 1 and starts.size >= 1
    gap_a, gap_b = float(plain["time_s"][ends[0]]), float(plain["time_s"][starts[starts > ends[0]][0]])
    assert 5.0 < gap_b - gap_a <= 40.0                      # under two holds: they meet
    assert out["present"][_between(out, gap_a, gap_b)].all()
    assert "bridged" in out["state"]
    assert not plain["present"][_between(plain, gap_a, gap_b)].all()


def test_holds_that_do_not_meet_leave_the_gap_empty() -> None:
    sig = _room(300.0, chest=(0.0, 60.0))
    sig += _room(300.0, chest=(200.0, 300.0), noise=0.0) - 1.0
    out = hybrid.hybrid_seconds(sig, FS, hold_seconds=20.0, lead_hold=True)
    breathing = np.asarray(out["breathing"])
    ends = np.flatnonzero(breathing[:-1] & ~breathing[1:])
    starts = np.flatnonzero(~breathing[:-1] & breathing[1:]) + 1
    gap_a, gap_b = float(out["time_s"][ends[0]]), float(out["time_s"][starts[starts > ends[0]][0]])
    assert gap_b - gap_a > 50.0
    assert out["present"][_between(out, gap_a, gap_a + 20.0)].all()          # trailing hold
    assert not out["present"][_between(out, gap_a + 22.0, gap_b - 21.0)].any()
    assert out["present"][_between(out, gap_b - 19.0, gap_b)].all()          # leading hold only
    assert "bridged" not in out["state"]


def test_the_verdict_does_not_depend_on_the_links_noise_scale() -> None:
    """Calibration-free means the same verdicts at five times the noise: the
    floor moves with the link and the threshold with it."""
    quiet = hybrid.hybrid_seconds(_room(200.0, walk=(60.0, 70.0), noise=0.02), FS)
    loud = hybrid.hybrid_seconds(_room(200.0, walk=(60.0, 70.0), noise=0.10), FS)
    assert loud["ratio_threshold"] > 3 * quiet["ratio_threshold"]
    assert np.array_equal(quiet["burst"], loud["burst"])
    assert np.array_equal(quiet["present"], loud["present"])


def test_dropouts_carry_no_evidence_and_no_verdict() -> None:
    sig = _room(200.0, chest=(0.0, 200.0))
    fab = np.zeros(sig.shape[0], dtype=bool)
    fab[int(100 * FS) : int(130 * FS)] = True
    out = hybrid.hybrid_seconds(sig, FS, fabricated=fab)
    inside = _between(out, 101.0, 129.0)
    assert out["unknown"][inside].all()
    assert (np.array(out["state"])[inside] == hybrid.STATE_UNKNOWN).all()
    assert not out["present"][inside].any()


def test_a_range_too_short_for_breathing_still_runs_the_motion_channel() -> None:
    """FarSense clamps its window to the range, so only a range shorter than
    one period at the band's floor loses the breathing channel."""
    out = hybrid.hybrid_seconds(_room(6.0, walk=(1.0, 4.0)), FS)
    assert out["breath_note"] is not None and "cannot hold" in out["breath_note"]
    assert out["burst"].any() and out["present"].any()


# --------------------------------------------------------------------------- #
#  The endpoint                                                                #
# --------------------------------------------------------------------------- #


def _capture_with_a_visit(tmp_path: Path, n: int = 4000) -> Path:
    """A 200 s MTK capture at 20 Hz: empty, a 10 s walk at 60 s, then a still
    breathing occupant to the end -- with a camera sidecar that saw them
    from 70 s on."""
    tpi0 = band(seed=1)
    tpi1 = band(seed=2)
    rng = np.random.default_rng(7)
    axis = np.exp(1j * rng.uniform(0, 2 * np.pi, tpi0.size))
    blob = []
    phase = 0.0
    for g in range(n):
        t = 0.05 * g
        mod = np.ones(tpi0.size, dtype=complex)
        if 60.0 <= t < 70.0:
            phase += rng.uniform(-2.0, 2.0)
            mod *= np.exp(1j * phase) * (1.0 + rng.uniform(-0.6, 0.6))
        elif t >= 70.0:
            mod += 0.03 * np.sin(2 * np.pi * 15.0 / 60.0 * t) * axis
        mod *= 1.0 + 0.01 * (rng.standard_normal(tpi0.size) + 1j * rng.standard_normal(tpi0.size))
        blob.append(group_records(g, {(0, 0): tpi0, (1, 0): tpi1 * mod}, ts=1000 + 50 * g))
    p = write(tmp_path, b"".join(blob), "visit.bin")
    base = 1_700_000_000.0
    frames = [
        {"epoch": base + s, "n": 1 if s >= 70 else 0, "max_conf": 0.9 if s >= 70 else 0.0}
        for s in range(200)
    ]
    p.with_name("visit_cv.json").write_text(json.dumps({"frames": frames}))
    return p


def test_endpoint_scores_the_visit_against_the_camera(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    r = TestClient(app).get("/api/hybrid", params={"path": str(p), "t0": 0.0, "t1": 200.0})
    assert r.status_code == 200, r.text
    body = r.json()
    n = len(body["time_s"])
    for key in ("present", "state", "motion_ratio", "breath_peak", "burst", "breathing"):
        assert len(body[key]) == n
    c = body["confusion"]
    assert c is not None and c["margin_s"] == 5.0
    assert c["recall"] > 0.8
    assert c["specificity"] > 0.8
    assert c["excluded"] > 0
    assert body["truth"]["present"][100] is True and body["truth"]["present"][10] is False
    assert "NaN" not in r.text


def test_endpoint_without_a_sidecar_returns_no_confusion(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    p.with_name("visit_cv.json").unlink()
    body = TestClient(app).get("/api/hybrid", params={"path": str(p), "t0": 0.0, "t1": 200.0}).json()
    assert body["confusion"] is None and body["truth"] is None
    assert any(body["present"])


def test_an_external_floor_lets_an_occupied_throughout_range_fire() -> None:
    """A person fidgeting for the whole range IS the range's 20th percentile,
    so the own-range threshold never trips; the link's quiet level from
    outside the range does."""
    rng = np.random.default_rng(11)
    t = np.arange(int(120 * FS)) / FS
    sig = np.ones((t.size, N_SC), dtype=complex)
    sig += 0.15 * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))   # fidget-level all along
    ev = hybrid.evidence_series(sig, FS)
    own = hybrid.verdict(ev)
    link = hybrid.verdict(ev, motion_floor=0.02)
    assert not own["burst"].any()
    assert link["burst"].mean() > 0.9
    assert link["params"]["motion_floor"] == 0.02 and own["params"]["motion_floor"] is None


def test_pooled_floor_is_the_links_quiet_level_across_captures(tmp_path: Path) -> None:
    """A noisy occupied capture and a quiet empty one an hour apart: the
    pooled floor is the quiet one's, and the noisy one's own is the occupant."""
    quiet = _capture_with_a_visit(tmp_path, n=2000)
    quiet_named = quiet.with_name("20260101_120000.bin")
    quiet.rename(quiet_named)
    quiet.with_name("visit_cv.json").rename(quiet_named.with_name("20260101_120000_cv.json"))
    own = hybrid.motion_levels(quiet_named)
    assert own.size > 50 and np.isfinite(own).any()
    floor_one = hybrid.pooled_floor([quiet_named])
    floor_two = hybrid.pooled_floor([quiet_named, quiet_named])
    assert floor_one == pytest.approx(floor_two)
    assert floor_one == pytest.approx(hybrid.floor_level(own))


def test_endpoint_pools_the_neighbours_for_the_floor(tmp_path: Path) -> None:
    p = _capture_with_a_visit(tmp_path)
    stamped = p.with_name("20260102_150000.bin")
    p.rename(stamped)
    p.with_name("visit_cv.json").rename(stamped.with_name("20260102_150000_cv.json"))
    client = TestClient(app)
    recent = client.get("/api/hybrid", params={"path": str(stamped), "t0": 0.0, "t1": 200.0}).json()
    own = client.get("/api/hybrid", params={"path": str(stamped), "t0": 0.0, "t1": 200.0, "floor_scope": "own"}).json()
    assert recent["floor_scope"] == "recent" and stamped.name in recent["floor_captures"]
    assert own["floor_scope"] == "own" and own["floor_captures"] == []
    explicit = client.get("/api/hybrid", params={"path": str(stamped), "t0": 0.0, "t1": 200.0, "motion_floor": 0.02}).json()
    assert explicit["floor_scope"] == "explicit" and explicit["ratio_floor"] == 0.02
    bad = client.get("/api/hybrid", params={"path": str(stamped), "t0": 0.0, "t1": 200.0, "floor_scope": "day"})
    assert bad.status_code == 400
