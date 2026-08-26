"""Tests for the /api/presence endpoint."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.presence import STATE_EMPTY, STATE_MOVING, STATE_PRESENT, STATE_UNKNOWN
from backend.tiles import PRESENCE_METRICS, get_index

CAPTURES = Path(__file__).resolve().parent.parent / "captures"
STATES = {STATE_UNKNOWN, STATE_MOVING, STATE_PRESENT, STATE_EMPTY}
SERIES = (
    "score", "periodicity", "tonality", "motion_gate", "motion_level", "rate_rpm",
)


def _capture_or_skip(name: str = "capture.dat") -> Path:
    p = CAPTURES / name
    if not p.is_file():
        pytest.skip(f"{name} not present")
    return p


def _full_range(p: Path) -> tuple[float, float]:
    idx = get_index(p)
    return float(idx.times[0]), float(idx.times[-1])


def _presence(p: Path, t0: float, t1: float, **params: float) -> dict:
    r = TestClient(app).get(
        "/api/presence", params={"path": str(p), "t0": t0, "t1": t1, **params}
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_presence_series_are_aligned_and_on_the_capture_clock() -> None:
    p = _capture_or_skip()
    t0, t1 = _full_range(p)
    r = TestClient(app).get("/api/presence", params={"path": str(p), "t0": t0, "t1": t1})

    assert r.status_code == 200, r.text
    body = r.json()

    n = len(body["time_s"])
    assert n > 0
    for key in (*SERIES, "state", "unknown"):
        assert len(body[key]) == n, f"{key} is not aligned with time_s"

    assert set(body["state"]) <= STATES
    times = body["time_s"]
    assert times == sorted(times)
    # Window centres sit inside the requested range, half a window in from
    # each end -- the same convention /api/doppler's column centres use.
    assert t0 <= times[0] <= times[-1] <= t1
    assert body["fs_hz"] > 0
    assert body["frames_used"] > 0


def test_blanked_windows_serialise_as_null_not_nan() -> None:
    """A bare NaN is not JSON and JSON.parse rejects the whole response.

    One blanked window would otherwise take the entire panel down rather than
    leaving a break in one line.
    """
    p = _capture_or_skip()
    t0, t1 = _full_range(p)
    r = TestClient(app).get("/api/presence", params={"path": str(p), "t0": t0, "t1": t1})

    assert "NaN" not in r.text, "response carries a bare NaN"
    body = json.loads(r.text)                       # strict: rejects NaN/Infinity
    for key in SERIES:
        for v in body[key]:
            assert v is None or math.isfinite(v)


def test_an_unknown_window_carries_no_score_or_rate() -> None:
    """Missing data must never arrive as a number a chart would draw."""
    p = _capture_or_skip()
    t0, t1 = _full_range(p)
    r = TestClient(app).get("/api/presence", params={
        # Blank on any interpolation at all, to force unknown windows on a
        # capture that has only small gaps.
        "path": str(p), "t0": t0, "t1": t1, "max_gap_fraction": 0.01,
    })
    body = r.json()

    for state, unknown, score, rate in zip(
        body["state"], body["unknown"], body["score"], body["rate_rpm"]
    ):
        if unknown:
            assert state == STATE_UNKNOWN
            assert score is None and rate is None
        else:
            assert state != STATE_UNKNOWN


def test_presence_reads_the_swap_corrected_ratio() -> None:
    """Not cosmetic: uncorrected, 1.2% of frame steps exceed pi outright.

    A pi step is a broadband impulse with energy inside 0.1-0.6 Hz, so on the
    raw ratio the detector would find respiration in an empty room.
    """
    assert PRESENCE_METRICS == (
        "csi_ratio_amplitude_corrected",
        "csi_ratio_phase_corrected",
    )


def test_a_transmitter_filter_narrows_the_channel() -> None:
    """Two transmitters interleaved are two channels, not one."""
    p = _capture_or_skip()
    t0, t1 = _full_range(p)
    idx = get_index(p)
    macs = sorted(set(idx.source_macs))
    if len(macs) < 2:
        pytest.skip("capture has a single transmitter")

    client = TestClient(app)
    mixed = client.get("/api/presence", params={"path": str(p), "t0": t0, "t1": t1})
    single = client.get("/api/presence", params={
        "path": str(p), "t0": t0, "t1": t1, "source_mac": macs[0],
    })

    assert single.status_code == 200, single.text
    assert single.json()["frames_used"] < mixed.json()["frames_used"]


@pytest.mark.parametrize("params,detail", [
    ({"channel": "quadrature"}, "channel"),
    ({"window_seconds": 3.0}, "cannot resolve"),
    # The Nyquist guard is not reachable from here -- 120 rpm is 2 Hz and this
    # capture's Nyquist is 2.54 Hz -- so it is pinned in test_presence.py
    # instead. What is reachable is asking for a rate band inverted.
    ({"rpm_lo": 30.0, "rpm_hi": 9.0}, "rate band"),
])
def test_presence_rejects_parameters_it_cannot_honour(params: dict, detail: str) -> None:
    p = _capture_or_skip()
    t0, t1 = _full_range(p)
    r = TestClient(app).get(
        "/api/presence", params={"path": str(p), "t0": t0, "t1": t1, **params}
    )

    assert r.status_code == 400, r.text
    assert detail in r.json()["detail"]


def test_an_empty_range_is_refused_rather_than_returning_nothing() -> None:
    p = _capture_or_skip()
    t0, _ = _full_range(p)
    r = TestClient(app).get("/api/presence", params={
        "path": str(p), "t0": t0 - 100.0, "t1": t0 - 90.0,
    })

    assert r.status_code == 400, r.text
    assert "fewer than 2 frames" in r.json()["detail"]


def test_a_window_longer_than_the_range_is_clamped_not_refused() -> None:
    """Zooming past the window length is ordinary use of a linked time axis."""
    p = _capture_or_skip()
    t0, t1 = _full_range(p)
    r = TestClient(app).get("/api/presence", params={
        "path": str(p), "t0": t0, "t1": t0 + 30.0, "window_seconds": 600.0,
    })

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["window_seconds"] < 600.0
    assert len(body["time_s"]) >= 1


def test_the_reported_geometry_is_what_ran_not_what_was_asked() -> None:
    p = _capture_or_skip()
    t0, t1 = _full_range(p)
    r = TestClient(app).get("/api/presence", params={
        "path": str(p), "t0": t0, "t1": t1, "window_seconds": 12.0, "hop_seconds": 1.0,
    })
    body = r.json()

    assert body["win"] == pytest.approx(body["window_seconds"] * body["fs_hz"], abs=1)
    assert body["params"]["window_seconds"] == 12.0    # what was asked
    assert body["rpm_floor_eff"] >= body["params"]["rate_band_rpm"][0]
    expected = int(round(body["window_seconds"] * body["fs_hz"] / 2))
    assert 60.0 * body["fs_hz"] / expected == pytest.approx(body["rpm_floor_eff"], rel=0.1)


def test_motion_level_is_not_a_flat_zero_on_a_real_capture() -> None:
    """Regression: structural nulls once made every motion sample read 0.0000.

    Measured on captures/20260821_170002.bin, an all-zero motion trace is the
    most confident possible claim that the room is perfectly still, produced
    from 11 dead subcarriers out of 256.
    """
    p = _capture_or_skip()
    t0, t1 = _full_range(p)
    body = TestClient(app).get(
        "/api/presence", params={"path": str(p), "t0": t0, "t1": t1}
    ).json()

    levels = np.array([v for v in body["motion_level"] if v is not None])
    assert levels.size > 0
    assert levels.max() > 0.0, "a real capture cannot be perfectly static"
    assert levels.std() > 0.0, "and cannot hold one value for its whole length"


# --------------------------------------------------------------------------- #
#  Empty-room reference                                                        #
# --------------------------------------------------------------------------- #


def test_without_a_reference_no_window_claims_an_empty_room() -> None:
    """The whole point of the reference: absence stops being free."""
    capture = _capture_or_skip()
    body = _presence(capture, 0.0, 20.0)

    assert STATE_EMPTY not in body["state"]
    assert body["reference"] is None
    assert body["baseline_dev_threshold"] is None
    assert all(v is None for v in body["baseline_dev"])
    assert all(v is None for v in body["motion_ratio"])
    assert any("reference" in w for w in body["warnings"])


def test_a_reference_range_produces_a_threshold_and_a_deviation() -> None:
    capture = _capture_or_skip()
    body = _presence(capture, 0.0, 20.0, ref_t0=0.0, ref_t1=20.0)

    ref = body["reference"]
    assert ref["dev_p95"] > 0.0
    assert ref["motion_floor"] > 0.0
    assert ref["n_windows"] > 0
    assert body["baseline_dev_threshold"] == pytest.approx(3.0 * ref["dev_p95"])
    assert any(v is not None for v in body["baseline_dev"])


def test_a_range_measured_against_itself_is_not_occupied() -> None:
    """A room cannot be displaced from where it already is."""
    capture = _capture_or_skip()
    body = _presence(capture, 0.0, 20.0, ref_t0=0.0, ref_t1=20.0)

    devs = [v for v in body["baseline_dev"] if v is not None]
    assert max(devs) < body["baseline_dev_threshold"]
    assert STATE_PRESENT not in body["state"]


def test_half_a_reference_range_is_refused() -> None:
    capture = _capture_or_skip()
    client = TestClient(app)
    for params in ({"ref_t0": 0.0}, {"ref_t1": 20.0}):
        resp = client.get(
            "/api/presence",
            params={"path": str(capture), "t0": 0.0, "t1": 20.0, **params},
        )
        assert resp.status_code == 400, params
        assert "together" in resp.json()["detail"]


def test_a_backwards_reference_range_is_refused() -> None:
    capture = _capture_or_skip()
    resp = TestClient(app).get(
        "/api/presence",
        params={
            "path": str(capture), "t0": 0.0, "t1": 20.0,
            "ref_t0": 20.0, "ref_t1": 5.0,
        },
    )
    assert resp.status_code == 400
    assert "must exceed" in resp.json()["detail"]


def test_the_breathing_rate_is_only_reported_where_it_is_believed() -> None:
    capture = _capture_or_skip()
    body = _presence(capture, 0.0, 20.0)

    for believed, rate in zip(body["breathing"], body["rate_rpm"]):
        if not believed:
            assert rate is None
