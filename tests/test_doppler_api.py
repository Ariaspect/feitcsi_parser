"""Tests for the /api/doppler endpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app

CAPTURES = Path(__file__).resolve().parent.parent / "captures"


def _capture_or_skip(name: str = "capture.dat") -> Path:
    p = CAPTURES / name
    if not p.is_file():
        pytest.skip(f"{name} not present")
    return p


def test_doppler_grid_matches_its_headers() -> None:
    from backend.tiles import get_index

    p = _capture_or_skip()
    idx = get_index(p)
    r = TestClient(app).get("/api/doppler", params={
        "path": str(p), "t0": float(idx.times[0]), "t1": float(idx.times[-1]),
        "metric": "amplitude", "win_seconds": 10.0,
    })
    assert r.status_code == 200
    h = r.headers
    grid = np.frombuffer(r.content, dtype="<f4")
    assert grid.size == int(h["X-Doppler-Width"]) * int(h["X-Doppler-Height"])
    assert int(h["X-Doppler-Height"]) > int(h["X-Doppler-Win"]) // 2   # zero-padded
    assert float(h["X-Doppler-FMax"]) == pytest.approx(float(h["X-Doppler-Fs"]) / 2)
    assert float(h["X-Doppler-ColT0"]) <= float(h["X-Doppler-ColT1"])


@pytest.mark.parametrize("params,detail", [
    ({"metric": "csi_cir"}, "metric"),
    ({"metric": "amplitude", "win_seconds": 30.0, "t1_offset": 1.0}, "too few"),
])
def test_doppler_rejects_bad_parameters(params: dict, detail: str) -> None:
    from backend.tiles import get_index

    p = _capture_or_skip()
    idx = get_index(p)
    t0 = float(idx.times[0])
    offset = params.pop("t1_offset", None)
    t1 = t0 + offset if offset else float(idx.times[-1])
    r = TestClient(app).get("/api/doppler", params={"path": str(p), "t0": t0, "t1": t1, **params})
    assert r.status_code == 400
    assert detail in r.json()["detail"]


def test_doppler_clamps_rather_than_refusing_a_long_window() -> None:
    """Zooming past the window length returns a spectrogram, not a 400."""
    from backend.tiles import get_index

    p = _capture_or_skip()
    t0 = float(get_index(p).times[0])
    r = TestClient(app).get("/api/doppler", params={
        "path": str(p), "t0": t0, "t1": t0 + 20.0,
        "metric": "amplitude", "win_seconds": 600.0,
    })
    assert r.status_code == 200
    assert float(r.headers["X-Doppler-WinSeconds"]) < 600.0
    assert "X-Doppler-Blank" in r.headers


def test_doppler_refuses_a_path_outside_the_capture_roots() -> None:
    r = TestClient(app).get("/api/doppler", params={
        "path": "/etc/hostname", "t0": 0, "t1": 1e9, "metric": "amplitude",
    })
    assert r.status_code == 404


def test_doppler_headers_are_exposed_to_browsers() -> None:
    """Without expose_headers a browser hides these and the body is unusable."""
    exposed: set[str] | None = None
    for mw in app.user_middleware:
        if "CORS" in str(mw.cls):
            exposed = set(mw.kwargs["expose_headers"])
    assert exposed is not None
    for name in ("X-Doppler-Width", "X-Doppler-Height", "X-Doppler-Fs",
                 "X-Doppler-FMax", "X-Doppler-Win", "X-Doppler-Hop",
                 "X-Doppler-WinSeconds", "X-Doppler-Frames",
                 "X-Doppler-ColT0", "X-Doppler-ColT1"):
        assert name in exposed, f"{name} missing from expose_headers"
