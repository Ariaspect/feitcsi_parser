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
                 "X-Doppler-FMin", "X-Doppler-FMax", "X-Doppler-Win",
                 "X-Doppler-Hop", "X-Doppler-WinSeconds", "X-Doppler-Frames",
                 "X-Doppler-ColT0", "X-Doppler-ColT1"):
        assert name in exposed, f"{name} missing from expose_headers"


def test_real_metrics_report_a_one_sided_axis() -> None:
    """A real signal has a conjugate-symmetric spectrum: the axis starts at 0."""
    from backend.tiles import get_index

    p = _capture_or_skip()
    idx = get_index(p)
    r = TestClient(app).get("/api/doppler", params={
        "path": str(p), "t0": float(idx.times[0]), "t1": float(idx.times[-1]),
        "metric": "amplitude", "win_seconds": 10.0,
    })
    assert r.status_code == 200
    assert float(r.headers["X-Doppler-FMin"]) == 0.0


def test_the_complex_ratio_axis_is_two_sided() -> None:
    """The point of the complex metric: negative Doppler is a separate row.

    A real metric folds approaching onto receding. This one must not, so the
    axis has to span both signs and the grid has to be about twice as tall for
    the same window -- one row per bin of a full transform rather than half.
    """
    from backend.tiles import get_index

    p = _capture_or_skip()
    idx = get_index(p)
    t0, t1 = float(idx.times[0]), float(idx.times[-1])
    common = {"path": str(p), "t0": t0, "t1": t1, "win_seconds": 10.0}

    cx = TestClient(app).get("/api/doppler", params={**common, "metric": "csi_ratio_complex"})
    real = TestClient(app).get("/api/doppler", params={**common, "metric": "amplitude"})
    assert cx.status_code == 200, cx.text
    assert real.status_code == 200

    fs = float(cx.headers["X-Doppler-Fs"])
    f_min = float(cx.headers["X-Doppler-FMin"])
    f_max = float(cx.headers["X-Doppler-FMax"])
    assert f_min == pytest.approx(-fs / 2, rel=1e-3)
    assert f_max == pytest.approx(fs / 2, rel=1e-2)

    rows_cx = int(cx.headers["X-Doppler-Height"])
    rows_real = int(real.headers["X-Doppler-Height"])
    assert rows_cx == pytest.approx(2 * rows_real, rel=0.02)

    grid = np.frombuffer(cx.content, dtype="<f4")
    assert grid.size == rows_cx * int(cx.headers["X-Doppler-Width"])
    assert np.isfinite(grid).any()
    # Served in normalised dB, so values are negative wherever the motion is
    # weaker than a unit-amplitude tone -- which is everywhere real. What must
    # hold is that nothing sits at the log floor's -120 dB, which would mean a
    # zero got through the normalisation rather than a measurement.
    finite = grid[np.isfinite(grid)]
    assert finite.size
    assert (finite > -120.0).all()
    assert finite.min() < 0.0


def test_a_positive_shift_lands_in_the_upper_half_of_the_grid(monkeypatch) -> None:
    """The served row order is the renderer's contract, and it inverts here.

    ``stft_complex`` returns ascending frequencies; the endpoint serves row 0 =
    *highest*, so the flip has to happen and has to happen once. Getting it
    wrong mirrors the panel and reports approach as recession -- which no test
    on real capture data can catch, because nothing in a capture says which
    way the occupant walked. So the decode is replaced with a known signal:
    ``exp(+j*2*pi*f*t)`` has all its energy at ``+f`` and none at ``-f``.
    """
    from backend import tiles

    p = _capture_or_skip()
    idx = tiles.get_index(p)
    t0, t1 = float(idx.times[0]), float(idx.times[-1])

    def fake_series(path, index, frame_ids, times, interpolate, *, mimo, source_mac):
        t = np.asarray(times, dtype=float)
        tone = np.exp(2j * np.pi * 0.3 * (t - t[0]))
        return np.repeat(tone[:, None], 4, axis=1), t, 0

    monkeypatch.setattr(tiles, "_complex_ratio_series", fake_series)

    r = TestClient(app).get("/api/doppler", params={
        "path": str(p), "t0": t0, "t1": t1,
        "metric": "csi_ratio_complex", "win_seconds": 5.0,
    })
    assert r.status_code == 200, r.text
    rows = int(r.headers["X-Doppler-Height"])
    cols = int(r.headers["X-Doppler-Width"])
    grid = np.frombuffer(r.content, dtype="<f4").reshape(rows, cols)

    f_min = float(r.headers["X-Doppler-FMin"])
    f_max = float(r.headers["X-Doppler-FMax"])
    peak_row = int(np.argmax(np.nansum(grid, axis=1)))
    # Row 0 is f_max and the last row is f_min, so the value falls with index.
    peak_hz = f_max - (f_max - f_min) * peak_row / (rows - 1)
    assert peak_hz == pytest.approx(0.3, abs=0.05)
    assert peak_row < rows // 2, "a +0.3 Hz tone was served below the zero row"


def test_the_complex_metric_needs_no_orientation_reference() -> None:
    """It reads the raw ratio, so it works without a transmitter selected.

    The time-unwrapped phase metric is built on the corrected ratio and needs
    one; this is the difference, and it is why the complex panel is the one
    that draws on an unfiltered capture.
    """
    from backend.tiles import get_index

    p = _capture_or_skip()
    idx = get_index(p)
    r = TestClient(app).get("/api/doppler", params={
        "path": str(p), "t0": float(idx.times[0]), "t1": float(idx.times[-1]),
        "metric": "csi_ratio_complex", "win_seconds": 10.0,
    })
    assert r.status_code == 200, r.text
    assert int(r.headers["X-Doppler-Frames"]) > 0


def test_doppler_values_do_not_move_with_the_window_length() -> None:
    """A fixed colour range is only meaningful if the numbers hold still.

    An unnormalised FFT bin scales with the window: a unit tone reads 213.5 at
    win=600 and 71.3 at win=200, exactly the ratio. Dividing by the taper's
    coherent gain leaves the tone's own amplitude, so the same motion reads the
    same dB whatever window it was measured in. The decode is replaced with a
    known tone because no capture says how much motion it contains.
    """
    from backend import tiles

    p = _capture_or_skip()
    idx = tiles.get_index(p)
    t0, t1 = float(idx.times[0]), float(idx.times[-1])

    def fake_series(path, index, frame_ids, times, interpolate, *, mimo, source_mac):
        t = np.asarray(times, dtype=float)
        tone = np.exp(2j * np.pi * 0.3 * (t - t[0]))
        return np.repeat(tone[:, None], 4, axis=1), t, 0

    import pytest as _pytest
    peaks = {}
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(tiles, "_complex_ratio_series", fake_series)
        for win_s in (4.0, 8.0):
            r = TestClient(app).get("/api/doppler", params={
                "path": str(p), "t0": t0, "t1": t1,
                "metric": "csi_ratio_complex", "win_seconds": win_s,
            })
            if r.status_code != 200:
                _pytest.skip("capture too short for both windows")
            h = r.headers
            g = np.frombuffer(r.content, dtype="<f4").reshape(
                int(h["X-Doppler-Height"]), int(h["X-Doppler-Width"]))
            peaks[win_s] = float(np.nanmax(g))
    # A unit tone normalises to 0 dB; doubling the window must not move it.
    assert abs(peaks[4.0] - peaks[8.0]) < 1.5, peaks
    assert all(abs(v) < 2.0 for v in peaks.values()), peaks


def test_doppler_serves_a_fixed_colour_range() -> None:
    from backend import tiles

    p = _capture_or_skip()
    idx = tiles.get_index(p)
    r = TestClient(app).get("/api/doppler", params={
        "path": str(p), "t0": float(idx.times[0]), "t1": float(idx.times[-1]),
        "metric": "csi_ratio_complex", "win_seconds": 10.0,
    })
    assert r.status_code == 200
    lo = float(r.headers["X-Doppler-ScaleMin"])
    hi = float(r.headers["X-Doppler-ScaleMax"])
    assert (lo, hi) == tiles.DOPPLER_SCALE_DB
    assert lo < hi


def test_the_static_profile_takes_the_ratio_gain_out_of_the_colour() -> None:
    """Two captures differing only in gain must draw the same colour.

    The complex ratio's absolute scale is set by how the two transmit chains
    happen to be gained -- 20.0 dB of spread across twelve September captures
    -- and the Doppler magnitude carried it straight through, at a correlation
    of +0.857. Scaling the whole series by a constant is exactly that
    difference, and after the static profile is divided out it must not move
    the panel at all.
    """
    from backend import tiles

    p = _capture_or_skip()
    idx = tiles.get_index(p)
    t0, t1 = float(idx.times[0]), float(idx.times[-1])
    rng = np.random.default_rng(0)
    base = None

    def series_scaled(scale: float):
        def fake(path, index, frame_ids, times, interpolate, *, mimo, source_mac):
            t = np.asarray(times, dtype=float)
            n = t.size
            # A static path plus a small moving one, so there is a real
            # modulation depth to preserve.
            static = np.exp(1j * rng.uniform(0, 2 * np.pi, 4))[None, :]
            moving = 0.05 * np.exp(2j * np.pi * 0.3 * (t - t[0]))[:, None]
            return scale * (static + moving) * np.ones((n, 4)), t, 0
        return fake

    import pytest as _pytest
    seen = []
    for scale in (1.0, 50.0):
        tiles.reset_tile_caches()
        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(tiles, "_complex_ratio_series", series_scaled(scale))
            r = TestClient(app).get("/api/doppler", params={
                "path": str(p), "t0": t0, "t1": t1,
                "metric": "csi_ratio_complex", "win_seconds": 10.0,
            })
            assert r.status_code == 200, r.text
            g = np.frombuffer(r.content, dtype="<f4")
            seen.append(float(np.nanmax(g)))
    tiles.reset_tile_caches()
    # 50x the gain is 34 dB. Without the static profile it would move the
    # panel by that much; with it, not at all.
    assert abs(seen[0] - seen[1]) < 1.0, seen
