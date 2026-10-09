"""/api/captures: what it reads from the sidecars beside each capture.

Kept apart from test_filters.py, whose module-level skip needs
captures/capture.dat -- none of these do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import _walk_captures, app


@pytest.fixture
def day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """captures/<day>/ laid out as the lg recorder writes it."""
    import backend.app as app_mod
    root = tmp_path / "captures"
    d = root / "20260930"
    d.mkdir(parents=True)
    monkeypatch.setattr(app_mod, "CAPTURES_DIR", root)
    return d


def _capture(d: Path, stem: str, cv: object = None, meta: object = None) -> None:
    (d / f"{stem}.bin").write_bytes(b"\x01" * 16)
    if cv is not None:
        (d / f"{stem}_cv.json").write_text(cv if isinstance(cv, str) else json.dumps(cv))
    if meta is not None:
        (d / f"{stem}_meta.json").write_text(meta if isinstance(meta, str) else json.dumps(meta))


def _listing() -> dict[str, dict]:
    res = TestClient(app).get("/api/captures")
    assert res.status_code == 200
    return {Path(c["filename"]).stem: c for c in res.json()}


def test_the_camera_sorts_captures_that_record_no_scenario(day: Path) -> None:
    _capture(day, "a", cv={"summary": {"fraction_occupied": 0.0}})
    _capture(day, "b", cv={"summary": {"fraction_occupied": 0.3}})
    _capture(day, "c", cv={"summary": {"fraction_occupied": 0.9}})
    _capture(day, "d")
    got = _listing()
    assert [got[s]["scenario"] for s in "abc"] == ["empty", "partial", "occupied"]
    assert got["b"]["occupancy"] == pytest.approx(0.3)
    assert "scenario" not in got["d"] and "occupancy" not in got["d"]


def test_a_recorded_scenario_wins_over_the_camera(day: Path) -> None:
    _capture(day, "a", cv={"summary": {"fraction_occupied": 0.0}},
             meta={"scenario": "seated", "room": "lab_a", "subject": ""})
    got = _listing()["a"]
    assert got["scenario"] == "seated" and got["room"] == "lab_a"
    assert "subject" not in got                 # a recorded blank is not reported
    assert "occupancy" not in got


def test_unreadable_sidecars_are_ignored_not_fatal(day: Path) -> None:
    _capture(day, "a", cv="{not json", meta="[1, 2]")
    _capture(day, "b", cv={"summary": None})
    _capture(day, "c")
    (day / "c_cv.json").write_bytes(b"\xff\xfe\x00")        # not UTF-8 at all
    got = _listing()
    for stem in "abc":
        assert "scenario" not in got[stem]


def test_the_camera_s_own_files_beside_a_capture_are_not_listed(day: Path) -> None:
    """Each capture sits beside a frames tarball and a directory of annotated
    stills; those are walked past, and only the capture itself is listed."""
    _capture(day, "20260930_120000", cv={"summary": {"fraction_occupied": 0.0}})
    (day / "20260930_120000_frames.tar").write_bytes(b"\0")
    annotated = day / "20260930_120000_cv_annotated"
    annotated.mkdir()
    for i in range(5):
        (annotated / f"20260930_1200{i:02d}_500.jpg").write_bytes(b"\0")
    got = _listing()
    assert list(got) == ["20260930_120000"]
    assert got["20260930_120000"]["filename"] == "20260930/20260930_120000.bin"


def test_the_path_walk_used_by_phase_1_finds_the_same_captures(day: Path) -> None:
    _capture(day, "a")
    (day / "nested").mkdir()
    _capture(day / "nested", "b")
    root = day.parent
    walked = {p.relative_to(root).as_posix() for p in _walk_captures(root, 8, set())}
    assert walked == {c["filename"] for c in _listing().values()}
