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


def test_the_operators_presence_label_and_the_camera_fraction_are_listed(day: Path) -> None:
    _capture(day, "a", cv={"summary": {"fraction_occupied": 0.97}},
             meta={"presence": "empty", "presence_source": "operator_manual"})
    _capture(day, "b", meta={"presence": "occupied"})
    _capture(day, "c", meta={"presence": "maybe"})
    got = _listing()
    assert got["a"]["presence"] == "empty" and got["a"]["presence_source"] == "operator_manual"
    assert got["a"]["camera_occupancy"] == pytest.approx(0.97)      # both shown, never merged
    assert got["b"]["presence"] == "occupied" and "camera_occupancy" not in got["b"]
    assert "presence" not in got["c"]                                 # only the two values the bar can send


def test_conditions_of_one_capture_are_read_fresh(day: Path) -> None:
    _capture(day, "a", cv={"summary": {"fraction_occupied": 0.0}})
    client = TestClient(app)
    path = _listing()["a"]["path"]
    first = client.get("/api/capture_conditions", params={"path": path}).json()
    assert first["camera_occupancy"] == 0.0 and "presence" not in first
    (day / "a_meta.json").write_text(json.dumps({"presence": "occupied", "presence_source": "operator_manual"}))
    again = client.get("/api/capture_conditions", params={"path": path}).json()
    assert again["presence"] == "occupied" and again["camera_occupancy"] == 0.0   # new sidecar seen without a reload
    assert client.get("/api/capture_conditions", params={"path": str(day / "nope.bin")}).status_code == 404


def test_a_rewritten_sidecar_is_read_again_not_served_from_the_cache(day: Path) -> None:
    import os
    _capture(day, "a", meta={"scenario": "seated"})
    assert _listing()["a"]["scenario"] == "seated"
    meta = day / "a_meta.json"
    meta.write_text(json.dumps({"scenario": "standing"}))
    st = meta.stat()
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert _listing()["a"]["scenario"] == "standing"


def test_a_capture_s_own_folders_are_not_walked_but_other_folders_are(day: Path) -> None:
    _capture(day, "a")
    frames = day / "a_cv_annotated"
    frames.mkdir()
    (frames / "stray.bin").write_bytes(b"\x01" * 16)   # never a capture in there
    other = day / "batch2"
    other.mkdir()
    _capture(other, "b")
    assert sorted(_listing()) == ["a", "b"]


def test_what_was_captured_comes_from_the_gt_table_then_the_operator(
        day: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.app as app_mod
    table = tmp_path / "gt_tables.csv"
    table.write_text("env,cond,y,stem,date,session,train,dur\n"
                     "env1,rot fan + small,1,a,1012,1012 21:16-21:47,True,\n"
                     "env1,static fan,0,b,1012,1012 22:08-22:28,True,\n")
    monkeypatch.setattr(app_mod, "GT_TABLES_CSV", table)
    _capture(day, "a", meta={"scenario": "typed at capture"})
    _capture(day, "b")
    _capture(day, "c", meta={"scenario": "fan mov / small mov"})
    _capture(day, "d", cv={"summary": {"fraction_occupied": 0.9}})
    got = _listing()
    assert got["a"]["recorded"] == "small move + rotating fan"      # table wins
    assert "1012 21:16-21:47" in got["a"]["recorded_source"]
    assert got["b"]["recorded"] == "static fan only"
    assert got["c"]["recorded"] == "fan mov / small mov"
    assert got["c"]["recorded_source"].startswith("operator")
    assert "recorded" not in got["d"]                                # never the camera's guess


def test_binary_gt_comes_from_the_gt_table_then_the_operators_presence(
        day: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The compare table's picker filters on one person / empty field: the
    user's GT table first, else the operator's label; never the camera."""
    import backend.app as app_mod
    table = tmp_path / "gt_tables.csv"
    table.write_text("env,cond,y,stem,date,session,train,dur\n"
                     "env1,rot fan + small,1,a,1012,1012 21:16-21:47,True,\n"
                     "env1,static fan,0,b,1012,1012 22:08-22:28,True,\n")
    monkeypatch.setattr(app_mod, "GT_TABLES_CSV", table)
    _capture(day, "a", meta={"presence": "empty"})                    # the table wins over the sidecar
    _capture(day, "b")
    _capture(day, "c", meta={"presence": "occupied", "presence_source": "operator_manual"})
    _capture(day, "d", cv={"summary": {"fraction_occupied": 0.9}})
    got = _listing()
    assert got["a"]["gt"] == "person" and got["a"]["gt_source"] == "GT table env1"
    assert got["b"]["gt"] == "empty"
    assert got["c"]["gt"] == "person" and got["c"]["gt_source"] == "operator_manual"
    assert "gt" not in got["d"]
