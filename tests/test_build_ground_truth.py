"""scripts/build_ground_truth.py: the table, the sidecars and the camera become one row per one-minute capture."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_ground_truth.py"
spec = importlib.util.spec_from_file_location("build_ground_truth", SCRIPT)
bgt = importlib.util.module_from_spec(spec)
sys.modules["build_ground_truth"] = bgt
spec.loader.exec_module(bgt)


def make(root: Path, stem: str, span=58.0, frac=None, scenario=None, edited=False, size=11_000_000):
    day = root / stem[:8]
    day.mkdir(exist_ok=True)
    (day / f"{stem}.bin").write_bytes(b"\0" * size)
    if span is not None:
        summary = {"span_s": span}
        if frac is not None:
            summary["fraction_occupied"] = frac
        (day / f"{stem}_cv.json").write_text(json.dumps({"summary": summary}))
        if edited:
            (day / f"{stem}_cv.json.orig").write_text("{}")
    if scenario:
        (day / f"{stem}_meta.json").write_text(json.dumps({"scenario": scenario}))


@pytest.fixture
def tree(tmp_path):
    make(tmp_path, "20260909_120000", frac=0.0)                   # stray one-minute file before the protocol
    make(tmp_path, "20260930_101500", frac=0.0)                   # env#1 empty by camera
    make(tmp_path, "20260930_160000", frac=1.0)                   # env#1 occupied by camera
    make(tmp_path, "20260930_163000", frac=0.4)                   # env#1 partial: person entering
    make(tmp_path, "20260930_180700", frac=0.0)                   # env#1 camera-blind block
    make(tmp_path, "20260930_170000", span=200.0, frac=0.0)       # three-minute capture: not listed
    make(tmp_path, "20260930_190000", span=None, size=11_000_000)  # no sidecar: size says one minute
    make(tmp_path, "20261001_151500", frac=0.0)                   # table: empty
    make(tmp_path, "20261001_184200", frac=1.0, edited=True)      # table: small_move
    make(tmp_path, "20261005_124802", frac=1.0)                   # env#2, not in the table
    make(tmp_path, "20261005_205501", frac=0.0, scenario="robot_vacuum_no_person")
    make(tmp_path, "20261005_205606", frac=0.0, scenario="robot_vacuum_no_person")
    make(tmp_path, "20261006_200331", frac=0.0, scenario="person_outside_camera_view_distance_test")
    return tmp_path


TABLE = {
    "empty": {"20261001": ["15:15"]},
    "small_move": {"20261001": ["18:42"]},
    "robot_vacuum": {"20261005": [("20:55", "20:56", 2)]},
}


def rows_for(tree):
    captures = bgt.scan(str(tree))
    problems = bgt.apply_table(captures, TABLE)
    return {r["capture"]: r for r in bgt.build_rows(captures)}, problems


def test_only_one_minute_captures_are_listed(tree):
    rows, problems = rows_for(tree)
    assert problems == []
    assert "20260930_170000" not in rows
    assert rows["20260930_190000"]["duration_s"] == "58"          # size proxy
    assert rows["20260930_190000"]["use"] == 0 and "no camera" in rows["20260930_190000"]["reason"]


def test_environment_boundary(tree):
    rows, _ = rows_for(tree)
    assert rows["20260930_101500"]["env"] == 1
    assert rows["20261001_151500"]["env"] == 2


def test_table_rows_are_authoritative(tree):
    rows, _ = rows_for(tree)
    e = rows["20261001_151500"]
    assert (e["label"], e["activity"], e["nonhuman"], e["source"], e["use"]) == ("empty", "none", 0, "table", 1)
    s = rows["20261001_184200"]
    assert (s["label"], s["activity"], s["source"], s["use"], s["camera_edited"]) == ("occupied", "small_move", "table", 1, 1)
    r = rows["20261005_205501"]
    assert (r["label"], r["activity"], r["nonhuman"], r["source"], r["use"]) == ("empty", "robot_vacuum", 1, "table", 1)


def test_env1_is_labelled_by_the_camera(tree):
    rows, _ = rows_for(tree)
    assert (rows["20260930_101500"]["label"], rows["20260930_101500"]["source"], rows["20260930_101500"]["use"]) == ("empty", "camera", 1)
    o = rows["20260930_160000"]
    assert (o["label"], o["activity"], o["use"]) == ("occupied", "unknown", 1)
    p = rows["20260930_163000"]
    assert p["use"] == 0 and "partial" in p["reason"]
    b = rows["20260930_180700"]
    assert b["label"] == "empty" and b["use"] == 0 and "camera-blind" in b["reason"]


def test_unlisted_env2_captures_are_evidence_not_labels(tree):
    rows, _ = rows_for(tree)
    u = rows["20261005_124802"]
    assert (u["label"], u["source"], u["use"]) == ("occupied", "camera", 0)
    d = rows["20261006_200331"]
    assert (d["label"], d["activity"], d["source"], d["use"]) == ("occupied", "static", "sidecar", 0)
    assert "distance_test" in d["reason"]
    assert rows["20260909_120000"]["use"] == 0 and "before the protocol" in rows["20260909_120000"]["reason"]


def test_table_file_disagreements_are_reported(tree):
    captures = bgt.scan(str(tree))
    problems = bgt.apply_table(captures, {"empty": {"20261001": ["15:16"]}, "static": {"20261005": [("20:55", "20:56", 3)]}})
    assert any("15:16" in p and "0 files" in p for p in problems)
    assert any("table 3, files 2" in p for p in problems)


def test_the_real_table_is_internally_consistent():
    # every expected count in a range is positive; no capture time is listed under two classes on the same day
    seen = {}
    for cls, days in bgt.TABLE.items():
        for day, items in days.items():
            for it in items:
                if isinstance(it, tuple):
                    assert it[2] > 0
                else:
                    assert seen.setdefault((day, it), cls) == cls, (day, it)
    assert set(bgt.OCCUPIED_ACTIVITIES) | set(bgt.NONHUMAN) | {"empty"} == set(bgt.TABLE)
