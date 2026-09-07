"""Tests for the /api/labels endpoint — ground truth beside a capture."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import app

CAPTURES = Path(__file__).resolve().parent.parent / "captures"
client = TestClient(app)


def test_a_labelled_run_publishes_detections_and_its_protocol() -> None:
    path = CAPTURES / "20260904_192623.bin"
    if not path.exists():
        pytest.skip("labelled capture not available")
    body = client.get("/api/labels", params={"path": str(path)}).json()

    assert body["source"].endswith("_cv.json")
    assert body["position"] == "position1"
    assert [ph["label"] for ph in body["phases"]] == ["empty", "sitting", "empty"]

    present = body["present"]
    # Relative to the capture's first sample, like every other series here --
    # not the local-time string the frame filenames carry.
    assert 0 <= present["timeS"][0] < 2
    assert present["timeS"][-1] < 301
    assert len(present["timeS"]) == len(present["present"])
    # The middle of the run is occupied and the opening stretch is not; this is
    # the protocol, and it is what makes the strip worth drawing.
    assert not any(present["present"][:30])
    assert any(present["present"])


def test_an_unlabelled_capture_is_not_an_error() -> None:
    path = CAPTURES / "20260827_165125.bin"
    if not path.exists():
        pytest.skip("capture not available")
    r = client.get("/api/labels", params={"path": str(path)})
    assert r.status_code == 200
    assert r.json()["present"] is None
    assert r.json()["source"] is None
