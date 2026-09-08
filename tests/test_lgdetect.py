"""The LG on-board detector, replayed over a capture under the board's NumPy."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import lgdetect
from backend.app import app

CAPTURES = Path(__file__).resolve().parent.parent / "captures"
client = TestClient(app)


def _capture() -> Path:
    p = CAPTURES / "20260904_193228.bin"
    if not p.exists():
        pytest.skip("capture not available")
    if not lgdetect.available():
        pytest.skip("board venv (.venv-board, numpy 1.x) not installed")
    return p


def test_it_runs_under_numpy_1_not_the_project_venv() -> None:
    """The interpreter is the experiment.

    Its length arithmetic shifts a numpy uint8 left by 8. NumPy 2 keeps that as
    uint8 and evaluates it to 0, so tags 8 and 9 (512 bytes each) read as
    length 0 and the TLV walk desynchronises at the first CSI field -- silently.
    A replay on the project venv would be measuring the NumPy version.
    """
    result = lgdetect.run(_capture())
    assert result["numpy"].startswith("1.")
    # A desynchronised walk shows up as parse failures; a correct one has none.
    assert result["parseFailures"] == 0
    assert result["records"] > 20000


def test_the_verdict_is_a_state_machine_over_its_events() -> None:
    result = lgdetect.run(_capture())
    spans = lgdetect.intervals(result["events"], result["duration"])
    assert spans, "the detector reported nothing at all"
    for a, b in spans:
        assert b > a
    # Ordered and non-overlapping: it is one state at a time.
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end


def test_a_second_run_is_served_from_cache() -> None:
    path = _capture()
    lgdetect.run(path)
    assert lgdetect.run(path)["cached"] is True


def test_the_endpoint_scores_it_against_the_camera() -> None:
    r = client.get("/api/lgdetect", params={"path": str(_capture())})
    assert r.status_code == 200
    body = r.json()
    truth = body["truth"]
    assert truth is not None, "this capture carries labels, so it must be scored"
    n = truth["tp"] + truth["fp"] + truth["fn"] + truth["tn"]
    assert n == len(truth["timeS"])
    # baseRate is what always saying "present" would score; reporting it is the
    # difference between a number and a claim.
    assert 0.0 < truth["baseRate"] < 1.0
