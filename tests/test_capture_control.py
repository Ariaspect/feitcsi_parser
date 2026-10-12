"""backend.capture_control: start, batch and stop the capture script from the web, against a stand-in script.

The stand-in behaves like capture_mtk_hourly.sh as far as the manager can see: it takes ``--duration N``, writes
START / DONE (or SKIP / ABORT) lines with a timestamp prefix to ``$CAPTURE_DIR/mtk_capture.log``, creates the
``.bin``, and spawns a child that lingers -- the webcam grabber of the real run -- so the process-group kill is tested.
"""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import capture_control
from backend.capture_control import CaptureManager

FAKE = r'''#!/usr/bin/env bash
set -u
DUR=60
while [ $# -gt 0 ]; do case $1 in --duration) DUR=$2; shift 2;; *) shift;; esac; done
LOG=$CAPTURE_DIR/mtk_capture.log
log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG"; }
mkdir -p "$CAPTURE_DIR"
echo "retain=$RETAIN_DAYS upload=$UPLOAD cv=$CV" >"$CAPTURE_DIR/env.seen"
if [ "${FAKE_MODE:-ok}" = skip ]; then log "SKIP  previous capture still running"; exit 0; fi
if [ "${FAKE_MODE:-ok}" = abort ]; then log "ABORT board 192.0.2.1 unreachable and serial wake failed"; exit 1; fi
STAMP=$(date '+%Y%m%d_%H%M%S')
sleep 5000 &          # the lingering child a real run has (webcam grabber)
log "START $STAMP.bin  ${DUR}s via enp1s0  (99999MB free)"
: >"$CAPTURE_DIR/$STAMP.bin"
sleep "$DUR"
echo data >>"$CAPTURE_DIR/$STAMP.bin"
log "DONE  $STAMP.bin  4.0K  ${DUR}s  rc=0  10 frames, ${DUR}s span"
exit 0
'''


@pytest.fixture
def fake(tmp_path: Path):
    script = tmp_path / "scripts" / "capture_mtk_hourly.sh"
    script.parent.mkdir()
    script.write_text(FAKE)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    capdir = tmp_path / "captures"
    capdir.mkdir()
    return script, capdir


def make_manager(script: Path, capdir: Path, tmp_path: Path) -> CaptureManager:
    return CaptureManager(script=script, capture_dir=capdir, board=None, remote=None,
                          local_lock=str(tmp_path / "lock"), frame_tmp=str(tmp_path / "frames"))


def wait_until(pred, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def pgid_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_single_capture_runs_to_done_and_reads_the_stamp(fake, tmp_path):
    script, capdir = fake
    m = make_manager(script, capdir, tmp_path)
    m.start(duration_s=10, delay_s=0, count=1)
    # the manager clamps nothing: 10 s is the floor, so the fake sleeps 10 s -- shorten through the script instead
    assert wait_until(lambda: m.status()["state"] in ("running", "finishing", "idle"))
    assert wait_until(lambda: not m.status()["busy"], timeout=25)
    s = m.status()
    assert s["state"] == "idle" and s["error"] is None
    assert len(s["runs"]) == 1
    run = s["runs"][0]
    assert run["outcome"] == "done" and run["rc"] == 0
    assert run["stamp"] and (capdir / f"{run['stamp']}.bin").read_text() == "data\n"
    assert run["summary"].startswith("DONE")
    assert (capdir / "env.seen").read_text().strip() == "retain=0 upload=1 cv=1"
    assert any("START" in line for line in s["log"]) and any(line.endswith("job: 1 x 10 s, delay 0 s, gap 0 s") for line in s["log"])


def test_countdown_then_capture_and_stop_kills_the_whole_group(fake, tmp_path):
    script, capdir = fake
    m = make_manager(script, capdir, tmp_path)
    m.start(duration_s=60, delay_s=2, count=1)
    s = m.status()
    assert s["state"] == "countdown" and 0 < s["countdown_remaining_s"] <= 2
    assert wait_until(lambda: m.status()["state"] == "running", timeout=10)
    pid = m.status()["current"]["pid"]
    stamp = m.status()["current"]["stamp"]
    assert pgid_alive(pid) and (capdir / f"{stamp}.bin").exists()
    m.stop()
    s = m.status()
    assert s["state"] == "idle" and not s["busy"]
    assert s["runs"][0]["outcome"] == "cancelled"
    assert not pgid_alive(pid), "the lingering child of the run must be gone with the process group"
    assert not (capdir / f"{stamp}.bin").exists(), "the partial capture is removed"
    assert any("stop requested" in line for line in s["log"]) and any("partial capture removed" in line for line in s["log"])


def test_stop_during_countdown_runs_nothing(fake, tmp_path):
    script, capdir = fake
    m = make_manager(script, capdir, tmp_path)
    m.start(duration_s=10, delay_s=30, count=3)
    m.stop()
    s = m.status()
    assert s["state"] == "idle" and s["runs"] == [] and not list(capdir.glob("*.bin"))


def test_batch_runs_count_captures_back_to_back(fake, tmp_path):
    script, capdir = fake
    m = make_manager(script, capdir, tmp_path)
    m.start(duration_s=10, delay_s=0, count=2, gap_s=1)
    assert wait_until(lambda: not m.status()["busy"], timeout=45)
    s = m.status()
    assert [r["outcome"] for r in s["runs"]] == ["done", "done"]
    assert len({r["stamp"] for r in s["runs"]}) == 2
    assert len(list(capdir.glob("*.bin"))) == 2


def test_a_skipped_or_aborted_run_stops_the_batch_with_the_scripts_reason(fake, tmp_path, monkeypatch):
    script, capdir = fake
    m = make_manager(script, capdir, tmp_path)
    monkeypatch.setenv("FAKE_MODE", "skip")
    m.start(duration_s=10, delay_s=0, count=3)
    assert wait_until(lambda: not m.status()["busy"], timeout=15)
    s = m.status()
    assert len(s["runs"]) == 1 and s["runs"][0]["outcome"] == "skipped"
    assert "previous capture still running" in s["error"]
    monkeypatch.setenv("FAKE_MODE", "abort")
    m.start(duration_s=10, delay_s=0, count=2)
    assert wait_until(lambda: not m.status()["busy"], timeout=15)
    s = m.status()
    assert s["runs"][0]["outcome"] == "aborted" and s["error"].startswith("ABORT")


def test_sidecar_is_written_when_the_operator_gave_a_scenario(fake, tmp_path):
    script, capdir = fake
    m = make_manager(script, capdir, tmp_path)
    m.start(duration_s=10, delay_s=0, count=1, room="env2", scenario="fan rotating", note="  AP height  ", upload=False, label=False)
    assert wait_until(lambda: not m.status()["busy"], timeout=25)
    run = m.status()["runs"][0]
    side = capdir / f"{run['stamp']}_meta.json"
    assert run["sidecar"] == side.name and side.exists()
    import json
    meta = json.loads(side.read_text())
    assert meta["room"] == "env2" and meta["scenario"] == "fan rotating" and meta["note"] == "AP height"
    assert meta["scenario_source"] == "operator_reported_at_capture" and meta["requested_duration_s"] == 10
    assert (capdir / "env.seen").read_text().strip() == "retain=0 upload=0 cv=0"


def test_validation_and_busy_rules(fake, tmp_path):
    script, capdir = fake
    m = make_manager(script, capdir, tmp_path)
    with pytest.raises(ValueError):
        m.start(duration_s=5)
    with pytest.raises(ValueError):
        m.start(count=0)
    with pytest.raises(ValueError):
        m.start(note="x" * 201)
    m.start(duration_s=10, delay_s=20, count=1)
    with pytest.raises(RuntimeError):
        m.start(duration_s=10, delay_s=0, count=1)
    m.stop()
    # the script's own lock held by someone else -> refuse
    import fcntl
    fd = os.open(str(tmp_path / "lock"), os.O_WRONLY | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError):
            m.start(duration_s=10, delay_s=0)
        assert m.status()["preflight"] is not None
    finally:
        os.close(fd)


def test_api_round_trip(fake, tmp_path, monkeypatch):
    script, capdir = fake
    monkeypatch.setenv("CAPTURE_SCRIPT", str(script))
    monkeypatch.setenv("CAPTURE_DIR", str(capdir))
    monkeypatch.setenv("CAPTURE_BOARD", "")
    monkeypatch.setenv("CAPTURE_REMOTE", "")
    capture_control.reset_manager()
    monkeypatch.setattr(capture_control, "LOCAL_LOCK", str(tmp_path / "lock"))
    from backend.app import app
    client = TestClient(app)
    try:
        s = client.get("/api/capture/status").json()
        assert s["state"] == "idle" and s["busy"] is False and "limits" in s
        bad = client.post("/api/capture/start", json={"duration_s": 1})
        assert bad.status_code == 400
        ok = client.post("/api/capture/start", json={"duration_s": 60, "delay_s": 30, "count": 2})
        assert ok.status_code == 200 and ok.json()["state"] == "countdown"
        again = client.post("/api/capture/start", json={"duration_s": 60})
        assert again.status_code == 409
        stopped = client.post("/api/capture/stop").json()
        assert stopped["state"] == "idle" and stopped["busy"] is False
    finally:
        capture_control.manager().stop()
        capture_control.reset_manager()
