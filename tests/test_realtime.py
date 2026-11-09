"""backend.realtime: the self ratio, reading a growing MTK file, cutting the stream into segments, following a running
capture, and the API -- all on synthetic MTK streams, no board."""

from __future__ import annotations

import fcntl
import math
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import realtime
from backend.mtk import MAGIC, MTKIndex
from backend.realtime import FrameReader, RealtimeMonitor, SegmentWriter, SelfRatio, flock_held
from tests.test_mtk import band, group_records

FS = 43.0


def mtk_stream(seconds: float, *, motion: tuple[float, float] | None = None, start: float = 0.0, seed: int = 0,
               t0_ms: int = 5_000_000) -> bytes:
    """Frames at 43 Hz: chain 1 = chain 0 x g + noise. Still, g is constant and only the noise moves the ratio; during
    ``motion`` g turns at 0.8 Hz, which moves the ratio far more over 0.5 s than over one frame."""
    rng = np.random.default_rng(seed)
    h0 = band(seed=1)
    out = []
    i0 = int(round(start * FS))
    for i in range(i0, i0 + int(round(seconds * FS))):
        t = i / FS
        g = 0.6 + 0.0j
        if motion is not None and motion[0] <= t < motion[1]:
            g += 0.4 * np.exp(2j * np.pi * 0.8 * t)
        noise = rng.normal(0, 25, h0.size) + 1j * rng.normal(0, 25, h0.size)
        h1 = np.where(h0 != 0, np.round(h0 * g + noise), 0)
        out.append(group_records(i + 1, {(0, 0): h0, (1, 0): h1}, ts=t0_ms + int(round(1000 * t))))
    return b"".join(out)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> tuple[np.ndarray, np.ndarray]:
    """20 s, a person-like movement from 10 to 14 s, read back as (times, ratio)."""
    path = tmp_path_factory.mktemp("rt") / "cap.bin"
    path.write_bytes(mtk_stream(20.0, motion=(10.0, 14.0)))
    t, r, n = FrameReader(path).read()
    assert n == t.size == int(20 * FS)
    return t, r


# ---------------------------------------------------------------------- the signal
def test_still_reads_about_one_and_movement_far_above(recorded) -> None:
    t, r = recorded
    out = SelfRatio().push(t, r)
    g = np.array([o[0] for o in out]) - t[0]
    x = np.array([o[1] for o in out])
    still = x[(g > 2) & (g < 9.5)]
    moving = x[(g > 10.6) & (g < 14)]
    assert np.isfinite(still).all() and 0.8 < np.median(still) < 1.25
    assert np.median(moving) > 3 * np.median(still)


def test_values_come_every_hop_and_never_ahead_of_the_frames(recorded) -> None:
    t, r = recorded
    sr = SelfRatio()
    seen: list[float] = []
    for a in range(0, t.size, 7):
        new = sr.push(t[a:a + 7], r[a:a + 7])
        assert all(g <= t[min(a + 7, t.size) - 1] for g, _ in new)    # the window of a value is complete when it is sent
        seen += [g for g, _ in new]
    assert np.allclose(np.diff(seen), realtime.HOP_S)


def test_chunking_does_not_change_the_values(recorded) -> None:
    t, r = recorded
    whole = SelfRatio().push(t, r)
    sr = SelfRatio()
    rng = np.random.default_rng(3)
    parts, a = [], 0
    while a < t.size:
        b = a + int(rng.integers(1, 60))
        parts += sr.push(t[a:b], r[a:b])
        a = b
    assert [g for g, _ in parts] == [g for g, _ in whole]
    np.testing.assert_allclose([x for _, x in parts], [x for _, x in whole], rtol=1e-12, equal_nan=True)


def test_a_gap_is_marked_and_a_clock_going_back_starts_afresh(recorded) -> None:
    t, r = recorded
    sr = SelfRatio()
    sr.push(t[:200], r[:200])
    out = sr.push(t[400:] + 60.0, r[400:])          # a minute of nothing
    assert math.isnan(out[0][1]) and out[1][0] > t[400] + 60.0
    again = sr.push(t[:100], r[:100])               # the board rebooted: its clock starts over
    assert all(g <= t[99] for g, _ in again)


# ---------------------------------------------------------------------- reading
def test_a_growing_file_is_read_once_frame_by_frame(tmp_path: Path) -> None:
    blob = mtk_stream(4.0)
    path = tmp_path / "grow.bin"
    path.write_bytes(blob[: len(blob) // 2 + 17])   # mid-record
    rd = FrameReader(path)
    t1, r1, _ = rd.read()
    with path.open("ab") as fh:
        fh.write(blob[len(blob) // 2 + 17:])
    t2, r2, _ = rd.read()
    assert rd.read()[2] == 0
    whole_t, whole_r, _ = FrameReader(tmp_path / "grow.bin").read()
    np.testing.assert_array_equal(np.concatenate([t1, t2]), whole_t)
    np.testing.assert_array_equal(np.concatenate([r1, r2]), whole_r)


def test_frames_from_another_sender_are_left_out(tmp_path: Path) -> None:
    ours = mtk_stream(6.0)
    other = group_records(9000, {(0, 0): band(seed=5), (1, 0): band(seed=6)}, ts=5_000_000, mac="aa:bb:cc:dd:ee:ff")
    path = tmp_path / "mixed.bin"
    path.write_bytes(other + ours)
    t, r, n = FrameReader(path).read()
    assert n == t.size + 1


class _Chunks:
    """A pipe that hands out the bytes in uneven pieces."""

    def __init__(self, blob: bytes, seed: int = 0) -> None:
        self.blob, self.pos, self.rng = blob, 0, np.random.default_rng(seed)

    def read1(self, n: int) -> bytes:
        k = int(self.rng.integers(1, 3000))
        out = self.blob[self.pos:self.pos + min(n, k)]
        self.pos += len(out)
        return out


def test_segments_are_cut_between_records_and_lose_nothing(tmp_path: Path) -> None:
    blob = mtk_stream(8.0)
    w = SegmentWriter(_Chunks(blob), tmp_path / "seg", segment_bytes=60_000).start()
    w.join(timeout=10)
    assert len(w.segments) > 3 and set(w.segments) == w.closed and not w.desynced
    data = [p.read_bytes() for p in w.segments]
    assert b"".join(data) == blob
    assert all(d[0] == MAGIC for d in data)
    frames = sum(MTKIndex(p).count for p in w.segments)
    assert frames >= int(8 * FS) - (len(w.segments) - 1)   # at most the group straddling each cut


# ---------------------------------------------------------------------- the capture lock
def test_a_held_lock_is_seen_without_taking_it(tmp_path: Path) -> None:
    lock = tmp_path / "mtk-capture.lock"
    lock.touch()
    assert not flock_held(lock)
    fd = os.open(lock, os.O_WRONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert flock_held(lock)
    finally:
        os.close(fd)
    assert not flock_held(lock)
    assert not flock_held(tmp_path / "missing.lock")


def test_the_monitor_reads_a_running_capture_and_lets_go_when_it_ends(tmp_path: Path) -> None:
    caps = tmp_path / "captures"
    caps.mkdir()
    lock = tmp_path / "mtk-capture.lock"
    lock.touch()
    cap = caps / "20261109_120000.bin"
    cap.write_bytes(mtk_stream(3.0))
    m = RealtimeMonitor(board=None, capture_dir=caps, local_lock=str(lock), rt_dir=tmp_path / "rt", threshold=2.0)
    fd = os.open(lock, os.O_WRONLY)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        m._tick()                                   # attaches near the file's end
        with cap.open("ab") as fh:
            fh.write(mtk_stream(9.0, start=3.0, motion=(6.0, 10.0)))
        m._tick()
        s = m.status(window_s=3600)
        assert s["source"] == "capture" and s["source_file"] == cap.name and s["state"] == "capture"
        assert len(s["t"]) > 20 and set(s["src"]) == {1}
        assert len(s["alarms"]) >= 1
        x = np.array([v for v in s["x"] if v is not None])
        assert np.nanmax(x) > 2.0 and np.nanmedian(x) < 2.0
    finally:
        os.close(fd)
    m._tick()                                       # the capture let go: a batch's next capture may follow at once
    s = m.status()
    assert s["source"] is None and s["state"] == "waiting" and "just ended" in s["detail"]
    m._free_since -= realtime.CAPTURE_SETTLE_S     # none did; no board here, so it waits for one
    m._tick()
    assert m.status()["state"] == "waiting" and "no board" in m.status()["detail"]


def test_a_capture_that_is_not_growing_here_is_reported(tmp_path: Path) -> None:
    lock = tmp_path / "mtk-capture.lock"
    lock.touch()
    m = RealtimeMonitor(board=None, capture_dir=tmp_path, local_lock=str(lock), rt_dir=tmp_path / "rt")
    fd = os.open(lock, os.O_WRONLY)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        m._tick()
    finally:
        os.close(fd)
    assert m.state == "capture" and "WiFi" in m.detail


# ---------------------------------------------------------------------- API
def test_the_api_reports_and_sets_the_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.app import app

    m = RealtimeMonitor(board=None, capture_dir=tmp_path, local_lock=str(tmp_path / "l.lock"), rt_dir=tmp_path / "rt")
    monkeypatch.setattr(realtime, "_monitor", m)
    client = TestClient(app)
    s = client.get("/api/realtime/status", params={"since": 0, "window_s": 600}).json()
    assert s["threshold"] == realtime.DEFAULT_THRESHOLD and s["ping_interval_s"] == 0.02
    assert s["signal"] == {"lag_s": 0.5, "window_s": 0.5, "hop_s": 0.25, "refractory_s": 5.0}
    assert s["rate_table"][0]["rate_hz"] == 43.0
    assert client.post("/api/realtime/threshold", json={"threshold": 1.8}).json()["threshold"] == 1.8
    assert client.post("/api/realtime/threshold", json={"threshold": 0.1}).status_code == 400
