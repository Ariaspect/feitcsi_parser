"""Real-time motion channel: a movement value every 0.25 s from an always-on CSI stream (the Realtime tab).

The presence verdict sees the room for one minute in five. This watches it all the time, and says *movement* --
never *person*: a robot or a rotating fan moves too.

The signal is the self ratio (measured offline 2026-11-02 on 1,097 labelled minutes of 4 rooms at 43 Hz)::

    x(g) = median over (g - W, g] of s_L  /  median over (g - W, g] of s_1
    s_L(i) = median over subcarriers of |r_i - r_(i-L)| / (|r_i| + |r_(i-L)|)

on the CSI ratio ``r`` (the AP's second transmit chain over its first, ``af8.load_ratio``), L = 0.5 s (s_1: one
frame), W = 0.5 s, one value every 0.25 s, causal. Noise moves the ratio as much in one frame as in half a second,
so an empty room reads about 1 in every room measured (all at one link state -- a noisy link is untested); a body
moving moves the half-second step far more than the one-frame step. Measured at threshold 2.14 (set on the other
rooms): 0.3 alarms/h on 3.4 h of empties; minutes that fire: big move 100 %, small 77 %, still person 28 %, static
fan 26 %, rotating fan 100 %, robot 47 %; 7 of 8 natural entries found within -9.3 / +10.3 s of the camera.
``RATE_TABLE`` holds the same measurement at lower frame rates (frames dropped from the 43 Hz recordings).

Where the CSI comes from:

* **its own stream** -- ``ssh root@<board> /var/csi/csi_stream.sh`` (the script ``capture_mtk_hourly.sh`` deploys),
  the stimulus of a capture (``capture_control.PING_INTERVAL``), written to rotating segment files under ``RT_DIR``
  and read back incrementally with ``MTKIndex.extend``. Segments are deleted once read: nothing is kept.
* **a capture's file** -- the board's stream takes one reader (``/var/csi/capture.lock``). When a capture starts on
  this laptop (its script holds ``LOCAL_LOCK``, from the web, cron or a shell) the own stream is stopped at once --
  the script spends several ssh round trips before it reaches the board's lock -- and the capture's growing ``.bin``
  is read instead, so the curve goes on. When the capture's lock is released the own stream starts again.

The board's frame clock counts from boot, so values are placed on the wall clock by the arrival time of the frames
(the smallest arrival lag seen, i.e. network and read delay folded in).
"""

from __future__ import annotations

import collections
import math
import os
import re
import signal
import socket
import subprocess
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from . import capture_control, mtk

LAG_S = 0.5
WINDOW_S = 0.5
HOP_S = 0.25
REFRACTORY_S = 5.0
# Measured 2026-11-02 at 43 Hz: the cut where an empty room alarms 0.3 times an hour, set on the other rooms.
DEFAULT_THRESHOLD = 2.14
# The same measurement with frames dropped from the 43 Hz recordings, the threshold reset per rate (about one empty
# alarm an hour or fewer). The lower rates needed a longer window or lag, so their rows are for a signal this module
# does not compute; they are shown so a rate drop can be read against them. % = share of labelled minutes that fire.
RATE_TABLE: tuple[dict[str, Any], ...] = (
    {"rate_hz": 43.0, "window_s": 0.5, "lag_s": 0.5, "threshold": 2.14, "empty_alarms_per_h": 0.3, "big": 100,
     "small": 77, "still": 28, "static_fan": 26, "robot": 47, "entries": "7/8", "onset_s": 0.5},
    {"rate_hz": 21.5, "window_s": 1.0, "lag_s": 0.5, "threshold": 1.57, "empty_alarms_per_h": 0.3, "big": 100,
     "small": 92, "still": 41, "static_fan": 41, "robot": 61, "entries": "7/8", "onset_s": 1.0},
    {"rate_hz": 10.8, "window_s": 1.0, "lag_s": 0.5, "threshold": 1.60, "empty_alarms_per_h": 1.2, "big": 100,
     "small": 81, "still": 66, "static_fan": 71, "robot": 72, "entries": "6/8", "onset_s": 2.9},
    {"rate_hz": 5.4, "window_s": 1.0, "lag_s": 1.0, "threshold": 1.83, "empty_alarms_per_h": 1.2, "big": 99,
     "small": 77, "still": 67, "static_fan": 75, "robot": 44, "entries": "6/8", "onset_s": 7.8},
    {"rate_hz": 2.7, "window_s": 2.0, "lag_s": 2.0, "threshold": 1.77, "empty_alarms_per_h": 1.8, "big": 98,
     "small": 77, "still": 66, "static_fan": 76, "robot": 55, "entries": "7/8", "onset_s": 3.5},
)

# What the alarm is judged on, in the project's own terms, and what it separates -- the 2026-11-02 measurement
# (1,097 labelled minutes of env1 / env2 / env3 / env3-1 at 43 Hz, threshold 2.14 set on the other rooms).
METRIC: dict[str, Any] = {
    "name": "self ratio x = 0.5 s step / one-frame step",
    "family": [
        "Its numerator is the step the sequence models read every second (s025 / s2 / s5 of GRU + 41 and of the "
        "Transformer tab: median over subcarriers of |r(t) - r(t - L)| / (|r(t)| + |r(t - L)|) on the CSI ratio), "
        "here at L = 0.5 s and on every frame.",
        "It is the symmetric relative step of the Hybrid 2 frame step (framediff), but on the CSI ratio, which "
        "cancels the receiver's gain steps that make up the raw-amplitude step's loudest tail.",
        "It is divided by its own one-frame step -- the noise of the moment -- so it needs no empty-room "
        "calibration, unlike the 11F / 41F / GRU inputs, which are read against a room's empty minutes.",
        "It is not a model and none of the 11F / 41 features: one number against one threshold (the 2026-11-02 "
        "real-time test).",
    ],
    "separates": [
        ["movement from an empty room, in every room measured, without calibration",
         "empty rooms: 0.3 alarms/h over 3.4 h; big moves fire in 100 % of minutes"],
        ["an arrival, within seconds",
         "7 of 8 camera-dated entries, -9.3 to +10.3 s from the camera (median -0.2 s); the one-minute-in-five "
         "verdict took a median 183 s"],
        ["small movement, most of the time", "77 % of small-move minutes fire"],
    ],
    "does_not": [
        ["a person from a non-human mover", "rotating fan 100 %, robot 47 %, static fan 26 % of minutes fire"],
        ["anything while a rotating fan runs",
         "x stays above the threshold 86-93 % of the time; the one missed entry was under a running fan"],
        ["a still person from an empty room",
         "28 % of still minutes fire -- presence stays the one-minute verdict's job"],
        ["big from small across rooms", "a 'big only' threshold is 5.0 / 6.0 / 5.3 / 2.6 in env1 / 2 / 3 / 3-1"],
        ["an exit (the room becoming empty)", "it fires on arrivals; 'empty' comes from the verdict"],
        ["a noisy link (untested)",
         "all four rooms were one link state; September's noisy-link captures (19.6 Hz) alarmed 12-73 times an "
         "hour on empties"],
    ],
}
KEEP_S = 2 * 3600.0                 # history the tab can show
PERF_KEEP_S = 600.0                 # computation-time statistics cover the last 10 minutes
RATE_BIN_S = 1.0                    # sampling rate reported per second
POLL_S = 0.2
SEGMENT_BYTES = 16 * 1024 * 1024    # ~90 s of stream at ~180 kB/s
# One ssh stream lasts this long, then the next one starts (a gap of a few seconds an hour). Odd on purpose: no
# capture asks for it (the capture tab caps at 3300 s), so a board process with this argument is one of ours.
STREAM_SECONDS = 3601
BOARD_SCRIPT = "/var/csi/csi_stream.sh"
RT_DIR = Path(os.environ.get("REALTIME_DIR", "/tmp/csi-realtime"))
RESTART_BACKOFF_S = 10.0
CAPTURE_FRESH_S = 10.0              # a capture file modified this recently is the one being written
# The own stream resumes only once no capture has held the lock for this long: a batch launches its next capture
# within a fraction of a second of the last (gap 0), and grabbing the board in between would only be undone at once.
CAPTURE_SETTLE_S = 5.0
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR"]
_CAPTURE_NAME = re.compile(r"^\d{8}_\d{6}\.bin$")


# ---------------------------------------------------------------------- signal
class SelfRatio:
    """The self ratio over a stream of CSI-ratio frames, fed chunk by chunk; causal and complete.

    A value at hop time ``g`` is emitted once a frame later than ``g`` has arrived, so its window is final. Frames
    must come in time order; a step back in time (a new board boot) or a change of width starts afresh.
    """

    def __init__(self, lag_s: float = LAG_S, window_s: float = WINDOW_S, hop_s: float = HOP_S) -> None:
        self.lag_s, self.window_s, self.hop_s = float(lag_s), float(window_s), float(hop_s)
        self.reset()

    def reset(self) -> None:
        self._t = np.zeros(0)
        self._r: np.ndarray | None = None
        self._st = np.zeros(0)
        self._s1 = np.zeros(0)
        self._sl = np.zeros(0)
        self._next: float | None = None
        self.dt: float | None = None
        # the last push: (seconds on the per-frame steps, seconds on the per-hop windows, frames, values)
        self.last_timing: tuple[float, float, int, int] = (0.0, 0.0, 0, 0)

    @staticmethod
    def _step(R: np.ndarray, mag: np.ndarray, T: np.ndarray, idx: np.ndarray, L: int, want: float, dt: float) -> np.ndarray:
        if idx.size == 0:
            return np.zeros(0)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN rows: a frame with no usable subcarrier
            v = np.abs(R[idx] - R[idx - L]) / (mag[idx] + mag[idx - L])
            s = np.nanmedian(v, axis=1)
        # a pair that straddles a dropout is not a step at this lag
        s[(T[idx] - T[idx - L]) > 1.5 * want + dt] = np.nan
        return s

    def push(self, t: np.ndarray, r: np.ndarray, first_hop: float | None = None) -> list[tuple[float, float]]:
        """Add frames (times in s, ratio ``(n, subcarriers)``); returns the new ``(hop time, x)`` values."""
        t = np.asarray(t, dtype=float)
        if t.size == 0:
            return []
        c0 = time.perf_counter()
        if self._r is not None and (r.shape[1] != self._r.shape[1] or t[0] < self._t[-1]):
            self.reset()
        T = t if self._r is None else np.concatenate([self._t, t])
        R = r if self._r is None else np.concatenate([self._r, r])
        m = T.size - t.size
        recent = np.diff(T[-400:])
        if recent.size:
            self.dt = float(np.median(recent))
        dt = self.dt
        if not dt or dt <= 0:
            self._t, self._r = T, R
            return []
        L = max(1, int(round(self.lag_s / dt)))
        mag = np.abs(R)
        new = np.arange(max(m, 1), T.size)
        s1 = self._step(R, mag, T, new, 1, dt, dt)
        sl = np.full(new.size, np.nan)
        has = new >= L
        sl[has] = self._step(R, mag, T, new[has], L, self.lag_s, dt)
        self._st = np.concatenate([self._st, T[new]])
        self._s1 = np.concatenate([self._s1, s1])
        self._sl = np.concatenate([self._sl, sl])
        # frames kept: enough for the lag of the next chunk's first frame
        cut = max(0, min(T.size - (L + 1), int(np.searchsorted(T, T[-1] - self.lag_s - 1.0))))
        self._t, self._r = T[cut:], R[cut:]

        c1 = time.perf_counter()
        out: list[tuple[float, float]] = []
        if self._next is None:
            start = T[0] + self.lag_s + self.window_s if first_hop is None else first_hop
            self._next = start if first_hop is not None else math.ceil(start / self.hop_s) * self.hop_s
        elif t[0] - self._next > 5.0:
            # a gap (stream restarted, source switched): one empty value marks it, then on from the new frames
            out.append((self._next, math.nan))
            self._next = math.ceil((t[0] + self.lag_s) / self.hop_s) * self.hop_s
        need = max(2, int(0.5 * self.window_s / dt))
        while self._next <= T[-1]:
            g = self._next
            a = int(np.searchsorted(self._st, g - self.window_s, side="right"))
            b = int(np.searchsorted(self._st, g, side="right"))
            v1, vl = self._s1[a:b], self._sl[a:b]
            v1, vl = v1[np.isfinite(v1)], vl[np.isfinite(vl)]
            x = float(np.median(vl) / np.median(v1)) if (b - a >= need and v1.size >= need and vl.size >= need) else math.nan
            out.append((g, x))
            self._next = g + self.hop_s
        keep = int(np.searchsorted(self._st, self._next - self.window_s - self.hop_s, side="right"))
        if keep:
            self._st, self._s1, self._sl = self._st[keep:], self._s1[keep:], self._sl[keep:]
        self.last_timing = (c1 - c0, time.perf_counter() - c1, int(t.size), len(out))
        return out


class FrameReader:
    """CSI-ratio frames appended to one growing MTK file since the last call.

    The frame set is the one every other view reads (``framediff._uniform_selection``): the dominant AP, 2x1
    (both of its transmit chains), full width. The AP is fixed once 200 frames have been seen; it can be handed
    on from the previous segment of the same stream.
    """

    def __init__(self, path: str | Path, mac: str | None = None) -> None:
        self.path = Path(path)
        self.index: mtk.MTKIndex | None = None
        self.done = 0
        self.mac = mac
        self._mac_fixed = mac is not None
        self.records_seen = 0

    def skip_to_tail(self, seconds: float) -> None:
        """Start reading near the end of what is already in the file (attaching to a capture mid-way)."""
        if not self._open():
            return
        idx = self.index
        assert idx is not None
        if idx.count:
            last = idx._stamps[-1] / 1e3
            self.done = int(np.searchsorted(idx._stamps / 1e3, last - seconds, side="left"))

    def _open(self) -> bool:
        try:
            if self.index is None:
                if not self.path.exists() or self.path.stat().st_size < mtk.PREFIX_BYTES:
                    return False
                self.index = mtk.MTKIndex(self.path)
            else:
                self.index.extend()
        except (OSError, ValueError, IndexError):
            return False
        return True

    def read(self) -> tuple[np.ndarray, np.ndarray, int]:
        """``(times s, ratio (n, subcarriers), frames read)`` -- the board's clock; empty when nothing is new."""
        empty = (np.zeros(0), np.zeros((0, 0), dtype=complex), 0)
        if not self._open():
            return empty
        idx = self.index
        assert idx is not None
        n = idx.count
        if n <= self.done:
            return empty
        ids = np.arange(self.done, n)
        self.done = n
        if not self._mac_fixed:
            counts = collections.Counter(idx.source_macs)
            self.mac = counts.most_common(1)[0][0] if counts else None
            self._mac_fixed = n >= 200
        macs = np.asarray(idx.source_macs, dtype=object)[ids]
        keep = ((idx.num_rx_arr[ids] == 2) & (idx.num_tx_arr[ids] == 1) & (macs == self.mac)
                & (idx._bins[ids] == idx.num_subcarriers))
        sel = ids[keep]
        if sel.size == 0:
            return np.zeros(0), np.zeros((0, 0), dtype=complex), int(ids.size)
        H = mtk.decode_complex(self.path, idx, sel)[:, :2, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = H[:, 1, :] / H[:, 0, :]
        r[~np.isfinite(r)] = np.nan
        return idx._stamps[sel] / 1e3, r, int(ids.size)


# ---------------------------------------------------------------------- helpers
def flock_held(path: str | Path) -> bool:
    """Whether any process holds a flock on ``path``, read from /proc/locks -- without taking it.

    Taking the lock to test it (``capture_control._lock_busy``) is harmless every ten seconds; five times a second
    it would, now and then, make the capture script's own ``flock -n`` fail and SKIP a capture.
    """
    try:
        st = os.stat(path)
    except OSError:
        return False
    key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    try:
        with open("/proc/locks") as fh:
            for line in fh:
                parts = line.split()
                if "FLOCK" in parts and "->" not in parts and key in parts:
                    return True
    except OSError:
        return False
    return False


def wired(board: str) -> tuple[bool, str]:
    """Whether the board is reached over the cable now -- the own stream runs only then (as captures do)."""
    try:
        route = subprocess.run(["ip", "route", "get", board], capture_output=True, text=True, timeout=5).stdout.split()
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"no route to the board ({exc})"
    dev = route[route.index("dev") + 1] if "dev" in route[:-1] else None
    if dev == capture_control.WIRED_IFACE:
        return True, f"cable ({dev})"
    return False, f"no cable: the board is routed via {dev or 'nothing'}, not {capture_control.WIRED_IFACE}"


def _stats(v: np.ndarray) -> dict[str, float | int]:
    """last / min / mean / p50 / p95 / max / n of a series (in the order it was recorded)."""
    v = np.asarray(v, dtype=float)
    return {"last": round(float(v[-1]), 4), "min": round(float(v.min()), 4), "mean": round(float(v.mean()), 4),
            "p50": round(float(np.median(v)), 4), "p95": round(float(np.percentile(v, 95)), 4),
            "max": round(float(v.max()), 4), "n": int(v.size)}


class SegmentWriter:
    """Copies the ssh stream into segment files, cutting at record boundaries (``magic | length u16 LE | body``)."""

    def __init__(self, stream, directory: Path, segment_bytes: int = SEGMENT_BYTES) -> None:
        self.stream = stream
        self.dir = Path(directory)
        self.segment_bytes = int(segment_bytes)
        self.segments: list[Path] = []
        self.closed: set[Path] = set()
        self.bytes = 0
        self.desynced = False
        self._lock = threading.Lock()
        self._fh = None
        self._remain = 0          # body bytes left in the current record
        self._carry = b""         # a prefix split across reads
        self._thread = threading.Thread(target=self._run, name="realtime-writer", daemon=True)

    def start(self) -> "SegmentWriter":
        self.dir.mkdir(parents=True, exist_ok=True)
        self._new_segment()
        self._thread.start()
        return self

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)

    def _new_segment(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self.closed.add(self.segments[-1])
            path = self.dir / f"seg_{len(self.segments):05d}.bin"
            self._fh = open(path, "wb", buffering=0)
            self.segments.append(path)
            self._written = 0

    def _boundaries(self, chunk: bytes) -> list[int]:
        """Offsets in ``chunk`` where a record starts."""
        out = []
        data = self._carry + chunk
        base = -len(self._carry)
        pos = 0
        if self._remain:
            take = min(self._remain, len(data))
            self._remain -= take
            pos = take
        while pos < len(data) and not self._remain:
            if len(data) - pos < mtk.PREFIX_BYTES:
                break
            if data[pos] != mtk.MAGIC:
                self.desynced = True
                pos = len(data)
                break
            length = data[pos + 1] | (data[pos + 2] << 8)
            out.append(base + pos)
            body = len(data) - pos - mtk.PREFIX_BYTES
            if body >= length:
                pos += mtk.PREFIX_BYTES + length
            else:
                self._remain = length - body
                pos = len(data)
        self._carry = data[pos:] if (pos < len(data) and not self._remain and not self.desynced) else b""
        return [b for b in out if b >= 0]

    def _run(self) -> None:
        # an unbuffered pipe (bufsize=0) has read, which returns what is there; a buffered one has read1 for that
        read = self.stream.read1 if hasattr(self.stream, "read1") else self.stream.read
        try:
            while True:
                chunk = read(65536)
                if not chunk:
                    break
                cut = None
                if self._written >= self.segment_bytes and not self.desynced:
                    starts = self._boundaries(chunk)
                    cut = starts[0] if starts else None
                elif not self.desynced:
                    self._boundaries(chunk)
                if cut is not None:
                    if cut:
                        self._fh.write(chunk[:cut])
                    self._new_segment()
                    chunk = chunk[cut:]
                self._fh.write(chunk)
                self._written += len(chunk)
                self.bytes += len(chunk)
        except (OSError, ValueError):
            pass
        finally:
            with self._lock:
                if self._fh is not None:
                    self._fh.close()
                    self._fh = None
                if self.segments:
                    self.closed.add(self.segments[-1])


# ---------------------------------------------------------------------- monitor
class RealtimeMonitor:
    """Keeps the stream running, the self ratio current, and the history the tab draws."""

    def __init__(self, board: str | None = capture_control.DEFAULT_BOARD, capture_dir: str | Path | None = None,
                 local_lock: str = capture_control.LOCAL_LOCK, rt_dir: str | Path = RT_DIR,
                 ping_interval: float = capture_control.PING_INTERVAL, threshold: float = DEFAULT_THRESHOLD) -> None:
        if capture_dir is None:
            shared = capture_control.SHARED_CAPTURE_DIR
            capture_dir = shared if shared.is_dir() else capture_control.REPO / "captures"
        self.board = board or None
        self.capture_dir = Path(capture_dir)
        self.local_lock = local_lock
        self.rt_dir = Path(rt_dir)
        self.ping_interval = float(ping_interval)
        self.threshold = float(threshold)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.enabled = False
        self.state = "stopped"          # stopped | waiting | starting | streaming | capture | error
        self.detail = ""
        self.source: str | None = None  # "stream" | "capture"
        self.source_file: str | None = None
        self.points: collections.deque = collections.deque()   # (wall t, x, source code 0 stream / 1 capture)
        self.rates: collections.deque = collections.deque()    # (wall t of the second, frames in it)
        self.alarms: collections.deque = collections.deque()   # (wall t, x)
        self.log: collections.deque = collections.deque(maxlen=80)
        self.started_at: float | None = None
        self.frames = 0
        self.perf: collections.deque = collections.deque()      # per batch: (wall t, frames, read s, step s, hop s, values)
        self.latency: collections.deque = collections.deque()   # per value: (wall t, s from its time to computed)
        self.host = socket.gethostname()
        self._sr = SelfRatio()
        self._offset: float | None = None
        self._rate_sec: int | None = None
        self._rate_n = 0
        self._last_alarm = -math.inf
        self._proc: subprocess.Popen | None = None
        self._writer: SegmentWriter | None = None
        self._board_pid: int | None = None
        self._board_busy = False        # the board said its lock is held
        self._stream_dir: Path | None = None
        self._reader: FrameReader | None = None
        self._seg_i = 0
        self._next_try = 0.0
        self._free_since: float | None = 0.0   # since when no capture has held LOCAL_LOCK (0: not since we started)

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self.enabled = True
                return
            self._stop.clear()
            self.enabled = True
            self.started_at = time.time()
            self._next_try = 0.0
            self.state, self.detail = "starting", ""
            self._note("monitor started")
            self._thread = threading.Thread(target=self._run, name="realtime-monitor", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self.enabled = False
            self._stop.set()
        if thread is not None:
            thread.join(timeout=20)

    def set_threshold(self, value: float) -> None:
        if not (math.isfinite(value) and 0.5 <= value <= 100):
            raise ValueError(f"threshold must be between 0.5 and 100, got {value}")
        with self._lock:
            self.threshold = float(value)
            self._note(f"threshold set to {value:g}")

    def status(self, since: float = 0.0, window_s: float = 900.0) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            lo = max(float(since), now - float(window_s))
            pts = [p for p in self.points if p[0] > lo]
            rates = [p for p in self.rates if p[0] > lo]
            recent = [n for (t, n) in self.rates if t > now - 6.0]
            return {
                "enabled": self.enabled,
                "state": self.state,
                "detail": self.detail,
                "source": self.source,
                "source_file": self.source_file,
                "board": self.board,
                "now": now,
                "started_at": self.started_at,
                "threshold": self.threshold,
                "default_threshold": DEFAULT_THRESHOLD,
                "ping_interval_s": self.ping_interval,
                "sampling_rate_hz": float(np.mean(recent)) if recent else None,
                "frames": self.frames,
                "signal": {"lag_s": self._sr.lag_s, "window_s": self._sr.window_s, "hop_s": self._sr.hop_s,
                           "refractory_s": REFRACTORY_S},
                "rate_table": list(RATE_TABLE),
                "metric": METRIC,
                "t": [round(p[0], 3) for p in pts],
                "x": [None if not math.isfinite(p[1]) else round(p[1], 4) for p in pts],
                "src": [p[2] for p in pts],
                "rate_t": [p[0] for p in rates],
                "rate": [p[1] for p in rates],
                "alarms": [{"t": a[0], "x": round(a[1], 3)} for a in self.alarms if a[0] > now - float(window_s)],
                "timing": self._timing(now),
                "log": list(self.log)[-30:],
            }

    # ------------------------------------------------------------------ loop
    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self._tick()
                except Exception as exc:  # the loop must not die silently
                    with self._lock:
                        self.state, self.detail = "error", f"{type(exc).__name__}: {exc}"
                        self._note(self.detail)
                    self._stop.wait(1.0)
                self._stop.wait(POLL_S)
        finally:
            self._stop_stream("monitor stopped")
            with self._lock:
                self.state, self.detail, self.source, self.source_file = "stopped", "", None, None

    def _tick(self) -> None:
        if flock_held(self.local_lock):
            self._free_since = None
            if self._proc is not None:
                self._stop_stream("a capture started; the board's stream is handed to it")
            self._follow_capture()
            return
        now = time.time()
        if self._free_since is None:
            self._free_since = now
        if self.source == "capture":
            with self._lock:
                self._note("capture finished; own stream again once no capture follows it")
                self.source, self.source_file = None, None
            self._reader = None
        if self._proc is None and now - self._free_since < CAPTURE_SETTLE_S:
            with self._lock:
                self.state, self.detail = "waiting", "a capture just ended; the own stream resumes unless another follows"
            return
        if self._proc is not None and self._proc.poll() is not None:
            self._drain_stream()
            if self._proc is not None:
                rc = self._proc.poll()
                self._stop_stream(f"stream ended (rc={rc})")
                self._next_try = time.time() + RESTART_BACKOFF_S
        if self._proc is None:
            if time.time() < self._next_try:
                return
            if not self.board:
                with self._lock:
                    self.state, self.detail = "waiting", "no board configured (CAPTURE_BOARD is empty)"
                self._next_try = time.time() + 60
                return
            ok, why = wired(self.board)
            if not ok:
                with self._lock:
                    self.state, self.detail = "waiting", why
                self._next_try = time.time() + RESTART_BACKOFF_S
                return
            self._start_stream()
            return
        self._drain_stream()

    # ------------------------------------------------------------------ own stream
    def _start_stream(self) -> None:
        self._stream_dir = self.rt_dir / f"stream_{int(time.time() * 1000)}"
        cmd = ["ssh", *SSH_OPTS, "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
               f"root@{self.board}",
               f"echo RT_PID $$ >&2; exec {BOARD_SCRIPT} {STREAM_SECONDS} {self.ping_interval:g}"]
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True, bufsize=0)
        self._proc = proc
        self._board_pid = None
        self._writer = SegmentWriter(proc.stdout, self._stream_dir).start()
        self._seg_i = 0
        self._reader = None
        threading.Thread(target=self._read_stderr, args=(proc,), name="realtime-stderr", daemon=True).start()
        with self._lock:
            self.state, self.detail = "starting", f"ssh root@{self.board} {BOARD_SCRIPT} (stimulus every {self.ping_interval:g} s)"
            self.source, self.source_file = "stream", None
            self._note(f"own stream started (pid {proc.pid})")

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            if line.startswith("RT_PID "):
                try:
                    self._board_pid = int(line.split()[1])
                except (IndexError, ValueError):
                    pass
                continue
            with self._lock:
                self._note(line)
                if "FATAL" in line:
                    self.state, self.detail = "error", line
                if "another capture already running" in line:
                    self._board_busy = True

    def _drain_stream(self) -> None:
        w = self._writer
        if w is None:
            return
        if w.desynced:
            # never seen: the driver hands out whole records. Without the framing no segment can be cut, so start over.
            self._stop_stream("the stream lost its record framing; restarting it")
            self._next_try = time.time() + 1.0
            return
        while True:
            with w._lock:
                if self._seg_i >= len(w.segments):
                    return
                path = w.segments[self._seg_i]
                closed = path in w.closed
            if self._reader is None or self._reader.path != path:
                self._reader = FrameReader(path, mac=self._reader.mac if self._reader else None)
            got = self._consume(self._reader, 0)
            if got or not closed:
                if not closed:
                    return
                continue
            # a closed segment read to its end: drop it, go on to the next
            try:
                path.unlink()
            except OSError:
                pass
            self._seg_i += 1

    def _board(self, command: str, timeout: float = 10.0) -> str:
        try:
            res = subprocess.run(["ssh", *SSH_OPTS, "-o", "ConnectTimeout=5", f"root@{self.board}", command],
                                 capture_output=True, text=True, timeout=timeout)
            return (res.stdout.strip() or res.stderr.strip() or f"rc={res.returncode}").replace("\n", " ")
        except (subprocess.TimeoutExpired, OSError) as exc:
            return f"ssh failed: {exc}"

    def _end_board_run(self, pid: int | None) -> str:
        """End our run on the board -- the script and every process under it -- while it is still one tree.

        Killing the local ssh is not enough (measured 2026-11-09): BusyBox ``timeout`` leaves a watcher child per
        command, and those, with ping, keep ``/var/csi/capture.lock`` open for up to an hour after the script is
        gone -- the next capture would fail with "another capture already running". So this runs *before* the local
        ssh is killed, while the run's processes still hang under its PID: disarm (the lock is still ours, so the
        driver is ours to disarm), then TERM and KILL the whole tree. Our runs are also found by their odd duration
        (the bracket keeps the pattern from matching this very command line).
        """
        cmd = (f"IW={capture_control.BOARD_IW}; P='{pid or ''}'; "
               "tree() { for c in $(pgrep -P \"$1\"); do tree \"$c\"; echo \"$c\"; done; }; "
               "{ [ -n \"$P\" ] && kill -0 \"$P\" 2>/dev/null; } "
               f"|| P=$(pgrep -f '[c]si_stream.sh {STREAM_SECONDS} ' | head -1); "
               "if [ -n \"$P\" ]; then $IW wlan0 driver 'set_csi 0' >/dev/null 2>&1; all=\"$(tree $P) $P\"; "
               "kill $all 2>/dev/null; sleep 0.3; kill -9 $all 2>/dev/null; fi; "
               f"i=0; while [ -n \"$(fuser {capture_control.BOARD_LOCK} 2>/dev/null)\" ] && [ $i -lt 20 ]; do "
               "sleep 0.1; i=$((i+1)); done; "
               f"echo \"run $P ended; lock now held by: [$(fuser {capture_control.BOARD_LOCK} 2>/dev/null | tr -s ' ')]\"")
        return self._board(cmd)

    def _clear_stale_lock(self) -> str:
        """The board refused our stream because its lock is held, and no capture runs here (``LOCAL_LOCK`` is free):
        every capture starts on this laptop and holds that lock first, so the holders are leftovers of a run that
        died (ours before a server restart, or a capture killed outright). Disarm, then end them."""
        lock = capture_control.BOARD_LOCK
        cmd = (f"pids=$(fuser {lock} 2>/dev/null); if [ -n \"$pids\" ]; then "
               f"{capture_control.BOARD_IW} wlan0 driver 'set_csi 0' >/dev/null 2>&1; kill $pids 2>/dev/null; sleep 0.5; "
               f"kill -9 $pids 2>/dev/null; fi; echo \"was held by: $pids; now: $(fuser {lock} 2>/dev/null || true)\"")
        return self._board(cmd)

    def _stop_stream(self, reason: str) -> None:
        proc = self._proc
        if proc is None:
            return
        # Just started: wait for the board to say which PID the run has (its first stderr line), so it can be ended.
        deadline = time.time() + 3.0
        while self._board_pid is None and proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        pid = self._board_pid
        if self.board and proc.poll() is None:
            out = self._end_board_run(pid)          # outside the lock: status() must not wait on ssh
            with self._lock:
                self._note("board: " + out)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        elif self.board and self._board_busy:
            if not flock_held(self.local_lock):
                out = self._clear_stale_lock()
                with self._lock:
                    self._note("board lock held with no capture running here; cleared it -- " + out)
            self._board_busy = False
        if self._writer is not None:
            self._writer.join(timeout=2)
        if self._stream_dir is not None:
            for p in self._stream_dir.glob("seg_*.bin"):
                try:
                    p.unlink()
                except OSError:
                    pass
            try:
                self._stream_dir.rmdir()
            except OSError:
                pass
        with self._lock:
            self._note(reason)
        self._proc, self._writer, self._board_pid, self._stream_dir = None, None, None, None
        if self.source == "stream":
            self._reader = None

    # ------------------------------------------------------------------ capture's file
    def _follow_capture(self) -> None:
        newest, mtime = None, 0.0
        try:
            for p in self.capture_dir.iterdir():
                if _CAPTURE_NAME.match(p.name):
                    m = p.stat().st_mtime
                    if m > mtime:
                        newest, mtime = p, m
        except OSError:
            pass
        fresh = newest is not None and time.time() - mtime < CAPTURE_FRESH_S
        if not fresh:
            with self._lock:
                self.source, self.state = "capture", "capture"
                self.detail = ("a capture is running, but no capture file is growing here (still waking the board, "
                               "or recording over WiFi on the board) -- no frames until it is")
            return
        if self._reader is None or self._reader.path != newest:
            self._reader = FrameReader(newest, mac=self._reader.mac if self._reader else None)
            self._reader.skip_to_tail(2.0)
            with self._lock:
                self.source, self.source_file = "capture", newest.name
                self._note(f"following the capture's file {newest.name}")
        self._consume(self._reader, 1)
        with self._lock:
            self.state, self.detail = "capture", f"reading the running capture's file {newest.name}"

    # ------------------------------------------------------------------ frames -> values
    def _consume(self, reader: FrameReader, src: int) -> int:
        c0 = time.perf_counter()
        t, r, n = reader.read()
        read_s = time.perf_counter() - c0
        if n == 0:
            return 0
        now = time.time()
        if t.size:
            lag = now - t[-1]
            if self._offset is None or lag < self._offset or lag > self._offset + 5.0:
                self._offset = lag
        off = self._offset if self._offset is not None else now - (t[-1] if t.size else 0.0)
        with self._lock:
            self.frames += int(t.size)
            # frames the signal used, per wall-clock second; a second closes when a later one starts
            secs, counts = np.unique(np.floor(t + off).astype(np.int64), return_counts=True)
            for sec, cnt in zip(secs.tolist(), counts.tolist()):
                if self._rate_sec is None:
                    self._rate_sec, self._rate_n = sec, cnt
                elif sec > self._rate_sec:
                    self.rates.append((float(self._rate_sec), self._rate_n))
                    for gap in range(self._rate_sec + 1, min(sec, self._rate_sec + 30)):
                        self.rates.append((float(gap), 0))
                    self._rate_sec, self._rate_n = sec, cnt
                else:
                    self._rate_n += cnt
            if t.size and r.size:
                new = self._sr.push(t, r)
                done = time.time()
                step_s, hop_s, _, _ = self._sr.last_timing
                self.perf.append((done, int(t.size), read_s, step_s, hop_s, len(new)))
                for g, x in new:
                    wall = g + off
                    self.points.append((wall, x, src))
                    if math.isfinite(x):
                        self.latency.append((done, done - wall))
                    if math.isfinite(x) and x >= self.threshold and wall - self._last_alarm >= REFRACTORY_S:
                        self.alarms.append((wall, x))
                        self._last_alarm = wall
            if self.source == "stream" and self.state != "streaming" and t.size:
                self.state, self.detail = "streaming", f"own stream from {self.board}, stimulus every {self.ping_interval:g} s"
            cut = now - KEEP_S
            for dq in (self.points, self.rates, self.alarms):
                while dq and dq[0][0] < cut:
                    dq.popleft()
            for dq in (self.perf, self.latency):
                while dq and dq[0][0] < now - PERF_KEEP_S:
                    dq.popleft()
        return n

    def _timing(self, now: float) -> dict[str, Any]:
        """Computation time over the last ``PERF_KEEP_S`` on this host. Frames are handled in batches (one per
        ``POLL_S`` tick), so a per-frame time is a batch's time over its frames; a per-value time is the batch's
        window medians over its values. Latency: from a value's time to when it was computed (stream transfer,
        the tick and the computation together)."""
        rows = [p for p in self.perf if p[1] > 0]
        f = np.array([p[1] for p in rows], dtype=float)
        rd, st, hp = (np.array([p[i] for p in rows], dtype=float) for i in (2, 3, 4))
        nv = np.array([p[5] for p in rows], dtype=float)
        has = nv > 0
        span = (now - rows[0][0]) if len(rows) > 1 else 0.0
        return {
            "host": self.host,
            "window_s": PERF_KEEP_S,
            "batches": len(rows),
            "per_frame_us": _stats((rd + st) / f * 1e6) if rows else None,
            "decode_per_frame_us": _stats(rd / f * 1e6) if rows else None,
            "steps_per_frame_us": _stats(st / f * 1e6) if rows else None,
            "per_value_us": _stats(hp[has] / nv[has] * 1e6) if has.any() else None,
            "batch_ms": _stats((rd + st + hp) * 1e3) if rows else None,
            "frames_per_batch": _stats(f) if rows else None,
            "latency_s": _stats(np.array([v for _, v in self.latency], dtype=float)) if self.latency else None,
            "cpu_percent": round(100.0 * float((rd + st + hp).sum()) / span, 3) if span > 0 else None,
        }

    def _note(self, text: str) -> None:
        self.log.append(f"{time.strftime('%H:%M:%S')}  {text}")


# ---------------------------------------------------------------------- singleton
_monitor: RealtimeMonitor | None = None
_monitor_lock = threading.Lock()


def monitor() -> RealtimeMonitor:
    """The process-wide monitor, configured like the capture manager (``CAPTURE_BOARD``, ``CAPTURE_DIR``,
    ``CAPTURE_LOCK``) plus ``REALTIME_DIR``."""
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            env = os.environ
            _monitor = RealtimeMonitor(board=env.get("CAPTURE_BOARD", capture_control.DEFAULT_BOARD) or None,
                                       capture_dir=env.get("CAPTURE_DIR") or None,
                                       local_lock=env.get("CAPTURE_LOCK", capture_control.LOCAL_LOCK))
        return _monitor


def reset_monitor() -> None:
    """Stop and forget the singleton (tests)."""
    global _monitor
    with _monitor_lock:
        m, _monitor = _monitor, None
    if m is not None:
        m.stop()
