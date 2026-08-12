"""Incremental FeitCSI .dat reader.

CSIKit's ``read_file`` re-reads and re-decodes the entire capture on every call,
so a polling UI pays O(file) per refresh and falls behind as the capture grows.
FeitCSI frames are self-delimiting -- a 272-byte header whose first word is the
payload length -- so bytes appended since the last poll can be decoded on their
own. ``CaptureStream`` keeps a byte offset plus a rolling window of decoded
frames and touches only what is new.

Decoding is kept bit-for-bit compatible with CSIKit:

* header fields come from ``FeitCSIBeamformReader.parseHeader``
* pilot interpolation and RSSI scaling call the same upstream functions
* the timestamp chain reproduces ``read_file``'s ftm_clock/mu_clock arithmetic
* amplitude is ``db(abs(csi))`` and phase ``angle(csi)``, matching
  ``csitools.get_CSI`` with ``extract_as_dBm=True``

The one intentional divergence is payload decoding: ``parseCsiData`` unpacks
each complex value in a Python triple loop, which dominates parse time. The
buffer is a flat array of little-endian int16 (real, imag) pairs ordered
rx-major, then tx, then subcarrier, so ``np.frombuffer`` reproduces it exactly
and vectorises the hot path.

``filter_mac`` is not supported here. Upstream compares ``filter_mac.casefold()``
against ``FeitCSIFrame.source_mac``, which is a tuple of six ints rather than a
string, so the filter raises before it can match any FeitCSI frame.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import numpy as np

from CSIKit.reader import FeitCSIBeamformReader
from CSIKit.util.csitools import scale_csi_frame
from CSIKit.util.matlab import db

from .parser import FeitCSICapture, mimo_safe_interpolate

HEADER_BYTES = 272

# Tick counter constants, mirrored from CSIKit.reader.readers.read_feitcsi.
MAX_TICK = 4294967295
TICK_RESOLUTION = 3.125
# ftm_clock wraps after ~13.4 s; past that gap upstream falls back to mu_clock.
FTM_WRAP_SECONDS = (MAX_TICK * TICK_RESOLUTION) / 1e9

# Upper bound on retained frames. The API caps max_packets at 10000, so a
# deeper buffer could never be displayed.
MAX_BUFFERED_FRAMES = 10000


def decode_payload(payload: bytes, header: dict) -> np.ndarray:
    """Vectorised equivalent of ``FeitCSIBeamformReader.parseCsiData``.

    Returns a (subcarriers, rx, tx) complex matrix.
    """
    num_rx = header["num_rx"]
    num_tx = header["num_tx"]
    num_sc = header["num_subcarriers"]
    expected = num_rx * num_tx * num_sc * 4

    if len(payload) < expected:
        raise ValueError(
            f"payload short: {len(payload)} bytes, need {expected} "
            f"({num_sc} subcarriers x {num_rx} rx x {num_tx} tx)"
        )

    flat = np.frombuffer(payload, dtype="<i2", count=num_rx * num_tx * num_sc * 2)
    pairs = flat.reshape(num_rx, num_tx, num_sc, 2)
    matrix = pairs[..., 0] + 1j * pairs[..., 1]
    # (rx, tx, subcarrier) -> (subcarrier, rx, tx)
    return np.ascontiguousarray(matrix.transpose(2, 0, 1), dtype=complex)


class CaptureStream:
    """Rolling view of one growing .dat file, decoded incrementally."""

    def __init__(
        self,
        path: str | Path,
        *,
        scaled: bool = True,
        interpolate: bool = True,
        max_frames: int = MAX_BUFFERED_FRAMES,
    ) -> None:
        self.path = Path(path)
        self.scaled = scaled
        self.interpolate = interpolate
        self.max_frames = max_frames

        self._reader = FeitCSIBeamformReader()
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._offset = 0
        self._amplitude: deque[np.ndarray] = deque(maxlen=self.max_frames)
        self._phase: deque[np.ndarray] = deque(maxlen=self.max_frames)
        self._times: deque[float] = deque(maxlen=self.max_frames)
        self._total_frames = 0
        self._num_subcarriers = 0
        self._bandwidth: str | None = None
        # Tail of the timestamp chain, carried across polls.
        self._prev_ftm: int | None = None
        self._prev_mu: int | None = None
        self._prev_time = 0.0

    def _next_timestamp(self, header: dict) -> float:
        """Reproduce read_file's cumulative timestamp arithmetic."""
        if self._prev_ftm is None or self._prev_mu is None:
            return 0.0

        ftm = header["ftm_clock"]
        mu = header["mu_clock"]

        if (mu - self._prev_mu) / 1e6 < FTM_WRAP_SECONDS:
            if ftm > self._prev_ftm:
                diff = ftm - self._prev_ftm
            else:  # ftm_clock overflow
                diff = ftm + (MAX_TICK - self._prev_ftm)
            return self._prev_time + (diff * TICK_RESOLUTION) / 1e9

        if mu > self._prev_mu:
            diff = mu - self._prev_mu
        else:  # mu_clock overflow
            diff = mu + (MAX_TICK - self._prev_mu)
        return self._prev_time + diff / 1e6

    def _decode_frame(self, header: dict, payload: bytes) -> tuple[np.ndarray, np.ndarray]:
        matrix = decode_payload(payload, header)

        if self.interpolate:
            data = mimo_safe_interpolate(
                self._reader.interpolate, {"header": header, "csi_matrix": matrix}
            )
            matrix = data["csi_matrix"]

        if self.scaled:
            for j in range(header["num_rx"]):
                matrix[:, j, :] = scale_csi_frame(matrix[:, j, :], header["rssi_1"])

        # csitools.get_CSI keeps a (frames, subcarriers, rx, tx) array; the
        # display path then collapses to the first stream. Do it here so the
        # buffer stores one row per frame.
        stream = matrix[:, 0, 0]
        return db(np.abs(stream)), np.angle(stream)

    def update(self) -> None:
        """Decode any frames appended since the last call."""
        with self._lock:
            self._update_locked()

    def _update_locked(self) -> None:
        size = self.path.stat().st_size

        # Shrinking means the capture was truncated or rotated; the retained
        # offset and timestamp chain no longer describe this file.
        if size < self._offset:
            self._reset()

        if size == self._offset:
            return

        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            buffer = handle.read()

        pos = 0
        consumed = 0
        limit = len(buffer)

        while pos + HEADER_BYTES <= limit:
            header = self._reader.parseHeader(buffer[pos : pos + HEADER_BYTES])
            end = pos + HEADER_BYTES + header["csi_length"]
            if end > limit:
                break  # frame still being written; retry next poll

            payload = buffer[pos + HEADER_BYTES : end]
            num_sc = header["num_subcarriers"]

            # A subcarrier-count change (bandwidth switch mid-capture) makes
            # earlier rows unstackable. Upstream drops the odd frames; here the
            # newer geometry wins and the buffer restarts.
            if self._num_subcarriers and num_sc != self._num_subcarriers:
                self._amplitude.clear()
                self._phase.clear()
                self._times.clear()
                self._bandwidth = None

            amplitude, phase = self._decode_frame(header, payload)
            timestamp = self._next_timestamp(header)

            self._amplitude.append(amplitude)
            self._phase.append(phase)
            self._times.append(timestamp)
            self._num_subcarriers = num_sc
            if self._bandwidth is None:
                self._bandwidth = header["channel_width"]
            self._prev_ftm = header["ftm_clock"]
            self._prev_mu = header["mu_clock"]
            self._prev_time = timestamp
            self._total_frames += 1

            pos = end
            consumed = end

        self._offset += consumed

    def snapshot(self, *, max_packets: int = 200) -> FeitCSICapture:
        """Trailing window of decoded frames, newest last."""
        with self._lock:
            count = len(self._times)
            if count == 0:
                empty = np.empty((0, 0), dtype=float)
                return FeitCSICapture(
                    amplitude=empty,
                    phase=empty,
                    time_seconds=np.empty(0, dtype=float),
                    bandwidth=self._bandwidth or "unknown",
                    chipset="Intel AX2xx",
                    filename=self.path.name,
                    num_subcarriers=0,
                )

            take = count if max_packets <= 0 else min(max_packets, count)
            start = count - take
            amplitude = np.stack(list(self._amplitude)[start:])
            phase = np.stack(list(self._phase)[start:])
            times = np.fromiter(
                list(self._times)[start:], dtype=float, count=take
            )

            return FeitCSICapture(
                amplitude=amplitude,
                phase=phase,
                time_seconds=times,
                bandwidth=self._bandwidth or "unknown",
                chipset="Intel AX2xx",
                filename=self.path.name,
                num_subcarriers=self._num_subcarriers,
            )

    @property
    def total_frames(self) -> int:
        """Frames decoded since the stream was opened or last reset."""
        return self._total_frames


_streams: dict[Path, CaptureStream] = {}
_registry_lock = threading.Lock()


def get_stream(path: str | Path) -> CaptureStream:
    """Return the shared stream for ``path``, creating it on first use.

    Streams are cached so successive polls decode only appended bytes. Keyed by
    resolved path so ``captures/x.dat`` and an absolute path to the same file
    share one buffer.
    """
    resolved = Path(path).resolve()
    with _registry_lock:
        stream = _streams.get(resolved)
        if stream is None:
            stream = CaptureStream(resolved)
            _streams[resolved] = stream
        return stream


def reset_streams() -> None:
    """Drop all cached streams. For tests and manual recovery."""
    with _registry_lock:
        _streams.clear()
