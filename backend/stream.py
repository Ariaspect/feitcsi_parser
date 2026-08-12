"""Incremental FeitCSI .dat reader.

CSIKit's ``read_file`` re-reads and re-decodes the entire capture on every call,
so a polling UI pays O(file) per refresh and falls behind as the capture grows.
FeitCSI frames are self-delimiting -- a 272-byte header whose first word is the
payload length -- so bytes appended since the last poll can be decoded on their
own. ``CaptureStream`` keeps a ``FrameIndex`` for the structural scan and a
rolling window of decoded frames, touching only what is new.

Decoding is kept compatible with CSIKit:

* header fields come from ``FeitCSIBeamformReader.parseHeader`` (via FrameIndex)
* pilot interpolation and RSSI scaling are vectorised across frames in
  ``backend.batch.decode_frames``, reproducing the same upstream functions
* the timestamp chain reproduces ``read_file``'s ftm_clock/mu_clock arithmetic,
  vectorised via ``FrameIndex._compute_timestamps``
* amplitude is ``db(abs(csi))`` and phase ``angle(csi)``, matching
  ``csitools.get_CSI`` with ``extract_as_dBm=True``

The output dtype is float32 (halving memory versus float64) because the batch
decode casts to float32 before returning.

``filter_mac`` is not supported here. Upstream compares ``filter_mac.casefold()``
against ``FeitCSIFrame.source_mac``, which is a tuple of six ints rather than a
string, so the filter raises before it can match any FeitCSI frame.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import numpy as np

from .batch import decode_frames
from .index import FrameIndex
from .parser import FeitCSICapture

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

        self._lock = threading.Lock()
        self._index = FrameIndex(self.path)
        self._amplitude: deque[np.ndarray] = deque(maxlen=max_frames)
        self._phase: deque[np.ndarray] = deque(maxlen=max_frames)
        self._times: deque[float] = deque(maxlen=max_frames)
        self._decoded_count = 0
        self._num_subcarriers = self._index.num_subcarriers
        self._bandwidth: str | None = (
            self._index.bandwidth if self._index.count > 0 else None
        )

    def update(self) -> None:
        """Decode any frames appended since the last call."""
        with self._lock:
            self._update_locked()

    def _update_locked(self) -> None:
        old_decoded = self._decoded_count

        # Extend the index (handles truncation via rebuild and new frames).
        self._index.extend()

        # Truncation: index was rebuilt and may now have fewer frames.
        if self._index.count < old_decoded:
            self._amplitude.clear()
            self._phase.clear()
            self._times.clear()
            self._bandwidth = None
            self._decoded_count = 0

        start = self._decoded_count
        end = self._index.count

        # Geometry change: new frames have a different csi_length. Clear the
        # buffer so the newer geometry wins, matching the old per-frame behavior.
        if end > start and start > 0:
            old_cl = int(self._index.csi_lengths[start - 1])
            new_cl = int(self._index.csi_lengths[start])
            if old_cl != new_cl:
                self._amplitude.clear()
                self._phase.clear()
                self._times.clear()
                self._bandwidth = None
                self._decoded_count = 0
                start = 0
                end = self._index.count

        if end <= start:
            return

        # Only the trailing max_frames survive the ring buffer, so decoding
        # anything older is work whose result is discarded on arrival. Opening a
        # 211 MB capture would otherwise decode all 95,787 frames -- 3.2 s and
        # 540 MB peak -- to keep the last 10,000. Timestamps are unaffected:
        # FrameIndex derives the whole chain from headers alone, so skipping
        # decode does not break the cumulative arithmetic.
        if end - start > self.max_frames:
            start = end - self.max_frames

        frame_ids = np.arange(start, end)
        amp, phase = decode_frames(
            self.path,
            self._index,
            frame_ids,
            scaled=self.scaled,
            interpolate=self.interpolate,
        )
        times = self._index.times[start:end]
        for i in range(len(frame_ids)):
            self._amplitude.append(amp[i])
            self._phase.append(phase[i])
            self._times.append(float(times[i]))

        self._decoded_count = end
        self._num_subcarriers = self._index.num_subcarriers
        if self._bandwidth is None:
            self._bandwidth = self._index.bandwidth

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
        """Frames present in the capture, whether or not they were decoded.

        Read from the index rather than counted while decoding, because the
        decoder now skips frames that fall outside the retained window. The UI
        reports this as the capture's packet total, which is a property of the
        file and not of how much of it was worth decoding.
        """
        return self._index.count


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
