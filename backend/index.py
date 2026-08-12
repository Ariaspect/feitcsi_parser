"""Structural scan of a FeitCSI .dat file — offsets and timestamps without decoding payloads.

Each frame is a 272-byte header followed by a ``csi_length``-byte payload. Real
captures have a uniform frame stride (bandwidth and antenna count do not change
mid-file), so the header scan and the timestamp chain both vectorise across all
frames at once via ``np.memmap``. A sequential fallback handles files with
variable-length frames or a trailing partial frame.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from CSIKit.reader import FeitCSIBeamformReader

HEADER_BYTES = 272

# Tick counter constants, mirrored from CSIKit.reader.readers.read_feitcsi.
MAX_TICK = 4294967295
TICK_RESOLUTION = 3.125
# ftm_clock wraps after ~13.4 s; past that gap upstream falls back to mu_clock.
FTM_WRAP_SECONDS = (MAX_TICK * TICK_RESOLUTION) / 1e9

# Rate-flag bit masks, mirrored from read_feitcsi.parseHeader.
RATE_MCS_MOD_TYPE_POS = 8
RATE_MCS_MOD_TYPE_MSK = 0x7 << RATE_MCS_MOD_TYPE_POS
RATE_MCS_CHAN_WIDTH_POS = 11
RATE_MCS_CHAN_WIDTH_MSK = 0x7 << RATE_MCS_CHAN_WIDTH_POS

_RATE_FORMATS = {
    0 << RATE_MCS_MOD_TYPE_POS: "CCK",
    1 << RATE_MCS_MOD_TYPE_POS: "LEGACY_OFDM",
    2 << RATE_MCS_MOD_TYPE_POS: "HT",
    3 << RATE_MCS_MOD_TYPE_POS: "VHT",
    4 << RATE_MCS_MOD_TYPE_POS: "HE",
    5 << RATE_MCS_MOD_TYPE_POS: "EHT",
}

_CHANNEL_WIDTHS = {
    0 << RATE_MCS_CHAN_WIDTH_POS: "20",
    1 << RATE_MCS_CHAN_WIDTH_POS: "40",
    2 << RATE_MCS_CHAN_WIDTH_POS: "80",
    3 << RATE_MCS_CHAN_WIDTH_POS: "160",
    4 << RATE_MCS_CHAN_WIDTH_POS: "320",
}


def _decode_rate_flags(rate_flags: np.ndarray) -> tuple[list[str], list[str]]:
    """Decode (rate_format, channel_width) per frame from raw rate_flags."""
    rf_raw = (rate_flags & RATE_MCS_MOD_TYPE_MSK).tolist()
    cw_raw = (rate_flags & RATE_MCS_CHAN_WIDTH_MSK).tolist()
    rate_formats = [_RATE_FORMATS.get(v, "unknown") for v in rf_raw]
    channel_widths = [_CHANNEL_WIDTHS.get(v, "unknown") for v in cw_raw]
    return rate_formats, channel_widths


def _compute_timestamps(ftm: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Vectorised cumulative timestamp chain reproducing ``read_file``.

    The delta for frame *i* depends only on frames *i* and *i-1*. The first
    frame's timestamp is 0.0. Both the ftm_clock overflow branch and the
    mu_clock fallback (when the gap exceeds the ftm wrap period) are reproduced
    exactly.
    """
    n = len(ftm)
    times = np.zeros(n, dtype=np.float64)
    if n < 2:
        return times

    ftm = ftm.astype(np.int64)
    mu = mu.astype(np.int64)

    ftm_prev, ftm_curr = ftm[:-1], ftm[1:]
    mu_prev, mu_curr = mu[:-1], mu[1:]

    # Upstream condition: (mu - prev_mu) / 1e6 < FTM_WRAP_SECONDS.
    # Python ints (upstream) never overflow, so int64 matches exactly.
    mu_gap = (mu_curr - mu_prev) / 1e6
    use_ftm = mu_gap < FTM_WRAP_SECONDS

    ftm_diff = np.where(
        ftm_curr > ftm_prev,
        ftm_curr - ftm_prev,
        ftm_curr + (MAX_TICK - ftm_prev),
    )
    ftm_delta = (ftm_diff * TICK_RESOLUTION) / 1e9

    mu_diff = np.where(
        mu_curr > mu_prev,
        mu_curr - mu_prev,
        mu_curr + (MAX_TICK - mu_prev),
    )
    mu_delta = mu_diff / 1e6

    deltas = np.where(use_ftm, ftm_delta, mu_delta)
    times[1:] = np.cumsum(deltas)
    return times


class FrameIndex:
    """Structural index of a FeitCSI capture: offsets, timestamps, and metadata.

    Never decodes CSI payloads. The uniform-stride fast path memory-maps the
    file and slices header columns for all frames in one shot. The sequential
    fallback walks frame by frame reading only 272-byte headers, seeking past
    payloads, and handles variable-length frames.

    Call ``extend()`` to index frames appended since the last scan without
    rescanning from zero. Truncation (file shrank) triggers a full rebuild.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._scan_end: int = 0  # byte offset past the last complete frame
        self._scan_full()

    # ------------------------------------------------------------------ #
    #  Full scan                                                          #
    # ------------------------------------------------------------------ #

    def _scan_full(self) -> None:
        size = self.path.stat().st_size
        if size < HEADER_BYTES:
            self._init_empty()
            return

        with self.path.open("rb") as fh:
            first_header = fh.read(HEADER_BYTES)
        if len(first_header) < HEADER_BYTES:
            self._init_empty()
            return

        csi_length_0 = struct.unpack("I", first_header[0:4])[0]
        stride_0 = HEADER_BYTES + csi_length_0

        if csi_length_0 > 0 and size % stride_0 == 0:
            self._scan_uniform(size, csi_length_0, stride_0)
        else:
            self._scan_sequential(size)

    def _scan_uniform(self, size: int, csi_length_0: int, stride: int) -> None:
        count = size // stride
        mm = np.memmap(self.path, dtype="<u1", mode="r").reshape(count, stride)

        def u32(col: int) -> np.ndarray:
            return mm[:, col : col + 4].copy().view("<u4").ravel().astype(np.int64)

        def u8(col: int) -> np.ndarray:
            return mm[:, col].copy().astype(np.int64)

        csi_lengths = u32(0)
        # Verify uniform stride; fall back if any frame differs.
        if not np.all(csi_lengths == csi_length_0):
            mm._mmap.close()
            self._scan_sequential(size)
            return

        ftm = u32(8)
        num_rx_arr = u8(46)
        num_tx_arr = u8(47)
        num_sc_arr = u32(52)
        rssi_1 = u32(60)
        mu = u32(88)
        rate_flags = u32(92)

        mm._mmap.close()

        # Verify geometry is constant; fall back if num_rx/num_tx vary. The
        # fast path broadcasts the first frame's scalars to all frames, so a
        # file with mixed 2x1/2x2 streams would be mislabelled.
        if not (np.all(num_rx_arr == num_rx_arr[0]) and np.all(num_tx_arr == num_tx_arr[0])):
            self._scan_sequential(size)
            return

        offsets = np.arange(count, dtype=np.int64) * stride
        times = _compute_timestamps(ftm, mu)

        rate_formats, channel_widths = _decode_rate_flags(rate_flags)

        self.offsets = offsets
        self.times = times
        self.csi_lengths = csi_lengths
        self.ftm_clocks = ftm
        self.mu_clocks = mu
        self.rssi_1 = rssi_1
        self.rate_flags = rate_flags
        self.rate_formats = rate_formats
        self.channel_widths = channel_widths
        self.count = count
        self.num_subcarriers = int(num_sc_arr[0])
        self.num_rx = int(num_rx_arr[0])
        self.num_tx = int(num_tx_arr[0])
        self.num_rx_arr = num_rx_arr
        self.num_tx_arr = num_tx_arr
        self.bandwidth = channel_widths[0]
        self.stride = stride
        self._uniform = True
        self._scan_end = count * stride

    def _scan_sequential(self, size: int) -> None:
        reader = FeitCSIBeamformReader()
        offsets: list[int] = []
        ftm_list: list[int] = []
        mu_list: list[int] = []
        rssi_list: list[int] = []
        rate_flags_list: list[int] = []
        csi_lengths: list[int] = []
        num_sc_list: list[int] = []
        num_rx_list: list[int] = []
        num_tx_list: list[int] = []

        with self.path.open("rb") as fh:
            pos = 0
            while pos + HEADER_BYTES <= size:
                fh.seek(pos)
                hb = fh.read(HEADER_BYTES)
                if len(hb) < HEADER_BYTES:
                    break
                h = reader.parseHeader(hb)
                # Withhold partial frames (payload still being written).
                frame_end = pos + HEADER_BYTES + h["csi_length"]
                if frame_end > size:
                    break
                offsets.append(pos)
                ftm_list.append(h["ftm_clock"])
                mu_list.append(h["mu_clock"])
                rssi_list.append(h["rssi_1"])
                rate_flags_list.append(h["rate_flags"])
                csi_lengths.append(h["csi_length"])
                num_sc_list.append(h["num_subcarriers"])
                num_rx_list.append(h["num_rx"])
                num_tx_list.append(h["num_tx"])
                pos = frame_end

        count = len(offsets)
        if count == 0:
            self._init_empty()
            return

        self.offsets = np.array(offsets, dtype=np.int64)
        self.ftm_clocks = np.array(ftm_list, dtype=np.int64)
        self.mu_clocks = np.array(mu_list, dtype=np.int64)
        self.rssi_1 = np.array(rssi_list, dtype=np.int64)
        self.rate_flags = np.array(rate_flags_list, dtype=np.int64)
        self.csi_lengths = np.array(csi_lengths, dtype=np.int64)
        self.num_rx_arr = np.array(num_rx_list, dtype=np.int64)
        self.num_tx_arr = np.array(num_tx_list, dtype=np.int64)
        self.times = _compute_timestamps(self.ftm_clocks, self.mu_clocks)

        rate_formats, channel_widths = _decode_rate_flags(self.rate_flags)
        self.rate_formats = rate_formats
        self.channel_widths = channel_widths

        self.count = count
        self.num_subcarriers = int(num_sc_list[0])
        self.num_rx = int(num_rx_list[0])
        self.num_tx = int(num_tx_list[0])
        self.bandwidth = channel_widths[0]
        # Stride is set only if every frame has the same csi_length.
        first_cl = csi_lengths[0]
        if all(cl == first_cl for cl in csi_lengths):
            self.stride = HEADER_BYTES + first_cl
        else:
            self.stride = None
        self._uniform = False
        self._scan_end = (
            int(offsets[-1]) + HEADER_BYTES + int(csi_lengths[-1])
        )

    def _init_empty(self) -> None:
        self.offsets = np.zeros(0, dtype=np.int64)
        self.times = np.zeros(0, dtype=np.float64)
        self.csi_lengths = np.zeros(0, dtype=np.int64)
        self.ftm_clocks = np.zeros(0, dtype=np.int64)
        self.mu_clocks = np.zeros(0, dtype=np.int64)
        self.rssi_1 = np.zeros(0, dtype=np.int64)
        self.rate_flags = np.zeros(0, dtype=np.int64)
        self.num_rx_arr = np.zeros(0, dtype=np.int64)
        self.num_tx_arr = np.zeros(0, dtype=np.int64)
        self.rate_formats = []
        self.channel_widths = []
        self.count = 0
        self.num_subcarriers = 0
        self.num_rx = 0
        self.num_tx = 0
        self.bandwidth = "unknown"
        self.stride = None
        self._uniform = False
        self._scan_end = 0

    # ------------------------------------------------------------------ #
    #  Incremental extension                                              #
    # ------------------------------------------------------------------ #

    def extend(self) -> int:
        """Index frames appended since the last scan.

        Returns the number of newly indexed frames. If the file shrank
        (truncation/rotation), the index is rebuilt from scratch; the return
        value is the number of frames in the rebuilt index (may be less than
        before, zero, or — if the file regrew — more).
        """
        size = self.path.stat().st_size

        # Truncation: file shrank below the end of the last scanned frame.
        if size < self._scan_end:
            old_count = self.count
            self._scan_full()
            return self.count  # caller compares with old decoded_count

        if self.count == 0:
            # Initial scan was empty; file may have grown.
            self._scan_full()
            return self.count

        if self._uniform and self.stride is not None:
            return self._extend_uniform(size)
        return self._extend_sequential(size)

    def _extend_uniform(self, size: int) -> int:
        old_count = self.count
        new_count = size // self.stride
        if new_count <= old_count:
            return 0

        stride = self.stride
        mm = np.memmap(self.path, dtype="<u1", mode="r", shape=(new_count, stride))

        def u32(col: int, lo: int, hi: int) -> np.ndarray:
            return mm[lo:hi, col : col + 4].copy().view("<u4").ravel().astype(np.int64)

        lo = old_count
        ftm_new = u32(8, lo, new_count)
        mu_new = u32(88, lo, new_count)
        rssi_new = u32(60, lo, new_count)
        rate_flags_new = u32(92, lo, new_count)
        csi_lengths_new = u32(0, lo, new_count)

        mm._mmap.close()

        # Verify the new frames still have the expected csi_length.
        if not np.all(csi_lengths_new == (stride - HEADER_BYTES)):
            self._scan_full()
            return self.count - old_count

        # Timestamps: prepend last old frame to chain correctly.
        ftm_seq = np.empty(new_count - lo + 1, dtype=np.int64)
        mu_seq = np.empty(new_count - lo + 1, dtype=np.int64)
        ftm_seq[0] = self.ftm_clocks[-1]
        mu_seq[0] = self.mu_clocks[-1]
        ftm_seq[1:] = ftm_new
        mu_seq[1:] = mu_new
        new_deltas = _compute_timestamps(ftm_seq, mu_seq)
        # _compute_timestamps returns cumulative timestamps starting from 0.0.
        # new_deltas[0] is 0.0 (the last old frame in the temp chain);
        # new_deltas[1:] is already the cumulative delta from that frame.
        if len(new_deltas) > 1:
            times_new = self.times[-1] + new_deltas[1:]
        else:
            times_new = np.zeros(0, dtype=np.float64)

        new_offsets = np.arange(lo, new_count, dtype=np.int64) * stride

        rf_new, cw_new = _decode_rate_flags(rate_flags_new)

        self.offsets = np.concatenate([self.offsets, new_offsets])
        self.times = np.concatenate([self.times, times_new])
        self.ftm_clocks = np.concatenate([self.ftm_clocks, ftm_new])
        self.mu_clocks = np.concatenate([self.mu_clocks, mu_new])
        self.rssi_1 = np.concatenate([self.rssi_1, rssi_new])
        self.rate_flags = np.concatenate([self.rate_flags, rate_flags_new])
        self.csi_lengths = np.concatenate([self.csi_lengths, csi_lengths_new])
        self.num_rx_arr = np.concatenate([
            self.num_rx_arr, np.full(new_count - lo, self.num_rx, dtype=np.int64)
        ])
        self.num_tx_arr = np.concatenate([
            self.num_tx_arr, np.full(new_count - lo, self.num_tx, dtype=np.int64)
        ])
        self.rate_formats.extend(rf_new)
        self.channel_widths.extend(cw_new)
        self.count = new_count
        self._scan_end = new_count * stride
        return new_count - old_count

    def _extend_sequential(self, size: int) -> int:
        old_count = self.count
        last_end = self._scan_end

        if size <= last_end:
            return 0

        reader = FeitCSIBeamformReader()
        offsets: list[int] = []
        ftm_list: list[int] = []
        mu_list: list[int] = []
        rssi_list: list[int] = []
        rate_flags_list: list[int] = []
        csi_lengths: list[int] = []
        num_sc_list: list[int] = []
        num_rx_list: list[int] = []
        num_tx_list: list[int] = []

        with self.path.open("rb") as fh:
            pos = last_end
            while pos + HEADER_BYTES <= size:
                fh.seek(pos)
                hb = fh.read(HEADER_BYTES)
                if len(hb) < HEADER_BYTES:
                    break
                h = reader.parseHeader(hb)
                frame_end = pos + HEADER_BYTES + h["csi_length"]
                if frame_end > size:
                    break  # partial frame, stop
                offsets.append(pos)
                ftm_list.append(h["ftm_clock"])
                mu_list.append(h["mu_clock"])
                rssi_list.append(h["rssi_1"])
                rate_flags_list.append(h["rate_flags"])
                csi_lengths.append(h["csi_length"])
                num_sc_list.append(h["num_subcarriers"])
                num_rx_list.append(h["num_rx"])
                num_tx_list.append(h["num_tx"])
                pos = frame_end

        if not offsets:
            return 0

        ftm_new = np.array(ftm_list, dtype=np.int64)
        mu_new = np.array(mu_list, dtype=np.int64)

        ftm_seq = np.empty(len(ftm_new) + 1, dtype=np.int64)
        mu_seq = np.empty(len(mu_new) + 1, dtype=np.int64)
        ftm_seq[0] = self.ftm_clocks[-1] if old_count else 0
        mu_seq[0] = self.mu_clocks[-1] if old_count else 0
        ftm_seq[1:] = ftm_new
        mu_seq[1:] = mu_new
        new_deltas = _compute_timestamps(ftm_seq, mu_seq)
        # _compute_timestamps returns cumulative timestamps starting from 0.0.
        # new_deltas[1:] is already the cumulative delta from the last old frame.
        if len(new_deltas) > 1 and old_count:
            times_new = self.times[-1] + new_deltas[1:]
        elif len(new_deltas) > 1:
            times_new = new_deltas[1:].copy()
        else:
            times_new = np.zeros(0, dtype=np.float64)

        rf_new, cw_new = _decode_rate_flags(np.array(rate_flags_list, dtype=np.int64))

        self.offsets = np.concatenate([self.offsets, np.array(offsets, dtype=np.int64)])
        self.times = np.concatenate([self.times, times_new])
        self.ftm_clocks = np.concatenate([self.ftm_clocks, ftm_new])
        self.mu_clocks = np.concatenate([self.mu_clocks, mu_new])
        self.rssi_1 = np.concatenate([self.rssi_1, np.array(rssi_list, dtype=np.int64)])
        self.rate_flags = np.concatenate([self.rate_flags, np.array(rate_flags_list, dtype=np.int64)])
        self.csi_lengths = np.concatenate([self.csi_lengths, np.array(csi_lengths, dtype=np.int64)])
        self.num_rx_arr = np.concatenate([self.num_rx_arr, np.array(num_rx_list, dtype=np.int64)])
        self.num_tx_arr = np.concatenate([self.num_tx_arr, np.array(num_tx_list, dtype=np.int64)])
        self.rate_formats.extend(rf_new)
        self.channel_widths.extend(cw_new)
        self.count += len(offsets)
        self._scan_end = (
            int(offsets[-1]) + HEADER_BYTES + int(csi_lengths[-1])
        )
        return len(offsets)
