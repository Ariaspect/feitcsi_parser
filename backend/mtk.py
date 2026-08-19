"""MediaTek CSI capture parser — TLV records grouped into frames.

The MediaTek driver (LG webOS board, ``/proc/net/wlan/csi_data``) emits a
stream of self-delimiting records::

    magic 0xAC | length u16 LE (body only) | tag(1) len(2 LE) value ...

Nothing about it resembles the FeitCSI layout, so this module mirrors
``index.FrameIndex`` and ``batch.decode_frames`` rather than reusing them.
Four differences drive the whole file:

*Records are not frames.* Tag 18 packs ``(group << 16) | (last << 15) | idx``;
one measurement is a group of up to four records, and bit 15 marks the last of
them. Grouping by *timestamp* looks tempting and is wrong — the millisecond
clock ticks mid-group, splitting real groups and merging unrelated ones.

*A group is a 2x1, not a 2x2.* Records carry ``tpi``/``rpi`` (cell index is
``2*rpi + tpi``), but the board is physically 1x1 — one antenna for 2.4 GHz,
one for 5 GHz — so ``rpi`` cannot be a second RF chain. Only ``tpi`` behaves
like a real antenna pair. Measured over 36 complete groups::

    tpi1/tpi0   temporal coherence 0.970 / 0.987, median step 0.08-0.10 rad
    rpi1/rpi0   temporal coherence 0.108 / 0.099, median step 1.64-1.70 rad
    raw cell    temporal coherence 0.170 / 0.132, median step 1.57-1.63 rad

Dividing along ``tpi`` — two AP transmit antennas seen through one receiver —
cancels the receiver's CFO/SFO. Dividing along ``rpi`` cancels nothing; it is
no better than taking no ratio at all, which is itself evidence that ``rpi``
is not a second chain. So **tpi is mapped onto the rx axis** and ``rpi`` plane
0 is used, because ``batch._decode_chunk`` and ``ratio`` both read the ratio
off ``rx1/rx0``. Our "rx" therefore holds transmit antennas. The alternative
was renaming the ratio axis through two tested modules for a naming gain.

*Samples are 14-bit signed, not int16*, and tag 8 / tag 9 are the real and
imaginary halves of one stream — not two chains.

*Subcarriers arrive in raw FFT bin order and need ``fftshift``* — the exact
opposite of FeitCSI, whose parser docstring warns against shifting. The
hardware reports pilots and guard bands as exact zeros, which is how nulls are
found here: no rate word exists, so ``batch._PILOT_INDICES`` can never apply.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .index import _mimo_label

MAGIC = 0xAC
PREFIX_BYTES = 3  # magic(1) + length u16 LE; the length counts the body only

TAG_VERSION = 0
TAG_TIMESTAMP = 2
TAG_RSSI = 3
TAG_SNR = 4
TAG_BANDWIDTH = 5
TAG_SOURCE_MAC = 7
TAG_CSI_REAL = 8
TAG_CSI_IMAG = 9
TAG_TPI = 15
TAG_RPI = 16
TAG_SEQUENCE = 18

# Bandwidth code -> FFT size, and the label FrameIndex uses for channel_width.
_SUBCARRIERS = {0: 64, 1: 128, 2: 256}
_CHANNEL_WIDTH = {0: "20", 1: "40", 2: "80"}

# Structural nulls up to this long are interpolated (pilots are 1 bin, the DC
# null is 3). Longer runs are the guard band and stay NaN — there is nothing
# on the far side to interpolate from.
MAX_NULL_RUN = 3

# Which rpi plane to read. Both ratio cleanly, but emitting two planes would
# put two frames on the same timestamp, and the tile column mapping resolves
# columns by time.
RPI_PLANE = 0

_MAX_SLOTS = 2  # tpi in {0, 1}


def _sign_extend_14(raw: np.ndarray) -> np.ndarray:
    """14-bit two's complement -> signed. Masking as int16 is wrong."""
    v = raw.astype(np.int32) & 0x3FFF
    return np.where(v & 0x2000, v - 0x4000, v)


def _walk_records(view: memoryview, size: int):
    """Yield ``(offset, end, tags)`` per record; tags map tag -> (off, len).

    Stops cleanly at a truncated trailing record so a file still being written
    indexes up to its last complete record.
    """
    pos = 0
    while pos + PREFIX_BYTES <= size:
        if view[pos] != MAGIC:
            return  # desynchronised; refuse to guess
        (body_len,) = struct.unpack_from("<H", view, pos + 1)
        end = pos + PREFIX_BYTES + body_len
        if end > size:
            return  # partial record, still being written
        tags: dict[int, tuple[int, int]] = {}
        i = pos + PREFIX_BYTES
        while i + 3 <= end:
            tag = view[i]
            (length,) = struct.unpack_from("<H", view, i + 1)
            if i + 3 + length > end:
                break
            tags[tag] = (i + 3, length)
            i += 3 + length
        yield pos, end, tags
        pos = end


def _u(view: memoryview, span: tuple[int, int] | None) -> int:
    """Little-endian unsigned int from a (offset, length) span; 0 if absent."""
    if span is None:
        return 0
    off, length = span
    return int.from_bytes(bytes(view[off : off + length]), "little")


def can_read(path: str | Path) -> bool:
    """True if *path* looks like an MTK capture: magic plus one clean record."""
    path = Path(path)
    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    if len(head) < PREFIX_BYTES or head[0] != MAGIC:
        return False
    (body_len,) = struct.unpack_from("<H", head, 1)
    if body_len < 3:
        return False
    # The first TLV of every observed record is tag 0 (version), length 1.
    end = min(PREFIX_BYTES + body_len, len(head))
    return head[PREFIX_BYTES] == TAG_VERSION and PREFIX_BYTES + 3 <= end


class MTKIndex:
    """Structural index of an MTK capture, shaped like ``index.FrameIndex``.

    Exposes the same attribute surface the tile pipeline consumes, so
    ``tiles``/``ratio`` need no knowledge of the format. ``stride`` is always
    ``None``: record length varies with bandwidth within a single file, so
    there is no uniform memmap fast path to take.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._scan_end: int = 0
        self._scan_full()

    # ------------------------------------------------------------------ #
    #  Scan                                                               #
    # ------------------------------------------------------------------ #

    def _scan_full(self) -> None:
        size = self.path.stat().st_size
        if size < PREFIX_BYTES:
            self._init_empty()
            return
        with self.path.open("rb") as fh:
            buf = fh.read()
        self._build(memoryview(buf), min(size, len(buf)), base=0, existing=None)

    def _build(self, view: memoryview, size: int, *, base: int, existing) -> int:
        """Walk records, assemble complete groups, and publish frame arrays.

        ``base`` offsets record positions into the whole file (non-zero when
        extending). Returns the number of frames appended.
        """
        groups: list[list[tuple[int, int, dict[int, tuple[int, int]]]]] = []
        current: list[tuple[int, int, dict[int, tuple[int, int]]]] = []
        current_id: int | None = None
        scan_end = 0

        for off, end, tags in _walk_records(view, size):
            seq = _u(view, tags.get(TAG_SEQUENCE))
            group_id, low = seq >> 16, seq & 0xFFFF
            if current_id is not None and group_id != current_id:
                current = []  # previous group never closed; drop it
            current_id = group_id
            current.append((off, end, tags))
            if low & 0x8000:  # last record of the group
                groups.append(current)
                scan_end = end
                current, current_id = [], None

        if not groups:
            if existing is None:
                self._init_empty()
            return 0

        n = len(groups)
        offsets = np.empty(n, dtype=np.int64)
        stamps = np.empty(n, dtype=np.int64)
        rssi = np.empty(n, dtype=np.int64)
        csi_lengths = np.empty(n, dtype=np.int64)
        bins = np.empty(n, dtype=np.int64)
        num_rx = np.empty(n, dtype=np.int64)
        real_off = np.full((n, _MAX_SLOTS), -1, dtype=np.int64)
        imag_off = np.full((n, _MAX_SLOTS), -1, dtype=np.int64)
        macs: list[str] = []
        widths: list[str] = []

        for gi, records in enumerate(groups):
            # Plane RPI_PLANE only; tpi ascending fills the rx slots.
            plane = [r for r in records if _u(view, r[2].get(TAG_RPI)) == RPI_PLANE]
            plane.sort(key=lambda r: _u(view, r[2].get(TAG_TPI)))
            if not plane:
                plane = records[:1]  # keep the frame, mark it single-stream

            head = plane[0][2]
            code = _u(view, head.get(TAG_BANDWIDTH))
            nbins = _SUBCARRIERS.get(code, 0)

            total = 0
            slots = 0
            for slot, (_, _, tags) in enumerate(plane[:_MAX_SLOTS]):
                r_span, i_span = tags.get(TAG_CSI_REAL), tags.get(TAG_CSI_IMAG)
                if r_span is None or i_span is None:
                    continue
                if r_span[1] != nbins * 2 or i_span[1] != nbins * 2:
                    continue  # payload disagrees with the bandwidth code
                real_off[gi, slot] = base + r_span[0]
                imag_off[gi, slot] = base + i_span[0]
                total += r_span[1] + i_span[1]
                slots += 1

            offsets[gi] = base + plane[0][0]
            stamps[gi] = _u(view, head.get(TAG_TIMESTAMP))
            rssi[gi] = _u(view, head.get(TAG_RSSI))
            csi_lengths[gi] = total
            bins[gi] = nbins
            num_rx[gi] = slots
            mac_span = head.get(TAG_SOURCE_MAC)
            macs.append(
                ":".join(f"{b:02x}" for b in bytes(view[mac_span[0] : mac_span[0] + 6]))
                if mac_span
                else "00:00:00:00:00:00"
            )
            widths.append(_CHANNEL_WIDTH.get(code, "unknown"))

        if existing is None:
            self.offsets = offsets
            self._stamps = stamps
            self.rssi_1 = rssi
            self.csi_lengths = csi_lengths
            self._bins = bins
            self.num_rx_arr = num_rx
            self.num_tx_arr = np.ones(n, dtype=np.int64)
            self._real_off = real_off
            self._imag_off = imag_off
            self.source_macs = macs
            self.channel_widths = widths
        else:
            self.offsets = np.concatenate([self.offsets, offsets])
            self._stamps = np.concatenate([self._stamps, stamps])
            self.rssi_1 = np.concatenate([self.rssi_1, rssi])
            self.csi_lengths = np.concatenate([self.csi_lengths, csi_lengths])
            self._bins = np.concatenate([self._bins, bins])
            self.num_rx_arr = np.concatenate([self.num_rx_arr, num_rx])
            self.num_tx_arr = np.concatenate([self.num_tx_arr, np.ones(n, dtype=np.int64)])
            self._real_off = np.concatenate([self._real_off, real_off])
            self._imag_off = np.concatenate([self._imag_off, imag_off])
            self.source_macs = self.source_macs + macs
            self.channel_widths = self.channel_widths + widths

        self.count = len(self.offsets)
        # tag 2 is an absolute millisecond clock; FrameIndex starts at 0.0.
        self.times = (self._stamps - self._stamps[0]).astype(np.float64) / 1e3
        self.num_subcarriers = int(self._bins.max())
        self.num_rx = int(self.num_rx_arr[0])
        self.num_tx = 1
        self.bandwidth = self.channel_widths[0]
        # No rate word exists — tag 19 (rx_rate) reads 0 on every record.
        self.rate_formats = ["unknown"] * self.count
        self.rate_flags = np.zeros(self.count, dtype=np.int64)
        self.ftm_clocks = np.zeros(self.count, dtype=np.int64)
        self.mu_clocks = np.zeros(self.count, dtype=np.int64)
        self.stride = None
        self._uniform = False
        self._scan_end = base + scan_end
        return n

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
        self._stamps = np.zeros(0, dtype=np.int64)
        self._bins = np.zeros(0, dtype=np.int64)
        self._real_off = np.zeros((0, _MAX_SLOTS), dtype=np.int64)
        self._imag_off = np.zeros((0, _MAX_SLOTS), dtype=np.int64)
        self.source_macs = []
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
    #  Shared surface                                                     #
    # ------------------------------------------------------------------ #

    def mimo_labels(self) -> list[str]:
        return [
            _mimo_label(int(rx), int(tx))
            for rx, tx in zip(self.num_rx_arr, self.num_tx_arr)
        ]

    def filter_mask(
        self,
        mimo: tuple[int, int] | None = None,
        source_mac: str | None = None,
    ) -> np.ndarray:
        n = self.count
        mask = np.ones(n, dtype=bool)
        if mimo is not None and n > 0:
            rx, tx = mimo
            mask &= (self.num_rx_arr == rx) & (self.num_tx_arr == tx)
        if source_mac and n > 0:
            macs = np.array(self.source_macs, dtype=object)
            mask &= macs == source_mac
        return mask

    def extend(self) -> int:
        """Index groups completed since the last scan.

        Bit 15 of tag 18 closes a group, so a partially-arrived group is simply
        never emitted — no timestamp holdback needed.
        """
        size = self.path.stat().st_size
        if size < self._scan_end:
            self._scan_full()
            return self.count
        if self.count == 0:
            self._scan_full()
            return self.count
        if size == self._scan_end:
            return 0
        with self.path.open("rb") as fh:
            fh.seek(self._scan_end)
            buf = fh.read()
        return self._build(
            memoryview(buf), len(buf), base=self._scan_end, existing=True
        )


# ---------------------------------------------------------------------- #
#  Decode                                                                 #
# ---------------------------------------------------------------------- #


def _null_runs(mask: np.ndarray):
    """Yield ``(start, stop)`` for each run of True in a 1-D boolean mask."""
    if not mask.any():
        return
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(edges[0::2], edges[1::2]):
        yield int(start), int(stop)


def _fill_nulls(z: np.ndarray) -> np.ndarray:
    """Interpolate structural nulls; leave the guard band NaN.

    Nulls are found from the data rather than a rate table: the hardware
    reports pilots, DC and guard as exact zeros, and a bin null in *every*
    frame is structural. Incidental zeros in a single frame are left alone —
    they are measurements.
    """
    out = z.astype(np.complex128, copy=True)
    structural = np.all(out == 0, axis=0)
    for start, stop in _null_runs(structural):
        left, right = start - 1, stop
        if stop - start > MAX_NULL_RUN or left < 0 or right >= out.shape[1]:
            out[:, start:stop] = np.nan  # guard band: nothing to span
            continue
        span = stop - start + 1
        for k in range(start, stop):
            mu = (k - left) / span
            out[:, k] = out[:, left] * (1 - mu) + out[:, right] * mu
    return out


def _read_plane(
    path: Path, offsets: np.ndarray, nbins: int
) -> np.ndarray:
    """Gather ``len(offsets)`` runs of ``nbins`` uint16 from scattered offsets."""
    mm = np.memmap(path, dtype="<u1", mode="r")
    idx = offsets[:, None] + np.arange(nbins * 2, dtype=np.int64)[None, :]
    raw = np.asarray(mm[idx])  # fancy indexing copies
    del mm
    return _sign_extend_14(raw.view("<u2")).astype(np.float64)


def decode_frames(
    path: str | Path,
    index: MTKIndex,
    frame_ids: np.ndarray,
    *,
    scaled: bool = False,
    interpolate: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode selected frames into (amplitude, phase, ratio_amp, ratio_phase).

    Matches ``batch.decode_frames``: float32, shape ``(len(frame_ids),
    num_subcarriers)``. Frames narrower than the file's widest are centred and
    NaN-padded — both bandwidths are centred on DC, so centring is the honest
    placement.

    ``scaled`` defaults to False, unlike the FeitCSI path. Tag 3 is a single
    byte whose sign is unresolved; read unsigned it would offset ``amplitude``
    by ~256 dB. Scaling touches only ``amplitude`` — both ratio metrics and
    both phases are provably independent of it — so leaving it off costs an
    absolute dB reference and nothing else.
    """
    path = Path(path)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    n = len(frame_ids)
    width = index.num_subcarriers

    empty = np.empty((0, width), dtype=np.float32)
    if n == 0:
        return empty, empty, empty, empty

    amp = np.full((n, width), np.nan, dtype=np.float32)
    phase = np.full((n, width), np.nan, dtype=np.float32)
    ratio_amp = np.full((n, width), np.nan, dtype=np.float32)
    ratio_phase = np.full((n, width), np.nan, dtype=np.float32)

    bins = index._bins[frame_ids]
    for nbins in np.unique(bins):
        nbins = int(nbins)
        if nbins == 0:
            continue
        sel = np.flatnonzero(bins == nbins)
        ids = frame_ids[sel]
        lo = (width - nbins) // 2  # centre the band in the output row

        streams: list[np.ndarray | None] = []
        for slot in range(_MAX_SLOTS):
            r_off = index._real_off[ids, slot]
            i_off = index._imag_off[ids, slot]
            present = (r_off >= 0) & (i_off >= 0)
            if not present.any():
                streams.append(None)
                continue
            z = np.full((len(ids), nbins), np.nan, dtype=np.complex128)
            rows = np.flatnonzero(present)
            real = _read_plane(path, r_off[rows], nbins)
            imag = _read_plane(path, i_off[rows], nbins)
            # Raw FFT bin order -> centred, the layout every metric assumes.
            z[rows] = np.fft.fftshift(real + 1j * imag, axes=1)
            if interpolate:
                z[rows] = _fill_nulls(z[rows])
            streams.append(z)

        rx0 = streams[0]
        if rx0 is None:
            continue
        if scaled:
            rx0 = _scale(rx0, index.rssi_1[ids], nbins)

        with np.errstate(divide="ignore", invalid="ignore"):
            amp[sel, lo : lo + nbins] = _db(np.abs(rx0))
            phase[sel, lo : lo + nbins] = np.angle(rx0)

            rx1 = streams[1]
            if rx1 is not None:
                if scaled:
                    rx1 = _scale(rx1, index.rssi_1[ids], nbins)
                ratio = rx1 / rx0
                ratio_amp[sel, lo : lo + nbins] = _db(np.abs(ratio))
                ratio_phase[sel, lo : lo + nbins] = np.angle(ratio)

    return amp, phase, ratio_amp, ratio_phase


def _db(x: np.ndarray) -> np.ndarray:
    """20*log10, matching CSIKit's voltage-metric db used by the FeitCSI path."""
    return 20 * np.log10(x)


def _scale(z: np.ndarray, rssi: np.ndarray, nbins: int) -> np.ndarray:
    """RSSI scaling, mirroring ``batch._batch_scale`` for a single stream."""
    rss_pwr = np.power(10, rssi.astype(np.float64) / 10)
    mag = np.nansum(np.abs(z) ** 2, axis=1) / nbins
    return z * np.sqrt(rss_pwr / mag)[:, None]
