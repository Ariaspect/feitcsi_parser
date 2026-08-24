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

*A group is a 2x2 grid, and the ratio is taken along ``tpi``.* Records carry
``tpi``/``rpi`` (cell index is ``2*rpi + tpi``). Which axis is which is not a
naming question — it is settled by the transmitter's cyclic shift, which the
802.11 standard applies per *transmit* chain and nothing else. Dividing along
``tpi`` yields a dead-constant +0.78 rad/subcarrier ramp; dividing along
``rpi`` yields none. So ``tpi`` indexes the AP's transmit chains and ``rpi``
indexes the board's receive side. See "the tx pair carries the transmitter's
cyclic shift" below.

Measured over the 1231 complete four-cell 80 MHz groups of ``capture1.bin``::

    tpi1/tpi0   temporal coherence 0.998 / 0.989, median |step| 0.025 / 0.047
    rpi1/rpi0   temporal coherence 0.678 / 0.678, median |step| 0.159 / 0.164
    raw cell    temporal coherence 0.003,         median |step| 1.565

An earlier revision of this docstring reported ``rpi1/rpi0`` at 0.108 / 0.099
with a median step of 1.64-1.70 rad — i.e. indistinguishable from no ratio at
all — and concluded that ``rpi`` could not be a second RF chain. That figure
does not reproduce, on the full capture or on a 36-group slice of it (0.699,
0.146). A step near 1.57 rad is what mis-pairing records *across* groups
produces, which is the very failure the tag-18 grouping above exists to
avoid, so the original measurement most likely predates that grouping.

The correction matters because the reasoning was wrong even though the
conclusion was right. ``rpi`` plane 1 is not empty: it carries 59.43 dB
against plane 0's 59.95 dB, is smooth across frequency (0.995), and its
ratio against plane 0 swings 24-31 dB across the band, so it is a genuinely
different receive path rather than a copy or a dead chain. Whether that is a
second antenna or a diversity-switched capture of one is not decidable from
the file, and the board is documented as 1x1.

``tpi`` still wins, on noise rather than on presence. Comparing how far each
ratio moves at the shortest available frame gap against how far it moves once
it has saturated separates jitter from signal::

    gap        tpi1/tpi0      rpi1/rpi0
    50 ms      0.158 rad      0.896 rad     <- rpi is 5.7x noisier per frame
    20 s       0.352 rad      1.559 rad     <- and already near the 1.571 rad
                                               limit of a random phase

Both cancel the receiver's CFO/SFO, but ``tpi0`` and ``tpi1`` come out of the
same packet, the same receive chain and the same timing recovery, so the
cancellation is exact; the two ``rpi`` planes evidently do not share phase
that tightly. So **tpi is mapped onto the rx axis** and ``rpi`` plane 0 is
used, because ``batch._decode_chunk`` and ``ratio`` both read the ratio off
``rx1/rx0``. Our "rx" therefore holds transmit antennas. The alternative was
renaming the ratio axis through two tested modules for a naming gain.

*The tx pair carries the transmitter's cyclic shift.* Because the two halves
of this ratio are two different transmit chains, the deliberate per-chain
cyclic shift the standard mandates does not cancel — it survives as a pure
linear phase ramp across the band. That is the one thing the tx-pair ratio
has that a genuine rx-pair ratio does not. Measured across every MTK capture
on hand, with no frame of any file dissenting on the sign::

    capture.bin          12 frames   +0.7811 rad/SC   397.8 ns
    capture1.bin       1231 frames   +0.7793 rad/SC   396.9 ns
    csi_ping34_30s.bin   36 frames   +0.7907 rad/SC   402.7 ns
    livetest.bin       2274 frames   +0.7787 rad/SC   396.6 ns
    tone_mask_off.bin  2158 frames   +0.7824 rad/SC   398.5 ns
    tone_mask_on.bin   2061 frames   +0.7796 rad/SC   397.1 ns

That is the -400 ns the standard specifies for the second stream, and at
80 MHz it wraps the phase about 30 times across the band, which swamps any
statistic taken along the subcarrier axis: the raw ratio phase of
``capture1.bin`` has a circular resultant of 0.010 — indistinguishable from
uniform — where removing the ramp lifts it to 0.811. ``decode_frames``
therefore removes it by default; see ``estimate_csd_slope``.

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

# Frames sampled, spread evenly across a capture, to measure its cyclic-shift
# ramp. The quantity is a hardware constant — the six captures on hand agree
# to within 6 ns of one another — so this sample is about outvoting per-frame
# noise, not about tracking anything that moves.
CSD_SAMPLE = 512

# Usable two-stream frames a capture needs before a ramp is measured at all.
# Below this the median is not a majority of anything.
CSD_MIN_FRAMES = 8

# Adjacent-subcarrier steps a frame must contribute before it gets a vote. A
# 20 MHz frame carries about 50 of them; the guard band and any single-stream
# frame carry none.
CSD_MIN_STEPS = 8

# Ramps shallower than this are left in place. 0.05 rad/SC is about 25 ns,
# under two wraps across an 80 MHz band — shallow enough that removing a
# noisy estimate of it would cost more than the ramp does. The real thing
# measures 0.78, fifteen times over the line.
CSD_MIN_SLOPE = 0.05

# Fraction of sampled frames whose slope must share the median's sign. A
# genuine cyclic shift is deterministic and scores 1.00 on every capture
# here. This is the same test that separates a tx pair from an rx pair: the
# MTK tx ratio has 0% of frames dissenting, where the FeitCSI rx ratio has
# 68% on the same statistic, because there the quantity is noise about zero.
CSD_MIN_AGREEMENT = 0.9

# "Not measured yet" — distinct from a measured "no ramp worth removing".
_CSD_UNSET = object()


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


# Records are self-delimiting, so finding them is a pointer chase. But their
# length varies only with *bandwidth*, and a capture holds very few bandwidths
# — a 1-hour 80 MHz capture measured 272 400 records of one size against 540 of
# another, in only 601 constant-size runs. So the chase can be replaced by
# predicting ``pos + k * stride`` and checking the prediction in bulk, which is
# what lets the scan below run out of a memmap instead of a 300 MB read().
_SCAN_CHUNK = 1 << 16

# Tags the vectorised scan reads. A class missing any of them falls back to the
# record-by-record walk rather than guessing at a layout.
_REQUIRED_TAGS = (
    TAG_TIMESTAMP,
    TAG_RSSI,
    TAG_BANDWIDTH,
    TAG_SOURCE_MAC,
    TAG_CSI_REAL,
    TAG_CSI_IMAG,
    TAG_TPI,
    TAG_RPI,
    TAG_SEQUENCE,
)


def _scan_record_offsets(mm: np.ndarray, start: int, end: int) -> np.ndarray:
    """Offsets of every whole record in ``[start, end)``.

    Same stopping rules as ``_walk_records``: the first byte that is not
    ``MAGIC`` ends the scan, and so does a trailing record that does not fit.
    Neither is resynchronised past — a desynchronised file is not guessed at.
    """
    chunks: list[np.ndarray] = []
    pos = start
    while pos + PREFIX_BYTES <= end:
        if mm[pos] != MAGIC:
            break
        body = int(mm[pos + 1]) | (int(mm[pos + 2]) << 8)
        stride = PREFIX_BYTES + body
        if stride <= PREFIX_BYTES or pos + stride > end:
            break
        # Extend the run of same-sized records as far as the prediction holds.
        while pos + stride <= end:
            k = min((end - pos) // stride, _SCAN_CHUNK)
            if k == 0:
                break
            cand = pos + stride * np.arange(k, dtype=np.int64)
            lens = mm[cand + 1].astype(np.int64) | (
                mm[cand + 2].astype(np.int64) << 8
            )
            good = (mm[cand] == MAGIC) & (lens == body)
            bad = np.flatnonzero(~good)
            taken = k if bad.size == 0 else int(bad[0])
            if taken == 0:
                break
            # .copy() when the run ends early: a slice keeps its whole
            # _SCAN_CHUNK-sized base alive, and with one per size-class run
            # that pinned ~300 MB on an hour-long capture.
            chunks.append(cand if taken == k else cand[:taken].copy())
            pos += stride * taken
            if taken < k:
                break  # size class changed; re-derive the stride
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(chunks)


def _class_layout(mm: np.ndarray, off: int, size: int) -> dict[int, tuple[int, int]]:
    """TLV layout of one record as ``tag -> (record-relative offset, length)``."""
    tags: dict[int, tuple[int, int]] = {}
    i = off + PREFIX_BYTES
    end = off + size
    while i + 3 <= end:
        tag = int(mm[i])
        length = int(mm[i + 1]) | (int(mm[i + 2]) << 8)
        if i + 3 + length > end:
            break
        tags[tag] = (i + 3 - off, length)
        i += 3 + length
    return tags


def _layout_holds(
    mm: np.ndarray, offs: np.ndarray, layout: dict[int, tuple[int, int]]
) -> bool:
    """True if every record at ``offs`` carries ``layout`` byte for byte.

    Checked rather than assumed: the whole speedup rests on one record's tag
    offsets standing in for a whole size class, so the claim is verified over
    the class before any field is read through it.
    """
    for tag, (rel, length) in layout.items():
        hdr = rel - 3
        if not bool(np.all(mm[offs + hdr] == tag)):
            return False
        lens = mm[offs + (hdr + 1)].astype(np.int64) | (
            mm[offs + (hdr + 2)].astype(np.int64) << 8
        )
        if not bool(np.all(lens == length)):
            return False
    return True


def _gather_le(mm: np.ndarray, offs: np.ndarray, rel: int, width: int) -> np.ndarray:
    """Little-endian unsigned ints at a fixed record-relative offset."""
    out = np.zeros(offs.size, dtype=np.uint64)
    for j in range(width):
        out |= mm[offs + (rel + j)].astype(np.uint64) << np.uint64(8 * j)
    return out


def _expand_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenation of ``arange(s, s + n)`` for each (s, n), without a loop."""
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    step = np.ones(total, dtype=np.int64)
    heads = np.zeros(starts.size, dtype=np.int64)
    heads[1:] = np.cumsum(lengths)[:-1]
    step[0] = starts[0]
    if starts.size > 1:
        step[heads[1:]] = starts[1:] - (starts[:-1] + lengths[:-1] - 1)
    return np.cumsum(step)


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
    ``tiles``/``ratio`` need no knowledge of the format.

    ``stride`` is always ``None``. Record length varies with bandwidth within
    a single file, and the two sizes *interleave* — an 80 MHz capture still
    receives the odd 20 MHz frame, in bursts of a few groups — so no single
    stride spans the file and ``batch``'s uniform fast path does not apply.

    That is a statement about the file, not about each record. Within one
    bandwidth every record is the same length and carries the same tags at
    byte-identical offsets, which is what ``_build_fast`` exploits: it parses
    the TLV layout once per size class and then reads each field for the whole
    class as one vectorised gather out of a memmap, instead of building a
    dict per record. On a 302 MB hour-long capture that is 2.7 s and a 923 MB
    Python heap peak down to 1.4 s and 69 MB, for a byte-identical index.
    ``_build`` remains as the fallback for anything that does not fit the
    assumption, and the two publish through the same ``_publish``.
    """

    chipset = "MediaTek"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._scan_end: int = 0
        self._scan_full()

    # ------------------------------------------------------------------ #
    #  Scan                                                               #
    # ------------------------------------------------------------------ #

    def _scan_full(self) -> None:
        self._csd: float | None | object = _CSD_UNSET
        size = self.path.stat().st_size
        if size < PREFIX_BYTES:
            self._init_empty()
            return
        mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        try:
            if self._build_fast(mm, 0, min(size, mm.size), existing=None) is not None:
                return
        finally:
            del mm
        with self.path.open("rb") as fh:
            buf = fh.read()
        self._build(memoryview(buf), min(size, len(buf)), base=0, existing=None)

    def _build_fast(self, mm: np.ndarray, start: int, end: int, *, existing) -> int | None:
        """Vectorised scan over ``mm[start:end]``, or ``None`` to fall back.

        Returns ``None`` — leaving no state touched — whenever the file does
        not meet the assumptions this path rests on, so the caller can run the
        record-by-record walk instead.
        """
        offs = _scan_record_offsets(mm, start, end)
        if offs.size == 0:
            return None
        sizes = np.empty(offs.size, dtype=np.int64)
        sizes[:-1] = np.diff(offs)
        last_body = int(mm[offs[-1] + 1]) | (int(mm[offs[-1] + 2]) << 8)
        sizes[-1] = PREFIX_BYTES + last_body

        # Per size class: one TLV parse, verified across the whole class.
        classes = np.unique(sizes)
        layouts: dict[int, dict[int, tuple[int, int]]] = {}
        for size_val in classes:
            size_i = int(size_val)
            sel = offs[sizes == size_i]
            layout = _class_layout(mm, int(sel[0]), size_i)
            if any(t not in layout for t in _REQUIRED_TAGS):
                return None
            if not _layout_holds(mm, sel, layout):
                return None
            layouts[size_i] = layout

        # Fields, gathered per class then scattered back into record order.
        seq = np.zeros(offs.size, dtype=np.int64)
        stamps_r = np.zeros(offs.size, dtype=np.int64)
        rssi_r = np.zeros(offs.size, dtype=np.int64)
        bw_code = np.zeros(offs.size, dtype=np.int64)
        tpi_r = np.zeros(offs.size, dtype=np.int64)
        rpi_r = np.zeros(offs.size, dtype=np.int64)
        real_rel = np.zeros(offs.size, dtype=np.int64)
        imag_rel = np.zeros(offs.size, dtype=np.int64)
        real_len = np.zeros(offs.size, dtype=np.int64)
        mac_rel = np.zeros(offs.size, dtype=np.int64)
        for size_i, layout in layouts.items():
            m = sizes == size_i
            sel = offs[m]
            s_rel, s_len = layout[TAG_SEQUENCE]
            seq[m] = _gather_le(mm, sel, s_rel, s_len).astype(np.int64)
            t_rel, t_len = layout[TAG_TIMESTAMP]
            stamps_r[m] = _gather_le(mm, sel, t_rel, t_len).astype(np.int64)
            r_rel, r_len = layout[TAG_RSSI]
            rssi_r[m] = _gather_le(mm, sel, r_rel, r_len).astype(np.int64)
            b_rel, b_len = layout[TAG_BANDWIDTH]
            bw_code[m] = _gather_le(mm, sel, b_rel, b_len).astype(np.int64)
            p_rel, p_len = layout[TAG_TPI]
            tpi_r[m] = _gather_le(mm, sel, p_rel, p_len).astype(np.int64)
            q_rel, q_len = layout[TAG_RPI]
            rpi_r[m] = _gather_le(mm, sel, q_rel, q_len).astype(np.int64)
            real_rel[m] = layout[TAG_CSI_REAL][0]
            real_len[m] = layout[TAG_CSI_REAL][1]
            imag_rel[m] = layout[TAG_CSI_IMAG][0]
            mac_rel[m] = layout[TAG_SOURCE_MAC][0]

        group_id = seq >> 16
        low = seq & 0xFFFF

        # Group boundaries, matching _build: a group is the run of records
        # sharing an id, and it only counts if a record with bit 15 closes it.
        closes = (low & 0x8000).astype(bool)
        new_run = np.empty(offs.size, dtype=bool)
        new_run[0] = True
        new_run[1:] = group_id[1:] != group_id[:-1]
        # A closing record also ends its run, so the next record starts a new one.
        new_run[1:] |= closes[:-1]
        run_of = np.cumsum(new_run) - 1
        n_runs = int(run_of[-1]) + 1

        run_start = np.flatnonzero(new_run)
        run_end = np.empty(n_runs, dtype=np.int64)  # exclusive
        run_end[:-1] = run_start[1:]
        run_end[-1] = offs.size
        # Keep only runs whose *last* record closes the group.
        keep = closes[run_end - 1]
        if not keep.any():
            return None
        run_start = run_start[keep]
        run_end = run_end[keep]
        n = run_start.size

        # rpi plane selection, tpi ascending — the same rule as _build.
        counts = run_end - run_start
        idx_all = _expand_ranges(run_start, counts)
        owner = np.repeat(np.arange(n, dtype=np.int64), counts)
        in_plane = rpi_r[idx_all] == RPI_PLANE
        # Order: plane first, then tpi ascending, stable within a group.
        order = np.lexsort((tpi_r[idx_all], ~in_plane, owner))
        idx_sorted = idx_all[order]
        owner_sorted = owner[order]
        rank = np.arange(idx_sorted.size, dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts
        )

        head_rec = idx_sorted[rank == 0]
        plane_counts = np.zeros(n, dtype=np.int64)
        np.add.at(plane_counts, owner_sorted[in_plane[order]], 1)

        nbins = np.array([_SUBCARRIERS.get(int(c), 0) for c in bw_code[head_rec]])

        offsets = offs[head_rec]
        stamps = stamps_r[head_rec]
        rssi = rssi_r[head_rec]
        bins = nbins.astype(np.int64)

        real_off = np.full((n, _MAX_SLOTS), -1, dtype=np.int64)
        imag_off = np.full((n, _MAX_SLOTS), -1, dtype=np.int64)
        total = np.zeros(n, dtype=np.int64)
        slots = np.zeros(n, dtype=np.int64)
        for slot in range(_MAX_SLOTS):
            sel_mask = rank == slot
            if not sel_mask.any():
                continue
            rec = idx_sorted[sel_mask]
            own = owner_sorted[sel_mask]
            # Only records in the chosen plane fill a slot, unless the group
            # has none at all — then _build keeps its first record as
            # single-stream, which is exactly rank 0.
            usable = in_plane[order][sel_mask] | (plane_counts[own] == 0)
            fits = real_len[rec] == bins[own] * 2
            ok = usable & fits
            rec, own = rec[ok], own[ok]
            real_off[own, slot] = offs[rec] + real_rel[rec]
            imag_off[own, slot] = offs[rec] + imag_rel[rec]
            total[own] += real_len[rec] * 2
            slots[own] += 1

        macs = [
            ":".join(f"{b:02x}" for b in bytes(mm[o + r : o + r + 6]))
            for o, r in zip(offs[head_rec], mac_rel[head_rec])
        ]
        widths = [_CHANNEL_WIDTH.get(int(c), "unknown") for c in bw_code[head_rec]]

        scan_end = int(offs[run_end[-1] - 1] + sizes[run_end[-1] - 1])
        return self._publish(
            n,
            offsets=offsets,
            stamps=stamps,
            rssi=rssi,
            csi_lengths=total,
            bins=bins,
            num_rx=slots,
            real_off=real_off,
            imag_off=imag_off,
            macs=macs,
            widths=widths,
            scan_end=scan_end,
            existing=existing,
        )

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

        return self._publish(
            n,
            offsets=offsets,
            stamps=stamps,
            rssi=rssi,
            csi_lengths=csi_lengths,
            bins=bins,
            num_rx=num_rx,
            real_off=real_off,
            imag_off=imag_off,
            macs=macs,
            widths=widths,
            scan_end=base + scan_end,
            existing=existing,
        )

    def _publish(
        self,
        n: int,
        *,
        offsets,
        stamps,
        rssi,
        csi_lengths,
        bins,
        num_rx,
        real_off,
        imag_off,
        macs,
        widths,
        scan_end: int,
        existing,
    ) -> int:
        """Install one scan's arrays, appending when ``existing`` is set.

        Shared by the vectorised scan and the record-by-record walk so the two
        cannot drift in what they expose.
        """
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
        self._scan_end = scan_end
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

    def csd_slope(self) -> float | None:
        """This capture's cyclic-shift ramp in rad/subcarrier, or ``None``.

        Measured once and remembered. Anchored to the *file* rather than to a
        batch for the reason ``ratio.Reference`` exists: an estimate rebuilt
        per view would let panning change the correction, and so change the
        picture, which is the one thing a correction must never do.

        ``extend`` keeps the measurement — a cyclic shift is a property of the
        transmitter, not of how much of the file has arrived. Only a rebuild
        after truncation discards it, since that may be a different file.

        Two callers racing may both measure it; the computation is
        deterministic, so the loser merely repeats the work.
        """
        if self._csd is _CSD_UNSET:
            self._csd = estimate_csd_slope(self.path, self)
        return self._csd  # type: ignore[return-value]

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
        mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        try:
            added = self._build_fast(
                mm, self._scan_end, min(size, mm.size), existing=True
            )
        finally:
            del mm
        if added is not None:
            return added
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


def estimate_csd_slope(path: str | Path, index: MTKIndex) -> float | None:
    """Measure a capture's cyclic-shift ramp, in radians per subcarrier.

    The transmitter delays each of its chains by a different fixed amount so
    that two antennas sending one stream cannot null each other out. A fixed
    delay is a linear phase across frequency, so dividing chain 1 by chain 0
    leaves that ramp behind — see the module docstring. It is deterministic,
    which is what makes it removable: the estimate here is a median of
    medians rather than a fit, because there is a single right answer and the
    job is only to find it through the noise.

    Estimating from *differences* between adjacent subcarriers keeps the
    answer independent of where the phase happens to be anchored, and keeps
    it valid for a 20 MHz frame and an 80 MHz frame alike — 802.11 spaces
    subcarriers 312.5 kHz apart at every one of these bandwidths, so a given
    delay is the same number of radians per subcarrier in all of them.

    Returns ``None`` rather than a number the caller should not trust: too
    few two-stream frames to vote, a ramp too shallow to be worth removing,
    or frames that disagree on its sign. That last gate is what stops this
    from inventing a ramp on a capture whose transmitter has only one chain,
    where the statistic is noise about zero and the sign splits near half.

    Unambiguous up to ``pi`` rad/SC, i.e. about 1600 ns, since the step
    between adjacent subcarriers is only known modulo ``2*pi``. Every cyclic
    shift the standard defines is far below that.
    """
    sel = np.flatnonzero(index.num_rx_arr >= 2)
    if sel.size < CSD_MIN_FRAMES:
        return None

    picks = sel[
        np.unique(
            np.linspace(0, sel.size - 1, min(CSD_SAMPLE, sel.size)).astype(np.int64)
        )
    ]
    _, _, _, ratio_phase = decode_frames(path, index, picks, deslope=False)

    steps = np.angle(np.exp(1j * np.diff(ratio_phase.astype(np.float64), axis=1)))
    usable = steps[np.isfinite(steps).sum(axis=1) >= CSD_MIN_STEPS]
    if usable.shape[0] < CSD_MIN_FRAMES:
        return None

    per_frame = np.nanmedian(usable, axis=1)
    slope = float(np.median(per_frame))
    if abs(slope) < CSD_MIN_SLOPE:
        return None
    if np.mean(np.sign(per_frame) == np.sign(slope)) < CSD_MIN_AGREEMENT:
        return None
    return slope


def decode_frames(
    path: str | Path,
    index: MTKIndex,
    frame_ids: np.ndarray,
    *,
    scaled: bool = False,
    interpolate: bool = True,
    deslope: bool = True,
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

    ``deslope`` removes the transmitter's cyclic-shift ramp from the ratio,
    using the whole file's estimate via ``MTKIndex.csd_slope``. It is aimed at
    ``ratio_phase``: the correction is a unit-magnitude rotation, so it leaves
    ``ratio_amp`` alone to within floating point — a few 1e-15 dB, since
    ``exp`` is not exactly unit-modulus in binary. ``amplitude`` and ``phase``
    read rx0 and never see it at all. Pass ``False`` to get the ratio as
    decoded — ``estimate_csd_slope`` does, to measure the ramp it removes.
    """
    path = Path(path)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    n = len(frame_ids)
    width = index.num_subcarriers

    empty = np.empty((0, width), dtype=np.float32)
    if n == 0:
        return empty, empty, empty, empty

    slope = index.csd_slope() if deslope else None

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
                if slope is not None:
                    # The ramp is a function of frequency offset from DC, and
                    # every bandwidth here is centred on DC, so anchoring the
                    # correction at the band's own centre leaves a 20 MHz and
                    # an 80 MHz frame on one phase reference instead of two.
                    k = np.arange(nbins) - nbins // 2
                    ratio = ratio * np.exp(-1j * slope * k)
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
