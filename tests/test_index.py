"""Tests for backend.index.FrameIndex — structural scan and timestamps."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from CSIKit.reader import FeitCSIBeamformReader

from backend.index import (
    FTM_WRAP_SECONDS,
    HEADER_BYTES,
    MAX_TICK,
    TICK_RESOLUTION,
    FrameIndex,
)

CAPTURE = Path(__file__).resolve().parent.parent / "captures" / "capture.dat"

pytestmark = pytest.mark.skipif(
    not CAPTURE.is_file(), reason="captures/capture.dat not present"
)


def _frame_offsets(raw: bytes) -> list[int]:
    """Byte offset of every frame boundary, matching test_stream.py."""
    offsets = []
    pos = 0
    while pos + HEADER_BYTES <= len(raw):
        csi_length = struct.unpack("I", raw[pos : pos + 4])[0]
        offsets.append(pos)
        pos += HEADER_BYTES + csi_length
        if pos > len(raw):
            break
    return offsets


def _reference_timestamps(ftm: list[int], mu: list[int]) -> list[float]:
    """Per-frame timestamp chain, mirroring read_file exactly."""
    n = len(ftm)
    times = [0.0]
    for i in range(1, n):
        if (mu[i] - mu[i - 1]) / 1e6 < FTM_WRAP_SECONDS:
            if ftm[i] > ftm[i - 1]:
                diff = ftm[i] - ftm[i - 1]
            else:
                diff = ftm[i] + (MAX_TICK - ftm[i - 1])
            times.append(times[-1] + (diff * TICK_RESOLUTION) / 1e9)
        else:
            if mu[i] > mu[i - 1]:
                diff = mu[i] - mu[i - 1]
            else:
                diff = mu[i] + (MAX_TICK - mu[i - 1])
            times.append(times[-1] + diff / 1e6)
    return times


@pytest.fixture(scope="module")
def raw() -> bytes:
    return CAPTURE.read_bytes()


@pytest.fixture(scope="module")
def index() -> FrameIndex:
    return FrameIndex(CAPTURE)


# ----------------------------------------------------------------------- #
#  Offset and timestamp parity with a sequential reference walk           #
# ----------------------------------------------------------------------- #


def test_offsets_match_sequential_walk(raw: bytes, index: FrameIndex) -> None:
    """FrameIndex offsets match a byte-by-byte frame walk."""
    ref = _frame_offsets(raw)
    assert index.count == len(ref)
    np.testing.assert_array_equal(index.offsets, np.array(ref, dtype=np.int64))


def test_times_match_sequential_walk(raw: bytes, index: FrameIndex) -> None:
    """FrameIndex timestamps match the per-frame reference computation."""
    reader = FeitCSIBeamformReader()
    ftm_list: list[int] = []
    mu_list: list[int] = []
    pos = 0
    while pos + HEADER_BYTES <= len(raw):
        h = reader.parseHeader(raw[pos : pos + HEADER_BYTES])
        ftm_list.append(h["ftm_clock"])
        mu_list.append(h["mu_clock"])
        pos += HEADER_BYTES + h["csi_length"]

    ref = np.array(_reference_timestamps(ftm_list, mu_list))
    assert index.times.shape == ref.shape
    np.testing.assert_allclose(index.times, ref, rtol=1e-12, atol=1e-12)


def test_vectorized_timestamps_match_iterative(raw: bytes) -> None:
    """Vectorized _compute_timestamps matches the iterative reference to 1e-12."""
    from backend.index import _compute_timestamps

    reader = FeitCSIBeamformReader()
    ftm_list: list[int] = []
    mu_list: list[int] = []
    pos = 0
    while pos + HEADER_BYTES <= len(raw):
        h = reader.parseHeader(raw[pos : pos + HEADER_BYTES])
        ftm_list.append(h["ftm_clock"])
        mu_list.append(h["mu_clock"])
        pos += HEADER_BYTES + h["csi_length"]

    ftm = np.array(ftm_list, dtype=np.int64)
    mu = np.array(mu_list, dtype=np.int64)
    vec = _compute_timestamps(ftm, mu)
    ref = np.array(_reference_timestamps(ftm_list, mu_list))

    np.testing.assert_allclose(vec, ref, rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------- #
#  Uniform fast path vs. sequential fallback                              #
# ----------------------------------------------------------------------- #


def test_fast_path_used_on_uniform_capture(index: FrameIndex) -> None:
    """The real capture has uniform stride, so the fast path is used."""
    assert index._uniform is True
    assert index.stride is not None
    assert index.stride == HEADER_BYTES + 1936  # 272 + 1936 = 2208


def test_fallback_matches_fast_path(tmp_path: Path, raw: bytes) -> None:
    """A trailing partial frame forces the sequential fallback; results match.

    The complete frames must have the same offsets, timestamps, and metadata
    as the fast path on the original file.
    """
    fast = FrameIndex(CAPTURE)
    assert fast._uniform is True

    # Append partial bytes so size % stride != 0, forcing sequential fallback.
    target = tmp_path / "partial.dat"
    target.write_bytes(raw + b"\x00" * 100)
    slow = FrameIndex(target)
    assert slow._uniform is False

    assert slow.count == fast.count
    np.testing.assert_array_equal(slow.offsets, fast.offsets)
    np.testing.assert_allclose(slow.times, fast.times, rtol=1e-12, atol=1e-12)
    assert slow.num_subcarriers == fast.num_subcarriers
    assert slow.num_rx == fast.num_rx
    assert slow.num_tx == fast.num_tx
    assert slow.bandwidth == fast.bandwidth
    assert slow.stride == fast.stride  # uniform csi_length detected


def test_fallback_with_variable_stride(tmp_path: Path, raw: bytes) -> None:
    """A synthetic file with varying csi_length uses the fallback correctly."""
    boundaries = _frame_offsets(raw)
    # Take first 5 frames; pad frame 2's payload to change its csi_length.
    data = bytearray(raw[: boundaries[5]])
    orig_cl = struct.unpack("I", data[boundaries[2] : boundaries[2] + 4])[0]
    pad = 8
    new_cl = orig_cl + pad
    struct.pack_into("I", data, boundaries[2], new_cl)
    data[boundaries[3] : boundaries[3]] = b"\x00" * pad

    target = tmp_path / "variable.dat"
    target.write_bytes(bytes(data))

    idx = FrameIndex(target)
    assert idx._uniform is False
    assert idx.stride is None  # variable stride
    assert idx.count == 5

    # Verify offsets: frame 3 shifts by pad bytes.
    expected_offsets = [
        boundaries[0],
        boundaries[1],
        boundaries[2],
        boundaries[3] + pad,
        boundaries[4] + pad,
    ]
    np.testing.assert_array_equal(idx.offsets, np.array(expected_offsets, dtype=np.int64))


# ----------------------------------------------------------------------- #
#  Incremental extension and truncation                                   #
# ----------------------------------------------------------------------- #


def test_incremental_extension_matches_full(raw: bytes) -> None:
    """Extending a file in chunks yields the same index as a full scan."""
    full = FrameIndex(CAPTURE)

    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
        target = Path(f.name)
    try:
        target.write_bytes(b"")
        idx = FrameIndex(target)

        boundaries = _frame_offsets(raw)
        cuts = sorted({
            boundaries[3],
            boundaries[3] + 100,  # mid-frame split
            boundaries[10] + HEADER_BYTES + 4,  # mid-payload split
            boundaries[len(boundaries) // 2],
            len(raw) - 7,
            len(raw),
        })

        written = 0
        for cut in cuts:
            with target.open("ab") as fh:
                fh.write(raw[written:cut])
            written = cut
            idx.extend()

        assert idx.count == full.count
        np.testing.assert_array_equal(idx.offsets, full.offsets)
        np.testing.assert_allclose(idx.times, full.times, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(idx.ftm_clocks, full.ftm_clocks)
        np.testing.assert_array_equal(idx.mu_clocks, full.mu_clocks)
        np.testing.assert_array_equal(idx.rssi_1, full.rssi_1)
    finally:
        target.unlink(missing_ok=True)


def test_truncation_rebuilds(raw: bytes) -> None:
    """A shrinking file triggers a full rebuild of the index."""
    boundaries = _frame_offsets(raw)

    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
        target = Path(f.name)
    try:
        target.write_bytes(raw[: boundaries[8]])
        idx = FrameIndex(target)
        assert idx.count == 8

        # Truncate to 3 frames.
        target.write_bytes(raw[: boundaries[3]])
        idx.extend()
        assert idx.count == 3
        np.testing.assert_array_equal(
            idx.offsets, np.arange(3, dtype=np.int64) * idx.stride
        )
        assert idx.times[0] == 0.0
    finally:
        target.unlink(missing_ok=True)


def test_empty_file_index() -> None:
    """An empty file produces an empty index with no crash."""
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
        target = Path(f.name)
    try:
        target.write_bytes(b"")
        idx = FrameIndex(target)
        assert idx.count == 0
        assert idx.offsets.shape == (0,)
        assert idx.times.shape == (0,)
        assert idx.num_subcarriers == 0
        assert idx.bandwidth == "unknown"
    finally:
        target.unlink(missing_ok=True)
