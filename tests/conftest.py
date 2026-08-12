"""Shared test fixtures across test modules."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from backend.index import HEADER_BYTES

CAPTURE = Path(__file__).resolve().parent.parent / "captures" / "capture.dat"


@pytest.fixture
def mixed_geometry_file(tmp_path: Path) -> Path:
    """A small .dat with interleaved 2x1 and 2x2 frames.

    Built from the first 6 frames of capture.dat. Frame 2 is rewritten as
    2x2: num_tx patched to 2, csi_length doubled, payload duplicated so the
    first (rx0, tx0) stream is bit-identical to the original 2x1 frame.

    Requires captures/capture.dat to exist.
    """
    if not CAPTURE.is_file():
        pytest.skip("captures/capture.dat not present")
    raw = CAPTURE.read_bytes()

    orig_cl = struct.unpack("I", raw[:4])[0]
    stride = HEADER_BYTES + orig_cl
    boundaries = [i * stride for i in range(7)]  # 6 frames + end

    data = bytearray(raw[: boundaries[6]])
    off = boundaries[2]
    new_cl = orig_cl * 2
    struct.pack_into("I", data, off, new_cl)
    data[off + 47] = 2  # num_tx: 1 -> 2
    payload_start = off + HEADER_BYTES
    payload_end = payload_start + orig_cl
    data[payload_end:payload_end] = data[payload_start:payload_end]

    target = tmp_path / "mixed.dat"
    target.write_bytes(bytes(data))
    return target
