"""Parity and incremental-correctness tests for backend.stream."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from CSIKit.reader import FeitCSIBeamformReader

from backend.parser import load_capture
from backend.stream import HEADER_BYTES, CaptureStream, decode_payload

CAPTURE = Path(__file__).resolve().parent.parent / "captures" / "capture.dat"

pytestmark = pytest.mark.skipif(
    not CAPTURE.is_file(), reason="captures/capture.dat not present"
)


def frame_offsets(raw: bytes) -> list[int]:
    """Byte offset of every frame boundary, including end-of-file."""
    offsets = [0]
    pos = 0
    while pos + HEADER_BYTES <= len(raw):
        csi_length = struct.unpack("I", raw[pos : pos + 4])[0]
        pos += HEADER_BYTES + csi_length
        if pos > len(raw):
            break
        offsets.append(pos)
    return offsets


def assert_matrices_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    """allclose that tolerates the -inf db() produces for zero-magnitude bins.

    CaptureStream stores float32 while load_capture returns float64, so the two
    cannot agree to float64 precision. The claim worth testing is narrower than
    "close": the arithmetic is identical and only the storage is narrower. So
    round the reference to the actual dtype first and demand agreement to a
    couple of ULP of that type. A blanket rtol=1e-5 would pass just as happily
    if the batched decoder started diverging for real.
    """
    assert actual.shape == expected.shape
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.isfinite(actual), finite)
    reference = expected[finite].astype(actual.dtype)
    np.testing.assert_allclose(actual[finite], reference, rtol=2e-7, atol=0)


@pytest.fixture(scope="module")
def raw() -> bytes:
    return CAPTURE.read_bytes()


@pytest.fixture(scope="module")
def reference():
    return load_capture(CAPTURE)


def test_decode_payload_matches_upstream(raw: bytes) -> None:
    """Vectorised payload decode is bit-identical to CSIKit's triple loop."""
    reader = FeitCSIBeamformReader()
    checked = 0
    for offset in frame_offsets(raw)[:20]:
        header = reader.parseHeader(raw[offset : offset + HEADER_BYTES])
        payload = raw[offset + HEADER_BYTES : offset + HEADER_BYTES + header["csi_length"]]
        if len(payload) < header["csi_length"]:
            break
        np.testing.assert_array_equal(
            decode_payload(payload, header), reader.parseCsiData(payload, header)
        )
        checked += 1
    assert checked > 0


def test_stream_matches_full_parse(reference) -> None:
    """One-shot stream read reproduces CSIKit's whole-file result."""
    stream = CaptureStream(CAPTURE)
    stream.update()
    snap = stream.snapshot(max_packets=0)

    assert stream.total_frames == len(reference)
    assert snap.num_subcarriers == reference.num_subcarriers
    assert snap.bandwidth == reference.bandwidth
    assert_matrices_equal(snap.amplitude, reference.amplitude)
    assert_matrices_equal(snap.phase, reference.phase)
    np.testing.assert_allclose(
        snap.time_seconds, reference.time_seconds, rtol=1e-12, atol=1e-12
    )


def test_growing_file_matches_full_parse(tmp_path: Path, raw: bytes, reference) -> None:
    """Appending in chunks -- including mid-frame splits -- changes nothing."""
    target = tmp_path / "growing.dat"
    target.write_bytes(b"")
    stream = CaptureStream(target)

    boundaries = frame_offsets(raw)
    # Cut points deliberately straddle frame boundaries so most polls land
    # partway through a frame that is still being written.
    cuts = sorted(
        {
            boundaries[3],
            boundaries[3] + 100,
            boundaries[10] + HEADER_BYTES + 4,
            boundaries[len(boundaries) // 2],
            len(raw) - 7,
            len(raw),
        }
    )

    written = 0
    for cut in cuts:
        with target.open("ab") as handle:
            handle.write(raw[written:cut])
        written = cut
        stream.update()

    snap = stream.snapshot(max_packets=0)
    assert stream.total_frames == len(reference)
    assert_matrices_equal(snap.amplitude, reference.amplitude)
    assert_matrices_equal(snap.phase, reference.phase)
    np.testing.assert_allclose(
        snap.time_seconds, reference.time_seconds, rtol=1e-12, atol=1e-12
    )


def test_partial_frame_is_withheld_until_complete(tmp_path: Path, raw: bytes) -> None:
    """A half-written frame is not decoded, and is picked up once finished."""
    boundaries = frame_offsets(raw)
    target = tmp_path / "partial.dat"

    target.write_bytes(raw[: boundaries[5] + HEADER_BYTES + 8])
    stream = CaptureStream(target)
    stream.update()
    assert stream.total_frames == 5

    with target.open("ab") as handle:
        handle.write(raw[boundaries[5] + HEADER_BYTES + 8 : boundaries[6]])
    stream.update()
    assert stream.total_frames == 6


def test_no_new_bytes_is_a_no_op(tmp_path: Path, raw: bytes) -> None:
    boundaries = frame_offsets(raw)
    target = tmp_path / "static.dat"
    target.write_bytes(raw[: boundaries[8]])

    stream = CaptureStream(target)
    stream.update()
    first = stream.snapshot(max_packets=0)
    for _ in range(3):
        stream.update()
    second = stream.snapshot(max_packets=0)

    assert stream.total_frames == 8
    assert_matrices_equal(second.amplitude, first.amplitude)
    np.testing.assert_array_equal(second.time_seconds, first.time_seconds)


def test_truncation_resets_stream(tmp_path: Path, raw: bytes) -> None:
    """A shrinking file invalidates the offset and timestamp chain."""
    boundaries = frame_offsets(raw)
    target = tmp_path / "rotated.dat"
    target.write_bytes(raw[: boundaries[8]])

    stream = CaptureStream(target)
    stream.update()
    assert stream.total_frames == 8

    target.write_bytes(raw[: boundaries[3]])
    stream.update()
    assert stream.total_frames == 3
    assert stream.snapshot(max_packets=0).time_seconds[0] == 0.0


def test_snapshot_returns_trailing_window(reference) -> None:
    stream = CaptureStream(CAPTURE)
    stream.update()
    snap = stream.snapshot(max_packets=50)

    assert snap.amplitude.shape[0] == 50
    assert_matrices_equal(snap.amplitude, reference.amplitude[-50:])
    np.testing.assert_allclose(snap.time_seconds, reference.time_seconds[-50:])


def test_empty_capture_snapshot(tmp_path: Path) -> None:
    target = tmp_path / "empty.dat"
    target.write_bytes(b"")
    stream = CaptureStream(target)
    stream.update()

    snap = stream.snapshot(max_packets=200)
    assert stream.total_frames == 0
    assert len(snap) == 0
    assert snap.num_subcarriers == 0


def test_buffer_is_bounded(tmp_path: Path, raw: bytes) -> None:
    """Retained frames are capped; total_frames still counts everything."""
    boundaries = frame_offsets(raw)
    target = tmp_path / "bounded.dat"
    target.write_bytes(raw[: boundaries[20]])

    stream = CaptureStream(target, max_frames=5)
    stream.update()
    assert stream.total_frames == 20
    assert stream.snapshot(max_packets=0).amplitude.shape[0] == 5


def test_subcarriers_are_centred_not_fft_ordered(reference) -> None:
    """Regression guard for the removed fftshift.

    FeitCSI emits a contiguous, centred spectrum. fftshift would split it and
    join the two outer edges, leaving a step far larger than any real
    bin-to-bin change. Assert no such seam exists.
    """
    profile = reference.amplitude.mean(axis=0)
    steps = np.abs(np.diff(profile))
    assert steps.max() < 10 * np.median(steps)

    shifted = np.fft.fftshift(profile)
    assert np.abs(np.diff(shifted)).max() > 5 * steps.max()


def test_load_capture_has_no_fftshift_option() -> None:
    import inspect

    assert "fftshift" not in inspect.signature(load_capture).parameters


def test_decode_is_capped_to_the_retained_window(
    tmp_path: Path, raw: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frames that the ring buffer will drop are never decoded.

    The deque keeps only max_frames rows, so decoding older frames produces
    values that are discarded the moment they are appended. On a 211 MB capture
    that is 95,787 frames decoded to retain 10,000. Assert the decoder is asked
    only for the tail, while total_frames still reports the whole file.
    """
    boundaries = frame_offsets(raw)
    target = tmp_path / "capped.dat"
    target.write_bytes(raw[: boundaries[20]])

    import backend.stream as stream_mod

    requested: list[int] = []
    real_decode = stream_mod.decode_frames

    def spy(path, index, frame_ids, **kwargs):
        requested.extend(int(i) for i in frame_ids)
        return real_decode(path, index, frame_ids, **kwargs)

    monkeypatch.setattr(stream_mod, "decode_frames", spy)

    stream = CaptureStream(target, max_frames=5)
    stream.update()

    assert requested == [15, 16, 17, 18, 19]
    assert stream.total_frames == 20
    assert stream.snapshot(max_packets=0).amplitude.shape[0] == 5


def test_capped_window_matches_the_reference_tail(reference) -> None:
    """Skipping older frames does not disturb the retained rows or their times.

    The timestamp chain is cumulative, so a decoder that skipped frames could
    plausibly lose the running offset. It does not: FrameIndex derives every
    timestamp from headers alone, independently of what gets decoded.
    """
    stream = CaptureStream(CAPTURE, max_frames=25)
    stream.update()
    snap = stream.snapshot(max_packets=0)

    assert snap.amplitude.shape[0] == 25
    assert_matrices_equal(snap.amplitude, reference.amplitude[-25:])
    assert_matrices_equal(snap.phase, reference.phase[-25:])
    np.testing.assert_allclose(
        snap.time_seconds, reference.time_seconds[-25:], rtol=1e-12, atol=1e-12
    )
