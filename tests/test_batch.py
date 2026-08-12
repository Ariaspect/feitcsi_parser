"""Tests for backend.batch.decode_frames — vectorised batched decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.batch import decode_frames
from backend.index import FrameIndex
from backend.parser import load_capture

CAPTURE = Path(__file__).resolve().parent.parent / "captures" / "capture.dat"

pytestmark = pytest.mark.skipif(
    not CAPTURE.is_file(), reason="captures/capture.dat not present"
)


def assert_matrices_close(actual: np.ndarray, expected: np.ndarray, *, rtol: float = 1e-5) -> None:
    """allclose that tolerates the -inf db() produces for zero-magnitude bins."""
    assert actual.shape == expected.shape
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.isfinite(actual), finite)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=rtol, atol=1e-6)


@pytest.fixture(scope="module")
def index() -> FrameIndex:
    return FrameIndex(CAPTURE)


@pytest.fixture(scope="module")
def reference():
    return load_capture(CAPTURE)


# ----------------------------------------------------------------------- #
#  Parity with load_capture (CSIKit reference path)                       #
# ----------------------------------------------------------------------- #


def test_decode_matches_load_capture(index: FrameIndex, reference) -> None:
    """Full batch decode matches CSIKit's whole-file parse within float32 tolerance."""
    all_ids = np.arange(index.count)
    amp, phase, _, _ = decode_frames(CAPTURE, index, all_ids, scaled=True, interpolate=True)

    assert amp.shape == (index.count, index.num_subcarriers)
    assert phase.shape == (index.count, index.num_subcarriers)
    assert amp.dtype == np.float32
    assert phase.dtype == np.float32

    assert_matrices_close(amp, reference.amplitude)
    assert_matrices_close(phase, reference.phase)


def test_ratio_matches_load_capture(index: FrameIndex, reference) -> None:
    """Batch CSI ratio decode matches load_capture's ratio within float32 tolerance."""
    all_ids = np.arange(index.count)
    _, _, ratio_amp, ratio_phase = decode_frames(CAPTURE, index, all_ids)

    assert ratio_amp.shape == (index.count, index.num_subcarriers)
    assert ratio_phase.shape == (index.count, index.num_subcarriers)
    assert ratio_amp.dtype == np.float32
    assert ratio_phase.dtype == np.float32

    assert_matrices_close(ratio_amp, reference.ratio_amplitude)

    # Phase wraps at ±π; at the seam, batch and load_capture can flip sign
    # (differ by 2π) due to floating-point reconstruction order. exp(1j*x)
    # is invariant to this flip, so compare via the complex exponential.
    np.testing.assert_allclose(
        np.exp(1j * ratio_phase.astype(np.float64)),
        np.exp(1j * reference.ratio_phase.astype(np.float64)),
        rtol=1e-5, atol=1e-6,
    )


def test_decode_times_match(index: FrameIndex, reference) -> None:
    """FrameIndex timestamps match the reference within 1e-12."""
    np.testing.assert_allclose(
        index.times, reference.time_seconds, rtol=1e-12, atol=1e-12
    )


def test_decode_unscaled_matches(index: FrameIndex) -> None:
    """Unscaled decode produces consistent (finite) output."""
    all_ids = np.arange(index.count)
    amp, phase, _, _ = decode_frames(
        CAPTURE, index, all_ids, scaled=False, interpolate=True
    )
    assert amp.dtype == np.float32
    assert phase.dtype == np.float32
    # Unscaled amplitude is still finite everywhere (no zero-magnitude bins
    # after interpolation, since raw int16 values are non-zero for this capture).
    assert np.all(np.isfinite(amp))


def test_decode_no_interpolation(index: FrameIndex) -> None:
    """Decode without interpolation still produces valid output."""
    all_ids = np.arange(index.count)
    amp, phase, _, _ = decode_frames(
        CAPTURE, index, all_ids, scaled=True, interpolate=False
    )
    assert amp.shape == (index.count, index.num_subcarriers)
    assert phase.shape == (index.count, index.num_subcarriers)


# ----------------------------------------------------------------------- #
#  Non-contiguous selection parity                                        #
# ----------------------------------------------------------------------- #


def test_strided_selection_matches_full(index: FrameIndex) -> None:
    """A non-contiguous frame_ids selection equals the same rows of a full decode."""
    all_ids = np.arange(index.count)
    amp_full, phase_full, _, _ = decode_frames(CAPTURE, index, all_ids)

    # Strided selection: every 7th frame.
    strided = np.arange(0, index.count, 7)
    amp_strided, phase_strided, _, _ = decode_frames(CAPTURE, index, strided)

    assert amp_strided.shape == (len(strided), index.num_subcarriers)
    np.testing.assert_allclose(
        amp_strided, amp_full[strided], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        phase_strided, phase_full[strided], rtol=1e-6, atol=1e-7
    )


def test_arbitrary_selection_matches_full(index: FrameIndex) -> None:
    """An arbitrary (unsorted, non-contiguous) selection matches full decode rows."""
    all_ids = np.arange(index.count)
    amp_full, phase_full, _, _ = decode_frames(CAPTURE, index, all_ids)

    rng = np.random.default_rng(42)
    picks = rng.choice(index.count, size=50, replace=False)
    amp_pick, phase_pick, _, _ = decode_frames(CAPTURE, index, picks)

    np.testing.assert_allclose(
        amp_pick, amp_full[picks], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        phase_pick, phase_full[picks], rtol=1e-6, atol=1e-7
    )


def test_single_frame_decode(index: FrameIndex) -> None:
    """A single-frame selection produces one row."""
    amp, phase, _, _ = decode_frames(CAPTURE, index, np.array([42]))
    assert amp.shape == (1, index.num_subcarriers)
    assert phase.shape == (1, index.num_subcarriers)


def test_empty_selection(index: FrameIndex) -> None:
    """An empty frame_ids selection returns empty arrays."""
    amp, phase, _, _ = decode_frames(CAPTURE, index, np.array([], dtype=np.int64))
    assert amp.shape == (0, index.num_subcarriers)
    assert phase.shape == (0, index.num_subcarriers)
    assert amp.dtype == np.float32
    assert phase.dtype == np.float32


# ----------------------------------------------------------------------- #
#  Mixed-geometry decode (interleaved 2x1 / 2x2 frames)                   #
# ----------------------------------------------------------------------- #


def test_decode_mixed_geometry_does_not_raise(
    mixed_geometry_file: Path,
) -> None:
    """Interleaved 2x1/2x2 frames decode without the varying-csi_length error."""
    idx = FrameIndex(mixed_geometry_file)
    all_ids = np.arange(idx.count)
    amp, phase, _, _ = decode_frames(mixed_geometry_file, idx, all_ids)

    assert amp.shape == (6, idx.num_subcarriers)
    assert phase.shape == (6, idx.num_subcarriers)
    assert amp.dtype == np.float32
    assert phase.dtype == np.float32


def test_decode_mixed_geometry_matches_grouped(
    mixed_geometry_file: Path,
) -> None:
    """Batch decode of mixed geometry matches per-group decode.

    Decoding all 6 frames together must produce the same result as decoding
    the 2x1 frames (0,1,3,4,5) and the 2x2 frame (2) separately. This proves
    the group-by-geometry scatter preserves order and values.
    """
    idx = FrameIndex(mixed_geometry_file)

    ids_2x1 = np.array([0, 1, 3, 4, 5])
    ids_2x2 = np.array([2])
    amp_2x1, phase_2x1, _, _ = decode_frames(mixed_geometry_file, idx, ids_2x1)
    amp_2x2, phase_2x2, _, _ = decode_frames(mixed_geometry_file, idx, ids_2x2)

    all_ids = np.arange(6)
    amp_all, phase_all, _, _ = decode_frames(mixed_geometry_file, idx, all_ids)

    np.testing.assert_allclose(amp_all[ids_2x1], amp_2x1, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(phase_all[ids_2x1], phase_2x1, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(amp_all[ids_2x2], amp_2x2, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(phase_all[ids_2x2], phase_2x2, rtol=1e-6, atol=1e-7)
