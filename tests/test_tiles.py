"""Tests for backend.tiles — pre-aggregated tile serving and the /api/tile, /api/meta endpoints."""

from __future__ import annotations

import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.batch import decode_frames
from backend.index import FrameIndex
from backend.parser import load_capture
from backend.tiles import (
    BLOCK_SIZE,
    TILE_FRAME_BUDGET,
    _block_cache,
    compute_tile,
    reset_tile_caches,
)

CAPTURE = Path(__file__).resolve().parent.parent / "captures" / "capture.dat"

pytestmark = pytest.mark.skipif(
    not CAPTURE.is_file(), reason="captures/capture.dat not present"
)


# ----------------------------------------------------------------------- #
#  Fixtures                                                               #
# ----------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def index() -> FrameIndex:
    return FrameIndex(CAPTURE)


@pytest.fixture(scope="module")
def reference():
    return load_capture(CAPTURE)


@pytest.fixture(autouse=True)
def _clean_caches():
    """Reset tile caches before each test so state does not leak."""
    reset_tile_caches()
    yield
    reset_tile_caches()


# ----------------------------------------------------------------------- #
#  Helpers                                                                #
# ----------------------------------------------------------------------- #


def _full_range(index: FrameIndex) -> tuple[float, float]:
    """Time range covering every frame.

    Exactly [times[0], times[-1]] -- no epsilon.  That is what /api/meta
    reports and therefore what a client asking for the whole capture sends, so
    it is the range the tests must exercise.  compute_tile treats the range as
    closed on both ends; an epsilon here would hide a regression to half-open.
    """
    return float(index.times[0]), float(index.times[-1])


def _assert_grids_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    """allclose that tolerates -inf from db(0) and NaN from gap columns."""
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    if finite.any():
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------- #
#  1. Full-range tile reproduces decode_frames                            #
# ----------------------------------------------------------------------- #


def test_full_range_tile_reproduces_decode(index: FrameIndex) -> None:
    """A tile whose width equals the frame count reproduces decode_frames,
    modulo the documented aggregation (max-hold where two frames share a
    column, NaN where a column has none)."""
    width = index.count
    t0, t1 = _full_range(index)

    grid, meta = compute_tile(CAPTURE, t0, t1, width, "amplitude")

    assert grid.shape == (index.num_subcarriers, width)
    assert meta["exact"] is True

    all_ids = np.arange(index.count)
    amp, _ = decode_frames(CAPTURE, index, all_ids)

    # Independently compute the expected grid using the same column logic.
    times = index.times
    span = t1 - t0
    col_edges = t0 + np.arange(width + 1, dtype=np.float64) / width * span
    col_starts = np.searchsorted(times, col_edges[:-1], side="left")
    col_ends = np.searchsorted(times, col_edges[1:], side="left")
    col_ends[-1] = index.count  # last column closed on the right

    expected = np.full((index.num_subcarriers, width), np.nan, dtype=np.float32)
    for x in range(width):
        s, e = int(col_starts[x]), int(col_ends[x])
        if e > s:
            expected[:, x] = amp[s:e].max(axis=0)
    expected = np.ascontiguousarray(expected[::-1, :])

    _assert_grids_equal(grid, expected)


# ----------------------------------------------------------------------- #
#  2. Max-hold aggregation                                                #
# ----------------------------------------------------------------------- #


def test_max_hold_two_frames_one_column(index: FrameIndex) -> None:
    """Two frames in one column: the per-subcarrier max comes out."""
    t0 = float(index.times[10])
    t1 = float(index.times[11]) + (float(index.times[12]) - float(index.times[11])) / 2
    grid, meta = compute_tile(CAPTURE, t0, t1, 1, "amplitude")

    amp, _ = decode_frames(CAPTURE, index, np.array([10, 11]))
    expected = np.max(amp, axis=0)[::-1].astype(np.float32).reshape(-1, 1)

    _assert_grids_equal(grid, expected)
    assert meta["frames_decoded"] == 2
    assert meta["exact"] is True


# ----------------------------------------------------------------------- #
#  3. Nearest-frame phase aggregation                                     #
# ----------------------------------------------------------------------- #


def test_phase_nearest_to_centre(index: FrameIndex) -> None:
    """Phase: the frame closest to the column centre wins."""
    t0 = float(index.times[10])
    t1 = float(index.times[12])
    grid, meta = compute_tile(CAPTURE, t0, t1, 1, "phase")

    centre = (t0 + t1) / 2
    # The range is closed at both ends, so frame 12 -- sitting exactly on t1 --
    # is a candidate alongside 10 and 11.
    candidates = np.array([10, 11, 12])
    dists = np.abs(index.times[candidates] - centre)
    nearest = int(candidates[np.argmin(dists)])

    _, phase = decode_frames(CAPTURE, index, np.array([nearest]))
    expected = phase[0][::-1].astype(np.float32).reshape(-1, 1)

    _assert_grids_equal(grid, expected)
    assert meta["frames_decoded"] == 3


# ----------------------------------------------------------------------- #
#  4. Gap columns are NaN; -inf from db(0) is preserved                   #
# ----------------------------------------------------------------------- #


def test_gap_columns_are_nan(index: FrameIndex) -> None:
    """Columns with no frames are NaN, not zero or carry-forward."""
    # Range extends well beyond the capture on both sides.
    t0 = float(index.times[0]) - 10.0
    t1 = float(index.times[-1]) + 10.0
    grid, _ = compute_tile(CAPTURE, t0, t1, 200, "amplitude")

    assert grid.shape == (index.num_subcarriers, 200)
    # The first and last columns are outside the frame range → NaN.
    assert np.all(np.isnan(grid[:, 0]))
    assert np.all(np.isnan(grid[:, -1]))


def test_neg_inf_preserved(index: FrameIndex) -> None:
    """-inf from db(0) is preserved as -inf, not turned into NaN."""
    t0, t1 = _full_range(index)
    width = index.count
    grid, _ = compute_tile(CAPTURE, t0, t1, width, "amplitude")

    all_ids = np.arange(index.count)
    amp, _ = decode_frames(CAPTURE, index, all_ids)

    # If the decoded data has -inf, the tile must have -inf at the
    # corresponding positions (for single-frame columns).
    if np.any(np.isneginf(amp)):
        # Find a column with exactly one frame that has -inf.
        times = index.times
        span = t1 - t0
        col_edges = t0 + np.arange(width + 1, dtype=np.float64) / width * span
        col_starts = np.searchsorted(times, col_edges[:-1], side="left")
        col_ends = np.searchsorted(times, col_edges[1:], side="left")

        found = False
        for x in range(width):
            s, e = int(col_starts[x]), int(col_ends[x])
            if e - s == 1 and np.any(np.isneginf(amp[s])):
                sc = np.where(np.isneginf(amp[s]))[0]
                # After flip, subcarrier sc maps to row num_sc - 1 - sc.
                for sc_idx in sc:
                    assert np.isneginf(grid[index.num_subcarriers - 1 - int(sc_idx), x])
                    found = True
        assert found, "expected at least one -inf position in a single-frame column"
    # If no -inf in the data, the tile should have none either.
    assert not np.any(np.isneginf(grid)) if not np.any(np.isneginf(amp)) else True


# ----------------------------------------------------------------------- #
#  5. Row 0 = highest subcarrier index                                    #
# ----------------------------------------------------------------------- #


def test_row_zero_is_highest_subcarrier(index: FrameIndex) -> None:
    """Row 0 of the grid corresponds to the highest subcarrier index."""
    t0 = float(index.times[5])
    t1 = float(index.times[5]) + 1e-9  # exactly one frame
    grid, _ = compute_tile(CAPTURE, t0, t1, 1, "amplitude")

    amp, _ = decode_frames(CAPTURE, index, np.array([5]))

    # Row 0 should be subcarrier num_sc - 1 (highest).
    np.testing.assert_allclose(grid[0, 0], amp[0, -1], rtol=1e-5)
    # Last row should be subcarrier 0 (lowest).
    np.testing.assert_allclose(grid[-1, 0], amp[0, 0], rtol=1e-5)


# ----------------------------------------------------------------------- #
#  6. Sampled range: exact=False, budget respected                        #
# ----------------------------------------------------------------------- #


def test_sampled_range_respects_budget(index: FrameIndex, monkeypatch) -> None:
    """A range with more frames than the budget returns exact=False,
    decodes no more than the budget, and still returns a full-width tile."""
    monkeypatch.setattr("backend.tiles.TILE_FRAME_BUDGET", 100)

    t0, t1 = _full_range(index)
    width = 800
    grid, meta = compute_tile(CAPTURE, t0, t1, width, "amplitude")

    assert meta["exact"] is False
    assert meta["frames_decoded"] <= 100
    assert meta["total_in_range"] == index.count
    assert grid.shape == (index.num_subcarriers, width)
    # The block cache should not have been used (sampled = direct decode).
    assert _block_cache.frames_decoded == 0


# ----------------------------------------------------------------------- #
#  7. Narrow range: exact=True                                            #
# ----------------------------------------------------------------------- #


def test_narrow_range_is_exact(index: FrameIndex) -> None:
    """A range that fits within the budget returns exact=True."""
    t0 = float(index.times[10])
    t1 = float(index.times[50])
    grid, meta = compute_tile(CAPTURE, t0, t1, 400, "amplitude")

    assert meta["exact"] is True
    assert meta["total_in_range"] <= TILE_FRAME_BUDGET
    assert grid.shape == (index.num_subcarriers, 400)


# ----------------------------------------------------------------------- #
#  8. Block cache: identical bytes, fewer decodes on overlap              #
# ----------------------------------------------------------------------- #


def test_block_cache_identical_bytes(index: FrameIndex) -> None:
    """A repeat request returns identical bytes from the block cache."""
    t0 = float(index.times[0])
    t1 = float(index.times[200])
    width = 300

    grid1, _ = compute_tile(CAPTURE, t0, t1, width, "amplitude")
    bytes1 = grid1.tobytes()
    decoded_after_first = _block_cache.frames_decoded

    grid2, _ = compute_tile(CAPTURE, t0, t1, width, "amplitude")
    bytes2 = grid2.tobytes()
    decoded_after_second = _block_cache.frames_decoded

    assert bytes1 == bytes2
    # Second request should decode zero additional frames (all cached).
    assert decoded_after_second == decoded_after_first


def test_block_cache_overlapping_decodes_fewer(index: FrameIndex) -> None:
    """A second request for an overlapping range decodes fewer frames."""
    t0_a = float(index.times[0])
    t1_a = float(index.times[400])
    width = 500

    compute_tile(CAPTURE, t0_a, t1_a, width, "amplitude")
    decoded_after_first = _block_cache.frames_decoded
    assert decoded_after_first > 0

    # Overlapping range: shares the same block(s) as the first.
    t0_b = float(index.times[200])
    t1_b = float(index.times[600])
    compute_tile(CAPTURE, t0_b, t1_b, width, "amplitude")
    decoded_after_second = _block_cache.frames_decoded

    # The second request should have decoded fewer frames (blocks cached).
    assert (decoded_after_second - decoded_after_first) < decoded_after_first


# ----------------------------------------------------------------------- #
#  9. Truncation invalidates cached blocks                                #
# ----------------------------------------------------------------------- #


def test_truncation_invalidates_cache(index: FrameIndex, raw: bytes) -> None:
    """Truncating a file invalidates cached blocks rather than serving stale data."""
    boundaries = []
    pos = 0
    while pos + 272 <= len(raw):
        csi_length = struct.unpack("I", raw[pos : pos + 4])[0]
        boundaries.append(pos)
        pos += 272 + csi_length
        if pos > len(raw):
            break

    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
        target = Path(f.name)
    try:
        # Write full capture, request a tile (caches blocks).
        target.write_bytes(raw)
        full_idx = FrameIndex(target)
        t0 = float(full_idx.times[0])
        t1 = float(full_idx.times[100])
        grid_full, _ = compute_tile(target, t0, t1, 100, "amplitude")

        # Truncate to 5 frames.
        target.write_bytes(raw[: boundaries[5]])
        grid_trunc, meta = compute_tile(target, t0, t1, 100, "amplitude")

        # The truncated tile must reflect the truncated file, not the cache.
        trunc_idx = FrameIndex(target)
        assert meta["total_in_range"] <= 5
        assert trunc_idx.count <= 5

        # Decode the truncated file independently for comparison.
        trunc_ids = np.arange(trunc_idx.count)
        if len(trunc_ids) > 0:
            amp_trunc, _ = decode_frames(target, trunc_idx, trunc_ids)
            # The grid should not equal the full-capture grid.
            assert not np.array_equal(
                np.isfinite(grid_trunc), np.isfinite(grid_full)
            ) or not np.allclose(
                grid_trunc[np.isfinite(grid_trunc)],
                grid_full[np.isfinite(grid_trunc)],
                rtol=1e-5, equal_nan=True,
            )
    finally:
        target.unlink(missing_ok=True)


# ----------------------------------------------------------------------- #
#  10. /api/meta parity with load_capture                                 #
# ----------------------------------------------------------------------- #


def test_meta_matches_load_capture(index: FrameIndex, reference) -> None:
    """/api/meta reports the same total_frames and num_subcarriers as load_capture."""
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    resp = client.get("/api/meta", params={"path": str(CAPTURE)})
    assert resp.status_code == 200

    body = resp.json()
    assert body["total_frames"] == len(reference)
    assert body["num_subcarriers"] == reference.num_subcarriers
    assert body["filename"] == CAPTURE.name
    assert body["bandwidth"] == reference.bandwidth


# ----------------------------------------------------------------------- #
#  11. Endpoint tests via TestClient                                      #
# ----------------------------------------------------------------------- #


def test_tile_endpoint_content_type_and_body(index: FrameIndex) -> None:
    """Correct Content-Type, body length = width * height * 4, headers parseable."""
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    width = 400
    t0 = float(index.times[0])
    t1 = float(index.times[-1]) + 1e-9

    resp = client.get("/api/tile", params={
        "path": str(CAPTURE),
        "t0": t0,
        "t1": t1,
        "width": width,
        "metric": "amplitude",
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"

    height = int(resp.headers["X-Tile-Height"])
    w = int(resp.headers["X-Tile-Width"])
    assert w == width
    assert height == index.num_subcarriers
    assert len(resp.content) == w * height * 4

    # All X-Tile-* headers present and parseable.
    for h in [
        "X-Tile-Width", "X-Tile-Height", "X-Capture-TMin", "X-Capture-TMax",
        "X-Tile-Frames", "X-Tile-Total", "X-Tile-Exact", "X-Tile-VMin",
        "X-Tile-VMax",
    ]:
        assert h in resp.headers, f"missing header {h}"
        float(resp.headers[h])  # parseable as float

    # Body is valid float32.
    arr = np.frombuffer(resp.content, dtype="<f4").reshape(height, w)
    assert arr.shape == (height, width)


def test_tile_endpoint_phase_metric(index: FrameIndex) -> None:
    """The /api/tile endpoint also serves phase tiles."""
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    t0 = float(index.times[10])
    t1 = float(index.times[50])

    resp = client.get("/api/tile", params={
        "path": str(CAPTURE),
        "t0": t0, "t1": t1, "width": 200, "metric": "phase",
    })
    assert resp.status_code == 200
    height = int(resp.headers["X-Tile-Height"])
    assert len(resp.content) == 200 * height * 4


def test_tile_endpoint_invalid_metric(index: FrameIndex) -> None:
    """An invalid metric returns 400."""
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    resp = client.get("/api/tile", params={
        "path": str(CAPTURE),
        "t0": 0.0, "t1": 1.0, "width": 100, "metric": "bogus",
    })
    assert resp.status_code == 400


def test_meta_endpoint_missing_path() -> None:
    """An empty path returns 400."""
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    resp = client.get("/api/meta", params={"path": ""})
    assert resp.status_code == 400


def test_tile_endpoint_not_found() -> None:
    """A non-existent path returns 404."""
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    resp = client.get("/api/tile", params={
        "path": "/nonexistent/file.dat",
        "t0": 0.0, "t1": 1.0, "width": 100, "metric": "amplitude",
    })
    assert resp.status_code == 404


# ----------------------------------------------------------------------- #
#  Closed-range boundary: the newest packet must survive                  #
# ----------------------------------------------------------------------- #


def test_frame_at_t1_is_in_range(index: FrameIndex) -> None:
    """t1 == times[-1] includes the final frame, not everything before it.

    /api/meta reports times[-1] as t_max, so this is the range a client sends
    for "the whole capture" and the one a follow-live window converges on. A
    half-open [t0, t1) drops exactly the newest packet -- the one a live view
    exists to show.
    """
    t0, t1 = _full_range(index)
    _, meta = compute_tile(CAPTURE, t0, t1, index.count, "amplitude")
    assert meta["total_in_range"] == index.count


def test_final_frame_reaches_the_last_column(index: FrameIndex) -> None:
    """The final frame contributes to the last column at any tile width.

    Selecting the frame is not enough: the per-column bucketing has its own
    right edge, and a half-open final bucket would drop the frame again after
    the range check let it through. Max-hold means the column is >= the frame
    everywhere, and equals it exactly where the frame is the maximum.
    """
    t0, t1 = _full_range(index)
    amp, _ = decode_frames(CAPTURE, index, np.arange(index.count))
    final = amp[-1]

    for width in (index.count, 800, 137, 1):
        grid, _ = compute_tile(CAPTURE, t0, t1, width, "amplitude")
        # grid row 0 is the highest subcarrier; flip back to source order.
        column = grid[:, -1][::-1]
        assert np.all(column >= final), f"final frame missing from last column at width={width}"

    # At one column per frame the last column holds exactly the last two
    # frames, so the max-hold is checkable against a closed-form expectation.
    grid, _ = compute_tile(CAPTURE, t0, t1, index.count, "amplitude")
    np.testing.assert_allclose(grid[:, -1][::-1], amp[-2:].max(axis=0), rtol=1e-6)


def test_frame_at_t0_is_in_range(index: FrameIndex) -> None:
    """The left edge is closed too -- the first frame is not dropped."""
    t0, t1 = _full_range(index)
    amp, _ = decode_frames(CAPTURE, index, np.arange(index.count))
    grid, _ = compute_tile(CAPTURE, t0, t1, index.count, "amplitude")
    assert np.all(grid[:, 0][::-1] >= amp[0])


def test_string_path_is_accepted(index: FrameIndex) -> None:
    """compute_tile coerces its path, and does not fork the index cache.

    The endpoint always passes a Path, but nothing in the signature enforces
    it, and a str would otherwise raise AttributeError on .stat() deep inside
    the decode -- and key a second, redundant FrameIndex in the cache.
    """
    t0, t1 = _full_range(index)
    from_path, _ = compute_tile(CAPTURE, t0, t1, 64, "amplitude")
    from_str, _ = compute_tile(str(CAPTURE), t0, t1, 64, "amplitude")
    _assert_grids_equal(from_str, from_path)


# ----------------------------------------------------------------------- #
#  Fixture: raw bytes for truncation test                                 #
# ----------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def raw() -> bytes:
    return CAPTURE.read_bytes()
