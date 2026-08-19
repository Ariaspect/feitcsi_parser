"""Tests for backend.tiles — pre-aggregated tile serving and the /api/tile, /api/meta endpoints."""

from __future__ import annotations

import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.batch import decode_frames
from backend.index import HEADER_BYTES, MAX_TICK, TICK_RESOLUTION, FrameIndex
from backend.parser import load_capture
from backend.tiles import (
    BLOCK_SIZE,
    TILE_FRAME_BUDGET,
    _base_with_context,
    _block_cache,
    _needs_reference,
    _ref_cache,
    compute_tile,
    get_index,
    get_reference,
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
    amp, _, _, _ = decode_frames(CAPTURE, index, all_ids)

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

    # Reproduce the sampling-gap fill: empty columns borrow the nearest
    # decoded frame's values when within the self-tuned gap_limit.  This
    # mirrors the fill in compute_tile exactly -- the bimodal timing of the
    # real capture means many columns receive no frame at width=frame_count,
    # and those within gap_limit are now filled rather than NaN.
    decoded_times = times
    empty = col_ends <= col_starts
    if len(decoded_times) >= 2:
        gap_limit = 2.0 * float(np.percentile(np.diff(decoded_times), 95))
    else:
        gap_limit = 0.0
    gap_limit = max(gap_limit, span / width)
    if len(decoded_times) > 0 and empty.any():
        centres = t0 + (np.arange(width) + 0.5) / width * span
        ec = centres[empty]
        j = np.searchsorted(decoded_times, ec)
        j_lo = np.clip(j - 1, 0, len(decoded_times) - 1)
        j_hi = np.clip(j, 0, len(decoded_times) - 1)
        d_lo = np.abs(decoded_times[j_lo] - ec)
        d_hi = np.abs(decoded_times[j_hi] - ec)
        nearest = np.where(d_lo <= d_hi, j_lo, j_hi)
        dist = np.minimum(d_lo, d_hi)
        ok = dist <= gap_limit
        cols = np.flatnonzero(empty)[ok]
        if cols.size > 0:
            expected[:, cols] = amp[nearest[ok]].T

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

    amp, _, _, _ = decode_frames(CAPTURE, index, np.array([10, 11]))
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

    _, phase, _, _ = decode_frames(CAPTURE, index, np.array([nearest]))
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
    amp, _, _, _ = decode_frames(CAPTURE, index, all_ids)

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

    amp, _, _, _ = decode_frames(CAPTURE, index, np.array([5]))

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
    # Width is capped at the number of frames in range (41: frames 10–50
    # inclusive) — never more columns than there are packets to fill.
    assert grid.shape == (index.num_subcarriers, meta["total_in_range"])


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
            amp_trunc, _, _, _ = decode_frames(target, trunc_idx, trunc_ids)
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
        "X-Tile-VMax", "X-Tile-PLow", "X-Tile-PHigh", "X-Tile-Filled",
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
    w = int(resp.headers["X-Tile-Width"])
    # Width is capped at total_in_range (41 frames in [times[10], times[50]]).
    assert w == 41
    assert len(resp.content) == w * height * 4


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
    amp, _, _, _ = decode_frames(CAPTURE, index, np.arange(index.count))
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
    amp, _, _, _ = decode_frames(CAPTURE, index, np.arange(index.count))
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
#  12. Robust percentile bounds in metadata                               #
# ----------------------------------------------------------------------- #


def test_tile_percentile_bounds(index: FrameIndex) -> None:
    """compute_tile returns p_low/p_high (1st/99th percentile of finite values).

    The raw min/max is dominated by outliers; percentile bounds are the robust
    scale the frontend locks to. -inf from db(0) must be excluded from the
    percentile computation, not clamped into it — clamping would drag p_low
    to -inf and defeat the purpose.
    """
    t0, t1 = _full_range(index)
    grid, meta = compute_tile(CAPTURE, t0, t1, 200, "amplitude")

    finite_mask = np.isfinite(grid)
    if finite_mask.any():
        finite_vals = grid[finite_mask]
        expected_low = float(np.nanpercentile(finite_vals, 1))
        expected_high = float(np.nanpercentile(finite_vals, 99))
        assert meta["p_low"] == pytest.approx(expected_low, rel=1e-5)
        assert meta["p_high"] == pytest.approx(expected_high, rel=1e-5)
        # Percentile bounds sit inside the finite extrema.
        assert meta["vmin"] <= meta["p_low"] <= meta["p_high"] <= meta["vmax"]
    else:
        assert meta["p_low"] == 0.0
        assert meta["p_high"] == 0.0


def test_tile_percentile_excludes_neg_inf(index: FrameIndex) -> None:
    """-inf from db(0) is excluded from the percentile, not clamped into it.

    If -inf were included, p_low would be -inf and the robust scale would be
    no better than the raw min. The finite mask excludes both NaN and -inf;
    the percentile is computed over that subset only.
    """
    t0, t1 = _full_range(index)
    grid, meta = compute_tile(CAPTURE, t0, t1, 200, "amplitude")

    if np.any(np.isneginf(grid)):
        # p_low must be finite — -inf was excluded from the computation.
        assert np.isfinite(meta["p_low"]), "p_low is -inf: -inf was not excluded"
        assert np.isfinite(meta["p_high"]), "p_high is -inf: -inf was not excluded"


def test_tile_endpoint_percentile_headers(index: FrameIndex) -> None:
    """X-Tile-PLow/X-Tile-PHigh headers are present and parseable."""
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    t0 = float(index.times[0])
    t1 = float(index.times[-1])
    resp = client.get("/api/tile", params={
        "path": str(CAPTURE),
        "t0": t0, "t1": t1, "width": 200, "metric": "amplitude",
    })
    assert resp.status_code == 200
    for h in ["X-Tile-PLow", "X-Tile-PHigh"]:
        assert h in resp.headers, f"missing header {h}"
        float(resp.headers[h])


# ----------------------------------------------------------------------- #
#  13. Tile width capped at frame count                                   #
# ----------------------------------------------------------------------- #


def test_tile_width_capped_at_frame_count(index: FrameIndex) -> None:
    """Tile width is capped at the number of frames in range.

    A full-extent request for more columns than packets returns a tile whose
    width equals the frame count — never more columns than there are packets
    to fill. At full extent on a 1101-packet capture, a 1230-wide request
    would otherwise produce 660 empty columns of 1230 (transparent stripes).
    """
    t0, t1 = _full_range(index)
    total = index.count

    # Request more columns than frames → capped.
    grid, meta = compute_tile(CAPTURE, t0, t1, total + 100, "amplitude")
    assert grid.shape == (index.num_subcarriers, total)
    assert meta["total_in_range"] == total

    # Request fewer columns than frames → no cap.
    narrow = max(1, total - 100)
    grid2, _ = compute_tile(CAPTURE, t0, t1, narrow, "amplitude")
    assert grid2.shape == (index.num_subcarriers, narrow)


def test_tile_width_capped_at_one_for_empty_range(index: FrameIndex) -> None:
    """An empty range caps width at 1, not 0.

    'Do not cap below 1' — a 0-wide tile would be a degenerate buffer the
    frontend cannot blit.
    """
    t0 = float(index.times[-1]) + 1.0
    t1 = t0 + 1.0
    grid, meta = compute_tile(CAPTURE, t0, t1, 400, "amplitude")
    assert grid.shape == (index.num_subcarriers, 1)
    assert meta["total_in_range"] == 0
    assert np.all(np.isnan(grid))


# ----------------------------------------------------------------------- #
#  Fixture: raw bytes for truncation test                                 #
# ----------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def raw() -> bytes:
    return CAPTURE.read_bytes()


# ----------------------------------------------------------------------- #
#  14. Sampling-gap fill: empty columns borrow nearest frame             #
# ----------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def template_frame(raw: bytes) -> bytes:
    """One frame's bytes from the real capture, as a template for synthetic captures."""
    csi_length = struct.unpack("I", raw[:4])[0]
    return raw[: HEADER_BYTES + csi_length]


def _write_synthetic_capture(
    path: Path, frame_deltas: list[float], template: bytes
) -> None:
    """Write a synthetic .dat file with the given inter-frame intervals.

    Reuses the real capture's frame bytes as a template (valid csi_length,
    num_sc, rate_flags, etc.) and rewrites only the ftm/mu clock fields to
    produce the requested timing.  The first frame is at t=0.0;
    *frame_deltas[i]* is the interval between frame i and frame i+1.
    """
    n = len(frame_deltas) + 1
    ftm = 0
    mu = 0
    frames: list[bytes] = []
    for i in range(n):
        frame = bytearray(template)
        struct.pack_into("<I", frame, 8, ftm)   # ftm_clock at offset 8
        struct.pack_into("<I", frame, 88, mu)   # mu_clock at offset 88
        frames.append(bytes(frame))
        if i < len(frame_deltas):
            dt = frame_deltas[i]
            ftm = (ftm + int(round(dt * 1e9 / TICK_RESOLUTION))) % (MAX_TICK + 1)
            mu = (mu + int(round(dt * 1e6))) % (MAX_TICK + 1)
    path.write_bytes(b"".join(frames))


def test_fill_closes_uniform_sampling_gaps(
    tmp_path: Path, template_frame: bytes
) -> None:
    """A capture whose burst cadence leaves sampling gaps (intervals wider
    than one column) is filled: no column is all-NaN, filled_columns > 0."""
    # 20 frames in 10 burst pairs: (0, 0.001), (0.2, 0.201), ...
    # Column width ≈ 0.09 s, burst cadence 0.2 s → every other column is empty.
    deltas: list[float] = []
    for i in range(10):
        deltas.append(0.001)        # intra-burst
        if i < 9:
            deltas.append(0.199)    # inter-burst
    # 19 deltas → 20 frames.

    path = tmp_path / "burst.dat"
    _write_synthetic_capture(path, deltas, template_frame)
    idx = FrameIndex(path)

    t0 = float(idx.times[0])
    t1 = float(idx.times[-1])
    grid, meta = compute_tile(path, t0, t1, 20, "amplitude")

    assert grid.shape == (idx.num_subcarriers, 20)
    # No column should be entirely NaN — all sampling gaps filled.
    assert not np.any(np.all(np.isnan(grid), axis=0)), "found an all-NaN column"
    assert meta["filled_columns"] > 0


def test_fill_preserves_genuine_dropout(
    tmp_path: Path, template_frame: bytes
) -> None:
    """A genuine multi-second dropout stays NaN: the gap is far wider than the
    self-tuned gap_limit, so columns inside it are not filled."""
    normal = 0.2
    huge = 50.0 * normal  # 10 s dropout
    n_side = 21  # frames on each side of the gap

    deltas: list[float] = []
    for _ in range(n_side - 1):
        deltas.append(normal)
    deltas.append(huge)
    for _ in range(n_side - 1):
        deltas.append(normal)
    # 41 deltas → 42 frames.

    path = tmp_path / "dropout.dat"
    _write_synthetic_capture(path, deltas, template_frame)
    idx = FrameIndex(path)

    t0 = float(idx.times[0])
    t1 = float(idx.times[-1])
    width = 42
    grid, meta = compute_tile(path, t0, t1, width, "amplitude")

    # Recompute gap_limit the same way the implementation does.
    decoded_times = idx.times  # full range, exact
    gap_limit = 2.0 * float(np.percentile(np.diff(decoded_times), 95))
    gap_limit = max(gap_limit, (t1 - t0) / width)

    # Find columns whose centre is deep inside the dropout (beyond gap_limit
    # from either edge frame) and assert they are all-NaN.
    span = t1 - t0
    centres = t0 + (np.arange(width) + 0.5) / width * span
    last_before = float(idx.times[n_side - 1])
    first_after = float(idx.times[n_side])
    deep = (centres > last_before + gap_limit) & (centres < first_after - gap_limit)
    assert deep.any(), "test misconfigured: no columns deep inside the dropout"
    for x in np.flatnonzero(deep):
        assert np.all(np.isnan(grid[:, x])), (
            f"column {x} inside dropout should be NaN but was filled"
        )
    # The dropout columns are not counted in filled_columns.
    assert meta["filled_columns"] == 0


def test_filled_column_carries_nearest_frame(
    tmp_path: Path, template_frame: bytes
) -> None:
    """A filled column carries the nearest decoded frame's values exactly."""
    # Same bimodal burst pattern as test_fill_closes_uniform_sampling_gaps.
    deltas: list[float] = []
    for i in range(10):
        deltas.append(0.001)
        if i < 9:
            deltas.append(0.199)

    path = tmp_path / "burst_exact.dat"
    _write_synthetic_capture(path, deltas, template_frame)
    idx = FrameIndex(path)

    t0 = float(idx.times[0])
    t1 = float(idx.times[-1])
    width = 20
    grid, meta = compute_tile(path, t0, t1, width, "amplitude")

    # Find an empty column that was filled.
    decoded_times = idx.times
    span = t1 - t0
    col_edges = t0 + np.arange(width + 1, dtype=np.float64) / width * span
    col_starts = np.searchsorted(decoded_times, col_edges[:-1], side="left")
    col_ends = np.searchsorted(decoded_times, col_edges[1:], side="left")
    col_ends[-1] = len(decoded_times)
    empty = col_ends <= col_starts
    # A filled column is empty (by bucketing) but not NaN (in the grid).
    col_all_nan = np.all(np.isnan(grid), axis=0)
    filled_cols = np.flatnonzero(empty & ~col_all_nan)
    assert filled_cols.size > 0, "expected at least one filled column"
    assert meta["filled_columns"] == filled_cols.size

    col = int(filled_cols[0])
    centre = t0 + (col + 0.5) / width * span
    nearest_frame = int(np.argmin(np.abs(decoded_times - centre)))

    # Decode that frame independently and compare.  The grid is row-flipped
    # (row 0 = highest subcarrier), so undo the flip before comparing.
    amp, _, _, _ = decode_frames(path, idx, np.array([nearest_frame]))
    np.testing.assert_array_equal(grid[:, col][::-1], amp[0])


def test_filled_columns_zero_when_no_gaps(
    tmp_path: Path, template_frame: bytes
) -> None:
    """When every column already has a frame, filled_columns == 0."""
    # Uniform 0.2 s cadence, 11 frames, width = 11.  Column width ≈ 0.182 s,
    # so each frame lands in its own column — no sampling gaps to fill.
    n = 11
    deltas = [0.2] * (n - 1)
    path = tmp_path / "uniform.dat"
    _write_synthetic_capture(path, deltas, template_frame)
    idx = FrameIndex(path)

    t0 = float(idx.times[0])
    t1 = float(idx.times[-1])
    grid, meta = compute_tile(path, t0, t1, n, "amplitude")
    assert meta["filled_columns"] == 0
    # And no column is NaN.
    assert not np.any(np.all(np.isnan(grid), axis=0))


def test_fill_handles_zero_and_one_decoded(index: FrameIndex) -> None:
    """n_decoded == 0 and n_decoded == 1 do not crash; filled_columns == 0."""
    # n_decoded == 0: range entirely outside the capture.
    t0 = float(index.times[-1]) + 1.0
    t1 = t0 + 1.0
    grid0, meta0 = compute_tile(CAPTURE, t0, t1, 10, "amplitude")
    assert meta0["filled_columns"] == 0
    assert np.all(np.isnan(grid0))

    # n_decoded == 1: a tiny range around a single frame.  Width is capped at 1.
    t0 = float(index.times[5])
    t1 = float(index.times[5]) + 1e-9
    grid1, meta1 = compute_tile(CAPTURE, t0, t1, 10, "amplitude")
    assert meta1["filled_columns"] == 0
    assert grid1.shape == (index.num_subcarriers, 1)
    assert not np.all(np.isnan(grid1))



# ----------------------------------------------------------------------- #
#  Derived phase metrics (unwrap / detrend)                               #
# ----------------------------------------------------------------------- #


def test_derived_metrics_are_served(index: FrameIndex) -> None:
    """Every derived metric produces a grid of the same shape as its base."""
    from backend.tiles import DERIVED_METRICS

    t0, t1 = float(index.times[0]), float(index.times[0]) + 1.0
    for metric, spec in DERIVED_METRICS.items():
        grid, _ = compute_tile(CAPTURE, t0, t1, 64, metric)
        base_grid, _ = compute_tile(CAPTURE, t0, t1, 64, spec.bases[0])
        assert grid.shape == base_grid.shape, metric
        if spec.preserves_coverage:
            # Same cells have data; the transform must not create or
            # destroy coverage, only change the values inside it.
            assert np.array_equal(np.isnan(grid), np.isnan(base_grid)), metric
        else:
            # A domain-changing transform (e.g. subcarrier -> delay tap)
            # has no per-cell correspondence to preserve, but a column
            # (frame) the base had no ratio for must still be empty here.
            base_has_data = ~np.all(np.isnan(base_grid), axis=0)
            grid_has_data = ~np.all(np.isnan(grid), axis=0)
            assert np.array_equal(base_has_data, grid_has_data), metric


def test_derived_metric_matches_transform_of_decoded_frames(index: FrameIndex) -> None:
    """The tile path and a direct transform of decoded frames agree.

    Guards the ordering rule: the transform runs per frame on full subcarrier
    vectors before column aggregation. Transforming after aggregation would
    unwrap across frames rather than across subcarriers and diverge here.
    """
    from backend.phase import unwrap_subcarrier

    # A window small enough that every frame in it is decoded exactly.
    t0 = float(index.times[0])
    t1 = float(index.times[min(50, index.count - 1)])
    grid, meta = compute_tile(CAPTURE, t0, t1, 8, "phase_unwrapped")
    assert meta["exact"]

    lo = int(np.searchsorted(index.times, t0, side="left"))
    hi = int(np.searchsorted(index.times, t1, side="right"))
    _, phase, _, _ = decode_frames(CAPTURE, index, np.arange(lo, hi))
    expected = unwrap_subcarrier(phase)

    # Each column holds one frame verbatim (nearest-frame aggregation), so
    # every column must appear among the transformed frames.
    for x in range(grid.shape[1]):
        col = grid[::-1, x]  # undo the row flip applied by compute_tile
        if np.isnan(col).all():
            continue
        assert np.isclose(expected, col, atol=1e-4, equal_nan=True).all(axis=1).any()


def test_derived_metric_reuses_the_base_block_decode(index: FrameIndex) -> None:
    """Asking for a derived metric after its base decodes no extra frames."""
    reset_tile_caches()
    t0, t1 = float(index.times[0]), float(index.times[0]) + 1.0

    compute_tile(CAPTURE, t0, t1, 64, "phase")
    after_base = _block_cache.frames_decoded
    assert after_base > 0

    compute_tile(CAPTURE, t0, t1, 64, "phase_unwrapped")
    compute_tile(CAPTURE, t0, t1, 64, "phase_detrended")
    assert _block_cache.frames_decoded == after_base


def test_derived_metric_is_not_computed_until_requested(index: FrameIndex) -> None:
    """A block decode must not populate derived entries nobody asked for."""
    reset_tile_caches()
    t0, t1 = float(index.times[0]), float(index.times[0]) + 1.0
    compute_tile(CAPTURE, t0, t1, 64, "phase")

    file_size = CAPTURE.stat().st_size
    keys = {k[1] for k in _block_cache._entries}
    assert "phase" in keys
    assert "phase_unwrapped" not in keys
    assert "phase_detrended" not in keys
    del file_size


def test_derived_metrics_survive_the_sampled_path(index: FrameIndex) -> None:
    """The stride-sampled branch (filtered/over-budget) also derives correctly."""
    t0, t1 = float(index.times[0]), float(index.times[-1])
    grid, meta = compute_tile(CAPTURE, t0, t1, 128, "phase_detrended", mimo=(2, 1))
    finite = grid[np.isfinite(grid)]
    if finite.size:
        # Detrended phase is centred by construction: the fit removes the mean.
        assert abs(float(np.nanmean(finite))) < 1.0


def test_tile_endpoint_serves_derived_metrics(index: FrameIndex) -> None:
    """/api/tile accepts the new metric names and returns a float32 grid."""
    from fastapi.testclient import TestClient
    from backend.app import app
    from backend.tiles import DERIVED_METRICS

    client = TestClient(app)
    for metric in DERIVED_METRICS:
        resp = client.get("/api/tile", params={
            "path": str(CAPTURE),
            "t0": float(index.times[0]), "t1": float(index.times[0]) + 1.0,
            "width": 32, "metric": metric,
        })
        assert resp.status_code == 200, (metric, resp.text)
        w = int(resp.headers["X-Tile-Width"])
        h = int(resp.headers["X-Tile-Height"])
        assert len(resp.content) == w * h * 4


# ----------------------------------------------------------------------- #
#  Orientation reference wiring                                           #
# ----------------------------------------------------------------------- #


def test_needs_reference_follows_the_derivation_chain():
    """The time-unwrapped ratio corrects nothing itself, but is built on one."""
    assert _needs_reference("csi_ratio_phase_corrected")
    assert _needs_reference("csi_ratio_amplitude_corrected")
    assert _needs_reference("csi_ratio_phase_time_unwrapped")
    assert not _needs_reference("csi_ratio_phase_unwrapped")
    assert not _needs_reference("phase_detrended")
    assert not _needs_reference("amplitude")


def test_reference_is_built_once_per_capture_and_filter():
    reset_tile_caches()
    index = get_index(CAPTURE)
    size = CAPTURE.stat().st_size
    first = get_reference(CAPTURE, index, size)
    second = get_reference(CAPTURE, index, size)
    assert first is second or (first is None and second is None)

    # A different filter is a different transmitter, so a different reference.
    mac = index.source_macs[0]
    other = get_reference(CAPTURE, index, size, source_mac=mac)
    assert (str(CAPTURE), size, None, mac) in _ref_cache

    reset_tile_caches()
    assert not _ref_cache
    assert other is None or other is not None  # only that the call succeeded


def test_tiles_report_whether_they_were_anchored():
    index = get_index(CAPTURE)
    t0, t1 = float(index.times[0]), float(index.times[-1])

    _, meta = compute_tile(CAPTURE, t0, t1, 64, "amplitude")
    # A metric that needs no reference is trivially anchored.
    assert meta["anchored"] is True

    _, meta = compute_tile(CAPTURE, t0, t1, 64, "csi_ratio_phase_corrected")
    assert isinstance(meta["anchored"], bool)


def test_corrected_tiles_are_stable_across_pan_and_zoom():
    """The regression, at the tile level: same frames, different views.

    Two views of the same instant rarely pick the same frame for a column —
    a wider view's column spans more frames — and raw wrapped phase differs
    between frames anyway, so comparing columns naively measures the sampling
    as much as the correction. The uncorrected metric settles it: where two
    views agree on the raw column they resolved the same frame, and any
    disagreement in the corrected column there is the correction's alone.
    """
    index = get_index(CAPTURE)
    macs, counts = np.unique(np.asarray(index.source_macs), return_counts=True)
    mac = str(macs[np.argmax(counts)])
    times = np.asarray(index.times)[np.asarray(index.source_macs) == mac]
    if len(times) < 100:
        pytest.skip("too few frames from one transmitter to pan across")

    lo, hi = len(times) // 4, len(times) // 2
    t0, t1 = float(times[lo]), float(times[hi])
    width = hi - lo
    span = t1 - t0

    def view(a, b, w):
        raw, _ = compute_tile(CAPTURE, a, b, w, "csi_ratio_phase", source_mac=mac)
        fix, meta = compute_tile(
            CAPTURE, a, b, w, "csi_ratio_phase_corrected", source_mac=mac
        )
        return raw, fix, meta

    base_raw, base_fix, base_meta = view(t0, t1, width)
    if not base_meta["anchored"]:
        pytest.skip("capture supports no absolute orientation")

    compared = 0
    for pad in (0.25, 0.5, 1.0):
        a = max(float(times[0]), t0 - pad * span)
        b = min(float(times[-1]), t1 + pad * span)
        w = max(2, int(width * (b - a) / span))
        raw, fix, _ = view(a, b, w)
        s0 = int(round((t0 - a) / (b - a) * w))
        e0 = int(round((t1 - a) / (b - a) * w))
        if e0 - s0 < 2:
            continue
        picks = np.linspace(0, width - 1, e0 - s0).astype(int)

        for k, col in enumerate(range(s0, e0)):
            mine, theirs = base_raw[:, picks[k]], raw[:, col]
            ok = np.isfinite(mine) & np.isfinite(theirs)
            if ok.sum() < 8 or not np.allclose(mine[ok], theirs[ok], atol=1e-4):
                continue  # different frames; nothing to say
            d = np.angle(
                np.exp(1j * (base_fix[:, picks[k]][ok] - fix[:, col][ok]))
            )
            compared += 1
            # Same frame, so the correction must have reached the same verdict.
            assert np.median(np.abs(d)) < 1e-3, (pad, col)

    assert compared > 0, "no column resolved to the same frame in both views"


def test_block_context_spans_the_neighbouring_blocks():
    reset_tile_caches()
    index = get_index(CAPTURE)
    size = CAPTURE.stat().st_size
    n_blocks = max(1, -(-index.count // BLOCK_SIZE))
    if n_blocks < 2:
        pytest.skip("capture is a single block")
    got = _base_with_context(
        CAPTURE, index, 1, "csi_ratio_phase", size, 32, 16, None
    )
    start, stop = BLOCK_SIZE, min(2 * BLOCK_SIZE, index.count)
    assert len(got) == 32 + (stop - start) + 16


def _assert_same_angle(a: np.ndarray, b: np.ndarray) -> None:
    """Equal up to the branch cut.

    The corrector re-wraps with ``angle(exp(i x))``, which sends exactly -pi
    to +pi. It is the same angle and the same colour on a cyclic map, but a
    raw comparison calls it a 2*pi miss.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    d = np.angle(np.exp(1j * (a[ok].astype(np.float64) - b[ok].astype(np.float64))))
    np.testing.assert_allclose(d, 0.0, atol=1e-5)


def test_no_reference_without_a_selected_transmitter():
    """Correction is per transmitter, so `all` gets no reference at all."""
    reset_tile_caches()
    index = get_index(CAPTURE)
    size = CAPTURE.stat().st_size
    assert get_reference(CAPTURE, index, size) is None
    assert get_reference(CAPTURE, index, size, source_mac="") is None
    assert get_reference(CAPTURE, index, size, mimo=(2, 1)) is None
    # Nothing was cached: there was nothing to build.
    assert not _ref_cache


def test_all_senders_leaves_the_ratio_exactly_as_decoded():
    """On `all`, skip the correction rather than apply a near-no-op version.

    Interleaved, _chain compares two different senders 86-93% of the time and
    the confidence gate declines, so the correction achieves almost nothing —
    but it still reported itself as done. Now the panel shows the raw ratio
    and says so.
    """
    index = get_index(CAPTURE)
    t0, t1 = float(index.times[0]), float(index.times[-1])

    for corrected, raw in (
        ("csi_ratio_phase_corrected", "csi_ratio_phase"),
        ("csi_ratio_amplitude_corrected", "csi_ratio_amplitude"),
    ):
        fixed, meta = compute_tile(CAPTURE, t0, t1, 128, corrected)
        plain, _ = compute_tile(CAPTURE, t0, t1, 128, raw)
        assert meta["anchored"] is False, corrected
        _assert_same_angle(fixed, plain)


def test_a_selected_transmitter_is_corrected_and_says_so():
    index = get_index(CAPTURE)
    macs, counts = np.unique(np.asarray(index.source_macs), return_counts=True)
    mac = str(macs[np.argmax(counts)])
    t0, t1 = float(index.times[0]), float(index.times[-1])

    _, meta = compute_tile(
        CAPTURE, t0, t1, 128, "csi_ratio_phase_corrected", source_mac=mac
    )
    ref = get_reference(CAPTURE, index, CAPTURE.stat().st_size, source_mac=mac)
    # The tile's claim and the reference's existence are the same fact.
    assert meta["anchored"] is (ref is not None)


def test_unanchored_tiles_do_not_run_the_batch_relative_fallback():
    """Falling back to the old per-batch answer is what made views disagree."""
    index = get_index(CAPTURE)
    times = index.times
    t0, t1 = float(times[0]), float(times[-1])
    mid = (t0 + t1) / 2

    whole, m1 = compute_tile(CAPTURE, t0, t1, 64, "csi_ratio_phase_corrected")
    half, m2 = compute_tile(CAPTURE, t0, mid, 32, "csi_ratio_phase_corrected")
    assert m1["anchored"] is False and m2["anchored"] is False
    # Both are the raw ratio, so neither can be a pi from the other.
    raw_whole, _ = compute_tile(CAPTURE, t0, t1, 64, "csi_ratio_phase")
    raw_half, _ = compute_tile(CAPTURE, t0, mid, 32, "csi_ratio_phase")
    _assert_same_angle(whole, raw_whole)
    _assert_same_angle(half, raw_half)
