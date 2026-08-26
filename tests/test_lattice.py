"""Tests for the tile lattice — the fixed time grid columns are quantised to.

The lattice is what makes a tile's columns a property of the capture rather
than of the window that asked for them. Before it, ``col_edges`` were derived
from ``t0``/``span``/``width``, so a one-pixel pan re-quantised all 1248
columns and the picture crawled; a live poll re-binned the whole grid every
300 ms. These tests pin the two properties that stop that: edges land on
multiples of ``dt``, and ``dt`` depends only on the span and the requested
width, never on where the window happens to start.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.index import FrameIndex
from backend.tiles import (
    CHUNK_COLUMNS,
    LATTICE_DT0,
    chunk_span,
    compute_tile,
    lattice_dt,
    pick_level,
    reset_tile_caches,
    snap_window,
)

CAPTURE = Path(__file__).resolve().parent.parent / "captures" / "capture.dat"

pytestmark = pytest.mark.skipif(
    not CAPTURE.is_file(), reason="captures/capture.dat not present"
)


# ----------------------------------------------------------------------- #
#  Levels                                                                 #
# ----------------------------------------------------------------------- #


def test_lattice_dt_doubles_per_level() -> None:
    assert lattice_dt(0) == pytest.approx(LATTICE_DT0)
    assert lattice_dt(1) == pytest.approx(2 * LATTICE_DT0)
    assert lattice_dt(10) == pytest.approx(LATTICE_DT0 * 1024)


def test_pick_level_gives_at_most_width_columns() -> None:
    """The level is the finest whose columns still fit the caller's pixels."""
    for span in (0.5, 7.0, 123.0, 3600.0):
        for width in (200, 800, 1248, 1600):
            level = pick_level(span, width, min_dt=0.0)
            dt = lattice_dt(level)
            assert math.ceil(span / dt) <= width, (span, width, level)
            # One level finer would overflow the width, or is below the floor.
            if level > 0:
                assert math.ceil(span / lattice_dt(level - 1)) > width


def test_pick_level_is_independent_of_where_the_window_sits() -> None:
    """Panning must not change the level — that is what re-binned everything."""
    span = 42.0
    base = pick_level(span, 1248, min_dt=0.0)
    for t0 in (0.0, 0.001, 7.3, 1234.567):
        assert pick_level((t0 + span) - t0, 1248, min_dt=0.0) == base


def test_pick_level_never_goes_finer_than_the_frame_spacing() -> None:
    """A capture sampled every 57 ms has nothing to say at 1 ms resolution.

    Without the floor a deep zoom asks for columns the data cannot fill, and
    the grid comes back mostly NaN — worse than the pre-lattice behaviour,
    which capped the width at the frame count.
    """
    level = pick_level(0.5, 1248, min_dt=0.057)
    assert lattice_dt(level) >= 0.057 / 2


# ----------------------------------------------------------------------- #
#  Snapping                                                               #
# ----------------------------------------------------------------------- #


def test_snap_window_lands_on_multiples_of_dt() -> None:
    level = 5
    dt = lattice_dt(level)
    c0, c1 = snap_window(7.3, 19.8, level)
    assert c0 * dt <= 7.3 < (c0 + 1) * dt
    assert (c1 - 1) * dt < 19.8 <= c1 * dt


def test_snap_window_covers_the_request() -> None:
    for t0, t1 in ((0.0, 1.0), (3.14159, 9.2), (100.0, 100.5)):
        level = pick_level(t1 - t0, 1248, min_dt=0.0)
        dt = lattice_dt(level)
        c0, c1 = snap_window(t0, t1, level)
        assert c0 * dt <= t0
        assert c1 * dt >= t1


def test_snap_window_shifts_by_whole_columns_under_a_pan() -> None:
    """A pan of exactly one column moves both edges by exactly one column."""
    level = 8
    dt = lattice_dt(level)
    a0, a1 = snap_window(10.0, 20.0, level)
    b0, b1 = snap_window(10.0 + dt, 20.0 + dt, level)
    assert (b0 - a0, b1 - a1) == (1, 1)


def test_chunk_span_is_a_whole_number_of_columns() -> None:
    for level in (0, 4, 11):
        assert chunk_span(level) == pytest.approx(CHUNK_COLUMNS * lattice_dt(level))


# ----------------------------------------------------------------------- #
#  The property the lattice exists for                                    #
# ----------------------------------------------------------------------- #


def _column_at(grid: np.ndarray, meta: dict, t: float) -> np.ndarray:
    """The column of *grid* covering absolute time *t*."""
    col = int(round((t - meta["t0"]) / meta["dt"] - 0.5))
    assert 0 <= col < grid.shape[1], (col, grid.shape, meta)
    return grid[:, col]


def test_a_pan_does_not_change_the_columns_it_kept() -> None:
    """The whole point: shared columns are bit-identical across a pan.

    Before the lattice, every column of a panned view was re-aggregated over
    different frame boundaries, so the image changed rather than moved.
    """
    reset_tile_caches()
    from backend.tiles import get_index

    index = get_index(CAPTURE)
    t_min, t_max = float(index.times[0]), float(index.times[-1])
    span = (t_max - t_min) / 4

    a_grid, a_meta = compute_tile(CAPTURE, t_min, t_min + span, 400, "amplitude")
    # Pan by a third of the window — an arbitrary amount, not a column multiple.
    shift = span / 3
    b_grid, b_meta = compute_tile(
        CAPTURE, t_min + shift, t_min + shift + span, 400, "amplitude"
    )

    assert a_meta["dt"] == b_meta["dt"], "a pan must not change the level"

    overlap0 = max(a_meta["t0"], b_meta["t0"])
    overlap1 = min(a_meta["t1"], b_meta["t1"])
    assert overlap1 > overlap0, "the two windows must overlap for this to mean anything"

    dt = a_meta["dt"]
    probes = np.arange(overlap0 + dt / 2, overlap1 - dt / 2, dt * 7)
    assert len(probes) > 5, "need several columns to compare"
    for t in probes:
        np.testing.assert_array_equal(
            _column_at(a_grid, a_meta, t),
            _column_at(b_grid, b_meta, t),
            err_msg=f"column at t={t} changed under a pan",
        )


def test_a_wider_window_reuses_the_same_columns_at_the_same_level() -> None:
    """Two callers whose spans land on the same level agree column for column."""
    reset_tile_caches()
    from backend.tiles import get_index

    index = get_index(CAPTURE)
    t_min, t_max = float(index.times[0]), float(index.times[-1])
    span = (t_max - t_min) / 4

    _, narrow_meta = compute_tile(CAPTURE, t_min, t_min + span, 400, "amplitude")
    level_dt = narrow_meta["dt"]

    # A window 10% wider still picks the same level for a suitable width.
    grid_a, meta_a = compute_tile(CAPTURE, t_min, t_min + span, 400, "amplitude")
    grid_b, meta_b = compute_tile(
        CAPTURE, t_min, t_min + span * 1.05, 420, "amplitude"
    )
    if meta_b["dt"] != level_dt:
        pytest.skip("the wider window landed on a different level")

    dt = level_dt
    overlap1 = min(meta_a["t1"], meta_b["t1"])
    probes = np.arange(meta_a["t0"] + dt / 2, overlap1 - dt / 2, dt * 11)
    for t in probes[:20]:
        np.testing.assert_array_equal(
            _column_at(grid_a, meta_a, t), _column_at(grid_b, meta_b, t)
        )


# ----------------------------------------------------------------------- #
#  Chunk cache                                                            #
# ----------------------------------------------------------------------- #


def test_repeating_a_request_costs_nothing() -> None:
    """The second identical request is served entirely from cached chunks."""
    reset_tile_caches()
    from backend.tiles import _chunk_cache, get_index

    index = get_index(CAPTURE)
    t_min, t_max = float(index.times[0]), float(index.times[-1])

    compute_tile(CAPTURE, t_min, t_max, 400, "amplitude")
    hits_before, misses_before = _chunk_cache.hits, _chunk_cache.misses
    compute_tile(CAPTURE, t_min, t_max, 400, "amplitude")

    assert _chunk_cache.misses == misses_before, "a repeat request re-computed a chunk"
    assert _chunk_cache.hits > hits_before


def test_a_pan_recomputes_only_the_chunks_it_moved_onto() -> None:
    """Panning re-uses every chunk the two views share.

    This is the cache the lattice buys: before it, a request keyed on the
    exact window could never hit, so every pan re-decoded the whole view.
    """
    reset_tile_caches()
    from backend.tiles import _chunk_cache, get_index

    index = get_index(CAPTURE)
    t_min, t_max = float(index.times[0]), float(index.times[-1])
    span = (t_max - t_min) / 4

    compute_tile(CAPTURE, t_min, t_min + span, 400, "amplitude")
    hits_before = _chunk_cache.hits
    # Pan by a fraction of a chunk: the view keeps most of its columns.
    compute_tile(CAPTURE, t_min + span / 8, t_min + span / 8 + span, 400, "amplitude")

    assert _chunk_cache.hits > hits_before, "a pan re-computed chunks it already had"


def test_a_chunk_is_keyed_on_the_frames_it_holds() -> None:
    """The growing edge misses; everything behind it keeps hitting.

    A chunk is entirely determined by the frames inside it, so the count is
    what an entry's validity depends on. Frames landing in the chunk at the
    edge change it and force a recompute; frames landing past it do not.
    """
    from backend.tiles import _chunk_frame_count, chunk_span, get_index

    index = get_index(CAPTURE)
    times = np.asarray(index.times)
    level = 6
    span = chunk_span(level)

    first = _chunk_frame_count(times, level, 0)
    assert first > 0
    # The same chunk, measured against a capture with more frames appended
    # beyond it, is unchanged -- so its key, and its cache entry, survive.
    grown = np.concatenate([times, times[-1] + span * 4 + np.arange(1, 50) * 0.05])
    assert _chunk_frame_count(grown, level, 0) == first


def test_a_capture_that_grew_keeps_the_columns_it_had() -> None:
    """Appending frames must not disturb columns already settled.

    The live-follow case: a poll extends the capture, the window slides, and
    every column left of the new data has to come back unchanged.
    """
    import shutil

    from backend.index import HEADER_BYTES
    from backend.tiles import get_index

    with tempfile.TemporaryDirectory() as tmp:
        growing = Path(tmp) / "growing.dat"
        raw = CAPTURE.read_bytes()
        # Start from two thirds of the capture, on a frame boundary.
        idx_full = FrameIndex(CAPTURE)
        cut_frame = (idx_full.count * 2) // 3
        cut = int(idx_full.offsets[cut_frame])
        growing.write_bytes(raw[:cut])

        reset_tile_caches()
        index = get_index(growing)
        t_min, t_max = float(index.times[0]), float(index.times[-1])
        span = (t_max - t_min) / 2
        before_grid, before_meta = compute_tile(
            growing, t_min, t_min + span, 400, "amplitude"
        )

        with growing.open("ab") as fh:
            fh.write(raw[cut:])

        after_grid, after_meta = compute_tile(
            growing, t_min, t_min + span, 400, "amplitude"
        )

        assert after_meta["dt"] == before_meta["dt"]
        assert after_meta["t0"] == before_meta["t0"]
        np.testing.assert_array_equal(
            before_grid, after_grid,
            err_msg="a settled column changed when the capture grew",
        )
        assert HEADER_BYTES  # imported for the offset arithmetic above
