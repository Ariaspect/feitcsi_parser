"""Pre-aggregated tile serving for offline capture exploration.

Builds display-resolution grids from arbitrary time ranges of a FeitCSI
capture. Cost per request is bounded: at most ``TILE_FRAME_BUDGET`` frames are
decoded, and the output grid has exactly ``width * num_subcarriers`` cells
regardless of how much of the file the view covers.

Two caches keep repeated work cheap:

* ``FrameIndex`` objects are cached per path (``extend()`` on each request so a
  growing capture stays current without a full rescan).
* Decoded blocks (contiguous runs of ``BLOCK_SIZE`` frames) are cached in an
  LRU keyed by ``(path, metric, block_index, file_size)``.  Including file size
  means a rewritten or truncated file cannot serve stale blocks.

Thread safety: FastAPI serves handlers from a threadpool, so both caches are
guarded with locks.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .batch import decode_frames
from .index import FrameIndex

# Maximum frames decoded per /api/tile request. When the requested time range
# holds more frames than this, stride-sample approximately BUDGET frames evenly
# across the range and mark the tile as sampled (X-Tile-Exact: 0).
#
# A sampled max-hold can miss transients that a full decode would catch —
# that is the price of a bounded request.  X-Tile-Exact lets the UI say so
# rather than silently lying.  Zooming in shrinks the range until it fits the
# budget and the tile becomes exact; that is the intended interaction.
TILE_FRAME_BUDGET = 8192

# Decode blocks are cached at this granularity (contiguous runs of frames
# aligned to BLOCK_SIZE boundaries).  A block is decoded in full on first
# request and reused on subsequent requests touching the same frames.
BLOCK_SIZE = 4096

# Upper bound on total bytes held in the block cache.
CACHE_MAX_BYTES = 256 * 1024 * 1024  # 256 MB


# ----------------------------------------------------------------------- #
#  Block cache                                                            #
# ----------------------------------------------------------------------- #


class _BlockCache:
    """LRU cache of decoded blocks, bounded by total bytes.

    Stores one metric per entry (key includes metric).  When a block is
    decoded, both amplitude and phase are cached under their respective keys
    so a later request for the other metric hits.
    """

    def __init__(self, max_bytes: int = CACHE_MAX_BYTES) -> None:
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0
        # Cumulative count of frames decoded on cache misses.  Reset by tests
        # via ``reset_tile_caches()``.
        self.frames_decoded = 0

    def get(self, key: tuple) -> np.ndarray | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry

    def put(self, key: tuple, block: np.ndarray) -> None:
        with self._lock:
            old = self._entries.get(key)
            if old is not None:
                self._bytes -= old.nbytes
            self._entries[key] = block
            self._bytes += block.nbytes
            self._entries.move_to_end(key)
            self._evict_locked()

    def _evict_locked(self) -> None:
        while self._bytes > self._max_bytes and self._entries:
            _, old = self._entries.popitem(last=False)
            self._bytes -= old.nbytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            self.frames_decoded = 0


_block_cache = _BlockCache()


# ----------------------------------------------------------------------- #
#  FrameIndex cache                                                       #
# ----------------------------------------------------------------------- #

_index_cache: dict[Path, FrameIndex] = {}
_index_lock = threading.Lock()


def get_index(path: Path) -> FrameIndex:
    """Return the shared FrameIndex for *path*, extending it if it exists.

    Mirrors the ``get_stream`` registry pattern in ``backend.stream``: the
    first call builds a full FrameIndex, subsequent calls call ``extend()``
    to pick up appended frames without a full rescan.  Truncation triggers a
    rebuild inside ``extend()``.
    """
    path = Path(path)
    with _index_lock:
        idx = _index_cache.get(path)
        if idx is None:
            idx = FrameIndex(path)
            _index_cache[path] = idx
        else:
            idx.extend()
        return idx


def reset_tile_caches() -> None:
    """Drop all cached FrameIndexes and decoded blocks.  For tests."""
    with _index_lock:
        _index_cache.clear()
    _block_cache.clear()


# ----------------------------------------------------------------------- #
#  Block-level decode with caching                                        #
# ----------------------------------------------------------------------- #


def _decode_block_cached(
    path: Path,
    index: FrameIndex,
    block_idx: int,
    metric: str,
    file_size: int,
) -> np.ndarray:
    """Return the decoded block for one metric, from cache or by decoding.

    On a cache miss, decodes the full block (both amplitude and phase) and
    caches both under their respective keys, so a later request for the other
    metric hits.
    """
    key = (str(path), metric, block_idx, file_size)
    cached = _block_cache.get(key)
    if cached is not None:
        return cached

    block_start = block_idx * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, index.count)
    block_ids = np.arange(block_start, block_end)
    amp, phase = decode_frames(path, index, block_ids)

    # Cache both metrics so a subsequent request for the other one hits.
    _block_cache.put((str(path), "amplitude", block_idx, file_size), amp)
    _block_cache.put((str(path), "phase", block_idx, file_size), phase)
    with _block_cache._lock:
        _block_cache.frames_decoded += len(block_ids)

    return amp if metric == "amplitude" else phase


def _decode_via_blocks(
    path: Path,
    index: FrameIndex,
    frame_ids: np.ndarray,
    metric: str,
    file_size: int,
) -> np.ndarray:
    """Decode a contiguous range of frames through the block cache.

    *frame_ids* must be a contiguous ascending run.  Each block overlapping
    the range is decoded in full (or fetched from cache); the requested
    frames are then sliced out.
    """
    n = len(frame_ids)
    num_sc = index.num_subcarriers
    out = np.empty((n, num_sc), dtype=np.float32)

    pos = 0
    while pos < n:
        fid = int(frame_ids[pos])
        block_idx = fid // BLOCK_SIZE
        block_start = block_idx * BLOCK_SIZE

        block = _decode_block_cached(path, index, block_idx, metric, file_size)

        # How many of our frame_ids fall in this block?
        block_end = min(block_start + BLOCK_SIZE, index.count)
        avail = block_end - fid
        take = min(avail, n - pos)

        local_start = fid - block_start
        out[pos : pos + take] = block[local_start : local_start + take]
        pos += take

    return out


# ----------------------------------------------------------------------- #
#  Tile computation                                                       #
# ----------------------------------------------------------------------- #


def compute_tile(
    path: Path,
    t0: float,
    t1: float,
    width: int,
    metric: str,
) -> tuple[np.ndarray, dict]:
    """Build a display-resolution grid for the requested time range.

    Returns ``(grid, metadata)`` where *grid* has shape
    ``(num_subcarriers, width)``, float32, row-major with row 0 = highest
    subcarrier index (matching the frontend's ``subcarrierSourceRect``
    convention).  Empty columns are NaN; ``-inf`` from ``db(0)`` is preserved.

    *metadata* keys: ``frames_decoded``, ``total_in_range``, ``exact``,
    ``vmin``, ``vmax``, ``t_min``, ``t_max``.
    """
    path = Path(path)
    index = get_index(path)
    num_sc = index.num_subcarriers
    times = index.times

    # Clamp width.
    width = max(1, min(width, 4096))

    # Guard against empty capture or invalid range.
    if index.count == 0 or t1 <= t0:
        grid = np.full((max(num_sc, 0), width), np.nan, dtype=np.float32)
        return grid, {
            "frames_decoded": 0,
            "total_in_range": 0,
            "exact": True,
            "vmin": 0.0,
            "vmax": 0.0,
            "t_min": float(times[0]) if len(times) else 0.0,
            "t_max": float(times[-1]) if len(times) else 0.0,
        }

    # Find frames in [t0, t1] -- CLOSED at both ends.
    #
    # The obvious half-open [t0, t1) drops a frame sitting exactly on t1, and
    # t1 == times[-1] is the single most common request there is: /api/meta
    # reports the last timestamp as t_max, so "show me the whole capture" and
    # every follow-live window land precisely on it.  The newest packet -- the
    # one a live view exists to show -- would be the one silently missing.
    # renderToImageData already special-cases its final column for the same
    # reason; the two must agree or live and explore render the same data
    # differently.
    lo = int(np.searchsorted(times, t0, side="left"))
    hi = int(np.searchsorted(times, t1, side="right"))
    total_in_range = hi - lo

    # Stride-sample if the range exceeds the budget.
    if total_in_range > TILE_FRAME_BUDGET:
        sampled = np.linspace(
            0, total_in_range - 1, TILE_FRAME_BUDGET, dtype=np.int64
        )
        frame_ids = np.arange(lo, hi)[sampled]
        exact = False
    else:
        frame_ids = np.arange(lo, hi)
        exact = True

    n_decoded = len(frame_ids)

    # Decode.
    file_size = path.stat().st_size
    if n_decoded == 0:
        data = np.empty((0, num_sc), dtype=np.float32)
    elif exact:
        # Exact range: decode through the block cache so scrubbing back and
        # forth over the same region hits the cache.
        data = _decode_via_blocks(path, index, frame_ids, metric, file_size)
    else:
        # Sampled: decode the strided frame_ids directly.  These are sparse
        # (not contiguous), so block-caching them would decode far more than
        # the budget.  Direct decode respects the budget exactly.
        amp, phase = decode_frames(path, index, frame_ids)
        data = amp if metric == "amplitude" else phase

    # Aggregate into columns.
    decoded_times = times[frame_ids] if n_decoded > 0 else np.zeros(0)
    span = t1 - t0
    col_edges = t0 + np.arange(width + 1, dtype=np.float64) / width * span
    col_starts = np.searchsorted(decoded_times, col_edges[:-1], side="left")
    col_ends = np.searchsorted(decoded_times, col_edges[1:], side="left")
    # The last column is closed on the right, mirroring the frame selection
    # above.  Assigning the decoded count rather than searchsorting col_edges[-1]
    # also sidesteps the float error in ``t0 + span``, which need not land on t1
    # exactly -- every decoded frame is in [t0, t1] by construction, so anything
    # not yet claimed by an earlier column belongs to this one.
    if width > 0:
        col_ends[-1] = n_decoded

    grid = np.full((num_sc, width), np.nan, dtype=np.float32)

    if n_decoded > 0:
        for x in range(width):
            s = int(col_starts[x])
            e = int(col_ends[x])
            if e <= s:
                continue
            if metric == "amplitude":
                # Max-hold over frames in this column, per subcarrier.
                # np.max handles -inf correctly: finite values win, and a
                # column where every frame has -inf (db(0)) yields -inf.
                grid[:, x] = data[s:e].max(axis=0)
            else:
                # Phase: the single frame nearest the column's centre time.
                centre = t0 + (x + 0.5) / width * span
                nearest = s + int(np.argmin(np.abs(decoded_times[s:e] - centre)))
                grid[:, x] = data[nearest]

    # Flip subcarrier axis so row 0 = highest subcarrier index, matching the
    # frontend's image convention (subcarrierSourceRect in render.ts).
    grid = np.ascontiguousarray(grid[::-1, :])

    # Finite value range (excludes NaN and -inf).
    finite_mask = np.isfinite(grid)
    if finite_mask.any():
        vmin = float(grid[finite_mask].min())
        vmax = float(grid[finite_mask].max())
    else:
        vmin = 0.0
        vmax = 0.0

    return grid, {
        "frames_decoded": n_decoded,
        "total_in_range": total_in_range,
        "exact": exact,
        "vmin": vmin,
        "vmax": vmax,
        "t_min": float(times[0]),
        "t_max": float(times[-1]),
    }
