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
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .batch import decode_frames
from .index import FrameIndex
from .phase import detrend_subcarrier, unwrap_subcarrier, unwrap_time
from .ratio import (
    CONTEXT_FRAMES,
    Reference,
    build_reference,
    correct_ratio_amplitude,
    correct_ratio_phase,
)


class Derived(NamedTuple):
    """How a derived metric is built from other metrics.

    ``bases`` may name base metrics or other derived ones — they are resolved
    recursively, so the time-unwrapped ratio can be built on the corrected
    ratio without either knowing about the other.

    ``needs_times`` metrics receive the decoded frames' timestamps as a
    keyword argument. Anything segmenting on capture gaps needs them, and the
    frame values alone cannot supply them.

    ``needs_reference`` metrics receive the capture's ``Reference`` and a
    ``native`` flag saying whether consecutive rows are consecutive frames.
    They are the ones whose answer would otherwise depend on which frames the
    view happened to contain — see ``backend.ratio``. They are also the ones
    decoded with a context margin, since their transform reads neighbours.
    """

    bases: tuple[str, ...]
    transform: Callable[..., np.ndarray]
    needs_times: bool = False
    needs_reference: bool = False

# Metrics decoded straight out of a frame payload.
BASE_METRICS = ("amplitude", "phase", "csi_ratio_amplitude", "csi_ratio_phase")

# Metrics derived from base metrics by a transform, mapped to
# (base metrics, transform). The transform receives the named base blocks as
# positional arguments, so a metric can be derived from more than one — the
# swap correction needs the ratio phase to decide which frames to flip even
# when it is the amplitude being corrected.
#
# Derived blocks are computed from decoded blocks rather than from the
# payload, and only when actually requested — an unopened derived panel costs
# nothing.
#
# The ratio gets unwrap but no detrend: rx1/rx0 already cancels the common
# per-packet offset and slope, so fitting and subtracting a line there would
# remove signal rather than nuisance. See backend/phase.py.
DERIVED_METRICS: dict[str, Derived] = {
    "phase_unwrapped": Derived(("phase",), unwrap_subcarrier),
    "phase_detrended": Derived(("phase",), detrend_subcarrier),
    "csi_ratio_phase_unwrapped": Derived(("csi_ratio_phase",), unwrap_subcarrier),
    # Both corrected metrics take phase *and* amplitude. The swap negates both,
    # and only the amplitude says which side of a boundary is the right way up
    # — its profile shape is fixed by the antennas, not by the moving channel.
    "csi_ratio_phase_corrected": Derived(
        ("csi_ratio_phase", "csi_ratio_amplitude"),
        correct_ratio_phase,
        needs_reference=True,
    ),
    "csi_ratio_amplitude_corrected": Derived(
        ("csi_ratio_amplitude", "csi_ratio_phase"),
        correct_ratio_amplitude,
        needs_reference=True,
    ),
    # Time-axis unwrapping is built on the *corrected* ratio, never the raw
    # one. Uncorrected, 1.2% of frame-to-frame steps exceed pi outright, and
    # each one an unwrapper misreads offsets everything after it by 2*pi for
    # good. Corrected, the 99th percentile step is 0.328 rad — a tenth of pi.
    "csi_ratio_phase_time_unwrapped": Derived(
        ("csi_ratio_phase_corrected",), unwrap_time, needs_times=True
    ),
}

TILE_METRICS = BASE_METRICS + tuple(DERIVED_METRICS)


def _needs_reference(metric: str) -> bool:
    """True if *metric* or anything it is derived from needs a Reference.

    ``csi_ratio_phase_time_unwrapped`` does not correct anything itself, but
    it is built on the corrected ratio — so it needs the reference just as
    much, one step removed.
    """
    derived = DERIVED_METRICS.get(metric)
    if derived is None:
        return False
    return derived.needs_reference or any(_needs_reference(m) for m in derived.bases)

# Metrics aggregated by max-hold within a display column. Everything else is
# nearest-frame: a maximum over angles is meaningless, and that holds for the
# unwrapped views too — an unwrapped row is still a phase curve, just one
# whose branch cuts have been removed.
MAX_HOLD_METRICS = ("amplitude", "csi_ratio_amplitude", "csi_ratio_amplitude_corrected")

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

# Frames decoded to measure a capture's orientation reference. They are drawn
# evenly across the whole capture rather than from one stretch, so a single
# badly-oriented block cannot carry the median. A few thousand is ample —
# every 2000-frame chunk of the captures at hand correlates +0.955 or better
# with the file's own profile.
REFERENCE_SAMPLE = 4096

# Orientation references, keyed by (path, file size, filter). One decode of
# REFERENCE_SAMPLE frames per capture per filter, then every tile of that
# capture is judged against the same absolute orientation no matter what the
# view is showing. File size is in the key for the same reason as in the
# block cache: a rewritten or truncated file must not serve a stale one.
_ref_cache: dict[tuple, Reference | None] = {}
_ref_lock = threading.Lock()


def get_reference(
    path: Path,
    index: FrameIndex,
    file_size: int,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
) -> Reference | None:
    """Return the cached orientation reference for this capture and filter.

    The reference is per transmitter, because the band profile it anchors
    against is a property of one pair of antennas — blending two senders'
    profiles together anchors to neither.

    So a *source_mac* is required, and without one this returns ``None``.
    That is the whole gate on whether the ratio gets corrected at all: on an
    interleaved capture the correction is very nearly a no-op anyway, because
    ``_chain`` fits adjacent pairs and 86-93% of those pairs are two
    different senders, so the confidence gate declines. Measured on an hour
    of frames, single-sender correction leaves 0.3-0.6% of steps above pi/2
    where interleaved leaves 11.2-11.6% — against 12.4-13.2% uncorrected.
    Running it there buys almost nothing and reports a correction that did
    not happen, so it is not run.

    ``None`` also comes back when the sender's own band is too flat to
    correlate against, or its phase names no clear direction. Either way the
    caller leaves the ratio alone and the tile reports ``anchored: False``.
    """
    if source_mac is None or not source_mac.strip():
        return None

    key = (str(path), file_size, mimo, source_mac)
    with _ref_lock:
        if key in _ref_cache:
            return _ref_cache[key]

    ids = np.flatnonzero(index.filter_mask(mimo=mimo, source_mac=source_mac))

    if len(ids) == 0:
        ref = None
    else:
        picks = np.unique(
            np.linspace(0, len(ids) - 1, min(REFERENCE_SAMPLE, len(ids))).astype(np.int64)
        )
        _, _, ratio_amp, ratio_phase = decode_frames(path, index, ids[picks])
        ref = build_reference(ratio_amp, ratio_phase)

    with _ref_lock:
        _ref_cache[key] = ref
    return ref


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
    with _ref_lock:
        _ref_cache.clear()
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
    reference: Reference | None = None,
) -> np.ndarray:
    """Return the decoded block for one metric, from cache or by decoding.

    On a cache miss, decodes the full block (all base metrics) and caches them
    under their respective keys, so a later request for another metric hits.

    A derived metric (see ``DERIVED_METRICS``) is computed from its base
    metric's block — itself fetched through this function, so the base decode
    is shared — and cached under its own key. Deriving lazily rather than
    alongside the base decode keeps the cache from carrying transforms of
    blocks nobody asked to see.

    A ``needs_reference`` metric is derived over a context margin taken from
    the neighbouring blocks, then trimmed back. Its transform judges a frame
    against its neighbours, and without the margin the frames at a block
    boundary would be decided on half of them — visible as a speckled seam
    every BLOCK_SIZE frames.
    """
    key = (str(path), metric, block_idx, file_size)
    cached = _block_cache.get(key)
    if cached is not None:
        return cached

    derived = DERIVED_METRICS.get(metric)
    if derived is not None:
        start = block_idx * BLOCK_SIZE
        stop = min(start + BLOCK_SIZE, index.count)
        if derived.needs_reference:
            lead = min(CONTEXT_FRAMES, start)
            trail = min(CONTEXT_FRAMES, index.count - stop)
            bases = [
                _base_with_context(
                    path, index, block_idx, m, file_size, lead, trail, reference
                )
                for m in derived.bases
            ]
            # native=False with no reference makes the transform an identity:
            # nothing left that could decide anything, so decide nothing.
            block = derived.transform(
                *bases, reference=reference, native=reference is not None
            )[lead : lead + (stop - start)]
        else:
            bases = [
                _decode_block_cached(path, index, block_idx, m, file_size, reference)
                for m in derived.bases
            ]
            if derived.needs_times:
                block = derived.transform(*bases, times=index.times[start:stop])
            else:
                block = derived.transform(*bases)
        _block_cache.put(key, block)
        return block

    block_start = block_idx * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, index.count)
    block_ids = np.arange(block_start, block_end)
    amp, phase, ratio_amp, ratio_phase = decode_frames(path, index, block_ids)

    _metrics = {
        "amplitude": amp,
        "phase": phase,
        "csi_ratio_amplitude": ratio_amp,
        "csi_ratio_phase": ratio_phase,
    }
    for m, arr in _metrics.items():
        _block_cache.put((str(path), m, block_idx, file_size), arr)
    with _block_cache._lock:
        _block_cache.frames_decoded += len(block_ids)

    return _metrics[metric]


def _base_with_context(
    path: Path,
    index: FrameIndex,
    block_idx: int,
    metric: str,
    file_size: int,
    lead: int,
    trail: int,
    reference: Reference | None,
) -> np.ndarray:
    """One block of *metric* with *lead*/*trail* frames of its neighbours.

    The neighbouring blocks come through the same cache, so the context is
    usually free — and when it is not, it is a block the view is about to
    want anyway.
    """
    parts = []
    if lead:
        prev = _decode_block_cached(
            path, index, block_idx - 1, metric, file_size, reference
        )
        parts.append(prev[len(prev) - lead :])
    parts.append(
        _decode_block_cached(path, index, block_idx, metric, file_size, reference)
    )
    if trail:
        nxt = _decode_block_cached(
            path, index, block_idx + 1, metric, file_size, reference
        )
        parts.append(nxt[:trail])
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _materialise(
    metric: str,
    available: dict[str, np.ndarray],
    times: np.ndarray,
    reference: Reference | None = None,
    native: bool = True,
) -> np.ndarray:
    """Resolve *metric* from already-decoded base arrays, deriving as needed.

    Used on the paths that decode a frame selection directly (filtered or
    stride-sampled requests), where the block cache does not apply. Results
    are memoised into *available* so a metric derived from another derived
    metric computes each stage once.
    """
    cached = available.get(metric)
    if cached is not None:
        return cached

    derived = DERIVED_METRICS[metric]
    bases = [_materialise(m, available, times, reference, native) for m in derived.bases]
    if derived.needs_reference:
        out = derived.transform(*bases, reference=reference, native=native)
    elif derived.needs_times:
        out = derived.transform(*bases, times=times)
    else:
        out = derived.transform(*bases)
    available[metric] = out
    return out


def _decode_via_blocks(
    path: Path,
    index: FrameIndex,
    frame_ids: np.ndarray,
    metric: str,
    file_size: int,
    reference: Reference | None = None,
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

        block = _decode_block_cached(
            path, index, block_idx, metric, file_size, reference
        )

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
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Build a display-resolution grid for the requested time range.

    Returns ``(grid, metadata)`` where *grid* has shape
    ``(num_subcarriers, width)``, float32, row-major with row 0 = highest
    subcarrier index (matching the frontend's ``subcarrierSourceRect``
    convention).  Empty columns are NaN; ``-inf`` from ``db(0)`` is preserved.

    *metadata* keys: ``frames_decoded``, ``total_in_range``, ``exact``,
    ``vmin``, ``vmax``, ``p_low``, ``p_high``, ``t_min``, ``t_max``,
    ``filled_columns``.

    ``mimo`` and ``source_mac`` restrict which frames are eligible for
    decoding. Filtered-out frames leave NaN holes — they are NOT filled from
    neighbours, so a 2x2 burst excluded by a '2x1 only' filter stays visible
    as a transparent stripe. The capture's full extent (``t_min``/``t_max``
    in metadata) is the unfiltered range so the live view keeps tracking
    growth; the tile window itself reflects the request.
    """
    path = Path(path)
    index = get_index(path)
    num_sc = index.num_subcarriers
    times = index.times

    filtered = mimo is not None or (source_mac is not None and source_mac.strip())
    if filtered:
        mask = index.filter_mask(mimo=mimo, source_mac=source_mac)
        filtered_idxs = np.flatnonzero(mask)
        filtered_times = times[filtered_idxs] if len(filtered_idxs) else np.zeros(0, dtype=np.float64)
    else:
        filtered_idxs = np.arange(index.count, dtype=np.int64)
        filtered_times = times

    # Clamp width.
    width = max(1, min(width, 4096))

    # Guard against empty capture or invalid range.
    if len(filtered_idxs) == 0 or t1 <= t0:
        grid = np.full((max(num_sc, 0), width), np.nan, dtype=np.float32)
        return grid, {
            "frames_decoded": 0,
            "total_in_range": 0,
            "exact": True,
            "vmin": 0.0,
            "vmax": 0.0,
            "p_low": 0.0,
            "p_high": 0.0,
            "t_min": float(times[0]) if index.count else 0.0,
            "t_max": float(times[-1]) if index.count else 0.0,
            "filled_columns": 0,
        }

    # Find filtered frames in [t0, t1] -- CLOSED at both ends (see /api/meta).
    lo = int(np.searchsorted(filtered_times, t0, side="left"))
    hi = int(np.searchsorted(filtered_times, t1, side="right"))
    total_in_range = hi - lo

    width = max(1, min(width, total_in_range))

    if total_in_range > TILE_FRAME_BUDGET:
        sampled = np.linspace(
            0, total_in_range - 1, TILE_FRAME_BUDGET, dtype=np.int64
        )
        sel_in_filtered = np.arange(lo, hi)[sampled]
        exact = False
    else:
        sel_in_filtered = np.arange(lo, hi)
        exact = True

    frame_ids = filtered_idxs[sel_in_filtered] if filtered else sel_in_filtered.astype(np.int64)
    n_decoded = len(frame_ids)

    file_size = path.stat().st_size

    # Metrics that undo the ratio corruption need the capture's own
    # orientation, or their answer is a property of this view rather than of
    # the data — pan or zoom and whole panels invert. See backend.ratio.
    #
    # A reference exists only for a single selected sender, and the ratio is
    # corrected only where there is one: `reference is not None` is the single
    # switch, so a view can never be half-corrected or claim a correction it
    # did not get. On `source_mac=all` the ratio is passed through untouched.
    needs_reference = _needs_reference(metric)
    reference = (
        get_reference(path, index, file_size, mimo=mimo, source_mac=source_mac)
        if needs_reference
        else None
    )
    correcting = reference is not None

    if n_decoded == 0:
        data = np.empty((0, num_sc), dtype=np.float32)
    elif exact and not filtered:
        data = _decode_via_blocks(path, index, frame_ids, metric, file_size, reference)
    else:
        # A stride-sampled selection is not a frame sequence: its rows are
        # seconds apart, so the passes that compare a frame to its neighbour
        # have nothing to compare against and are skipped (native=False).
        # An exact selection is contiguous, so it gets a context margin
        # instead — decoded, corrected, then trimmed off — which gives its
        # edge frames the same neighbours a full-capture pass would.
        lead = 0
        ctx_sel = sel_in_filtered
        if correcting and exact:
            lo_ctx = max(0, int(sel_in_filtered[0]) - CONTEXT_FRAMES)
            hi_ctx = min(len(filtered_idxs), int(sel_in_filtered[-1]) + 1 + CONTEXT_FRAMES)
            ctx_sel = np.arange(lo_ctx, hi_ctx)
            lead = int(sel_in_filtered[0]) - lo_ctx
        ctx_ids = filtered_idxs[ctx_sel] if filtered else ctx_sel.astype(np.int64)

        amp, phase, ratio_amp, ratio_phase = decode_frames(path, index, ctx_ids)
        available = {
            "amplitude": amp,
            "phase": phase,
            "csi_ratio_amplitude": ratio_amp,
            "csi_ratio_phase": ratio_phase,
        }
        data = _materialise(
            metric, available, times[ctx_ids], reference, native=exact and correcting
        )
        if lead or len(ctx_ids) != n_decoded:
            data = data[lead : lead + n_decoded]

    decoded_times = times[frame_ids] if n_decoded > 0 else np.zeros(0)
    span = t1 - t0
    col_edges = t0 + np.arange(width + 1, dtype=np.float64) / width * span
    col_starts = np.searchsorted(decoded_times, col_edges[:-1], side="left")
    col_ends = np.searchsorted(decoded_times, col_edges[1:], side="left")
    if width > 0:
        col_ends[-1] = n_decoded

    grid = np.full((num_sc, width), np.nan, dtype=np.float32)

    if n_decoded > 0:
        for x in range(width):
            s = int(col_starts[x])
            e = int(col_ends[x])
            if e <= s:
                continue
            if metric in MAX_HOLD_METRICS:
                grid[:, x] = data[s:e].max(axis=0)
            else:
                centre = t0 + (x + 0.5) / width * span
                nearest = s + int(np.argmin(np.abs(decoded_times[s:e] - centre)))
                grid[:, x] = data[nearest]

    # Gap fill: only sampling gaps (sub-gap-limit intervals) get filled from
    # the nearest decoded frame. When a filter is active, frames excluded by
    # the filter must stay NaN — they are not sampling gaps, they are
    # intentional omissions. So skip the fill entirely while filtered.
    filled_columns = 0
    empty = col_ends <= col_starts
    if not filtered and n_decoded >= 2:
        gap_limit = 2.0 * float(np.percentile(np.diff(decoded_times), 95))
    elif not filtered:
        gap_limit = 0.0
    else:
        gap_limit = 0.0
    gap_limit = max(gap_limit, span / width)
    if not filtered and n_decoded > 0 and empty.any():
        centres = t0 + (np.arange(width) + 0.5) / width * span
        ec = centres[empty]
        j = np.searchsorted(decoded_times, ec)
        j_lo = np.clip(j - 1, 0, n_decoded - 1)
        j_hi = np.clip(j, 0, n_decoded - 1)
        d_lo = np.abs(decoded_times[j_lo] - ec)
        d_hi = np.abs(decoded_times[j_hi] - ec)
        nearest = np.where(d_lo <= d_hi, j_lo, j_hi)
        dist = np.minimum(d_lo, d_hi)
        ok = dist <= gap_limit
        cols = np.flatnonzero(empty)[ok]
        if cols.size > 0:
            grid[:, cols] = data[nearest[ok]].T
            filled_columns = int(cols.size)

    # Flip subcarrier axis so row 0 = highest subcarrier index, matching the
    # frontend's image convention (subcarrierSourceRect in render.ts).
    grid = np.ascontiguousarray(grid[::-1, :])

    # Finite value range (excludes NaN and -inf) and robust percentile bounds.
    # The raw min/max is dominated by outliers — on the real capture, 98.5% of
    # amplitude values fall in [40, 60] while extrema span [7.7, 84.7]. A
    # min/max scale compresses the visible structure into one narrow slice of
    # the colormap. The 1st/99th percentile bounds are the robust scale the
    # frontend locks to. -inf from db(0) is excluded by the finite mask, not
    # clamped into the percentile — clamping would drag p_low to -inf.
    finite_mask = np.isfinite(grid)
    if finite_mask.any():
        finite_vals = grid[finite_mask]
        vmin = float(finite_vals.min())
        vmax = float(finite_vals.max())
        p_low = float(np.nanpercentile(finite_vals, 1))
        p_high = float(np.nanpercentile(finite_vals, 99))
    else:
        vmin = 0.0
        vmax = 0.0
        p_low = 0.0
        p_high = 0.0

    return grid, {
        "frames_decoded": n_decoded,
        "total_in_range": total_in_range,
        "exact": exact,
        # Whether this tile's ratio was corrected. False on a metric that
        # needs an orientation reference and could not get one — no sender
        # selected, most often — in which case the ratio is shown exactly as
        # decoded rather than corrected against a reference that isn't there.
        "anchored": correcting if needs_reference else True,
        "vmin": vmin,
        "vmax": vmax,
        "p_low": p_low,
        "p_high": p_high,
        "t_min": float(times[0]),
        "t_max": float(times[-1]),
        "filled_columns": filled_columns,
    }
