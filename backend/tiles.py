"""Pre-aggregated tile serving for offline capture exploration.

Builds display-resolution grids from arbitrary time ranges of a FeitCSI
capture. Cost per request is bounded: the output grid never exceeds the
requested width whatever the view covers, and each chunk of it decodes at most
``CHUNK_FRAME_BUDGET`` frames.

Columns are quantised to a fixed lattice rather than derived from the window
that asked for them -- see "The lattice" below. A tile therefore covers the
smallest lattice-aligned range containing the request, reports it in
``meta["t0"]``/``["t1"]``, and leaves the crop to the caller.

Three caches keep repeated work cheap:

* ``FrameIndex`` objects are cached per path (``extend()`` on each request so a
  growing capture stays current without a full rescan).
* Decoded blocks (contiguous runs of ``BLOCK_SIZE`` frames) are cached in an
  LRU keyed by ``(path, metric, block_index, frames_in_that_block)``. Keying
  a block on how many frames *it* holds rather than on the size of the whole
  file is what lets a completed block survive the capture growing: its own
  count can never change again, while the tail block's does. Keying on file size
  means a rewritten or truncated file cannot serve stale blocks.
* Lattice chunks (runs of ``CHUNK_COLUMNS`` reduced columns) are cached in an
  LRU keyed by ``(path, metric, level, chunk, filters, frames_in_that_chunk)``.
  Same rule as the block cache, one level up: a chunk is determined by the
  frames inside it, so the entry stays valid exactly as long as that count
  does. This is the cache the lattice buys -- a request keyed on an exact
  window could never hit.

Thread safety: FastAPI serves handlers from a threadpool, so both caches are
guarded with locks.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import mtk, presence
from .batch import decode_frames as _decode_feitcsi
from .cir import csi_to_cir_centred
from .index import FrameIndex
from .doppler import (
    DEFAULT_MAX_GAP_FRACTION,
    DEFAULT_ZERO_PAD,
    gap_limit_for,
    resample_uniform,
    stft_average,
    uniform_grid,
)
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
    # False for a transform that changes what a row *is* — the CIR's rows are
    # delay taps, not subcarriers, so a base's per-subcarrier NaN pattern
    # (guard band, dropped pilots) has no counterpart to preserve. What must
    # still hold, and does, is coverage per *frame*: a column the base has no
    # ratio for gets no CIR either. See ``test_derived_metrics_are_served``.
    preserves_coverage: bool = True

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
    # Delay-domain view of the raw channel (rx0/tx0), not the ratio: no
    # correction applies because there is no second chain to have been
    # swapped with, so this is built straight on the base amplitude/phase
    # and needs no Reference. See backend.cir for why this is deliberately
    # the uncorrected channel rather than the ratio's own IFFT — the two
    # answer different questions and are not interchangeable.
    "csi_cir": Derived(
        ("amplitude", "phase"),
        csi_to_cir_centred,
        preserves_coverage=False,
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
MAX_HOLD_METRICS = (
    "amplitude",
    "csi_ratio_amplitude",
    "csi_ratio_amplitude_corrected",
    # A CIR magnitude is the same kind of quantity as an amplitude — real,
    # non-negative, meaningfully peaked — so the same peak-preserving
    # aggregation applies rather than nearest-frame, which would just pick
    # one frame's echo pattern and discard the rest of a zoomed-out column.
    "csi_cir",
)

# Metrics whose values are angles wrapped to (-pi, pi] — the ones the
# frontend gives a fixed [-pi, pi] scale and the TWILIGHT palette. Averaging
# two wrapped angles with plain arithmetic is wrong exactly at the branch
# cut: a frame at +3.1 rad and its neighbour at -3.1 rad are 0.08 rad apart
# on the circle, and linear interpolation would walk the *long* way around
# and report something near 0. The gap-fill below detours through
# exp(i*angle) for these so it interpolates along the circle instead of
# through the cut. Every other metric here — including the *_unwrapped and
# *_detrended views, whose whole point is to no longer be an angle on a
# circle — is a plain number and takes ordinary linear interpolation.
CIRCULAR_METRICS = ("phase", "csi_ratio_phase", "csi_ratio_phase_corrected")

# Maximum frames decoded per /api/tile request. When the requested time range
# holds more frames than this, stride-sample approximately BUDGET frames evenly
# across the range and mark the tile as sampled (X-Tile-Exact: 0).
#
# A sampled max-hold can miss transients that a full decode would catch —
# that is the price of a bounded request.  X-Tile-Exact lets the UI say so
# rather than silently lying.  Zooming in shrinks the range until it fits the
# budget and the tile becomes exact; that is the intended interaction.
TILE_FRAME_BUDGET = 8192

# ----------------------------------------------------------------------- #
#  The lattice                                                            #
# ----------------------------------------------------------------------- #
#
# A tile's columns are quantised to a fixed time grid instead of being derived
# from the window that asked for them. Column *c* at level *L* always covers
# ``[c*dt, (c+1)*dt)`` with ``dt = LATTICE_DT0 * 2**L``, measured from the
# capture's own t=0 — so a column is a property of the capture, not of the
# request.
#
# Before this, ``col_edges = t0 + arange(width+1)/width * span`` made every
# edge a function of the view. A one-pixel pan re-quantised all 1248 columns
# and a live poll re-binned the whole grid every 300 ms: the picture crawled
# even where the data had not changed. On the lattice a pan shifts columns
# that keep their values, and a live poll appends columns on the right.
#
# Powers of two, not a finer ladder: doubling keeps every coarse column an
# exact union of two finer ones, so levels nest and a coarse chunk could be
# folded from finer ones later. The cost is that resolution steps by 2x
# between levels rather than tracking the pixel width continuously.
LATTICE_DT0 = 1e-3  # finest column width, 1 ms
LATTICE_MAX_LEVEL = 40  # dt ~ 12 days; a guard, never reached in practice

# Columns per cached chunk. The unit of work and of caching: a request
# assembles its grid from these, so two overlapping requests share everything
# but their edges. 256 columns at 4 bytes x 256 subcarriers is 256 KB.
CHUNK_COLUMNS = 256

# Frames decoded per chunk before stride-sampling kicks in. Per column that is
# CHUNK_FRAME_BUDGET / CHUNK_COLUMNS = 8, matching the ~6.5 the old whole-tile
# budget of 8192 over 1248 columns allowed. The stride is anchored to the
# chunk's own frame range, which is fixed by the lattice, so the sample does
# not shift when the window moves -- the old budget selected with
# ``linspace`` over the *request*, so a pan changed which frames were shown.
CHUNK_FRAME_BUDGET = 2048

# Decode blocks are cached at this granularity (contiguous runs of frames
# aligned to BLOCK_SIZE boundaries).  A block is decoded in full on first
# request and reused on subsequent requests touching the same frames.
BLOCK_SIZE = 4096

# Upper bound on total bytes held in the block cache.
CACHE_MAX_BYTES = 256 * 1024 * 1024  # 256 MB

# Upper bound on total bytes held in the chunk cache. Smaller than the block
# cache because a chunk is already reduced: 256 columns against the thousands
# of frames they were built from.
CHUNK_CACHE_MAX_BYTES = 128 * 1024 * 1024  # 128 MB

# Frames sampled across the whole capture to measure the colour scale, and how
# far the capture may grow before that sample is taken again. The scale must
# not depend on the caller's pixel width — two browsers on the same capture
# have to lock to the same scale — so it cannot be read off the tile, whose
# level *is* a function of width. Sampling the capture instead makes it a
# property of the data. See ``_scale_source``.
STATS_FRAMES = 2048
STATS_REFRESH_FRAMES = 4096

# Below this many sampled frames inside the requested range, the capture-wide
# sample says too little about it and the range's own frames are decoded
# instead. Such a range is small by definition, so that is cheap -- and it
# still depends only on the range, never on the width.
STATS_MIN_FRAMES = 16

# Hard cap on the columns one tile may hold, whatever the caller asks for.
MAX_TILE_COLUMNS = 4096


def lattice_dt(level: int) -> float:
    """Seconds covered by one column at *level*."""
    return LATTICE_DT0 * (2.0**level)


def pick_level(span: float, width: int, min_dt: float) -> int:
    """Finest level whose columns still fit *width* pixels.

    Depends only on the span and the requested width — never on where the
    window sits — so panning cannot change the level, and the columns of two
    overlapping views line up.

    *min_dt* is the capture's own median frame spacing, and the level never
    goes finer than it. Going finer asks for columns no frame can fill: at
    full extent a 1101-packet capture asked for 1230 columns would come back
    with hundreds of empty ones, which is the reason the pre-lattice code
    capped the width at the frame count. It cannot cap the width any more --
    that is exactly the view-dependent quantisation the lattice removes -- so
    it caps the resolution instead, which is the same guarantee expressed on
    the axis that belongs to the data.
    """
    if not (span > 0) or width < 1:
        return 0
    level = 0
    if span > width * LATTICE_DT0:
        level = int(math.ceil(math.log2(span / (width * LATTICE_DT0))))
    # Nudge down for float error: log2 of an exact power of two can land a
    # hair above the integer and cost a whole level of resolution.
    while level > 0 and math.ceil(span / lattice_dt(level - 1)) <= width:
        level -= 1
    while math.ceil(span / lattice_dt(level)) > width:
        level += 1
    if min_dt > 0:
        floor_level = max(0, int(math.ceil(math.log2(min_dt / LATTICE_DT0))))
        level = max(level, floor_level)
    return min(level, LATTICE_MAX_LEVEL)


def snap_window(t0: float, t1: float, level: int) -> tuple[int, int]:
    """Column indices `[c0, c1)` at *level* covering `[t0, t1]`.

    Snaps outwards: the tile always contains the requested window, and the
    caller crops. ``tileSourceRect`` in the frontend already maps a tile whose
    window is wider than the view onto the view, so nothing else has to know.
    """
    dt = lattice_dt(level)
    c0 = int(math.floor(t0 / dt))
    c1 = int(math.ceil(t1 / dt))
    if c1 <= c0:
        c1 = c0 + 1
    return c0, c1


def chunk_span(level: int) -> float:
    """Seconds covered by one cached chunk at *level*."""
    return CHUNK_COLUMNS * lattice_dt(level)


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

    def drop_path(self, path: str) -> None:
        """Evict every block belonging to *path*.

        For a truncated capture: the frame ids its cached blocks were decoded
        under no longer refer to the same frames.
        """
        with self._lock:
            for key in [k for k in self._entries if k[0] == path]:
                self._bytes -= self._entries.pop(key).nbytes

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
#  Chunk cache                                                            #
# ----------------------------------------------------------------------- #


class _Chunk(NamedTuple):
    """One cached run of ``CHUNK_COLUMNS`` lattice columns.

    ``grid`` is ``(num_subcarriers, CHUNK_COLUMNS)`` with row 0 = lowest
    subcarrier index — the flip to display orientation happens once, on the
    assembled tile, so a chunk stays in the same orientation the decoders use.
    """

    grid: np.ndarray
    frames_decoded: int
    total_in_range: int
    exact: bool
    # Which of this chunk's columns the gap fill wrote. Kept per column, not
    # as a count, because a tile reports the fills inside *its* slice -- a
    # chunk usually overhangs both ends of the window that pulled it in.
    filled_mask: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.grid.nbytes + self.filled_mask.nbytes)


class _ChunkCache:
    """LRU cache of computed lattice chunks, bounded by total bytes.

    Shares the shape of ``_BlockCache`` but holds reduced columns rather than
    decoded frames: a chunk is the unit two overlapping requests have in
    common, so a pan re-computes only the columns that actually entered the
    view.

    The key carries the number of frames the chunk holds, so an entry stays
    valid exactly as long as its own contents do: the chunk at a live
    capture's growing edge misses as soon as a frame lands in it, while every
    chunk behind it keeps hitting. See ``_chunk_frame_count``.
    """

    def __init__(self, max_bytes: int = CHUNK_CACHE_MAX_BYTES) -> None:
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple, _Chunk] = OrderedDict()
        self._bytes = 0
        # Cumulative hits/misses, for tests and for reasoning about a live
        # view's steady state. Reset by ``reset_tile_caches()``.
        self.hits = 0
        self.misses = 0

    def drop_path(self, path: str) -> None:
        with self._lock:
            for key in [k for k in self._entries if k[0] == path]:
                self._bytes -= self._entries.pop(key).nbytes

    def get(self, key: tuple) -> _Chunk | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def put(self, key: tuple, chunk: _Chunk) -> None:
        with self._lock:
            old = self._entries.get(key)
            if old is not None:
                self._bytes -= old.nbytes
            self._entries[key] = chunk
            self._bytes += chunk.nbytes
            self._entries.move_to_end(key)
            while self._bytes > self._max_bytes and self._entries:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.nbytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            self.hits = 0
            self.misses = 0


_chunk_cache = _ChunkCache()

# Capture-wide value samples backing the colour scale, keyed per capture,
# metric and filter. See ``_scale_source``.
_stats_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
_stats_lock = threading.Lock()


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
    interpolate: bool = True,
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

    key = (str(path), file_size, mimo, source_mac, interpolate)
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
        _, _, ratio_amp, ratio_phase = decode_frames(
            path, index, ids[picks], interpolate=interpolate
        )
        ref = build_reference(ratio_amp, ratio_phase)

    with _ref_lock:
        _ref_cache[key] = ref
    return ref


def decode_frames(path, index, frame_ids, **kwargs):
    """Decode via whichever reader owns *index*.

    The two readers produce the same four float32 arrays; everything above
    this line is format-blind.
    """
    if isinstance(index, mtk.MTKIndex):
        return mtk.decode_frames(path, index, frame_ids, **kwargs)
    return _decode_feitcsi(path, index, frame_ids, **kwargs)


def get_index(path: Path) -> FrameIndex | mtk.MTKIndex:
    """Return the shared index for *path*, extending it if it exists.

    Mirrors the ``get_stream`` registry pattern in ``backend.stream``: the
    first call builds a full FrameIndex, subsequent calls call ``extend()``
    to pick up appended frames without a full rescan.  Truncation triggers a
    rebuild inside ``extend()``.
    """
    path = Path(path)
    with _index_lock:
        idx = _index_cache.get(path)
        if idx is None:
            idx = mtk.MTKIndex(path) if mtk.can_read(path) else FrameIndex(path)
            _index_cache[path] = idx
        else:
            before = idx.count
            idx.extend()
            # extend() rebuilds from scratch when the file has been truncated,
            # so frame ids shift and any cached block for this path is stale.
            if idx.count < before:
                _block_cache.drop_path(str(path))
                _chunk_cache.drop_path(str(path))
        return idx


def reset_tile_caches() -> None:
    """Drop every cached FrameIndex, decoded block, chunk and scale sample."""
    with _index_lock:
        _index_cache.clear()
    with _ref_lock:
        _ref_cache.clear()
    with _stats_lock:
        _stats_cache.clear()
    _block_cache.clear()
    _chunk_cache.clear()


# ----------------------------------------------------------------------- #
#  Block-level decode with caching                                        #
# ----------------------------------------------------------------------- #


def _block_frame_count(index: FrameIndex, block_idx: int) -> int:
    """Frames currently held by *block_idx*.

    A block that is already full returns BLOCK_SIZE for the rest of the
    process's life, which is what lets a completed block stay cached while a
    growing capture appends to the tail.
    """
    start = block_idx * BLOCK_SIZE
    return max(0, min(BLOCK_SIZE, index.count - start))


def _decode_block_cached(
    path: Path,
    index: FrameIndex,
    block_idx: int,
    metric: str,
    reference: Reference | None = None,
    interpolate: bool = True,
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
    # One observation of index.count for the whole call. A growing capture is
    # extended in place by other requests, so reading it again below could size
    # the block against a different count than the key names: the lookup key
    # and the key the block is finally stored under would disagree, and every
    # poll on a growing capture would miss the cache and re-decode.
    total = index.count
    n_block = max(0, min(BLOCK_SIZE, total - block_idx * BLOCK_SIZE))

    key = (str(path), metric, block_idx, n_block, interpolate)
    cached = _block_cache.get(key)
    if cached is not None:
        return cached

    derived = DERIVED_METRICS.get(metric)
    if derived is not None:
        start = block_idx * BLOCK_SIZE
        stop = start + n_block
        if derived.needs_reference:
            lead = min(CONTEXT_FRAMES, start)
            trail = min(CONTEXT_FRAMES, total - stop)
            bases = [
                _base_with_context(
                    path, index, block_idx, m, lead, trail, reference,
                    interpolate=interpolate,
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
                _decode_block_cached(
                    path, index, block_idx, m, reference,
                    interpolate=interpolate,
                )
                for m in derived.bases
            ]
            if derived.needs_times:
                block = derived.transform(*bases, times=index.times[start:stop])
            else:
                block = derived.transform(*bases)
        _block_cache.put(key, block)
        return block

    block_start = block_idx * BLOCK_SIZE
    block_end = block_start + n_block
    block_ids = np.arange(block_start, block_end)
    amp, phase, ratio_amp, ratio_phase = decode_frames(
        path, index, block_ids, interpolate=interpolate
    )

    _metrics = {
        "amplitude": amp,
        "phase": phase,
        "csi_ratio_amplitude": ratio_amp,
        "csi_ratio_phase": ratio_phase,
    }
    for m, arr in _metrics.items():
        # Same n_block the key above was built from, so what the key promises
        # and what the array holds cannot diverge.
        _block_cache.put((str(path), m, block_idx, n_block, interpolate), arr)
    with _block_cache._lock:
        _block_cache.frames_decoded += len(block_ids)

    return _metrics[metric]


def _base_with_context(
    path: Path,
    index: FrameIndex,
    block_idx: int,
    metric: str,
    lead: int,
    trail: int,
    reference: Reference | None,
    *,
    interpolate: bool = True,
) -> np.ndarray:
    """One block of *metric* with *lead*/*trail* frames of its neighbours.

    The neighbouring blocks come through the same cache, so the context is
    usually free — and when it is not, it is a block the view is about to
    want anyway.
    """
    parts = []
    if lead:
        prev = _decode_block_cached(
            path, index, block_idx - 1, metric, reference,
            interpolate=interpolate,
        )
        parts.append(prev[len(prev) - lead :])
    parts.append(
        _decode_block_cached(
            path, index, block_idx, metric, reference,
            interpolate=interpolate,
        )
    )
    if trail:
        nxt = _decode_block_cached(
            path, index, block_idx + 1, metric, reference,
            interpolate=interpolate,
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
    reference: Reference | None = None,
    *,
    interpolate: bool = True,
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
            path, index, block_idx, metric, reference,
            interpolate=interpolate,
        )

        # How many of our frame_ids fall in this block? Bounded by what the
        # block actually holds, not by index.count, which another request may
        # have moved since these frame_ids were chosen.
        local_start = fid - block_start
        avail = len(block) - local_start
        take = min(avail, n - pos)
        if take <= 0:
            # The capture shrank under us; return the rows that are still real.
            return out[:pos]

        out[pos : pos + take] = block[local_start : local_start + take]
        pos += take

    return out


def _interpolate_time_gaps(
    grid: np.ndarray,
    fillable: np.ndarray,
    data: np.ndarray,
    decoded_times: np.ndarray,
    t0: float,
    span: float,
    width: int,
    gap_limit: float,
    *,
    circular: bool,
) -> int:
    """Fill the *fillable* columns of *grid* by linearly interpolating between
    the two decoded frames bracketing each one; return how many were filled.

    *fillable* marks the columns eligible for the fill — empty ones that are
    genuine sampling gaps rather than filter omissions; ``compute_tile`` makes
    that call. Of those, only columns within *gap_limit* of both neighbours
    are touched — the sampling-gap/dropped-packet distinction, also from
    ``compute_tile`` and not repeated here. *circular* selects the interpolation itself: a plain
    weighted average is correct for a magnitude or an unwrapped/accumulated
    phase, but wrong for an angle wrapped to (-pi, pi] — averaging +3.1 rad
    and -3.1 rad directly walks the long way around the circle and lands
    near 0 rad, when the two are 0.08 rad apart the short way. Blending
    ``exp(i*angle)`` instead and taking the angle back out follows the
    circle rather than the branch cut.
    """
    n_decoded = len(decoded_times)
    centres = t0 + (np.arange(width) + 0.5) / width * span
    ec = centres[fillable]
    j = np.searchsorted(decoded_times, ec)
    j_lo = np.clip(j - 1, 0, n_decoded - 1)
    j_hi = np.clip(j, 0, n_decoded - 1)
    t_lo = decoded_times[j_lo]
    t_hi = decoded_times[j_hi]
    dist = np.minimum(np.abs(t_lo - ec), np.abs(t_hi - ec))
    ok = dist <= gap_limit
    cols = np.flatnonzero(fillable)[ok]
    if cols.size == 0:
        return 0

    lo_vals = data[j_lo[ok]]
    hi_vals = data[j_hi[ok]]
    # Position within [t_lo, t_hi], 0 at t_lo and 1 at t_hi. The denominator
    # is only ever 0 when j_lo == j_hi (a gap past the first or last decoded
    # frame, clamped to it on both sides) — lo_vals and hi_vals are then
    # identical and mu is moot, so the divide-by-zero is masked rather than
    # branched around.
    span_t = t_hi[ok] - t_lo[ok]
    mu = np.where(span_t > 0, (ec[ok] - t_lo[ok]) / np.where(span_t > 0, span_t, 1.0), 0.0)
    mu = mu[:, None]
    if circular:
        blended = np.angle((1 - mu) * np.exp(1j * lo_vals) + mu * np.exp(1j * hi_vals))
    else:
        blended = (1 - mu) * lo_vals + mu * hi_vals
    grid[:, cols] = blended.T
    return int(cols.size)


# ----------------------------------------------------------------------- #
#  Tile computation                                                       #
# ----------------------------------------------------------------------- #


# Doppler runs on real-valued series only. Amplitude is the raw channel's
# magnitude; the phase panel is built on the *time-unwrapped* ratio phase
# because raw phase is wrapped, and its 2*pi jumps are broadband steps that
# would dominate an FFT and read as motion that is not there.
DOPPLER_METRICS: tuple[str, ...] = ("amplitude", "csi_ratio_phase_time_unwrapped")

# Floor on a clamped window. Below this a spectrogram column is too few samples
# to carry a meaningful spectrum, and the honest answer is an error.
MIN_DOPPLER_WIN = 8


def compute_doppler(
    path: Path,
    t0: float,
    t1: float,
    metric: str,
    *,
    win_seconds: float = 10.0,
    overlap: float = 0.5,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    zero_pad: int = DEFAULT_ZERO_PAD,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
) -> tuple[np.ndarray, dict]:
    """Subcarrier-averaged Doppler spectrogram for a time range.

    Returns ``(spectrogram, metadata)``. The grid is
    ``(win // 2 + 1, n_windows)`` float32, row 0 = highest Doppler frequency
    -- the same row order ``compute_tile`` uses, so the same renderer draws it.

    The sample rate is the capture's median frame rate over the frames
    actually in range, never a function of a requested display width. The
    window is given in *seconds* for the same reason: frame rate varies from
    5 Hz to 18 Hz across captures, so a fixed frame count would mean a
    different physical window on every file.

    A window longer than the range holds is *clamped* to what is available
    rather than refused. Zooming in is the normal way to use these panels, and
    every zoom past the window length would otherwise return 400 and leave the
    panel showing stale pixels. The window actually used comes back in
    ``win_seconds``, so the caller can report what it got rather than what it
    asked for.
    """
    if metric not in DOPPLER_METRICS:
        raise ValueError(
            "metric must be one of: " + ", ".join(repr(m) for m in DOPPLER_METRICS)
        )
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    index = get_index(path)
    times_all = np.asarray(index.times, dtype=float)
    mask = index.filter_mask(mimo=mimo, source_mac=source_mac)

    frame_ids = np.flatnonzero(mask & (times_all >= t0) & (times_all <= t1))
    if frame_ids.size < 2:
        raise ValueError("fewer than 2 frames in range")

    times = times_all[frame_ids]

    # Sample rate comes from the whole (filtered) capture, not from the frames
    # in view. Taking it from the visible slice makes the frequency axis
    # rescale as the user zooms: capture.dat's 1st-percentile interval is
    # 0.42 ms, so a short slice that happens to contain a burst reports 1144 Hz
    # against the capture's true 5.1 Hz, and the panel's Hz labels become
    # nonsense. A filter may legitimately change the rate, so it is applied --
    # a time window may not.
    filtered_times = times_all[mask]
    _, fs = uniform_grid(filtered_times if filtered_times.size >= 2 else times)
    step = 1.0 / fs
    n_grid = int(np.floor((times[-1] - times[0]) / step)) + 1
    grid_times = times[0] + np.arange(n_grid, dtype=float) * step
    # Clamp rather than refuse: a zoom past the window length is ordinary use.
    # MIN_DOPPLER_WIN keeps a clamped window wide enough to mean something.
    win = int(round(win_seconds * fs))
    win = min(win, int(grid_times.size))
    # Round DOWN to even -- rounding up would push a clamped window back past
    # the samples that are actually there.
    win -= win % 2
    if win < MIN_DOPPLER_WIN:
        raise ValueError(
            f"range holds {grid_times.size} samples at {fs:.2f} Hz, too few for "
            f"a {MIN_DOPPLER_WIN}-sample minimum window"
        )

    reference = (
        get_reference(
            path, index, path.stat().st_size, mimo=mimo, source_mac=source_mac,
            interpolate=interpolate,
        )
        if _needs_reference(metric)
        else None
    )
    values = _decode_for_doppler(
        path, index, frame_ids, metric, reference, interpolate
    )

    samples, fabricated = resample_uniform(
        times, values, grid_times, gap_limit_for(times)
    )
    hop = max(1, int(round(win * (1.0 - overlap))))
    spec, freqs = stft_average(
        samples, fs, win, hop,
        fabricated=fabricated,
        max_gap_fraction=max_gap_fraction,
        zero_pad=zero_pad,
    )

    finite = spec[np.isfinite(spec)]
    n_out = spec.shape[1]
    return spec.astype(np.float32), {
        "fs": float(fs),
        "f_max": float(freqs[-1]),
        "win": int(win),
        "hop": int(hop),
        "win_seconds": float(win * step),   # what was used, not what was asked
        "blank_columns": int(np.all(~np.isfinite(spec), axis=0).sum()),
        "frames_used": int(frame_ids.size),
        "t_min": float(times_all[0]) if times_all.size else 0.0,
        "t_max": float(times_all[-1]) if times_all.size else 0.0,
        # A column is centred on its window, so the first and last column
        # centres sit half a window inside the requested range.
        "col_t0": float(grid_times[0] + win * step / 2.0),
        "col_t1": float(grid_times[0] + ((n_out - 1) * hop + win / 2.0) * step),
        "vmin": float(finite.min()) if finite.size else 0.0,
        "vmax": float(finite.max()) if finite.size else 1.0,
        "p_low": float(np.percentile(finite, 1)) if finite.size else 0.0,
        "p_high": float(np.percentile(finite, 99)) if finite.size else 1.0,
    }


def _decode_for_doppler(
    path: Path,
    index: FrameIndex,
    frame_ids: np.ndarray,
    metric: str,
    reference: Reference | None,
    interpolate: bool,
) -> np.ndarray:
    """Decode *metric* for *frame_ids* as ``(n_frames, n_subcarriers)``.

    Goes through the same block cache the tile path uses, so a Doppler panel
    and a heatmap over the same window share one decode.
    """
    rows: list[np.ndarray] = []
    for block_idx in sorted({int(i) // BLOCK_SIZE for i in frame_ids}):
        block = _decode_block_cached(
            path, index, block_idx, metric, reference, interpolate=interpolate
        )
        start = block_idx * BLOCK_SIZE
        wanted = frame_ids[(frame_ids >= start) & (frame_ids < start + BLOCK_SIZE)]
        # frame_ids was taken from an earlier observation of the index. If the
        # capture shrank since -- a replaced or truncated live file rebuilds the
        # index from scratch -- ids past the end of the block no longer refer to
        # anything. Drop them rather than raising: the window is a frame or two
        # short for one poll, and the next one sees the rebuilt capture.
        wanted = wanted[wanted - start < len(block)]
        if wanted.size:
            rows.append(block[wanted - start])
    if not rows:
        return np.empty((0, index.num_subcarriers), dtype=np.float32)
    return np.concatenate(rows, axis=0)


# The two planes the presence detector needs, and the reason both are the
# *corrected* ones. The swap correction is not a cosmetic tidy-up here: rx0
# and rx1 trade places on some frames, and uncorrected, 1.2% of frame-to-frame
# steps in the ratio phase exceed pi outright. A pi step is a broadband
# impulse with plenty of energy inside 0.1-0.6 Hz, so on the uncorrected ratio
# the detector would find respiration in an empty room -- manufactured
# entirely by the decode. See backend.ratio.
PRESENCE_METRICS: tuple[str, str] = (
    "csi_ratio_amplitude_corrected",
    "csi_ratio_phase_corrected",
)


def compute_presence(
    path: Path,
    t0: float,
    t1: float,
    *,
    channel: str = "complex",
    window_seconds: float = presence.DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = presence.DEFAULT_HOP_SECONDS,
    rate_band_rpm: tuple[float, float] = presence.DEFAULT_RATE_BAND_RPM,
    bandpass_hz: tuple[float, float] = presence.DEFAULT_BANDPASS_HZ,
    motion_frac_lo: float = presence.DEFAULT_MOTION_FRAC_LO,
    motion_frac_hi: float = presence.DEFAULT_MOTION_FRAC_HI,
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    smooth_windows: int = presence.DEFAULT_SMOOTH_WINDOWS,
    present_threshold: float = presence.DEFAULT_PRESENT_THRESHOLD,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
) -> dict:
    """Motion level and static-presence verdicts for a time range.

    Returns the ``presence_windows`` result with its window centres moved onto
    the capture's own clock, plus the metadata a caller needs to draw it.

    Shares ``compute_doppler``'s shape deliberately -- the same index, the same
    filter mask, the same block cache, the same uniform grid -- so a presence
    panel and a spectrogram over one window decode the capture once between
    them, and so the two panels can never disagree about what a dropout is.

    The sample rate comes from the whole filtered capture rather than from the
    frames in view, for the reason spelled out in ``compute_doppler``: a short
    slice containing a burst reports a rate the capture never ran at, and
    every frequency-derived quantity below -- the respiration band, the
    autocorrelation lags, the reported rate in rpm -- would be scaled by it.
    """
    index = get_index(path)
    times_all = np.asarray(index.times, dtype=float)
    mask = index.filter_mask(mimo=mimo, source_mac=source_mac)

    frame_ids = np.flatnonzero(mask & (times_all >= t0) & (times_all <= t1))
    if frame_ids.size < 2:
        raise ValueError("fewer than 2 frames in range")

    reference = get_reference(
        path, index, path.stat().st_size, mimo=mimo, source_mac=source_mac,
        interpolate=interpolate,
    )
    amplitude_db = _decode_for_doppler(
        path, index, frame_ids, PRESENCE_METRICS[0], reference, interpolate
    )
    phase_rad = _decode_for_doppler(
        path, index, frame_ids, PRESENCE_METRICS[1], reference, interpolate
    )
    ratio = presence.complex_ratio(amplitude_db, phase_rad)

    # Frames carrying no ratio at all -- a single-rx frame has nothing to
    # divide by -- are dropped from the series rather than interpolated
    # through. Dropping them lets the hole they leave be measured as a gap
    # like any other, so a long run of them blanks its windows instead of
    # being silently bridged into a flat stretch.
    usable = np.isfinite(ratio).any(axis=1)
    times = times_all[frame_ids][: ratio.shape[0]][usable]
    ratio = ratio[usable]
    if times.size < 2:
        raise ValueError("no frames in range carry a two-antenna CSI ratio")

    filtered_times = times_all[mask]
    _, fs = uniform_grid(filtered_times if filtered_times.size >= 2 else times)
    step = 1.0 / fs
    n_grid = int(np.floor((times[-1] - times[0]) / step)) + 1
    grid_times = times[0] + np.arange(n_grid, dtype=float) * step

    # Resampled as two real planes rather than as a phase: interpolating a
    # wrapped phase across the +/-pi seam averages the two ends of the circle
    # and lands halfway round it. The angle is taken afterwards, downstream.
    gap_limit = gap_limit_for(times)
    real, fabricated = resample_uniform(times, ratio.real, grid_times, gap_limit)
    imag, _ = resample_uniform(times, ratio.imag, grid_times, gap_limit)

    result = presence.presence_windows(
        real + 1j * imag,
        fs,
        fabricated=fabricated,
        channel=channel,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        rate_band_rpm=rate_band_rpm,
        bandpass_hz=bandpass_hz,
        motion_frac_lo=motion_frac_lo,
        motion_frac_hi=motion_frac_hi,
        max_gap_fraction=max_gap_fraction,
        smooth_windows=smooth_windows,
        present_threshold=present_threshold,
    )

    # Window centres arrive relative to the grid; the caller draws them on the
    # capture's clock, shared with every other panel.
    result["time_s"] = grid_times[0] + result["time_s"]
    result["frames_used"] = int(times.size)
    result["frames_without_ratio"] = int(usable.size - int(usable.sum()))
    result["t_min"] = float(times_all[0]) if times_all.size else 0.0
    result["t_max"] = float(times_all[-1]) if times_all.size else 0.0
    return result


# ----------------------------------------------------------------------- #
#  Lattice chunks                                                         #
# ----------------------------------------------------------------------- #


def _frame_spacing(times: np.ndarray) -> float:
    """Median interval between consecutive frames, or 0.0 if unknowable.

    A property of the capture rather than of any view, which is what makes it
    safe as the lattice floor: a level derived from it cannot change when the
    window moves.
    """
    if len(times) < 2:
        return 0.0
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    return float(np.median(diffs))


def _capture_gap_limit(times: np.ndarray, dt: float) -> float:
    """Longest interval still treated as a sampling gap rather than a dropout.

    Measured over the whole capture, not over the request. The old code took
    the 95th percentile of the *decoded* frames' spacing, which changed with
    the range and the sampling stride — so the same column could be filled in
    one view and left blank in the next. On the lattice this has to be a
    constant of the capture, or cached chunks would disagree with fresh ones.
    """
    if len(times) < 2:
        return dt
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return dt
    return max(2.0 * float(np.percentile(diffs, 95)), dt)


def _reference_tag(reference: Reference | None) -> int | None:
    """Identity of a ``Reference``, for cache keying.

    A chunk corrected against one orientation must not be served once the
    orientation has changed. The reference is re-measured as a capture grows,
    so keying chunks on the file size would discard every corrected chunk on
    every poll; keying on the reference's own contents keeps them exactly as
    long as they stay valid.
    """
    if reference is None:
        return None
    return hash((reference.amp_profile.tobytes(), complex(reference.phase_dir)))


def _column_reduce(
    data: np.ndarray,
    decoded_times: np.ndarray,
    edges: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reduce decoded frames to one value per column.

    Half-open columns throughout: a frame at an edge belongs to the column
    starting there. *data* may carry frames outside ``edges`` — the chunk
    decode includes the frames bracketing its range so a gap at the boundary
    interpolates the same way it would mid-chunk — and those simply fall
    outside every column.

    Returns ``(grid, empty_mask, num_sc)``.
    """
    n_cols = len(edges) - 1
    num_sc = int(data.shape[1]) if data.ndim == 2 and data.shape[0] else 0
    grid = np.full((num_sc, n_cols), np.nan, dtype=np.float32)
    starts = np.searchsorted(decoded_times, edges[:-1], side="left")
    ends = np.searchsorted(decoded_times, edges[1:], side="left")
    if num_sc:
        max_hold = metric in MAX_HOLD_METRICS
        for x in range(n_cols):
            s, e = int(starts[x]), int(ends[x])
            if e <= s:
                continue
            if max_hold:
                grid[:, x] = data[s:e].max(axis=0)
            else:
                centre = 0.5 * (edges[x] + edges[x + 1])
                nearest = s + int(np.argmin(np.abs(decoded_times[s:e] - centre)))
                grid[:, x] = data[nearest]
    return grid, ends <= starts, num_sc


def _decode_selection(
    path: Path,
    index: FrameIndex,
    frame_ids: np.ndarray,
    metric: str,
    reference: Reference | None,
    *,
    contiguous: bool,
    filtered: bool,
    interpolate: bool,
) -> np.ndarray:
    """Decode *frame_ids* into *metric*'s values, through the block cache when
    the selection allows it.

    A contiguous unfiltered run is what the block cache is keyed for. Anything
    else — a filtered selection, or a stride-sampled one — is decoded directly
    and told ``native=False``: its rows are not consecutive frames, so the
    passes that compare a frame with its neighbour have nothing to compare
    against.
    """
    if len(frame_ids) == 0:
        return np.empty((0, index.num_subcarriers), dtype=np.float32)
    if contiguous and not filtered:
        return _decode_via_blocks(
            path, index, frame_ids, metric, reference, interpolate=interpolate
        )
    amp, phase, ratio_amp, ratio_phase = decode_frames(
        path, index, frame_ids, interpolate=interpolate
    )
    available = {
        "amplitude": amp,
        "phase": phase,
        "csi_ratio_amplitude": ratio_amp,
        "csi_ratio_phase": ratio_phase,
    }
    return _materialise(
        metric,
        available,
        index.times[frame_ids],
        reference,
        native=contiguous and reference is not None,
    )


def _compute_chunk(
    path: Path,
    index: FrameIndex,
    filtered_idxs: np.ndarray,
    filtered_times: np.ndarray,
    metric: str,
    level: int,
    chunk_index: int,
    *,
    reference: Reference | None,
    correcting: bool,
    filtered: bool,
    interpolate: bool,
    gap_limit: float,
    num_sc: int,
) -> _Chunk:
    """Build one ``CHUNK_COLUMNS``-wide run of lattice columns.

    Everything here is a function of ``(level, chunk_index)`` and the capture
    — never of the window that asked for it. That is what makes a chunk
    cacheable, and what makes two overlapping requests agree column for
    column.
    """
    dt = lattice_dt(level)
    col0 = chunk_index * CHUNK_COLUMNS
    edges = (col0 + np.arange(CHUNK_COLUMNS + 1, dtype=np.float64)) * dt

    lo = int(np.searchsorted(filtered_times, edges[0], side="left"))
    hi = int(np.searchsorted(filtered_times, edges[-1], side="left"))
    total_in_range = hi - lo

    if total_in_range > CHUNK_FRAME_BUDGET:
        # Stride anchored to the chunk's own frame range, which the lattice
        # fixes. The old budget selected with ``linspace`` over the *request*,
        # so a pan changed which frames were shown even where the window
        # still covered the same data.
        stride = int(math.ceil(total_in_range / CHUNK_FRAME_BUDGET))
        sel = np.arange(lo, hi, stride, dtype=np.int64)
        exact = False
    else:
        sel = np.arange(lo, hi, dtype=np.int64)
        exact = True

    # Frames bracketing the chunk, so a gap spanning its edge interpolates
    # from the same two frames a whole-range fill would have used.
    lead = 1 if lo > 0 else 0
    trail = 1 if hi < len(filtered_times) else 0
    if exact:
        sel_ctx = np.arange(lo - lead, hi + trail, dtype=np.int64)
    elif len(sel):
        sel_ctx = np.unique(
            np.concatenate(
                [np.arange(lo - lead, lo, dtype=np.int64), sel,
                 np.arange(hi, hi + trail, dtype=np.int64)]
            )
        )
    else:
        sel_ctx = np.arange(lo - lead, hi + trail, dtype=np.int64)

    contiguous = bool(
        len(sel_ctx) > 0 and sel_ctx[-1] - sel_ctx[0] + 1 == len(sel_ctx)
    )

    # A correcting metric reads neighbours, so an exact selection is decoded
    # with a margin and trimmed: its edge frames then see the same neighbours
    # a whole-capture pass would give them.
    margin = CONTEXT_FRAMES if (correcting and exact) else 0
    lo_ctx = max(0, int(sel_ctx[0]) - margin) if len(sel_ctx) else 0
    hi_ctx = (
        min(len(filtered_times), int(sel_ctx[-1]) + 1 + margin) if len(sel_ctx) else 0
    )
    if margin and contiguous:
        decode_sel = np.arange(lo_ctx, hi_ctx, dtype=np.int64)
        keep = slice(int(sel_ctx[0]) - lo_ctx, int(sel_ctx[0]) - lo_ctx + len(sel_ctx))
    else:
        decode_sel = sel_ctx
        keep = slice(None)

    frame_ids = filtered_idxs[decode_sel] if filtered else decode_sel
    decode_contiguous = bool(
        len(frame_ids) > 0 and frame_ids[-1] - frame_ids[0] + 1 == len(frame_ids)
    )
    data = _decode_selection(
        path, index, frame_ids, metric, reference,
        contiguous=decode_contiguous, filtered=filtered, interpolate=interpolate,
    )
    data = data[keep]
    kept_ids = filtered_idxs[sel_ctx] if filtered else sel_ctx
    decoded_times = index.times[kept_ids] if len(kept_ids) else np.zeros(0)

    grid, empty, sc = _column_reduce(data, decoded_times, edges, metric)
    if sc == 0:
        grid = np.full((num_sc, CHUNK_COLUMNS), np.nan, dtype=np.float32)

    # A column a filter emptied is an omission, not a sampling gap: a 2x2
    # burst dropped by a '2x1 only' filter has to stay a visible stripe.
    if filtered:
        all_starts = np.searchsorted(index.times, edges[:-1], side="left")
        all_ends = np.searchsorted(index.times, edges[1:], side="left")
        # Frames were there, none of them passed: an omission, not a gap.
        filter_emptied = (all_ends > all_starts) & empty
        fillable = empty & ~filter_emptied
    else:
        fillable = empty

    if interpolate and len(decoded_times) and fillable.any() and grid.shape[0]:
        _interpolate_time_gaps(
            grid,
            fillable,
            data,
            decoded_times,
            float(edges[0]),
            float(edges[-1] - edges[0]),
            CHUNK_COLUMNS,
            gap_limit,
            circular=metric in CIRCULAR_METRICS,
        )

    # A fill is a column that had no frame of its own and came back with
    # values anyway. Deriving it from the grid rather than trusting the
    # returned count keeps the mask and the pixels in agreement.
    filled_mask = fillable & ~np.all(np.isnan(grid), axis=0) if grid.shape[0] else fillable

    return _Chunk(grid, int(len(sel)), total_in_range, exact, filled_mask)


def _chunk_frame_count(
    filtered_times: np.ndarray, level: int, chunk_index: int
) -> int:
    """How many frames fall inside this chunk right now.

    Part of the cache key, which is what makes a chunk safe to cache on a
    growing capture: a chunk is entirely determined by the frames inside it,
    so an entry stays valid exactly as long as that count does. Frames landing
    in the chunk at the growing edge change the count and miss; frames landing
    past it do not, and every settled chunk keeps hitting. This is the rule the
    block cache already uses -- key on what the unit itself holds, never on the
    size of the whole file, which changes on every poll.
    """
    span = chunk_span(level)
    lo = int(np.searchsorted(filtered_times, span * chunk_index, side="left"))
    hi = int(np.searchsorted(filtered_times, span * (chunk_index + 1), side="left"))
    return hi - lo


def _chunk_for(
    path: Path,
    index: FrameIndex,
    filtered_idxs: np.ndarray,
    filtered_times: np.ndarray,
    metric: str,
    level: int,
    chunk_index: int,
    *,
    mimo: tuple[int, int] | None,
    source_mac: str | None,
    reference: Reference | None,
    correcting: bool,
    filtered: bool,
    interpolate: bool,
    gap_limit: float,
    num_sc: int,
) -> _Chunk:
    """Cached ``_compute_chunk``."""
    key = (
        str(path), metric, level, chunk_index,
        mimo, source_mac, interpolate, _reference_tag(reference),
        _chunk_frame_count(filtered_times, level, chunk_index),
    )
    hit = _chunk_cache.get(key)
    if hit is not None:
        return hit
    chunk = _compute_chunk(
        path, index, filtered_idxs, filtered_times, metric, level, chunk_index,
        reference=reference, correcting=correcting, filtered=filtered,
        interpolate=interpolate, gap_limit=gap_limit, num_sc=num_sc,
    )
    _chunk_cache.put(key, chunk)
    return chunk


# ----------------------------------------------------------------------- #
#  Colour scale source                                                    #
# ----------------------------------------------------------------------- #


def _scale_source(
    path: Path,
    index: FrameIndex,
    filtered_idxs: np.ndarray,
    metric: str,
    *,
    mimo: tuple[int, int] | None,
    source_mac: str | None,
    reference: Reference | None,
    filtered: bool,
    interpolate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """A width-independent sample of the capture's values, as (times, values).

    The colour scale must be a property of the data, not of the browser
    window: two laptops on the same capture have to lock to the same scale.
    Reading it off the tile cannot give that any more — the tile's level is a
    function of the requested width — so the scale is measured from frames
    drawn evenly across the whole capture instead, and cached.

    Re-sampled only once the capture has grown by ``STATS_REFRESH_FRAMES``, so
    a live view does not re-measure on every poll.
    """
    n = len(filtered_idxs)
    key = (
        str(path), metric, mimo, source_mac, interpolate,
        n // STATS_REFRESH_FRAMES, _reference_tag(reference),
    )
    with _stats_lock:
        hit = _stats_cache.get(key)
    if hit is not None:
        return hit

    if n == 0:
        empty = (np.zeros(0), np.empty((0, index.num_subcarriers), dtype=np.float32))
        with _stats_lock:
            _stats_cache[key] = empty
        return empty

    picks = np.unique(
        np.linspace(0, n - 1, min(STATS_FRAMES, n)).astype(np.int64)
    )
    frame_ids = filtered_idxs[picks] if filtered else picks
    contiguous = bool(frame_ids[-1] - frame_ids[0] + 1 == len(frame_ids))
    values = _decode_selection(
        path, index, frame_ids, metric, reference,
        contiguous=contiguous, filtered=filtered, interpolate=interpolate,
    )
    sample = (index.times[frame_ids], values)
    with _stats_lock:
        _stats_cache[key] = sample
    return sample


def _scale_bounds(values: np.ndarray) -> tuple[float, float, float, float]:
    """``(vmin, vmax, p_low, p_high)`` over the finite values of *values*.

    ``-inf`` from ``db(0)`` is excluded by the finite mask rather than clamped
    in: clamping would drag ``p_low`` to ``-inf`` and make the robust scale no
    better than the raw minimum.
    """
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return 0.0, 0.0, 0.0, 0.0
    finite_vals = values[finite_mask]
    return (
        float(finite_vals.min()),
        float(finite_vals.max()),
        float(np.nanpercentile(finite_vals, 1)),
        float(np.nanpercentile(finite_vals, 99)),
    )


def compute_tile(
    path: Path,
    t0: float,
    t1: float,
    width: int,
    metric: str,
    *,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
) -> tuple[np.ndarray, dict]:
    """Build a display-resolution grid covering the requested time range.

    Returns ``(grid, metadata)``. *grid* is ``(num_subcarriers, columns)``
    float32, row-major with row 0 = highest subcarrier index (matching the
    frontend's ``subcarrierSourceRect`` convention). Empty columns are NaN;
    ``-inf`` from ``db(0)`` is preserved.

    **The grid is not the window that was asked for.** Columns are quantised
    to the lattice (see ``pick_level``/``snap_window``), so the tile covers
    ``[meta["t0"], meta["t1"]]`` — the smallest lattice-aligned range
    containing ``[t0, t1]`` — at ``meta["dt"]`` seconds per column. The caller
    crops, which ``tileSourceRect`` in the frontend already does. Everything
    the crawling picture came from lives in that one change: a pan now shifts
    columns that keep their values instead of re-aggregating all of them over
    new boundaries, and a live poll appends columns on the right instead of
    re-binning the grid every refresh.

    *metadata* keys: ``t0``, ``t1``, ``dt``, ``level``, ``frames_decoded``,
    ``total_in_range``, ``exact``, ``anchored``, ``vmin``, ``vmax``,
    ``p_low``, ``p_high``, ``t_min``, ``t_max``, ``filled_columns``.

    ``interpolate`` is one flag governing two different axes. Along
    subcarrier, it controls whether structural nulls (pilots, the DC/guard
    band) are filled or left ``NaN`` — see ``batch.decode_frames`` and
    ``mtk.decode_frames``. It reaches every decode this function does,
    including the orientation ``Reference``, and is part of the block, chunk
    and reference cache keys, so toggling it never serves data decoded under
    the other setting. Along time, it controls whether a sampling gap is
    linearly interpolated between its two bracketing frames or left ``NaN``.

    ``mimo`` and ``source_mac`` restrict which frames are eligible for
    decoding. Filtered-out frames leave NaN holes — they are NOT filled from
    neighbours, so a 2x2 burst excluded by a '2x1 only' filter stays visible
    as a stripe. The capture's full extent (``t_min``/``t_max`` in metadata)
    is the unfiltered range so the live view keeps tracking growth; the tile
    window itself reflects the request.
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

    width = max(1, min(width, MAX_TILE_COLUMNS))

    if len(filtered_idxs) == 0 or t1 <= t0:
        grid = np.full((max(num_sc, 0), width), np.nan, dtype=np.float32)
        return grid, {
            "t0": float(t0),
            "t1": float(t1),
            "dt": lattice_dt(0),
            "level": 0,
            "frames_decoded": 0,
            "total_in_range": 0,
            "exact": True,
            "anchored": True,
            "vmin": 0.0,
            "vmax": 0.0,
            "p_low": 0.0,
            "p_high": 0.0,
            "t_min": float(times[0]) if index.count else 0.0,
            "t_max": float(times[-1]) if index.count else 0.0,
            "filled_columns": 0,
        }

    # --- Lattice ------------------------------------------------------- #
    level = pick_level(t1 - t0, width, _frame_spacing(times))
    c0, c1 = snap_window(t0, t1, level)
    while c1 - c0 > MAX_TILE_COLUMNS and level < LATTICE_MAX_LEVEL:
        level += 1
        c0, c1 = snap_window(t0, t1, level)
    dt = lattice_dt(level)

    # Frames the caller asked about, CLOSED at both ends (see /api/meta).
    # Reported as-is: it describes the request, not the snapped tile.
    lo = int(np.searchsorted(filtered_times, t0, side="left"))
    hi = int(np.searchsorted(filtered_times, t1, side="right"))
    total_in_range = hi - lo

    # Metrics that undo the ratio corruption need the capture's own
    # orientation, or their answer is a property of this view rather than of
    # the data — pan or zoom and whole panels invert. See backend.ratio.
    needs_reference = _needs_reference(metric)
    reference = (
        get_reference(
            path, index, path.stat().st_size, mimo=mimo, source_mac=source_mac,
            interpolate=interpolate,
        )
        if needs_reference
        else None
    )
    correcting = reference is not None

    gap_limit = _capture_gap_limit(filtered_times, dt)

    # --- Assemble from chunks ------------------------------------------ #
    k0 = int(math.floor(c0 / CHUNK_COLUMNS))
    k1 = int(math.floor((c1 - 1) / CHUNK_COLUMNS))
    chunks = [
        _chunk_for(
            path, index, filtered_idxs, filtered_times, metric, level, k,
            mimo=mimo, source_mac=source_mac, reference=reference,
            correcting=correcting, filtered=bool(filtered),
            interpolate=interpolate, gap_limit=gap_limit, num_sc=num_sc,
        )
        for k in range(k0, k1 + 1)
    ]
    assembled = (
        chunks[0].grid if len(chunks) == 1
        else np.concatenate([c.grid for c in chunks], axis=1)
    )
    offset = c0 - k0 * CHUNK_COLUMNS
    grid = assembled[:, offset : offset + (c1 - c0)]

    # Flip subcarrier axis so row 0 = highest subcarrier index, matching the
    # frontend's image convention (subcarrierSourceRect in render.ts).
    grid = np.ascontiguousarray(grid[::-1, :])

    # --- Colour scale -------------------------------------------------- #
    # Measured on frames sampled across the capture, never on the grid. A
    # grid column reduces the frames that fall in it — a maximum, for
    # amplitude — and that reduction's distribution depends on how many frames
    # share a column, i.e. on the level, i.e. on the caller's pixel width.
    # Bounds read off the grid would make the colour scale a function of the
    # browser window, and two laptops would lock to different scales for the
    # same capture.
    stat_times, stat_values = _scale_source(
        path, index, filtered_idxs, metric,
        mimo=mimo, source_mac=source_mac, reference=reference,
        filtered=bool(filtered), interpolate=interpolate,
    )
    in_range = (
        (stat_times >= t0) & (stat_times <= t1) if len(stat_times) else np.zeros(0, bool)
    )
    if int(in_range.sum()) >= STATS_MIN_FRAMES or total_in_range == 0:
        vmin, vmax, p_low, p_high = _scale_bounds(stat_values[in_range])
    else:
        # Too narrow a window for the capture-wide sample to describe. The
        # range is small by definition here, so decoding it outright is cheap,
        # and the answer still depends only on the range — not on the width.
        picks = np.unique(
            np.linspace(lo, hi - 1, min(STATS_FRAMES, max(total_in_range, 1)))
            .astype(np.int64)
        )
        frame_ids = filtered_idxs[picks] if filtered else picks
        contiguous = bool(len(frame_ids) and frame_ids[-1] - frame_ids[0] + 1 == len(frame_ids))
        vmin, vmax, p_low, p_high = _scale_bounds(
            _decode_selection(
                path, index, frame_ids, metric, reference,
                contiguous=contiguous, filtered=bool(filtered),
                interpolate=interpolate,
            )
        )

    return grid, {
        # The window this tile actually covers, which is the snapped one.
        "t0": float(c0 * dt),
        "t1": float(c1 * dt),
        "dt": float(dt),
        "level": int(level),
        "frames_decoded": int(sum(c.frames_decoded for c in chunks)),
        "total_in_range": total_in_range,
        "exact": all(c.exact for c in chunks),
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
        # Fills inside the tile's own slice, not inside the chunks it drew
        # from: a chunk overhangs the window at both ends.
        "filled_columns": int(
            np.concatenate([c.filled_mask for c in chunks])[
                offset : offset + (c1 - c0)
            ].sum()
        ),
    }
