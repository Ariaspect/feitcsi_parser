"""Pre-aggregated tile serving for offline capture exploration.

Builds display-resolution grids from arbitrary time ranges of a FeitCSI
capture. Cost per request is bounded: at most ``TILE_FRAME_BUDGET`` frames are
decoded, and the output grid has exactly ``width * num_subcarriers`` cells
regardless of how much of the file the view covers.

Two caches keep repeated work cheap:

* ``FrameIndex`` objects are cached per path (``extend()`` on each request so a
  growing capture stays current without a full rescan).
* Decoded blocks (contiguous runs of ``BLOCK_SIZE`` frames) are cached in an
  LRU keyed by ``(path, metric, block_index, frames_in_that_block)``. Keying
  a block on how many frames *it* holds rather than on the size of the whole
  file is what lets a completed block survive the capture growing: its own
  count can never change again, while the tail block's does. Keying on file size
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

from . import mtk
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
    """Build a display-resolution grid for the requested time range.

    Returns ``(grid, metadata)`` where *grid* has shape
    ``(num_subcarriers, width)``, float32, row-major with row 0 = highest
    subcarrier index (matching the frontend's ``subcarrierSourceRect``
    convention).  Empty columns are NaN; ``-inf`` from ``db(0)`` is preserved.

    *metadata* keys: ``frames_decoded``, ``total_in_range``, ``exact``,
    ``vmin``, ``vmax``, ``p_low``, ``p_high``, ``t_min``, ``t_max``,
    ``filled_columns``.

    ``interpolate`` is one flag governing two different axes. Along
    subcarrier, it controls whether structural nulls (pilots, the DC/guard
    band) are filled or left ``NaN`` — see ``batch.decode_frames`` and
    ``mtk.decode_frames``. It reaches every decode this function does,
    including the orientation ``Reference``, and is part of the block and
    reference cache keys, so toggling it never serves a block decoded under
    the other setting. Along time, it controls whether a sampling gap (see
    "Gap fill" below) is linearly interpolated between its two bracketing
    frames or left ``NaN`` — off means every gap in the data, in either
    axis, stays visible as a gap.

    ``mimo`` and ``source_mac`` restrict which frames are eligible for
    decoding. Filtered-out frames leave NaN holes — they are NOT filled from
    neighbours, so a 2x2 burst excluded by a '2x1 only' filter stays visible
    as a stripe. A filter narrows which columns the gap fill may touch; it
    does not switch the fill off, so a column that held no frames at all is
    still filled as the sampling gap it is. The capture's full extent (``t_min``/``t_max``
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
        get_reference(
            path, index, file_size, mimo=mimo, source_mac=source_mac,
            interpolate=interpolate,
        )
        if needs_reference
        else None
    )
    correcting = reference is not None

    if n_decoded == 0:
        data = np.empty((0, num_sc), dtype=np.float32)
    elif exact and not filtered:
        data = _decode_via_blocks(
            path, index, frame_ids, metric, reference,
            interpolate=interpolate,
        )
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

        amp, phase, ratio_amp, ratio_phase = decode_frames(
            path, index, ctx_ids, interpolate=interpolate
        )
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

    # Gap fill: only sampling gaps (sub-gap-limit intervals) get filled, by
    # linear interpolation in time between the two decoded frames bracketing
    # the gap — not a nearest-frame copy, which would hold each value flat
    # until the next real sample and understate how fast the channel moves
    # between them. Skipped entirely when *interpolate* is off — the whole
    # point of turning it off is to see gaps as gaps, not smoothed over by a
    # guess.
    #
    # A filter narrows *which* columns are eligible, and does not switch the
    # fill off. Frames a filter excluded must stay NaN: a 2x2 burst dropped by
    # a '2x1 only' filter is an intentional omission and has to remain a
    # visible stripe, not get painted over from its neighbours. But a column
    # holding no frames at all is a sampling gap whether or not a filter is
    # set, and there is no reason a sender selection should stop it being
    # filled. The two are told apart by asking the unfiltered frame times
    # whether anything was ever there — without that distinction the only safe
    # move was to disable the fill wholesale, which is what made the
    # interpolate toggle inert as soon as any filter was chosen.
    empty = col_ends <= col_starts
    if filtered:
        all_starts = np.searchsorted(times, col_edges[:-1], side="left")
        all_ends = np.searchsorted(times, col_edges[1:], side="left")
        all_ends[-1] = int(np.searchsorted(times, col_edges[-1], side="right"))
        kept_starts = np.searchsorted(filtered_times, col_edges[:-1], side="left")
        kept_ends = np.searchsorted(filtered_times, col_edges[1:], side="left")
        kept_ends[-1] = int(np.searchsorted(filtered_times, col_edges[-1], side="right"))
        # Frames were there, none of them passed: an omission, not a gap.
        filter_emptied = (all_ends > all_starts) & (kept_ends <= kept_starts)
        fillable = empty & ~filter_emptied
    else:
        fillable = empty
    if n_decoded >= 2:
        gap_limit = 2.0 * float(np.percentile(np.diff(decoded_times), 95))
    else:
        gap_limit = 0.0
    gap_limit = max(gap_limit, span / width)
    if interpolate and n_decoded > 0 and fillable.any():
        filled_columns = _interpolate_time_gaps(
            grid, fillable, data, decoded_times, t0, span, width, gap_limit,
            circular=metric in CIRCULAR_METRICS,
        )
    else:
        filled_columns = 0

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
    #
    # Measured on *data*, not on *grid*: grid is the wrong population. Each of
    # its columns reduces the ~len(data)/width frames that fall in it, and for
    # a MAX_HOLD_METRICS metric that reduction is a maximum — an order
    # statistic whose distribution depends on how many frames share a column,
    # i.e. on the caller's pixel width. Bounds taken from it move with the
    # browser window: on a 2285-frame capture, p_low reads 39.4 at width 900
    # against 38.0 at 2560 (true 37.8), and vmin 17.2 against 0.0, while vmax
    # stays pinned because max-of-max is the one statistic the reduction
    # cannot bias. Two laptops would lock to different color scales for the
    # same capture. *data* holds the frames themselves, whose selection is a
    # function of TILE_FRAME_BUDGET alone, so these bounds are width-invariant.
    # Gap-filled columns would not have added extrema either way — they are
    # convex blends of data rows.
    finite_mask = np.isfinite(data)
    if finite_mask.any():
        finite_vals = data[finite_mask]
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
