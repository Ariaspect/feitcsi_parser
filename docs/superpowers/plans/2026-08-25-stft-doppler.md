# STFT Doppler Panels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add subcarrier-averaged STFT Doppler spectrograms for amplitude and time-unwrapped phase, served from a new `/api/doppler` endpoint and rendered in their own frontend tab, working on both finished and growing captures.

**Architecture:** A new `backend/doppler.py` holds pure functions — resample irregular frame times onto a uniform grid, then per-window detrend + Hann + `rfft`, averaged incrementally across subcarriers. `/api/doppler` returns the **same binary contract as `/api/tile`** (row-major float32 + headers) so the existing canvas renderer is reused; only the row axis changes meaning, exactly as `csi_cir` already does. Live support comes from fixing the block cache's growth invalidation, not from an incremental spectrogram — measurements show the FFT is cheap and decode is the bottleneck.

**Tech Stack:** Python 3.12, numpy (no scipy), FastAPI, uvicorn; React 18.3 + Vite 5.4, TypeScript, raw `<canvas>`, d3-zoom/d3-scale, Base UI primitives (`@base-ui/react`, **not** Radix), vitest, pytest.

**Spec:** No separate spec file — the design was approved in conversation on 2026-08-25 and is restated in full under "Design Decisions" below. Executors read this document alone.

## Global Constraints

- **No new Python dependencies.** `pyproject.toml` declares CSIKit, fastapi, numpy, uvicorn only. There is **no scipy** — implement the STFT with `numpy.fft` and `numpy.hanning`.
- **UI primitives come from `@base-ui/react`**, already a dependency. The `components/ui/` files are thin wrappers over it. Do not add Radix or any second primitive library.
- **No new frontend charting dependency.** Axes, colorbars, and heatmaps are drawn directly onto a raw `<canvas>`. d3-zoom and d3-scale are the only permitted helpers.
- Python ≥3.12. Frontend Node ≥18, vite 5.4.21, vitest `environment: "node"`.
- Backend binary responses are little-endian float32 (`astype("<f4")`), metadata in headers, body stays a bare buffer.
- Custom response headers **must** be added to the `expose_headers` list in `backend/app.py` or browsers hide them from JavaScript and the body is unusable.
- All filesystem access goes through `resolve_capture_path`. Never open a caller-supplied path directly.
- Run the full suite with `uv run pytest -q`; frontend tests with `npm --prefix frontend run test`.

---

## Design Decisions (read before Task 1)

These were settled by measurement. Do not re-litigate them mid-implementation.

**1. Doppler is unsigned.** Amplitude and unwrapped phase are real-valued signals, so their spectra are conjugate-symmetric and negative frequencies carry no new information. Use `numpy.fft.rfft` and a **one-sided 0 … fs/2 axis**. Approaching and receding motion are not distinguishable. Signed Doppler would require the complex CSI, which is out of scope here.

**2. Per-window mean removal is mandatory.** Measured on `captures/20260822_070002.bin`: DC's share of total spectrogram power is 47.8% undetrended and 0.2% with per-window mean removal. Skipping it makes bin 0 swamp the panel.

**3. The resample rate comes from frame timing, never from requested width.** Measured trap: feeding 2048 tile columns spanning 113 s of a 5 Hz capture into an FFT produced a 4.74 Hz "peak" — above that capture's 2.55 Hz Nyquist, so pure interpolation artifact. The same shortcut decimates the MTK captures (2048 cols / 200 s = 10.2 Hz from an 18.3 Hz capture). Derive `fs` from the median inter-frame interval of the frames actually in range.

**4. Frame timing is irregular and must be resampled with gap awareness.** Measured `dt` percentiles (p1/p50/p99) and worst gap:

| capture | frames | PRF | p1/p50/p99 dt | max gap | within ±10% |
|---|---|---|---|---|---|
| `capture.dat` | 1,101 | 5.1 Hz | 0.42 / 197 / 202 ms | 401 ms | 50% |
| `csi_20260813_030001.dat` | 65,219 | 11.6 Hz | 0.17 / 86 / 107 ms | **22,946 ms** | **0.98%** |
| `20260822_070002.bin` | 60,796 | 17.9 Hz | 47 / 56 / 61 ms | 116 ms | 70% |

A 23-second gap must not become signal. Windows spanning a gap larger than the limit are NaN.

**5. Live needs the cache fixed, not an incremental STFT.** Measured costs:

| operation | cost |
|---|---|
| subcarrier-averaged STFT, 60k samples × 256 sc | 229 ms |
| STFT, one new column | 4.7 ms |
| decode cold, 200 s / full capture | 967 / 1206 ms |
| decode warm (block cache hit) | 13 / 382 ms |
| frontend `DEFAULT_REFRESH_MS` | 300 ms |

The FFT is cheap enough to redo wholesale each poll. Decode is not, and `_decode_block_cached` keys on `file_size` (`backend/tiles.py:391`), so **every poll on a growing capture is a cold decode**. Verified directly: the same request cost 2.0 ms warm, then 28.6 ms after appending bytes, against a 39.4 ms cold baseline. Task 1 fixes this and benefits all six existing panels too.

**6. The panel earns its place.** Measured contrast (peak / median, above 0.05 Hz) across six slices per capture: `20260821_170002.bin` shows 0.08–0.28 Hz (5–17 breaths/min) at 2.4–4.2× on every slice; `20260822_070002.bin` is flat at 1.05–1.23×. Occupied vs empty is legible. At these PRFs this is a respiration-band instrument — 1 m/s hand motion sits near 33 Hz at 5 GHz and aliases on every capture on hand.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/doppler.py` (create) | Pure functions: `uniform_grid`, `resample_uniform`, `stft_average`. No I/O, no caching, no FastAPI. |
| `backend/tiles.py` (modify) | Block cache key fix (Task 1); `compute_doppler` orchestration (Task 4) alongside `compute_tile`, reusing `get_index`, filters, and `_decode_block_cached`. |
| `backend/app.py` (modify) | `/api/doppler` route, `X-Doppler-*` headers, `expose_headers` additions. |
| `tests/test_doppler.py` (create) | Pure-function tests: synthetic tones, gap handling, detrending, Nyquist. |
| `tests/test_doppler_api.py` (create) | Endpoint tests: headers, shapes, errors, filters. |
| `tests/test_tiles.py` (modify) | Block cache growth-invalidation tests (Task 1). |
| `frontend/src/api.ts` (modify) | `DopplerTile` type + `fetchDoppler`. |
| `frontend/src/Heatmap.tsx` (modify) | Optional `source` fetcher seam and `yDomain` prop. |
| `frontend/src/components/ui/tabs.tsx` (create) | shadcn-style Tabs primitive; none exists yet. |
| `frontend/src/App.tsx` (modify) | Tab split: existing six panels in one tab, two Doppler panels in another. |
| `README.md` (modify) | `/api/doppler` documentation and a Doppler section. |

---

### Task 1: Block cache survives capture growth

A fully-written block is immutable; only the block holding the tail can change. Keying every block on whole-file size throws away all of them whenever one byte is appended.

**Files:**
- Modify: `backend/tiles.py` — `_decode_block_cached` signature and key, and its call sites
- Test: `tests/test_tiles.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_decode_block_cached(path, index, block_idx, metric, block_frames: int, reference=None, interpolate=True)` — the `file_size: int` parameter is replaced by `block_frames: int`, the number of frames this block currently holds. Task 4 calls the same helper.

- [ ] **Step 1: Write the failing test**

```python
def test_block_cache_survives_growth(tmp_path: Path) -> None:
    """Appending frames must not invalidate blocks that were already complete.

    The cache keyed on whole-file size, so one appended byte re-decoded the
    entire capture -- which made every live poll a cold decode.
    """
    import time
    from backend.tiles import compute_tile, get_index, reset_tile_caches

    src = Path(__file__).resolve().parent.parent / "captures" / "capture.dat"
    if not src.is_file():
        pytest.skip("captures/capture.dat not present")
    raw = src.read_bytes()
    half = len(raw) // 2

    grow = tmp_path / "grow.dat"
    grow.write_bytes(raw[:half])

    reset_tile_caches()
    idx = get_index(grow)
    t0, t1 = float(idx.times[0]), float(idx.times[-1])

    compute_tile(grow, t0, t1, 400, "amplitude")          # cold
    start = time.perf_counter()
    compute_tile(grow, t0, t1, 400, "amplitude")          # warm
    warm = time.perf_counter() - start

    with grow.open("ab") as fh:
        fh.write(raw[half:half + 200_000])
    get_index(grow)                                        # picks up new frames

    start = time.perf_counter()
    compute_tile(grow, t0, t1, 400, "amplitude")          # same window, after growth
    after = time.perf_counter() - start

    # Completed blocks must still hit. Allow generous headroom for the one
    # tail block that legitimately re-decodes, but nothing near a cold pass.
    assert after < warm * 8, f"growth cost {after*1e3:.1f} ms vs warm {warm*1e3:.1f} ms"


def test_block_cache_rebuilds_on_truncation(tmp_path: Path) -> None:
    """A truncated capture must not serve stale blocks from the old file."""
    from backend.tiles import compute_tile, get_index, reset_tile_caches

    src = Path(__file__).resolve().parent.parent / "captures" / "capture.dat"
    if not src.is_file():
        pytest.skip("captures/capture.dat not present")
    raw = src.read_bytes()

    shrink = tmp_path / "shrink.dat"
    shrink.write_bytes(raw)
    reset_tile_caches()
    idx = get_index(shrink)
    t0, t1 = float(idx.times[0]), float(idx.times[-1])
    full, _ = compute_tile(shrink, t0, t1, 200, "amplitude")

    shrink.write_bytes(raw[: len(raw) // 3])
    idx = get_index(shrink)
    small, meta = compute_tile(shrink, float(idx.times[0]), float(idx.times[-1]), 200, "amplitude")

    assert meta["total_in_range"] < full.shape[1] * 4  # fewer frames than before
    assert idx.count > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tiles.py::test_block_cache_survives_growth -v`
Expected: FAIL — the growth pass costs roughly a cold decode, far more than `warm * 8`.

- [ ] **Step 3: Change the cache key from file size to per-block frame count**

In `backend/tiles.py`, replace the `file_size` parameter of `_decode_block_cached` with `block_frames`, and build the key from it:

```python
def _decode_block_cached(
    path: Path,
    index: FrameIndex,
    block_idx: int,
    metric: str,
    block_frames: int,
    reference: Reference | None = None,
    interpolate: bool = True,
) -> np.ndarray:
    """Return the decoded block for one metric, from cache or by decoding.

    Keyed on how many frames *this block* currently holds, not on the size of
    the whole file. A block that is already full has a frame count that can
    never change again, so it stays cached for the life of the process while
    a growing capture appends to the tail. Keying on file size instead threw
    away every block whenever one byte arrived, which made each live poll a
    cold decode of the entire visible window.

    Truncation is handled upstream: ``FrameIndex.extend`` rebuilds, and
    ``get_index`` clears this cache for the path when it does, so a shrunken
    file cannot serve blocks indexed from the old one.
    """
    key = (str(path), metric, block_idx, block_frames, interpolate)
    cached = _block_cache.get(key)
    if cached is not None:
        return cached
    ...
```

Add a helper next to it, and use it at every call site:

```python
def _block_frame_count(index: FrameIndex, block_idx: int) -> int:
    """Frames currently in *block_idx*. Full blocks return BLOCK_SIZE forever."""
    start = block_idx * BLOCK_SIZE
    return max(0, min(BLOCK_SIZE, index.count - start))
```

Replace the `file_size = path.stat().st_size` line in `compute_tile` and pass
`_block_frame_count(index, block_idx)` at each `_decode_block_cached(...)` call
in place of `file_size`. There are call sites around `tiles.py:416`, `:426`,
`:443`, `:470`, `:476`, `:482`, and `:546` — grep for `file_size` and
`_decode_block_cached` and convert every one; leaving a single site passing a
whole-file size silently reintroduces the bug for that path.

- [ ] **Step 4: Clear the block cache when an index rebuilds**

In `backend/tiles.py`, `get_index` currently calls `idx.extend()` unconditionally. Detect a rebuild and drop that path's blocks:

```python
def get_index(path: Path) -> FrameIndex | mtk.MTKIndex:
    path = Path(path)
    with _index_lock:
        idx = _index_cache.get(path)
        if idx is None:
            idx = mtk.MTKIndex(path) if mtk.can_read(path) else FrameIndex(path)
            _index_cache[path] = idx
        else:
            before = idx.count
            idx.extend()
            # extend() rebuilds from scratch on truncation, so a shrinking
            # count means the frame ids the cached blocks were decoded under
            # no longer refer to the same frames.
            if idx.count < before:
                _block_cache.drop_path(str(path))
        return idx
```

Add `drop_path` to the block cache class (find `class _BlockCache` in `tiles.py`):

```python
    def drop_path(self, path: str) -> None:
        """Evict every block belonging to *path*. For truncation/rebuild."""
        with self._lock:
            for key in [k for k in self._entries if k[0] == path]:
                self._bytes -= self._entries.pop(key).nbytes
```

`_BlockCache` (`backend/tiles.py:192`) stores entries in an `OrderedDict` named `_entries` and tracks `_bytes` for its LRU bound — decrement it when evicting, or the cache will believe it is fuller than it is and start evicting live blocks.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_tiles.py -v -k "block_cache"`
Expected: PASS, both tests.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass. This changes shared caching used by every panel — a regression here shows up as wrong pixels, not as an error, so do not proceed on a red suite.

- [ ] **Step 7: Commit**

```bash
git add backend/tiles.py tests/test_tiles.py
git commit -m "perf: key block cache on per-block frame count, not file size"
```

---

### Task 2: Uniform resampling of irregular frame times

**Files:**
- Create: `backend/doppler.py`
- Test: `tests/test_doppler.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DEFAULT_GAP_FACTOR: float = 2.0`
  - `uniform_grid(times: np.ndarray) -> tuple[np.ndarray, float]` — returns `(grid_times, fs)`
  - `resample_uniform(times: np.ndarray, values: np.ndarray, grid_times: np.ndarray, gap_limit: float) -> np.ndarray` — `values` is `(n_frames, n_cols)`, result is `(len(grid_times), n_cols)`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for backend.doppler pure functions."""

from __future__ import annotations

import numpy as np
import pytest

from backend.doppler import resample_uniform, uniform_grid


def test_uniform_grid_uses_median_interval() -> None:
    """fs comes from the median gap, so outliers cannot drag the rate."""
    times = np.array([0.0, 0.1, 0.2, 0.3, 5.0, 5.1, 5.2])
    grid, fs = uniform_grid(times)
    assert fs == pytest.approx(10.0, rel=1e-6)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] <= times[-1] + 1e-9
    assert np.allclose(np.diff(grid), 0.1)


def test_uniform_grid_rejects_too_few_frames() -> None:
    with pytest.raises(ValueError, match="at least"):
        uniform_grid(np.array([0.0]))


def test_resample_interpolates_between_frames() -> None:
    """A linear ramp resamples to the same ramp."""
    times = np.array([0.0, 0.1, 0.2, 0.3])
    values = np.array([[0.0], [1.0], [2.0], [3.0]])
    grid, _ = uniform_grid(times)
    out = resample_uniform(times, values, grid, gap_limit=1.0)
    assert out.shape == (len(grid), 1)
    assert np.allclose(out[:, 0], grid * 10.0, atol=1e-9)


def test_resample_blanks_samples_inside_a_large_gap() -> None:
    """A 5 s hole in a 10 Hz capture must not be interpolated into signal."""
    times = np.array([0.0, 0.1, 0.2, 5.2, 5.3, 5.4])
    values = np.arange(6, dtype=float).reshape(6, 1)
    grid, fs = uniform_grid(times)
    out = resample_uniform(times, values, grid, gap_limit=0.25)

    inside = (grid > 0.2 + 1e-9) & (grid < 5.2 - 1e-9)
    assert inside.any(), "test needs samples strictly inside the gap"
    assert np.all(np.isnan(out[inside, 0]))
    assert np.isfinite(out[grid <= 0.2, 0]).all()


def test_resample_carries_nan_columns_through() -> None:
    """An all-NaN subcarrier stays NaN rather than becoming zero."""
    times = np.array([0.0, 0.1, 0.2, 0.3])
    values = np.full((4, 2), np.nan)
    values[:, 0] = [1.0, 2.0, 3.0, 4.0]
    grid, _ = uniform_grid(times)
    out = resample_uniform(times, values, grid, gap_limit=1.0)
    assert np.isfinite(out[:, 0]).all()
    assert np.isnan(out[:, 1]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_doppler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.doppler'`.

- [ ] **Step 3: Write the implementation**

```python
"""Irregularly-sampled CSI -> uniform time grid -> Doppler spectrogram.

An STFT assumes samples arrive at a constant rate. CSI frames do not: on the
captures this was built against the median interval ranges from 56 ms to
197 ms, the 1st percentile can be under half a millisecond, and one FeitCSI
capture holds a single 23-second hole. So the series is resampled onto a
uniform grid before any FFT touches it.

The grid's rate is taken from the *median* inter-frame interval, never from a
requested display width. Deriving it from width is a trap worth naming: 2048
columns spanning 113 s of a 5 Hz capture implies an 18 Hz sample rate, and the
resulting spectrogram shows peaks above that capture's own 2.55 Hz Nyquist --
entirely manufactured by the interpolator. The same mistake decimates a faster
capture and silently truncates the top of its band.

Real gaps stay holes. A window that spans one comes out NaN rather than
carrying an interpolated ramp that would read as signal.
"""

from __future__ import annotations

import numpy as np

# A gap wider than this many times the 95th-percentile inter-frame interval is
# a dropout rather than jitter. Matches the convention backend.tiles already
# uses for filling tile columns, so the two panels agree about what a hole is.
DEFAULT_GAP_FACTOR = 2.0


def uniform_grid(times: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(grid_times, fs)`` spanning *times* at the median frame rate.

    *times* must be sorted ascending, as a FrameIndex produces them.
    """
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        raise ValueError("need at least 2 frames to infer a sample rate")

    dt = np.diff(times)
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("frames carry no positive time deltas")

    step = float(np.median(dt))
    fs = 1.0 / step
    span = float(times[-1] - times[0])
    n = int(np.floor(span / step)) + 1
    grid = times[0] + np.arange(n, dtype=float) * step
    return grid, fs


def gap_limit_for(times: np.ndarray, factor: float = DEFAULT_GAP_FACTOR) -> float:
    """Longest inter-frame interval still treated as jitter, not a dropout."""
    times = np.asarray(times, dtype=float)
    dt = np.diff(times)
    dt = dt[dt > 0]
    if dt.size == 0:
        return float("inf")
    return float(np.percentile(dt, 95)) * factor


def resample_uniform(
    times: np.ndarray,
    values: np.ndarray,
    grid_times: np.ndarray,
    gap_limit: float,
) -> np.ndarray:
    """Linearly resample ``(n_frames, n_cols)`` *values* onto *grid_times*.

    Samples that land strictly inside a gap wider than *gap_limit* are NaN.
    A column that is entirely NaN in the input stays entirely NaN.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    grid_times = np.asarray(grid_times, dtype=float)

    if values.ndim != 2:
        raise ValueError(f"values must be 2-D (n_frames, n_cols), got {values.shape}")
    if values.shape[0] != times.shape[0]:
        raise ValueError(f"values has {values.shape[0]} rows for {times.shape[0]} times")

    out = np.empty((grid_times.size, values.shape[1]), dtype=float)
    for col in range(values.shape[1]):
        series = values[:, col]
        finite = np.isfinite(series)
        if not finite.any():
            out[:, col] = np.nan
            continue
        # np.interp needs finite samples; a per-column mask keeps one dead
        # subcarrier from blanking its neighbours.
        out[:, col] = np.interp(grid_times, times[finite], series[finite])

    # Blank grid samples that fall strictly inside a real dropout. Computed
    # once against the frame times, not per column, because a gap is a
    # property of when frames arrived rather than of any one subcarrier.
    dt = np.diff(times)
    for start, width in zip(times[:-1], dt):
        if width > gap_limit:
            inside = (grid_times > start) & (grid_times < start + width)
            out[inside, :] = np.nan

    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_doppler.py -v`
Expected: PASS, all five.

- [ ] **Step 5: Commit**

```bash
git add backend/doppler.py tests/test_doppler.py
git commit -m "feat: resample irregular CSI frame times onto a uniform grid"
```

---

### Task 3: Subcarrier-averaged STFT

**Files:**
- Modify: `backend/doppler.py`
- Test: `tests/test_doppler.py`

**Interfaces:**
- Consumes: nothing from Task 2 at call time (independent pure function).
- Produces: `stft_average(samples: np.ndarray, fs: float, win: int, hop: int) -> tuple[np.ndarray, np.ndarray]` returning `(spectrogram, freqs)` where `spectrogram` is `(win // 2 + 1, n_cols)` with **row 0 = highest frequency** and `freqs` is ascending `0 … fs/2`.

- [ ] **Step 1: Write the failing test**

```python
from backend.doppler import stft_average


def _tone(freq: float, fs: float, n: int, n_cols: int = 1) -> np.ndarray:
    t = np.arange(n) / fs
    return np.tile(np.sin(2 * np.pi * freq * t)[:, None], (1, n_cols))


def test_stft_finds_a_known_tone() -> None:
    """A 2 Hz tone at 20 Hz lands in the 2 Hz bin, in every column."""
    fs, win, hop = 20.0, 128, 64
    samples = _tone(2.0, fs, 2048, n_cols=4)
    spec, freqs = stft_average(samples, fs, win, hop)

    assert spec.shape[0] == win // 2 + 1
    assert freqs[0] == pytest.approx(0.0)
    assert freqs[-1] == pytest.approx(fs / 2)
    # Row 0 is the HIGHEST frequency, matching the tile contract's row order.
    peak_row = int(np.argmax(spec.mean(axis=1)))
    assert freqs[::-1][peak_row] == pytest.approx(2.0, abs=fs / win)


def test_stft_removes_the_mean_per_window() -> None:
    """A large DC offset must not dominate; detrending is not optional."""
    fs, win, hop = 20.0, 128, 64
    samples = _tone(2.0, fs, 2048) + 1000.0
    spec, freqs = stft_average(samples, fs, win, hop)
    profile = spec.mean(axis=1)
    dc_row = int(np.argmin(np.abs(freqs[::-1] - 0.0)))
    assert profile[dc_row] < profile.max() * 0.1


def test_stft_propagates_nan_windows() -> None:
    """A window containing a resampling hole is NaN, not zero-filled."""
    fs, win, hop = 20.0, 128, 64
    samples = _tone(2.0, fs, 512)
    samples[200:260, :] = np.nan
    spec, _ = stft_average(samples, fs, win, hop)
    assert np.isnan(spec).any()
    assert np.isfinite(spec).any(), "only the affected columns should be NaN"


def test_stft_averages_across_columns() -> None:
    """Two subcarriers with tones at different frequencies both appear."""
    fs, win, hop = 20.0, 256, 128
    samples = np.concatenate([_tone(2.0, fs, 2048), _tone(5.0, fs, 2048)], axis=1)
    spec, freqs = stft_average(samples, fs, win, hop)
    profile = spec.mean(axis=1)[::-1]  # ascending frequency
    for want in (2.0, 5.0):
        near = np.abs(freqs - want) < fs / win
        assert profile[near].max() > np.median(profile) * 3


def test_stft_rejects_a_window_longer_than_the_series() -> None:
    with pytest.raises(ValueError, match="shorter than"):
        stft_average(_tone(2.0, 20.0, 64), 20.0, 128, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_doppler.py -v -k stft`
Expected: FAIL — `ImportError: cannot import name 'stft_average'`.

- [ ] **Step 3: Write the implementation**

Append to `backend/doppler.py`:

```python
def stft_average(
    samples: np.ndarray,
    fs: float,
    win: int,
    hop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Subcarrier-averaged magnitude spectrogram of ``(n_samples, n_cols)``.

    Returns ``(spectrogram, freqs)``. *spectrogram* is
    ``(win // 2 + 1, n_cols_out)`` with **row 0 = highest frequency**, matching
    the tile contract's "row 0 = highest subcarrier" convention so the same
    renderer draws both. *freqs* is ascending, ``0 .. fs/2``.

    Three things the maths requires, each of which silently ruins the output
    if skipped:

    * **Real input means a one-sided spectrum.** Amplitude and unwrapped phase
      are real, so their spectra are conjugate-symmetric and the negative half
      carries no information. ``rfft`` is correct, and the consequence is that
      Doppler here is *unsigned*: approaching and receding motion are not
      distinguishable. Signed Doppler needs the complex CSI.

    * **The per-window mean must go.** Measured on a real capture, DC holds
      47.8% of total power undetrended and 0.2% detrended. Leaving it in makes
      bin 0 swamp every panel.

    * **A NaN anywhere in a window poisons that whole column**, because each
      output bin sums over the entire window. That is the correct behaviour --
      the window spans a real dropout -- so the column is left NaN rather than
      zero-filled, which would report silence as signal.

    Subcarriers are accumulated one at a time rather than stacked. The stacked
    form is ``(n_cols, n_windows, win)``, which for a full capture is a
    quarter of a gigabyte; accumulating holds one subcarrier's windows.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2-D (n_samples, n_cols), got {samples.shape}")
    if win < 2:
        raise ValueError("win must be at least 2")
    if hop < 1:
        raise ValueError("hop must be at least 1")
    if samples.shape[0] < win:
        raise ValueError(
            f"series of {samples.shape[0]} samples is shorter than the {win}-sample window"
        )

    n_samples, n_cols = samples.shape
    n_out = (n_samples - win) // hop + 1
    starts = np.arange(n_out) * hop
    offsets = np.arange(win)
    taper = np.hanning(win)

    acc = np.zeros((win // 2 + 1, n_out), dtype=float)
    for col in range(n_cols):
        seg = samples[:, col][starts[:, None] + offsets]     # (n_out, win)
        seg = seg - seg.mean(axis=1, keepdims=True)          # detrend per window
        acc += np.abs(np.fft.rfft(seg * taper, axis=1)).T

    spec = acc / n_cols
    freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    # Row 0 = highest frequency, so the renderer's existing top-down row order
    # puts fast motion at the top of the panel.
    return spec[::-1, :], freqs
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_doppler.py -v`
Expected: PASS, all nine.

- [ ] **Step 5: Commit**

```bash
git add backend/doppler.py tests/test_doppler.py
git commit -m "feat: subcarrier-averaged STFT with per-window detrending"
```

---

### Task 4: `compute_doppler` orchestration

**Files:**
- Modify: `backend/tiles.py`
- Test: `tests/test_doppler.py`

**Interfaces:**
- Consumes: `uniform_grid`, `gap_limit_for`, `resample_uniform`, `stft_average` (Tasks 2–3); `_decode_block_cached` and `_block_frame_count` (Task 1).
- Produces:
  - `DOPPLER_METRICS: tuple[str, ...] = ("amplitude", "csi_ratio_phase_time_unwrapped")`
  - `compute_doppler(path, t0, t1, metric, *, win_seconds=30.0, overlap=0.5, mimo=None, source_mac=None, interpolate=True) -> tuple[np.ndarray, dict]`
  - metadata keys: `fs`, `f_max`, `win`, `hop`, `win_seconds`, `frames_used`, `t_min`, `t_max`, `col_t0`, `col_t1`, `vmin`, `vmax`, `p_low`, `p_high`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

CAPTURES = Path(__file__).resolve().parent.parent / "captures"


def _capture_or_skip(name: str) -> Path:
    p = CAPTURES / name
    if not p.is_file():
        pytest.skip(f"{name} not present")
    return p


def test_compute_doppler_shape_and_metadata() -> None:
    from backend.tiles import compute_doppler, get_index

    p = _capture_or_skip("capture.dat")
    idx = get_index(p)
    t0, t1 = float(idx.times[0]), float(idx.times[-1])

    spec, meta = compute_doppler(p, t0, t1, "amplitude", win_seconds=10.0)

    assert spec.ndim == 2
    assert spec.shape[0] == meta["win"] // 2 + 1
    assert spec.shape[1] >= 1
    assert meta["f_max"] == pytest.approx(meta["fs"] / 2)
    assert meta["hop"] == meta["win"] // 2          # 0.5 overlap
    assert meta["frames_used"] > 0
    assert meta["col_t1"] <= t1 + 1e-6


def test_compute_doppler_rejects_unknown_metric() -> None:
    from backend.tiles import compute_doppler

    p = _capture_or_skip("capture.dat")
    with pytest.raises(ValueError, match="metric"):
        compute_doppler(p, 0.0, 1e9, "csi_cir")


def test_compute_doppler_window_never_exceeds_the_range() -> None:
    """Asking for a 10-minute window over 30 s of data is an error, not a crash."""
    from backend.tiles import compute_doppler, get_index

    p = _capture_or_skip("capture.dat")
    idx = get_index(p)
    t0 = float(idx.times[0])
    with pytest.raises(ValueError, match="shorter than"):
        compute_doppler(p, t0, t0 + 5.0, "amplitude", win_seconds=600.0)


def test_compute_doppler_nyquist_matches_frame_rate() -> None:
    """f_max is half the median frame rate -- the capture's real ceiling."""
    import numpy as np

    from backend.tiles import compute_doppler, get_index

    p = _capture_or_skip("capture.dat")
    idx = get_index(p)
    t = np.asarray(idx.times, dtype=float)
    t0, t1 = float(t[0]), float(t[-1])
    expected_fs = 1.0 / float(np.median(np.diff(t)[np.diff(t) > 0]))

    _, meta = compute_doppler(p, t0, t1, "amplitude", win_seconds=10.0)
    assert meta["fs"] == pytest.approx(expected_fs, rel=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_doppler.py -v -k compute_doppler`
Expected: FAIL — `ImportError: cannot import name 'compute_doppler'`.

- [ ] **Step 3: Write the implementation**

Add to `backend/tiles.py`, next to `compute_tile`:

```python
from .doppler import gap_limit_for, resample_uniform, stft_average, uniform_grid

# Doppler is computed on real-valued series only. Amplitude is the raw
# channel's magnitude; the phase panel is built on the *time-unwrapped* ratio
# phase because raw phase is wrapped, and its 2*pi jumps are broadband steps
# that would dominate an FFT and read as motion that is not there.
DOPPLER_METRICS: tuple[str, ...] = ("amplitude", "csi_ratio_phase_time_unwrapped")


def compute_doppler(
    path: Path,
    t0: float,
    t1: float,
    metric: str,
    *,
    win_seconds: float = 30.0,
    overlap: float = 0.5,
    mimo: tuple[int, int] | None = None,
    source_mac: str | None = None,
    interpolate: bool = True,
) -> tuple[np.ndarray, dict]:
    """Subcarrier-averaged Doppler spectrogram for a time range.

    Returns ``(spectrogram, metadata)``. The grid is
    ``(win // 2 + 1, n_windows)`` float32, row 0 = highest Doppler frequency,
    which is the same row order ``compute_tile`` uses so the same renderer
    draws it.

    The sample rate is the median frame rate over the frames actually in
    range, never a function of any requested display width -- resampling a
    5 Hz capture onto a wider grid manufactures peaks above its own Nyquist.
    The window is specified in *seconds* for that reason: frame rate varies
    per capture, so a fixed frame count would mean a different physical
    window on every file.
    """
    if metric not in DOPPLER_METRICS:
        raise ValueError(
            "metric must be one of: " + ", ".join(repr(m) for m in DOPPLER_METRICS)
        )
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    index = get_index(path)
    mask = index.filter_mask(mimo=mimo, source_mac=source_mac)
    times_all = np.asarray(index.times, dtype=float)

    in_range = mask & (times_all >= t0) & (times_all <= t1)
    frame_ids = np.flatnonzero(in_range)
    if frame_ids.size < 2:
        raise ValueError("fewer than 2 frames in range")

    times = times_all[frame_ids]
    values = _decode_for_doppler(path, index, frame_ids, metric, interpolate)

    grid_times, fs = uniform_grid(times)
    win = int(round(win_seconds * fs))
    if win < 2:
        raise ValueError(f"win_seconds={win_seconds} is under two samples at {fs:.2f} Hz")
    if grid_times.size < win:
        raise ValueError(
            f"range holds {grid_times.size} samples, shorter than the "
            f"{win}-sample ({win_seconds:.1f} s) window"
        )

    samples = resample_uniform(times, values, grid_times, gap_limit_for(times))
    hop = max(1, int(round(win * (1.0 - overlap))))
    spec, freqs = stft_average(samples, fs, win, hop)

    finite = spec[np.isfinite(spec)]
    n_out = spec.shape[1]
    step = 1.0 / fs
    return spec.astype(np.float32), {
        "fs": float(fs),
        "f_max": float(freqs[-1]),
        "win": int(win),
        "hop": int(hop),
        "win_seconds": float(win / fs),
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
    interpolate: bool,
) -> np.ndarray:
    """Decode *metric* for *frame_ids*, as ``(n_frames, n_subcarriers)``.

    Goes through the same block cache the tile path uses, so a Doppler panel
    and a heatmap over the same window share one decode.
    """
    rows: list[np.ndarray] = []
    for block_idx in sorted({int(i) // BLOCK_SIZE for i in frame_ids}):
        block = _decode_block_cached(
            path,
            index,
            block_idx,
            metric,
            _block_frame_count(index, block_idx),
            interpolate=interpolate,
        )
        start = block_idx * BLOCK_SIZE
        wanted = frame_ids[(frame_ids >= start) & (frame_ids < start + BLOCK_SIZE)]
        rows.append(block[wanted - start])
    return np.concatenate(rows, axis=0)
```

If `_decode_block_cached` needs a `Reference` for `csi_ratio_phase_time_unwrapped` (it has `needs_reference` one step removed — check `_needs_reference` in `tiles.py:129`), build the reference the same way `compute_tile` does and pass it through. Mirror `compute_tile`'s existing reference construction rather than inventing a second path.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_doppler.py -v`
Expected: PASS.

- [ ] **Step 5: Verify against real data that the respiration line appears**

Run:

```bash
uv run python -c "
import numpy as np
from pathlib import Path
from backend.tiles import compute_doppler, get_index
p = Path('captures/20260821_170002.bin')
idx = get_index(p); t = np.asarray(idx.times, float)
spec, meta = compute_doppler(p, t[0], t[0]+600, 'amplitude', win_seconds=30.0)
freqs = np.linspace(meta['f_max'], 0, spec.shape[0])
prof = np.nanmean(spec, axis=1)
band = freqs < 1.0
print('fs', round(meta['fs'],2), 'f_max', round(meta['f_max'],2), 'shape', spec.shape)
print('peak below 1 Hz:', round(float(freqs[band][np.nanargmax(prof[band])]),3), 'Hz')
print('contrast:', round(float(np.nanmax(prof[band])/np.nanmedian(prof[band])),2))
"
```

Expected: a peak between roughly 0.08 and 0.3 Hz with contrast above 2. If the peak is at 0 Hz or contrast is near 1, detrending or the row order is wrong — fix before continuing. Skip this step if that capture is absent.

- [ ] **Step 6: Commit**

```bash
git add backend/tiles.py tests/test_doppler.py
git commit -m "feat: compute_doppler orchestration over the shared block cache"
```

---

### Task 5: `/api/doppler` endpoint

**Files:**
- Modify: `backend/app.py`
- Test: `tests/test_doppler_api.py`

**Interfaces:**
- Consumes: `compute_doppler`, `DOPPLER_METRICS` (Task 4); `resolve_capture_path`.
- Produces: `GET /api/doppler` with query params `path`, `t0`, `t1`, `metric`, `win_seconds`, `overlap`, `mimo`, `source_mac`, `interpolate`; body is little-endian float32; headers `X-Doppler-Width`, `X-Doppler-Height`, `X-Doppler-Fs`, `X-Doppler-FMax`, `X-Doppler-Win`, `X-Doppler-Hop`, `X-Doppler-WinSeconds`, `X-Doppler-Frames`, `X-Doppler-ColT0`, `X-Doppler-ColT1`, `X-Capture-TMin`, `X-Capture-TMax`, `X-Tile-VMin`, `X-Tile-VMax`, `X-Tile-PLow`, `X-Tile-PHigh`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the /api/doppler endpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app

CAPTURES = Path(__file__).resolve().parent.parent / "captures"


def _capture_or_skip(name: str = "capture.dat") -> Path:
    p = CAPTURES / name
    if not p.is_file():
        pytest.skip(f"{name} not present")
    return p


def test_doppler_returns_a_float32_grid_matching_its_headers() -> None:
    from backend.tiles import get_index

    p = _capture_or_skip()
    idx = get_index(p)
    client = TestClient(app)
    r = client.get("/api/doppler", params={
        "path": str(p), "t0": float(idx.times[0]), "t1": float(idx.times[-1]),
        "metric": "amplitude", "win_seconds": 10.0,
    })
    assert r.status_code == 200
    h = r.headers
    w = int(h["X-Doppler-Width"])
    ht = int(h["X-Doppler-Height"])
    grid = np.frombuffer(r.content, dtype="<f4")
    assert grid.size == w * ht
    assert float(h["X-Doppler-FMax"]) == pytest.approx(float(h["X-Doppler-Fs"]) / 2)
    assert int(h["X-Doppler-Win"]) // 2 + 1 == ht
    assert float(h["X-Doppler-ColT0"]) <= float(h["X-Doppler-ColT1"])


def test_doppler_rejects_a_tile_only_metric() -> None:
    p = _capture_or_skip()
    r = TestClient(app).get("/api/doppler", params={
        "path": str(p), "t0": 0, "t1": 1e9, "metric": "csi_cir",
    })
    assert r.status_code == 400
    assert "metric" in r.json()["detail"]


def test_doppler_rejects_a_window_longer_than_the_range() -> None:
    from backend.tiles import get_index

    p = _capture_or_skip()
    idx = get_index(p)
    t0 = float(idx.times[0])
    r = TestClient(app).get("/api/doppler", params={
        "path": str(p), "t0": t0, "t1": t0 + 5.0,
        "metric": "amplitude", "win_seconds": 600.0,
    })
    assert r.status_code == 400
    assert "shorter than" in r.json()["detail"]


def test_doppler_404s_on_a_path_outside_the_capture_roots() -> None:
    r = TestClient(app).get("/api/doppler", params={
        "path": "/etc/hostname", "t0": 0, "t1": 1e9, "metric": "amplitude",
    })
    assert r.status_code == 404


def test_doppler_headers_are_exposed_to_browsers() -> None:
    """Without expose_headers the body is unusable from JavaScript."""
    from backend.app import app as fastapi_app

    exposed = None
    for mw in fastapi_app.user_middleware:
        if "CORS" in str(mw.cls):
            exposed = set(mw.kwargs["expose_headers"])
    assert exposed is not None
    for name in ("X-Doppler-Width", "X-Doppler-Height", "X-Doppler-Fs",
                 "X-Doppler-FMax", "X-Doppler-Win", "X-Doppler-Hop",
                 "X-Doppler-WinSeconds", "X-Doppler-Frames",
                 "X-Doppler-ColT0", "X-Doppler-ColT1"):
        assert name in exposed, f"{name} missing from expose_headers"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_doppler_api.py -v`
Expected: FAIL — 404 from FastAPI, route not registered.

- [ ] **Step 3: Add the ten header names to `expose_headers`**

In `backend/app.py`, extend the existing `expose_headers` list in the `CORSMiddleware` block with:

```python
        "X-Doppler-Width",
        "X-Doppler-Height",
        "X-Doppler-Fs",
        "X-Doppler-FMax",
        "X-Doppler-Win",
        "X-Doppler-Hop",
        "X-Doppler-WinSeconds",
        "X-Doppler-Frames",
        "X-Doppler-ColT0",
        "X-Doppler-ColT1",
```

- [ ] **Step 4: Add the route**

In `backend/app.py`, after the `/api/tile` handler:

```python
@app.get("/api/doppler")
def doppler(
    path: str = Query(..., description="Path to capture file"),
    t0: float = Query(..., description="Start of requested time window (seconds)"),
    t1: float = Query(..., description="End of requested time window (seconds)"),
    metric: str = Query("amplitude", description=f"One of: {', '.join(DOPPLER_METRICS)}"),
    win_seconds: float = Query(30.0, gt=0, le=600, description="STFT window length in seconds"),
    overlap: float = Query(0.5, ge=0.0, lt=1.0, description="Window overlap fraction"),
    mimo: str | None = Query(None, description="MIMO filter: 'all' or 'NxM'"),
    source_mac: str | None = Query(None, description="Source MAC filter"),
    interpolate: bool = Query(True, description="Fill structural subcarrier nulls before transforming"),
) -> Response:
    """Subcarrier-averaged Doppler spectrogram, as raw little-endian float32.

    Body is a bare ``(win // 2 + 1, n_windows)`` array, row-major, row 0 =
    highest Doppler frequency -- the same row order ``/api/tile`` uses.

    The frequency axis runs 0 to ``X-Doppler-FMax`` and is **one-sided**:
    amplitude and unwrapped phase are real signals, so their spectra are
    symmetric and the sign of the Doppler shift is not recoverable.

    ``X-Doppler-Fs`` is the capture's own median frame rate over the frames in
    range, so ``FMax`` is this file's true Nyquist rather than a function of
    the requested width.
    """
    p = resolve_capture_path(path)

    try:
        mimo_filter = parse_mimo_filter(mimo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        spec, meta = compute_doppler(
            p, t0, t1, metric,
            win_seconds=win_seconds,
            overlap=overlap,
            mimo=mimo_filter,
            source_mac=parse_mac_filter(source_mac),
            interpolate=interpolate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=spec.astype("<f4", copy=False).tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Doppler-Width": str(spec.shape[1]),
            "X-Doppler-Height": str(spec.shape[0]),
            "X-Doppler-Fs": str(meta["fs"]),
            "X-Doppler-FMax": str(meta["f_max"]),
            "X-Doppler-Win": str(meta["win"]),
            "X-Doppler-Hop": str(meta["hop"]),
            "X-Doppler-WinSeconds": str(meta["win_seconds"]),
            "X-Doppler-Frames": str(meta["frames_used"]),
            "X-Doppler-ColT0": str(meta["col_t0"]),
            "X-Doppler-ColT1": str(meta["col_t1"]),
            "X-Capture-TMin": str(meta["t_min"]),
            "X-Capture-TMax": str(meta["t_max"]),
            "X-Tile-VMin": str(meta["vmin"]),
            "X-Tile-VMax": str(meta["vmax"]),
            "X-Tile-PLow": str(meta["p_low"]),
            "X-Tile-PHigh": str(meta["p_high"]),
        },
    )
```

Add `compute_doppler` and `DOPPLER_METRICS` to the existing `from .tiles import ...` line.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_doppler_api.py -v && uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app.py tests/test_doppler_api.py
git commit -m "feat: /api/doppler endpoint serving spectrograms as float32 tiles"
```

---

### Task 6: Frontend `fetchDoppler`

**Files:**
- Modify: `frontend/src/api.ts`
- Test: `frontend/src/api.test.ts`

**Interfaces:**
- Consumes: `/api/doppler` (Task 5), the existing `Tile` interface and `filterParams` helper.
- Produces:
  - `export type DopplerMetric = "amplitude" | "csi_ratio_phase_time_unwrapped"`
  - `export interface DopplerTile extends Tile { fs: number; fMax: number; win: number; hop: number; winSeconds: number }`
  - `export async function fetchDoppler(path, t0, t1, metric, winSeconds, signal?, mimo?, sourceMac?, interpolate?): Promise<DopplerTile>`

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/api.test.ts`:

```ts
import { describe, expect, it, vi, afterEach } from "vitest";
import { fetchDoppler } from "./api";

function dopplerResponse(width: number, height: number) {
  const grid = new Float32Array(width * height).fill(1);
  return new Response(grid.buffer, {
    status: 200,
    headers: {
      "X-Doppler-Width": String(width),
      "X-Doppler-Height": String(height),
      "X-Doppler-Fs": "17.86",
      "X-Doppler-FMax": "8.93",
      "X-Doppler-Win": String((height - 1) * 2),
      "X-Doppler-Hop": String(height - 1),
      "X-Doppler-WinSeconds": "28.7",
      "X-Doppler-Frames": "5524",
      "X-Doppler-ColT0": "15.0",
      "X-Doppler-ColT1": "185.0",
      "X-Capture-TMin": "0",
      "X-Capture-TMax": "200",
      "X-Tile-VMin": "0",
      "X-Tile-VMax": "10",
      "X-Tile-PLow": "1",
      "X-Tile-PHigh": "9",
    },
  });
}

describe("fetchDoppler", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("parses the grid and the doppler headers", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => dopplerResponse(20, 129)));
    const tile = await fetchDoppler("c.dat", 0, 200, "amplitude", 30);
    expect(tile.width).toBe(20);
    expect(tile.height).toBe(129);
    expect(tile.grid.length).toBe(20 * 129);
    expect(tile.fMax).toBeCloseTo(8.93);
    expect(tile.fs).toBeCloseTo(17.86);
    expect(tile.winSeconds).toBeCloseTo(28.7);
    // Column centres, not the requested range -- the panel labels its x-axis
    // from these so the first column is not drawn half a window too early.
    expect(tile.t0).toBeCloseTo(15.0);
    expect(tile.t1).toBeCloseTo(185.0);
  });

  it("sends win_seconds and the filter params", async () => {
    const spy = vi.fn(async () => dopplerResponse(4, 5));
    vi.stubGlobal("fetch", spy);
    await fetchDoppler("c.dat", 1, 2, "amplitude", 45, undefined, "2x2", "aa:bb");
    const url = String(spy.mock.calls[0][0]);
    expect(url).toContain("/api/doppler?");
    expect(url).toContain("win_seconds=45");
    expect(url).toContain("metric=amplitude");
    expect(url).toContain("mimo=2x2");
  });

  it("throws with the server detail on an error status", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("too short", { status: 400 })));
    await expect(fetchDoppler("c.dat", 0, 1, "amplitude", 600)).rejects.toThrow(/400/);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm --prefix frontend run test -- api.test.ts`
Expected: FAIL — `fetchDoppler` is not exported.

- [ ] **Step 3: Write the implementation**

Append to `frontend/src/api.ts`:

```ts
/** Metrics /api/doppler accepts. Both are real-valued series: amplitude, and
 *  the time-unwrapped ratio phase. Raw wrapped phase is deliberately absent —
 *  its 2π jumps are broadband steps that dominate an FFT. */
export type DopplerMetric = "amplitude" | "csi_ratio_phase_time_unwrapped";

export interface DopplerTile extends Tile {
  /** The capture's own median frame rate over the frames in range. */
  fs: number;
  /** Nyquist, fs/2. The frequency axis runs 0..fMax and is one-sided:
   *  real input means the sign of the Doppler shift is not recoverable. */
  fMax: number;
  win: number;
  hop: number;
  winSeconds: number;
}

export async function fetchDoppler(
  path: string,
  t0: number,
  t1: number,
  metric: DopplerMetric,
  winSeconds: number,
  signal?: AbortSignal,
  mimo?: string | null,
  sourceMac?: string | null,
  interpolate?: boolean,
): Promise<DopplerTile> {
  const url =
    `/api/doppler?path=${encodeURIComponent(path)}` +
    `&t0=${t0}&t1=${t1}&metric=${metric}&win_seconds=${winSeconds}` +
    filterParams(mimo, sourceMac) +
    (interpolate === false ? "&interpolate=false" : "");
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  const grid = new Float32Array(await res.arrayBuffer());
  const h = res.headers;
  return {
    grid,
    width: parseInt(h.get("X-Doppler-Width") ?? "0", 10),
    height: parseInt(h.get("X-Doppler-Height") ?? "0", 10),
    // Column centres sit half a window inside the requested range, so the
    // panel's own extent is what the backend reports, not what was asked for.
    t0: parseFloat(h.get("X-Doppler-ColT0") ?? "0"),
    t1: parseFloat(h.get("X-Doppler-ColT1") ?? "0"),
    captureTMin: parseFloat(h.get("X-Capture-TMin") ?? "0"),
    captureTMax: parseFloat(h.get("X-Capture-TMax") ?? "0"),
    framesDecoded: parseInt(h.get("X-Doppler-Frames") ?? "0", 10),
    totalInRange: parseInt(h.get("X-Doppler-Frames") ?? "0", 10),
    exact: true,
    anchored: true,
    vmin: parseFloat(h.get("X-Tile-VMin") ?? "0"),
    vmax: parseFloat(h.get("X-Tile-VMax") ?? "0"),
    pLow: parseFloat(h.get("X-Tile-PLow") ?? "0"),
    pHigh: parseFloat(h.get("X-Tile-PHigh") ?? "0"),
    fs: parseFloat(h.get("X-Doppler-Fs") ?? "0"),
    fMax: parseFloat(h.get("X-Doppler-FMax") ?? "0"),
    win: parseInt(h.get("X-Doppler-Win") ?? "0", 10),
    hop: parseInt(h.get("X-Doppler-Hop") ?? "0", 10),
    winSeconds: parseFloat(h.get("X-Doppler-WinSeconds") ?? "0"),
  };
}
```

- [ ] **Step 4: Run the tests**

Run: `npm --prefix frontend run test -- api.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.ts frontend/src/api.test.ts
git commit -m "feat: fetchDoppler client for /api/doppler"
```

---

### Task 7: Heatmap y-axis domain and fetcher seam

`Heatmap` hardcodes `fetchTile` and labels its y-axis by subcarrier index. Doppler needs a different fetcher and a y-axis in Hz. Add two optional props; every existing call site keeps its current behaviour by omitting them.

**Files:**
- Modify: `frontend/src/Heatmap.tsx`
- Test: `frontend/src/heatmap.test.ts` (create)

**Interfaces:**
- Consumes: `fetchDoppler`, `DopplerTile` (Task 6).
- Produces: two new optional `HeatmapProps` fields:
  - `source?: (t0: number, t1: number, width: number, signal: AbortSignal) => Promise<Tile>` — defaults to the existing `fetchTile` call.
  - `yDomain?: [number, number]` — `[valueAtRow0, valueAtLastRow]`. When present the y-axis labels interpolate this domain instead of showing subcarrier indices.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/heatmap.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { yAxisTicks } from "./Heatmap";

describe("yAxisTicks", () => {
  it("labels subcarrier indices when no domain is given", () => {
    const ticks = yAxisTicks(256, undefined, 4);
    expect(ticks[0].label).toBe("255");
    expect(ticks[ticks.length - 1].label).toBe("0");
  });

  it("interpolates a Hz domain from row 0 downward", () => {
    // Row 0 is the highest frequency, matching the backend's row order.
    const ticks = yAxisTicks(129, [8.93, 0], 4);
    expect(parseFloat(ticks[0].label)).toBeCloseTo(8.93, 2);
    expect(parseFloat(ticks[ticks.length - 1].label)).toBeCloseTo(0, 2);
    const values = ticks.map((t) => parseFloat(t.label));
    for (let i = 1; i < values.length; i++) {
      expect(values[i]).toBeLessThan(values[i - 1]);
    }
  });

  it("keeps enough precision for a sub-Hz respiration line", () => {
    const ticks = yAxisTicks(129, [8.93, 0], 10);
    const labels = ticks.map((t) => t.label);
    expect(new Set(labels).size).toBe(labels.length);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm --prefix frontend run test -- heatmap.test.ts`
Expected: FAIL — `yAxisTicks` is not exported.

- [ ] **Step 3: Extract and export the tick helper**

In `frontend/src/Heatmap.tsx`, add near the other module-level helpers:

```ts
export interface AxisTick {
  /** Fractional position down the plot, 0 at the top row. */
  position: number;
  label: string;
}

/** Y-axis ticks for a plot of `rows` rows.
 *
 * Without `domain` the axis is a row index — subcarrier number, or a delay tap
 * for the CIR panel. With one, `domain[0]` is the value at row 0 and
 * `domain[1]` the value at the last row, which is how the Doppler panel gets a
 * Hz axis without the renderer knowing what a hertz is.
 *
 * Doppler's interesting band is the bottom few percent of a ±9 Hz axis, so
 * labels carry enough decimals to stay distinct at high tick counts. */
export function yAxisTicks(
  rows: number,
  domain: [number, number] | undefined,
  count: number,
): AxisTick[] {
  const ticks: AxisTick[] = [];
  for (let i = 0; i < count; i++) {
    const position = count === 1 ? 0 : i / (count - 1);
    if (domain) {
      const value = domain[0] + (domain[1] - domain[0]) * position;
      const span = Math.abs(domain[1] - domain[0]);
      const decimals = span >= 100 ? 0 : span >= 10 ? 1 : span >= 1 ? 2 : 3;
      ticks.push({ position, label: value.toFixed(decimals) });
    } else {
      const index = Math.round((rows - 1) * (1 - position));
      ticks.push({ position, label: String(index) });
    }
  }
  return ticks;
}
```

Replace the component's existing y-axis label loop with a call to `yAxisTicks(props.numSubcarriers, props.yDomain, tickCount)`, drawing each tick's `label` at its `position`. Keep the existing pixel maths and fonts; only the label source changes.

- [ ] **Step 4: Add the two props and the fetcher seam**

Add to the `HeatmapProps` interface:

```ts
  /** Fetches the grid for a view. Defaults to /api/tile via fetchTile.
   *  The Doppler panels pass a fetchDoppler-backed function instead, which is
   *  why this is a function rather than a metric name: the two endpoints take
   *  different parameters but return the same binary contract. */
  source?: (t0: number, t1: number, width: number, signal: AbortSignal) => Promise<Tile>;
  /** [valueAtRow0, valueAtLastRow] for the y-axis. Omit for a subcarrier
   *  index axis. The Doppler panels pass [fMax, 0]. */
  yDomain?: [number, number];
```

At the fetch site (currently `Heatmap.tsx:753`), replace the direct call:

```ts
        const tile = props.source
          ? await props.source(t0, t1, width, controller.signal)
          : await fetchTile(
              props.path,
              t0,
              t1,
              width,
              props.metric,
              controller.signal,
              props.mimo,
              props.sourceMac,
              props.interpolate,
            );
```

Leave every other line of the effect — the stale-response guard, the abort handling, the scale locking — untouched.

- [ ] **Step 5: Run the tests**

Run: `npm --prefix frontend run test && npm --prefix frontend run build`
Expected: tests pass, build succeeds with no TypeScript errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/Heatmap.tsx frontend/src/heatmap.test.ts
git commit -m "feat: Heatmap fetcher seam and y-axis domain for non-subcarrier rows"
```

---

### Task 8: Tabs, and the Doppler panels in their own tab

No Tabs primitive exists — `frontend/src/components/ui/` has alert, badge, button, card, collapsible, input, label, select, separator only.

**Files:**
- Create: `frontend/src/components/ui/tabs.tsx`
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `fetchDoppler` (Task 6); `source` and `yDomain` props (Task 7).
- Produces: no exports other tasks depend on.

- [ ] **Step 1: Add the Tabs primitive**

**No install needed.** This project does not use Radix — `collapsible.tsx` imports from `@base-ui/react`, which is already a dependency at `^1.7.0` and ships a `tabs` entry point. Adding `@radix-ui/react-tabs` would pull a second, conflicting primitive library into a codebase that has none.

Base UI's parts are `Root`, `List`, `Tab`, `Panel`, `Indicator` — note that the trigger is `Tab` (not `Trigger`) and the content is `Panel` (not `Content`), which is why the wrappers below rename them. Open `frontend/src/components/ui/collapsible.tsx` first and mirror its structure exactly; it maps `Content` to `Panel` the same way.

Create `frontend/src/components/ui/tabs.tsx`:

```tsx
import { Tabs as TabsPrimitive } from "@base-ui/react/tabs"

import { cn } from "@/lib/utils"

function Tabs({ className, ...props }: TabsPrimitive.Root.Props) {
  return (
    <TabsPrimitive.Root
      data-slot="tabs"
      className={cn("flex flex-col gap-2", className)}
      {...props}
    />
  )
}

function TabsList({ className, ...props }: TabsPrimitive.List.Props) {
  return (
    <TabsPrimitive.List
      data-slot="tabs-list"
      className={cn(
        "bg-muted text-muted-foreground inline-flex h-9 w-fit items-center justify-center rounded-lg p-[3px]",
        className,
      )}
      {...props}
    />
  )
}

function TabsTrigger({ className, ...props }: TabsPrimitive.Tab.Props) {
  return (
    <TabsPrimitive.Tab
      data-slot="tabs-trigger"
      className={cn(
        "data-[selected]:bg-background data-[selected]:text-foreground inline-flex h-[calc(100%-1px)] flex-1 items-center justify-center gap-1.5 rounded-md border border-transparent px-2 py-1 text-sm font-medium whitespace-nowrap transition-colors disabled:pointer-events-none disabled:opacity-50",
        className,
      )}
      {...props}
    />
  )
}

function TabsContent({ className, ...props }: TabsPrimitive.Panel.Props) {
  return (
    <TabsPrimitive.Panel
      data-slot="tabs-content"
      className={cn("flex-1 outline-none", className)}
      {...props}
    />
  )
}

export { Tabs, TabsList, TabsTrigger, TabsContent }
```

Base UI marks the active tab with a `data-selected` attribute rather than Radix's `data-state="active"` — the class strings above already reflect that. Verify against the rendered DOM in Step 5 rather than assuming.

- [ ] **Step 2: Split the existing panels into a tab**

In `frontend/src/App.tsx`, import the primitive and wrap the current panel stack. The six existing `<Heatmap>` panels (at roughly lines 384, 404, 425, 441, 476, 509) move inside the first `TabsContent` unchanged — do not alter their props:

```tsx
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// ...inside the render, wrapping the existing panel stack:
<Tabs defaultValue="channel">
  <TabsList>
    <TabsTrigger value="channel">Channel</TabsTrigger>
    <TabsTrigger value="doppler">Doppler</TabsTrigger>
  </TabsList>

  <TabsContent value="channel">
    {/* the six existing Heatmap panels, moved verbatim */}
  </TabsContent>

  <TabsContent value="doppler">
    {/* Task 8 Step 3 */}
  </TabsContent>
</Tabs>
```

- [ ] **Step 3: Add the two Doppler panels**

Inside the `doppler` tab. Add a `winSeconds` control alongside the existing toolbar state (`const [winSeconds, setWinSeconds] = useState(30);`):

```tsx
<Heatmap
  path={path}
  metric="amplitude"
  filename={meta.filename}
  numSubcarriers={dopplerRows}
  captureTMin={meta.t_min}
  captureTMax={meta.t_max}
  title="Doppler — amplitude"
  colorLabel="magnitude"
  axisLabel="Doppler (Hz)"
  yDomain={[dopplerFMax, 0]}
  timeLink={timeLink}
  dark={dark}
  source={useCallback(
    (t0, t1, _width, signal) =>
      fetchDoppler(path, t0, t1, "amplitude", winSeconds, signal, mimo, sourceMac),
    [path, winSeconds, mimo, sourceMac],
  )}
/>
```

and the same again with `metric="csi_ratio_phase_time_unwrapped"`, `title="Doppler — unwrapped phase"`.

`dopplerRows` and `dopplerFMax` come from the first response; hold them in state, seeded from a `fetchDoppler` call in an effect, and render a short "loading" placeholder until they arrive. The `_width` parameter is ignored on purpose: the number of columns is set by the window and hop, not by the plot width, and pretending otherwise would reintroduce the width-drives-sample-rate bug.

- [ ] **Step 4: Add the caveat line under the panels**

Users will read a flat panel as a broken panel. Render this beneath the two plots:

```tsx
<p className="text-[11px] text-muted-foreground">
  Doppler is unsigned — amplitude and unwrapped phase are real signals, so
  approaching and receding motion are indistinguishable. The axis tops out at
  this capture's own Nyquist ({dopplerFMax.toFixed(2)} Hz); motion above that
  aliases. Shift + wheel zooms the frequency axis — the respiration band sits
  in the bottom few percent of it.
</p>
```

- [ ] **Step 5: Verify in a real browser**

Run `npm run dev:all`, open http://localhost:5173, select `20260821_170002.bin`, and switch to the Doppler tab.

Expected: a horizontal band near the bottom of the frequency axis. Shift+wheel to zoom into 0–1 Hz and confirm a line around 0.08–0.28 Hz. Then select `20260822_070002.bin` and confirm the panel is comparatively flat. A canvas defect will not show up in any unit test — check the pixels.

- [ ] **Step 6: Run the tests and build**

Run: `npm --prefix frontend run test && npm --prefix frontend run build && uv run pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/ui/tabs.tsx frontend/src/App.tsx
git commit -m "feat: Doppler tab with amplitude and unwrapped-phase spectrograms"
```

---

### Task 9: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:** none.

- [ ] **Step 1: Document the endpoint**

Add a `### \`GET /api/doppler\`` section to the API list in `README.md`, after `/api/tile`. Cover: the query parameters from Task 5, the one-sided frequency axis and why Doppler is unsigned here, the `X-Doppler-*` header table, and that `X-Doppler-Fs` is the capture's median frame rate rather than a function of requested width.

- [ ] **Step 2: Add a Doppler section**

Add a `## Doppler` section near `## Channel impulse response`, covering:
- what the panel shows and that the two inputs are amplitude and time-unwrapped phase, with raw wrapped phase excluded and why
- the measured PRF/Nyquist table from "Design Decisions" point 4
- that this is a respiration-band instrument at these PRFs, and that 1 m/s motion at 5 GHz sits near 33 Hz and aliases
- the measured contrast figures from point 6, so a reader can tell a working panel from a broken one
- that windows spanning a gap are NaN by design

- [ ] **Step 3: Update the Usage panel count**

`README.md` currently says the frontend renders eight heatmaps and lists them. Update that sentence for the tab split and the two new panels.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: /api/doppler and the Doppler panel's limits"
```

---

## Self-Review

**Spec coverage.** Approach A (own endpoint, tile pipeline reused for decode) — Tasks 4–5. Subcarrier-averaged spectrograms — Task 3. Amplitude + time-unwrapped phase — `DOPPLER_METRICS`, Task 4. Full Nyquist per capture — `fs` from median frame interval, Tasks 2 and 4. Separate tab — Task 8. Live support — Task 1 (the measured bottleneck) plus the unchanged existing poll loop; no incremental spectrogram, because the FFT measured 4.7 ms per new column against a 967 ms cold decode.

**Known gap, deliberately left open.** Task 1 makes live polling affordable but does not make the *first* view of a large window instant — a cold decode is still around 1.2 s on a 3300 s capture. Bounding that would mean a decode-ahead or a coarser first pass, which is a separate change to shared machinery and is not in this plan.

**Type consistency.** `_decode_block_cached(..., block_frames, ...)` is introduced in Task 1 and consumed with the same name in Task 4. `compute_doppler`'s metadata keys are produced in Task 4 and read in Task 5 headers, which are parsed in Task 6 into `DopplerTile`, whose `fMax` feeds `yDomain` in Task 8. `yAxisTicks(rows, domain, count)` is defined and tested in Task 7 and used only there. `DopplerMetric` in Task 6 matches `DOPPLER_METRICS` in Task 4.

**Ordering.** Tasks 1–5 are backend and independently testable. Tasks 6–8 are frontend and depend on 5. Task 9 depends on everything. Task 1 stands alone and is worth landing first regardless of the rest of the feature.
