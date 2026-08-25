"""Block-cache behaviour when a live capture changes size mid-request.

These run without a capture file on disk: the index surface the block layer
touches is small enough to stand in for, and the decode is stubbed, so the
tests exercise the bookkeeping rather than the reader.

The bug they pin: ``compute_doppler`` chooses ``frame_ids`` from one reading of
``index.count``, but the block that serves them was sized by a later reading.
A request extending the shared index in between -- or a replaced live file,
whose ``extend()`` rebuilds from scratch and can come back *shorter* -- leaves
ids pointing past the end of the block.

In the Doppler path that raised ``IndexError``. In the tile path it was worse:
``take`` fell to zero while ``pos`` still trailed ``n``, so the loop stopped
advancing and spun forever, hanging the worker thread rather than failing.

A live view polls continuously, so both re-triggered on every poll for as long
as the capture kept changing under it.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.tiles import (
    BLOCK_SIZE,
    _block_cache,
    _decode_block_cached,
    _decode_for_doppler,
    _decode_via_blocks,
    reset_tile_caches,
)


class _FakeIndex:
    """The slice of the index surface the block layer actually reads."""

    def __init__(self, count: int, num_sc: int = 8) -> None:
        self.count = count
        self.num_subcarriers = num_sc
        self.times = np.arange(count, dtype=float)


def _stub_decode(path, index, frame_ids, **kwargs):
    """Four metrics whose value is the frame id, so rows are identifiable."""
    frame_ids = np.asarray(frame_ids)
    rows = np.tile(
        frame_ids.astype(np.float32)[:, None], (1, index.num_subcarriers)
    )
    return rows, rows.copy(), rows.copy(), rows.copy()


@pytest.fixture
def stub(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.tiles.decode_frames", _stub_decode)
    reset_tile_caches()
    path = tmp_path / "live.bin"
    path.write_bytes(b"\xac")  # only needs to exist
    yield path
    reset_tile_caches()


class _ShiftingIndex(_FakeIndex):
    """``count`` changes between reads, as a concurrently-extended one does.

    The first read is the one that built the cache key; a later read inside the
    same call is what used to size the block. Anything reading ``count`` once
    is unaffected by the shift.
    """

    def __init__(self, counts: list[int], num_sc: int = 8) -> None:
        super().__init__(counts[0], num_sc)
        self._counts = list(counts)

    @property                                   # type: ignore[override]
    def count(self) -> int:
        value = self._counts[0]
        if len(self._counts) > 1:
            self._counts.pop(0)
        return value

    @count.setter
    def count(self, value: int) -> None:
        self._counts = [value]


def test_cached_block_length_matches_the_count_in_its_key(stub) -> None:
    """A cached block holds exactly the frames its key names.

    A guard rather than a reproducer -- the old code satisfied it too, because
    the ``put`` re-read the count and so stored a self-consistent entry. What it
    pins is that one reading of ``count`` now drives the key, the block, and the
    put, so a shifting count cannot make the lookup key and the stored key
    disagree and turn every poll on a growing capture into a re-decode.
    """
    index = _ShiftingIndex([BLOCK_SIZE + 50, BLOCK_SIZE + 47])

    _decode_block_cached(stub, index, 1, "amplitude")

    assert _block_cache._entries, "the decode must have cached something"
    for key, arr in _block_cache._entries.items():
        assert key[3] == len(arr), (
            f"key {key} promises {key[3]} frames but the block holds {len(arr)}"
        )


def test_doppler_drops_frames_that_vanished_instead_of_raising(stub) -> None:
    """A capture replaced mid-request rebuilds shorter; ids past the end go."""
    index = _FakeIndex(BLOCK_SIZE + 50)
    frame_ids = np.arange(index.count)

    # Another request rebuilt the index from a now-shorter file.
    index.count -= 3

    out = _decode_for_doppler(stub, index, frame_ids, "amplitude", None, True)

    assert out.shape[0] == index.count, "keeps every frame that still exists"
    assert np.array_equal(out[:, 0], np.arange(index.count, dtype=np.float32))


def test_tile_slice_returns_the_rows_that_are_still_real(stub) -> None:
    """The tile path takes the same shrink without a shape mismatch."""
    index = _FakeIndex(BLOCK_SIZE + 50)
    frame_ids = np.arange(index.count)

    index.count -= 3

    out = _decode_via_blocks(stub, index, frame_ids, "amplitude", None, interpolate=True)

    assert out.shape[0] == index.count
    assert np.array_equal(out[:, 0], np.arange(index.count, dtype=np.float32))


def test_growth_does_not_serve_the_stale_short_tail_block(stub) -> None:
    """A tail block cached while short must not answer for the longer one."""
    index = _FakeIndex(BLOCK_SIZE + 10)
    first = _decode_block_cached(stub, index, 1, "amplitude")
    assert len(first) == 10

    index.count = BLOCK_SIZE + 40  # capture grew
    second = _decode_block_cached(stub, index, 1, "amplitude")

    assert len(second) == 40, "grown tail re-decodes rather than serving 10 rows"
    assert np.array_equal(
        second[:, 0],
        np.arange(BLOCK_SIZE, BLOCK_SIZE + 40, dtype=np.float32),
    )
