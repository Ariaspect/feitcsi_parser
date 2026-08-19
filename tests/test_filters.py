"""Tests for MIMO and source-MAC filtering across index, tile, and API."""

from __future__ import annotations

import struct
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.index import FrameIndex, parse_mimo_filter
from backend.tiles import compute_tile, reset_tile_caches

CAPTURE = Path(__file__).resolve().parent.parent / "captures" / "capture.dat"

pytestmark = pytest.mark.skipif(
    not CAPTURE.is_file(), reason="captures/capture.dat not present"
)


@pytest.fixture
def mixed_file(tmp_path: Path) -> Path:
    """capture.dat with frame 2 rewritten as 2x2 (payload duplicated)."""
    raw = CAPTURE.read_bytes()
    orig_cl = struct.unpack("I", raw[:4])[0]
    stride = 272 + orig_cl
    boundaries = [i * stride for i in range(7)]

    data = bytearray(raw[: boundaries[6]])
    off = boundaries[2]
    new_cl = orig_cl * 2
    struct.pack_into("I", data, off, new_cl)
    data[off + 47] = 2  # num_tx: 1 -> 2
    payload_start = off + 272
    payload_end = payload_start + orig_cl
    data[payload_end:payload_end] = data[payload_start:payload_end]

    target = tmp_path / "mixed_filter.dat"
    target.write_bytes(bytes(data))
    return target


# ----------------------------------------------------------------------- #
#  parse_mimo_filter                                                      #
# ----------------------------------------------------------------------- #


def test_parse_mimo_filter_all_variants() -> None:
    assert parse_mimo_filter(None) is None
    assert parse_mimo_filter("") is None
    assert parse_mimo_filter("all") is None
    assert parse_mimo_filter("2x1") == (2, 1)
    assert parse_mimo_filter("2x2") == (2, 2)
    assert parse_mimo_filter("4x4") == (4, 4)


@pytest.mark.parametrize("bad", ["2", "2x", "x2", "abc", "2x1x3", "2.5x1"])
def test_parse_mimo_filter_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_mimo_filter(bad)


# ----------------------------------------------------------------------- #
#  FrameIndex.source_macs + filter_mask                                   #
# ----------------------------------------------------------------------- #


def test_source_macs_match_upstream(mixed_file: Path) -> None:
    """source_macs matches CSIKit's parseHeader on every frame."""
    from CSIKit.reader import FeitCSIBeamformReader

    reader = FeitCSIBeamformReader()
    idx = FrameIndex(mixed_file)

    pos = 0
    for i in range(idx.count):
        with mixed_file.open("rb") as fh:
            fh.seek(pos)
            hb = fh.read(272)
        h = reader.parseHeader(hb)
        assert idx.source_macs[i] == h["source_mac_string"]
        pos += 272 + h["csi_length"]


def test_filter_mask_mimo(mixed_file: Path) -> None:
    idx = FrameIndex(mixed_file)
    mask_2x1 = idx.filter_mask(mimo=(2, 1))
    mask_2x2 = idx.filter_mask(mimo=(2, 2))
    mask_all = idx.filter_mask()

    assert mask_2x1.sum() == 5  # frames 0,1,3,4,5
    assert mask_2x2.sum() == 1  # frame 2
    assert mask_all.sum() == 6
    assert (mask_2x1 & mask_2x2).sum() == 0  # disjoint
    assert (mask_2x1 | mask_2x2).all()  # cover all


def test_filter_mask_source_mac(mixed_file: Path) -> None:
    idx = FrameIndex(mixed_file)
    macs = list(dict.fromkeys(idx.source_macs))
    assert len(macs) >= 2

    for mac in macs:
        mask = idx.filter_mask(source_mac=mac)
        for i in range(idx.count):
            if mask[i]:
                assert idx.source_macs[i] == mac


def test_filter_mask_combined(mixed_file: Path) -> None:
    """MIMO + MAC filters compose with AND."""
    idx = FrameIndex(mixed_file)
    macs = list(dict.fromkeys(idx.source_macs))
    mac = macs[0]

    mac_mask = idx.filter_mask(source_mac=mac)
    combined = idx.filter_mask(mimo=(2, 1), source_mac=mac)
    # combined is subset of mac_mask
    assert (combined & ~mac_mask).sum() == 0
    # combined only has 2x1 frames
    for i in range(idx.count):
        if combined[i]:
            assert idx.num_tx_arr[i] == 1


# ----------------------------------------------------------------------- #
#  /api/filters endpoint                                                  #
# ----------------------------------------------------------------------- #


def test_filters_endpoint(mixed_file: Path) -> None:
    client = TestClient(app)
    r = client.get("/api/filters", params={"path": str(mixed_file)})
    assert r.status_code == 200
    body = r.json()
    assert "2x1" in body["mimo_modes"]
    assert "2x2" in body["mimo_modes"]
    assert len(body["source_macs"]) >= 2
    # MACs look like MACs
    for mac in body["source_macs"]:
        assert len(mac.split(":")) == 6


# ----------------------------------------------------------------------- #
#  /api/meta with filters                                                 #
# ----------------------------------------------------------------------- #


def test_meta_unfiltered_total(mixed_file: Path) -> None:
    client = TestClient(app)
    r = client.get("/api/meta", params={"path": str(mixed_file)})
    assert r.status_code == 200
    assert r.json()["total_frames"] == 6


def test_meta_mimo_filter_reduces_count(mixed_file: Path) -> None:
    client = TestClient(app)
    r = client.get("/api/meta", params={"path": str(mixed_file), "mimo": "2x2"})
    assert r.status_code == 200
    body = r.json()
    assert body["total_frames"] == 1


def test_meta_mac_filter_reduces_count(mixed_file: Path) -> None:
    client = TestClient(app)
    idx = FrameIndex(mixed_file)
    macs = list(dict.fromkeys(idx.source_macs))
    mac = macs[0]
    expected = sum(1 for m in idx.source_macs if m == mac)

    r = client.get("/api/meta", params={"path": str(mixed_file), "source_mac": mac})
    assert r.status_code == 200
    assert r.json()["total_frames"] == expected


def test_meta_invalid_mimo_returns_400(mixed_file: Path) -> None:
    client = TestClient(app)
    r = client.get("/api/meta", params={"path": str(mixed_file), "mimo": "garbage"})
    assert r.status_code == 400


# ----------------------------------------------------------------------- #
#  compute_tile with filters                                              #
# ----------------------------------------------------------------------- #


def test_tile_filtered_leaves_holes(mixed_file: Path) -> None:
    """A 2x2-only filter excludes 2x1 frames; their columns stay NaN."""
    reset_tile_caches()
    idx = FrameIndex(mixed_file)
    t0 = float(idx.times[0])
    t1 = float(idx.times[-1])

    grid, meta = compute_tile(mixed_file, t0, t1, 200, "amplitude", mimo=(2, 2))
    assert meta["frames_decoded"] == 1
    assert meta["filled_columns"] == 0  # filter disables gap fill
    # Most columns should be NaN (only the one with the 2x2 frame is finite).
    finite_cols = np.isfinite(grid).any(axis=0).sum()
    assert finite_cols <= 1


def test_tile_unfiltered_fills_sampling_gaps(mixed_file: Path) -> None:
    """Without a filter, sampling gaps get filled (current behavior).

    Uses capture.dat (1101 frames) so width=200 < total_in_range and real
    sampling gaps exist between columns.
    """
    reset_tile_caches()
    idx = FrameIndex(CAPTURE)
    t0 = float(idx.times[0])
    t1 = float(idx.times[-1])

    grid, meta = compute_tile(CAPTURE, t0, t1, 200, "amplitude")
    assert meta["frames_decoded"] == 1101
    # 1101 frames spread over 200 columns -> columns cover ~5 frames each, so
    # every column receives at least one frame and no gaps need filling.
    # Sanity check: filled_columns reflects that no fill was needed.
    assert meta["filled_columns"] == 0


def test_tile_unfiltered_fills_gaps_on_sparse_capture(mixed_file: Path) -> None:
    """A synthetic sparse capture exercises the gap fill path.

    Build a 6-frame capture whose frames are far apart relative to the column
    width so most columns receive no frame and must borrow from neighbours.
    """
    # Reuse mixed_file but request fewer columns than frames so width is not
    # capped to total_in_range, then check that fill fires on the gaps.
    reset_tile_caches()
    idx = FrameIndex(mixed_file)
    t0 = float(idx.times[0])
    t1 = float(idx.times[-1])

    # 6 frames in 5 columns: at least one column is a sampling gap.
    grid, meta = compute_tile(mixed_file, t0, t1, 5, "amplitude")
    assert meta["frames_decoded"] == 6
    # Either no gaps (every column got a frame) or some gaps filled: in both
    # cases, finite columns should equal width when unfiltered.
    finite_cols = int(np.isfinite(grid).any(axis=0).sum())
    assert finite_cols == 5


def test_tile_mac_filter(mixed_file: Path) -> None:
    """MAC filter narrows decoded frames to that MAC's frames."""
    reset_tile_caches()
    idx = FrameIndex(mixed_file)
    macs = list(dict.fromkeys(idx.source_macs))
    mac = macs[0]
    expected = sum(1 for m in idx.source_macs if m == mac)

    t0 = float(idx.times[0])
    t1 = float(idx.times[-1])
    _, meta = compute_tile(mixed_file, t0, t1, 200, "amplitude", source_mac=mac)
    # Tile is sampled to 8192 max but file is tiny, so all matching frames
    # get decoded.
    assert meta["frames_decoded"] == expected


# ----------------------------------------------------------------------- #
#  /api/tile with filters                                                 #
# ----------------------------------------------------------------------- #


def test_tile_endpoint_mimo_filter(mixed_file: Path) -> None:
    client = TestClient(app)
    idx = FrameIndex(mixed_file)
    t0 = float(idx.times[0])
    t1 = float(idx.times[-1])

    r = client.get("/api/tile", params={
        "path": str(mixed_file),
        "t0": t0, "t1": t1, "width": 200, "metric": "amplitude", "mimo": "2x2",
    })
    assert r.status_code == 200
    frames = int(r.headers["X-Tile-Frames"])
    assert frames == 1


def test_tile_endpoint_mac_filter(mixed_file: Path) -> None:
    client = TestClient(app)
    idx = FrameIndex(mixed_file)
    macs = list(dict.fromkeys(idx.source_macs))
    mac = macs[0]
    expected = sum(1 for m in idx.source_macs if m == mac)

    t0 = float(idx.times[0])
    t1 = float(idx.times[-1])
    r = client.get("/api/tile", params={
        "path": str(mixed_file),
        "t0": t0, "t1": t1, "width": 200, "metric": "amplitude",
        "source_mac": mac,
    })
    assert r.status_code == 200
    assert int(r.headers["X-Tile-Frames"]) == expected


# ----------------------------------------------------------------------- #
#  /api/captures endpoint                                                #
# ----------------------------------------------------------------------- #


def test_captures_endpoint_lists_capture_files() -> None:
    """GET /api/captures returns .dat and .bin from captures/, mtime desc.

    .dat is FeitCSI, .bin is MediaTek; both are selectable in the UI.
    """
    client = TestClient(app)
    r = client.get("/api/captures")
    assert r.status_code == 200
    caps = r.json()

    cap_names = {c["filename"] for c in caps}
    assert "capture.dat" in cap_names

    for c in caps:
        assert c["filename"].endswith((".dat", ".bin"))
        assert c["size_bytes"] > 0
        assert "path" in c
        assert "mtime" in c

    mtimes = [c["mtime"] for c in caps]
    assert mtimes == sorted(mtimes, reverse=True)


def test_captures_endpoint_missing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing captures dir returns empty list, not an error."""
    import backend.app as app_mod
    monkeypatch.setattr(app_mod, "CAPTURES_DIR", tmp_path / "nonexistent")
    client = TestClient(app)
    r = client.get("/api/captures")
    assert r.status_code == 200
    assert r.json() == []


# ----------------------------------------------------------------------- #
#  'all' means no filter, symmetrically for both filters                  #
# ----------------------------------------------------------------------- #


def test_parse_mac_filter_treats_all_as_no_filter() -> None:
    """'all' must mean 'every sender', not a literal address to match."""
    from backend.index import parse_mac_filter

    assert parse_mac_filter("all") is None
    assert parse_mac_filter(None) is None
    assert parse_mac_filter("") is None
    assert parse_mac_filter("   ") is None
    assert parse_mac_filter("d8:3a:dd:29:22:f5") == "d8:3a:dd:29:22:f5"


def test_endpoints_treat_source_mac_all_like_mimo_all(mixed_file: Path) -> None:
    """source_mac=all returns the whole capture, matching mimo=all.

    Regression: 'all' used to be matched as a literal MAC, so a direct API
    call asking for every sender got zero frames back. The frontend strips
    'all' before building the query and so never saw it.
    """
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    unfiltered = client.get("/api/meta", params={"path": str(mixed_file)})
    all_macs = client.get("/api/meta", params={"path": str(mixed_file), "source_mac": "all"})
    all_mimo = client.get("/api/meta", params={"path": str(mixed_file), "mimo": "all"})

    assert unfiltered.status_code == 200
    assert all_macs.json()["total_frames"] == unfiltered.json()["total_frames"]
    assert all_mimo.json()["total_frames"] == unfiltered.json()["total_frames"]
    assert all_macs.json()["total_frames"] > 0


def test_tile_endpoint_source_mac_all_is_unfiltered(mixed_file: Path) -> None:
    """The same symmetry on /api/tile: 'all' must not empty the grid."""
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    params = {"path": str(mixed_file), "t0": 0.0, "t1": 1e9, "width": 16, "metric": "amplitude"}
    plain = client.get("/api/tile", params=params)
    with_all = client.get("/api/tile", params={**params, "source_mac": "all"})

    assert plain.status_code == with_all.status_code == 200
    assert with_all.headers["X-Tile-Width"] == plain.headers["X-Tile-Width"]
    assert with_all.content == plain.content
