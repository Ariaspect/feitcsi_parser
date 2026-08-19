"""MTK TLV format: record framing, group assembly, and decode."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from backend.mtk import (
    CSD_MIN_FRAMES,
    MAGIC,
    RPI_PLANE,
    TAG_BANDWIDTH,
    TAG_CSI_IMAG,
    TAG_CSI_REAL,
    TAG_RPI,
    TAG_RSSI,
    TAG_SEQUENCE,
    TAG_SOURCE_MAC,
    TAG_TIMESTAMP,
    TAG_TPI,
    TAG_VERSION,
    MTKIndex,
    can_read,
    decode_frames,
    estimate_csd_slope,
)

CAPTURES = Path(__file__).resolve().parent.parent / "captures"
PING = CAPTURES / "csi_ping34_30s.bin"
BEACON = CAPTURES / "csi_beacon_30s.bin"
FEITCSI = CAPTURES / "capture.dat"

MAC = "08:bf:b8:95:80:04"


# ---------------------------------------------------------------------- #
#  Synthetic record builder                                               #
# ---------------------------------------------------------------------- #


def _tlv(tag: int, value: bytes) -> bytes:
    return bytes([tag]) + struct.pack("<H", len(value)) + value


def _csi_bytes(values: np.ndarray) -> bytes:
    """Encode as 14-bit two's complement in u16 LE, as the hardware does."""
    return (np.asarray(values).astype(np.int32) & 0x3FFF).astype("<u2").tobytes()


def record(
    *,
    group: int,
    idx: int,
    last: bool,
    tpi: int,
    rpi: int,
    csi: np.ndarray,
    bw_code: int = 2,
    ts: int = 1000,
    rssi: int = 200,
    mac: str = MAC,
) -> bytes:
    seq = (group << 16) | ((0x8000 if last else 0) | idx)
    body = b"".join([
        _tlv(TAG_VERSION, bytes([13])),
        _tlv(TAG_TIMESTAMP, struct.pack("<Q", ts)),
        _tlv(TAG_RSSI, bytes([rssi])),
        _tlv(TAG_BANDWIDTH, bytes([bw_code])),
        _tlv(TAG_SOURCE_MAC, bytes(int(x, 16) for x in mac.split(":"))),
        _tlv(TAG_CSI_REAL, _csi_bytes(np.real(csi))),
        _tlv(TAG_CSI_IMAG, _csi_bytes(np.imag(csi))),
        _tlv(TAG_TPI, bytes([tpi])),
        _tlv(TAG_RPI, bytes([rpi])),
        _tlv(TAG_SEQUENCE, struct.pack("<I", seq)),
    ])
    return bytes([MAGIC]) + struct.pack("<H", len(body)) + body


def group_records(
    group: int, cells: dict[tuple[int, int], np.ndarray], **kw
) -> bytes:
    """One group; ``cells`` maps (tpi, rpi) -> csi. Last record gets bit 15."""
    items = sorted(cells.items(), key=lambda kv: 2 * kv[0][1] + kv[0][0])
    return b"".join(
        record(group=group, idx=i, last=(i == len(items) - 1),
               tpi=tpi, rpi=rpi, csi=csi, **kw)
        for i, ((tpi, rpi), csi) in enumerate(items)
    )


def band(n: int = 256, seed: int = 0) -> np.ndarray:
    """A synthetic band with realistic structural nulls (guard, DC, pilots)."""
    rng = np.random.default_rng(seed)
    z = (rng.integers(-2000, 2000, n) + 1j * rng.integers(-2000, 2000, n)).astype(complex)
    k = np.fft.fftfreq(n, 1 / n).astype(int)  # raw FFT bin -> signed subcarrier
    z[np.abs(k) >= 123] = 0                   # guard
    z[np.abs(k) <= 1] = 0                     # DC null
    z[np.isin(np.abs(k), [11, 39, 75, 103])] = 0  # VHT80 pilots
    return z


def write(tmp_path: Path, blob: bytes, name: str = "cap.bin") -> Path:
    target = tmp_path / name
    target.write_bytes(blob)
    return target


# ---------------------------------------------------------------------- #
#  1. Detection                                                           #
# ---------------------------------------------------------------------- #


def test_detects_an_mtk_capture(tmp_path: Path) -> None:
    assert can_read(write(tmp_path, group_records(1, {(0, 0): band()})))


def test_rejects_a_feitcsi_capture() -> None:
    if not FEITCSI.is_file():
        pytest.skip("captures/capture.dat not present")
    assert not can_read(FEITCSI)


def test_rejects_a_file_without_the_magic(tmp_path: Path) -> None:
    assert not can_read(write(tmp_path, b"\x00" * 64))
    assert not can_read(write(tmp_path, b""))


# ---------------------------------------------------------------------- #
#  2. Grouping is driven by tag 18, not the clock                         #
# ---------------------------------------------------------------------- #


def test_a_group_spanning_a_millisecond_boundary_stays_one_frame(tmp_path: Path) -> None:
    """The real captures tick mid-group; timestamps must not split a frame."""
    blob = b"".join([
        record(group=1, idx=0, last=False, tpi=0, rpi=0, csi=band(), ts=6746499),
        record(group=1, idx=1, last=False, tpi=1, rpi=0, csi=band(seed=1), ts=6746499),
        record(group=1, idx=2, last=False, tpi=0, rpi=1, csi=band(seed=2), ts=6746499),
        record(group=1, idx=3, last=True, tpi=1, rpi=1, csi=band(seed=3), ts=6746500),
    ])
    idx = MTKIndex(write(tmp_path, blob))
    assert idx.count == 1
    assert idx.num_rx_arr.tolist() == [2]


def test_records_sharing_a_timestamp_across_groups_stay_separate(tmp_path: Path) -> None:
    """Two groups at the same millisecond are two frames, not one."""
    blob = b"".join([
        group_records(1, {(0, 0): band(), (1, 0): band(seed=1)}, ts=500),
        group_records(2, {(0, 0): band(seed=2), (1, 0): band(seed=3)}, ts=500),
    ])
    idx = MTKIndex(write(tmp_path, blob))
    assert idx.count == 2
    assert idx.times.tolist() == [0.0, 0.0]


def test_an_unclosed_group_is_withheld(tmp_path: Path) -> None:
    """No bit-15 record yet: the group is still arriving."""
    blob = group_records(1, {(0, 0): band(), (1, 0): band(seed=1)})
    blob += record(group=2, idx=0, last=False, tpi=0, rpi=0, csi=band(seed=2))
    idx = MTKIndex(write(tmp_path, blob))
    assert idx.count == 1


def test_extend_picks_up_a_group_once_it_closes(tmp_path: Path) -> None:
    path = write(tmp_path, group_records(1, {(0, 0): band(), (1, 0): band(seed=1)}))
    idx = MTKIndex(path)
    assert idx.count == 1

    with path.open("ab") as fh:  # a partial group adds nothing
        fh.write(record(group=2, idx=0, last=False, tpi=0, rpi=0, csi=band(seed=2)))
    assert idx.extend() == 0 and idx.count == 1

    with path.open("ab") as fh:  # closing it yields the frame
        fh.write(record(group=2, idx=1, last=True, tpi=1, rpi=0, csi=band(seed=3)))
    assert idx.extend() == 1 and idx.count == 2


def test_a_truncated_record_is_not_indexed(tmp_path: Path) -> None:
    blob = group_records(1, {(0, 0): band(), (1, 0): band(seed=1)})
    idx = MTKIndex(write(tmp_path, blob + record(
        group=2, idx=0, last=True, tpi=0, rpi=0, csi=band(seed=2))[:40]))
    assert idx.count == 1


# ---------------------------------------------------------------------- #
#  3. Cell layout                                                         #
# ---------------------------------------------------------------------- #


def test_tpi_fills_the_rx_slots_and_rpi_selects_the_plane(tmp_path: Path) -> None:
    """Only plane RPI_PLANE is read; tpi ascending becomes rx0, rx1."""
    other = 1 - RPI_PLANE
    blob = group_records(1, {
        (0, RPI_PLANE): band(seed=1),
        (1, RPI_PLANE): band(seed=2),
        (0, other): band(seed=3),
        (1, other): band(seed=4),
    })
    path = write(tmp_path, blob)
    idx = MTKIndex(path)
    assert idx.num_rx_arr.tolist() == [2] and idx.num_tx_arr.tolist() == [1]
    assert idx.mimo_labels() == ["2x1"]

    amp, _, _, _ = decode_frames(path, idx, np.array([0]))
    solo = MTKIndex(write(tmp_path, group_records(
        2, {(0, RPI_PLANE): band(seed=1)}), "solo.bin"))
    solo_amp, _, _, _ = decode_frames(tmp_path / "solo.bin", solo, np.array([0]))
    np.testing.assert_allclose(amp, solo_amp, equal_nan=True)


def test_a_single_stream_group_has_no_ratio(tmp_path: Path) -> None:
    idx = MTKIndex(write(tmp_path, group_records(1, {(0, RPI_PLANE): band()})))
    assert idx.num_rx_arr.tolist() == [1]
    _, _, ratio_amp, ratio_phase = decode_frames(
        tmp_path / "cap.bin", idx, np.array([0]))
    assert np.isnan(ratio_amp).all() and np.isnan(ratio_phase).all()


# ---------------------------------------------------------------------- #
#  4. Sample decoding                                                     #
# ---------------------------------------------------------------------- #


def test_samples_are_sign_extended_from_14_bits(tmp_path: Path) -> None:
    """0x2000 is negative in 14-bit; read as int16 it would be +8192."""
    n = 256
    z = np.zeros(n, dtype=complex)
    k = np.fft.fftfreq(n, 1 / n).astype(int)
    live = np.flatnonzero(np.abs(k) == 5)[0]
    z[live] = -8192 + 0j  # 0x2000 once masked to 14 bits

    path = write(tmp_path, group_records(1, {(0, RPI_PLANE): z}))
    idx = MTKIndex(path)
    amp, phase, _, _ = decode_frames(path, idx, np.array([0]), interpolate=False)
    col = n // 2 + int(k[live])  # fftshift moves bin k to centre+k
    assert amp[0, col] == pytest.approx(20 * np.log10(8192), abs=1e-3)
    assert phase[0, col] == pytest.approx(np.pi, abs=1e-5)


def test_subcarriers_are_fftshifted_to_centre_dc(tmp_path: Path) -> None:
    """MTK arrives in raw FFT bin order — the opposite of FeitCSI."""
    n = 256
    z = np.ones(n, dtype=complex) * 100
    k = np.fft.fftfreq(n, 1 / n).astype(int)
    z[k == 40] = 3000  # a marker at a known signed subcarrier

    path = write(tmp_path, group_records(1, {(0, RPI_PLANE): z}))
    amp, _, _, _ = decode_frames(
        path, MTKIndex(path), np.array([0]), interpolate=False)
    assert int(np.nanargmax(amp[0])) == n // 2 + 40


def test_the_ratio_divides_along_the_tpi_axis(tmp_path: Path) -> None:
    """rx1/rx0 must be tpi1/tpi0 — the axis that cancels receiver offsets."""
    base = band(seed=7)
    scaled = base * 2  # exactly +6.02 dB, zero phase shift
    path = write(tmp_path, group_records(
        1, {(0, RPI_PLANE): base, (1, RPI_PLANE): scaled}))
    _, _, ratio_amp, ratio_phase = decode_frames(
        path, MTKIndex(path), np.array([0]), interpolate=False)
    live = np.isfinite(ratio_amp[0])
    np.testing.assert_allclose(ratio_amp[0][live], 20 * np.log10(2), atol=1e-4)
    np.testing.assert_allclose(ratio_phase[0][live], 0.0, atol=1e-5)


# ---------------------------------------------------------------------- #
#  5. Nulls                                                               #
# ---------------------------------------------------------------------- #


def test_pilots_are_interpolated_and_the_guard_band_is_not(tmp_path: Path) -> None:
    """Structural nulls are found in the data; no rate table exists."""
    path = write(tmp_path, b"".join(
        group_records(g, {(0, RPI_PLANE): band(seed=g)}) for g in (1, 2, 3)))
    idx = MTKIndex(path)
    amp, _, _, _ = decode_frames(path, idx, np.arange(3))

    n = 256
    k = np.fft.fftfreq(n, 1 / n).astype(int)
    col = lambda kk: n // 2 + kk  # noqa: E731
    for pilot in (11, 39, 75, 103):
        assert np.isfinite(amp[:, col(pilot)]).all(), f"pilot {pilot} left null"
    assert np.isfinite(amp[:, col(0)]).all(), "DC null not interpolated"
    for guard in (123, 127, -123, -128):
        assert np.isnan(amp[:, col(guard)]).all(), f"guard {guard} was filled"


def test_a_long_interior_null_is_not_spanned(tmp_path: Path) -> None:
    """A wide dead region is not guessed across; the guard runs only prove the
    edge condition, so this is what MAX_NULL_RUN actually guards."""
    n = 256
    k = np.fft.fftfreq(n, 1 / n).astype(int)
    wide, short = np.isin(k, np.arange(50, 59)), np.isin(k, np.arange(-62, -59))

    def notched(seed: int) -> np.ndarray:
        z = band(n, seed=seed)
        z[wide] = 0
        z[short] = 0  # exactly MAX_NULL_RUN long
        return z

    path = write(tmp_path, b"".join(
        group_records(g, {(0, RPI_PLANE): notched(g)}) for g in (1, 2, 3)))
    amp, _, _, _ = decode_frames(path, MTKIndex(path), np.arange(3))

    assert np.isnan(amp[:, n // 2 + 50: n // 2 + 59]).all(), "9-bin gap invented"
    assert np.isfinite(amp[:, n // 2 - 62: n // 2 - 59]).all(), "3-bin gap left null"


def test_interpolation_can_be_turned_off(tmp_path: Path) -> None:
    path = write(tmp_path, group_records(1, {(0, RPI_PLANE): band()}))
    idx = MTKIndex(path)
    raw, _, _, _ = decode_frames(path, idx, np.array([0]), interpolate=False)
    assert np.isneginf(raw[0, 256 // 2 + 11])  # db(0) at an untouched pilot


# ---------------------------------------------------------------------- #
#  6. Mixed bandwidth                                                     #
# ---------------------------------------------------------------------- #


def test_narrow_frames_are_centred_and_nan_padded(tmp_path: Path) -> None:
    """Both bandwidths centre on DC, so centring is the honest placement."""
    wide = band(256, seed=1)
    narrow = np.ones(64, dtype=complex) * 500
    path = write(tmp_path, b"".join([
        group_records(1, {(0, RPI_PLANE): wide}, bw_code=2),
        group_records(2, {(0, RPI_PLANE): narrow}, bw_code=0),
    ]))
    idx = MTKIndex(path)
    assert idx.num_subcarriers == 256
    assert idx.channel_widths == ["80", "20"]

    amp, _, _, _ = decode_frames(path, idx, np.arange(2), interpolate=False)
    assert amp.shape == (2, 256)
    lo = (256 - 64) // 2
    assert np.isnan(amp[1, :lo]).all() and np.isnan(amp[1, lo + 64:]).all()
    assert np.isfinite(amp[1, lo:lo + 64]).all()


# ---------------------------------------------------------------------- #
#  7. Filters                                                             #
# ---------------------------------------------------------------------- #


def test_filter_mask_selects_by_mimo_and_mac(tmp_path: Path) -> None:
    other = "60:38:e0:bb:ee:02"
    path = write(tmp_path, b"".join([
        group_records(1, {(0, RPI_PLANE): band(), (1, RPI_PLANE): band(seed=1)}),
        group_records(2, {(0, RPI_PLANE): band(seed=2)}),
        group_records(3, {(0, RPI_PLANE): band(seed=3)}, mac=other),
    ]))
    idx = MTKIndex(path)
    assert idx.filter_mask(mimo=(2, 1)).tolist() == [True, False, False]
    assert idx.filter_mask(source_mac=other).tolist() == [False, False, True]
    assert idx.filter_mask().all()


# ---------------------------------------------------------------------- #
#  8. Real captures                                                       #
# ---------------------------------------------------------------------- #


@pytest.mark.skipif(not PING.is_file(), reason="MTK ping capture not present")
def test_the_ping_capture_indexes_as_expected() -> None:
    idx = MTKIndex(PING)
    assert idx.count == 38
    assert idx._scan_end == PING.stat().st_size  # every byte accounted for
    assert idx.num_subcarriers == 256 and idx.bandwidth == "80"
    assert idx.stride is None  # record length varies with bandwidth
    labels = dict(zip(*np.unique(idx.mimo_labels(), return_counts=True)))
    assert labels == {"2x1": 36, "1x1": 2}
    assert set(idx.source_macs) == {MAC}
    assert idx.times[0] == 0.0 and idx.times[-1] == pytest.approx(28.039)


@pytest.mark.skipif(not PING.is_file(), reason="MTK ping capture not present")
def test_the_tpi_ratio_is_temporally_coherent() -> None:
    """The property the tpi mapping exists for: rpi scores ~0.68 here."""
    idx = MTKIndex(PING)
    ids = np.flatnonzero(idx.filter_mask(mimo=(2, 1)))
    _, _, _, ratio_phase = decode_frames(PING, idx, ids)
    live = np.isfinite(ratio_phase).all(axis=0)
    coherence = np.abs(np.exp(1j * ratio_phase[:, live]).mean(axis=0)).mean()
    step = np.angle(np.exp(1j * np.diff(ratio_phase[:, live], axis=0)))
    assert coherence > 0.9
    assert np.median(np.abs(step)) < 0.2
    assert np.mean(np.abs(step) > np.pi / 2) < 0.01


@pytest.mark.skipif(not BEACON.is_file(), reason="MTK beacon capture not present")
def test_the_beacon_capture_is_a_single_frame() -> None:
    idx = MTKIndex(BEACON)
    assert idx.count == 1
    assert idx.num_subcarriers == 128 and idx.bandwidth == "40"
    assert idx.num_rx_arr.tolist() == [1]


@pytest.mark.skipif(not PING.is_file(), reason="MTK ping capture not present")
def test_incremental_extension_matches_a_full_scan(tmp_path: Path) -> None:
    blob = PING.read_bytes()
    path = write(tmp_path, blob[:100_000], "grow.bin")
    incremental = MTKIndex(path)
    partial = incremental.count
    with path.open("ab") as fh:
        fh.write(blob[100_000:])
    added = incremental.extend()
    full = MTKIndex(path)

    assert partial + added == full.count == incremental.count
    for field in ("offsets", "times", "csi_lengths", "num_rx_arr", "rssi_1"):
        np.testing.assert_array_equal(
            getattr(incremental, field), getattr(full, field))
    assert incremental.source_macs == full.source_macs
    ids = np.arange(full.count)
    for a, b in zip(decode_frames(path, incremental, ids),
                    decode_frames(path, full, ids)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not PING.is_file(), reason="MTK ping capture not present")
def test_decoding_a_subset_matches_decoding_everything() -> None:
    idx = MTKIndex(PING)
    every = decode_frames(PING, idx, np.arange(idx.count))
    picks = np.array([3, 11, 12, 30])
    subset = decode_frames(PING, idx, picks)
    for whole, part in zip(every, subset):
        np.testing.assert_array_equal(whole[picks], part)


def test_an_empty_file_indexes_to_nothing(tmp_path: Path) -> None:
    idx = MTKIndex(write(tmp_path, b""))
    assert idx.count == 0 and idx.num_subcarriers == 0
    for arr in decode_frames(tmp_path / "cap.bin", idx, np.array([], dtype=np.int64)):
        assert arr.shape == (0, 0)


# ---------------------------------------------------------------------- #
#  9. Dispatch — the format is chosen by sniffing, not by extension       #
# ---------------------------------------------------------------------- #


def test_get_index_picks_the_reader_from_the_bytes(tmp_path: Path) -> None:
    from backend.index import FrameIndex
    from backend.tiles import get_index, reset_tile_caches

    reset_tile_caches()
    mtk_path = write(tmp_path, group_records(1, {(0, RPI_PLANE): band()}), "x.dat")
    assert isinstance(get_index(mtk_path), MTKIndex), "extension must not decide"
    if FEITCSI.is_file():
        assert isinstance(get_index(FEITCSI), FrameIndex)
    reset_tile_caches()


def test_tiles_decode_frames_routes_to_the_owning_reader(tmp_path: Path) -> None:
    from backend.tiles import decode_frames as dispatched

    path = write(tmp_path, group_records(
        1, {(0, RPI_PLANE): band(seed=1), (1, RPI_PLANE): band(seed=2)}))
    idx = MTKIndex(path)
    ids = np.array([0])
    for routed, direct in zip(dispatched(path, idx, ids), decode_frames(path, idx, ids)):
        np.testing.assert_array_equal(routed, direct)


@pytest.mark.skipif(not PING.is_file(), reason="MTK ping capture not present")
def test_the_api_serves_an_mtk_capture() -> None:
    from fastapi.testclient import TestClient

    from backend.app import app
    from backend.tiles import reset_tile_caches

    reset_tile_caches()
    client = TestClient(app)

    listed = {c["filename"] for c in client.get("/api/captures").json()}
    assert PING.name in listed, "MTK captures must appear in the picker"

    meta = client.get("/api/meta", params={"path": str(PING)}).json()
    assert meta["chipset"] == "MediaTek"  # not the hardcoded Intel label
    assert (meta["total_frames"], meta["num_subcarriers"]) == (38, 256)

    assert client.get("/api/filters", params={"path": str(PING)}).json() == {
        "mimo_modes": ["1x1", "2x1"],
        "source_macs": [MAC],
    }

    body = client.get("/api/tile", params={
        "path": str(PING), "metric": "csi_ratio_phase", "width": 120,
        "t0": meta["t_min"], "t1": meta["t_max"],
        "mimo": "2x1", "source_mac": MAC,
    })
    assert body.status_code == 200
    grid = np.frombuffer(body.content, dtype="<f4").reshape(256, -1)
    # Only the guard band and empty time columns may be blank.
    blank_rows = (~np.isfinite(grid)).all(axis=1).sum()
    blank_cols = (~np.isfinite(grid)).all(axis=0).sum()
    total = (~np.isfinite(grid)).sum()
    assert blank_rows == 11
    assert total == blank_rows * grid.shape[1] + blank_cols * grid.shape[0] \
        - blank_rows * blank_cols, "NaN outside the guard band and time gaps"
    reset_tile_caches()


# ---------------------------------------------------------------------- #
#  10. Cyclic shift                                                       #
# ---------------------------------------------------------------------- #


def _slope(ratio_phase: np.ndarray) -> float:
    """Median per-subcarrier phase step, the quantity the estimator votes on."""
    steps = np.angle(np.exp(1j * np.diff(ratio_phase.astype(float), axis=1)))
    return float(np.nanmedian(np.nanmedian(steps, axis=1)))


def ramped(z: np.ndarray, slope: float) -> np.ndarray:
    """*z* with a linear phase ramp of *slope* rad per subcarrier about DC.

    ``band`` is in raw FFT bin order, and ``fftfreq`` gives each bin its
    signed distance from DC there — the same axis ``decode_frames`` corrects
    along once it has fftshifted. Rounding keeps the result representable in
    the 14 bits the hardware uses.
    """
    k = np.fft.fftfreq(len(z), 1 / len(z))
    out = z * np.exp(1j * slope * k)
    return np.round(out.real) + 1j * np.round(out.imag)


def _ramped_capture(
    tmp_path: Path, slopes, n_bins: int = 256, seed: int = 3, **kw
) -> Path:
    """One group per entry in *slopes*, tpi1 being tpi0 under that ramp."""
    blob = b"".join(
        group_records(
            g,
            {(0, RPI_PLANE): band(n_bins, seed=seed + g),
             (1, RPI_PLANE): ramped(band(n_bins, seed=seed + g), sl)},
            ts=1000 + 50 * g,
            **kw,
        )
        for g, sl in enumerate(slopes)
    )
    return write(tmp_path, blob)


def test_a_constant_ramp_is_measured_and_removed(tmp_path: Path) -> None:
    path = _ramped_capture(tmp_path, [0.7793] * 12)
    idx = MTKIndex(path)
    assert idx.csd_slope() == pytest.approx(0.7793, abs=0.01)

    ids = np.arange(idx.count)
    assert _slope(decode_frames(path, idx, ids, deslope=False)[3]) == pytest.approx(
        0.7793, abs=0.01
    )
    assert abs(_slope(decode_frames(path, idx, ids)[3])) < 0.01


def test_deslope_touches_the_ratio_phase_and_nothing_else(tmp_path: Path) -> None:
    """Only one of the four arrays moves.

    ``amplitude`` and ``phase`` read rx0, which the correction never reaches.
    ``ratio_amp`` is reached but not moved: a unit-magnitude rotation cannot
    change a magnitude, up to ``exp`` not being exactly unit-modulus in
    binary — the residual measures a few 1e-15 dB.
    """
    path = _ramped_capture(tmp_path, [0.7793] * 12)
    idx = MTKIndex(path)
    ids = np.arange(idx.count)
    off = decode_frames(path, idx, ids, deslope=False)
    on = decode_frames(path, idx, ids)

    for name, a, b in zip(("amplitude", "phase"), off, on):
        np.testing.assert_array_equal(
            np.nan_to_num(a, nan=-999.0), np.nan_to_num(b, nan=-999.0),
            err_msg=f"{name} must not move",
        )
    np.testing.assert_allclose(
        np.nan_to_num(off[2], nan=-999.0), np.nan_to_num(on[2], nan=-999.0),
        atol=1e-12, err_msg="ratio_amp must not move beyond floating point",
    )
    assert not np.allclose(off[3], on[3], equal_nan=True)


def test_a_capture_without_a_ramp_is_left_alone(tmp_path: Path) -> None:
    """A transmitter with one chain applies no cyclic shift; invent none."""
    path = _ramped_capture(tmp_path, [0.0] * 12)
    idx = MTKIndex(path)
    assert idx.csd_slope() is None

    ids = np.arange(idx.count)
    np.testing.assert_array_equal(
        np.nan_to_num(decode_frames(path, idx, ids, deslope=False)[3], nan=-999.0),
        np.nan_to_num(decode_frames(path, idx, ids)[3], nan=-999.0),
    )


def test_frames_disagreeing_on_the_sign_are_refused(tmp_path: Path) -> None:
    """A ramp deep enough to pass the floor, but only 7 of 12 frames agree.

    The magnitude gate alone would accept this; a real cyclic shift is
    deterministic, so a split this wide means the quantity is not one.
    """
    path = _ramped_capture(tmp_path, [0.7793] * 7 + [-0.7793] * 5)
    idx = MTKIndex(path)
    assert _slope(decode_frames(path, idx, np.arange(12), deslope=False)[3]) > 0.5
    assert idx.csd_slope() is None


def test_too_few_two_stream_frames_to_vote(tmp_path: Path) -> None:
    path = _ramped_capture(tmp_path, [0.7793] * (CSD_MIN_FRAMES - 1))
    assert MTKIndex(path).csd_slope() is None


def test_both_bandwidths_are_corrected_about_their_own_dc(tmp_path: Path) -> None:
    """A 20 MHz frame sits centred in a 256-wide row; the ramp is about DC.

    802.11 spaces subcarriers 312.5 kHz apart at 20 and 80 MHz alike, so one
    cyclic shift is the same rad/subcarrier in both, and both bands are
    centred on DC. Anchoring the correction at each band's own centre is
    what keeps the narrow frame from being corrected about the wide frame's
    edge. No capture on hand mixes bandwidths among two-stream frames, so
    this case exists only here.
    """
    wide = b"".join(
        group_records(
            g,
            {(0, RPI_PLANE): band(256, seed=g),
             (1, RPI_PLANE): ramped(band(256, seed=g), 0.7793)},
            ts=1000 + 50 * g,
        )
        for g in range(12)
    )
    narrow = group_records(
        99,
        {(0, RPI_PLANE): band(64, seed=99),
         (1, RPI_PLANE): ramped(band(64, seed=99), 0.7793)},
        bw_code=0,
        ts=2000,
    )
    path = write(tmp_path, wide + narrow)
    idx = MTKIndex(path)
    assert idx.num_subcarriers == 256
    assert idx.channel_widths[-1] == "20"

    ratio_phase = decode_frames(path, idx, np.array([idx.count - 1]))[3]
    live = np.isfinite(ratio_phase[0])
    assert live.sum() > 30, "the narrow frame should decode"
    assert abs(_slope(ratio_phase)) < 0.01

    # Centred, not left-aligned: the live span straddles the row's midpoint.
    span = np.flatnonzero(live)
    assert span.min() < 128 < span.max()


def test_the_estimate_is_anchored_to_the_file_not_the_batch(tmp_path: Path) -> None:
    """Decoding a slice must give that slice the same numbers as the whole.

    The reason ``ratio.Reference`` exists, applied to the ramp: an estimate
    rebuilt per view would let panning change the picture.
    """
    path = _ramped_capture(tmp_path, [0.7793] * 12)
    idx = MTKIndex(path)
    whole = decode_frames(path, idx, np.arange(12))[3]
    part = decode_frames(path, idx, np.arange(4, 8))[3]
    np.testing.assert_allclose(part, whole[4:8], equal_nan=True)


def test_extend_keeps_the_measurement(tmp_path: Path) -> None:
    """A cyclic shift belongs to the transmitter, not to how much has arrived."""
    blob = _ramped_capture(tmp_path, [0.7793] * 12).read_bytes()
    path = write(tmp_path, blob, name="growing.bin")
    idx = MTKIndex(path)
    first = idx.csd_slope()
    assert first is not None

    with path.open("ab") as fh:
        fh.write(blob)
    idx.extend()
    assert idx.count == 24
    assert idx.csd_slope() == first


@pytest.mark.skipif(not PING.is_file(), reason="MTK ping capture not present")
def test_the_ping_capture_carries_the_standard_cyclic_shift() -> None:
    """~400 ns is what 802.11 specifies for a second stream."""
    idx = MTKIndex(PING)
    slope = idx.csd_slope()
    assert slope is not None
    delay_ns = slope / (2 * np.pi * 312_500) * 1e9
    assert delay_ns == pytest.approx(400, abs=10)

    ids = np.flatnonzero(idx.filter_mask(mimo=(2, 1)))
    raw = decode_frames(PING, idx, ids, deslope=False)[3]
    fixed = decode_frames(PING, idx, ids)[3]
    assert _slope(raw) == pytest.approx(slope, abs=0.01)
    assert abs(_slope(fixed)) < 0.01

    # 30 wraps across the band leave the raw phase indistinguishable from
    # uniform; removing them is what makes a subcarrier-axis mean mean anything.
    live = np.isfinite(raw)
    assert abs(np.mean(np.exp(1j * raw[live]))) < 0.05
    assert abs(np.mean(np.exp(1j * fixed[live]))) > 0.3

