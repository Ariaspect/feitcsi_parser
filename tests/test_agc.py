"""Tests for backend.agc — per-gain-state amplitude correction."""

from __future__ import annotations

import numpy as np
import pytest

from backend import agc
from backend.agc import GainTable, apply_gain_table, build_gain_table

N_SC = 64
DOMINANT = -45


def _capture(
    n: int = 2000,
    *,
    off_state: int = -49,
    off_every: int = 20,
    tilt: np.ndarray | None = None,
    drift_db: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A synthetic capture with one injected gain-state distortion.

    Returns ``(amplitude, rssi, tilt)``. The room is a fixed per-subcarrier
    profile, optionally drifting linearly over the capture so a test can check
    the correction does not absorb it. Frames in *off_state* get *tilt* added.
    """
    rng = np.random.default_rng(seed)
    if tilt is None:
        tilt = np.linspace(-6.0, 6.0, N_SC)
    tilt = tilt - np.median(tilt)

    room = 20.0 + 5.0 * np.sin(np.linspace(0, 3 * np.pi, N_SC))
    amp = np.tile(room, (n, 1)).astype(float)
    if drift_db:
        amp += np.linspace(0, drift_db, n)[:, None] * np.linspace(
            -1.0, 1.0, N_SC
        )[None, :]
    if noise:
        amp += rng.normal(0, noise, amp.shape)

    rssi = np.full(n, DOMINANT, dtype=np.int64)
    rssi[off_every // 2 :: off_every] = off_state
    amp[rssi == off_state] += tilt
    return amp, rssi, tilt


def test_single_gain_state_yields_no_table():
    """Nothing to correct against, so nothing is claimed."""
    amp, rssi, _ = _capture(off_every=10**9)
    assert build_gain_table([(amp, rssi)]) is None


def test_recovers_the_injected_offset():
    amp, rssi, tilt = _capture()
    table = build_gain_table([(amp, rssi)])
    assert table is not None
    assert table.dominant == DOMINANT
    assert table.n_states == 1
    assert np.allclose(table.offsets[-49], tilt, atol=1e-6)


def test_correction_removes_the_distortion():
    amp, rssi, _ = _capture()
    table = build_gain_table([(amp, rssi)])
    out = apply_gain_table(amp, rssi, table)

    def shape(a, m):
        p = a - np.median(a, axis=1, keepdims=True)
        return np.median(p[m], axis=0)

    off, dom = rssi == -49, rssi == DOMINANT
    before = np.mean(np.abs(shape(amp, off) - shape(amp, dom)))
    after = np.mean(np.abs(shape(out, off) - shape(out, dom)))
    assert before > 2.0
    assert after < 0.01


def test_offsets_carry_no_level_term():
    """The measured artifact is shape-only, so the correction must be too."""
    amp, rssi, _ = _capture()
    table = build_gain_table([(amp, rssi)])
    for off in table.offsets.values():
        assert abs(float(np.median(off))) < 1e-9


def test_dominant_state_is_untouched():
    amp, rssi, _ = _capture()
    table = build_gain_table([(amp, rssi)])
    out = apply_gain_table(amp, rssi, table)
    dom = rssi == DOMINANT
    assert np.array_equal(out[dom], amp[dom].astype(np.float32))


def test_rare_state_is_left_uncorrected():
    """A state seen a handful of times gets no offset rather than a bad one."""
    amp, rssi, _ = _capture()
    # A third state appearing far fewer times than MIN_STATE_FRAMES.
    rare = np.flatnonzero(rssi == DOMINANT)[:5]
    rssi[rare] = -60
    amp[rare] += 9.0
    table = build_gain_table([(amp, rssi)])
    assert -60 not in table.offsets
    out = apply_gain_table(amp, rssi, table)
    assert np.array_equal(out[rare], amp[rare].astype(np.float32))


def test_no_table_is_an_identity():
    amp, rssi, _ = _capture()
    assert apply_gain_table(amp, rssi, None) is amp


def test_nan_bins_stay_nan():
    amp, rssi, _ = _capture()
    amp[:, 3] = np.nan
    table = build_gain_table([(amp, rssi)])
    out = apply_gain_table(amp, rssi, table)
    assert np.isnan(out[:, 3]).all()
    assert np.isfinite(out[:, 0]).all()


def test_room_drift_is_not_absorbed():
    """The comparison is local, so a slowly changing room stays in the data.

    A global off-minus-dominant difference would book part of the drift as a
    gain offset. With the drift spread evenly over the capture and the
    off-state frames evenly spaced, that error averages out — so the drift is
    made to correlate with nothing and the offset must still come back as the
    tilt alone.
    """
    tilt = np.linspace(-4.0, 4.0, N_SC)
    amp, rssi, _ = _capture(tilt=tilt, drift_db=8.0)
    table = build_gain_table([(amp, rssi)])
    assert np.allclose(table.offsets[-49], tilt - np.median(tilt), atol=0.05)


def test_segments_are_pooled():
    a1, r1, tilt = _capture(n=1000, seed=1)
    a2, r2, _ = _capture(n=1000, seed=2)
    table = build_gain_table([(a1, r1), (a2, r2)])
    assert table is not None
    assert np.allclose(table.offsets[-49], tilt, atol=1e-6)


def test_mismatched_segment_widths_raise():
    a1, r1, _ = _capture(n=500)
    a2, r2, _ = _capture(n=500)
    with pytest.raises(ValueError, match="same subcarrier count"):
        build_gain_table([(a1, r1), (a2[:, :16], r2)])


def test_rssi_length_must_match():
    amp, rssi, _ = _capture(n=500)
    with pytest.raises(ValueError, match="rssi must be 1-D"):
        build_gain_table([(amp, rssi[:-1])])
    table = build_gain_table([(amp, rssi)])
    with pytest.raises(ValueError, match="entries for"):
        apply_gain_table(amp, rssi[:-1], table)


def test_amplitude_must_be_2d():
    with pytest.raises(ValueError, match="must be 2-D"):
        build_gain_table([(np.zeros(10), np.zeros(10))])


def test_empty_input_is_none():
    assert build_gain_table([]) is None
    assert build_gain_table([(np.zeros((0, N_SC)), np.zeros(0))]) is None


def test_long_off_run_without_local_dominant_is_skipped():
    """A frame with no dominant neighbours contributes nothing.

    It is the guard that keeps the local comparison local: a stretch where the
    gain never returns has no nearby reference, and guessing one from the far
    side of the capture is the global comparison this deliberately avoids.
    """
    amp, rssi, _ = _capture(n=600, off_every=10**9)
    rssi[:400] = -49            # one long run, no dominant frame within reach
    amp[:400] += 5.0
    table = build_gain_table([(amp, rssi)], half_width=20)
    # The first frames near the boundary can still see dominant neighbours, so
    # what must hold is that the run does not produce a confident offset from
    # frames that had none.
    if table is not None:
        assert table.n_frames < 400


def test_apply_on_empty_is_safe():
    table = GainTable(offsets={-49: np.zeros(N_SC)}, dominant=-45, n_frames=0)
    out = apply_gain_table(np.zeros((0, N_SC)), np.zeros(0), table)
    assert out.shape == (0, N_SC)


def test_module_constants_are_coherent():
    assert agc.MIN_LOCAL_FRAMES <= agc.MIN_STATE_FRAMES
    assert agc.DEFAULT_HALF_WIDTH >= agc.MIN_LOCAL_FRAMES


# ---------------------------------------------------------------------- #
#  Integration: the correction as the tile pipeline applies it            #
# ---------------------------------------------------------------------- #

from pathlib import Path  # noqa: E402

from backend import tiles  # noqa: E402
from tests.test_mtk import band, group_records, write  # noqa: E402

RPI = 0


def _mtk_capture_with_gain_steps(tmp_path: Path, n: int = 900) -> Path:
    """A synthetic MTK capture whose every 20th frame sits in a lower gain state.

    The distortion is injected the way the radio produces it: the off-state
    frames' tpi0 stream is scaled per subcarrier, so their amplitude SHAPE
    differs while the room behind them does not.
    """
    tpi0, tpi1 = band(seed=1), band(seed=2)
    tilt = 10 ** (np.linspace(-4.0, 4.0, 256) / 20.0)   # +/-4 dB across the band
    blob = []
    for g in range(n):
        off = (g % 20) == 10
        a = tpi0 * tilt if off else tpi0
        blob.append(
            group_records(
                g,
                {(0, RPI): a, (1, RPI): tpi1},
                ts=1000 + 50 * g,
                # tag 3 is a signed byte; 211 -> -45, 207 -> -49
                rssi=207 if off else 211,
            )
        )
    return write(tmp_path, b"".join(blob), "gain.bin")


def test_tiles_builds_a_gain_table(tmp_path: Path):
    p = _mtk_capture_with_gain_steps(tmp_path)
    idx = tiles.get_index(p)
    table = tiles.get_gain_table(p, idx, p.stat().st_size)
    assert table is not None
    assert table.dominant == -45
    assert table.n_states == 1
    assert -49 in table.offsets


def test_tile_amplitude_is_corrected_and_the_ratio_is_not(tmp_path: Path):
    p = _mtk_capture_with_gain_steps(tmp_path)
    tiles.reset_tile_caches()

    def grid(metric, agc_correct):
        g, meta = tiles.compute_tile(
            p, 0.0, 45.0, 400, metric, agc_correct=agc_correct
        )
        return g, meta

    amp_on, meta_on = grid("amplitude", True)
    amp_off, meta_off = grid("amplitude", False)
    assert meta_on["agc_corrected"] is True
    assert meta_on["agc_states"] == 1
    assert meta_off["agc_corrected"] is False
    assert np.nanmax(np.abs(amp_on - amp_off)) > 1.0

    # The ratio divides the gain out, so the table must not reach it.
    r_on, r_meta = grid("csi_ratio_amplitude", True)
    r_off, _ = grid("csi_ratio_amplitude", False)
    assert r_meta["agc_corrected"] is False
    assert np.array_equal(np.nan_to_num(r_on), np.nan_to_num(r_off))


def test_agc_affected_follows_the_derivation_graph():
    assert tiles._agc_affected("amplitude")
    # csi_cir moved onto the ratio planes, which divide the gain out, so it
    # no longer inherits the correction -- and the recursion worked that out
    # on its own when the metric's bases changed.
    assert not tiles._agc_affected("csi_cir")
    assert not tiles._agc_affected("csi_ratio_amplitude")
    assert not tiles._agc_affected("phase")
    assert not tiles._agc_affected("csi_ratio_phase_corrected")


def test_toggling_agc_does_not_serve_a_stale_tile(tmp_path: Path):
    """The tag is in the cache key, so the two settings cannot collide."""
    p = _mtk_capture_with_gain_steps(tmp_path)
    tiles.reset_tile_caches()
    a1, _ = tiles.compute_tile(p, 0.0, 45.0, 400, "amplitude", agc_correct=True)
    b1, _ = tiles.compute_tile(p, 0.0, 45.0, 400, "amplitude", agc_correct=False)
    a2, _ = tiles.compute_tile(p, 0.0, 45.0, 400, "amplitude", agc_correct=True)
    assert np.array_equal(np.nan_to_num(a1), np.nan_to_num(a2))
    assert not np.array_equal(np.nan_to_num(a1), np.nan_to_num(b1))


# ---------------------------------------------------------------------- #
#  Per-frame correction along the shared direction                        #
# ---------------------------------------------------------------------- #


def _bimodal_capture(n: int = 2000, seed: int = 3):
    """One reported RSSI value covering TWO real gain states.

    This is the case a per-state offset cannot serve: half the frames at
    RSSI -48 carry the deep distortion and half the shallow one, so any single
    offset for that state is wrong for one half. Measured on
    20260911_095127, where 39 of its -48 frames sit with the deep states and
    85 with the shallow ones.
    """
    rng = np.random.default_rng(seed)
    shape = np.linspace(-1.0, 1.0, N_SC) ** 3
    shape = shape - np.median(shape)
    shape = shape / np.linalg.norm(shape)

    room = 20.0 + 5.0 * np.sin(np.linspace(0, 3 * np.pi, N_SC))
    amp = np.tile(room, (n, 1)).astype(float)
    amp += rng.normal(0, 0.02, amp.shape)

    rssi = np.full(n, DOMINANT, dtype=np.int64)
    amb = np.arange(10, n, 20)
    rssi[amb] = -48
    deep = amb[::2]
    shallow = amb[1::2]
    amp[deep] += 90.0 * shape
    amp[shallow] += 30.0 * shape
    return amp, rssi, shape, deep, shallow


def test_per_frame_separates_states_one_rssi_cannot():
    amp, rssi, shape, deep, shallow = _bimodal_capture()
    table = build_gain_table([(amp, rssi)])
    assert table.basis is not None

    per_state = apply_gain_table(amp, rssi, table, contiguous=False)
    per_frame = apply_gain_table(amp, rssi, table, contiguous=True)

    def residual(a, rows):
        p = a - np.median(a, axis=1, keepdims=True)
        base = np.median(p[rssi == DOMINANT], axis=0)
        return float(np.mean(np.abs(np.median(p[rows], axis=0) - base)))

    for rows, name in ((deep, "deep"), (shallow, "shallow")):
        raw = residual(amp, rows)
        state = residual(per_state, rows)
        frame = residual(per_frame, rows)
        assert frame < state, f"{name}: per-frame {frame} not better than {state}"
        assert frame < 0.1 * raw, f"{name}: per-frame left {frame} of {raw}"


def test_per_frame_needs_contiguous_rows():
    """Without consecutive frames there are no neighbours, so it must not try."""
    amp, rssi, *_ = _bimodal_capture()
    table = build_gain_table([(amp, rssi)])
    a = apply_gain_table(amp, rssi, table, contiguous=False)
    b = apply_gain_table(amp, rssi, table, contiguous=True)
    assert not np.allclose(a, b)


def test_dominant_frames_are_untouched_per_frame():
    amp, rssi, *_ = _bimodal_capture()
    table = build_gain_table([(amp, rssi)])
    out = apply_gain_table(amp, rssi, table, contiguous=True)
    dom = rssi == DOMINANT
    assert np.array_equal(out[dom], amp[dom].astype(np.float32))


def test_the_basis_is_orthonormal_and_carries_no_level():
    """Both properties are load-bearing, and one of them broke silently.

    A plain dot product is only a projection against an ORTHONORMAL basis.
    Re-centring each row on its own median makes each row level-free but
    rotates them by different amounts, and two rows that started perpendicular
    stop being so -- measured here at a correlation of 0.75, whose
    coefficients then counted the same direction twice and over-subtracted by
    1.8x. Orthogonality to the constant vector is what carries "no level"
    instead, because unlike a median it survives being combined.
    """
    amp, rssi, *_ = _bimodal_capture()
    table = build_gain_table([(amp, rssi)])
    B = table.basis
    assert B.ndim == 2
    assert B.shape[0] <= agc.N_COMPONENTS
    live = np.any(B != 0.0, axis=0)
    gram = B[:, live] @ B[:, live].T
    assert np.allclose(gram, np.eye(B.shape[0]), atol=1e-8), gram
    for row in B:
        assert abs(float(np.mean(row[live]))) < 1e-8


def test_a_correction_in_the_basis_span_has_no_level_term():
    amp, rssi, *_ = _bimodal_capture()
    table = build_gain_table([(amp, rssi)])
    B = table.basis
    live = np.any(B != 0.0, axis=0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        k = rng.normal(0, 50, B.shape[0])
        assert abs(float(np.mean((k @ B)[live]))) < 1e-7


def test_the_leading_direction_matches_the_injected_one():
    amp, rssi, shape, *_ = _bimodal_capture()
    table = build_gain_table([(amp, rssi)])
    # Sign is arbitrary in an SVD, so compare on absolute correlation.
    c = abs(float(np.corrcoef(table.basis[0], shape)[0, 1]))
    assert c > 0.99


def test_null_floor_is_measured_and_positive():
    amp, rssi, *_ = _bimodal_capture()
    table = build_gain_table([(amp, rssi)])
    assert table.null_k > 0.0
    # It has to sit far below the real coefficients, or nothing would fire.
    assert table.null_k < 10.0


def test_a_frame_below_the_null_floor_is_left_alone():
    """Below the floor the artifact cannot be told from the room."""
    amp, rssi, shape, deep, shallow = _bimodal_capture()
    table = build_gain_table([(amp, rssi)])
    # A frame whose distortion is a fraction of the measured floor.
    quiet = np.array([5])
    rssi2 = rssi.copy()
    amp2 = amp.copy()
    rssi2[quiet] = -46
    amp2[quiet] += (table.null_k * 0.1) * shape
    out = apply_gain_table(amp2, rssi2, table, contiguous=True)
    assert np.allclose(out[quiet], amp2[quiet].astype(np.float32), atol=1e-4)


def test_too_few_deltas_yields_no_shape():
    amp, rssi, _ = _capture(n=400, off_every=200)   # only a couple of off frames
    table = build_gain_table([(amp, rssi)])
    if table is not None:
        assert table.basis is None or table.n_frames >= agc.MIN_SHAPE_FRAMES


def test_scalar_projection_ignores_nan_bins():
    amp, rssi, *_ = _bimodal_capture()
    amp[:, 5] = np.nan
    table = build_gain_table([(amp, rssi)])
    out = apply_gain_table(amp, rssi, table, contiguous=True)
    assert np.isnan(out[:, 5]).all()
    assert np.isfinite(out[:, 0]).all()


def test_below_the_floor_a_frame_still_gets_its_state_offset():
    """The floor decides who corrects a frame, never whether one does.

    Marking a frame handled when the floor rejected it would strip the
    fallback from every frame below it, which on a capture where almost
    nothing clears the floor leaves the artifact whole -- measured on
    20260909_200646, where 0.1% of frames clear it.
    """
    amp, rssi, *_ = _bimodal_capture()
    table = build_gain_table([(amp, rssi)])
    assert table.offsets, "the state should be populated enough to measure"

    # A floor nothing can clear, so the per-frame pass must defer entirely.
    blocked = table._replace(null_k=1e9)
    per_state = apply_gain_table(amp, rssi, blocked, contiguous=False)
    per_frame = apply_gain_table(amp, rssi, blocked, contiguous=True)
    assert np.array_equal(per_frame, per_state)

    off = rssi != DOMINANT
    assert not np.allclose(per_frame[off], amp[off].astype(np.float32))


def test_per_frame_never_undoes_the_state_correction():
    """Whatever the floor decides, the result is no further from the room."""
    amp, rssi, *_ = _bimodal_capture()
    table = build_gain_table([(amp, rssi)])
    a = apply_gain_table(amp, rssi, table, contiguous=False)
    b = apply_gain_table(amp, rssi, table, contiguous=True)

    def err(x):
        p = x - np.median(x, axis=1, keepdims=True)
        base = np.median(p[rssi == DOMINANT], axis=0)
        off = rssi != DOMINANT
        return float(np.mean(np.abs(p[off] - base)))

    assert err(b) <= err(a) + 1e-6
