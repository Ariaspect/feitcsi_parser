"""Tests for backend.farsense -- the FarSense blind copy."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import farsense as fz

FS = 19.61          # the link's 2x1 frame rate on the September captures
N_SC = 32


def _chest(
    rpm: float,
    *,
    seconds: float = 120.0,
    depth: float = 0.03,
    noise: float = 0.03,
    fs: float = FS,
    n_sc: int = N_SC,
    blind: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """A static ratio of 1 plus one chest, seen along a random axis per subcarrier.

    ``blind`` puts the chest purely along the imaginary axis of a real static
    term, where the magnitude sees only the second-order term -- the paper's
    "blind spot" for amplitude-only sensing.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    wave = np.sin(2 * np.pi * rpm / 60.0 * t)[:, None]
    axis = (
        1j * np.ones(n_sc)
        if blind
        else np.exp(1j * rng.uniform(0, 2 * np.pi, n_sc))
    )
    sig = 1.0 + depth * wave * axis[None, :]
    return sig + noise * (
        rng.standard_normal((t.size, n_sc)) + 1j * rng.standard_normal((t.size, n_sc))
    )


# --------------------------------------------------------------------------- #
#  Projection with maximal periodicity (Sec. 5.2)                              #
# --------------------------------------------------------------------------- #


def test_bnr_of_a_pure_tone_is_n_over_fft_size() -> None:
    """The one-sided convention that reproduces Fig. 16's 0.13 weights."""
    win = 240
    t = np.arange(win) / FS
    tone = (0.0 + 1j * np.sin(2 * np.pi * 0.25 * t))[:, None]
    ext = fz.extract_patterns(tone, FS)
    assert ext["bnr"][0] == pytest.approx(win / fz.FFT_SIZE, rel=0.05)


def test_the_best_axis_is_the_axis_the_chest_moves_along() -> None:
    """A chest along the imaginary axis is found near theta = pi/2, with a
    pattern that IS the chest -- and the magnitude alone sees nothing.

    Near, not at: BNR is scale-free, so every projection of a clean arc
    scores the same and only the noise decides between neighbouring angles.
    What must hold is that the pattern is the breath and |r| is not.
    """
    sig = _chest(15.0, blind=True, noise=0.005, seconds=12.0)
    ext = fz.extract_patterns(sig, FS)
    t = np.arange(sig.shape[0]) / FS
    wave = np.sin(2 * np.pi * 0.25 * t)

    theta = ext["theta"] % np.pi
    assert np.all(np.abs(theta - np.pi / 2) < np.pi / 8)
    for k in range(sig.shape[1]):
        assert abs(np.corrcoef(ext["pattern"][:, k], wave)[0, 1]) > 0.95
    magnitude = np.abs(sig[:, 0])
    assert abs(np.corrcoef(magnitude, wave)[0, 1]) < 0.3, "the magnitude really is blind here"


def test_periodicity_beats_variance_when_choosing_the_axis() -> None:
    """Sec. 5.2.2: a max-variance rule picks the drift, BNR picks the chest.

    The real axis carries a large slow drift (all its energy below the band)
    and the imaginary axis a small breath. Variance points at the drift; BNR
    points at the breath.
    """
    win = 240
    t = np.arange(win) / FS
    drift = 0.5 * (t / t[-1] - 0.5)                 # linear, out of band
    breath = 0.02 * np.sin(2 * np.pi * 0.3 * t)
    sig = (drift + 1j * breath)[:, None]
    ext = fz.extract_patterns(sig, FS)

    theta = ext["theta"][0] % np.pi
    assert abs(theta - np.pi / 2) < np.pi / 25
    # ...and the axis of maximal variance really is the other one.
    x, y = sig.real[:, 0], sig.imag[:, 0]
    assert np.var(x) > 10 * np.var(y)


def test_dead_subcarriers_score_zero_and_stay_nan() -> None:
    sig = _chest(15.0, seconds=12.0)
    sig[:, 3] = np.nan
    ext = fz.extract_patterns(sig, FS)
    assert ext["bnr"][3] == 0.0
    assert np.isnan(ext["pattern"][:, 3]).all()
    assert np.isfinite(ext["pattern"][:, 0]).all()


# --------------------------------------------------------------------------- #
#  Rate by autocorrelation (Sec. 6.4)                                          #
# --------------------------------------------------------------------------- #


def test_subcarriers_below_seven_tenths_of_the_best_are_excluded() -> None:
    pattern = np.random.default_rng(1).standard_normal((240, 5))
    bnr = np.array([0.10, 0.08, 0.06, 0.01, 0.0])
    _, selected = fz.combine_autocorrelations(pattern, bnr)
    assert selected.tolist() == [True, True, False, False, False]


def test_the_combined_autocorrelation_is_the_weighted_sum_of_eq_11() -> None:
    rng = np.random.default_rng(2)
    pattern = rng.standard_normal((240, 3))
    bnr = np.array([0.10, 0.09, 0.08])
    r_msc, selected = fz.combine_autocorrelations(pattern, bnr)
    assert selected.all()
    assert r_msc[0] == pytest.approx(bnr.sum())


def test_first_peak_is_the_first_local_maximum_in_band_not_the_largest() -> None:
    """Two peaks in band: the paper reads the first one."""
    lags = np.arange(240)
    r = 0.4 * np.cos(2 * np.pi * lags / 50) + 0.6 * np.cos(2 * np.pi * lags / 100)
    pk = fz.first_peak(r, FS)
    # Lag 50 is the first local maximum (23.5 rpm at 19.61 Hz); lag 100 is higher.
    assert pk["lag"] == 50
    assert r[100] > r[50]


def test_no_peak_in_band_reports_no_rate() -> None:
    r = np.exp(-np.arange(240) / 30.0)           # monotone: nothing periodic
    pk = fz.first_peak(r, FS)
    assert np.isnan(pk["lag"]) and np.isnan(pk["lag_refined"])


def test_the_refined_lag_undoes_the_biased_estimator_lean() -> None:
    """Eq. 10's taper leans every peak towards lag zero. The refinement
    divides the taper back out before fitting, so the read lands on the
    true period rather than a few samples short of it."""
    n = 1200
    period = 400.3
    k = np.arange(n)
    r = (1.0 - k / n) * np.cos(2 * np.pi * k / period)
    pk = fz.first_peak(r, 100.0)
    assert pk["lag"] < period - 3                       # the raw peak leans short
    assert pk["lag_refined"] == pytest.approx(period, abs=0.1)


# --------------------------------------------------------------------------- #
#  The whole pipeline                                                          #
# --------------------------------------------------------------------------- #


# The paper's own parameters, pinned: the module defaults are the user's set
# (docs/farsense.md) and a 10 s window holds only two periods at 12 rpm.
PAPER = dict(window_seconds=12.0, band_rpm=(10.0, 37.0), keep_fraction=0.7, n_theta=100, savgol_seconds=0.5)


@pytest.mark.parametrize("rpm", [12.0, 15.0, 25.0, 30.0])
def test_a_chest_is_recovered_within_the_papers_half_rpm(rpm: float) -> None:
    out = fz.farsense_windows(_chest(rpm), FS, **PAPER)
    est = out["rpm"]
    assert np.isfinite(est).mean() > 0.95
    assert np.mean(np.abs(est[np.isfinite(est)] - rpm) < 0.5) > 0.95
    assert out["stationary"].all()


def test_the_blind_spot_is_not_blind_here() -> None:
    out = fz.farsense_windows(_chest(15.0, blind=True), FS)
    est = out["rpm"]
    assert np.nanmedian(est) == pytest.approx(15.0, abs=0.3)


def test_gross_motion_blanks_the_rate_and_says_so() -> None:
    sig = _chest(15.0)
    rng = np.random.default_rng(3)
    n = sig.shape[0]
    a, b = n // 3, n // 3 + int(20 * FS)
    # A walk: the ratio wanders by more than itself between samples.
    sig[a:b] *= np.exp(1j * np.cumsum(rng.uniform(-2.0, 2.0, (b - a, 1)), axis=0))
    sig[a:b] *= 1.0 + rng.uniform(-0.6, 0.6, (b - a, 1))
    out = fz.farsense_windows(sig, FS)
    inside = (out["time_s"] > a / FS + 2) & (out["time_s"] < b / FS - 2)
    assert not out["stationary"][inside].any()
    assert np.isnan(out["rpm"][inside]).all()
    outside = out["time_s"] < a / FS - out["window_seconds"]
    assert out["stationary"][outside].all()


def test_the_paper_reports_a_rate_for_an_empty_room_and_min_peak_removes_it() -> None:
    """The copy is faithful: with no gate the empty room gets a number. The
    normalised peak is what separates it, and `min_peak` applies it."""
    rng = np.random.default_rng(4)
    n = int(120 * FS)
    empty = 1.0 + 0.03 * (rng.standard_normal((n, N_SC)) + 1j * rng.standard_normal((n, N_SC)))
    faithful = fz.farsense_windows(empty, FS)
    gated = fz.farsense_windows(empty, FS, min_peak=0.2)
    chest = fz.farsense_windows(_chest(15.0), FS, min_peak=0.2)

    assert np.isfinite(faithful["rpm"]).mean() > 0.3
    assert np.nanmedian(faithful["acf_peak_norm"]) < 0.1
    assert np.isfinite(gated["rpm"]).mean() < 0.05
    assert np.isfinite(chest["rpm"]).mean() > 0.95


def test_the_stitched_pattern_is_continuous_across_windows() -> None:
    """Consecutive windows can carry the chest inverted; the stitch aligns them."""
    out = fz.farsense_windows(_chest(15.0, noise=0.01), FS)
    p = out["pattern"]
    t = out["pattern_t"]
    wave = np.sin(2 * np.pi * 0.25 * t)
    ok = np.isfinite(p)
    assert ok.mean() > 0.99
    assert abs(np.corrcoef(p[ok], wave[ok])[0, 1]) > 0.95


def test_fabricated_windows_report_nothing() -> None:
    sig = _chest(15.0)
    fab = np.zeros(sig.shape[0], dtype=bool)
    fab[200:500] = True
    out = fz.farsense_windows(sig, FS, fabricated=fab)
    centre = (out["time_s"] > 350 / FS - 3) & (out["time_s"] < 350 / FS + 3)
    assert out["unknown"][centre].all()
    assert np.isnan(out["rpm"][centre]).all()
    assert np.isnan(out["pattern"][200:500]).all()


def test_detail_returns_the_requested_window_in_full() -> None:
    out = fz.farsense_windows(_chest(18.0), FS, detail_index=7)
    d = out["detail"]
    assert d is not None and d["index"] == 7
    assert d["iq"].shape == (out["win"], 2)
    assert d["pattern"].shape == (out["win"],)
    assert d["acf"].shape == (out["win"],)
    assert d["bnr"].shape == d["theta"].shape == d["selected"].shape == (N_SC,)
    assert d["lag_lo"] <= d["lag"] <= d["lag_hi"]


def test_a_band_the_window_cannot_hold_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot hold"):
        fz.farsense_windows(_chest(15.0, seconds=30.0), FS, window_seconds=4.0)


# --------------------------------------------------------------------------- #
#  The endpoint                                                                #
# --------------------------------------------------------------------------- #

from pathlib import Path  # noqa: E402

from backend.app import app  # noqa: E402
from tests.test_mtk import band, group_records, write  # noqa: E402


def _mtk_capture_with_a_chest(tmp_path: Path, rpm: float = 15.0, n: int = 1200) -> Path:
    """A synthetic MTK capture at 20 Hz whose tpi1/tpi0 ratio carries a chest."""
    tpi0 = band(seed=1)
    tpi1 = band(seed=2)
    rng = np.random.default_rng(5)
    axis = np.exp(1j * rng.uniform(0, 2 * np.pi, tpi0.size))
    blob = []
    for g in range(n):
        t = 0.05 * g
        breath = 1.0 + 0.03 * np.sin(2 * np.pi * rpm / 60.0 * t) * axis
        noise = 1.0 + 0.01 * (rng.standard_normal(tpi0.size) + 1j * rng.standard_normal(tpi0.size))
        blob.append(
            group_records(g, {(0, 0): tpi0, (1, 0): tpi1 * breath * noise}, ts=1000 + 50 * g)
        )
    return write(tmp_path, b"".join(blob), "chest.bin")


def test_endpoint_reads_the_chest_and_returns_a_detail(tmp_path: Path) -> None:
    p = _mtk_capture_with_a_chest(tmp_path)
    client = TestClient(app)
    r = client.get("/api/farsense", params={"path": str(p), "t0": 0.0, "t1": 60.0, "detail_t": 30.0})
    assert r.status_code == 200, r.text
    body = r.json()
    n = len(body["time_s"])
    assert n > 0
    for key in ("stationary", "rpm", "acf_peak_norm", "bnr_max", "n_selected", "best_sc"):
        assert len(body[key]) == n
    assert len(body["bnr_map"]) == len(body["sc_index"])
    assert all(len(row) == n for row in body["bnr_map"])
    assert len(body["pattern"]) == len(body["pattern_t"])
    rates = [v for v in body["rpm"] if v is not None]
    assert rates and abs(float(np.median(rates)) - 15.0) < 0.5
    d = body["detail"]
    assert d is not None
    assert abs(d["t_s"][0] + body["window_seconds"] / 2 - 30.0) <= body["hop"] / body["fs_hz"] + 1e-6
    assert len(d["iq"]) == body["win"]
    assert "NaN" not in r.text


def test_endpoint_refuses_a_window_that_cannot_hold_one_period(tmp_path: Path) -> None:
    p = _mtk_capture_with_a_chest(tmp_path, n=400)
    r = TestClient(app).get(
        "/api/farsense", params={"path": str(p), "t0": 0.0, "t1": 20.0, "window_seconds": 4.0}
    )
    assert r.status_code == 400
    assert "cannot hold" in r.json()["detail"]


def test_a_detail_request_reuses_the_cached_sweep(tmp_path: Path) -> None:
    """A click on the tab asks for one window in full; that must not redo
    the whole sweep. The cache is keyed on the file's identity and every
    parameter, so a changed parameter is a fresh sweep."""
    import time

    p = _mtk_capture_with_a_chest(tmp_path)
    fz.reset_cache()
    tic = time.perf_counter()
    first = fz.compute_farsense(p, 0.0, 60.0)
    full = time.perf_counter() - tic
    tic = time.perf_counter()
    again = fz.compute_farsense(p, 0.0, 60.0, detail_t=30.0)
    cached = time.perf_counter() - tic

    assert first["detail"] is None and again["detail"] is not None
    assert np.array_equal(np.nan_to_num(first["rpm"]), np.nan_to_num(again["rpm"]))
    assert cached < full / 3
    assert abs(again["detail"]["t_s"][0] + again["window_seconds"] / 2 - 30.0) <= again["hop"] / again["fs_hz"] + 1e-6

    other = fz.compute_farsense(p, 0.0, 60.0, keep_fraction=0.5)
    assert other["params"]["keep_fraction"] == 0.5


def test_positive_only_skips_a_negative_wiggle_on_the_rising_slope() -> None:
    """A trough at the half period with a small bump on the way back up: the
    literal first local maximum is that bump (negative); the first positive
    one is the period."""
    lags = np.arange(240)
    r = np.cos(2 * np.pi * lags / 80)                 # 14.7 rpm at 19.61 Hz
    r[45] += 0.05                                     # a wiggle at lag 45, where r < 0
    literal = fz.first_peak(r, FS)
    positive = fz.first_peak(r, FS, positive_only=True)
    assert literal["lag"] == 45 and literal["height"] < 0
    assert positive["lag"] == 80 and positive["height"] > 0.9
