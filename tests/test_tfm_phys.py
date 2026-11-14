"""Tests for backend.tfm_phys (the Physics tab): seq13pw / seq15pwu and the 17-channel base under the install
calibration, with the weights of scripts/fit_tfm_phys.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend import tfm_model, tfm_phys as TP, tfm_series as TS

ROOT = Path(__file__).resolve().parents[1]
META = json.loads((ROOT / "backend" / "tfm_phys.json").read_text())
SEQ13PW = ["s025", "s2", "lgrow", "g15_res", "sT", "u_g15v", "pk_w", "bs_w", "rpm", "bs_f", "plf_res", "plf_g", "swing_ab"]


def _grid(rows: list) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)


def test_channel_sets_are_the_studys() -> None:
    m = META["models"]
    assert m["seq13pw"]["channels"] == SEQ13PW
    assert m["seq15pwu"]["channels"] == SEQ13PW + ["pk_u", "bs_u"]
    assert m["seq17"]["channels"] == list(TS.CH)
    never = {"lgrow", "swing_ab", "rpm", "bs_f", "plf_res", "plf_g"}
    for name in TP.NEW_MODELS:
        assert never.isdisjoint(m[name]["calibrated_channels"])
        assert set(m[name]["log_channels"]) == {"s025", "s2", "g15_res", "sT", "u_g15v"}
    assert TP.PHYS_CHANNELS == SEQ13PW + ["pk_u", "bs_u"]


def test_folds_never_see_the_judged_room() -> None:
    for name in TP.MODELS:
        for f, d in META["models"][name]["folds"].items():
            assert len(TP.NETS[name][f]) == 5
            assert f not in d["trained_on"]
        assert META["models"][name]["folds"]["scnC"] == {**META["models"][name]["folds"]["scnC"],
                                                         "trained_on": ["env1", "env2"], "calibrated_with": "env3-1"}
    assert all(len(k) == 5 for k in META["ref_keys"].values())


def test_reproduces_the_benchmarks_probabilities() -> None:
    for row in META["check"]:
        room = tfm_model.room_for(row["stem"])
        assert room["fold"] == row["fold"]
        S = dict(zip(TP.PHYS_CHANNELS, _grid(row["phys"])))
        for name in TP.MODELS:
            X = _grid(row["base"]) if name == "seq17" else TP.matrix(S, name)
            v = TP.judge(X, name, room)[1]
            assert v["p"] == pytest.approx(row["p"][name], abs=1e-9)
            assert v["label"] == int(v["p"] > 0.5) and len(v["p_seeds"]) == 5


def test_a_calibration_minute_is_flagged() -> None:
    ref = [r for r in META["check"] if r["reference"]]
    assert ref and TP.is_reference(ref[0]["stem"], tfm_model.room_for(ref[0]["stem"]))
    other = [r for r in META["check"] if not r["reference"] and r["fold"] != "scnC"][0]
    assert not TP.is_reference(other["stem"], tfm_model.room_for(other["stem"]))


def test_lean_breathing_lines_equal_the_deployed_ones() -> None:
    rng = np.random.default_rng(0)
    fs, n = 40.0, 2440
    t = np.arange(n) / fs
    gf = (rng.normal(size=(n, 40)) + 1j * rng.normal(size=(n, 40))) * 0.05 + np.exp(2j * np.pi * 0.25 * t)[:, None] * rng.normal(size=40)
    a, f = TP._bs_lines(gf, fs)
    ref = TS._bs_series(gf, fs)
    assert np.array_equal(np.isfinite(a), np.isfinite(ref["bs_all"]))
    assert np.allclose(a, ref["bs_all"], equal_nan=True) and np.allclose(f, ref["bs_f"], equal_nan=True)


def test_swing_share_sees_a_swing_and_not_noise() -> None:
    rng = np.random.default_rng(1)
    fs, T = 20.0, 400                                           # 20 s swing
    n = int(61 * fs)
    ph = 2 * np.pi * np.arange(n) / T
    swing = (np.sin(3 * ph) + 0.5 * np.sin(7 * ph))[:, None] * rng.normal(size=8)
    noise = rng.normal(size=(n, 8))
    assert TP.swing_alpha(swing + 0.05 * noise, T, fs, band=True) > 0.8
    assert TP.swing_alpha(noise, T, fs, band=True) < 0.1
    x = swing + 0.05 * noise
    assert np.allclose(TP.sync_average(x, T)[:T], TP.sync_average(x, T)[T:2 * T])


def test_series_reproduces_the_benchmarks_channels() -> None:
    row = next((r for r in META["check"] if r["fold"] != "scnC" and (ROOT / "captures" / f"{r['stem']}.bin").exists()), None)
    if row is None:
        pytest.skip("no check capture on this machine")
    s = TP.series(ROOT / "captures" / f"{row['stem']}.bin", 0.0, 61.0)
    X = np.array([s["S"][c] for c in TP.PHYS_CHANNELS])
    ref = _grid(row["phys"])
    assert np.array_equal(np.isfinite(X), np.isfinite(ref))
    assert np.allclose(X, ref, rtol=1e-3, equal_nan=True)       # lg's arithmetic; a laptop's BLAS differs ~1e-5


# ------------------------------------------------------------------ the board-speed series (backend.tfm_phys_fast)
from backend import af8, af8_env1 as E, breath_components as BC, farsense, mtk, tfm_phys_fast as TF  # noqa: E402


def _cap():
    row = next((r for r in META["check"] if r["fold"] != "scnC" and (ROOT / "captures" / f"{r['stem']}.bin").exists()), None)
    if row is None:
        pytest.skip("no check capture on this machine")
    return ROOT / "captures" / f"{row['stem']}.bin"


def test_fast_pieces_equal_the_originals() -> None:
    rng = np.random.default_rng(3)
    fs, n = 41.6667, 2500
    t = np.arange(n) / fs
    g = (1.0 + 0.02 * rng.normal(size=(n, 31)) + 0.02j * rng.normal(size=(n, 31))
         + 0.05 * np.exp(2j * np.pi * 0.25 * t)[:, None] * rng.normal(size=31))
    for k in (1, 10, 42, 250):                                   # running-sum moving average = np.convolve "same"
        ref = np.apply_along_axis(lambda c: np.convolve(c, np.ones(k) / k, "same"), 0, g)
        assert np.allclose(TF._ma(g, k), ref, rtol=0, atol=1e-12)
    A = np.abs(g)
    for L in (10, 83, 830):
        assert np.array_equal(TF._step(g, A, L), af8._step(g, L))
    assert TF.swing_period(g, A, fs) == af8.swing_period(g, fs)
    T = af8.swing_period(g, fs)
    for band in (False, True):
        assert TF.swing_alpha(g, T, fs, band) == pytest.approx(TP.swing_alpha(g, T, fs, band), abs=1e-12)
    z0, V0, _ = TP._fast_subspace(g, fs)
    z1, V1, _ = TF.fast_subspace(g, fs)
    assert np.allclose(np.abs(V0.conj().T @ V1), np.eye(V0.shape[1]), atol=1e-9)
    fab = np.zeros(n, bool); fab[1200:1230] = True
    for rpm in (True, False):
        a, b = TF.farsense_pk(g, fab, fs, rpm)
        ra, rb = TS._fs_windows(g, fab, fs, 0)
        assert np.allclose(a, ra, rtol=1e-9, equal_nan=True)
        assert b is None if not rpm else np.allclose(b, rb, rtol=1e-9, equal_nan=True)
    gf = (1.0 + 0.02 * rng.normal(size=(n, 245)) + 0.02j * rng.normal(size=(n, 245))
          + 0.05 * np.exp(2j * np.pi * 0.25 * t)[:, None] * rng.normal(size=245))
    ra, rf = TP._bs_lines(gf, fs)                                # reads the full grid's every 8th column
    for gram in ("stretch", "direct"):
        a, f = TF.bs_lines(gf[:, ::af8.STRIDE], fs, gram)
        assert np.allclose(a, ra, rtol=1e-9, equal_nan=True) and np.array_equal(f, rf, equal_nan=True)
    Y = rng.normal(size=(833, 8)).cumsum(0)
    v, fr = TF._prominences(Y, fs)
    ref = [BC.prominence(Y[:, j], fs) for j in range(8)]
    assert np.allclose(v, [p[0] for p in ref], rtol=1e-9) and np.array_equal(fr, [p[1] for p in ref])
    pat = rng.normal(size=(417, 31)); bnr = rng.random(31)
    r0, s0 = farsense.combine_autocorrelations(pat, bnr, keep_fraction=TF.FS_KEEP)
    r1, s1 = TF.combine_autocorrelations(pat, bnr)
    assert np.array_equal(s0, s1) and np.allclose(r0, r1, rtol=0, atol=1e-12 * np.abs(r0).max())


def test_fast_decode_and_gating_equal_the_originals() -> None:
    p = _cap()
    idx = mtk.MTKIndex(p)
    G = TF.load_grid(p, idx, 0.0, 61.0)
    H, tt, live, nb = E.load_chains(p, idx, 0.0, 61.0)
    assert np.array_equal(G["tt"], tt) and np.array_equal(G["live"], live)
    assert np.array_equal(G["h0"], H[:, 0, ::af8.STRIDE]) and np.array_equal(G["h1"], H[:, 1, ::af8.STRIDE])
    ref = E.direct_path_gated(H, live, nb)[:, ::af8.STRIDE]
    assert np.allclose(TF.gated8(G["full1"], G["live"]), ref, rtol=0, atol=1e-9 * np.abs(ref).max())


def test_fast_series_equals_the_reference() -> None:
    p = _cap()
    idx = mtk.MTKIndex(p)
    s0, s1 = TP.series(p, 0.0, 61.0, idx), TF.series(p, 0.0, 61.0, idx)
    assert s0["T"] == s1["T"] and s0["w"] == pytest.approx(s1["w"], abs=1e-12)
    for c in TP.PHYS_CHANNELS:
        assert np.allclose(s0["S"][c], s1["S"][c], rtol=1e-9, atol=0, equal_nan=True), c
    room = tfm_model.room_for(p.stem)
    for name in TP.NEW_MODELS:
        assert TP.judge(TP.matrix(s1["S"], name), name, room)[1]["p"] == pytest.approx(
            TP.judge(TP.matrix(s0["S"], name), name, room)[1]["p"], abs=1e-12)
