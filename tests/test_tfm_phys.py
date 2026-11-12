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
