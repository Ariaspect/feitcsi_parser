"""Tests for the e2e deployment (user 2026-11-18): backend.e2e_np (numpy inference), backend.stream_e2e (the
streaming window, calibration and judgment) and the bundle backend/e2e_deploy (experiments/e2e/deploy_bundle.py)."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from backend import e2e_np as N, mtk, stream_e2e as SE

CAP = SE.BUNDLE.parents[1] / "captures" / "20261013_120426.bin"       # an env1 minute; skipped without captures/
needs_cap = pytest.mark.skipif(not CAP.exists(), reason="needs captures/20261013_120426.bin")


def test_bundle_folds_never_see_the_judged_room() -> None:
    folds = SE.META["folds"]
    assert set(folds) == {"env1", "env2", "env3", "env3-1", "scnC"}
    for fold, rooms in folds.items():
        if fold != "scnC":
            assert fold not in rooms and len(rooms) == 3
        for s in SE.META["seeds"]:
            assert (SE.BUNDLE / f"{fold}__s{s}.npz").exists()
    assert folds["scnC"] == ["env1", "env2"]
    assert SE.META["model"] == "sbo.B.lfhftl.st16.att.noaug" and SE.META["seeds"] == [0, 1, 2]


def test_bundle_reproduces_the_evaluated_verdicts() -> None:
    c = json.loads((SE.BUNDLE / "check.json").read_text())
    assert c["flips"] == [] and c["n"] > 1400
    for row in c["table"].values():
        assert row["numpy"] == row["torch"]


def test_used_bins_are_two_groups_of_fifteen() -> None:
    assert len(SE.NEED) == 30 and list(SE.NEED[:3]) == [0, 8, 16] and int(SE.NEED[-1]) == 232
    assert SE.OFFS == [0, 8] and SE.STARTS == [0, 1, 2]


def test_scale_is_the_training_formula() -> None:
    rng = np.random.default_rng(0)
    minutes = [(0.01 * rng.standard_normal((SE.T, N.N_BINS, 2))).astype(np.float16) for _ in range(5)]
    s = SE.scale_from(minutes)
    ref = (np.median(np.stack([N.noise_level(m) for m in minutes]), axis=0) / SE.META["glob"]).astype(np.float32)
    assert np.array_equal(s[SE.NEED], ref[SE.NEED])
    assert (np.delete(s, SE.NEED) == 1).all()


def test_judgment_is_finite_for_every_fold() -> None:
    rng = np.random.default_rng(1)
    arr = (0.01 * rng.standard_normal((SE.T, N.N_BINS, 2))).astype(np.float16)
    sc = np.asarray(SE.META["room_scale"]["env2"], np.float32)
    for fold in SE.META["folds"]:
        assert math.isfinite(SE.Model(fold).logit(arr, sc))


@needs_cap
def test_stream_window_equals_the_one_minute_file() -> None:
    idx = mtk.MTKIndex(CAP)
    a = N.prep_fast(str(CAP), 0.0, 61.0, SE.NEED, idx)
    assert np.array_equal(a.view(np.uint16), N.prep_cols(str(CAP), 0.0, 61.0, SE.NEED, idx).view(np.uint16))
    ring = SE.Ring(keep_s=3600)
    ring.ingest(CAP, idx, np.arange(idx.count))
    W = ring.frames(int(np.asarray(idx._stamps)[0]) + 61000)
    assert W is not None
    assert np.array_equal(a.view(np.uint16), SE.prep_frames(W["D"], W["tt"]).view(np.uint16))


@needs_cap
def test_fast_index_is_mtkindex() -> None:
    a, b = vars(mtk.MTKIndex(CAP)), vars(N.fast_index(CAP))
    assert set(a) == set(b)
    for k, x in a.items():
        if isinstance(x, np.ndarray):
            assert x.dtype == b[k].dtype and x.tobytes() == b[k].tobytes(), k
        elif k != "path":
            assert x == b[k], k


@needs_cap
def test_one_core_judge_par_is_judge_fast() -> None:
    keep = os.sched_getaffinity(0)
    try:
        m = SE.Model("env1")
        sc = np.asarray(SE.META["room_scale"]["env1"], np.float32)
        arr = N.prep_fast(str(CAP), 0.0, 61.0, SE.NEED)
        l0 = N.judge_fast(arr, sc, SE.CFG, m.nets, SE.STARTS, SE.OFFS)
        l1 = N.judge_par(str(CAP), 0.0, 61.0, sc, SE.CFG, m.nets, SE.STARTS, SE.OFFS, cores=(min(keep),))
        assert l1 == l0
    finally:
        os.sched_setaffinity(0, keep)
