"""Tests for backend.af8 -- the A+F 8-feature presence classifier."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from backend import af8
from tests.test_hybrid2 import _capture

REPO = Path(__file__).resolve().parent.parent
CAPTURES = Path("/home/cyphy/feitcsi_parser/captures")

NAN = float("nan")
# The implementation guide's §8-1: stem -> (window end, the eight features, p(person)).
# The four inventory windows end at the reference inventory's t1 (the camera's
# span); the 10/7 vacuum was scored on [0, 61).
GUIDE = {
    "20261002_200002": (58.01800012588501, [0.139047, 0.715367, 1.070326, 0.039499, 0.588157, 0.474106, 25, 0.865930], 1.0000),
    "20261001_190002": (58.00699996948242, [0.110295, 0.788755, 1.094998, 0.164787, 0.231604, 0.079279, 0, NAN], 0.7963),
    "20261006_134123": (58.00399994850159, [0.114377, 0.796932, 1.126240, 0.041756, 0.304580, 0.102428, 2, 0.974911], 0.1896),
    "20261002_030003": (59.01599979400635, [0.017029, 0.970799, 1.023366, 0.013313, 0.221556, 0.055651, 0, NAN], 0.0015),
    "20261007_142632": (61.0, [0.016134, 0.958254, 1.002992, 0.045931, 0.380780, 0.080732, 1, 0.943273], 0.0583),
}


# --------------------------------------------------------------------------- #
#  The model                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stem", sorted(GUIDE))
def test_the_model_gives_the_guides_probability(stem: str) -> None:
    """Pure arithmetic: the guide's features in, its p(person) out, NaN imputed."""
    _, feats, p = GUIDE[stem]
    assert af8.probability(dict(zip(af8.FEATURES, feats))) == pytest.approx(p, abs=5e-5)


def test_the_model_has_the_guides_shape() -> None:
    for arr in (af8.IMPUTER_MEDIAN, af8.SCALER_MEAN, af8.SCALER_SCALE, af8.COEF):
        assert arr.shape == (len(af8.FEATURES),)
    assert (af8.SCALER_SCALE > 0).all()


# --------------------------------------------------------------------------- #
#  The features, on the guide's own captures                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stem", sorted(GUIDE))
def test_the_features_are_the_guides(stem: str) -> None:
    path = CAPTURES / f"{stem}.bin"
    if not path.is_file():
        pytest.skip(f"{stem} is not on this machine (captures are kept 7 days)")
    t1, feats, p = GUIDE[stem]
    out = af8.evaluate(path, 0.0, t1)
    assert out["subcarriers"] == 31
    for name, want in zip(af8.FEATURES, feats):
        got = out["features"][name]
        if math.isnan(want):
            assert got is None, name
        else:
            assert got == pytest.approx(want, abs=1e-6), name     # the guide prints 6 places
    assert out["p_person"] == pytest.approx(p, abs=5e-5)


# --------------------------------------------------------------------------- #
#  Where it runs                                                               #
# --------------------------------------------------------------------------- #


def test_it_judges_a_synthetic_capture(tmp_path: Path) -> None:
    out = af8.evaluate(_capture(tmp_path))
    assert out["label"] in (0, 1)
    assert 0.0 <= out["p_person"] <= 1.0
    assert out["window_s"] == [0.0, af8.WINDOW_SECONDS]


def test_it_runs_without_scipy_or_csikit(tmp_path: Path) -> None:
    p = _capture(tmp_path)
    code = (
        "import sys; sys.modules['scipy'] = None; sys.modules['CSIKit'] = None\n"
        "from backend import af8\n"
        "sys.exit(af8.main(sys.argv[1:]))\n"
    )
    res = subprocess.run([sys.executable, "-c", code, str(p), "--json"],
                         cwd=REPO, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["p_person"] == pytest.approx(af8.evaluate(p)["p_person"], abs=0)


def test_a_window_too_short_is_not_judged(tmp_path: Path) -> None:
    """Under 30 s of usable grid the guide says do not judge -- exit 2, not a 0."""
    p = _capture(tmp_path)
    with pytest.raises(ValueError, match="usable grid"):
        af8.evaluate(p, 0.0, 20.0)
    assert af8.main([str(p), "--t1", "20"]) == 2


def test_cli_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                        capsys: pytest.CaptureFixture) -> None:
    p = _capture(tmp_path)
    assert af8.main([str(p)]) == 0
    assert capsys.readouterr().out.strip() in ("0", "1")
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"\x00" * 64)
    assert af8.main([str(junk)]) == 2
    assert af8.main([str(tmp_path / "missing.bin")]) == 2
    monkeypatch.setattr(np, "__version__", "1.25.2")
    assert af8.main([str(p)]) == 3
