"""Tests for backend.hybrid2_calc -- the board's numpy-only one-minute verdict."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import savgol_filter

from backend import farsense, hybrid2, hybrid2_calc
from backend.tiles import get_index
from tests.test_hybrid2 import _capture

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
#  The numpy Savitzky-Golay that replaced scipy's                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n, window, order", [
    (2580, 43, 3),   # 1 s at the 43 Hz the captures run at now
    (1200, 21, 3),   # 1 s at the default 20 Hz
    (43, 43, 3),     # window equal to the series
    (6, 5, 2),
])
def test_savgol_is_scipys_interp_mode(n: int, window: int, order: int) -> None:
    x = np.random.default_rng(n).standard_normal((n, 7)).cumsum(axis=0)
    ours = farsense.savgol(x, window, order)
    theirs = savgol_filter(x, window, order, axis=0, mode="interp")
    assert np.max(np.abs(ours - theirs)) <= 1e-12 * np.max(np.abs(theirs))


def test_the_band_dft_cache_changes_nothing() -> None:
    """The cached in-band DFT is the matrix the expression builds, so a sweep
    with it is the sweep without it, bit for bit."""
    seg = np.random.default_rng(3).standard_normal((476, 12, 2)) @ np.array([1.0, 1j])
    farsense._DFT_CACHE.clear()
    cold = farsense.extract_patterns(seg, 47.6)
    warm = farsense.extract_patterns(seg, 47.6)
    assert farsense._DFT_CACHE                      # it did cache
    for key in ("pattern", "bnr", "theta", "bnr_all"):
        assert np.array_equal(cold[key], warm[key], equal_nan=True), key


def test_savgol_refuses_what_scipy_would() -> None:
    with pytest.raises(ValueError):
        farsense.savgol(np.zeros((10, 2)), 4, 2)      # even window
    with pytest.raises(ValueError):
        farsense.savgol(np.zeros((10, 2)), 11, 3)     # longer than the series


# --------------------------------------------------------------------------- #
#  The calculator is the server's verdict                                      #
# --------------------------------------------------------------------------- #


def test_block_size_is_the_servers() -> None:
    """The null fill is decided per block, so the boundaries are part of the
    answer: a different block size would decode a different ratio."""
    from backend.tiles import BLOCK_SIZE
    assert hybrid2_calc.BLOCK_SIZE == BLOCK_SIZE


def test_it_returns_the_servers_range_verdict(tmp_path: Path) -> None:
    p = _capture(tmp_path)
    hybrid2.reset_cache()
    t = np.asarray(get_index(p).times, float)
    server = hybrid2.compute_hybrid2(p, float(t[0]), float(t[-1]), mimo=(2, 1),
                                     lag_seconds=hybrid2.RANGE_LAG_SECONDS)["range_verdict"]
    ours = hybrid2_calc.evaluate(p)
    assert ours["present"] == server["present"]
    assert ours["label"] == int(server["present"])
    assert ours["by"] == server["by"]
    assert ours["breath_run"] == server["breath_run"]
    assert ours["seconds"] == server["seconds"]
    assert ours["motion_p90"] == server["motion_p90"]          # bit for bit


def test_it_runs_without_scipy_or_csikit(tmp_path: Path) -> None:
    """The board has neither. Blocking both in a fresh interpreter is the
    closest this machine gets to it: any import that reaches for one fails."""
    p = _capture(tmp_path)
    code = (
        "import sys; sys.modules['scipy'] = None; sys.modules['CSIKit'] = None\n"
        "from backend import hybrid2_calc\n"
        "sys.exit(hybrid2_calc.main(sys.argv[1:]))\n"
    )
    res = subprocess.run([sys.executable, "-c", code, str(p), "--json"],
                         cwd=REPO, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["label"] in (0, 1)
    assert out["label"] == hybrid2_calc.evaluate(p)["label"]


# --------------------------------------------------------------------------- #
#  The command line                                                            #
# --------------------------------------------------------------------------- #


def test_cli_prints_the_label_alone(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    p = _capture(tmp_path)
    assert hybrid2_calc.main([str(p)]) == 0
    assert capsys.readouterr().out.strip() in ("0", "1")


def test_cli_refuses_a_file_it_cannot_score(tmp_path: Path) -> None:
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"\x00" * 64)
    assert hybrid2_calc.main([str(junk)]) == 2
    assert hybrid2_calc.main([str(tmp_path / "missing.bin")]) == 2


def test_cli_refuses_a_numpy_older_than_the_boards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """1.26 (the board's) and 2.x are checked; anything older must stop, not
    answer on a library the calculator was never run against."""
    monkeypatch.setattr(np, "__version__", "1.25.2")
    assert hybrid2_calc.main([str(_capture(tmp_path, n=400))]) == 3
