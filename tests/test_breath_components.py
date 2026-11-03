"""Tests for backend.breath_components -- the object-agnostic breathing columns (bs_*)."""

from __future__ import annotations

import numpy as np

from backend import breath_components as BC

FS = 40.0


def _grid(person: bool, mover: bool, seed: int = 0, n_sc: int = 64, seconds: float = 60.0) -> np.ndarray:
    """Ratio-like grid: unit level, noise, an optional strong mover pattern and a weak breathing pattern."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    g = 1.0 + 0.01 * (rng.standard_normal((t.size, n_sc)) + 1j * rng.standard_normal((t.size, n_sc)))
    if mover:            # a rotating mover: a 1.3 Hz blade line on its own spatial pattern, 20x the breathing
        pat = np.exp(1j * rng.uniform(0, 2 * np.pi, n_sc))
        g += 0.2 * np.sin(2 * np.pi * 1.3 * t)[:, None] * pat[None, :]
    if person:           # breathing at 15 rpm on another pattern
        pat = np.exp(1j * rng.uniform(0, 2 * np.pi, n_sc))
        g += 0.01 * np.sin(2 * np.pi * 0.25 * t)[:, None] * pat[None, :]
    return g


def test_a_breathing_line_shows_beside_a_strong_mover_and_not_without_a_person() -> None:
    empty = BC.from_grid(_grid(False, False), FS)
    person = BC.from_grid(_grid(True, False), FS)
    mover = BC.from_grid(_grid(False, True), FS)
    both = BC.from_grid(_grid(True, True), FS)
    assert set(empty) == set(BC.KEYS)
    assert person["bs_all_p50"] > empty["bs_all_p50"] + 2
    # the mover takes the leading component; the person's line is still there without it
    assert both["bs_r1_p50"] > mover["bs_r1_p50"] + 2
    # alone, the person is the leading component: removing it drops the line
    assert person["bs_all_p50"] > person["bs_r1_p50"]
    # one breathing rate holds over the windows
    assert person["bs_rate_iqr"] < empty["bs_rate_iqr"] or person["bs_rate_iqr"] <= 0.01


def test_the_eigh_components_match_an_svd() -> None:
    g = _grid(True, True, seed=3)
    W = int(round(BC.WINDOW_SECONDS * FS))
    x = g[:W]
    X = np.c_[x.real - x.real.mean(axis=0), x.imag - x.imag.mean(axis=0)]
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    _, V = np.linalg.eigh(X.T @ X)
    comps = X @ V[:, ::-1][:, :BC.N_COMPONENTS]
    for j in range(BC.N_COMPONENTS):          # the same series up to sign: the same line
        assert abs(BC.prominence(comps[:, j], FS)[0] - BC.prominence(u[:, j] * s[j], FS)[0]) < 1e-6


def test_under_one_window_is_blank() -> None:
    out = BC.from_grid(_grid(True, False, seconds=10.0), FS)
    assert all(np.isnan(out[k]) for k in BC.KEYS)


def test_an_unreadable_capture_is_blank_not_an_error(tmp_path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"\0" * 64)
    out = BC.unit_features(p, 0.0, 60.0)
    assert set(out) == set(BC.KEYS) and all(np.isnan(v) for v in out.values())
