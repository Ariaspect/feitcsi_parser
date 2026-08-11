"""Heatmap plotting helpers for FeitCSI amplitude and phase.

Adapted from nexmon_csi_parser/streamlit_app.py for FeitCSI captures.
Subcarriers arrive fftshifted from parser.py (signed-frequency order).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .parser import FeitCSICapture


def _finite_bounds(values: np.ndarray, fallback: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    vmin = float(np.percentile(finite, 5))
    vmax = float(np.percentile(finite, 95))
    if np.isclose(vmin, vmax):
        vmin = float(finite.min())
        vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0
    return vmin, vmax


def _subcarrier_axis(num_subcarriers: int) -> np.ndarray:
    """Return signed-frequency bin labels centered at 0.

    For N subcarriers in fftshifted order, bins run -N//2 .. +N//2-1.
    """
    n = num_subcarriers
    return np.arange(-n // 2, n // 2, dtype=int)


def plot_heatmap(
    capture: FeitCSICapture,
    *,
    metric: str = "amplitude",
    title: str | None = None,
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    interpolation: str = "linear",
) -> plt.Figure:
    """Plot a single heatmap of `metric` (amplitude or phase).

    Parameters
    ----------
    capture : FeitCSICapture
    metric : {"amplitude", "phase"}
    title : str | None
    cmap_name : str
        Matplotlib colormap name.
    vmin, vmax : float | None
        Color bounds. If None, auto-computed from finite values.
    interpolation : {"linear", "none"}
        Per-subcarrier 1D linear interpolation across NaN gaps in time.
    """
    data = capture.amplitude if metric == "amplitude" else capture.phase
    t = capture.time_seconds
    n_sc = capture.num_subcarriers

    height = min(8.0, max(3.0, n_sc / 20.0 if n_sc else 3.0))
    fig, ax = plt.subplots(figsize=(14, height), constrained_layout=True)

    if data.size == 0 or t.size == 0:
        ax.set_title(title or f"FeitCSI {metric}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Subcarrier bin")
        return fig

    # Time-axis bucketing: average amplitude within uniform dt cells.
    use_time = np.isfinite(t).any() and t.size > 1
    if use_time:
        diffs = np.diff(t)
        positive = diffs[diffs > 0]
        dt = float(np.median(positive)) if positive.size else 1.0
        if dt <= 0:
            dt = 1.0
        t_min = float(np.nanmin(t))
        t_max = float(np.nanmax(t))
        n_cells = int(np.ceil((t_max - t_min) / dt)) + 1
        cell_idx = np.clip(((t - t_min) / dt).astype(int), 0, max(0, n_cells - 1))

        row_has = np.isfinite(data).any(axis=1)
        cells = cell_idx[row_has]
        rows = data[row_has]

        grid_sum = np.zeros((n_cells, n_sc), dtype=np.float64)
        grid_count = np.zeros(n_cells, dtype=np.int32)
        np.add.at(grid_sum, cells, np.nan_to_num(rows))
        np.add.at(grid_count, cells, 1)

        plot_data = np.full((n_cells, n_sc), np.nan, dtype=float)
        has = grid_count > 0
        plot_data[has] = grid_sum[has] / grid_count[has, None]
        plot_y = t_min + (np.arange(n_cells) + 0.5) * dt

        if interpolation == "linear":
            plot_data = _fill_nan_grid(plot_data)
    else:
        plot_data = data
        plot_y = np.arange(t.size, dtype=float)

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = _finite_bounds(plot_data)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=cmap(0.0))

    sc_bins = _subcarrier_axis(n_sc)
    mesh = ax.pcolormesh(
        plot_y,
        sc_bins,
        plot_data.T,
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    label = "Amplitude (dBm)" if metric == "amplitude" else "Phase (rad)"
    ax.set_title(title or f"FeitCSI {metric} ({data.shape[0]} packets)")
    ax.set_xlabel("Time (s)" if use_time else "Packet index")
    ax.set_ylabel("Subcarrier bin")
    fig.colorbar(mesh, ax=ax, label=label)
    return fig


def _fill_nan_grid(grid: np.ndarray) -> np.ndarray:
    """Per-column 1D linear interpolation of NaN gaps."""
    if grid.size == 0:
        return grid
    nan_mask = np.isnan(grid)
    if not nan_mask.any():
        return grid
    filled = grid.copy()
    for col in range(filled.shape[1]):
        col_data = filled[:, col]
        valid = ~np.isnan(col_data)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        if n_valid == 1:
            filled[:, col] = col_data[valid][0]
            continue
        filled[:, col] = np.interp(
            np.arange(filled.shape[0]),
            np.where(valid)[0],
            col_data[valid],
        )
    return filled
