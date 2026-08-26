"""Stage 2 (preprocessing) on one capture: artefacts + the comparison plot.

Writes an ``.npz`` holding every intermediate so the figures can be redrawn
without decoding the capture again, and a three-panel PNG: the ratio amplitude
as it arrives, then after static removal at each path's window.

The panel is the acceptance test for this stage. The fixed horizontal bands in
the raw heatmap are the link's own fading nulls, not the room; if they survive
static removal, something upstream is wrong and the later stages are being
built on it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backend.doppler import gap_limit_for, resample_uniform
from backend.preprocess import (
    BREATHING_DETREND_SECONDS,
    MOTION_DETREND_SECONDS,
    derive_sample_rate,
    normalize_subcarriers,
    remove_static,
    subcarrier_mask,
)
from backend.presence import complex_ratio
from backend.tiles import PRESENCE_METRICS, _decode_for_doppler, get_index, get_reference


def load(path: Path, t0: float, t1: float) -> dict:
    index = get_index(path)
    times_all = np.asarray(index.times, dtype=float)
    frame_ids = np.flatnonzero((times_all >= t0) & (times_all <= t1))

    reference = get_reference(path, index, path.stat().st_size, interpolate=True)
    decode = lambda metric: _decode_for_doppler(
        path, index, frame_ids, metric, reference, True
    )
    ratio = complex_ratio(decode(PRESENCE_METRICS[0]), decode(PRESENCE_METRICS[1]))
    h0_db = decode("amplitude")

    usable = np.isfinite(ratio).any(axis=1)
    times = times_all[frame_ids][: ratio.shape[0]][usable]
    return {"ratio": ratio[usable], "h0_db": h0_db[: ratio.shape[0]][usable], "times": times}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--out", type=Path, default=Path("artifacts/stage2"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = load(args.capture, args.t0, args.t1)
    rate = derive_sample_rate(raw["times"])
    print(f"capture      {args.capture.name}")
    print(f"frames       {raw['times'].size}   span {raw['times'][-1] - raw['times'][0]:.1f} s")
    print(f"rate         derived {rate['fs_hz']:.3f} Hz (mean) / {rate['median_hz']:.3f} Hz "
          f"(median), nominal {rate['nominal_hz']:g} Hz, Nyquist {rate['nyquist_hz']:.2f} Hz")
    for w in rate["warnings"]:
        print(f"  ! {w}")

    mask = subcarrier_mask(raw["ratio"], raw["h0_db"])
    print(f"subcarriers  {raw['ratio'].shape[1]} total, {mask['n_kept']} kept")
    print(f"  dead ({mask['dropped_dead'].size}): {mask['dropped_dead'].tolist()}")
    print(f"  weak ({mask['dropped_weak'].size}): {mask['dropped_weak'].tolist()}")

    fs = rate["fs_hz"]
    times = raw["times"]
    step = 1.0 / fs
    n_grid = int(np.floor((times[-1] - times[0]) / step)) + 1
    grid_times = times[0] + np.arange(n_grid) * step
    kept = raw["ratio"][:, mask["keep"]]
    gap = gap_limit_for(times)
    real, fabricated = resample_uniform(times, kept.real, grid_times, gap)
    imag, _ = resample_uniform(times, kept.imag, grid_times, gap)
    grid = real + 1j * imag
    print(f"grid         {grid.shape[0]} samples x {grid.shape[1]} subcarriers, "
          f"{100 * fabricated.mean():.2f}% fabricated")

    motion, motion_scale = normalize_subcarriers(remove_static(grid, fs, MOTION_DETREND_SECONDS))
    breathing, breathing_scale = normalize_subcarriers(
        remove_static(grid, fs, BREATHING_DETREND_SECONDS)
    )
    resid = 20 * np.log10(np.std(breathing, axis=0) + 1e-12)
    lo, hi = np.percentile(resid, [10, 90])
    print(f"normalised   per-subcarrier residual spread p10-p90 = {hi - lo:.2f} dB "
          f"(scale range {20 * np.log10(breathing_scale.max() / breathing_scale.min()):.1f} dB removed)")

    npz = args.out / f"{args.capture.stem}_stage2.npz"
    np.savez_compressed(
        npz,
        grid_times=grid_times, fabricated=fabricated,
        ratio=grid.astype(np.complex64),
        motion=motion.astype(np.complex64),
        breathing=breathing.astype(np.complex64),
        motion_scale=motion_scale, breathing_scale=breathing_scale,
        keep=mask["keep"], dropped_dead=mask["dropped_dead"],
        dropped_weak=mask["dropped_weak"], fs_hz=fs,
    )
    print(f"artefacts    {npz} ({npz.stat().st_size / 1e6:.1f} MB)")

    plot(args.out / f"{args.capture.stem}_stage2.png", grid_times, grid, motion, breathing, fs)


def plot(png: Path, t, grid, motion, breathing, fs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def db(x):
        return 20 * np.log10(np.abs(x) + 1e-12)

    panels = [
        ("raw ratio |r| (dB)", db(grid)),
        (f"static removed + normalised, motion window {MOTION_DETREND_SECONDS:g} s", db(motion)),
        (f"static removed + normalised, breathing window {BREATHING_DETREND_SECONDS:g} s", db(breathing)),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    for ax, (title, data) in zip(axes, panels):
        lo, hi = np.percentile(data, [2, 98])
        im = ax.imshow(
            data.T, aspect="auto", origin="lower", cmap="viridis", vmin=lo, vmax=hi,
            extent=[t[0], t[-1], 0, data.shape[1]],
        )
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("kept subcarrier")
        fig.colorbar(im, ax=ax, pad=0.01, label="dB")
    axes[-1].set_xlabel("time (s)")
    fig.savefig(png, dpi=110)
    print(f"figure       {png}")


if __name__ == "__main__":
    main()
