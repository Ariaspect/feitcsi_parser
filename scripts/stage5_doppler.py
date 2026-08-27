"""Stage 5 (Doppler) from the stage-2 artefacts: two scales, one figure each.

Motion and breathing get separate configurations because one cannot serve
both: 0.033 Hz resolution needs a 30 s window, and 30 s of a walk-through is
a smear. Both run on the complex ratio so the Doppler keeps its sign.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backend.spectro import (
    doppler_sign_bias,
    find_sidebands,
    stft_complex,
    stft_config,
    velocity_of,
)

REGIMES = [(30.0, 380.0, "occupied still"), (378.0, 392.0, "the walk-out"),
           (392.0, 505.0, "post-exit"), (525.0, 595.0, "settled empty")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage2_npz", type=Path)
    ap.add_argument("--out", type=Path, default=Path("artifacts/stage5"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.stage2_npz)
    fs = float(d["fs_hz"])
    t0 = float(d["grid_times"][0])
    stem = args.stage2_npz.stem.replace("_stage2", "")
    print(f"input        {args.stage2_npz.name} at {fs:.3f} Hz")

    results = {}
    for purpose, key in (("motion", "motion"), ("breathing", "breathing")):
        cfg = stft_config(fs, purpose)
        print(f"\n{purpose}: window {cfg['window_seconds']:.2f} s, hop {cfg['hop_seconds']:.2f} s, "
              f"resolution {cfg['resolution_hz']:.3f} Hz, bin {cfg['bin_hz']:.4f} Hz, "
              f"axis {cfg['display_hz'][0]:+.2f}..{cfg['display_hz'][1]:+.2f} Hz "
              f"({velocity_of(cfg['display_hz'][1]):.2f} m/s)")
        for w in cfg["warnings"]:
            print(f"  ! {w}")
        spec, freqs, times = stft_complex(
            d[key].astype(np.complex128), fs, cfg["win"], cfg["hop"],
            zero_pad=cfg["zero_pad"], fabricated=d["fabricated"],
        )
        results[purpose] = (cfg, spec, freqs, times + t0)

    cfg, spec, freqs, times = results["breathing"]
    print("\nsidebands, per regime (symmetric +/-f pair inside 0.1-0.6 Hz)")
    for lo, hi, label in REGIMES:
        m = (times >= lo) & (times <= hi)
        if not m.any():
            continue
        found = find_sidebands(np.nanmean(spec[:, m], axis=1), freqs)
        verdict = "FOUND" if found["found"] else "  -  "
        print(f"  {label:16s} {verdict}  {found['rpm']:5.2f} rpm  "
              f"prominence {found['prominence']:5.2f}  symmetry {found['symmetry']:.2f}"
              + (f"  harmonics rejected {found['harmonics_rejected']}"
                 if found["harmonics_rejected"] else ""))

    mcfg, mspec, mfreqs, mtimes = results["motion"]
    bias = doppler_sign_bias(mspec, mfreqs)
    print("\ndoppler sign bias (P+ - P-)/(P+ + P-), motion scale")
    for lo, hi, label in REGIMES:
        m = (mtimes >= lo) & (mtimes <= hi)
        if m.any():
            print(f"  {label:16s} {np.nanmedian(bias[m]):+.4f}   "
                  f"p10 {np.nanpercentile(bias[m], 10):+.3f}  "
                  f"p90 {np.nanpercentile(bias[m], 90):+.3f}")
    print("  note: this capture holds no entry event -- the occupant was already "
          "present at t=0 -- so the opposite-sign half of the check needs another file")

    npz = args.out / f"{stem}_stage5.npz"
    np.savez_compressed(
        npz,
        motion_spec=results["motion"][1].astype(np.float32), motion_freqs=results["motion"][2],
        motion_times=results["motion"][3], sign_bias=bias,
        breathing_spec=spec.astype(np.float32), breathing_freqs=freqs, breathing_times=times,
    )
    print(f"\nartefacts    {npz} ({npz.stat().st_size / 1e6:.1f} MB)")
    plot(args.out / f"{stem}_stage5.png", results, bias)


def plot(png, results, bias) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)

    for ax, purpose in zip(axes[:2], ("motion", "breathing")):
        cfg, spec, freqs, times = results[purpose]
        lo, hi = cfg["display_hz"]
        keep = (freqs >= lo) & (freqs <= hi)
        data = 10 * np.log10(spec[keep] + 1e-12)
        vlo, vhi = np.nanpercentile(data, [5, 99.5])
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="magma",
                       vmin=vlo, vmax=vhi,
                       extent=[times[0], times[-1], freqs[keep][0], freqs[keep][-1]])
        ax.set_ylabel(f"{purpose} Doppler (Hz)")
        ax.axhline(0, color="w", lw=0.4, alpha=0.4)
        ax.set_title(
            f"{purpose}: {cfg['window_seconds']:.2f} s window, "
            f"{cfg['resolution_hz']:.3f} Hz resolution, Blackman-Harris, signed",
            fontsize=10, loc="left")
        fig.colorbar(im, ax=ax, pad=0.01, label="dB")

    ax = axes[2]
    mtimes = results["motion"][3]
    ax.plot(mtimes, bias, lw=0.6, color="#2f6fed")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("sign bias\n(P+ - P-)/(P+ + P-)")
    ax.set_xlabel("time (s)")
    ax.set_xlim(mtimes[0], mtimes[-1])
    ax.set_title("Approaching (+) against receding (-). Respiration sits near zero: "
                 "its sidebands are symmetric.", fontsize=10, loc="left")

    fig.savefig(png, dpi=110)
    print(f"figure       {png}")


if __name__ == "__main__":
    main()
