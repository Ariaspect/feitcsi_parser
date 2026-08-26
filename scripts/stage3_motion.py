"""Stage 3 (motion) from the stage-2 artefacts: figures + threshold calibration.

Reads the ``.npz`` stage 2 wrote rather than decoding the capture again, and
appends its own series to a stage-3 ``.npz`` so the figures stay redrawable.

The motion threshold is not a constant. It is calibrated from a stretch the
operator declares empty -- mean + 3 sigma of the score over that stretch --
which is the only way a score that means "how correlated is this window" gets
a floor that belongs to this room and this radio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backend.motion import (
    FIDGET_BAND_HZ,
    MOTION_HOP_SECONDS,
    MOTION_WINDOW_SECONDS,
    eigenvalue_ratio,
    fidget_energy,
    lag1_correlation,
    motion_score,
)

# The room settles after the occupant leaves: measured on
# captures/lg/20260825_185637.bin the channel offset runs 0.44-0.91 dB over
# 390-500 s and 0.14-0.29 dB over 510-600 s. Calibrating across the whole
# post-exit stretch folds that settling into the "empty" noise floor and
# raises every threshold derived from it.
EMPTY_RANGE_S = (520.0, 600.0)

# Better still, calibrate from a capture recorded when the room was empty for
# its whole length. The tail of a measurement capture is a short sample of a
# room that has not finished settling: 80 s of
# captures/lg/20260825_185637.bin gives mean 0.198 with sd 0.153, while 600 s
# of captures/lg/20260826_0400-0500.bin (nobody home, 04:00) gives mean 0.179
# with sd 0.075 -- the same level, half the spread, and a threshold of 0.403
# rather than 0.657. Pass --empty-npz to use one.


def calibrate(centres: np.ndarray, values: np.ndarray, empty: tuple[float, float]) -> dict:
    lo, hi = empty
    inside = (centres >= lo) & (centres <= hi)
    if not inside.any():
        raise ValueError(f"no window falls inside the empty range {empty}")
    # Blanked windows are excluded: a bridged hole reads as motion, so leaving
    # them in raises the very floor they are being measured against.
    quiet = values[inside]
    quiet = quiet[np.isfinite(quiet)]
    if quiet.size == 0:
        raise ValueError(f"every window inside {empty} is blanked")
    return {
        "mean": float(np.mean(quiet)),
        "std": float(np.std(quiet)),
        "threshold": float(np.mean(quiet) + 3.0 * np.std(quiet)),
        "n": int(quiet.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage2_npz", type=Path)
    ap.add_argument("--empty", type=float, nargs=2, default=EMPTY_RANGE_S,
                    help="known-empty range within this capture, used when --empty-npz is absent")
    ap.add_argument("--empty-npz", type=Path, default=None,
                    help="stage-2 artefacts of a capture that was empty throughout")
    ap.add_argument("--out", type=Path, default=Path("artifacts/stage3"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.stage2_npz)
    fs = float(d["fs_hz"])
    t0 = float(d["grid_times"][0])
    sig = d["motion"].astype(np.complex128)
    print(f"input        {args.stage2_npz.name}  {sig.shape[0]} x {sig.shape[1]} at {fs:.3f} Hz")
    print(f"windows      {MOTION_WINDOW_SECONDS:g} s, hop {MOTION_HOP_SECONDS:g} s")

    fabricated = d["fabricated"]
    centres, rho = lag1_correlation(sig, fs, fabricated=fabricated)
    centres = centres + t0
    score = motion_score(rho)
    _, eig = eigenvalue_ratio(sig, fs, fabricated=fabricated)
    _, fidget, band = fidget_energy(sig, fs, fabricated=fabricated)
    blanked = np.isnan(score)
    print(f"dropouts     {100 * fabricated.mean():.2f}% of samples fabricated, "
          f"{blanked.sum()} of {blanked.size} windows blanked")
    print(f"fidget band  {FIDGET_BAND_HZ} requested -> {band[0]:.2f}-{band[1]:.2f} Hz used")

    if args.empty_npz is not None:
        ref = np.load(args.empty_npz)
        ref_fs = float(ref["fs_hz"])
        ref_t0 = float(ref["grid_times"][0])
        ref_sig = ref["motion"].astype(np.complex128)
        ref_centres, ref_rho = lag1_correlation(ref_sig, ref_fs, fabricated=ref["fabricated"])
        ref_centres = ref_centres + ref_t0
        whole = (ref_centres[0], ref_centres[-1])
        cal = calibrate(ref_centres, motion_score(ref_rho), whole)
        # The fidget floor stays in-capture. The motion score is a
        # correlation and carries across captures unchanged; the fidget
        # reading is an absolute power and does not. Measured on two
        # stretches that were both empty, the night reference reads 50.8 and
        # this capture's settled tail reads 14.8 -- a 3.4x disagreement about
        # what an empty room costs, which would put the threshold above
        # anything this capture ever reaches. Until that is understood the
        # fidget series is diagnostic, not a verdict.
        fid_cal = calibrate(centres, fidget, tuple(args.empty))
        print(f"calibration  motion from {args.empty_npz.name}, whole "
              f"{whole[0]:.0f}-{whole[1]:.0f} s, {cal['n']} windows")
        print(f"             fidget in-capture {args.empty[0]:g}-{args.empty[1]:g} s "
              "(absolute, does not transfer between captures)")
    else:
        cal = calibrate(centres, score, tuple(args.empty))
        fid_cal = calibrate(centres, fidget, tuple(args.empty))
        print(f"calibration  in-capture {args.empty[0]:g}-{args.empty[1]:g} s, {cal['n']} windows")
    print(f"  motion     mean {cal['mean']:.4f}  sd {cal['std']:.4f}  -> threshold {cal['threshold']:.4f}")
    print(f"  fidget     mean {fid_cal['mean']:.3f}  sd {fid_cal['std']:.3f}  -> threshold {fid_cal['threshold']:.3f}")

    occupied = (centres <= 378)
    empty = (centres >= args.empty[0])
    print(f"  occupied p50 {np.nanmedian(score[occupied]):.4f}   "
          f"empty p50 {np.nanmedian(score[empty]):.4f}")
    print(f"  above threshold: occupied {100 * np.nanmean(score[occupied] > cal['threshold']):.1f}%, "
          f"empty {100 * np.nanmean(score[empty] > cal['threshold']):.1f}%")
    settle = (centres > 383) & (centres < EMPTY_RANGE_S[0])
    print(f"  post-exit {383:g}-{args.empty[0]:g} s p50 {np.nanmedian(score[settle]):.4f}, "
          f"above threshold {100 * np.nanmean(score[settle] > cal['threshold']):.1f}%")

    npz = args.out / f"{args.stage2_npz.stem.replace('_stage2', '')}_stage3.npz"
    np.savez_compressed(
        npz, centres=centres, rho=rho.astype(np.float32), score=score,
        eigenvalue_ratio=eig, fidget=fidget, fidget_band=np.array(band),
        motion_threshold=cal["threshold"], fidget_threshold=fid_cal["threshold"],
        empty_range=np.array(args.empty),
    )
    print(f"artefacts    {npz} ({npz.stat().st_size / 1e6:.1f} MB)")
    plot(args.out / f"{npz.stem}.png", centres, rho, score, eig, fidget, cal, fid_cal, args.empty)


def plot(png, centres, rho, score, eig, fidget, cal, fid_cal, empty) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(centres, score, lw=0.7, color="#d97a2f")
    ax.axhline(cal["threshold"], color="#d62728", lw=1.2, ls="--",
               label=f"empty mean+3sd = {cal['threshold']:.3f}")
    ax.axvspan(empty[0], empty[1], color="#2f9c6f", alpha=0.15, label="calibration (empty)")
    ax.set_ylabel("motion score\n(median lag-1 rho)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("M(t): how much of each window is a trajectory rather than noise",
                 fontsize=10, loc="left")

    ax = axes[1]
    lo, hi = np.nanpercentile(rho, [2, 98])
    im = ax.imshow(rho, aspect="auto", origin="lower", cmap="magma", vmin=lo, vmax=hi,
                   extent=[centres[0], centres[-1], 0, rho.shape[0]])
    ax.set_ylabel("kept subcarrier")
    ax.set_title("rho[k, t]: which subcarriers decorrelated, i.e. at which delays",
                 fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, pad=0.01)

    ax = axes[2]
    ax.plot(centres, eig, lw=0.7, color="#2f6fed")
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$\lambda_1 / \sum \lambda_i$")
    ax.set_title("Covariance concentration: near 1 = one shared waveform (a body), "
                 "spread = per-subcarrier directions (a chest) or noise",
                 fontsize=10, loc="left")

    ax = axes[3]
    ax.semilogy(centres, fidget, lw=0.7, color="#7b5ea7")
    ax.axhline(fid_cal["threshold"], color="#d62728", lw=1.2, ls="--",
               label=f"empty mean+3sd = {fid_cal['threshold']:.2f}")
    ax.set_ylabel("fidget power\n(noise floors)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Fidget band, absolute in units of the noise floor", fontsize=10, loc="left")

    fig.savefig(png, dpi=110)
    print(f"figure       {png}")


if __name__ == "__main__":
    main()
