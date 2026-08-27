"""Stage 4 (breathing) from the stage-2 and stage-3 artefacts.

Reads the detrended, normalised signal stage 2 wrote at the *breathing*
window, and the motion score stage 3 wrote, so nothing is decoded twice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backend.breathing import BREATHING_HOP_SECONDS, BREATHING_WINDOW_SECONDS, breathing_windows

# Where the occupant was reported sitting still, used for the sanity check
# rather than for any threshold.
STILL_RANGE_S = (30.0, 380.0)
SANE_RPM = (12.0, 20.0)


def motion_gate_for(
    breath_centres: np.ndarray, win_s: float, motion_centres: np.ndarray,
    score: np.ndarray, threshold: float, fraction: float,
) -> np.ndarray:
    """One flag per breathing window: was this window driven by gross motion."""
    gate = np.zeros(breath_centres.size, dtype=bool)
    for i, c in enumerate(breath_centres):
        inside = (motion_centres >= c - win_s / 2) & (motion_centres <= c + win_s / 2)
        vals = score[inside]
        vals = vals[np.isfinite(vals)]
        gate[i] = vals.size > 0 and (vals > threshold).mean() > fraction
    return gate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage2_npz", type=Path)
    ap.add_argument("--stage3-npz", type=Path, default=None)
    ap.add_argument("--gate-fraction", type=float, default=0.5,
                    help="fraction of a breathing window that must be above the motion "
                         "threshold before the estimate is held")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("artifacts/stage4"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.stage2_npz)
    fs = float(d["fs_hz"])
    t0 = float(d["grid_times"][0])
    sig = d["breathing"].astype(np.complex128)
    print(f"input        {args.stage2_npz.name}  {sig.shape[0]} x {sig.shape[1]} at {fs:.3f} Hz")
    print(f"windows      {BREATHING_WINDOW_SECONDS:g} s, hop {BREATHING_HOP_SECONDS:g} s")

    gate = None
    if args.stage3_npz is not None and not args.no_gate:
        m = np.load(args.stage3_npz)
        probe = breathing_windows(sig, fs, fabricated=d["fabricated"])
        gate = motion_gate_for(
            probe["time_s"] + t0, probe["window_seconds"],
            m["centres"], m["score"], float(m["motion_threshold"]), args.gate_fraction,
        )
        print(f"motion gate  threshold {float(m['motion_threshold']):.3f}, "
              f"fraction {args.gate_fraction:g} -> {100 * gate.mean():.1f}% of windows held")

    out = breathing_windows(sig, fs, fabricated=d["fabricated"], motion_gate=gate)
    centres = out["time_s"] + t0
    print(f"resolution   {out['rpm_resolution']:.2f} rpm raw, interpolated below that")

    detected = np.isfinite(out["rpm"])
    print(f"detection    {100 * detected.mean():.1f}% of windows report a rate")
    for lo, hi, label in [(*STILL_RANGE_S, "occupied still"), (525, 595, "settled empty")]:
        m = (centres >= lo) & (centres <= hi)
        got = out["rpm"][m & detected]
        conf = out["confidence"][m]
        print(f"  {label:16s} {100 * detected[m].mean():5.1f}% detected  "
              f"conf p50 {np.nanmedian(conf):.3f}  " +
              (f"rpm p50 {np.median(got):5.2f}  p10-p90 {np.percentile(got, 10):.1f}-"
               f"{np.percentile(got, 90):.1f}" if got.size else "no rate"))

    m = (centres >= STILL_RANGE_S[0]) & (centres <= STILL_RANGE_S[1]) & detected
    if m.any():
        inside = ((out["rpm"][m] >= SANE_RPM[0]) & (out["rpm"][m] <= SANE_RPM[1])).mean()
        print(f"  sanity     {100 * inside:.1f}% of detected rates inside "
              f"{SANE_RPM[0]:g}-{SANE_RPM[1]:g} rpm")

    npz = args.out / f"{args.stage2_npz.stem.replace('_stage2', '')}_stage4.npz"
    np.savez_compressed(
        npz, centres=centres, **{k: v for k, v in out.items() if k != "time_s"}
    )
    print(f"artefacts    {npz} ({npz.stat().st_size / 1e6:.1f} MB)")
    plot(args.out / f"{npz.stem}.png", centres, out)


def plot(png, centres, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(centres, out["rpm"], lw=1.2, color="#2f6fed", label="rate")
    for lo, hi in [SANE_RPM]:
        ax.axhspan(lo, hi, color="#2f9c6f", alpha=0.12, label=f"{lo:g}-{hi:g} rpm")
    if out["gated"].any():
        ax.fill_between(centres, *out["band_rpm"], where=out["gated"],
                        color="#d62728", alpha=0.15, label="held (gross motion)")
    ax.set_ylim(*out["band_rpm"])
    ax.set_ylabel("rate (rpm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Respiration rate, blank where nothing was believable",
                 fontsize=10, loc="left")

    ax = axes[1]
    ax.plot(centres, out["confidence"], lw=1.0, color="#7b5ea7")
    ax.axhline(0.4, color="#d62728", ls="--", lw=1.2, label="report threshold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("confidence")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("tone x estimator agreement x group consensus", fontsize=10, loc="left")

    ax = axes[2]
    for g in range(out["group_rpm"].shape[0]):
        ax.plot(centres, out["group_rpm"][g], lw=0.7, alpha=0.75)
    ax.set_ylim(*out["band_rpm"])
    ax.set_ylabel("per-group rate\n(rpm)")
    ax.set_title("Subcarrier groups estimated separately: agreement is evidence, "
                 "spread is a second target or noise", fontsize=10, loc="left")

    ax = axes[3]
    w = out["weights"]
    im = ax.imshow(w, aspect="auto", origin="lower", cmap="viridis",
                   extent=[centres[0], centres[-1], 0, w.shape[0]],
                   vmin=0, vmax=np.percentile(w, 99))
    ax.set_ylabel("kept subcarrier")
    ax.set_xlabel("time (s)")
    ax.set_title("w[k]: in-band power fraction, i.e. which delays carry the chest",
                 fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, pad=0.01)

    fig.savefig(png, dpi=110)
    print(f"figure       {png}")


if __name__ == "__main__":
    main()
