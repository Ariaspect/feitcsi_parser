#!/usr/bin/env python3
"""Score our detector and LG's against camera ground truth, on one time grid.

Written to be pointed at data that does not exist yet. Everything it needs is
discovered from the capture directory: which runs the camera saw as empty, which
of those can calibrate which, and what each detector says about the rest.

The calibration rule is the point of the whole thing, so it is explicit:

  * A capture is NEVER calibrated on itself, nor on any stretch of itself. The
    reference comes only from OTHER captures the camera labelled completely
    empty. Using a capture's own empty stretches means knowing the answer
    before measuring, which is how the earlier AUC 0.86-0.97 was obtained and
    why it did not survive contact with a deployable protocol.

  * The reference needs SEVERAL empty captures, not one. The threshold is a
    multiple of how far two empty captures sit apart, and a single capture can
    only report how far it wanders from itself -- about 0.04 dB overnight
    against the ~0.2 dB that separates captures, i.e. a threshold roughly 5x too
    low. On 20260914 one empty capture was all there was and every threshold
    from 0.08 to 3.7 dB failed: 100% recall at 0% specificity, or the reverse.

  * The reference also has to be NEAR IN TIME, and --ref-age-h is the only
    thing enforcing that. The pool screen cannot: it sees the reference
    profiles and never the capture's, so it can only ask whether the references
    agree with each other. Five camera-empty captures from one 20260827 morning
    agree to 0.161 dB, give a healthy 0.175 dB scale, and put 100% of a
    20260904 capture's windows above the threshold that follows -- its quietest
    second included, at 9x. Widening the window buys silently wrong verdicts.

  * There is no numeric test for "does this pool apply to this capture", and
    one was looked for. The capture's distance from its pool, in thresholds,
    spans 0.09-5.65 over 22 calibrations from the August protocol that scored
    94% recall at 91% specificity, while two known-broken calibrations read
    4.94 and 5.20 -- inside that range. It cannot separate them because the
    numerator also carries how occupied the capture is: every healthy case
    above 5 is occupied 96-100% of the time and is therefore far from any empty
    reference for honest reasons. The time window is the guard; there is no
    second one.

  * When a capture cannot be calibrated under those rules it is reported as
    UNSCOREABLE, never scored against a fallback. The September controlled
    sessions are all unscoreable and should read that way rather than
    contributing a made-up number.

Both detectors are resampled onto the same grid before scoring, because they
disagree about time by construction: ours reduces a window to one verdict, LG's
emits movement events and holds a state between them. The grid defaults to 1 s,
which is the camera's own rate -- there is no ground truth finer than that.

    eval_pipeline.py --days 20260915 20260916
    eval_pipeline.py --days 20260915 --lg-sweep
"""
import argparse
import json
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from backend import presence, tiles  # noqa: E402

CONF = 0.5          # camera box confidence below this is not a person
BOARD_VENV = REPO / ".venv-board" / "bin" / "python"


# --------------------------------------------------------------- discovery --
def survey(roots, days=None):
    """Every capture with both a .bin and camera labels, with its occupancy."""
    out = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        # captures/ is flat; the collection host nests by day.
        dirs = [root] + [d for d in sorted(root.iterdir()) if d.is_dir()]
        for d in dirs:
            for cv in sorted(d.glob("*_cv.json")):
                stamp = cv.name[:15]
                binp = d / f"{stamp}.bin"
                if not binp.is_file():
                    continue
                if days and stamp[:8] not in days:
                    continue
                frames = json.load(open(cv))["frames"]
                if not frames:
                    continue
                t0 = frames[0]["epoch"]
                occ = np.array([
                    [f["epoch"] - t0,
                     1.0 if (f.get("n", 0) > 0 and f.get("max_conf", 0) >= CONF) else 0.0]
                    for f in frames])
                out.append({
                    "stamp": stamp,
                    "path": binp,
                    "occ": occ,
                    "frac": float(occ[:, 1].mean()),
                    "dt": datetime.strptime(stamp, "%Y%m%d_%H%M%S"),
                })
    out.sort(key=lambda c: c["dt"])
    return out


def pick_reference(cap, empties, max_age_h, min_pool, pool_size):
    """The empty captures allowed to calibrate this one.

    Prefers the most recent, and takes them from either side in time: a
    reference recorded an hour after the run describes the same room as one from
    an hour before, and insisting on "before" alone throws away half the
    evidence for no physical reason. It must never be the capture itself.
    """
    near = [e for e in empties
            if e["stamp"] != cap["stamp"]
            and abs((cap["dt"] - e["dt"]).total_seconds()) <= max_age_h * 3600]
    if len(near) < min_pool:
        return None
    near.sort(key=lambda e: abs((cap["dt"] - e["dt"]).total_seconds()))
    return near[:pool_size]


# ------------------------------------------------------------- our verdict --
def score_ours(cap, refs, grid_s, k, ref_seconds):
    grids, fs_ref = [], None
    for r in refs:
        g, _, f, *_ = tiles._presence_grid(
            r["path"], 0.0, ref_seconds, mimo=None, source_mac=None, interpolate=True)
        grids.append(g)
        fs_ref = f if fs_ref is None else fs_ref
    g, _, fs, gtimes, *_ = tiles._presence_grid(
        cap["path"], 0.0, 1e9, mimo=None, source_mac=None, interpolate=True)

    width = g.shape[1]
    grids = [x for x in grids if x.shape[1] == width]
    if len(grids) < 2:
        return None, "reference subcarrier width does not match"

    # Screen before trusting: a pool spanning two different room states makes
    # dev_scale the gap between them rather than the wander within either. Two
    # camera-empty captures six hours apart on 20260914 sat 10.6 dB apart and
    # produced a 63 dB threshold that nothing could reach.
    profs = [presence.amplitude_profile(x) for x in grids]
    keep, spread = presence.screen_reference_pool(profs)
    if len(keep) < 2:
        return None, (f"reference pool disagrees by {spread:.2f} dB; no subset of "
                      f"2 or more agrees within "
                      f"{presence.DEFAULT_MAX_POOL_SPREAD_DB:g} dB")
    grids = [grids[i] for i in keep]

    # loo: the pool stands in for "captures that are not this one", which is
    # exactly what this capture is to them.
    ref = presence.presence_reference(grids, fs_ref, scale_mode="loo")
    thr = k * ref["dev_scale"]

    n = max(1, int(round(grid_s * fs)))
    profiles = ref["profiles"]
    centres, devs = [], []
    for start in range(0, g.shape[0] - n + 1, n):
        w = presence.amplitude_profile(g[start:start + n])
        # A window is empty if it matches ANY known-empty state, so the verdict
        # takes the distance to the nearest -- the same rule the scale was built
        # under, which is what makes their ratio mean anything.
        near = [presence.baseline_deviation(w, p) for p in profiles]
        near = [v for v in near if np.isfinite(v)]
        if not near:
            continue
        centres.append(float(gtimes[0]) + (start + n / 2) / fs)
        devs.append(min(near))
    devs = np.asarray(devs)
    return {"t": np.asarray(centres), "dev": devs,
            "threshold": thr, "dev_scale": ref["dev_scale"],
            "state": devs > thr, "n_ref": len(grids)}, None


# -------------------------------------------------------------- lg verdict --
def score_lg(cap, grid_s, threshold, absence, duration):
    """Replay LG's detector and turn its +/- events into state on our grid.

    It must run under .venv-board: its TLV length arithmetic depends on NumPy
    1.x integer promotion, and on NumPy 2 the walk desynchronises silently.
    """
    if not BOARD_VENV.is_file():
        return None, f"missing {BOARD_VENV}"
    cmd = [str(BOARD_VENV), str(HERE / "lg_detect_replay.py"), str(cap["path"]),
           "--threshold", str(threshold), "--absence", str(absence)]
    try:
        raw = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None, "replay timed out"
    if raw.returncode != 0:
        return None, (raw.stderr or "replay failed").strip().splitlines()[-1][:80]
    res = json.loads(raw.stdout)

    centres = np.arange(grid_s / 2, duration, grid_s)
    state = np.zeros(centres.shape, dtype=bool)
    cur = False                      # its current_status starts False
    idx = 0
    for ev in sorted(res["events"], key=lambda e: e["t"]):
        while idx < centres.size and centres[idx] < ev["t"]:
            state[idx] = cur
            idx += 1
        cur = (ev["kind"] == "+")
    state[idx:] = cur
    return {"t": centres, "state": state, "events": len(res["events"])}, None


# ------------------------------------------------------------------ score ---
def confusion(t, state, occ, grid_s):
    """Compare to the camera on the same grid; drop windows it cannot call."""
    tp = fp = fn = tn = 0
    for c, p in zip(t, state):
        m = (occ[:, 0] >= c - grid_s / 2) & (occ[:, 0] <= c + grid_s / 2)
        if not m.any():
            continue
        f = occ[m, 1].mean()
        if f == 0.0:
            fp, tn = (fp + 1, tn) if p else (fp, tn + 1)
        elif f > 0.5:
            tp, fn = (tp + 1, fn) if p else (tp, fn + 1)
    return tp, fp, fn, tn


def report(name, rows):
    tp = sum(r[0] for r in rows); fp = sum(r[1] for r in rows)
    fn = sum(r[2] for r in rows); tn = sum(r[3] for r in rows)
    tot = tp + fp + fn + tn
    if tot == 0:
        print(f"  {name:10} no scoreable window")
        return None
    rec = tp / (tp + fn) * 100 if tp + fn else float("nan")
    spec = tn / (fp + tn) * 100 if fp + tn else float("nan")
    prec = tp / (tp + fp) * 100 if tp + fp else float("nan")
    print(f"  {name:10} acc {(tp+tn)/tot*100:5.1f}%  recall {rec:5.1f}%  "
          f"spec {spec:5.1f}%  prec {prec:5.1f}%   "
          f"[tp {tp} fp {fp} fn {fn} tn {tn}]")
    return {"accuracy": (tp + tn) / tot, "recall": rec / 100, "specificity": spec / 100,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=[str(REPO / "captures"), "/home/lg_csi/lg_csi_captures"])
    ap.add_argument("--days", nargs="*", help="YYYYMMDD filter; default all")
    ap.add_argument("--grid", type=float, default=1.0, help="seconds per verdict")
    ap.add_argument("--k", type=float, default=presence.DEFAULT_BASELINE_DEV_K)
    ap.add_argument("--ref-seconds", type=float, default=120.0)
    ap.add_argument("--ref-age-h", type=float, default=6.0)
    ap.add_argument("--pool", type=int, default=5)
    ap.add_argument("--min-pool", type=int, default=2)
    ap.add_argument("--lg-threshold", type=float, default=26.0)
    ap.add_argument("--lg-absence", type=float, default=10.0)
    ap.add_argument("--lg-sweep", nargs="*", type=float,
                    help="thresholds to try for LG instead of a single one")
    ap.add_argument("--skip-lg", action="store_true")
    ap.add_argument("--out", help="write per-capture results as JSON")
    args = ap.parse_args()

    caps = survey(args.roots, set(args.days) if args.days else None)
    if not caps:
        sys.exit("no labelled captures found")
    empties = [c for c in caps if c["frac"] == 0.0]
    tests = [c for c in caps if c["frac"] > 0.0]
    print(f"{len(caps)} labelled captures  |  {len(empties)} camera-empty  "
          f"|  {len(tests)} with an occupant")
    print(f"grid {args.grid}s   k {args.k}   pool<={args.pool} within {args.ref_age_h}h\n")

    ours_rows, lg_rows, per_capture, unscoreable = [], [], [], []
    for cap in tests:
        refs = pick_reference(cap, empties, args.ref_age_h, args.min_pool, args.pool)
        if refs is None:
            unscoreable.append((cap["stamp"], f"<{args.min_pool} empty captures "
                                              f"within {args.ref_age_h}h"))
            continue
        res, err = score_ours(cap, refs, args.grid, args.k, args.ref_seconds)
        if res is None:
            unscoreable.append((cap["stamp"], err))
            continue
        o = confusion(res["t"], res["state"], cap["occ"], args.grid)
        ours_rows.append(o)
        row = {"stamp": cap["stamp"], "occupancy": cap["frac"],
               "n_ref": res["n_ref"], "threshold": res["threshold"],
               "ours": o}
        if not args.skip_lg:
            duration = float(cap["occ"][-1, 0])
            lg, lerr = score_lg(cap, args.grid, args.lg_threshold,
                                args.lg_absence, duration)
            if lg is not None:
                l = confusion(lg["t"], lg["state"], cap["occ"], args.grid)
                lg_rows.append(l)
                row["lg"] = l
            else:
                row["lg_error"] = lerr
        per_capture.append(row)
        print(f"  {cap['stamp']}  occ {cap['frac']*100:4.0f}%  refs {res['n_ref']}  "
              f"thr {res['threshold']:.3f} dB")

    print(f"\n=== pooled over {len(ours_rows)} captures, {args.grid}s grid ===")
    summary = {"ours": report("ours", ours_rows)}
    if lg_rows:
        summary["lg"] = report(f"lg t={args.lg_threshold:g}", lg_rows)

    if args.lg_sweep is not None and not args.skip_lg:
        grid = args.lg_sweep or [10, 14, 18, 22, 26, 30, 34, 40, 50]
        print("\n=== LG threshold sweep ===")
        best = None
        for t in grid:
            rows = []
            for cap in tests:
                duration = float(cap["occ"][-1, 0])
                lg, _ = score_lg(cap, args.grid, t, args.lg_absence, duration)
                if lg is not None:
                    rows.append(confusion(lg["t"], lg["state"], cap["occ"], args.grid))
            m = report(f"t={t:g}", rows)
            if m and (best is None or m["accuracy"] > best[1]["accuracy"]):
                best = (t, m)
        if best:
            print(f"\n  best LG threshold {best[0]:g} dB "
                  f"at {best[1]['accuracy']*100:.1f}% accuracy")
            summary["lg_best"] = {"threshold": best[0], **best[1]}

    if unscoreable:
        print(f"\n{len(unscoreable)} UNSCOREABLE (no fallback was used):")
        for stamp, why in unscoreable:
            print(f"  {stamp}  {why}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"summary": summary, "captures": per_capture,
             "unscoreable": [{"stamp": s, "reason": w} for s, w in unscoreable],
             "params": vars(args) | {"roots": [str(r) for r in args.roots]}},
            indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
