"""Refit the fixed weights in ``backend.motionsig`` and print them for pasting.

The tab ships constants on purpose -- a panel that fits on the capture it is
displaying reports its own training error -- so this is the script that
produced them and the only thing that should ever change them.

What it does, per capture: decode through ``motionsig.capture_features`` (the
same code path the tab uses, so the fit and the display cannot drift apart),
read the camera sidecar, normalise each capture against *its own* scale, and
pool the normalised windows. That per-capture normalisation is not a detail:
pooling the features raw measures the link rather than the room -- empty-room
variance spans 36x between sample rates in this corpus -- and it inverts the
ranking of the conditions.

One logistic regression per normalisation, class-weight balanced so the 33%
occupied base rate does not set the operating point, and the threshold at the
90th percentile of the empty windows' score. Both are reported with the
per-capture breakdown, because the corpus median matters more than the pooled
number: the pooled figure is dominated by whichever captures are longest.

    python -m scripts.fit_motionsig --captures captures --out weights.json

Adding ``--holdout STAMP`` leaves those captures out of the fit and scores
them separately, which is how the two-capture demonstration figure was made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend import motionsig, truth as truthmod
from backend.app import _camera_truth


def collect(stamps: list[str], root: Path, margin_s: float) -> dict[str, dict]:
    """Per-capture raw features and the camera's verdict on each window."""
    out: dict[str, dict] = {}
    for stamp in stamps:
        path = root / f"{stamp}.bin"
        if not path.is_file():
            print(f"  missing  {stamp}")
            continue
        cam = _camera_truth(path)
        if cam is None or not cam.size:
            print(f"  no camera {stamp}")
            continue
        try:
            feat = motionsig.capture_features(path, 0.0, float("inf"))
        except ValueError as exc:
            print(f"  failed   {stamp}: {exc}")
            continue
        cells, excluded = truthmod.cell_truth(
            feat["time_s"], cam[:, 0], cam[:, 1] > 0.5, feat["window_seconds"] / 2, margin_s,
        )
        scored = np.isfinite(cells)
        out[stamp] = {
            "raw": feat["raw"],
            "occupied": cells > 0.5,
            "scored": scored,
            "empty": scored & (cells <= 0.5),
            "fs": feat["fs"],
        }
        print(f"  ok       {stamp}  {len(feat['time_s']):4d} windows  {feat['fs']:5.1f} Hz  "
              f"occupied {100 * float(np.mean(cells[scored] > 0.5)):4.0f}%  excluded {int(excluded.sum())}")
    return out


def design(recs: dict[str, dict], mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Normalised windows pooled across captures, and their camera labels."""
    rows, labels = [], []
    for rec in recs.values():
        ref = motionsig.reference(rec["raw"], mode, rec["empty"])
        z = {f: (np.asarray(rec["raw"][f], float) - ref[f][0]) / ref[f][1] for f in motionsig.FEATURES}
        ok = rec["scored"]
        for f in motionsig.FEATURES:
            ok = ok & np.isfinite(z[f])
        rows.append(np.stack([z[f][ok] for f in motionsig.FEATURES], axis=1))
        labels.append(rec["occupied"][ok])
    return np.concatenate(rows), np.concatenate(labels)


def per_capture(recs: dict[str, dict], mode: str, coef: dict[str, float]) -> list[tuple]:
    """Recall, specificity and balanced accuracy for each capture on its own."""
    rows = []
    for stamp, rec in sorted(recs.items()):
        ref = motionsig.reference(rec["raw"], mode, rec["empty"])
        _, s = motionsig.score(rec["raw"], ref, coef)
        pred = np.isfinite(s) & (s > coef["threshold"])
        cells = np.where(rec["scored"], rec["occupied"].astype(float), np.nan)
        c = truthmod.confusion(cells, pred)
        rows.append((stamp, c["recall"], c["specificity"]))
    return rows


def report(recs: dict[str, dict], mode: str, coef: dict[str, float], title: str) -> None:
    rows = per_capture(recs, mode, coef)
    print(f"\n  {title}")
    balanced = []
    for stamp, rc, sp in rows:
        b = None if rc is None or sp is None else (rc + sp) / 2
        if b is not None:
            balanced.append(b)
        fmt = lambda v: "   —  " if v is None else f"{100 * v:5.1f}%"  # noqa: E731
        print(f"    {stamp}  recall {fmt(rc)}  spec {fmt(sp)}  balanced {fmt(b)}")
    if balanced:
        print(f"    median balanced {100 * float(np.median(balanced)):.1f}%  "
              f"mean {100 * float(np.mean(balanced)):.1f}%  over {len(balanced)} captures")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", type=Path, default=Path("captures"),
                    help="directory holding the .bin files and their _cv.json sidecars")
    ap.add_argument("--stamps", nargs="*", default=sorted(motionsig.CORPUS),
                    help="capture stamps to fit on; defaults to motionsig.CORPUS")
    ap.add_argument("--holdout", nargs="*", default=[],
                    help="stamps to exclude from the fit and score separately")
    ap.add_argument("--margin-s", type=float, default=truthmod.DEFAULT_MARGIN_S,
                    help="empty camera frames within this many seconds of a transition are not scored")
    ap.add_argument("--specificity", type=float, default=90.0,
                    help="percentile of the empty windows' score taken as the threshold")
    ap.add_argument("--out", type=Path, default=None, help="also write the weights as JSON here")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression

    hold = set(args.holdout)
    print(f"decoding {len(args.stamps)} captures from {args.captures}")
    recs = collect([s for s in args.stamps if s not in hold], args.captures, args.margin_s)
    if not recs:
        raise SystemExit("nothing to fit on")
    held = collect(sorted(hold), args.captures, args.margin_s) if hold else {}

    fitted: dict[str, dict[str, float]] = {}
    for mode in motionsig.MODES:
        X, y = design(recs, mode)
        m = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced").fit(X, y)
        d = m.decision_function(X)
        thr = float(np.percentile(d[~y], args.specificity))
        coef = {f: float(w) for f, w in zip(motionsig.FEATURES, m.coef_[0])}
        coef["intercept"] = float(m.intercept_[0])
        coef["threshold"] = thr
        fitted[mode] = coef
        print(f"\n{mode}: {X.shape[0]} windows, {100 * y.mean():.1f}% occupied")
        print("  " + "  ".join(f"{k} {v:+.6f}" for k, v in coef.items()))
        report(recs, mode, coef, "fitted on these:")
        if held:
            report(held, mode, coef, "held out:")

    print("\npaste into backend/motionsig.COEFFICIENTS:")
    print(json.dumps(fitted, indent=4))
    if args.out:
        args.out.write_text(json.dumps(fitted, indent=2) + "\n")
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
