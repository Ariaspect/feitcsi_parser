#!/usr/bin/env python3
"""Present the captures as a Room / Configuration / Scenario tree.

The tree is built out of SYMLINKS and can be thrown away and rebuilt at any
time. The captures stay where they are, and ``_meta.json`` stays the single
place a condition is recorded. Three reasons it is not a move:

  * Most captures do not carry these fields yet -- the 195 unattended runs from
    August have no meta sidecar at all -- so moving files would file almost
    everything under "unspecified" and then need moving again each time a
    condition is recovered.

  * One capture belongs to several groupings at once. Sliced by room today,
    by subject tomorrow, by scenario for a paper. Directories force one choice;
    a rebuildable view does not.

  * Reference selection finds empty captures by time proximity and the
    collection host stores by day. Relocating files puts those two layouts out
    of step for no gain.

The tree deliberately lands OUTSIDE the API's capture roots. ``_walk_captures``
follows symlinked directories and only de-duplicates directories, not files, so
a tree inside ``captures/`` would make every capture appear twice -- once as
itself and once through the tree -- and be eligible twice as a calibration
reference.

    build_dataset_tree.py                      # dry run
    build_dataset_tree.py --apply
    build_dataset_tree.py --by room subject --apply
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = [REPO / "captures", Path("/home/lg_csi/lg_csi_captures")]
DEFAULT_TREE = REPO / "datasets"
DEFAULT_AXES = ["room", "configuration", "scenario"]
UNSET = "unspecified"

# Sidecars that must follow the capture, or the tree is useless: without _cv the
# capture has no ground truth and without _meta it has no conditions.
SIDECAR_SUFFIXES = ("_cv.json", "_meta.json", "_frames.tar", "_frames.tar.zst")


def safe(value: str) -> str:
    """A directory name that survives every shell and filesystem here."""
    out = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(value))
    return out.strip("_") or UNSET


def find_captures(roots):
    out = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in (".bin", ".dat") or not path.is_file():
                continue
            # A capture already inside a tree we built is not a new capture.
            if DEFAULT_TREE in path.parents:
                continue
            out.append(path)
    return out


def conditions(capture: Path, axes):
    meta_path = capture.with_name(f"{capture.stem}_meta.json")
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
    # An occupancy-derived fallback for scenario, so the 195 unattended runs
    # land somewhere more useful than "unspecified": the camera already knows
    # whether anyone was in them.
    if "scenario" not in meta:
        cv_path = capture.with_name(f"{capture.stem}_cv.json")
        if cv_path.is_file():
            try:
                frac = json.loads(cv_path.read_text())["summary"]["fraction_occupied"]
            except (OSError, json.JSONDecodeError, KeyError):
                frac = None
            if frac is not None:
                meta["scenario"] = ("empty" if frac == 0.0
                                    else "occupied" if frac > 0.5
                                    else "partial")
    return [safe(meta.get(a, UNSET)) for a in axes], meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[str(p) for p in DEFAULT_ROOTS])
    ap.add_argument("--tree", default=str(DEFAULT_TREE))
    ap.add_argument("--by", nargs="+", default=DEFAULT_AXES,
                    help=f"meta fields to nest by; default {' '.join(DEFAULT_AXES)}")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="remove the existing tree first; it is only symlinks")
    args = ap.parse_args()

    tree = Path(args.tree)
    captures = find_captures(args.roots)
    if not captures:
        sys.exit(f"no captures under {', '.join(args.roots)}")

    plan = {}
    for capture in captures:
        parts, _ = conditions(capture, args.by)
        plan.setdefault(tuple(parts), []).append(capture)

    width = max(len(" / ".join(k)) for k in plan)
    print(f"{len(captures)} captures -> {len(plan)} groups, nested by "
          f"{' / '.join(args.by)}\n")
    for key in sorted(plan):
        label = " / ".join(key)
        print(f"  {label:<{width}}  {len(plan[key]):4d}")

    unspecified = sum(len(v) for k, v in plan.items() if all(p == UNSET for p in k))
    if unspecified:
        print(f"\n{unspecified} captures carry none of these fields. They are "
              f"filed under {'/'.join([UNSET] * len(args.by))} rather than guessed; "
              f"record the fields at capture time (run_experiment.sh --room/"
              f"--config/--scenario) or backfill with set_meta.py.")

    if not args.apply:
        print("\ndry run -- rerun with --apply to build the symlink tree")
        return

    if args.clean and tree.exists():
        shutil.rmtree(tree)
    made = relinked = 0
    for key, group in plan.items():
        target_dir = tree.joinpath(*key)
        target_dir.mkdir(parents=True, exist_ok=True)
        for capture in group:
            for path in [capture] + [
                capture.with_name(f"{capture.stem}{suf}")
                for suf in SIDECAR_SUFFIXES
            ]:
                if not path.exists():
                    continue
                link = target_dir / path.name
                rel = os.path.relpath(path.resolve(), target_dir)
                if link.is_symlink():
                    if os.readlink(link) == rel:
                        continue
                    link.unlink()
                    relinked += 1
                elif link.exists():
                    continue          # a real file already there is not ours
                link.symlink_to(rel)
                made += 1
    print(f"\n{tree}: {made} links created, {relinked} repointed")
    print("symlinks only -- delete the tree and rebuild whenever the "
          "conditions change")


if __name__ == "__main__":
    main()
