#!/usr/bin/env python3
"""Build data/ground_truth.csv: one row per one-minute capture, for training and evaluating the presence models.

Sources, in order of authority
  1. The operator's ground-truth table of 2026-10-09 (env#2, 10/01 -> 10/09), transcribed in TABLE below with two
     corrections against the files on disk: 10/02 "17:34" -> 17:33 (there is no capture at 17:34; the one at 17:33:02
     is otherwise unlisted) and the 10/09 rotating-fan block ends at 12:20, not 12:19 (its 35th capture starts 12:20:25).
  2. Operator sidecars (<stem>_meta.json) for captures the table does not cover (the 10/06 20:03-20:12 distance test).
  3. The camera (<stem>_cv.json, fraction of frames with a person) for env#1 (09/29 20:25 -> 09/30), which has no table.

Rules
  * A capture is one minute when its camera span is 45-75 s (file size when there is no sidecar); nothing else is listed.
  * env#1 = up to and including 09/30, env#2 = from 10/01 (user, 2026-10-09; the 09/29 -> 09/30 cron ran unbroken).
  * label is binary: "occupied" means a human is present; robot and fan captures are "empty" with nonhuman=1.
  * activity is a stratum for error analysis, never a training target.
  * A capture the table does not mention is never given a guessed label for training: it gets use=0 and a reason.
    The label column still carries the best available evidence (camera, sidecar) so the user can flip use to 1.

Usage
  python scripts/build_ground_truth.py --captures /home/lg_csi/lg_csi_captures [--out data/ground_truth.csv]
Exit status is 1 when the table and the files disagree (a listed time without a file, a range count that is off).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime

ENV2_FROM = "20261001"              # first day in env#2; everything before is env#1
ONE_MINUTE = (45.0, 75.0)           # camera span in seconds that counts as a one-minute capture
PROTOCOL_FROM = "20260929"          # first day of the one-minute protocol; earlier one-minute files are strays
CAMERA_OCCUPIED = 0.5               # env#1 label: fraction of camera frames with a person
CAMERA_PARTIAL = (0.05, 0.95)       # env#1: a fraction strictly inside this band is a person entering/leaving -> use=0
CAMERA_BLIND = ("20260930", "18:05", "18:25")   # steady detector firing between occupied captures, person out of view
                                                 # or a link episode -- undecidable (docs/hybrid2.md, held-out section)

# The operator's table. An item is "HH:MM" (one capture) or (first, last, expected_count) with last=None for "to the
# end of the day". Activities for occupied rows; for empty rows the key names the non-human source.
TABLE = {
    "empty": {
        "20261001": ["15:15", "15:18", "15:54", "15:57", "16:00", "16:03", "16:06", "20:06", "20:09", "20:12"],
        "20261002": [("02:00", "04:27", 50)],
    },
    "small_move": {   # 작은 움직임
        "20261001": ["18:42", "18:45", "18:48", "18:51", "18:54", "18:57", "19:06", "19:09", "19:12", "19:21", "19:24",
                     "19:30", "19:45", "19:51", "19:57"],
        "20261002": ["17:36", "17:39", "17:45", "17:51", "17:54", "17:57", "18:03", "18:06", "19:57"],
    },
    "large_move": {   # 큰 움직임
        "20261001": ["19:00", "19:03", "19:15", "19:18", "19:27", "19:36", "19:39", "19:42", "19:48", "19:54"],
        "20261002": ["17:18", "17:24", "17:42", "17:48", "18:00", "20:24", "20:27", "20:36"],
        "20261005": ["15:53", "15:54"],
    },
    "static": {       # 정적 재실
        "20261002": ["17:27", "17:30", "17:33", "18:21", "18:24", "18:27", "19:54", "20:00", "20:03", "20:06", "20:09",
                     "20:12", "20:15", "20:18", "20:21", "20:30", "20:33", "20:39", "20:42"],   # 17:33 was "17:34"
        "20261005": ["12:36", "12:39", "12:42", "12:45", "12:57", "13:27", "14:19", "14:22", "14:25", "14:30", "14:33",
                     ("14:49", "14:58", 10), "15:07", "15:08", ("15:10", "15:16", 7), ("16:27", "16:36", 10),
                     ("16:45", "16:49", 5), "16:55", "18:02", "18:03"],
    },
    "standing_walk": {   # 서서 움직임 / 돌아다님
        "20261001": ["20:00", "20:03"],
        "20261005": ["15:31", "15:32", ("15:58", "16:07", 10)],
    },
    "robot_vacuum": {
        "20261005": [("20:55", "21:04", 10)],
        "20261006": [("13:27", "15:31", 35)],
        "20261007": [("14:26", "16:04", 33)],
    },
    "fan_static": {"20261008": [("20:08", "20:28", 20)]},
    "fan_rotating": {"20261008": [("20:33", None, 5)], "20261009": [("10:48", "11:31", 40), ("11:39", "12:20", 35)]},
}
OCCUPIED_ACTIVITIES = ("static", "small_move", "large_move", "standing_walk")
NONHUMAN = ("robot_vacuum", "fan_static", "fan_rotating")
# Sidecar scenarios the table does not cover, and what they mean.
SIDECAR_SCENARIOS = {
    "robot_vacuum_no_person": ("empty", "robot_vacuum"),
    "person_outside_camera_view_distance_test": ("occupied", "static"),
    "person_moving_outside_view_beyond_3m": ("occupied", "unknown"),
}

COLUMNS = ("capture", "env", "day", "time", "duration_s", "label", "activity", "nonhuman", "source",
           "camera_fraction", "camera_edited", "use", "reason")


def _hhmm(stem: str) -> str:
    return f"{stem[9:11]}:{stem[11:13]}"


def scan(captures_root: str) -> list[dict]:
    """Every .bin under <root>/<day>/ with what its sidecars say."""
    out = []
    for path in sorted(glob.glob(os.path.join(captures_root, "2026*", "*.bin"))):
        stem = os.path.basename(path)[:-4]
        day = os.path.basename(os.path.dirname(path))
        if len(stem) < 15 or not stem[:8].isdigit() or stem[:8] != day:
            continue
        cv_path = path[:-4] + "_cv.json"
        meta_path = path[:-4] + "_meta.json"
        span = frac = None
        if os.path.exists(cv_path):
            with open(cv_path) as fh:
                summary = json.load(fh).get("summary", {})
            span = summary.get("span_s")
            frac = summary.get("fraction_occupied")
        if span is None:
            span = os.path.getsize(path) / 11.0e6 * 58.0      # ~11 MB per minute at 42 Hz
        scenario = None
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                scenario = json.load(fh).get("scenario")
        out.append({"capture": stem, "day": day, "hhmm": _hhmm(stem), "span": float(span), "frac": frac,
                    "scenario": scenario, "camera_edited": os.path.exists(cv_path + ".orig")})
    return out


def apply_table(captures: list[dict], table: dict) -> list[str]:
    """Attach the table's class to each capture it names; return human-readable mismatches."""
    by_day: dict[str, list[dict]] = {}
    for c in captures:
        by_day.setdefault(c["day"], []).append(c)
    problems = []
    for cls, days in table.items():
        for day, items in days.items():
            rows = by_day.get(day, [])
            for item in items:
                if isinstance(item, str):
                    hits = [r for r in rows if r["hhmm"] == item]
                    if len(hits) != 1:
                        problems.append(f"{cls} {day} {item}: {len(hits)} files")
                else:
                    first, last, expected = item
                    hits = [r for r in rows if r["hhmm"] >= first and (last is None or r["hhmm"] <= last)]
                    if len(hits) != expected:
                        problems.append(f"{cls} {day} {first}-{last or 'end'}: table {expected}, files {len(hits)}")
                for h in hits:
                    if h.get("table_class") and h["table_class"] != cls:
                        problems.append(f"{h['capture']}: in both {h['table_class']} and {cls}")
                    h["table_class"] = cls
    return problems


def _row(c: dict, **kw) -> dict:
    env = 2 if c["day"] >= ENV2_FROM else 1
    row = {"capture": c["capture"], "env": env, "day": c["day"], "time": f"{c['capture'][9:11]}:{c['capture'][11:13]}:{c['capture'][13:15]}",
           "duration_s": f"{c['span']:.0f}", "label": "", "activity": "", "nonhuman": 0, "source": "",
           "camera_fraction": "" if c["frac"] is None else f"{c['frac']:.3f}", "camera_edited": int(bool(c["camera_edited"])),
           "use": 1, "reason": ""}
    row.update(kw)
    return row


def _camera_label(c: dict) -> tuple[str, str]:
    frac = c["frac"]
    if frac is None:
        return "", "no camera sidecar"
    label = "occupied" if frac >= CAMERA_OCCUPIED else "empty"
    if CAMERA_PARTIAL[0] < frac < CAMERA_PARTIAL[1]:
        return label, f"camera partial ({frac:.2f}): person entering or leaving"
    return label, ""


def build_rows(captures: list[dict]) -> list[dict]:
    rows = []
    for c in captures:
        if not (ONE_MINUTE[0] <= c["span"] <= ONE_MINUTE[1]):
            continue
        env = 2 if c["day"] >= ENV2_FROM else 1
        cls = c.get("table_class")
        if cls:
            if cls in OCCUPIED_ACTIVITIES:
                rows.append(_row(c, label="occupied", activity=cls, source="table"))
            elif cls in NONHUMAN:
                rows.append(_row(c, label="empty", activity=cls, nonhuman=1, source="table"))
            else:
                rows.append(_row(c, label="empty", activity="none", source="table"))
            continue
        if c["day"] < PROTOCOL_FROM:
            label, _ = _camera_label(c)
            rows.append(_row(c, label=label, activity="unknown" if label == "occupied" else ("none" if label else ""),
                             source="camera" if label else "", use=0,
                             reason=f"one-minute file before the protocol started ({PROTOCOL_FROM})"))
            continue
        if env == 1:
            label, why = _camera_label(c)
            activity = "unknown" if label == "occupied" else ("none" if label else "")
            use, reason = (1, "") if label and not why else (0, why)
            if (label == "empty" and c["day"] == CAMERA_BLIND[0] and CAMERA_BLIND[1] <= c["hhmm"] <= CAMERA_BLIND[2]):
                use, reason = 0, "camera-blind block: detector fired steadily between occupied captures, undecidable"
            rows.append(_row(c, label=label, activity=activity, source="camera" if label else "", use=use, reason=reason))
            continue
        # env#2 but not in the table: sidecar, then camera, as evidence only
        if c["scenario"] in SIDECAR_SCENARIOS:
            label, activity = SIDECAR_SCENARIOS[c["scenario"]]
            rows.append(_row(c, label=label, activity=activity, nonhuman=int(activity in NONHUMAN), source="sidecar",
                             use=0, reason=f"not in the operator table (sidecar: {c['scenario']})"))
            continue
        label, _ = _camera_label(c)
        rows.append(_row(c, label=label, activity="unknown" if label == "occupied" else ("none" if label else ""),
                         source="camera" if label else "", use=0, reason="not in the operator table"))
    return rows


def summarise(rows: list[dict]) -> str:
    lines = []
    for env in (1, 2):
        R = [r for r in rows if r["env"] == env]
        used = [r for r in R if r["use"] == 1]
        lines.append(f"env#{env}: {len(R)} one-minute captures, {len(used)} usable "
                     f"(occupied {sum(r['label'] == 'occupied' for r in used)}, "
                     f"empty {sum(r['label'] == 'empty' and not r['nonhuman'] for r in used)}, "
                     f"non-human {sum(bool(r['nonhuman']) for r in used)}); use=0: {len(R) - len(used)}")
        acts: dict[str, int] = {}
        for r in used:
            acts[r["activity"]] = acts.get(r["activity"], 0) + 1
        lines.append("    by activity: " + ", ".join(f"{k} {v}" for k, v in sorted(acts.items())))
        reasons: dict[str, int] = {}
        for r in R:
            if r["use"] == 0:
                key = r["reason"].split(" (")[0].split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
        if reasons:
            lines.append("    use=0 reasons: " + "; ".join(f"{k} {v}" for k, v in sorted(reasons.items())))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--captures", required=True, help="root holding <YYYYMMDD>/<stem>.bin and sidecars")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ground_truth.csv"))
    args = ap.parse_args(argv)
    captures = scan(args.captures)
    problems = apply_table(captures, TABLE)
    rows = build_rows(captures)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"{len(captures)} captures scanned, {len(rows)} one-minute rows written to {args.out}")
    print(summarise(rows))
    for p in problems:
        print("MISMATCH:", p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
