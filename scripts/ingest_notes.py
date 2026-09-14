#!/usr/bin/env python3
"""Merge an externally-kept experiment log (CSV) into capture sidecars.

The log is maintained by hand -- one row per run, with a minute-resolution
timestamp.  Captures are named to the second, so each row is matched to the
capture that starts nearest to it inside a tolerance window.

Writes captures/<stamp>_meta.json, preserving any field already there.
Dry run unless --apply.
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
CAPTURES = Path(__file__).resolve().parent.parent / "captures"

STAMP_RE = re.compile(r"^(\d{8}_\d{6})$")
DATE_RE = re.compile(r"^([A-Z][a-z]+ \d{1,2}, \d{4} \d{1,2}:\d{2})")

# The log records NIC handling in free text.  It is the one field we cannot
# afford to leave unparsed: a moved NIC shifts the empty-room profile by
# 3-5 dB, which is the same order as a person.
NIC_DETACHED = re.compile(r"떨어[짐진]|떨어졌")
NIC_ATTACHED = re.compile(r"붙임|다시 ?부착|부착(?!떨어)")


def capture_stamps():
    seen = set()
    for p in CAPTURES.iterdir():
        name = p.name
        for suffix in ("_meta.json", "_cv.json", "_frames.tar.zst", "_frames.tar", ".dat"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        else:
            name = name.split(".")[0]
        if STAMP_RE.match(name):
            seen.add(name)
    return sorted(seen)


def stamp_to_dt(stamp):
    return datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=KST)


def parse_row_time(raw):
    m = DATE_RE.match(raw.strip())
    if not m:
        return None
    return datetime.strptime(m.group(1), "%B %d, %Y %H:%M").replace(tzinfo=KST)


def nic_state(note):
    """Explicit NIC mention in a row, or None when the row is silent."""
    if NIC_DETACHED.search(note):
        return False
    if NIC_ATTACHED.search(note):
        return True
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--apply", action="store_true", help="write the sidecars")
    ap.add_argument("--before", type=int, default=90,
                    help="seconds a capture may start before the logged time")
    ap.add_argument("--after", type=int, default=240,
                    help="seconds a capture may start after the logged time")
    args = ap.parse_args()

    stamps = capture_stamps()
    if not stamps:
        sys.exit(f"no captures under {CAPTURES}")
    times = {s: stamp_to_dt(s) for s in stamps}

    with open(args.csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    parsed = []
    for row in rows:
        raw = (row.get("Date") or "").strip()
        if raw:
            parsed.append((parse_row_time(raw), raw, row))
    parsed.sort(key=lambda r: (r[0] is None, r[0]))

    used, plans, unmatched = {}, [], []
    carried, carried_day = None, None
    for when, raw, row in parsed:
        if when is None:
            unmatched.append((raw, "unparseable date"))
            continue

        best, best_delta = None, None
        for stamp, t in times.items():
            delta = (t - when).total_seconds()
            if -args.before <= delta <= args.after:
                if best_delta is None or abs(delta) < abs(best_delta):
                    best, best_delta = stamp, delta
        if best is None:
            unmatched.append((raw, "no capture in window"))
            continue
        if best in used:
            unmatched.append((raw, f"{best} already claimed by {used[best]}"))
            continue
        used[best] = raw

        note = (row.get("비고") or "").strip()
        explicit = nic_state(note)
        if explicit is None and carried_day == when.date():
            mounted, inferred = carried, True
        else:
            mounted, inferred = explicit, False
        if explicit is not None:
            carried, carried_day = explicit, when.date()

        fields = {
            "subject": (row.get("Person") or "").strip() or None,
            "session_type": (row.get("유형") or "").strip() or None,
            "log_note": note or None,
            "log_source": Path(args.csv_path).name,
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        if mounted is not None:
            fields["nic_mounted"] = mounted
            fields["nic_mounted_inferred"] = inferred
        plans.append((best, best_delta, fields))

    plans.sort()
    n_written = 0
    for stamp, delta, fields in plans:
        path = CAPTURES / f"{stamp}_meta.json"
        existing = json.loads(path.read_text()) if path.exists() else {}
        changed = {k: v for k, v in fields.items() if existing.get(k) != v}
        mark = "new " if not existing else ("edit" if changed else "same")
        if "nic_mounted" in fields:
            nic = "  nic=" + ("on" if fields["nic_mounted"] else "OFF")
            if fields["nic_mounted_inferred"]:
                nic += "?"
        else:
            nic = "  nic=??"
        print(f"{mark} {stamp}  {delta:+5.0f}s  {fields.get('subject','')}"
              f"  [{fields.get('session_type','')}]{nic}  {fields.get('log_note','')}")
        if args.apply and changed:
            merged = dict(existing)
            merged.update(fields)
            path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
            n_written += 1

    print(f"\n{len(plans)} matched, {len(unmatched)} unmatched"
          + (f", {n_written} written" if args.apply else ""))
    for raw, why in unmatched:
        print(f"  unmatched row: {raw}  ({why})")

    lo = min((times[s] for s, _, _ in plans), default=None)
    hi = max((times[s] for s, _, _ in plans), default=None)
    if lo and hi:
        gaps = [s for s in stamps if lo <= times[s] <= hi and s not in used]
        if gaps:
            print(f"\n{len(gaps)} captures inside the logged span carry no row:")
            for s in gaps:
                print(f"  {s}")

    if not args.apply:
        print("\ndry run -- rerun with --apply to write")


if __name__ == "__main__":
    main()
