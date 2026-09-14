#!/usr/bin/env python3
"""Set fields on capture sidecars, selected by stamp prefix.

For facts that apply to a whole day or a whole session and were not recorded
at capture time:

    scripts/set_meta.py 20260911 --set distance_m=1.2 distance_source=recalled

Values parse as JSON when they can (numbers, true/false, null), else stay
strings.  Dry run unless --apply.
"""
import argparse
import json
import re
from pathlib import Path

CAPTURES = Path(__file__).resolve().parent.parent / "captures"
STAMP_RE = re.compile(r"^(\d{8}_\d{6})")


def coerce(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="stamp prefix, e.g. 20260911 or 20260911_14")
    ap.add_argument("--set", dest="pairs", nargs="+", required=True, metavar="KEY=VALUE")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    fields = {}
    for pair in args.pairs:
        if "=" not in pair:
            ap.error(f"expected KEY=VALUE, got {pair!r}")
        k, v = pair.split("=", 1)
        fields[k] = coerce(v)

    stamps = set()
    for p in CAPTURES.iterdir():
        m = STAMP_RE.match(p.name)
        if m and m.group(1).startswith(args.prefix):
            stamps.add(m.group(1))

    if not stamps:
        raise SystemExit(f"no captures matching {args.prefix!r}")

    n = 0
    for stamp in sorted(stamps):
        path = CAPTURES / f"{stamp}_meta.json"
        existing = json.loads(path.read_text()) if path.exists() else {}
        changed = {k: v for k, v in fields.items() if existing.get(k) != v}
        was = {k: existing[k] for k in changed if k in existing}
        mark = "same" if not changed else ("new " if not existing else "edit")
        print(f"{mark} {stamp}  " + " ".join(f"{k}={v!r}" for k, v in fields.items())
              + (f"   (was {was})" if was else ""))
        if args.apply and changed:
            merged = dict(existing)
            merged.update(fields)
            path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
            n += 1

    print(f"\n{len(stamps)} captures matched" + (f", {n} written" if args.apply else "\ndry run -- rerun with --apply"))


if __name__ == "__main__":
    main()
