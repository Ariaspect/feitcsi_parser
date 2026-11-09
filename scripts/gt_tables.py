"""User's GT tables (2026-10-19) -> stems, then a date/session-balanced pick of
the 'Data' count per condition. Leftovers of env1 = the evaluation set.
Balancing: water-fill across dates, then across sessions within a date, then
evenly spaced in time inside a session.

Writes data/gt_tables.csv (one row per capture in a table session; `train`
marks the balanced pick). The web shows its `cond` beside the GT badge.
Run on lg: python scripts/gt_tables.py data/gt_tables.csv"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/lg_csi/lg_csi_captures")
R = lambda d, a, b: ("r", d, a, b)          # range of start times, inclusive
L = lambda d, *ts: ("l", d, ts)             # explicit start times

SPEC = {
 "env1": {
  "empty": (50, [R("1013", "03:00", "03:53"), R("1013", "11:50", "12:16")]),
  "small move": (20, [R("1012", "20:15", "20:36")]),
  "large move": (20, [R("1013", "14:12", "14:18"), R("1013", "14:22", "14:28"), R("1013", "19:59", "20:05"), R("1014", "13:18", "13:24")]),
  "still": (20, [R("1013", "13:56", "14:09"), R("1014", "13:29", "13:42")]),
  "vacuum": (20, [R("1012", "21:53", "22:06"), R("1013", "15:48", "16:05")]),
  "vacuum + still": (20, [R("1013", "16:13", "16:24"), R("1014", "14:13", "14:27")]),
  "static fan": (20, [R("1012", "22:08", "22:28"), R("1014", "12:33", "12:53")]),
  "static fan + still": (20, [R("1013", "15:40", "15:44"), R("1013", "19:25", "19:37"), R("1013", "19:49", "19:56"), R("1014", "13:01", "13:13")]),
  "static fan + small": (20, [R("1013", "15:18", "15:31"), R("1013", "18:44", "19:14")]),
  "rot fan": (20, [R("1012", "20:47", "21:08"), R("1013", "16:30", "16:57")]),
  "rot fan + still": (20, [R("1013", "14:54", "15:07"), R("1014", "13:47", "14:00")]),
  "rot fan + small": (20, [R("1012", "21:16", "21:47")]),
 },
 "env2": {
  "empty": (50, [L("1001", "15:15", "15:18", "15:54", "15:57", "16:00", "16:03", "16:06"), L("1001", "20:06", "20:09", "20:12"),
                 R("1002", "02:00", "04:27")]),
  "small move": (20, [L("1001", "18:42", "18:45", "18:48", "18:51", "18:54", "18:57", "19:06", "19:09", "19:12", "19:21", "19:24", "19:30", "19:45", "19:51", "19:57"),
                      L("1002", "17:36", "17:39", "17:45", "17:51", "17:54", "17:57", "18:03", "18:06", "19:57")]),
  "large move": (20, [L("1001", "19:00", "19:03", "19:15", "19:18", "19:27", "19:36", "19:39", "19:42", "19:48", "19:54"),
                      L("1002", "17:18", "17:24", "17:42", "17:48", "18:00", "20:24", "20:27", "20:36"), L("1005", "15:53", "15:54")]),
  "still": (50, [L("1002", "17:27", "17:30", "17:33", "18:21", "18:24", "18:27", "19:54", "20:00", "20:03", "20:06", "20:09", "20:12", "20:15", "20:18", "20:21", "20:30", "20:33", "20:39", "20:42"),
                 L("1005", "12:36", "12:39", "12:42", "12:45", "12:57", "13:27", "14:19", "14:22", "14:25", "14:30", "14:33"),
                 R("1005", "14:49", "14:58"), L("1005", "15:07", "15:08"), R("1005", "15:10", "15:16"), R("1005", "16:27", "16:36"),
                 R("1005", "16:45", "16:49"), L("1005", "16:55", "18:02", "18:03")]),
  "standing/walking": (10, [L("1001", "20:00", "20:03"), L("1005", "15:31", "15:32"), R("1005", "15:58", "16:07")]),
  "vacuum": (20, [R("1005", "20:55", "21:04"), R("1006", "13:27", "15:31"), R("1007", "14:26", "16:04")]),
  "static fan": (20, [R("1008", "20:08", "20:28"), R("1012", "14:07", "14:12"), R("1019", "17:56", "18:11")]),
  "static fan + still": (20, [R("1012", "12:03", "12:13"), R("1019", "18:16", "18:20")]),
  "static fan + small": (20, [R("1019", "18:24", "18:28")]),
  "rot fan": (20, [("first", "1008", "20:33", 5), R("1009", "10:48", "11:31"), R("1009", "11:39", "12:19"),
                   R("1012", "14:16", "14:20"), R("1012", "14:27", "14:31")]),
  "rot fan + still": (20, [R("1012", "12:45", "12:50"), R("1019", "18:43", "18:48")]),
  "rot fan + small": (20, [R("1012", "12:28", "12:40"), R("1019", "18:34", "18:38"), R("1019", "18:53", "18:58")]),
 },
}
HUMAN = lambda c: c not in ("empty", "vacuum", "static fan", "rot fan")

# Sessions the user tabled after the models were fitted: GT for the web and for tests, never in a
# balanced pick, so adding one cannot change any earlier row's `train`. (spec, the user's note)
LATER = {
 "env2": {
  "small move": [(R("1030", "17:13", "17:18"), ""), (R("1030", "17:27", "17:32"), ""),
                 (R("1030", "17:40", "17:44"), "typing"), (R("1030", "17:48", "17:52"), "typing")],
  "still": [(R("1030", "18:00", "18:05"), ""), (R("1030", "18:06", "18:11"), "prone"),
            (R("1030", "18:17", "18:21"), ""), (R("1030", "18:26", "18:31"), "other seat facing back"),
            (R("1030", "18:43", "18:47"), "other seat"), (R("1030", "19:08", "19:12"), "")],
  "static fan + still": [(R("1102", "20:30", "20:32"), ""), (R("1102", "20:47", "20:49"), ""),
                         (R("1102", "20:58", "21:03"), "")],
  "rot fan + still": [(R("1102", "21:09", "21:14"), ""), (R("1102", "21:16", "21:20"), ""),
                      (R("1104", "19:43", "19:47"), "prone"),
                      (R("1105", "21:41", "21:45"), ""), (R("1105", "21:47", "21:52"), "")],
  "rot fan + small": [(R("1104", "18:21", "18:26"), ""), (R("1104", "18:29", "18:34"), ""),
                      (R("1104", "19:52", "19:56"), "typing"),
                      (R("1105", "19:51", "19:55"), ""), (R("1105", "20:06", "20:10"), "")],
 },
}
# The user's table (2026-11-09, corrected the same day) dates the 1102 static-fan / rot-fan + still sessions 11/3; every
# capture at those times is stamped 20261102 (file name and capture_start_utc_epoch) and none exists for 11/03, so they
# stay under 1102. The corrected table adds rot fan + still on 11/05 21:41-21:45 and 21:47-21:52.

_cache = {}
def day_stems(d):
    day = "2026" + d
    if day not in _cache:
        rows = []
        for b in sorted((ROOT / day).glob("*.bin")):
            st = b.stem; m = b.with_name(st + "_meta.json")
            dur = None
            if m.exists():
                try: dur = json.loads(m.read_text()).get("requested_duration_s")
                except Exception: pass
            rows.append((st, st[9:11] + ":" + st[11:13], dur))
        _cache[day] = rows
    return _cache[day]

def resolve(s):
    if s[0] == "r":
        _, d, a, b = s; return [st for st, hm, _ in day_stems(d) if a <= hm <= b], f"{d} {a}-{b}"
    if s[0] == "l":
        _, d, ts = s; out = []
        for t in ts:
            hit = [st for st, hm, _ in day_stems(d) if hm == t]
            out += hit[:1]
            if not hit: print("  MISSING", d, t, file=sys.stderr)
        return out, f"{d} list({len(ts)})"
    _, d, a, n = s
    return [st for st, hm, _ in day_stems(d) if hm >= a][:n], f"{d} from {a} x{n}"

def waterfill(caps, n):
    caps = np.array(caps); take = np.zeros(len(caps), int)
    while take.sum() < min(n, caps.sum()):
        open_ = np.flatnonzero(take < caps); i = open_[np.argmin(take[open_])]; take[i] += 1
    return take

def spread(lst, k):
    if k >= len(lst): return list(lst)
    return [lst[int(i)] for i in np.round(np.linspace(0, len(lst) - 1, k))]

rows = []
for env, conds in SPEC.items():
    for cond, (n, specs) in conds.items():
        sess = []
        for s in specs:
            st, name = resolve(s); sess.append((name[:4], name, st))
        dates = sorted({d for d, _, _ in sess})
        date_take = waterfill([sum(len(x) for dd, _, x in sess if dd == d) for d in dates], n)
        for d, k in zip(dates, date_take):
            ss = [x for x in sess if x[0] == d]
            for (_, name, st), kk in zip(ss, waterfill([len(x[2]) for x in ss], k)):
                pick = set(spread(st, kk))
                for x in st:
                    rows.append(dict(env=env, cond=cond, y=int(HUMAN(cond)), stem=x, date=d, session=name, train=x in pick))
for env, conds in LATER.items():
    for cond, specs in conds.items():
        for s, note in specs:
            st, name = resolve(s)
            rows += [dict(env=env, cond=cond, y=int(HUMAN(cond)), stem=x, date=s[1], session=f"{name} {note}".strip(), train=False) for x in st]
D = pd.DataFrame(rows)
dup = D[D.duplicated(["env", "stem"], keep=False)]
if len(dup): print("DUPLICATE stems across conditions:\n", dup.sort_values("stem").to_string(), file=sys.stderr)
dur = {st: du for day in _cache.values() for st, _, du in day}
D["dur"] = D.stem.map(dur)
D.to_csv(sys.argv[1], index=False)
pd.set_option("display.width", 200)
print(D.groupby(["env", "cond"]).agg(found=("stem", "size"), train=("train", "sum")).to_string())
print("\ntrain by session:")
print(D[D.train].groupby(["env", "cond", "session"]).size().to_string())
print("\nnon-60s durations:", D[D.dur != 60][["stem", "dur"]].to_string() if (D.dur != 60).any() else "none")
