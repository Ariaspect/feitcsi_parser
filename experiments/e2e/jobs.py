"""Window list for the end-to-end study (cyphy session, 2026-11-15).

Four parts, each window = (part, row_id, capture path, t0, t1 relative to the capture's first frame, labels):
  bench   the 1,426 windows of the Physics-tab check (bench_rows.csv.gz = scratchpad phys/verify_rows.csv.gz):
          LOEO 1,097 (20 fixed5 references among them), scnC 149, env2_later 100, unused 80
  fresh   scenario A (37 min) / B (15 min) of 20261022 (env3 room) and the 7 captures of 11/09 (env2), labels as in
          ml_lab scripts/cy_phys_fresh.py; never used for any choice in this study
  unlab   unlabelled minutes of single-room days 10/01-10/23 (rule of ml_lab scripts/cy_unlab2_extract.py), not a
          labelled minute, not scenario A / B / C; days >= 10/30 (env2_later test days) and 09/xx (other set-up) left out
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CAP = Path("/home/lg_csi/lg_csi_captures")
LAB = "/home/lg_csi/ml_lab/data"
HERE = Path(__file__).resolve().parent
SCN = "20261023_132301"
SKIP = {"20261022_172809", "20261022_183500", SCN}


def capture(stem: str) -> Path:
    return CAP / stem[:8] / f"{stem}.bin"


def bench() -> list[dict]:
    R = pd.read_csv(HERE / "bench_rows.csv.gz", dtype={"stem": str, "row_id": str})
    out = []
    for r in R.itertuples():
        base = dict(part="bench", row_id=r.row_id, set=r.set, room=r.room, fold=r.fold, cond=r.cond, y=int(r.y),
                    obj=int(r.obj), stratum=r.stratum, reference=bool(r.reference))
        if r.set == "scnC":
            k = int(str(r.row_id).split(":C")[1])
            out.append(dict(base, path=str(capture(SCN)), t0=60.0 * k, t1=60.0 * k + 60.0, key=SCN))
        else:
            out.append(dict(base, path=str(capture(r.stem)), t0=0.0, t1=61.0, key=r.stem))
    return out


def fresh() -> list[dict]:
    sys.path.insert(0, "/home/lg_csi/ml_lab/scripts")
    argv = sys.argv
    import cy_tfm_fine as F, lgg_scn_ab as AB  # noqa: E401
    sys.argv = argv
    out = []
    WA = pd.read_csv(f"{LAB}/lgg_scnA_41.csv")
    objA = WA.obj.to_numpy(int)
    for i, w in enumerate(WA.w.astype(int)):
        out.append(dict(part="fresh", row_id=f"A:{w}", set="A", room="env3", fold="env3", cond="scnA", y=int(F.yA[i]),
                        obj=int(objA[i]), stratum="", reference=False, path=str(capture("20261022_172809")),
                        t0=60.0 * w, t1=60.0 * w + 60.0, key="20261022_172809"))
    bl = AB.b_labels()
    objB = bl.obj.to_numpy(int) if "obj" in bl else np.zeros(15, int)
    for w in range(15):
        out.append(dict(part="fresh", row_id=f"B:{w}", set="B", room="env3", fold="env3", cond="scnB", y=int(F.yB[w]),
                        obj=int(objB[w]), stratum="", reference=False, path=str(capture("20261022_183500")),
                        t0=60.0 * w, t1=60.0 * w + 60.0, key="20261022_183500"))
    for s in ["210454", "210616", "211825", "212330", "212836", "213341", "213846"]:
        m = json.load(open(CAP / "20261109" / f"20261109_{s}_meta.json"))
        out.append(dict(part="fresh", row_id=f"1109:{s}", set="1109", room="env2", fold="env2", cond="1109",
                        y=int(m["presence"] == "occupied"), obj=0, stratum="", reference=False,
                        path=str(CAP / "20261109" / f"20261109_{s}.bin"), t0=0.0, t1=61.0, key=f"20261109_{s}"))
    return out


def unlab(durations: dict[str, float]) -> list[dict]:
    D = pd.read_csv("/home/lg_csi/feitcsi_parser-ml/data/ml_heads_train.csv")[["key", "ds"]]
    lab = D[D.ds != "scnC"].assign(day=lambda d: d.key.str[:8])
    rooms = lab.groupby("day").ds.agg(lambda s: sorted(set(s)))
    single = {d: r[0] for d, r in rooms.items() if len(r) == 1 and "20261001" <= d <= "20261023"}
    known = set(D.key) | set(pd.read_csv(HERE / "bench_rows.csv.gz", dtype={"stem": str}).stem)
    out = []
    for day, room in sorted(single.items()):
        for b in sorted((CAP / day).glob("*.bin")):
            if b.stem in known or b.stem in SKIP or b.stem not in durations:
                continue
            dur = durations[b.stem]
            for w in range(int((dur - 59.0) // 60.0) + 1 if dur >= 59.0 else 0):
                out.append(dict(part="unlab", row_id=f"U:{b.stem}:{w}", set="unlab", room=room, fold=room, cond="",
                                y=-1, obj=-1, stratum="", reference=False, path=str(b), t0=60.0 * w,
                                t1=60.0 * w + 61.0, key=b.stem))
    return out


def unlab_files() -> list[Path]:
    D = pd.read_csv("/home/lg_csi/feitcsi_parser-ml/data/ml_heads_train.csv")[["key", "ds"]]
    lab = D[D.ds != "scnC"].assign(day=lambda d: d.key.str[:8])
    rooms = lab.groupby("day").ds.agg(lambda s: sorted(set(s)))
    single = [d for d, r in rooms.items() if len(r) == 1 and "20261001" <= d <= "20261023"]
    known = set(D.key) | set(pd.read_csv(HERE / "bench_rows.csv.gz", dtype={"stem": str}).stem)
    return [b for d in sorted(single) for b in sorted((CAP / d).glob("*.bin")) if b.stem not in known | SKIP]
