"""How much the training augmentations of the e2e candidate contribute (cyphy session, 2026-11-17; user: "증강이 정확도에
얼마나 기여하는지 확인해봐").

Base: sbo.B.lfhftl.st16.att (random 40-s crop, random bin offset, time reversal 50 %, conjugation 50 %, global phase
rotation -- the last is a no-op for this spectral-only model). Variants switch one thing off, fitted by
  tta_study.py fit <variants> 4 5      (same seeds 0-4, same room sets, same rows, B with its own reference)
  .noaug   reversal + conjugation + rotation off      .norev  reversal off      .noconj  conjugation off
  .fcrop   training crop always the middle 40 s       .foff   training bin offset always 0
  .noaug.fcrop.foff   everything off
Scored at the deployment test setting 3 crops x 2 offsets x 3 seeds; fcrop / foff also at the test inputs they were
trained on (middle crop / offset 0). Paired session bootstrap against the base (B = 2000).
  aug_study.py OUT.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fair_compare as FC  # noqa: E402

BASE = "sbo.B.lfhftl.st16.att"
VARS = [("base (모두 켬)", BASE, (3, 2)), ("뒤집기 끔 (noaug)", BASE + ".noaug", (3, 2)),
        ("시간 뒤집기만 끔", BASE + ".norev", (3, 2)), ("켤레만 끔", BASE + ".noconj", (3, 2)),
        ("위치 고정", BASE + ".fcrop", (3, 2)), ("위치 고정, 판정도 가운데 1개", BASE + ".fcrop", (1, 2)),
        ("묶음 고정", BASE + ".foff", (3, 2)), ("묶음 고정, 판정도 offset 0", BASE + ".foff", (3, 1)),
        ("전부 끔", BASE + ".noaug.fcrop.foff", (3, 2)), ("전부 끔, 판정도 학습 입력만", BASE + ".noaug.fcrop.foff", (1, 1))]
OUTER_ROWS = ["Env 1", "Env 2", "Env 3", "Env 3-1", "Total", "Scenario A", "Scenario B", "Scenario C", "Env 2 + 110"]


def inner_frame(inn: dict) -> pd.DataFrame:
    """One row per (pair, scored window) with ok_<name>; room = pair, session = pair|session (for FC.boot)."""
    Mi = FC.M.set_index("row_id")
    out = []
    first = next(iter(inn.values()))
    for key in sorted(first):
        idx = first[key].index
        q = pd.DataFrame({"obj": Mi.obj.reindex(idx).to_numpy(), "y": Mi.y.reindex(idx).to_numpy(),
                          "pair": "+".join(key[0]) + ">" + key[1],
                          "session": FC.ROWS.set_index("row_id").session.reindex(idx).astype(str).to_numpy()}, index=idx)
        for n, d in inn.items():
            q[f"ok_{n}"] = (d[key].reindex(idx).to_numpy() > 0) == (q.y.to_numpy() == 1)
        out.append(q.reset_index(names="row_id"))
    D = pd.concat(out, ignore_index=True)
    D["room"] = D.pair
    D["session"] = D.pair + "|" + D.session
    return D


def main():
    out = Path(sys.argv[1])
    res = {"variants": {}}
    rows, inn, ens5 = {}, {}, {}
    for name, cfg, (nc, no) in VARS:
        e = FC.e2e_ens(cfg, nc, no, seeds=range(3))
        D = FC.e2e_rows(e)
        rows[name] = D
        inn[name] = FC.e2e_inner(e)
        ens5[name] = (cfg, nc, no)
    D0 = rows[VARS[0][0]]
    S = FC.subsets(D0)
    J = D0[["room", "obj", "y", "set", "part"]].copy()
    J["session"] = FC.sessions(D0)
    for name, D in rows.items():
        J[f"ok_{name}"] = ((D.logit.reindex(J.index) > 0) == (J.y == 1)).to_numpy()
    I = inner_frame(inn)
    base = VARS[0][0]
    for name, cfg, (nc, no) in VARS:
        v = {"cfg": cfg, "test": f"{nc}x{no}x3"}
        v["inner"] = FC.inner_scores(inn[name])
        v["table"] = {k: FC.kpi(J[m][f"ok_{name}"].to_numpy(), J[m].obj.to_numpy()) for k, m in S.items()}
        v["inner_groups"] = FC.group_acc(inn[name])
        if name != base:
            v["boot_inner"] = FC.boot(I, pd.Series(True, index=I.index), name, base)
            v["boot_total"] = FC.boot(J, S["Total"], name, base)
            v["boot_C"] = FC.boot(J, S["Scenario C"], name, base)
            v["boot_env2_110"] = FC.boot(J, S["Env 2 + 110"], name, base)
        # seed draw: every 3-of-5 subset, inner BA and Total Overall
        ba, tot = [], []
        for sub in itertools.combinations(range(5), 3):
            e = FC.e2e_ens(cfg, nc, no, seeds=sub)
            ba.append(FC.inner_scores(FC.e2e_inner(e))["BA"])
            Dk = FC.e2e_rows(e).reindex(J.index)
            tot.append(100 * ((Dk.logit > 0) == (J.y == 1))[S["Total"]].mean())
        v["seeds_inner_BA"] = [min(ba), max(ba)]
        v["seeds_total"] = [round(min(tot), 1), round(max(tot), 1)]
        res["variants"][name] = v
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    report(res)


def report(res):
    V = res["variants"]
    print("== inner (12 pairs) and seed ranges")
    for n, v in V.items():
        b = v.get("boot_inner")
        bs = "" if b is None else f" | vs base pooled acc {b['diff']['all']} {b['ci'].get('all')} K1 {b['diff']['k1']} {b['ci'].get('k1')} K2 {b['diff']['k2']} {b['ci'].get('k2')}"
        print(f"{n:30s} {v['test']}  BA {v['inner']['BA']} acc {v['inner']['acc']} K1 {v['inner']['k1']} K2 {v['inner']['k2']}"
              f"  seeds BA {v['seeds_inner_BA']} Total {v['seeds_total']}{bs}")
    print("\n== outer table Overall (K1/K2)")
    print("variant".ljust(30) + " ".join(r.ljust(17) for r in OUTER_ROWS))
    for n, v in V.items():
        print(n.ljust(30) + " ".join(f"{v['table'][r]['all']} ({v['table'][r]['k1']}/{v['table'][r]['k2']})".ljust(17) for r in OUTER_ROWS))
    print("\n== outer bootstrap vs base: Total, C, Env2+110 (Overall diff [CI])")
    for n, v in V.items():
        if "boot_total" in v:
            print(f"{n:30s} Total {v['boot_total']['diff']['all']} {v['boot_total']['ci'].get('all')} | "
                  f"C {v['boot_C']['diff']['all']} {v['boot_C']['ci'].get('all')} | "
                  f"E2+110 {v['boot_env2_110']['diff']['all']} {v['boot_env2_110']['ci'].get('all')}")
    print("\n== inner condition groups")
    names = [g for g, _, _ in FC.GROUPS[:10]]
    print("variant".ljust(30) + " ".join(g[:7].ljust(8) for g in names))
    for n, v in V.items():
        print(n.ljust(30) + " ".join(f"{v['inner_groups'].get(g, float('nan')):6.1f}  " for g in names))


if __name__ == "__main__":
    main()
