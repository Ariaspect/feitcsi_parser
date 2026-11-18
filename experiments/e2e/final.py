"""Final step of the end-to-end study (cyphy session, 2026-11-15/16).

  final.py picks [K]       per outer fold (env1..env3-1 and scnC): the candidates ranked by the fold's inner score
                           (models that never saw the fold's room), the single pick and the top-K ensemble; prints the
                           configs whose outer models must be trained (run.py final ...)
  final.py report OUTDIR   after run.py final: outer results of the nested picks (single, top-K ensemble), of every
                           picked config, rate check, condition table against the feature models, exports for
                           compare.py; OUTDIR/final.json
Candidates: every config in out/pred with 3+ seeds on all eight dev room sets. Selection never reads outer data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2e as E  # noqa: E402
import evaluate as V  # noqa: E402

FOLDS = list(E.ROOMS) + ["scnC"]
DEV_SETS = [rs for rs in E.room_sets() if len(rs) < 3]


def candidates() -> list[str]:
    out = []
    for p in sorted(V.PRED.iterdir()):
        if p.is_dir() and all(V.n_seeds(p.name, rs) >= 3 for rs in DEV_SETS):
            out.append(p.name)
    return out


def picks(K: int = 3) -> dict:
    C = candidates()
    res = {}
    for f in FOLDS:
        sc = sorted(((V.inner_score(c, f), c) for c in C), reverse=True)
        top = [c for _, c in sc[:K]]
        ens = "ens:" + "+".join(top)
        res[f] = {"single": sc[0][1], "single_inner": round(sc[0][0], 1), "topk": ens,
                  "topk_inner": round(V.inner_score(ens, f), 1), "ranking": [(c, round(v, 1)) for v, c in sc[:10]]}
    return res


def fold_logit(name_by_fold: dict[str, str]) -> dict[str, np.ndarray]:
    L = {}
    for f, c in name_by_fold.items():
        Lc = V.outer_logit(c)
        if f in Lc:
            L[f] = Lc[f]
    return L


def feature_rows() -> pd.DataFrame:
    R = pd.read_csv(HERE / "bench_rows.csv.gz", dtype={"row_id": str})
    return R.set_index("row_id")[["p_seq17", "p_seq13pw", "p_seq15pwu"]]


def main():
    mode = sys.argv[1]
    if mode == "picks":
        K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        P = picks(K)
        need = sorted({c for v in P.values() for c in [v["single"]] + v["topk"][4:].split("+")})
        print(json.dumps(P, indent=1))
        print("configs to train for the outer step:", ",".join(need))
        (HERE / "out" / "picks.json").write_text(json.dumps({"K": K, "picks": P, "need": need}, indent=1))
        return
    outdir = Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
    P = json.loads((HERE / "out" / "picks.json").read_text())["picks"]
    rep = {"picks": P}
    single = {f: P[f]["single"] for f in FOLDS}
    topk = {f: P[f]["topk"] for f in FOLDS}
    for name, sel in (("nested_single", single), ("nested_topk", topk)):
        D = V.assemble(fold_logit(sel))
        rep[name] = {"summary": V.summary(D), "rate": V.rate_check(D)}
        V.export(D, outdir / f"{name}_rows.csv.gz")
    configs = sorted({c for v in P.values() for c in [v["single"]] + v["topk"][4:].split("+")})
    rep["per_config"] = {}
    for c in configs:
        L = V.outer_logit(c)
        if len(L) == 5:
            D = V.assemble(L)
            rep["per_config"][c] = V.summary(D)
            V.export(D, outdir / f"{c}_rows.csv.gz")
    # condition table: nested picks vs the feature models on the same rows (bench part; fresh sets have no features)
    Ds = {"e2e_single": V.assemble(fold_logit(single)), "e2e_topk": V.assemble(fold_logit(topk))}
    Fr = feature_rows()
    base = Ds["e2e_single"]
    for col, nm in (("p_seq17", "seq17"), ("p_seq13pw", "seq13pw"), ("p_seq15pwu", "seq15pwu")):
        q = base.copy()
        p = q.row_id.map(Fr[col])
        q["logit"] = np.log(p.clip(1e-9, 1 - 1e-9) / (1 - p.clip(1e-9, 1 - 1e-9)))
        Ds[nm] = q[q.part == "bench"]
        rep.setdefault("features", {})[nm] = V.summary(Ds[nm])
    T = V.cond_table(Ds)
    T.to_csv(outdir / "conditions.csv")
    (outdir / "final.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k]["summary"] for k in ("nested_single", "nested_topk")}, indent=1))
    pd.set_option("display.width", 250)
    print(T.to_string())


if __name__ == "__main__":
    main()
