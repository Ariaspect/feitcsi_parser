"""Full-dataset check that backend/tfm_phys_fast.series (the board-speed series) reproduces backend/tfm_phys.series.

Every window of a rows table from scripts/verify_tfm_phys.py (verify_rows.csv.gz: the benchmark's 1,346 + the 80
unused GT windows, 1,426) is computed both ways on this machine from the raw capture. Reported: per channel the
largest relative difference and any NaN mismatch, swing-period mismatches, the largest |w| difference, and for
seq13pw and seq15pwu the largest |P| difference, verdict flips and the accuracy tables (KPI#1 = no object, KPI#2 =
object) of both, plus the median time per window of each series.

Usage:  .venv/bin/python scripts/verify_tfm_phys_fast.py ROWS.csv.gz CAPTURES_DIR OUT_DIR [NPROC]
CAPTURES_DIR holds <stem>.bin or lg_archive/<day>/<stem>.bin; scenario C rows ("scnC:Cxxx") read minute xxx of
20261023_132301. Writes OUT_DIR/verify_fast_rows.csv.gz and OUT_DIR/verify_fast_summary.json. cyphy session 2026-11-13.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend import mtk, tfm_phys as TP, tfm_phys_fast as TF  # noqa: E402

SCN = "20261023_132301"
_IDX: dict[str, mtk.MTKIndex] = {}


def capture(caps: Path, stem: str) -> Path:
    p = caps / f"{stem}.bin"
    return p if p.exists() else caps / "lg_archive" / stem[:8] / f"{stem}.bin"


def one(job: tuple) -> dict:
    rid, path, t0, t1, room = job
    try:
        if path not in _IDX:
            _IDX.clear()                                   # one capture's index at a time per worker
            _IDX[path] = mtk.MTKIndex(Path(path))
        idx = _IDX[path]
        a = time.perf_counter(); s0 = TP.series(path, t0, t1, idx); t_ref = time.perf_counter() - a
        a = time.perf_counter(); s1 = TF.series(path, t0, t1, idx); t_fast = time.perf_counter() - a
        out = {"row_id": rid, "t_ref": t_ref, "t_fast": t_fast, "T_ref": s0["T"], "T_fast": s1["T"],
               "w_diff": abs(s0["w"] - s1["w"]), "error": ""}
        for c in TP.PHYS_CHANNELS:
            x, y = s0["S"][c], s1["S"][c]
            fx, fy = np.isfinite(x), np.isfinite(y)
            m = fx & fy
            out[f"rel_{c}"] = float((np.abs(x[m] - y[m]) / np.maximum(np.abs(x[m]), 1e-12)).max()) if m.any() else 0.0
            out[f"nan_{c}"] = int((fx != fy).sum())
        for mdl in TP.NEW_MODELS:
            out[f"p_ref_{mdl}"] = TP.judge(TP.matrix(s0["S"], mdl), mdl, room)[1]["p"]
            out[f"p_fast_{mdl}"] = TP.judge(TP.matrix(s1["S"], mdl), mdl, room)[1]["p"]
        return out
    except Exception as exc:  # noqa: BLE001 -- recorded and counted
        return {"row_id": rid, "error": repr(exc)}


def acc(q: pd.DataFrame, col: str) -> float | None:
    return None if len(q) == 0 else round(100.0 * float(((q[col] > 0.5).astype(int) == q.y).mean()), 1)


def ba(q: pd.DataFrame, col: str) -> float | None:
    ok = (q[col] > 0.5).astype(int) == q.y
    if not (q.y == 1).any() or not (q.y == 0).any():
        return None
    return round(50.0 * (float(ok[q.y == 1].mean()) + float(ok[q.y == 0].mean())), 1)


def main() -> None:
    rows_csv, caps, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    R = pd.read_csv(rows_csv, dtype={"stem": str, "row_id": str})
    jobs = []
    for r in R.itertuples():
        room = {"room": r.room, "fold": r.fold, "calibrated_with": "env3-1" if r.fold == "scnC" else r.fold}
        if r.set == "scnC":
            k = int(str(r.row_id).split(":C")[1])
            jobs.append((r.row_id, str(capture(caps, SCN)), 60.0 * k, 60.0 * k + 60.0, room))
        else:
            jobs.append((r.row_id, str(capture(caps, r.stem)), 0.0, 61.0, room))
    jobs.sort(key=lambda j: (j[1], j[2]))                    # scenario C minutes stay together per worker
    t = time.perf_counter()
    with Pool(nproc) as pool:
        res = pool.map(one, jobs, chunksize=8)
    wall = time.perf_counter() - t
    D = pd.DataFrame([r for r in res if not r["error"]])
    err = [r for r in res if r["error"]]
    M = R.merge(D, on="row_id", how="left")
    summ: dict = {"windows": int(len(R)), "computed": int(len(D)), "n_errors": len(err), "errors": err[:20],
                  "wall_s": round(wall, 1), "nproc": nproc,
                  "median_s_per_window": {"ref": float(D.t_ref.median()), "fast": float(D.t_fast.median())},
                  "T_mismatch": int((D.T_ref != D.T_fast).sum()), "max_w_diff": float(D.w_diff.max()),
                  "channels": {c: {"max_rel_diff": float(D[f"rel_{c}"].max()), "nan_mismatch": int(D[f"nan_{c}"].sum())}
                               for c in TP.PHYS_CHANNELS}, "models": {}}
    sc = M[~M.reference.astype(bool)]
    sets = {"loeo": sc[sc.set == "loeo"], "scnC": sc[sc.set == "scnC"], "env2_later": sc[sc.set == "env2_later"],
            "unused": sc[sc.set == "unused"]}
    for mdl in TP.NEW_MODELS:
        d = (D[f"p_ref_{mdl}"] - D[f"p_fast_{mdl}"]).abs()
        e = {"max_abs_dP": float(d.max()), "flips": int(((D[f"p_ref_{mdl}"] > 0.5) != (D[f"p_fast_{mdl}"] > 0.5)).sum()),
             "max_abs_dP_vs_lg_tab": float((M[f"p_fast_{mdl}"] - M[f"p_{mdl}"]).abs().max()) if f"p_{mdl}" in M else None,
             "accuracy": {}}
        for kind in ("ref", "fast"):
            col = f"p_{kind}_{mdl}"
            e["accuracy"][kind] = {k: {"n": int(len(q)), "all": acc(q, col), "kpi1": acc(q[q.obj == 0], col),
                                       "kpi2": acc(q[q.obj == 1], col), "ba": ba(q, col)} for k, q in sets.items()}
        summ["models"][mdl] = e
    out_dir.mkdir(parents=True, exist_ok=True)
    M.to_csv(out_dir / "verify_fast_rows.csv.gz", index=False)
    (out_dir / "verify_fast_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    brief = {k: summ[k] for k in ("windows", "computed", "n_errors", "wall_s", "median_s_per_window", "T_mismatch", "max_w_diff")}
    brief["max_rel_diff"] = max(v["max_rel_diff"] for v in summ["channels"].values())
    brief["nan_mismatch"] = sum(v["nan_mismatch"] for v in summ["channels"].values())
    brief["models"] = {m: {k: v for k, v in e.items() if k != "accuracy"} for m, e in summ["models"].items()}
    print(json.dumps(brief, indent=1), flush=True)


if __name__ == "__main__":
    main()
