"""Scenario B with its own reference (run.py b5): accuracy (KPI#1; B has no object) per config and context, and the
nested picks of out/picks.json (fold env3 -> context env1+env2+env3-1, fold scnC -> env1+env2). Self-calibrated
configs do not use a reference: their B logits come from out/pred unchanged."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e as E  # noqa: E402
import configs as C  # noqa: E402

HERE = Path(__file__).resolve().parent
M = E.load_meta()
B = np.flatnonzero((M.set == "B").to_numpy())
y = M.y.to_numpy()[B]
CTX = {"scnC": ("env1", "env2"), "env3": ("env1", "env2", "env3-1")}


def logit(cfg, rs):
    if cfg.startswith("ens:"):
        parts = [logit(c, rs) for c in cfg[4:].split("+")]
        return None if any(p is None for p in parts) else np.mean(parts, 0)
    if C.get(cfg).cal in ("persc", "scalar"):
        fs = sorted((HERE / "out" / "pred_b5" / cfg).glob(f"{E.rs_name(rs)}__s*.npy"))
        return np.mean([np.load(f) for f in fs], 0) if fs else None
    fs = sorted((HERE / "out" / "pred" / cfg).glob(f"{E.rs_name(rs)}__s*.npy"))
    return np.mean([np.load(f)[B] for f in fs], 0) if fs else None


def acc(lo):
    ok = (lo > 0) == (y == 1)
    return {"acc": round(100 * ok.mean(), 1), "empty": round(100 * ok[y == 0].mean(), 1), "occupied": round(100 * ok[y == 1].mean(), 1)}


P = json.loads((HERE / "out" / "picks.json").read_text())["picks"]
res = {"nested_single": {}, "nested_top3": {}, "per_config": {}}
for ctx, rs in CTX.items():
    res["nested_single"][ctx] = acc(logit(P[ctx]["single"], rs))
    res["nested_top3"][ctx] = acc(logit(P[ctx]["topk"], rs))
for d in sorted((HERE / "out" / "pred_b5").iterdir()):
    res["per_config"][d.name] = {ctx: acc(lo) for ctx, rs in CTX.items() if (lo := logit(d.name, rs)) is not None}
(HERE / "out" / "final" / "b5.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
