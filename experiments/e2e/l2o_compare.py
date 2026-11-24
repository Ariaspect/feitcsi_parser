"""Inner L2O table of the feature models (out/l2o_feat.csv) in evaluate.dev_table's terms (per pair BA / K1 / K2,
averaged; per held-out room BA), next to e2e configs. Dev data only.
  l2o_compare.py CFG[,CFG...]"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as V  # noqa: E402

Fd = pd.read_csv(Path(__file__).resolve().parent / "out" / "l2o_feat.csv", dtype={"row_id": str})
rows = []
for v, d in Fd.groupby("variant"):
    per, k1, k2 = {}, [], []
    for (t, h), q in d.groupby(["train", "held"]):
        ok = (q.dec == q.y).to_numpy(); y = q.y.to_numpy(); o = q.obj.to_numpy()
        per.setdefault(h, []).append(50 * (ok[y == 1].mean() + ok[y == 0].mean()))
        k1.append(100 * ok[o == 0].mean() if (o == 0).any() else np.nan); k2.append(100 * ok[o == 1].mean() if (o == 1).any() else np.nan)
    r = {"cfg": v, "L2O_BA": np.mean([x for l in per.values() for x in l])}
    r.update({f"BA_{h}": np.mean(l) for h, l in per.items()}); r["K1"] = np.nanmean(k1); r["K2"] = np.nanmean(k2)
    rows.append(r)
T = pd.concat([pd.DataFrame(rows).set_index("cfg"),
               V.dev_table(sys.argv[1].split(",") + [a for a in sys.argv[2:]])[["L2O_BA", "BA_env1", "BA_env2", "BA_env3", "BA_env3-1", "K1", "K2"]]])
print(T.sort_values("L2O_BA", ascending=False).round(1).to_string())
