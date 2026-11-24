"""Compact table of out/final/final.json: accuracy (KPI#1 / KPI#2) per scoring set for the nested picks, every
picked config and the feature models (bench rows only)."""
import json
import sys
from pathlib import Path

R = json.loads((Path(__file__).resolve().parent / "out" / "final" / "final.json").read_text())
keys = ["LOEO", "env1", "env2", "env3", "env3-1", "scnC", "env2_later", "unused", "A|scnC", "A|env3", "B|scnC", "B|env3", "1109|env2"]


def fmt(s):
    if s is None:
        return "-"
    k2 = "-" if s["k2"] is None else s["k2"]
    return "%s (%s/%s)" % (s["acc"], s["k1"], k2)


models = {"nested_single": R["nested_single"]["summary"], "nested_top3": R["nested_topk"]["summary"]}
models.update(R["per_config"])
models.update({"F:" + c: s for c, s in R["features"].items()})
print("model".ljust(26), " | ".join(k.ljust(17) for k in keys))
for m, s in models.items():
    print(m.ljust(26), " | ".join(fmt(s.get(k)).ljust(17) for k in keys))
if len(sys.argv) > 1:
    print(json.dumps(R["nested_single"]["rate"], indent=0))
    print(json.dumps(R["nested_topk"]["rate"], indent=0))
