"""Merge the Physics tab's display-only results into backend/tfm_phys_results.json.

  python scripts/tfm_phys_results.py --verify verify_summary.json [--lg lg.json --lg-machine TEXT --lg-note TEXT]
                                     [--board board.json --board-machine TEXT --board-note TEXT]

--verify: scripts/verify_tfm_phys.py's summary (every labelled window through the tab's code). --lg / --board:
scripts/bench_tfm_phys_board.py's output on that machine; the board's probabilities are compared with lg's on the same
windows. A part not given keeps its value from the existing file. cyphy session 2026-11-12.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "backend" / "tfm_phys_results.json"
STAGES = ["total seq13pw", "total seq15pwu", "both physics models, one series", "total base 17",
          "total seq13pw, reference arithmetic", "total seq15pwu, reference arithmetic", "index",
          "series seq13pw", "calibrate + forward seq13pw", "series seq15pwu", "calibrate + forward seq15pwu",
          "  seq15pwu series: decode", "  seq15pwu series: steps + swing period", "  seq15pwu series: plf",
          "  seq15pwu series: swing share", "  seq15pwu series: FarSense", "  seq15pwu series: reflection (g15_res, u_g15v)",
          "  seq15pwu series: breathing lines (bs)", "series base 17", "calibrate + forward base"]


def timing(path: str, machine: str, note: str, ref: dict | None = None) -> dict:
    d = json.loads(Path(path).read_text())
    rows = d["rows"]
    stages = [s for s in STAGES if all(s in r["t"] for r in rows)]
    out = {"machine": f"{machine} (python {d['info']['python']}, numpy {d['info']['numpy']}, cores {d['info']['affinity']})",
           "note": note, "reps": d["info"]["reps"], "stages": stages, "rows": rows,
           "median": {s: float(np.median([r["t"][s] for r in rows])) for s in stages},
           "max": {s: float(np.max([r["t"][s] for r in rows])) for s in stages}, "p_match": None}
    if ref is not None:
        lg = {r["stem"]: r["p"] for r in ref["rows"]}
        common = [r for r in rows if r["stem"] in lg]
        diffs = {m: max(abs(r["p"][m] - lg[r["stem"]][m]) for r in common) for m in rows[0]["p"]} if common else {}
        out["p_match"] = (f"P vs lg on the same {len(common)} windows: max |ΔP| " +
                          ", ".join(f"{m} {v:.1e}" for m, v in diffs.items()) + ".") if common else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify"); ap.add_argument("--lg"); ap.add_argument("--lg-machine", default="lg"); ap.add_argument("--lg-note", default="")
    ap.add_argument("--board"); ap.add_argument("--board-machine", default="board"); ap.add_argument("--board-note", default="")
    a = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if a.verify:
        v = json.loads(Path(a.verify).read_text())
        res["verification"] = {k: v[k] for k in ("windows", "computed", "n_errors", "wall_s", "tab_room_lookup_ok", "references_n",
                                                 "sets_n", "channels", "probabilities", "accuracy", "strata_loeo", "conditions")}
    lg_raw = json.loads(Path(a.lg).read_text()) if a.lg else None
    if a.lg:
        res["lg"] = timing(a.lg, a.lg_machine, a.lg_note)
    if a.board:
        res["board"] = timing(a.board, a.board_machine, a.board_note, lg_raw)
    res["generated"] = str(date.today())
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
