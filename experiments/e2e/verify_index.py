"""fast_index vs mtk.MTKIndex on every capture of the windows: every published field equal (cyphy session, 2026-11-18).
  verify_index.py [N_PROC]   lg"""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/lg_csi/feitcsi_parser-ml"); sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e as E  # noqa: E402
import e2e_np as N  # noqa: E402
from backend import mtk  # noqa: E402

M = E.load_meta()


def job(path):
    a, b = vars(mtk.MTKIndex(Path(path))), vars(N.fast_index(path))
    if set(a) != set(b):
        return path, "keys"
    for k, x in a.items():
        y = b[k]
        if k == "path":
            continue
        if isinstance(x, np.ndarray):
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                return path, k
        elif x != y:
            return path, k
    return path, None


if __name__ == "__main__":
    paths = sorted({str(p) for p, e in zip(M.path, M.err) if str(e) in ("nan", "", "None")})
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        res = p.map(job, paths, chunksize=4)
    bad = [(q, k) for q, k in res if k]
    print(f"captures {len(res)}: every index field equal {len(res) - len(bad)}/{len(res)}; differences {bad[:10]}")
