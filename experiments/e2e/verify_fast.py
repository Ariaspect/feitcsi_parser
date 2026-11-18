"""prep_fast vs x_40 (the training tensor, = prep()) on every window, compared as float16 bit patterns (cyphy session,
2026-11-18).  verify_fast.py [N_PROC]   lg; the 30 bins of the 3x2x3 setting (offsets 0 and 8) on every window and,
as a wider check, bins 0-239 (all 16 offsets) on 1 window in 10. Also reports which windows fell back to prep_cols."""
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/lg_csi/feitcsi_parser-ml"); sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e as E  # noqa: E402
import e2e_np as N  # noqa: E402

M = E.load_meta()
X = np.load(E.OUT / "x_40.npy", mmap_mode="r")
CFG = {"stride": 16}
NEED = N.bins_used(CFG, [0, 8])
ALL = N.bins_used(CFG, range(16))
_cols = N.prep_cols
FELL = []


def _spy(*a, **k):
    FELL.append(1)
    return _cols(*a, **k)


N.prep_cols = _spy


def job(i):
    r = M.iloc[i]
    if str(r.err) not in ("nan", "", "None"):
        return i, None, None, False
    FELL.clear()
    a = N.prep_fast(r.path, r.t0, r.t1, NEED)
    fell = bool(FELL)
    x = np.asarray(X[i])
    eq = bool(np.array_equal(a[:, NEED].view(np.uint16), x[:, NEED].view(np.uint16)))
    eqall = None
    if i % 10 == 0:
        b = N.prep_fast(r.path, r.t0, r.t1, ALL)
        eqall = bool(np.array_equal(b[:, ALL].view(np.uint16), x[:, ALL].view(np.uint16)))
    return i, eq, eqall, fell


if __name__ == "__main__":
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        res = p.map(job, range(len(M)), chunksize=8)
    ok = [e for _, e, _, _ in res if e is not None]
    okall = [e for _, _, e, _ in res if e is not None]
    fell = [i for i, e, _, f in res if e is not None and f]
    bad = [i for i, e, _, _ in res if e is False]
    print(f"windows {len(ok)}: used bins bit-identical {sum(ok)}/{len(ok)}; bins 0-239 (1 in 10) {sum(okall)}/{len(okall)}; "
          f"fell back to prep_cols {len(fell)} {fell[:10]}; mismatches {bad[:10]}")
