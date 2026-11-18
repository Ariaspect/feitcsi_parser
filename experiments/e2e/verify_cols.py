"""prep_cols vs x_40 (the training tensor, = prep()) on the used bins of every window (cyphy session, 2026-11-17).
  verify_cols.py [N_PROC]   lg; bins of stride 16, offsets 0 and 8 (the 3x2x3 setting) and, as a wider check, all
                            16 offsets (every bin)."""
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


def job(i):
    r = M.iloc[i]
    if str(r.err) not in ("nan", "", "None"):
        return i, None, None, int(r.live)
    a = N.prep_cols(r.path, r.t0, r.t1, NEED)
    b = N.prep_cols(r.path, r.t0, r.t1, ALL) if i % 10 == 0 else None
    x = np.asarray(X[i])
    eq = bool(np.array_equal(a[:, NEED], x[:, NEED]))
    eqall = None if b is None else bool(np.array_equal(b, x))
    return i, eq, eqall, int(r.live)


if __name__ == "__main__":
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        res = p.map(job, range(len(M)), chunksize=8)
    ok = [e for _, e, _, _ in res if e is not None]
    okall = [e for _, _, e, _ in res if e is not None]
    nl = [l for _, e, _, l in res if e is not None and l != 245]
    bad = [(i, l) for i, e, _, l in res if e is False]
    print(f"windows {len(ok)}: used bins identical {sum(ok)}/{len(ok)}; every bin (1 in 10) {sum(okall)}/{len(okall)}; "
          f"windows with live != 245: {len(nl)}; mismatches {bad[:10]}")
