"""Raw-CSI input tensors for the end-to-end study (cyphy session, 2026-11-15). No feature extraction.

Per window: the frames the production series uses (framediff._uniform_selection, decode2, live bins = both chains finite
in > 90 % of frames, frames finite on every live bin), the antenna ratio r = h1 / h0 on EVERY live bin (the series use
every 8th), linearly interpolated onto the capture's own uniform grid at its median frame rate (af8.resample, as in
production: no line in the noise envelope, see memory resampling-rate-fingerprint), then band-limited to a common
rate with scipy resample_poly (never a second linear re-grid between rates).

  probe              -> out/probe.csv: per window fs, frames, live bins, fabricated share (no arrays)
  extract_gated RATE [N] -> out/x_<RATE>g.npy, the direct-path-removed ratio (gated_ratio), same rows and grid
  extract RATE [N]   -> out/x_<RATE>.npy (windows, T, bins, 2) float16 of z = r / mean_t(r) - 1 (re, im),
                        out/meta_<RATE>.csv (row order), N processes
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, "/home/lg_csi/feitcsi_parser-ml")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from backend import af8, framediff, mtk, tfm_phys_fast as TF  # noqa: E402
import jobs as J  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
_IDX: dict[str, mtk.MTKIndex] = {}
SECONDS = 60.0


def index(path: str) -> mtk.MTKIndex:
    if path not in _IDX:
        _IDX.clear()
        _IDX[path] = mtk.MTKIndex(Path(path))
    return _IDX[path]


def ratio(path: str, t0: float, t1: float):
    idx = index(path)
    t = np.asarray(idx.times, dtype=float)
    tr = t - t[0]
    sel = framediff._uniform_selection(idx, None, None)
    ids = np.flatnonzero(np.asarray(sel["mask"], bool) & (tr >= t0) & (tr < t1))
    if ids.size < 100:
        raise ValueError(f"{ids.size} frames")
    D = TF.decode2(Path(path), idx, ids)
    fin = np.isfinite(D)
    live = fin.all(axis=1).mean(axis=0) > af8.LIVE_FRACTION
    rows = np.flatnonzero(fin[:, :, live].all(axis=(1, 2)) & (D[:, 0, :] != 0)[:, live].all(axis=1))
    r = D[rows, 1][:, live] / D[rows, 0][:, live]
    g, fab, fs = af8.resample(r, t[ids][rows])
    return g, fab, fs, live, D.shape[2]


def gated_ratio(path: str, t0: float, t1: float):
    """Chain 1 with the direct path removed in the delay domain (af8_env1.direct_path_gated: IFFT zero-padded x4,
    taps within +-1.5 of the strongest zeroed, FFT back) on every live bin, over chain 0 -- the reflection ratio the
    g15 channel is built on, here as a full (time x bin) signal."""
    from backend import af8_env1 as AE
    idx = index(path)
    t = np.asarray(idx.times, dtype=float)
    tr = t - t[0]
    sel = framediff._uniform_selection(idx, None, None)
    ids = np.flatnonzero(np.asarray(sel["mask"], bool) & (tr >= t0) & (tr < t1))
    D = TF.decode2(Path(path), idx, ids)
    fin = np.isfinite(D)
    live = fin.all(axis=1).mean(axis=0) > af8.LIVE_FRACTION
    rows = np.flatnonzero(fin[:, :, live].all(axis=(1, 2)) & (D[:, 0, :] != 0)[:, live].all(axis=1))
    full1 = D[rows, 1, :].copy()
    full1[:, ~live] = 0
    L = full1.shape[1] * AE.GATE_PAD
    h = np.fft.ifft(full1, n=L, axis=1)
    pk = (np.abs(h) ** 2).argmax(axis=1)
    d = (np.arange(L)[None, :] - pk[:, None] + L // 2) % L - L // 2
    h = np.where(np.abs(d) < AE.GATE_TAPS * AE.GATE_PAD, 0, h)
    G = np.fft.fft(h, axis=1)[:, : full1.shape[1]]
    r = G[:, live] / D[rows, 0][:, live]
    g, fab, fs = af8.resample(r, t[ids][rows])
    return g, fab, fs, live, D.shape[2]


def probe_one(job: dict) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g, fab, fs, live, nb = ratio(job["path"], job["t0"], job["t1"])
        return dict(row_id=job["row_id"], fs=fs, n=len(g), live=int(live.sum()), width=nb, fab=float(fab.mean()),
                    first_live=int(np.flatnonzero(live)[0]), last_live=int(np.flatnonzero(live)[-1]), err="")
    except Exception as e:  # noqa: BLE001
        return dict(row_id=job["row_id"], err=repr(e))


def duration(path: Path) -> tuple[str, float]:
    try:
        t = np.asarray(mtk.MTKIndex(path).times, float)
        return path.stem, float(t[-1] - t[0])
    except Exception:  # noqa: BLE001
        return path.stem, -1.0


def all_jobs(nproc: int) -> list[dict]:
    files = J.unlab_files()
    with Pool(nproc) as pl:
        dur = dict(pl.map(duration, files, chunksize=4))
    return J.bench() + J.fresh() + J.unlab(dur)


def to_rate(g: np.ndarray, fs: float, rate: float) -> np.ndarray:
    fr = Fraction(rate / fs).limit_denominator(200)
    x = g - g.mean(axis=0)
    y = resample_poly(x, fr.numerator, fr.denominator, axis=0, padtype="line")
    return y + g.mean(axis=0)


def extract_one(args) -> dict:
    job, rate, nbins, kind = args
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g, fab, fs, live, nb = (gated_ratio if kind == "g" else ratio)(job["path"], job["t0"], job["t1"])
            y = to_rate(g, fs, rate)
        T = int(round(SECONDS * rate))
        if len(y) < T:
            y = np.concatenate([y, np.repeat(y[-1:], T - len(y), axis=0)])
        y = y[:T]
        z = y / y.mean(axis=0) - 1.0
        if z.shape[1] != nbins:            # fixed width: resample the bin axis (only if a capture differs)
            src = np.linspace(0, 1, z.shape[1]); dst = np.linspace(0, 1, nbins)
            z = np.stack([np.interp(dst, src, row.real) + 1j * np.interp(dst, src, row.imag) for row in z])
        arr = np.stack([z.real, z.imag], axis=-1).astype(np.float16)
        return dict(row_id=job["row_id"], arr=arr, fs=fs, live=int(live.sum()), fab=float(fab.mean()), err="")
    except Exception as e:  # noqa: BLE001
        return dict(row_id=job["row_id"], err=repr(e))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    mode = sys.argv[1]
    if mode == "probe":
        nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        jl = all_jobs(nproc)
        pd.DataFrame(jl).to_csv(OUT / "jobs.csv", index=False)
        jl.sort(key=lambda j: (j["path"], j["t0"]))
        t = time.time()
        with Pool(nproc) as pl:
            P = pl.map(probe_one, jl, chunksize=8)
        P = pd.DataFrame(P)
        P.to_csv(OUT / "probe.csv", index=False)
        print("jobs", len(jl), "errors", int((P.err != "").sum()), f"{time.time() - t:.0f} s", flush=True)
        return
    rate = float(sys.argv[2]); nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    nbins = 245
    kind = "g" if mode == "extract_gated" else ""
    jl = pd.read_csv(OUT / "jobs.csv", dtype={"row_id": str, "key": str}).to_dict("records")
    jl.sort(key=lambda j: (j["path"], j["t0"]))
    T = int(round(SECONDS * rate))
    X = np.lib.format.open_memmap(OUT / f"x_{rate:g}{kind}.npy", mode="w+", dtype=np.float16, shape=(len(jl), T, nbins, 2))
    meta = []
    t = time.time()
    with Pool(nproc) as pl:
        for i, r in enumerate(pl.imap(extract_one, [(j, rate, nbins, kind) for j in jl], chunksize=4)):
            if r["err"] == "":
                X[i] = r.pop("arr")
            meta.append(dict(jl[i], **{k: v for k, v in r.items() if k != "row_id"}))
            if i % 200 == 0:
                print(i, f"{time.time() - t:.0f} s", flush=True)
    X.flush()
    pd.DataFrame(meta).to_csv(OUT / f"meta_{rate:g}{kind}.csv", index=False)
    print("done", len(jl), "errors", sum(m["err"] != "" for m in meta), f"{time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
