"""Streaming e2e model (user 2026-11-18: "앞으로 e2e 모델은 현재 후보로 배포해"): sbo.B.lfhftl.st16.att.noaug on the
always-on board stream -- every 5 s the last 61 s, the same windows as the 15F streaming test (user 11-17).

Model (bundle ``backend/e2e_deploy``, made by experiments/e2e/deploy_bundle.py on lg): spectral branches lf / hf / tl
on 15-bin subsets (every 16th grid bin), test inputs 3 crops (0 / 10 / 20 s) x bin offsets 0 / 8 x seeds 0-2, person
if the mean logit > 0. ``fold`` picks the weights: the room the model never saw (env1, env2, env3, env3-1) or scnC
(env1 + env2).

The input of a window is ``e2e_np.prep_cols`` of a one-minute capture holding the same frames: the frames are the
ring's (uniform set, full width, nulls filled over the window), then the live-bin and
frame masks, the antenna ratio on the 30 used bins, ``af8.resample``, resample_poly to 40 Hz, z = r / mean - 1, float16.

Self-contained: the frame ring, the per-frame decode and the window null fill are copies of the 15F stream test's
``stream_phys.FrameRing`` / ``tfm_phys_fast.decode_raw2`` / ``fill_nulls2`` (wt-ml, uncommitted on 11-18), whose
arithmetic is ``tfm_phys_fast.decode2``'s split in two (integer decode, then the fill), so this module needs neither.

Calibration (fixed5, the training's): a room's per-bin scale = the median over its install empty minutes of
``noise_level`` (median |first difference| / sqrt 2 of the prepared minute) divided by the training constant
``glob``. Only the used bins are computed (noise_level is per bin); the others stay 1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from backend import af8, e2e_np as N, framediff, mtk

BUNDLE = Path(__file__).with_name("e2e_deploy")
META: dict[str, Any] = json.loads((BUNDLE / "meta.json").read_text())
CFG: dict[str, Any] = META["cfg"]
OFFS: list[int] = list(META["offsets"])
STARTS: list[int] = list(META["starts_idx"])
NEED = N.bins_used(CFG, OFFS)
WINDOW_S = 61.0          # the training windows: [0, 61) s of a one-minute capture
HOP_S = 5.0
# A window runs WINDOW_S from its first frame, which falls up to a frame interval after ``end - WINDOW_S``; ending
# this far behind the newest frame keeps every frame of it already in the ring.
LAG_MS = 500
T = int(round(60.0 * N.RATE))


def decode_raw(path: str | Path, index: mtk.MTKIndex, ids: np.ndarray) -> np.ndarray:
    """``tfm_phys_fast.decode2`` before the null fill: (frames, 2, width) complex128, a missing slot all NaN; full-width
    frames only. The fftshift is folded into the byte gather and the 14-bit values are sign-extended by shifts."""
    width = index.num_subcarriers
    perm = np.fft.fftshift(np.arange(width))
    cols = (2 * perm[:, None] + np.arange(2)[None, :]).ravel()
    mm = np.memmap(path, dtype="<u1", mode="r")
    out = np.empty((len(ids), 2, width), np.complex128)
    for slot in range(2):
        r_off = index._real_off[ids, slot]
        i_off = index._imag_off[ids, slot]
        present = (r_off >= 0) & (i_off >= 0)
        rows = np.flatnonzero(present)
        out[~present, slot] = np.nan
        if not rows.size:
            continue
        z = out[rows, slot]
        for off, part in ((r_off, z.real), (i_off, z.imag)):
            raw = np.asarray(mm[off[rows, None] + cols[None, :]]).view("<u2")
            part[...] = (raw << 2).view(np.int16) >> 2
        out[rows, slot] = z
    del mm
    return out


def fill_nulls(out: np.ndarray) -> np.ndarray:
    """``tfm_phys_fast.decode2``'s null fill, in place on ``decode_raw``'s output: a bin zero in every present frame of
    a slot is structural, interpolated across (a run longer than ``mtk.MAX_NULL_RUN`` or at an edge: NaN)."""
    width = out.shape[2]
    for slot in range(2):
        rows = np.flatnonzero(~np.isnan(out[:, slot, 0].real))
        if not rows.size:
            continue
        z = out[rows, slot]
        structural = np.all(z == 0, axis=0)
        for start, stop in mtk._null_runs(structural):
            left, right = start - 1, stop
            if stop - start > mtk.MAX_NULL_RUN or left < 0 or right >= width:
                z[:, start:stop] = np.nan
                continue
            span = stop - start + 1
            for k in range(start, stop):
                mu = (k - left) / span
                z[:, k] = z[:, left] * (1 - mu) + z[:, right] * mu
        out[rows, slot] = z
    return out


class Ring:
    """Decoded frames of a growing capture (or several, one after another), the last ``keep_s`` seconds."""

    def __init__(self, keep_s: float = WINDOW_S + 2 * HOP_S) -> None:
        self.keep_ms = int(keep_s * 1000)
        self.width = 0
        self._chunks: list[dict[str, Any]] = []

    def ingest(self, path: str | Path, index: mtk.MTKIndex, ids: np.ndarray) -> int:
        """Frames ``ids`` of ``index`` (new ones, in file order); returns how many were added."""
        ids = np.asarray(ids, dtype=np.int64)
        if not ids.size:
            return 0
        width = int(index.num_subcarriers)
        if self.width and width != self.width:
            raise ValueError(f"width changed {self.width} -> {width}")
        self.width = width
        full = np.asarray(index._bins)[ids] == width
        raw = np.full((ids.size, 2, width), np.nan, np.complex64)   # 14-bit integers: exact in float32
        if full.any():
            raw[full] = decode_raw(Path(path), index, ids[full])
        self._chunks.append({"stamp": np.asarray(index._stamps)[ids].astype(np.int64),
                             "rx": np.asarray(index.num_rx_arr)[ids], "tx": np.asarray(index.num_tx_arr)[ids],
                             "mac": np.array(index.source_macs, dtype=object)[ids], "full": full, "raw": raw})
        last = self.last_stamp()
        while len(self._chunks) > 1 and self._chunks[0]["stamp"][-1] < last - self.keep_ms:
            self._chunks.pop(0)
        return int(ids.size)

    def last_stamp(self) -> int | None:
        return int(self._chunks[-1]["stamp"].max()) if self._chunks else None

    def nbytes(self) -> int:
        return sum(c["raw"].nbytes for c in self._chunks)

    def frames(self, end_stamp: int, window_s: float = WINDOW_S) -> dict[str, Any] | None:
        """The frames of the window ending at ``end_stamp`` (ms): those from the first one at or after end - window,
        ``framediff._uniform_selection`` inside the window (dominant MAC, STRICT_MIMO when present, else the
        commonest mode), full width; nulls filled over the window's frames."""
        C = self._chunks
        if not C:
            return None
        st = np.concatenate([c["stamp"] for c in C])
        lo = end_stamp - int(round(window_s * 1000))
        first = np.flatnonzero(st >= lo)
        if not first.size:
            return None
        i0 = int(first[0])
        tr = (st - st[i0]).astype(np.float64) / 1e3
        win = (np.arange(st.size) >= i0) & (tr < window_s)
        rx = np.concatenate([c["rx"] for c in C]); tx = np.concatenate([c["tx"] for c in C])
        mac = np.concatenate([c["mac"] for c in C]); full = np.concatenate([c["full"] for c in C])
        counts: dict[Any, int] = {}
        for v in mac[win]:
            counts[v] = counts.get(v, 0) + 1
        dom = max(counts, key=lambda k: counts[k]) if counts else None
        base = win & (mac == dom) if dom else win
        strict = base & (rx == framediff.STRICT_MIMO[0]) & (tx == framediff.STRICT_MIMO[1])
        if strict.any():
            sel = strict
        else:
            pairs, cnt = np.unique(np.stack([rx[base], tx[base]], 1), axis=0, return_counts=True)
            if not cnt.size:
                return None
            m = pairs[cnt.argmax()]
            sel = base & (rx == m[0]) & (tx == m[1])
        sel &= full
        ids = np.flatnonzero(sel)
        if ids.size < 2:
            return None
        D = np.empty((ids.size, 2, self.width), np.complex128)
        o = 0; k = 0
        for c in C:
            n = len(c["stamp"]); j = ids[(ids >= o) & (ids < o + n)] - o
            D[k:k + j.size] = c["raw"][j]; k += j.size; o += n
        return {"D": fill_nulls(D), "tt": tr[ids], "first_stamp": int(st[i0]), "frames": int(ids.size),
                "span_s": float(tr[ids[-1]]),
                "complete": bool(st.max() >= st[i0] + int(round(window_s * 1000)))}


def prep_frames(D: np.ndarray, tt: np.ndarray, need=NEED) -> np.ndarray:
    """``e2e_np.prep_cols`` on decoded, null-filled frames (``decode2``'s output) and their times: (2400, 245, 2)
    float16, the bins ``need`` filled, the others 0."""
    fin = np.isfinite(D)
    live = fin.all(axis=1).mean(axis=0) > af8.LIVE_FRACTION
    rows = np.flatnonzero(fin[:, :, live].all(axis=(1, 2)) & (D[:, 0, :] != 0)[:, live].all(axis=1))
    lv = np.flatnonzero(live); nl = lv.size
    need = np.asarray(need)
    if nl == N.N_BINS:
        src_cols = need
    else:
        src = np.linspace(0, 1, nl); dst = np.linspace(0, 1, N.N_BINS)[need]
        j = np.clip(np.searchsorted(src, dst, side="right") - 1, 0, nl - 2)
        src_cols = np.unique(np.concatenate([j, j + 1]))
    cols = lv[src_cols]
    r = D[rows, 1][:, cols] / D[rows, 0][:, cols]
    g, fab, fs = af8.resample(r, tt[rows])
    y = N.to_rate(g, fs)
    if len(y) < T:
        y = np.concatenate([y, np.repeat(y[-1:], T - len(y), axis=0)])
    y = y[:T]
    z = y / y.mean(axis=0) - 1.0
    if nl != N.N_BINS:
        full = np.zeros((T, nl), z.dtype); full[:, src_cols] = z
        src = np.linspace(0, 1, nl); dst = np.linspace(0, 1, N.N_BINS)
        z = np.stack([np.interp(dst, src, q.real) + 1j * np.interp(dst, src, q.imag) for q in full])[:, need]
    out = np.zeros((T, N.N_BINS, 2), np.float16)
    out[:, need] = np.stack([z.real, z.imag], axis=-1).astype(np.float16)
    return out


def scale_from(minutes: list[np.ndarray], need=NEED) -> np.ndarray:
    """The room's per-bin divisor from its install empty minutes (prepared arrays): the training's fixed5 formula,
    median over the minutes of noise_level, over ``glob``; bins outside ``need`` stay 1."""
    need = np.asarray(need)
    L = np.stack([N.noise_level(np.asarray(a)[:, need]) for a in minutes])
    s = np.ones(N.N_BINS, np.float32)
    s[need] = (np.median(L, axis=0) / META["glob"]).astype(np.float32)
    return s


class Model:
    """The bundle's nets for one fold; ``judge`` = e2e_np.judge_fast (one core) or judge over a Pool."""

    def __init__(self, fold: str, cores: list[int] | None = None) -> None:
        if fold not in META["folds"]:
            raise ValueError(f"fold {fold!r} not in {sorted(META['folds'])}")
        self.fold = fold
        self.nets = [N.FastNet(N.Net(CFG, dict(np.load(BUNDLE / f"{fold}__s{s}.npz")))) for s in META["seeds"]]
        self.pool = N.Pool(cores, self.nets) if cores and len(cores) > 1 else None

    def logit(self, arr: np.ndarray, scale: np.ndarray, timings: dict[str, float] | None = None) -> float:
        if self.pool is None:
            return N.judge_fast(arr, scale, CFG, self.nets, STARTS, OFFS, timings=timings)
        return _judge_pool(self.pool, arr, scale)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()


def _judge_pool(pool: "N.Pool", arr: np.ndarray, scale: np.ndarray) -> float:
    """judge_fast's logit with the six test inputs split over the pool's processes (per input on the board's numpy,
    see e2e_np.judge_par)."""
    inputs = [(s0, o) for s0 in STARTS for o in OFFS]
    ig = [g for g in np.array_split(np.arange(len(inputs)), len(pool.cores)) if g.size]
    vals = arr[:, NEED]
    L = np.concatenate(pool.map("inputs", [(CFG, scale, NEED, vals, [inputs[i] for i in g]) for g in ig]), axis=1)
    tot = np.zeros(L.shape[0])
    for s in range(L.shape[0]):
        tot[s] += L[s].sum()
    return float(tot.sum() / (len(inputs) * L.shape[0]))


def window_input(ring: Ring, end_stamp: int | None, timings: dict[str, float] | None = None
                 ) -> tuple[np.ndarray, dict[str, Any]] | None:
    """The prepared (2400, 245, 2) input of the window ending at ``end_stamp`` (None: ``LAG_MS`` before the newest
    frame) and its extras; None if empty or not every frame of it has arrived."""
    tm = timings if timings is not None else {}
    a = time.perf_counter()
    last = ring.last_stamp()
    if last is None:
        return None
    W = ring.frames(last - LAG_MS if end_stamp is None else end_stamp)
    b = time.perf_counter(); tm["window"] = b - a
    if W is None or not W["complete"]:
        return None
    arr = prep_frames(W["D"], W["tt"])
    tm["prep"] = time.perf_counter() - b
    return arr, {k: W[k] for k in ("first_stamp", "frames", "span_s")}
