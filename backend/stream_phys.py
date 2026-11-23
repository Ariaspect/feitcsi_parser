"""Streaming seq15pwu / seq13pw (user 2026-11-17, full deployment test): the board captures all the time and every
5 s judges the last minute, keeping what it can in memory instead of reading a capture file per judgment.

A window here is what a one-minute capture was in training: the frames from the first one at or after
``t_end - WINDOW_S`` up to ``WINDOW_S`` past it, times counted from that first frame, the uniform set (dominant MAC,
``framediff.STRICT_MIMO`` when present, full width) chosen inside the window, structural nulls filled over the
window's frames. On that, ``tfm_phys_fast.series_grid`` and the model's calibration + 5-seed forward.

What is kept between judgments (``FrameRing``): each frame's stamp, MIMO mode, MAC, width and its decoded CSI before
the null fill (``tfm_phys_fast.decode_raw2``), decoded once as it arrives; frames older than the window are dropped. The CSI is
kept as complex64: before the null fill it is 14-bit integers, exact in float32.
Everything that depends on the whole window (uniform set, null fill, live bins, swing period, subspaces, FarSense,
breathing lines) is recomputed per judgment, so a judgment equals ``tfm_phys_fast.series`` on a one-minute capture
holding the same frames.

Calibration is the deployment's own: ``Reference`` from install empty minutes of the room (fixed5: five), the
model's log channels logged, median / MAD x 1.4826 per channel over every second, shrunk halfway to the fold's pool
(``tfm_phys.calibrate``'s arithmetic with the room's stats in place of the logged references).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from backend import framediff, mtk, tfm_phys as TP, tfm_phys_fast as TF

WINDOW_S = 61.0          # the training windows: [0, 61) s of a one-minute capture, 61 one-second columns
HOP_S = 5.0
# A window runs WINDOW_S from its first frame, which falls up to a frame interval after ``end - WINDOW_S``; ending
# this far behind the newest frame keeps every frame of it already in the ring.
LAG_MS = 500


class FrameRing:
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
        bins = np.asarray(index._bins)[ids]
        full = bins == width
        raw = np.full((ids.size, 2, width), np.nan, np.complex64)   # 14-bit integers: exact in float32
        if full.any():
            raw[full] = TF.decode_raw2(Path(path), index, ids[full])
        macs = np.array(index.source_macs, dtype=object)[ids]
        self._chunks.append({"stamp": np.asarray(index._stamps)[ids].astype(np.int64),
                             "rx": np.asarray(index.num_rx_arr)[ids], "tx": np.asarray(index.num_tx_arr)[ids],
                             "mac": macs, "full": full, "raw": raw})
        self._prune()
        return int(ids.size)

    def _prune(self) -> None:
        last = self.last_stamp()
        while len(self._chunks) > 1 and self._chunks[0]["stamp"][-1] < last - self.keep_ms:
            self._chunks.pop(0)

    def last_stamp(self) -> int | None:
        return int(self._chunks[-1]["stamp"].max()) if self._chunks else None

    def nbytes(self) -> int:
        return sum(c["raw"].nbytes for c in self._chunks)

    def window(self, end_stamp: int, window_s: float = WINDOW_S) -> dict[str, Any] | None:
        """``load_grid``'s grid of the window ending at ``end_stamp`` (ms); None with fewer than 2 uniform frames."""
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
        # framediff._uniform_selection on the window's frames: dominant MAC, STRICT_MIMO if any, else the commonest
        counts: dict[Any, int] = {}
        for v in mac[win]:
            counts[v] = counts.get(v, 0) + 1
        dom = max(counts, key=lambda k: counts[k]) if counts else None
        base = win & (mac == dom) if dom else win
        strict = base & (rx == framediff.STRICT_MIMO[0]) & (tx == framediff.STRICT_MIMO[1])
        if strict.any():
            sel = strict
        else:
            pairs, counts = np.unique(np.stack([rx[base], tx[base]], 1), axis=0, return_counts=True)
            if not counts.size:
                return None
            m = pairs[counts.argmax()]
            sel = base & (rx == m[0]) & (tx == m[1])
        sel &= full
        ids = np.flatnonzero(sel)
        if ids.size < 2:
            return None
        D = np.empty((ids.size, 2, self.width), np.complex128)
        o = 0; k = 0
        for c in C:                                    # gather the window's rows chunk by chunk
            n = len(c["stamp"]); j = ids[(ids >= o) & (ids < o + n)] - o
            D[k:k + j.size] = c["raw"][j]; k += j.size; o += n
        D = TF.fill_nulls2(D)
        G = TF.grid_from_frames(D, tr[ids])
        G["span_s"] = float(tr[ids[-1]])
        G["first_stamp"] = int(st[i0])
        G["frames"] = int(ids.size)
        G["complete"] = bool(st.max() >= st[i0] + int(round(window_s * 1000)))   # a later frame has arrived
        return G


class Reference:
    """The room's calibration from its install empty minutes (each a (C, 61) raw series of the model)."""

    def __init__(self, model: str, minutes: list[np.ndarray]) -> None:
        m = TP.META["models"][model]
        self.model = model
        logged = np.stack([_log(np.asarray(X, float), m) for X in minutes])
        self.med, self.mad = TP._stats(logged)
        self.n = len(minutes)


def _log(X: np.ndarray, m: dict[str, Any]) -> np.ndarray:
    x = X.copy()
    lg = [m["channels"].index(c) for c in m["log_channels"]]
    with np.errstate(invalid="ignore"):
        x[lg] = np.log(np.clip(x[lg], 1e-6, None))
    return x


def calibrate(X: np.ndarray, ref: Reference, fold: str) -> np.ndarray:
    """``tfm_phys.calibrate`` with ``ref``'s median / MAD."""
    m = TP.META["models"][ref.model]
    x = _log(X.astype(float), m)
    cal = [m["channels"].index(c) for c in m["calibrated_channels"]]
    pool = TP.FOLD[ref.model][fold]["pool"]
    sd = np.exp((1 - 0.5) * np.log(ref.mad) + 0.5 * np.log(pool))
    x[cal] = (x[cal] - ref.med[cal, None]) / sd[cal, None]
    return x


def series_at(ring: FrameRing, end_stamp: int | None, model: str, timings: dict[str, float] | None = None
              ) -> tuple[np.ndarray, dict[str, Any]] | None:
    """The model's (C, 61) raw input for the window ending at ``end_stamp`` (None: ``LAG_MS`` before the newest
    frame) and the series' extras; None if empty or not every frame of it has arrived."""
    tm = timings if timings is not None else {}
    a = time.perf_counter()
    last = ring.last_stamp()
    if last is None:
        return None
    G = ring.window(last - LAG_MS if end_stamp is None else end_stamp)
    tm["window"] = time.perf_counter() - a
    if G is None or not G["complete"]:
        return None
    s = TF.series_grid(G, TP.META["models"][model]["channels"], tm)
    if s is None:
        return None
    s.update(frames=G["frames"], span_s=G["span_s"], first_stamp=G["first_stamp"])
    return TP.matrix(s["S"], model), s


def judge(X: np.ndarray, ref: Reference, fold: str) -> dict[str, Any]:
    return TP.predict(calibrate(X, ref, fold), np.isfinite(X), ref.model, fold)
