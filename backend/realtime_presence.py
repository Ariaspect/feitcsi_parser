"""Presence every 5 s on the Realtime tab's stream: seq13pw (13F) and seq15pwu (15F) on the last minute.

The streaming deployment of 2026-11-17 (``stream_phys``, ``stream_run``), run here on the frames the Realtime monitor
already reads (its own board stream, or a running capture's file) instead of in a process of its own on the board:

* every frame the monitor reads is decoded once into a ``stream_phys.FrameRing``;
* every ``HOP_S`` the window ending ``LAG_MS`` before the newest frame -- what a one-minute capture was in training --
  is judged by both models: ``tfm_phys_fast.series_grid`` + calibration + the 5-seed mean P (> 0.5: occupied);
* calibration = the room's install reference: the first ``INSTALL_MINUTES`` non-overlapping complete windows after
  "install" is pressed (the room must be empty then), kept in ``REF_PATH`` across restarts;
* the weights are the ``FOLD`` fold, trained on env1 + env2 only.

A judgment equals ``tfm_phys_fast.series`` on a one-minute capture holding the same frames (stream_phys docstring).
"""

from __future__ import annotations

import collections
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import mtk, stream_phys as SP, tfm_phys as TP, tfm_phys_fast as TF

MODELS = ("seq13pw", "seq15pwu")
LABELS = {"seq13pw": "13F", "seq15pwu": "15F"}
FOLD = "scnC"                        # trained on env1 + env2
INSTALL_MINUTES = 5                  # fixed5: the room's five install empty minutes
HOP_S = SP.HOP_S
KEEP_S = 2 * 3600.0
REF_PATH = Path(os.environ.get("REALTIME_REF", str(Path.home() / ".cache" / "csi-realtime" / "reference.npz")))
# both models' channels from one series: each channel is computed on its own, the set only decides which are
CHANNELS = sorted({c for m in MODELS for c in TP.META["models"][m]["channels"]})


class PresenceJudge:
    """The ring, the install reference and the judgments; ``feed`` from the monitor's thread, judged on its own."""

    def __init__(self, ref_path: str | Path = REF_PATH) -> None:
        self.ref_path = Path(ref_path)
        self._lock = threading.RLock()        # ring + state
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ring = SP.FrameRing()
        self.offset: float | None = None      # wall clock - board clock (s), from the monitor
        self.refs: dict[str, SP.Reference] | None = None
        self.ref_info: dict[str, Any] | None = None
        self.installing = False
        self._install: list[np.ndarray] = []  # (C, 61) per install minute, CHANNELS order
        self._install_first: list[int] = []
        self._install_end: int | None = None
        self.rows: collections.deque = collections.deque()   # judgments, oldest first
        self.detail = ""
        self.errors = 0
        self._load()

    # ------------------------------------------------------------------ reference
    def _load(self) -> None:
        if not self.ref_path.exists():
            return
        try:
            z = np.load(self.ref_path, allow_pickle=False)
            ch = [str(c) for c in z["channels"]]
            M = np.asarray(z["minutes"], float)
            self._set_refs(M, ch, [float(v) for v in z["first_wall"]], float(z["saved"]))
        except (OSError, KeyError, ValueError) as exc:
            self.detail = f"saved reference unreadable ({exc}); install a new one"

    def _set_refs(self, M: np.ndarray, ch: list[str], first_wall: list[float], saved: float) -> None:
        refs = {}
        for m in MODELS:
            rows = [ch.index(c) for c in TP.META["models"][m]["channels"]]
            refs[m] = SP.Reference(m, [X[rows] for X in M])
        self.refs = refs
        self.ref_info = {"minutes": int(M.shape[0]), "first_wall": first_wall, "saved": saved,
                         "path": str(self.ref_path)}

    def start_install(self) -> None:
        """Collect a new reference from the next complete windows; judging goes on with the old one meanwhile."""
        with self._lock:
            self.installing = True
            self._install, self._install_first, self._install_end = [], [], None
            # windows that start before now hold frames from before the press
            last = self.ring.last_stamp()
            self._install_end = None if last is None else last + 1

    def cancel_install(self) -> None:
        with self._lock:
            self.installing = False
            self._install, self._install_first, self._install_end = [], [], None

    # ------------------------------------------------------------------ frames in
    def feed(self, path: Path, index: mtk.MTKIndex, ids: np.ndarray, offset: float | None) -> None:
        """New frames ``ids`` of ``index`` (any MIMO mode / AP / width; the window picks its uniform set)."""
        ids = np.asarray(ids, dtype=np.int64)
        if not ids.size:
            return
        with self._lock:
            self.offset = offset
            last = self.ring.last_stamp()
            first = int(np.asarray(index._stamps)[ids].min())
            if (last is not None and first < last - 10_000) or (
                    self.ring.width and int(index.num_subcarriers) != self.ring.width):
                self.ring = SP.FrameRing()            # a new board boot or another width: start afresh
            self.ring.ingest(path, index, ids)

    # ------------------------------------------------------------------ loop
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="realtime-presence", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=20)
        self._thread = None

    def _run(self) -> None:
        next_t = math.ceil(time.time() / HOP_S) * HOP_S
        while not self._stop.is_set():
            self._stop.wait(max(0.0, next_t - time.time()))
            if self._stop.is_set():
                break
            next_t += HOP_S
            try:
                self.step()
            except Exception as exc:  # the loop must not die silently
                with self._lock:
                    self.errors += 1
                    self.detail = f"{type(exc).__name__}: {exc}"
            if time.time() > next_t:                  # fell behind: skip to the next hop rather than pile up
                next_t = math.ceil(time.time() / HOP_S) * HOP_S

    def step(self) -> dict[str, Any] | None:
        """One judgment of the window ending ``LAG_MS`` before the newest frame; the row, or None."""
        tm: dict[str, float] = {}
        c0 = time.perf_counter()
        with self._lock:
            last = self.ring.last_stamp()
            if last is None:
                self.detail = "no frames yet"
                return None
            G = self.ring.window(last - SP.LAG_MS)
            off = self.offset
        tm["window"] = time.perf_counter() - c0
        if G is None or not G["complete"]:
            self.detail = "collecting the first minute of frames"
            return None
        s = TF.series_grid(G, CHANNELS, tm)
        if s is None:
            self.detail = "window not finite once resampled; skipped"
            return None
        X = np.array([s["S"][c] for c in CHANNELS], dtype=float)      # TP.matrix's rows, both models' channels
        first = int(G["first_stamp"])
        end_wall = (first / 1e3 + SP.WINDOW_S + off) if off is not None else time.time()
        row: dict[str, Any] = {"t": round(end_wall, 3), "first_stamp": first, "frames": int(G["frames"]),
                               "span_s": round(float(G["span_s"]), 2)}
        with self._lock:
            if self.installing and (self._install_end is None or first >= self._install_end):
                self._install.append(X)
                self._install_first.append(round(end_wall - SP.WINDOW_S, 3))
                self._install_end = first + int(SP.WINDOW_S * 1000)
                row["install_minute"] = len(self._install)
                if len(self._install) >= INSTALL_MINUTES:
                    M = np.stack(self._install)
                    saved = time.time()
                    self.ref_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = self.ref_path.with_suffix(".tmp.npz")
                    np.savez(tmp, minutes=M, channels=np.array(CHANNELS), first_wall=np.array(self._install_first),
                             saved=saved, fold=FOLD)
                    tmp.replace(self.ref_path)
                    self._set_refs(M, CHANNELS, list(self._install_first), saved)
                    self.installing = False
                    self._install, self._install_first = [], []
                    row["reference_saved"] = True
            refs = self.refs
        if refs is not None:
            for m in MODELS:
                v = SP.judge(X[[CHANNELS.index(c) for c in TP.META["models"][m]["channels"]]], refs[m], FOLD)
                row[m] = round(float(v["p"]), 5)
        tm["total"] = time.perf_counter() - c0
        row["compute_s"] = round(tm["total"], 3)
        with self._lock:
            self.rows.append(row)
            while self.rows and self.rows[0]["t"] < time.time() - KEEP_S:
                self.rows.popleft()
            self.detail = "judging" if refs is not None else "no reference yet: install one with the room empty"
        return row

    # ------------------------------------------------------------------ out
    def status(self, since: float, window_s: float) -> dict[str, Any]:
        with self._lock:
            lo = max(float(since), time.time() - float(window_s))
            rows = [r for r in self.rows if r["t"] > lo]
            cs = [r["compute_s"] for r in self.rows if r["t"] > time.time() - 600]
            return {
                "models": [{"key": m, "label": LABELS[m]} for m in MODELS],
                "fold": FOLD,
                "trained_on": TP.META["models"][MODELS[0]]["folds"][FOLD]["trained_on"],
                "threshold": TP.THRESHOLD,
                "hop_s": HOP_S,
                "window_s": SP.WINDOW_S,
                "install_minutes": INSTALL_MINUTES,
                "reference": self.ref_info,
                "installing": self.installing,
                "install_done": len(self._install),
                "detail": self.detail,
                "errors": self.errors,
                "ring_mb": round(self.ring.nbytes() / 1e6, 1),
                "compute_s": ({"p50": round(float(np.median(cs)), 3), "max": round(float(np.max(cs)), 3),
                               "n": len(cs)} if cs else None),
                "rows": rows,
            }
